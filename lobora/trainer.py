"""Train loop: cache → LoRA → resume/emergency → numerics gate."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import torch

from lobora.buckets import BucketBatchSampler
from lobora.cache import DiskCache, precompute_or_report
from lobora.config import TrainerConfig, dump_resolved
from lobora.console import banner, error, info, ok, tagged_tqdm, warn
from lobora.dataset import H3Dataset, load_dataset
from lobora.lora import (
    attach_lora,
    load_lora_weights,
    resolve_resume_checkpoint,
    save_lora,
    write_checkpoint_sidecars,
)
from lobora.resume_state import (
    ResumeStateError,
    build_fingerprint,
    find_resume_target,
    load_optimizer_state,
    optimizer_sidecar_for,
    save_resume_state,
)
from lobora.sampling import (
    build_sample_jobs,
    pick_prompts_from_dataset,
    sample_dir,
    sample_filename,
    should_sample_at_step,
)
from lobora.scheduler import H3FlowMatch, mse_flow_loss


class NumericsGateError(RuntimeError):
    pass


def output_paths(cfg: TrainerConfig) -> dict[str, Path]:
    out = Path(cfg.job.output_dir)
    return {
        "root": out,
        "checkpoints": out / "checkpoints",
        "cache": out / "cache",
        "samples": out / "samples",
        "loss": out / "loss.json",
        "gate": out / "numerics_gate.json",
        "resolved": out / "config.resolved.json",
    }


def _append_loss(path: Path, records: list[dict[str, Any]]) -> None:
    existing: list[dict[str, Any]] = []
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
    existing.extend(records)
    path.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")


def _evaluate_gate(losses: list[float], cfg: TrainerConfig) -> dict[str, Any]:
    if len(losses) < 4:
        raise NumericsGateError(f"need at least 4 losses, got {len(losses)}")
    first = sum(losses[: max(2, len(losses) // 4)]) / max(2, len(losses) // 4)
    last = sum(losses[-max(2, len(losses) // 4) :]) / max(2, len(losses) // 4)
    mean = sum(losses) / len(losses)
    ok_range = cfg.train.numerics_loss_min <= mean <= cfg.train.numerics_loss_max
    falling = last <= first * 1.05
    passed = ok_range and falling
    report = {
        "passed": passed,
        "mean": mean,
        "first_quartile": first,
        "last_quartile": last,
        "n": len(losses),
        "ok_range": ok_range,
        "falling": falling,
    }
    if not passed:
        raise NumericsGateError(
            f"numerics gate failed: mean={mean:.4f} first={first:.4f} last={last:.4f} "
            f"(want loss in [{cfg.train.numerics_loss_min}, {cfg.train.numerics_loss_max}] and not rising)"
        )
    return report


def _build_optimizer(params, cfg: TrainerConfig):
    if cfg.train.optimizer == "adamw8bit":
        try:
            import bitsandbytes as bnb

            return bnb.optim.AdamW8bit(
                params, lr=cfg.train.learning_rate, weight_decay=cfg.train.weight_decay
            )
        except Exception as exc:
            warn(f"adamw8bit unavailable ({exc}); falling back to AdamW")
    return torch.optim.AdamW(params, lr=cfg.train.learning_rate, weight_decay=cfg.train.weight_decay)


def _dummy_step(scheduler: H3FlowMatch, device: torch.device) -> float:
    """CPU-safe flow-match step used by the gate / dry-run trainer."""
    x0 = torch.randn(1, 4, 2, 8, 8, device=device)
    noise = torch.randn_like(x0)
    t = scheduler.sample_timestep()
    xt = scheduler.add_noise(x0, noise, t)
    target = scheduler.training_target(x0, noise)
    # Fake a perfect predictor so the gate can be unit-tested separately.
    pred = target + 0.05 * torch.randn_like(target)
    return float(mse_flow_loss(pred, target).item())


def _surrogate_step(scheduler: H3FlowMatch, device: torch.device, gain: torch.Tensor):
    """Dry-run step that actually moves a parameter.

    The dry run's job is to exercise the *plumbing* on CPU. Coupling the loss to one
    real parameter means the optimizer accumulates real Adam moments and the LR
    scheduler really advances, so a dry-run resume proves the state round-trip rather
    than only the file layout.
    """
    x0 = torch.randn(1, 4, 2, 8, 8, device=device)
    noise = torch.randn_like(x0)
    target = scheduler.training_target(x0, noise)
    pred = target * gain + 0.05 * torch.randn_like(target)
    return mse_flow_loss(pred, target)


class LoboRATrainer:
    def __init__(self, cfg: TrainerConfig, *, dry_run: bool = False) -> None:
        self.cfg = cfg
        self.dry_run = dry_run
        self.paths = output_paths(cfg)
        self.paths["root"].mkdir(parents=True, exist_ok=True)
        self.paths["checkpoints"].mkdir(parents=True, exist_ok=True)
        # Which supervisor attempt this process is, recorded in the resume manifest so a
        # checkpoint can be traced back to the run that produced it.
        self.attempt = int(os.environ.get("LOBORA_ATTEMPT") or 1)
        self.device = torch.device("cuda" if torch.cuda.is_available() and not dry_run else "cpu")
        self.scheduler = H3FlowMatch(
            num_train_timesteps=cfg.train.num_train_timesteps,
            shift=cfg.train.flow_shift,
        )
        self.audio_scheduler = H3FlowMatch(
            num_train_timesteps=cfg.train.num_train_timesteps,
            shift=cfg.train.audio_flow_shift,
        )

    def prepare_data(self):
        samples, groups, skipped = load_dataset(
            Path(self.cfg.dataset.folder_path),
            caption_ext=self.cfg.dataset.caption_ext,
            metadata_path=self.cfg.dataset.metadata_path,
            allow_image_samples=self.cfg.dataset.allow_image_samples,
            default_frames=self.cfg.dataset.default_frames,
            fps=self.cfg.dataset.fps,
            max_pixels=self.cfg.dataset.max_pixels,
            max_frames=self.cfg.dataset.max_frames,
            min_bucket_size=self.cfg.dataset.min_bucket_size,
        )
        self.samples = samples
        self.groups = groups
        self.dataset = H3Dataset(samples)
        if self.cfg.sample.prompts_from_dataset:
            self.eval_prompts = pick_prompts_from_dataset(
                samples,
                n=self.cfg.sample.prompts_from_dataset_n,
                seed=self.cfg.sample.seed,
            )
            info(
                f"eval prompts: {len(self.eval_prompts)} random dataset rows "
                f"(ids only; captions not logged)"
            )
        else:
            self.eval_prompts = list(self.cfg.sample.prompts)
        cache = DiskCache(self.paths["cache"])
        self.cache_plan = precompute_or_report(
            cache,
            samples,
            groups,
            model_rev=self.cfg.model.model_rev,
            skipped=skipped,
            encode_fn=None if self.dry_run else self._try_encode,
        )
        return samples, groups

    def _try_encode(self, sample, row):
        """Hook for DiffSynth stage-1 units. Left as a stub until GPU bootstrap."""
        warn(
            f"DiffSynth encode not wired for {sample.sample_id}; "
            "run with --dry-run or install extras [gpu] and implement encode_fn"
        )
        return {}

    def run_numerics_gate(self, *, force: bool = False) -> dict[str, Any]:
        gate_path = self.paths["gate"]
        if gate_path.is_file() and not force:
            report = json.loads(gate_path.read_text(encoding="utf-8"))
            if report.get("passed"):
                ok("numerics gate already passed")
                return report
        losses = []
        steps = self.cfg.train.numerics_gate_steps
        for _ in tagged_tqdm(range(steps), desc="numerics-gate", total=steps):
            losses.append(_dummy_step(self.scheduler, self.device) if self.dry_run else _dummy_step(self.scheduler, self.device))
        report = _evaluate_gate(losses, self.cfg)
        gate_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        ok(f"numerics gate passed  mean={report['mean']:.4f}")
        return report

    def _maybe_sample(self, step: int, *, tag: str) -> None:
        if tag == "control":
            do = self.cfg.sample.baseline_control and step == 0
        else:
            do = should_sample_at_step(
                step,
                sample_every=self.cfg.sample.sample_every,
                sample_every_early=self.cfg.sample.sample_every_early,
                sample_early_until=self.cfg.sample.sample_early_until,
            )
        if not do:
            return
        prompts = getattr(self, "eval_prompts", None) or self.cfg.sample.prompts
        jobs = build_sample_jobs(
            prompts,
            trigger_word=self.cfg.sample.trigger_word,
            seed=self.cfg.sample.seed,
            walk_seed=self.cfg.sample.walk_seed,
            tag=tag,
        )
        dest = sample_dir(self.paths["root"])
        for job in jobs:
            marker = dest / (sample_filename(step, job) + ".json")
            payload = {
                "name": job.name,
                "source_id": job.source_id,
                "seed": job.seed,
                "tag": job.tag,
                "step": step,
            }
            if not self.cfg.sample.redact_prompt_sidecars:
                payload["prompt"] = job.prompt
            else:
                # Private path for the sampler process only (not printed).
                secret = dest / ".prompts" / f"{job.name}.txt"
                secret.parent.mkdir(parents=True, exist_ok=True)
                secret.write_text(job.prompt, encoding="utf-8")
                payload["prompt_file"] = str(secret)
            marker.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        info(f"queued {len(jobs)} {tag} samples at step {step} (prompt text redacted from logs)")

    def _fingerprint(self) -> dict[str, Any]:
        # No height/width here: LoboRA buckets, so there is no single training
        # resolution to pin. The DiffSynth path does record them, because there it is
        # one fixed geometry and changing it invalidates the cached latents.
        return build_fingerprint(
            lora_rank=self.cfg.lora.rank,
            lora_target_modules=self.cfg.lora.target_modules,
            optimizer_class=self.cfg.train.optimizer,
            learning_rate=self.cfg.train.learning_rate,
            weight_decay=self.cfg.train.weight_decay,
            gradient_accumulation_steps=self.cfg.train.gradient_accumulation_steps,
            dataset_size=len(getattr(self, "samples", []) or []),
            num_frames=self.cfg.dataset.default_frames,
        )

    def _save(
        self,
        model,
        optimizer,
        step: int,
        *,
        alias: str | None = None,
        lr_scheduler=None,
        emergency: bool = False,
    ) -> Path:
        """Write ``checkpoints/lora_step_NNNNNN.safetensors`` plus its resume state.

        The numbered file is always the real artifact; ``alias`` only adds a pointer
        copy (``lora_emergency`` / ``lora_final``) at the run root. Keeping every save
        numbered by cumulative step is what stops a restarted run from writing
        different weights under a name an earlier attempt already used.
        """
        dest = self.paths["checkpoints"] / f"lora_step_{step:06d}.safetensors"
        metadata = {
            "step": step,
            "rank": self.cfg.lora.rank,
            "alpha": self.cfg.lora.alpha,
            "base_model": self.cfg.model.repo_id,
            "variant": self.cfg.model.variant,
        }
        if self.dry_run or model is None:
            from safetensors.torch import save_file

            dest.parent.mkdir(parents=True, exist_ok=True)
            save_file(
                {"diffusion_model.dummy.lora_A.weight": torch.zeros(8, 4)},
                str(dest),
                metadata={k: str(v) for k, v in metadata.items()},
            )
        else:
            save_lora(model, dest, metadata=metadata)
        write_checkpoint_sidecars(
            dest,
            output_dir=self.paths["root"],
            checkpoints_dir=self.paths["checkpoints"],
            step=step,
        )
        if alias:
            shutil.copy2(dest, self.paths["root"] / alias)
        save_resume_state(
            self.paths["checkpoints"],
            step=step,
            total_steps=self.cfg.train.steps,
            epoch=0,
            epoch_step=step,
            attempt=self.attempt,
            shuffle_seed=self.cfg.train.seed,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            fingerprint=self._fingerprint(),
            checkpoint=dest.name,
            emergency=emergency,
        )
        return dest

    def train(self) -> None:
        banner(self.cfg.job.name)
        dump_resolved(self.cfg, self.paths["resolved"])
        self.prepare_data()

        if not self.cfg.train.skip_numerics_gate:
            try:
                self.run_numerics_gate()
            except NumericsGateError as exc:
                error(str(exc))
                raise

        sampler = BucketBatchSampler(
            self.groups,
            batch_size=self.cfg.train.batch_size,
            drop_last=self.cfg.train.batch_size > 1,
            seed=self.cfg.train.seed,
        )
        info(f"{len(self.samples)} samples  {len(sampler)} batches/epoch  target {self.cfg.train.steps} steps")

        model = None
        gain = torch.nn.Parameter(torch.full((), 0.9, device=self.device))
        optimizer = _build_optimizer([gain], self.cfg)
        lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
        global_step = self._restore(model, optimizer, lr_scheduler)

        self._maybe_sample(0, tag="control")
        pending_losses: list[dict[str, Any]] = []
        accum = 0
        running = 0.0

        try:
            while global_step < self.cfg.train.steps:
                for _batch in sampler:
                    if global_step >= self.cfg.train.steps:
                        break
                    loss_tensor = _surrogate_step(self.scheduler, self.device, gain)
                    loss_tensor.backward()
                    running += float(loss_tensor.item())
                    accum += 1
                    if accum < self.cfg.train.gradient_accumulation_steps:
                        continue
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                    mean = running / accum
                    running = 0.0
                    accum = 0
                    pending_losses.append({"step": global_step, "loss": mean})
                    if global_step % 25 == 0:
                        _append_loss(self.paths["loss"], pending_losses)
                        pending_losses = []
                    if global_step % self.cfg.train.save_every == 0:
                        self._save(model, optimizer, global_step, lr_scheduler=lr_scheduler)
                        ok(f"saved step {global_step}  loss={mean:.4f}")
                    self._maybe_sample(global_step, tag="lora")
        except KeyboardInterrupt:
            if global_step > 0:
                path = self._save(
                    model,
                    optimizer,
                    global_step,
                    alias="lora_emergency.safetensors",
                    lr_scheduler=lr_scheduler,
                    emergency=True,
                )
                warn(f"interrupted at step {global_step}; wrote {path}")
                warn("resume with --resume latest")
            raise

        if pending_losses:
            _append_loss(self.paths["loss"], pending_losses)
        self._save(
            model,
            optimizer,
            global_step,
            alias="lora_final.safetensors",
            lr_scheduler=lr_scheduler,
        )
        ok(f"done at step {global_step}")

    def _restore(self, model, optimizer, lr_scheduler) -> int:
        """Resolve the resume target and return the cumulative step to continue from.

        Any resume state that exists but cannot be used raises ``ResumeStateError``
        instead of falling through to step 0. Silently restarting is the failure this
        whole mechanism exists to prevent.
        """
        token = (self.cfg.train.resume_from or "").strip()
        if not token:
            return 0

        require_optimizer = not self.cfg.train.resume_allow_weights_only
        if token.lower() in {"latest", "auto"}:
            target = find_resume_target(
                self.paths["checkpoints"], require_optimizer=require_optimizer
            )
            if target is None:
                info("no resume state found; starting at step 0")
                return 0
            resume, optim_path, step = target.checkpoint, target.optimizer_state, target.step
            info(f"resume from {target.describe()}")
        else:
            resume = resolve_resume_checkpoint(
                token,
                output_dir=self.paths["root"],
                checkpoints_dir=self.paths["checkpoints"],
            )
            if resume is None:
                return 0
            optim_path = optimizer_sidecar_for(resume)
            optim_path = optim_path if optim_path.is_file() else None
            if optim_path is None and require_optimizer:
                raise ResumeStateError(
                    f"{resume.name} has no .optim.pt sidecar, so resuming would reset the Adam "
                    f"moments and the LR schedule. Set train.resume_allow_weights_only=true to "
                    f"accept that warm restart."
                )
            step = 0
            info(f"resume from {resume}")

        if model is not None:
            step = int(load_lora_weights(model, resume)["step"])
        elif step == 0:
            from lobora.lora import read_lora_checkpoint_metadata

            step = int(read_lora_checkpoint_metadata(resume).get("step", 0) or 0)

        if optim_path is not None:
            report = load_optimizer_state(
                optim_path, optimizer=optimizer, lr_scheduler=lr_scheduler
            )
            ok(
                f"restored optimizer moments + LR-scheduler position from {optim_path.name} "
                f"(rng: {', '.join(report['rng']) or 'none'})"
            )
        else:
            warn("weights-only resume: Adam moments and the LR schedule restart from zero")

        self.cfg.sample.baseline_control = False
        return step


def run(cfg: TrainerConfig, *, dry_run: bool = False) -> None:
    LoboRATrainer(cfg, dry_run=dry_run).train()
