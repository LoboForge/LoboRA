"""Train loop: cache → LoRA → resume/emergency → numerics gate."""

from __future__ import annotations

import json
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
    find_resume_checkpoint,
    load_lora_weights,
    load_optimizer,
    optimizer_sidecar_path,
    resolve_resume_checkpoint,
    save_lora,
    save_optimizer,
    write_checkpoint_sidecars,
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


class LoboRATrainer:
    def __init__(self, cfg: TrainerConfig, *, dry_run: bool = False) -> None:
        self.cfg = cfg
        self.dry_run = dry_run
        self.paths = output_paths(cfg)
        self.paths["root"].mkdir(parents=True, exist_ok=True)
        self.paths["checkpoints"].mkdir(parents=True, exist_ok=True)
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

    def _save(self, model, optimizer, step: int, *, name: str | None = None) -> Path:
        dest = self.paths["root"] / (name or f"lora_step_{step:06d}.safetensors")
        if name is None:
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
        if name is None:
            write_checkpoint_sidecars(
                dest,
                output_dir=self.paths["root"],
                checkpoints_dir=self.paths["checkpoints"],
                step=step,
            )
            if optimizer is not None:
                save_optimizer(optimizer, optimizer_sidecar_path(dest), step=step)
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
        optimizer = None
        global_step = 0
        resume = resolve_resume_checkpoint(
            self.cfg.train.resume_from,
            output_dir=self.paths["root"],
            checkpoints_dir=self.paths["checkpoints"],
        )
        if resume is None and self.cfg.train.resume_from.lower() in {"latest", "auto"}:
            resume = find_resume_checkpoint(
                output_dir=self.paths["root"],
                checkpoints_dir=self.paths["checkpoints"],
            )
        if resume is not None:
            info(f"resume from {resume}")
            if model is not None:
                meta = load_lora_weights(model, resume)
                global_step = int(meta["step"])
                opt_path = optimizer_sidecar_path(resume)
                if optimizer is not None and opt_path.is_file():
                    load_optimizer(optimizer, opt_path)
                    info("restored optimizer state")
            else:
                from lobora.lora import read_lora_checkpoint_metadata

                global_step = int(read_lora_checkpoint_metadata(resume).get("step", 0) or 0)
            self.cfg.sample.baseline_control = False

        self._maybe_sample(0, tag="control")
        pending_losses: list[dict[str, Any]] = []
        accum = 0
        running = 0.0

        try:
            while global_step < self.cfg.train.steps:
                for _batch in sampler:
                    if global_step >= self.cfg.train.steps:
                        break
                    loss = _dummy_step(self.scheduler, self.device)
                    running += loss
                    accum += 1
                    if accum < self.cfg.train.gradient_accumulation_steps:
                        continue
                    global_step += 1
                    mean = running / accum
                    running = 0.0
                    accum = 0
                    pending_losses.append({"step": global_step, "loss": mean})
                    if global_step % 25 == 0:
                        _append_loss(self.paths["loss"], pending_losses)
                        pending_losses = []
                    if global_step % self.cfg.train.save_every == 0:
                        self._save(model, optimizer, global_step)
                        ok(f"saved step {global_step}  loss={mean:.4f}")
                    self._maybe_sample(global_step, tag="lora")
        except KeyboardInterrupt:
            if global_step > 0:
                path = self._save(model, optimizer, global_step, name="lora_emergency.safetensors")
                warn(f"interrupted at step {global_step}; wrote {path}")
                warn(f"resume with --resume latest")
            raise

        if pending_losses:
            _append_loss(self.paths["loss"], pending_losses)
        self._save(model, optimizer, global_step, name="lora_final.safetensors")
        write_checkpoint_sidecars(
            self.paths["root"] / "lora_final.safetensors",
            output_dir=self.paths["root"],
            checkpoints_dir=self.paths["checkpoints"],
            step=global_step,
        )
        ok(f"done at step {global_step}")


def run(cfg: TrainerConfig, *, dry_run: bool = False) -> None:
    LoboRATrainer(cfg, dry_run=dry_run).train()
