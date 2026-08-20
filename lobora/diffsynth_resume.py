"""Make DiffSynth's MiniMax-H3 training loop actually resumable.

This is the **production** path. Live runs go through
``examples/minimax_h3/model_training/train.py`` under ``accelerate launch``, which
calls ``diffsynth.diffusion.runner.launch_training_task``. That function builds a
fresh AdamW, a fresh ``ConstantLR``, and a ``ModelLogger`` whose ``num_steps`` starts
at zero, and it persists none of them. So a supervisor restart carries the adapter
tensors (via ``--lora_checkpoint``) and nothing else: the step counter, the Adam
moments and the LR-schedule position all reset.

``install()`` swaps in a logger and a training loop that keep the four things a
restart needs — cumulative step, optimizer moments, scheduler position, and the
position in the sample order — using the sidecar format in ``lobora.resume_state``.
Both replacements are drop-in: same names, same call signatures, so the upstream
example script runs unmodified under ``scripts/train_h3_resumable.py``.

DiffSynth is imported lazily. Importing this module on CPU without DiffSynth is fine
and is what the tests do.
"""

from __future__ import annotations

import os
import signal
from pathlib import Path
from typing import Any

import torch

from lobora.console import error, info, ok, warn
from lobora.resume_state import (
    ResumeStateError,
    build_fingerprint,
    checkpoint_name,
    compare_fingerprints,
    find_resume_target,
    load_optimizer_state,
    save_resume_state,
)


def epoch_sample_order(dataset_size: int, *, seed: int, epoch: int) -> list[int]:
    """Deterministic shuffle for one epoch.

    Upstream uses ``DataLoader(dataset, shuffle=True)`` with no generator, so the
    order depends on global RNG at loader-construction time and is not reproducible
    across processes. Deriving it from an explicit seed makes "resume where we left
    off" mean something: the same epoch replays the same permutation, so skipping the
    first ``epoch_step`` entries genuinely skips the samples already trained on.
    """
    generator = torch.Generator()
    generator.manual_seed((int(seed) * 1_000_003 + int(epoch)) % (2**63 - 1))
    return torch.randperm(dataset_size, generator=generator).tolist()


def run_fingerprint(args: Any, dataset_size: int) -> dict[str, Any]:
    return build_fingerprint(
        lora_rank=getattr(args, "lora_rank", None),
        lora_target_modules=getattr(args, "lora_target_modules", None),
        lora_base_model=getattr(args, "lora_base_model", None),
        optimizer_class=getattr(args, "customized_optimizer", None) or "torch.optim.AdamW",
        learning_rate=getattr(args, "learning_rate", None),
        weight_decay=getattr(args, "weight_decay", None),
        gradient_accumulation_steps=getattr(args, "gradient_accumulation_steps", None),
        dataset_size=dataset_size,
        dataset_repeat=getattr(args, "dataset_repeat", None),
        height=getattr(args, "height", None),
        width=getattr(args, "width", None),
        num_frames=getattr(args, "num_frames", None),
    )


def _current_attempt() -> int:
    try:
        return int(os.environ.get("ANATOMY_ATTEMPT") or 0)
    except ValueError:
        return 0


def build_resumable_logger_class(base_logger_class):
    """Subclass DiffSynth's ``ModelLogger`` so saves are cumulative and carry state."""

    class ResumableModelLogger(base_logger_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # Everything below is filled in by the training loop once it knows the
            # resolved resume target; until then the logger behaves like upstream.
            self.num_steps = 0
            self.total_steps = 0
            self.epoch = 0
            self.epoch_step = 0
            self.attempt = _current_attempt()
            self.shuffle_seed = 42
            self.fingerprint: dict[str, Any] = {}
            self.optimizer = None
            self.lr_scheduler = None

        def adopt_resume(self, *, cumulative_step: int, total_steps: int, epoch: int, epoch_step: int):
            self.num_steps = int(cumulative_step)
            self.total_steps = int(total_steps)
            self.epoch = int(epoch)
            self.epoch_step = int(epoch_step)

        def bind_optimizer(self, optimizer, lr_scheduler) -> None:
            self.optimizer = optimizer
            self.lr_scheduler = lr_scheduler

        def save_model(self, accelerator, model, file_name):
            super().save_model(accelerator, model, file_name)
            if not accelerator.is_main_process:
                return
            self.write_resume_state(file_name)

        def write_resume_state(self, file_name: str, *, emergency: bool = False) -> None:
            try:
                state = save_resume_state(
                    Path(self.output_path),
                    step=self.num_steps,
                    total_steps=self.total_steps,
                    epoch=self.epoch,
                    epoch_step=self.epoch_step,
                    attempt=self.attempt,
                    shuffle_seed=self.shuffle_seed,
                    optimizer=self.optimizer,
                    lr_scheduler=self.lr_scheduler,
                    fingerprint=self.fingerprint,
                    checkpoint=file_name,
                    emergency=emergency,
                )
            except Exception as exc:  # noqa: BLE001 - a save failure must be loud, not fatal
                error(f"failed to write resume state at step {self.num_steps}: {exc}")
                error("the adapter file is on disk but a restart will NOT continue cleanly")
                return
            ok(f"saved {file_name} + resume state ({state.describe()})")

    return ResumableModelLogger


def _resolve_start(output_path: Path, args: Any, fingerprint: dict[str, Any]):
    """Decide where this attempt starts. Raises ``ResumeStateError`` rather than guessing."""
    require_optimizer = not bool(os.environ.get("LOBORA_ALLOW_WEIGHTS_ONLY_RESUME"))
    target = find_resume_target(output_path, require_optimizer=False)
    if target is None:
        info("no resume state found; starting a fresh run at cumulative step 0")
        return None

    info(f"resume target: {target.describe()}")

    lora_checkpoint = getattr(args, "lora_checkpoint", None)
    if lora_checkpoint:
        given = Path(lora_checkpoint).resolve()
        if given != target.checkpoint.resolve():
            raise ResumeStateError(
                f"--lora_checkpoint is {given.name} but the resume state says the run is at "
                f"{target.checkpoint.name} (cumulative step {target.step}). Loading the older "
                f"adapter against the newer optimizer moments would corrupt the resume. "
                f"Point --lora_checkpoint at {target.checkpoint} or clear the resume state."
            )
    else:
        raise ResumeStateError(
            f"resume state says this run is at cumulative step {target.step} "
            f"({target.checkpoint.name}) but no --lora_checkpoint was passed, so the adapter "
            f"tensors would start from scratch while the step counter continues. "
            f"Pass --lora_checkpoint {target.checkpoint}, or delete the resume state to "
            f"deliberately start over."
        )

    if target.state is not None and target.state.fingerprint:
        drift = compare_fingerprints(target.state.fingerprint, fingerprint)
        if drift:
            detail = ", ".join(
                f"{k}: {target.state.fingerprint[k]!r} -> {fingerprint[k]!r}" for k in drift
            )
            raise ResumeStateError(
                f"run config changed since the checkpoint was written ({detail}). The saved "
                f"Adam moments belong to the old configuration. Set "
                f"LOBORA_ALLOW_WEIGHTS_ONLY_RESUME=1 to resume weights-only, or start a new run."
            )

    if target.optimizer_state is None and require_optimizer:
        raise ResumeStateError(
            f"{target.checkpoint.name} has no optimizer sidecar, so resuming would reset the "
            f"Adam moments and the LR schedule — a warm restart, not a continuation. Set "
            f"LOBORA_ALLOW_WEIGHTS_ONLY_RESUME=1 to accept that (expected for the first restart "
            f"after this patch lands, since older checkpoints predate the sidecar)."
        )
    return target


def launch_resumable_training_task(
    accelerator,
    dataset,
    model,
    model_logger,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 1,
    save_steps: int | None = None,
    num_epochs: int = 1,
    enable_model_cpu_offload: bool = False,
    enable_optimizer_cpu_offload: bool = False,
    cpu_offload_split_threshold: int | None = None,
    customized_optimizer: str | None = None,
    args=None,
    **kwargs,
):
    """Drop-in replacement for ``diffsynth.diffusion.runner.launch_training_task``.

    Same construction and same step semantics as upstream (``num_steps`` counts
    dataloader items, ``save_steps`` is measured in those). What is added: the counter
    is cumulative across attempts, optimizer/scheduler/RNG are restored and saved, the
    sample order is seeded so the consumed prefix can be skipped, and the remaining
    budget shrinks so repeated crashes still converge on the scheduled total.
    """
    from diffsynth.core import OffloadTrainingManager
    from diffsynth.diffusion.runner import (
        get_optimizer_class,
        initialize_deepspeed_gradient_checkpointing,
        save_training_args,
    )
    from tqdm import tqdm

    if args is not None:
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        num_epochs = args.num_epochs
        enable_model_cpu_offload = args.enable_model_cpu_offload
        enable_optimizer_cpu_offload = args.enable_optimizer_cpu_offload
        cpu_offload_split_threshold = args.cpu_offload_split_threshold
        customized_optimizer = args.customized_optimizer

    if accelerator.is_main_process:
        save_training_args(args)

    if not hasattr(model_logger, "adopt_resume"):
        raise ResumeStateError(
            "the resumable runner was installed but the ModelLogger was not. Launch through "
            "scripts/train_h3_resumable.py so lobora.diffsynth_resume.install() patches both."
        )

    dataset_size = len(dataset)
    total_steps = dataset_size * max(1, int(num_epochs))
    output_path = Path(model_logger.output_path)
    fingerprint = run_fingerprint(args, dataset_size)
    shuffle_seed = int(os.environ.get("LOBORA_SHUFFLE_SEED") or getattr(args, "seed", 0) or 42)

    # Resolved on every rank: it is read-only, and every rank must agree on the budget.
    target = _resolve_start(output_path, args, fingerprint)
    start_step = target.step if target is not None else 0
    start_epoch = target.state.epoch if (target and target.state) else start_step // max(1, dataset_size)
    start_in_epoch = (
        target.state.epoch_step if (target and target.state) else start_step % max(1, dataset_size)
    )
    if target is not None and target.state is not None and target.state.shuffle_seed:
        shuffle_seed = int(target.state.shuffle_seed)

    if start_step >= total_steps:
        ok(f"already at cumulative step {start_step}/{total_steps}; nothing left to train")
        return

    optimizer_class = get_optimizer_class(customized_optimizer)
    optimizer = optimizer_class(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)

    if enable_model_cpu_offload:
        optimizer, scheduler = accelerator.prepare(optimizer, scheduler)
        model.pipe.device = accelerator.device
        offload_manager = OffloadTrainingManager(
            model, accelerator.device, enable_optimizer_cpu_offload, cpu_offload_split_threshold
        )
    else:
        model.to(device=accelerator.device)
        model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
        offload_manager = None

    if target is not None and target.optimizer_state is not None:
        report = load_optimizer_state(
            target.optimizer_state, optimizer=optimizer, lr_scheduler=scheduler
        )
        ok(
            f"restored optimizer moments + LR-scheduler position from "
            f"{target.optimizer_state.name} (rng: {', '.join(report['rng']) or 'none'})"
        )
    elif target is not None:
        warn(
            f"resuming weights-only from {target.checkpoint.name}: Adam moments and the LR "
            f"schedule restart from zero (this attempt is a warm restart, not a continuation)"
        )

    model_logger.adopt_resume(
        cumulative_step=start_step,
        total_steps=total_steps,
        epoch=start_epoch,
        epoch_step=start_in_epoch,
    )
    model_logger.bind_optimizer(optimizer, scheduler)
    model_logger.shuffle_seed = shuffle_seed
    model_logger.fingerprint = fingerprint

    env_offset = os.environ.get("DIFFSYNTH_STEP_OFFSET")
    if env_offset and env_offset.strip() != str(start_step):
        warn(
            f"ignoring DIFFSYNTH_STEP_OFFSET={env_offset}; the resume state on disk says "
            f"cumulative step {start_step} and that is authoritative"
        )

    info(
        f"attempt {model_logger.attempt or '?'}: cumulative step {start_step}/{total_steps}, "
        f"{total_steps - start_step} to go (epoch {start_epoch}, {start_in_epoch} items consumed, "
        f"shuffle_seed={shuffle_seed})"
    )

    initialize_deepspeed_gradient_checkpointing(accelerator)

    stop_requested = {"value": False}

    def _request_stop(signum, _frame):
        stop_requested["value"] = True
        warn(f"signal {signum} received; finishing this step then saving an emergency checkpoint")

    previous_handlers = {}
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            previous_handlers[sig] = signal.signal(sig, _request_stop)
        except ValueError:
            pass  # not on the main thread; emergency save falls back to the except clause

    def _emergency_save(reason: str) -> None:
        if not accelerator.is_main_process or model_logger.num_steps <= start_step:
            return
        warn(f"{reason}: saving {checkpoint_name(model_logger.num_steps)} before exiting")
        try:
            model_logger.save_model(accelerator, model, checkpoint_name(model_logger.num_steps))
        except Exception as exc:  # noqa: BLE001 - best effort; the original failure matters more
            error(f"emergency save failed: {exc}")

    try:
        for epoch_id in range(start_epoch, num_epochs):
            model_logger.epoch = epoch_id
            order = epoch_sample_order(dataset_size, seed=shuffle_seed, epoch=epoch_id)
            consumed = start_in_epoch if epoch_id == start_epoch else 0
            model_logger.epoch_step = consumed
            if consumed:
                info(f"epoch {epoch_id}: skipping {consumed} already-trained samples")
            dataloader = torch.utils.data.DataLoader(
                dataset,
                sampler=order[consumed:],
                collate_fn=lambda x: x[0],
                num_workers=num_workers,
            )
            dataloader = accelerator.prepare(dataloader)

            progress = tqdm(
                dataloader,
                initial=model_logger.num_steps,
                total=total_steps,
                desc=f"epoch {epoch_id}",
            )
            for data in progress:
                with accelerator.accumulate(model):
                    if dataset.load_from_cache:
                        loss = model({}, inputs=data)
                    else:
                        loss = model(data)
                    accelerator.backward(loss)
                    if offload_manager is not None:
                        offload_manager.after_backward()
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    model_logger.epoch_step += 1
                    model_logger.on_step_end(accelerator, model, save_steps, loss=loss)
                if stop_requested["value"]:
                    break
            if stop_requested["value"]:
                break
            start_in_epoch = 0
    except BaseException as exc:  # noqa: BLE001 - includes KeyboardInterrupt / SystemExit
        _emergency_save(f"training aborted ({type(exc).__name__})")
        raise
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)

    if stop_requested["value"]:
        _emergency_save("stop requested")
        raise SystemExit(130)

    model_logger.on_training_end(accelerator, model, save_steps)


def install() -> None:
    """Swap the resumable logger + runner into ``diffsynth.diffusion``.

    Must run before the example script's ``from diffsynth.diffusion import *``, which
    is why ``scripts/train_h3_resumable.py`` calls this and then executes the example.
    """
    import diffsynth.diffusion as diffusion
    from diffsynth.diffusion import logger as logger_module
    from diffsynth.diffusion import runner as runner_module

    resumable_logger = build_resumable_logger_class(logger_module.ModelLogger)
    for module in (diffusion, logger_module, runner_module):
        module.ModelLogger = resumable_logger
    for module in (diffusion, runner_module):
        module.launch_training_task = launch_resumable_training_task
    info("resumable training loop installed (cumulative step + optimizer/scheduler state)")
