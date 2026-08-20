"""Cumulative-step resume state: the LensTrainer manifest idea in flat-file form.

LensTrainer (mirrored in Project_SAVANT's ``savant/training/engine.py``) resumes from
a numbered ``checkpoint-NNNNNN`` directory holding weights, ``optimizer.pt`` and a
``manifest.json`` whose ``step`` is authoritative, with ``-latest`` / ``-emergency``
pointers and a resolve order of *numbered → emergency → latest*. Half-written saves
are skipped so a crash mid-checkpoint can never become the resume target.

DiffSynth writes flat ``step-N.safetensors`` adapter files and nothing else, so the
same contract is expressed as sidecars alongside them:

    step-700.safetensors    adapter tensors (DiffSynth writes this)
    step-700.optim.pt       optimizer moments + LR-scheduler position + RNG
    train_state.json        manifest: authoritative cumulative step + run identity

``step-N`` is always the **cumulative** micro-step across every supervisor attempt, so
a filename can never be reused with different weights.

Nothing here imports DiffSynth or touches a GPU; it is exercised on CPU by
``tests/test_resume_state.py``.
"""

from __future__ import annotations

import json
import os
import random
import re
import struct
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

STATE_FILENAME = "train_state.json"
STATE_FORMAT = "lobora-resume/1"

#: Pointer names, mirroring LensTrainer's ``checkpoint-latest`` / ``checkpoint-emergency``.
LATEST_POINTER = "train_state.latest.txt"
EMERGENCY_POINTER = "train_state.emergency.txt"

#: Exit status meaning "resume state exists but is unusable". A supervisor must treat
#: this as fatal: retrying cannot fix it and each retry burns an attempt.
EXIT_RESUME_UNUSABLE = 3

#: Fields that must agree between the saved optimizer state and the current run for
#: the restored Adam moments to mean anything.
FINGERPRINT_KEYS = (
    "lora_rank",
    "lora_target_modules",
    "lora_base_model",
    "optimizer_class",
    "learning_rate",
    "weight_decay",
    "gradient_accumulation_steps",
    "dataset_size",
    "dataset_repeat",
    "height",
    "width",
    "num_frames",
)


class ResumeStateError(RuntimeError):
    """Resume state is present but unusable. Never swallow this."""


#: Two checkpoint spellings share this state format. ``step-N`` is what DiffSynth's
#: ModelLogger writes on the box; ``lora_step_NNNNNN`` is LoboRA's own (README).
CHECKPOINT_PATTERN = re.compile(r"^(?:step-|lora_step_)(\d+)\.safetensors$")


def checkpoint_name(step: int, *, style: str = "diffsynth") -> str:
    if style == "lobora":
        return f"lora_step_{step:06d}.safetensors"
    return f"step-{step}.safetensors"


def optimizer_sidecar_for(checkpoint: Path) -> Path:
    """``step-700.safetensors`` → ``step-700.optim.pt`` (README ``.optim.pt`` convention)."""
    return Path(checkpoint).with_suffix(".optim.pt")


def parse_checkpoint_step(name: str) -> int | None:
    """Cumulative step encoded in a checkpoint filename, or None if it is not one of ours.

    Deliberately anchored: files parked aside by hand (``attempt1_step-600.safetensors``)
    and the ``lora_latest`` / ``lora_final`` pointers do not match, so they are never
    resume candidates.
    """
    match = CHECKPOINT_PATTERN.match(Path(name).name)
    return int(match.group(1)) if match else None


def verify_safetensors(path: Path) -> int:
    """Structural check that a safetensors file is complete. Returns the tensor count.

    Same test the on-box supervisor applies before offering a file as a resume target:
    a checkpoint truncated by a crash mid-write must not be mistaken for a good one.
    """
    size = path.stat().st_size
    with path.open("rb") as handle:
        raw = handle.read(8)
        if len(raw) != 8:
            raise ResumeStateError(f"{path.name}: no safetensors header length (truncated)")
        header_len = struct.unpack("<Q", raw)[0]
        if header_len <= 0 or 8 + header_len > size:
            raise ResumeStateError(
                f"{path.name}: declares a {header_len}-byte header but the file is {size} bytes"
            )
        try:
            header = json.loads(handle.read(header_len))
        except json.JSONDecodeError as exc:
            raise ResumeStateError(f"{path.name}: unparseable safetensors header ({exc})") from exc
    tensors = {k: v for k, v in header.items() if k != "__metadata__"}
    end = max((v["data_offsets"][1] for v in tensors.values()), default=0)
    if size != 8 + header_len + end:
        raise ResumeStateError(
            f"{path.name}: {size} bytes but header+payload need {8 + header_len + end} (truncated)"
        )
    if not tensors:
        raise ResumeStateError(f"{path.name}: contains no tensors")
    return len(tensors)


@dataclass
class ResumeState:
    """Authoritative training position. ``cumulative_step`` spans all attempts."""

    cumulative_step: int = 0
    total_steps: int = 0
    epoch: int = 0
    epoch_step: int = 0
    attempt: int = 0
    shuffle_seed: int = 42
    checkpoint: str = ""
    optimizer_state: str = ""
    fingerprint: dict[str, Any] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)
    updated: str = ""
    format: str = STATE_FORMAT

    @property
    def steps_remaining(self) -> int:
        return max(0, self.total_steps - self.cumulative_step)

    def checkpoint_path(self, output_dir: Path) -> Path:
        return Path(output_dir) / self.checkpoint

    def optimizer_path(self, output_dir: Path) -> Path:
        return Path(output_dir) / self.optimizer_state

    def describe(self) -> str:
        return (
            f"cumulative step {self.cumulative_step}/{self.total_steps or '?'} "
            f"(epoch {self.epoch} item {self.epoch_step}, written on attempt {self.attempt})"
        )


def state_path(output_dir: Path) -> Path:
    return Path(output_dir) / STATE_FILENAME


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Write via tmp + rename so a crash mid-write cannot truncate the manifest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def capture_rng() -> dict[str, Any]:
    blob: dict[str, Any] = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        blob["cuda"] = torch.cuda.get_rng_state_all()
    try:
        import numpy as np

        blob["numpy"] = np.random.get_state()
    except ImportError:
        pass
    return blob


def restore_rng(blob: dict[str, Any]) -> list[str]:
    """Restore what is present; report what was restored. Missing entries are not fatal."""
    restored: list[str] = []
    if "python" in blob:
        random.setstate(blob["python"])
        restored.append("python")
    if "torch" in blob:
        torch.set_rng_state(blob["torch"].to(torch.uint8).cpu())
        restored.append("torch")
    if "cuda" in blob and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(blob["cuda"])
            restored.append("cuda")
        except (RuntimeError, ValueError):
            # Different device count than the run that saved it; CPU RNG still applies.
            pass
    if "numpy" in blob:
        try:
            import numpy as np

            np.random.set_state(blob["numpy"])
            restored.append("numpy")
        except ImportError:
            pass
    return restored


def save_optimizer_state(
    path: Path,
    *,
    step: int,
    optimizer: Any,
    lr_scheduler: Any = None,
    rng: dict[str, Any] | None = None,
) -> Path:
    """Persist Adam moments + LR-scheduler position + RNG next to the adapter file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    blob: dict[str, Any] = {
        "format": STATE_FORMAT,
        "step": int(step),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "lr_scheduler": lr_scheduler.state_dict() if lr_scheduler is not None else None,
        "rng": rng if rng is not None else capture_rng(),
    }
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(blob, tmp)
    os.replace(tmp, path)
    return path


def load_optimizer_state(
    path: Path,
    *,
    optimizer: Any = None,
    lr_scheduler: Any = None,
    restore_rng_state: bool = True,
) -> dict[str, Any]:
    """Restore optimizer / scheduler / RNG from a sidecar.

    Raises ``ResumeStateError`` on anything unusable. A resume that cannot restore the
    moments it was told to restore must stop, not quietly warm-start.
    """
    path = Path(path)
    if not path.is_file():
        raise ResumeStateError(f"optimizer sidecar missing: {path}")
    try:
        blob = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001 - torch raises many unrelated types here
        raise ResumeStateError(f"cannot read optimizer sidecar {path.name}: {exc}") from exc
    if not isinstance(blob, dict) or "optimizer" not in blob:
        raise ResumeStateError(f"{path.name} is not a LoboRA optimizer sidecar")

    report: dict[str, Any] = {"step": int(blob.get("step", 0) or 0), "rng": []}
    if optimizer is not None:
        if blob["optimizer"] is None:
            raise ResumeStateError(f"{path.name} holds no optimizer state")
        try:
            optimizer.load_state_dict(blob["optimizer"])
        except (ValueError, KeyError, RuntimeError) as exc:
            raise ResumeStateError(
                f"optimizer state in {path.name} does not fit the current optimizer ({exc}). "
                f"This usually means lora_rank / target_modules / optimizer class changed."
            ) from exc
        report["optimizer"] = True
    if lr_scheduler is not None:
        if blob.get("lr_scheduler") is None:
            raise ResumeStateError(f"{path.name} holds no LR-scheduler state")
        lr_scheduler.load_state_dict(blob["lr_scheduler"])
        report["lr_scheduler"] = True
    if restore_rng_state and isinstance(blob.get("rng"), dict):
        report["rng"] = restore_rng(blob["rng"])
    return report


def build_fingerprint(**values: Any) -> dict[str, Any]:
    """Normalised subset of the run config the optimizer state depends on."""
    out: dict[str, Any] = {}
    for key in FINGERPRINT_KEYS:
        if key in values and values[key] is not None:
            value = values[key]
            out[key] = list(value) if isinstance(value, (list, tuple)) else value
    return out


def compare_fingerprints(saved: dict[str, Any], current: dict[str, Any]) -> list[str]:
    """Keys present in both that disagree. Absent keys are not a conflict."""
    return sorted(k for k in saved if k in current and saved[k] != current[k])


def save_resume_state(
    output_dir: Path,
    *,
    step: int,
    total_steps: int,
    epoch: int,
    epoch_step: int,
    attempt: int,
    shuffle_seed: int,
    optimizer: Any = None,
    lr_scheduler: Any = None,
    fingerprint: dict[str, Any] | None = None,
    checkpoint: str | None = None,
    emergency: bool = False,
) -> ResumeState:
    """Write the optimizer sidecar and rewrite ``train_state.json`` to point at ``step``.

    The manifest is written last and atomically, so it only ever names a sidecar that
    is already fully on disk.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt = checkpoint or checkpoint_name(step)
    optim_path = optimizer_sidecar_for(output_dir / ckpt)
    if optimizer is not None or lr_scheduler is not None:
        save_optimizer_state(optim_path, step=step, optimizer=optimizer, lr_scheduler=lr_scheduler)

    previous = read_state_file(output_dir, strict=False)
    history = list(previous.history) if previous else []
    if previous and previous.cumulative_step != step:
        history.append(
            {
                "step": previous.cumulative_step,
                "attempt": previous.attempt,
                "checkpoint": previous.checkpoint,
                "updated": previous.updated,
            }
        )

    state = ResumeState(
        cumulative_step=int(step),
        total_steps=int(total_steps),
        epoch=int(epoch),
        epoch_step=int(epoch_step),
        attempt=int(attempt),
        shuffle_seed=int(shuffle_seed),
        checkpoint=ckpt,
        optimizer_state=optim_path.name if optim_path.is_file() else "",
        fingerprint=dict(fingerprint or {}),
        history=history[-32:],
        updated=_now(),
    )
    write_json_atomic(state_path(output_dir), asdict(state))
    pointer = EMERGENCY_POINTER if emergency else LATEST_POINTER
    (output_dir / pointer).write_text(ckpt + "\n", encoding="utf-8")
    return state


def read_state_file(output_dir: Path, *, strict: bool = True) -> ResumeState | None:
    """Parse ``train_state.json``. Returns None only when the file genuinely is absent.

    With ``strict`` (the default) a present-but-broken manifest raises rather than
    reading as "no state", which would silently restart the run from zero.
    """
    path = state_path(output_dir)
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        if strict:
            raise ResumeStateError(f"{path} is present but unreadable: {exc}") from exc
        return None
    if not isinstance(raw, dict):
        if strict:
            raise ResumeStateError(f"{path} is not a JSON object")
        return None
    fmt = raw.get("format")
    if strict and fmt != STATE_FORMAT:
        raise ResumeStateError(f"{path} has format {fmt!r}, expected {STATE_FORMAT!r}")
    allowed = {f for f in ResumeState.__dataclass_fields__}
    return ResumeState(**{k: v for k, v in raw.items() if k in allowed})


def scan_checkpoints(output_dir: Path) -> list[tuple[int, Path]]:
    """Every structurally valid numbered checkpoint, ascending by cumulative step."""
    output_dir = Path(output_dir)
    found: list[tuple[int, Path]] = []
    for path in sorted(output_dir.glob("*.safetensors")):
        step = parse_checkpoint_step(path.name)
        if step is None:
            continue
        try:
            verify_safetensors(path)
        except ResumeStateError:
            continue
        found.append((step, path))
    return sorted(found)


@dataclass
class ResumeTarget:
    checkpoint: Path
    step: int
    optimizer_state: Path | None
    state: ResumeState | None
    source: str

    def describe(self) -> str:
        optim = self.optimizer_state.name if self.optimizer_state else "none (weights only)"
        return f"{self.checkpoint.name} @ cumulative step {self.step} [{self.source}] optimizer={optim}"


def find_resume_target(
    output_dir: Path,
    *,
    require_optimizer: bool = False,
) -> ResumeTarget | None:
    """Resolve the newest usable checkpoint, LensTrainer order: manifest → numbered → pointer.

    ``require_optimizer`` makes a weights-only checkpoint an error instead of a
    degraded resume, so an operator who asked for full continuation is told when they
    would silently get a warm restart.
    """
    output_dir = Path(output_dir)
    state = read_state_file(output_dir)
    candidates = scan_checkpoints(output_dir)

    if state is not None:
        ckpt = state.checkpoint_path(output_dir)
        if not ckpt.is_file():
            raise ResumeStateError(
                f"{STATE_FILENAME} names {state.checkpoint} at cumulative step "
                f"{state.cumulative_step} but that file is not in {output_dir}. "
                f"Refusing to restart from zero — restore the checkpoint or delete "
                f"{state_path(output_dir)} to start over deliberately."
            )
        verify_safetensors(ckpt)
        newest = candidates[-1][0] if candidates else state.cumulative_step
        if newest > state.cumulative_step:
            raise ResumeStateError(
                f"{STATE_FILENAME} points at step {state.cumulative_step} but "
                f"step-{newest}.safetensors is newer. The manifest and the checkpoint "
                f"directory disagree; resolve by hand rather than losing "
                f"{newest - state.cumulative_step} steps."
            )
        optim = state.optimizer_path(output_dir) if state.optimizer_state else None
        if optim is not None and not optim.is_file():
            raise ResumeStateError(
                f"{STATE_FILENAME} names optimizer state {state.optimizer_state} but it is "
                f"missing from {output_dir}."
            )
        if optim is None and require_optimizer:
            raise ResumeStateError(
                f"no optimizer state recorded for {state.checkpoint}. Resuming would reset "
                f"the Adam moments and the LR schedule; pass --allow-weights-only-resume to "
                f"accept that warm restart."
            )
        return ResumeTarget(ckpt, state.cumulative_step, optim, state, "train_state.json")

    if not candidates:
        return None

    # No manifest: an older run, or the very first one after this patch lands. Fall back
    # to the highest numbered checkpoint and treat its number as the cumulative step.
    step, ckpt = candidates[-1]
    optim = optimizer_sidecar_for(ckpt)
    if not optim.is_file():
        if require_optimizer:
            raise ResumeStateError(
                f"{ckpt.name} has no {optim.name} sidecar and no {STATE_FILENAME}. This "
                f"checkpoint predates resume-state support: resuming from it restarts the "
                f"Adam moments and the LR schedule. Pass --allow-weights-only-resume to accept."
            )
        optim = None
    return ResumeTarget(ckpt, step, optim, None, "highest numbered step-N")


def read_pointer(output_dir: Path, name: str) -> Path | None:
    pointer = Path(output_dir) / name
    if not pointer.is_file():
        return None
    target = Path(output_dir) / pointer.read_text(encoding="utf-8").strip()
    return target if target.is_file() else None
