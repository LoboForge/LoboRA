"""CPU cover for the DiffSynth production-path glue.

Only the pure parts are exercised here: sample ordering, the fingerprint, and the
resume-target guards. The training loop itself needs DiffSynth + a GPU, but every
decision it makes about *where to restart* is made by the functions below.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from lobora.diffsynth_resume import _resolve_start, epoch_sample_order, run_fingerprint
from lobora.resume_state import ResumeStateError, checkpoint_name, save_resume_state


def _write_adapter(directory: Path, step: int) -> Path:
    path = directory / checkpoint_name(step)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({"diffusion_model.x.lora_A.weight": torch.zeros(4, 2)}, str(path))
    return path


def _args(**overrides):
    base = dict(
        lora_rank=32,
        lora_target_modules="qkv_proj,out_proj",
        lora_base_model="dit",
        customized_optimizer=None,
        learning_rate=1e-4,
        weight_decay=0.01,
        gradient_accumulation_steps=4,
        dataset_repeat=7,
        height=480,
        width=832,
        num_frames=73,
        lora_checkpoint=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


# --------------------------------------------------------------------------- #
# Deterministic sample order
# --------------------------------------------------------------------------- #

def test_sample_order_is_reproducible_and_a_permutation():
    first = epoch_sample_order(6741, seed=42, epoch=0)
    again = epoch_sample_order(6741, seed=42, epoch=0)
    assert first == again
    assert sorted(first) == list(range(6741))


def test_sample_order_differs_by_epoch_and_by_seed():
    assert epoch_sample_order(500, seed=42, epoch=0) != epoch_sample_order(500, seed=42, epoch=1)
    assert epoch_sample_order(500, seed=42, epoch=0) != epoch_sample_order(500, seed=43, epoch=0)


def test_resume_skips_exactly_the_consumed_prefix():
    """676 items consumed means the restart trains items 676.. of the same permutation."""
    order = epoch_sample_order(6741, seed=42, epoch=0)
    consumed = 676
    remaining = epoch_sample_order(6741, seed=42, epoch=0)[consumed:]
    assert len(remaining) == 6741 - consumed
    assert set(remaining).isdisjoint(order[:consumed])
    assert order[:consumed] + remaining == order


# --------------------------------------------------------------------------- #
# Fingerprint
# --------------------------------------------------------------------------- #

def test_fingerprint_records_geometry_and_optimizer_shape():
    fingerprint = run_fingerprint(_args(), dataset_size=6741)
    assert fingerprint["dataset_size"] == 6741
    assert fingerprint["optimizer_class"] == "torch.optim.AdamW"
    assert (fingerprint["height"], fingerprint["width"], fingerprint["num_frames"]) == (480, 832, 73)


# --------------------------------------------------------------------------- #
# Resume-target guards: every failure mode must be loud
# --------------------------------------------------------------------------- #

def test_fresh_run_resolves_to_none(tmp_path: Path):
    assert _resolve_start(tmp_path, _args(), {}) is None


def test_resume_without_lora_checkpoint_raises(tmp_path: Path):
    _write_adapter(tmp_path, 600)
    with pytest.raises(ResumeStateError, match="no --lora_checkpoint was passed"):
        _resolve_start(tmp_path, _args(), {})


def test_lora_checkpoint_disagreeing_with_the_state_raises(tmp_path: Path):
    _write_adapter(tmp_path, 600)
    _write_adapter(tmp_path, 700)
    with pytest.raises(ResumeStateError, match="would corrupt the resume"):
        _resolve_start(tmp_path, _args(lora_checkpoint=str(tmp_path / "step-600.safetensors")), {})


def test_matching_lora_checkpoint_resolves(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("LOBORA_ALLOW_WEIGHTS_ONLY_RESUME", "1")
    ckpt = _write_adapter(tmp_path, 700)
    target = _resolve_start(tmp_path, _args(lora_checkpoint=str(ckpt)), {})
    assert target.step == 700
    assert target.checkpoint == ckpt


def test_weights_only_resume_is_fatal_by_default(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("LOBORA_ALLOW_WEIGHTS_ONLY_RESUME", raising=False)
    ckpt = _write_adapter(tmp_path, 700)
    with pytest.raises(ResumeStateError, match="no optimizer sidecar"):
        _resolve_start(tmp_path, _args(lora_checkpoint=str(ckpt)), {})


def test_config_drift_since_the_checkpoint_is_fatal(tmp_path: Path):
    ckpt = _write_adapter(tmp_path, 700)
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    save_resume_state(
        tmp_path,
        step=700,
        total_steps=6741,
        epoch=0,
        epoch_step=700,
        attempt=1,
        shuffle_seed=42,
        optimizer=optimizer,
        lr_scheduler=torch.optim.lr_scheduler.ConstantLR(optimizer),
        fingerprint=run_fingerprint(_args(), dataset_size=6741),
    )
    drifted = run_fingerprint(_args(lora_rank=64), dataset_size=6741)
    with pytest.raises(ResumeStateError, match="run config changed"):
        _resolve_start(tmp_path, _args(lora_checkpoint=str(ckpt)), drifted)
