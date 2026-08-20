"""CPU-only cover for the resume contract: cumulative step, optimizer + scheduler
state, checkpoint-name collisions, and fail-loud behaviour on unusable state."""

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from lobora.resume_state import (
    STATE_FILENAME,
    ResumeStateError,
    build_fingerprint,
    checkpoint_name,
    compare_fingerprints,
    find_resume_target,
    load_optimizer_state,
    optimizer_sidecar_for,
    parse_checkpoint_step,
    read_state_file,
    save_resume_state,
    scan_checkpoints,
    verify_safetensors,
)


def _write_adapter(directory: Path, step: int, *, fill: float = 1.0, style: str = "diffsynth") -> Path:
    path = directory / checkpoint_name(step, style=style)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {"diffusion_model.blocks.0.qkv_proj.lora_A.weight": torch.full((4, 2), fill)},
        str(path),
        metadata={"step": str(step)},
    )
    return path


def _trained_optimizer(steps: int, *, lr: float = 1e-3):
    """A real AdamW + ConstantLR advanced ``steps`` times, so the state is non-trivial."""
    model = torch.nn.Linear(4, 4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    for _ in range(steps):
        model(torch.randn(2, 4)).pow(2).mean().backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    return model, optimizer, scheduler


def _fingerprint():
    return build_fingerprint(
        lora_rank=32,
        lora_target_modules="qkv_proj,out_proj",
        learning_rate=1e-4,
        gradient_accumulation_steps=4,
        height=480,
        width=832,
        num_frames=73,
    )


# --------------------------------------------------------------------------- #
# Naming
# --------------------------------------------------------------------------- #

def test_parse_checkpoint_step_accepts_both_spellings():
    assert parse_checkpoint_step("step-700.safetensors") == 700
    assert parse_checkpoint_step("lora_step_000700.safetensors") == 700


@pytest.mark.parametrize(
    "name",
    [
        "attempt1_step-600.safetensors",  # parked aside by hand
        "lora_latest.safetensors",
        "lora_final.safetensors",
        "epoch-0.safetensors",
        "step-.safetensors",
    ],
)
def test_parse_checkpoint_step_rejects_non_lineage_names(name):
    assert parse_checkpoint_step(name) is None


def test_cumulative_names_never_collide_across_attempts(tmp_path: Path):
    """Attempt 2 resuming at 600 must not reuse a name attempt 1 already wrote."""
    lora = tmp_path / "lora"
    attempt1 = [_write_adapter(lora, step, fill=1.0) for step in (100, 200, 300, 400, 500, 600)]
    before = {p.name: p.read_bytes() for p in attempt1}

    # Attempt 2 continues the lineage from the resume point with different weights.
    attempt2 = [_write_adapter(lora, step, fill=2.0) for step in (700, 800)]

    assert not {p.name for p in attempt2} & set(before)
    for path in attempt1:
        assert path.read_bytes() == before[path.name], f"{path.name} was overwritten"
    assert [step for step, _ in scan_checkpoints(lora)] == [100, 200, 300, 400, 500, 600, 700, 800]


def test_optimizer_sidecar_naming_follows_the_checkpoint():
    assert optimizer_sidecar_for(Path("/x/step-700.safetensors")).name == "step-700.optim.pt"
    assert (
        optimizer_sidecar_for(Path("/x/lora_step_000700.safetensors")).name
        == "lora_step_000700.optim.pt"
    )


# --------------------------------------------------------------------------- #
# Round-trip: cumulative step, Adam moments, scheduler position
# --------------------------------------------------------------------------- #

def test_round_trip_restores_step_moments_and_scheduler(tmp_path: Path):
    lora = tmp_path / "lora"
    _write_adapter(lora, 676)
    _, optimizer, scheduler = _trained_optimizer(12)

    saved_moments = {
        id_: {k: v.clone() for k, v in state.items() if torch.is_tensor(v)}
        for id_, state in optimizer.state_dict()["state"].items()
    }
    saved_lr = scheduler.get_last_lr()
    saved_last_epoch = scheduler.state_dict()["last_epoch"]
    assert saved_last_epoch == 12, "scheduler must have actually advanced"

    state = save_resume_state(
        lora,
        step=676,
        total_steps=6741,
        epoch=0,
        epoch_step=676,
        attempt=2,
        shuffle_seed=42,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        fingerprint=_fingerprint(),
    )
    assert state.cumulative_step == 676
    assert state.steps_remaining == 6741 - 676

    # A restart: brand-new optimizer and scheduler, zeroed moments and position.
    _, fresh_optimizer, fresh_scheduler = _trained_optimizer(0)
    assert fresh_optimizer.state_dict()["state"] == {}
    assert fresh_scheduler.state_dict()["last_epoch"] == 0

    target = find_resume_target(lora, require_optimizer=True)
    assert target is not None
    assert target.step == 676, "a resumed run must know it is at 676, not 0"

    report = load_optimizer_state(
        target.optimizer_state, optimizer=fresh_optimizer, lr_scheduler=fresh_scheduler
    )
    assert report["optimizer"] and report["lr_scheduler"]

    restored = fresh_optimizer.state_dict()["state"]
    assert restored, "Adam moments did not come back"
    for id_, moments in saved_moments.items():
        for key, value in moments.items():
            assert torch.allclose(restored[id_][key], value), f"param {id_} moment {key} drifted"
    assert fresh_scheduler.state_dict()["last_epoch"] == saved_last_epoch
    assert fresh_scheduler.get_last_lr() == saved_lr


def test_saved_step_survives_a_second_save(tmp_path: Path):
    lora = tmp_path / "lora"
    for step in (600, 700):
        _write_adapter(lora, step)
        _, optimizer, scheduler = _trained_optimizer(2)
        save_resume_state(
            lora,
            step=step,
            total_steps=6741,
            epoch=0,
            epoch_step=step,
            attempt=2,
            shuffle_seed=42,
            optimizer=optimizer,
            lr_scheduler=scheduler,
        )
    state = read_state_file(lora)
    assert state.cumulative_step == 700
    assert [entry["step"] for entry in state.history] == [600]
    assert (lora / "step-600.optim.pt").is_file()
    assert (lora / "step-700.optim.pt").is_file()


def test_state_file_is_valid_json_naming_the_sidecar(tmp_path: Path):
    lora = tmp_path / "lora"
    _write_adapter(lora, 300)
    _, optimizer, scheduler = _trained_optimizer(3)
    save_resume_state(
        lora,
        step=300,
        total_steps=1000,
        epoch=0,
        epoch_step=300,
        attempt=1,
        shuffle_seed=7,
        optimizer=optimizer,
        lr_scheduler=scheduler,
    )
    raw = json.loads((lora / STATE_FILENAME).read_text(encoding="utf-8"))
    assert raw["cumulative_step"] == 300
    assert raw["checkpoint"] == "step-300.safetensors"
    assert raw["optimizer_state"] == "step-300.optim.pt"
    assert raw["shuffle_seed"] == 7


# --------------------------------------------------------------------------- #
# Resolution order
# --------------------------------------------------------------------------- #

def test_no_state_at_all_is_not_an_error(tmp_path: Path):
    assert find_resume_target(tmp_path) is None


def test_falls_back_to_highest_numbered_when_no_manifest(tmp_path: Path):
    lora = tmp_path / "lora"
    for step in (100, 900, 200):
        _write_adapter(lora, step)
    target = find_resume_target(lora)
    assert target.step == 900
    assert target.source == "highest numbered step-N"
    assert target.optimizer_state is None


def test_truncated_checkpoint_is_never_a_resume_target(tmp_path: Path):
    lora = tmp_path / "lora"
    _write_adapter(lora, 100)
    good = _write_adapter(lora, 200)
    partial = lora / checkpoint_name(300)
    partial.write_bytes(good.read_bytes()[: len(good.read_bytes()) // 2])

    with pytest.raises(ResumeStateError):
        verify_safetensors(partial)
    assert [step for step, _ in scan_checkpoints(lora)] == [100, 200]
    assert find_resume_target(lora).step == 200


# --------------------------------------------------------------------------- #
# Fail loudly, never silently
# --------------------------------------------------------------------------- #

def test_manifest_naming_a_missing_checkpoint_raises(tmp_path: Path):
    lora = tmp_path / "lora"
    _write_adapter(lora, 500)
    save_resume_state(lora, step=500, total_steps=6741, epoch=0, epoch_step=500, attempt=1, shuffle_seed=42)
    (lora / checkpoint_name(500)).unlink()

    with pytest.raises(ResumeStateError, match="Refusing to restart from zero"):
        find_resume_target(lora)


def test_manifest_older_than_the_directory_raises(tmp_path: Path):
    lora = tmp_path / "lora"
    _write_adapter(lora, 500)
    save_resume_state(lora, step=500, total_steps=6741, epoch=0, epoch_step=500, attempt=1, shuffle_seed=42)
    _write_adapter(lora, 900)  # a newer checkpoint the manifest does not know about

    with pytest.raises(ResumeStateError, match="disagree"):
        find_resume_target(lora)


def test_corrupt_manifest_raises_instead_of_reading_as_absent(tmp_path: Path):
    lora = tmp_path / "lora"
    lora.mkdir()
    (lora / STATE_FILENAME).write_text("{ not json", encoding="utf-8")

    with pytest.raises(ResumeStateError, match="unreadable"):
        read_state_file(lora)


def test_weights_only_resume_is_refused_unless_asked_for(tmp_path: Path):
    lora = tmp_path / "lora"
    _write_adapter(lora, 600)

    with pytest.raises(ResumeStateError, match="predates resume-state support"):
        find_resume_target(lora, require_optimizer=True)
    assert find_resume_target(lora, require_optimizer=False).step == 600


def test_missing_optimizer_sidecar_raises(tmp_path: Path):
    with pytest.raises(ResumeStateError, match="missing"):
        load_optimizer_state(tmp_path / "step-100.optim.pt", optimizer=None)


def test_incompatible_optimizer_state_raises(tmp_path: Path):
    lora = tmp_path / "lora"
    _write_adapter(lora, 100)
    _, optimizer, scheduler = _trained_optimizer(4)
    save_resume_state(
        lora,
        step=100,
        total_steps=200,
        epoch=0,
        epoch_step=100,
        attempt=1,
        shuffle_seed=42,
        optimizer=optimizer,
        lr_scheduler=scheduler,
    )
    # A differently shaped model, as if lora_rank or target_modules had changed.
    other = torch.optim.AdamW(torch.nn.Linear(8, 8).parameters(), lr=1e-3)
    other.add_param_group({"params": list(torch.nn.Linear(3, 3).parameters())})

    with pytest.raises(ResumeStateError, match="does not fit"):
        load_optimizer_state(lora / "step-100.optim.pt", optimizer=other)


def test_fingerprint_drift_is_detected():
    saved = _fingerprint()
    current = build_fingerprint(
        lora_rank=64,
        lora_target_modules="qkv_proj,out_proj",
        learning_rate=1e-4,
        gradient_accumulation_steps=4,
        height=480,
        width=832,
        num_frames=73,
    )
    assert compare_fingerprints(saved, current) == ["lora_rank"]
    assert compare_fingerprints(saved, saved) == []


def test_cache_invalidating_geometry_is_fingerprinted():
    """480x832 / 73 frames cost 8 hours to cache; a silent change must be catchable."""
    saved = _fingerprint()
    for key, value in (("height", 512), ("width", 768), ("num_frames", 56)):
        current = dict(saved, **{key: value})
        assert compare_fingerprints(saved, current) == [key]
