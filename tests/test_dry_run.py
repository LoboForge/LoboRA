import json
from pathlib import Path

import torch
from PIL import Image

from lobora.cli import main
from lobora.resume_state import STATE_FILENAME, read_state_file


def test_dry_run_train(tmp_path: Path):
    data = tmp_path / "data"
    data.mkdir()
    Image.new("RGB", (480, 832), color=(20, 80, 20)).save(data / "s.png")
    (data / "s.txt").write_text("a green field, wide shot", encoding="utf-8")
    out = tmp_path / "out"
    cfg = Path(__file__).resolve().parents[1] / "configs" / "concept_minimal.yaml"
    rc = main(
        [
            str(cfg),
            "--dataset-path",
            str(data),
            "--output-dir",
            str(out),
            "--steps",
            "4",
            "--save-every",
            "2",
            "--dry-run",
            "--skip-numerics-gate",
            "--set",
            "train.gradient_accumulation_steps=1",
        ]
    )
    assert rc == 0
    assert (out / "lora_final.safetensors").is_file()
    assert (out / "lora_latest.safetensors").is_file()
    assert (out / "config.resolved.json").is_file()
    assert list((out / "checkpoints").glob("lora_step_*.safetensors"))


def _dataset(tmp_path: Path) -> Path:
    data = tmp_path / "data"
    data.mkdir()
    Image.new("RGB", (480, 832), color=(20, 80, 20)).save(data / "s.png")
    (data / "s.txt").write_text("a green field, wide shot", encoding="utf-8")
    return data


def _run(data: Path, out: Path, steps: int, *extra: str) -> int:
    cfg = Path(__file__).resolve().parents[1] / "configs" / "concept_minimal.yaml"
    return main(
        [
            str(cfg),
            "--dataset-path", str(data),
            "--output-dir", str(out),
            "--steps", str(steps),
            "--save-every", "2",
            "--dry-run",
            "--skip-numerics-gate",
            "--set", "train.gradient_accumulation_steps=1",
            *extra,
        ]
    )


def test_dry_run_resume_continues_the_cumulative_step(tmp_path: Path):
    """A second invocation must continue at 4, not restart at 0."""
    data = _dataset(tmp_path)
    out = tmp_path / "out"
    checkpoints = out / "checkpoints"

    assert _run(data, out, 4) == 0
    first = read_state_file(checkpoints)
    assert first.cumulative_step == 4
    first_names = {p.name for p in checkpoints.glob("lora_step_*.safetensors")}
    sidecar = checkpoints / "lora_step_000004.optim.pt"
    assert sidecar.is_file(), "optimizer state was not persisted"
    moments_before = torch.load(sidecar, map_location="cpu", weights_only=False)
    assert moments_before["lr_scheduler"]["last_epoch"] == 4

    assert _run(data, out, 8, "--resume", "latest") == 0
    second = read_state_file(checkpoints)
    assert second.cumulative_step == 8
    assert [entry["step"] for entry in second.history][-1] == 6

    # The first run's checkpoints are still there under their own step numbers, and
    # the second run only added higher ones.
    names = {p.name for p in checkpoints.glob("lora_step_*.safetensors")}
    assert first_names <= names
    assert {"lora_step_000006.safetensors", "lora_step_000008.safetensors"} <= names

    # The scheduler kept walking rather than restarting its position.
    resumed = torch.load(checkpoints / "lora_step_000008.optim.pt", map_location="cpu", weights_only=False)
    assert resumed["lr_scheduler"]["last_epoch"] == 8
    assert resumed["step"] == 8
    assert json.loads((checkpoints / STATE_FILENAME).read_text())["cumulative_step"] == 8
