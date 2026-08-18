from pathlib import Path

from PIL import Image

from lobora.cli import main


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
