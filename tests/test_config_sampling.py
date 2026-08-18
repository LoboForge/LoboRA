from pathlib import Path

from lobora.cli import resolve_config
from lobora.config import load_config
from lobora.sampling import build_sample_jobs, should_sample_at_step


def test_unknown_key_does_not_crash(tmp_path: Path):
    path = tmp_path / "cfg.yaml"
    path.write_text(
        "job:\n  name: t\n  not_a_real_field: 1\nlora:\n  rank: 8\n",
        encoding="utf-8",
    )
    cfg = load_config(path)
    assert cfg.job.name == "t"
    assert cfg.lora.rank == 8


def test_cli_set_and_flag_precedence(tmp_path: Path):
    src = Path(__file__).resolve().parents[1] / "configs" / "concept_minimal.yaml"
    cfg = resolve_config(
        type(
            "A",
            (),
            {
                "config": str(src),
                "env_file": None,
                "set": ["lora.rank=24"],
                "dataset_path": "/tmp/ds",
                "output_dir": None,
                "job_name": None,
                "model_repo": None,
                "steps": 33,
                "save_every": None,
                "lora_rank": 12,  # flag wins over --set
                "lora_alpha": None,
                "batch_size": None,
                "skip_numerics_gate": False,
            },
        )()
    )
    assert cfg.lora.rank == 12
    assert cfg.train.steps == 33
    assert cfg.dataset.folder_path == "/tmp/ds"


def test_sample_cadence():
    assert should_sample_at_step(50, sample_every=250, sample_every_early=50, sample_early_until=500)
    assert not should_sample_at_step(0, sample_every=250, sample_every_early=50, sample_early_until=500)
    assert should_sample_at_step(250, sample_every=250, sample_every_early=50, sample_early_until=100)


def test_trigger_expand():
    jobs = build_sample_jobs(
        [{"name": "a", "prompt": "photo of [trigger] smiling"}],
        trigger_word="ember",
        seed=1,
        walk_seed=False,
        tag="lora",
    )
    assert jobs[0].prompt == "photo of ember smiling"
