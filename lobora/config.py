"""YAML config with Lens-style precedence and unknown-key warnings."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from lobora.console import warn


@dataclass
class JobConfig:
    name: str = "h3-lora"
    output_dir: str = "./output/h3-lora"


@dataclass
class ModelConfig:
    repo_id: str = "MiniMaxAI/MiniMax-H3"
    variant: str = "ref2va"  # ref2va | fl2va
    precision: str = "bf16"  # bf16 | nf4 | fp8_frozen
    model_rev: str = "main"
    cache_text_embeddings: bool = True
    cache_latents: bool = True
    fp8_frozen_dit: bool = True


@dataclass
class DatasetConfig:
    folder_path: str = "./dataset"
    caption_ext: str = "txt"
    metadata_path: str = ""  # optional Ref2VA JSON / FL2VA CSV
    max_pixels: int = 480 * 832
    max_frames: int = 124
    default_frames: int = 73
    min_bucket_size: int = 1
    allow_image_samples: bool = True
    fps: int = 24


@dataclass
class LoraConfig:
    rank: int = 32
    alpha: int = 32
    dropout: float = 0.0
    target_modules: list[str] = field(default_factory=lambda: ["qkv_proj", "out_proj"])
    extra_ffn_modules: list[str] = field(default_factory=list)


@dataclass
class TrainConfig:
    steps: int = 2000
    batch_size: int = 1
    gradient_accumulation_steps: int = 2
    learning_rate: float = 1e-4
    optimizer: str = "adamw"  # adamw | adamw8bit
    weight_decay: float = 0.01
    gradient_checkpointing: bool = True
    max_grad_norm: float = 1.0
    save_every: int = 250
    seed: int = 42
    resume_from: str = ""
    skip_numerics_gate: bool = False
    numerics_gate_steps: int = 50
    numerics_loss_min: float = 0.15
    numerics_loss_max: float = 1.5
    flow_shift: float = 2.22
    audio_flow_shift: float = 2.22
    num_train_timesteps: int = 1000


@dataclass
class SampleConfig:
    prompts: list[Any] = field(
        default_factory=lambda: [
            {
                "name": "establishing",
                "prompt": "[trigger] walking through a sunlit courtyard, medium wide, slow dolly in",
            },
            {
                "name": "portrait",
                "prompt": "close-up of [trigger], soft window light, shallow depth of field",
            },
        ]
    )
    trigger_word: str = "h3lora"
    width: int = 480
    height: int = 832
    frames: int = 22
    steps: int = 12
    seed: int = 42
    walk_seed: bool = True
    baseline_control: bool = True
    sample_every: int = 250
    sample_every_early: int = 50
    sample_early_until: int = 500


@dataclass
class TrainerConfig:
    job: JobConfig = field(default_factory=JobConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    sample: SampleConfig = field(default_factory=SampleConfig)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _section(data: dict[str, Any], cls: type, key: str):
    section = data.get(key, {}) or {}
    if not isinstance(section, dict):
        raise ValueError(f"Config section '{key}' must be a mapping")
    allowed = {f.name for f in fields(cls)}
    unknown = sorted(k for k in section if k not in allowed)
    if unknown:
        warn(f"unknown keys in '{key}': {', '.join(unknown)} (ignored)")
    filtered = {k: v for k, v in section.items() if k in allowed}
    return cls(**filtered)


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> TrainerConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError("Config root must be a mapping")
    if overrides:
        raw = deep_merge(raw, overrides)
    return TrainerConfig(
        job=_section(raw, JobConfig, "job"),
        model=_section(raw, ModelConfig, "model"),
        dataset=_section(raw, DatasetConfig, "dataset"),
        lora=_section(raw, LoraConfig, "lora"),
        train=_section(raw, TrainConfig, "train"),
        sample=_section(raw, SampleConfig, "sample"),
    )


def set_dotted(overrides: dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    cursor = overrides
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
        if not isinstance(cursor, dict):
            raise ValueError(f"Cannot set '{dotted}': '{part}' is not a mapping")
    cursor[parts[-1]] = _coerce(value)


def _coerce(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "yes"}:
        return True
    if lowered in {"false", "no"}:
        return False
    if lowered in {"null", "none", ""}:
        return ""
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def dump_resolved(cfg: TrainerConfig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(cfg), indent=2) + "\n", encoding="utf-8")
