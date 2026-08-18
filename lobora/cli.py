"""YAML + env-file + --set + flags. Precedence: YAML < env < --set < flags."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from lobora.config import TrainerConfig, deep_merge, load_config, set_dotted
from lobora.console import info
from lobora.trainer import LoboRATrainer, run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lobora",
        description="Train MiniMax-H3 LoRAs with resume, two-stage cache, and mixed buckets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("config", nargs="?", help="YAML preset")
    parser.add_argument("--env-file", type=str, default=None)
    parser.add_argument("--dataset-path", type=str)
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--job-name", type=str)
    parser.add_argument("--model-repo", type=str)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--save-every", type=int)
    parser.add_argument("--lora-rank", type=int)
    parser.add_argument("--lora-alpha", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--resume", type=str, default=argparse.SUPPRESS)
    parser.add_argument("--skip-numerics-gate", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="CPU dummy loop (no DiffSynth)")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="Dotted override: --set lora.rank=16",
    )
    parser.add_argument(
        "--task",
        choices=("train", "cache", "gate"),
        default="train",
    )
    return parser


def _load_env_file(path: Path) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())
        if key.strip().startswith("LOBORA_"):
            dotted = key.strip()[len("LOBORA_") :].lower().replace("__", ".")
            set_dotted(overrides, dotted, value.strip())
    return overrides


def cli_overrides(args: argparse.Namespace) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if args.dataset_path:
        set_dotted(out, "dataset.folder_path", args.dataset_path)
    if args.output_dir:
        set_dotted(out, "job.output_dir", args.output_dir)
    if args.job_name:
        set_dotted(out, "job.name", args.job_name)
    if args.model_repo:
        set_dotted(out, "model.repo_id", args.model_repo)
    if args.steps is not None:
        set_dotted(out, "train.steps", str(args.steps))
    if args.save_every is not None:
        set_dotted(out, "train.save_every", str(args.save_every))
    if args.lora_rank is not None:
        set_dotted(out, "lora.rank", str(args.lora_rank))
    if args.lora_alpha is not None:
        set_dotted(out, "lora.alpha", str(args.lora_alpha))
    if args.batch_size is not None:
        set_dotted(out, "train.batch_size", str(args.batch_size))
    if hasattr(args, "resume"):
        set_dotted(out, "train.resume_from", args.resume)
    if args.skip_numerics_gate:
        set_dotted(out, "train.skip_numerics_gate", "true")
    return out


def resolve_config(args: argparse.Namespace) -> TrainerConfig:
    if not args.config:
        raise SystemExit("config YAML is required")
    overrides: dict[str, Any] = {}
    if args.env_file:
        overrides = deep_merge(overrides, _load_env_file(Path(args.env_file)))
    for item in args.set:
        if "=" not in item:
            raise SystemExit(f"--set expects key=value, got {item!r}")
        key, value = item.split("=", 1)
        set_dotted(overrides, key.strip(), value.strip())
    overrides = deep_merge(overrides, cli_overrides(args))
    return load_config(args.config, overrides)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = resolve_config(args)
    info(f"job={cfg.job.name}  out={cfg.job.output_dir}  steps={cfg.train.steps}")
    trainer = LoboRATrainer(cfg, dry_run=args.dry_run)
    if args.task == "cache":
        trainer.prepare_data()
        return 0
    if args.task == "gate":
        trainer.run_numerics_gate(force=True)
        return 0
    run(cfg, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
