#!/usr/bin/env python3
"""Emit the two --model_paths JSON files the DiffSynth H3 trainer takes.

The two stages want *different* model sets, and getting this wrong is expensive:

  stage 1 (sft:data_process) needs text encoder + video VAE + audio VAE, because
          it is encoding the dataset to latents. It does not need the DiT.
  stage 2 (sft:train) needs ONLY the sharded DiT. Loading the text encoder or the
          VAEs here wastes tens of GiB of VRAM for nothing -- stage 2 reads
          latents straight off the cache.

Sharded groups must be passed as a nested list so DiffSynth treats them as one
model. Shards are ordered by their model-NNNNN-of-NNNNN suffix, not by string
sort, so a 14-shard group does not come out as 1, 10, 11, ... , 2.

    make_model_paths.py --models-root /workspace/models/MiniMax-H3 --out-dir /workspace/output
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

SHARD_RE = re.compile(r"model-(\d+)-of-(\d+)\.safetensors$")


def shard_group(directory: Path) -> list[str]:
    """All shards in `directory`, ordered by shard index, as a nested group."""
    shards = sorted(
        (p for p in directory.glob("model-*-of-*.safetensors") if SHARD_RE.search(p.name)),
        key=lambda p: int(SHARD_RE.search(p.name).group(1)),
    )
    if not shards:
        raise SystemExit(f"error: no model-*-of-*.safetensors shards in {directory}")
    declared = int(SHARD_RE.search(shards[0].name).group(2))
    if len(shards) != declared:
        raise SystemExit(
            f"error: {directory} has {len(shards)} shards but they declare {declared} "
            f"-- the download is incomplete, do not start a run on it"
        )
    return [str(p) for p in shards]


def single(path: Path) -> str:
    if not path.is_file():
        raise SystemExit(f"error: missing {path}")
    return str(path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models-root", type=Path, default=Path("/workspace/models/MiniMax-H3"))
    ap.add_argument("--out-dir", type=Path, default=Path("/workspace/output"))
    ap.add_argument("--variant", default="Ref2VA")
    args = ap.parse_args()

    root = args.models_root / args.variant
    args.out_dir.mkdir(parents=True, exist_ok=True)

    stage1 = [
        shard_group(root / "text_encoder"),
        single(root / "video_vae" / "source" / "model.safetensors"),
        single(root / "audio_vae" / "model.safetensors"),
    ]
    stage2 = [shard_group(root / "transformer")]

    for name, payload in (("stage1_model_paths.json", stage1), ("stage2_model_paths.json", stage2)):
        dest = args.out_dir / name
        tmp = dest.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(dest)
        print(f"wrote {dest}")

    print(f"stage1: text_encoder x{len(stage1[0])} + video_vae + audio_vae")
    print(f"stage2: transformer x{len(stage2[0])} (DiT only)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
