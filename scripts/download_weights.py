#!/usr/bin/env python3
"""Download MiniMax-H3 Ref2VA bf16 pieces (DiT + TE + VAEs). ~124 GB."""

from __future__ import annotations

import argparse
from pathlib import Path

REF2VA_ALLOW = [
    "Ref2VA/",
    "text_encoder/",
    "video_vae/",
    "audio_vae/",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="MiniMaxAI/MiniMax-H3")
    parser.add_argument("--dest", default="./models/MiniMax-H3")
    parser.add_argument("--revision", default="main")
    args = parser.parse_args()

    from huggingface_hub import snapshot_download

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=args.repo,
        local_dir=str(dest),
        revision=args.revision,
        allow_patterns=[f"*{p}*" for p in REF2VA_ALLOW],
    )
    print(f"downloaded {args.repo} → {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
