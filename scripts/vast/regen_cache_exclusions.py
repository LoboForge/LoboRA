#!/usr/bin/env python3
"""Regenerate the stage-2 cache exclusion list by measuring packed sequence length.

Why this script exists instead of a checked-in list: the list's entries are derived from
dataset filenames, and dataset filenames are media-derived and must never be committed.
So the repo carries the *rule*, and the box regenerates the list on demand.

    python scripts/vast/regen_cache_exclusions.py --cache $OUT/split-cache \\
        --out $OUT/cache_exclude.txt
    export DIFFSYNTH_CACHE_EXCLUDE_FILE=$OUT/cache_exclude.txt   # then relaunch stage 2

WHAT IT MEASURES

Every cached sample carries the packed sequence it will be trained on. For the ref2va
layout (`MiniMaxH3Pipeline._build_packed_ref2va`) that is:

    used     = text_len
             + ref_visual_rows          (sum over reference blocks)
             + ref_audio_rows
             + target_audio_rows        (audio_t * audio_channels, channels = 2)
             + target_video_rows        (latent_t * (latent_h // 2) * (latent_w // 2))
    seq_len  = ceil(used / 64) * 64     (_SEQ_ALIGN = 64)

At 480x832 the target video term is 390 rows per latent frame ((480/16/2) * (832/16/2)),
so the familiar shorthand is

    seq_len ~ 390 * latent_frames + 2 * ref_tokens + text_tokens + audio_tokens, /64-aligned

The `2 *` on the reference term is not a typo. Reference-image conditioning enters the
sequence TWICE: once as visual rows in the image stream, and once again inside `text_len`,
because the references are encoded by the Qwen3-VL processor into `prompt_embeds`. On the
measured run that term was 54-78% of total sequence length -- it dominates. Which is why
clip length and source fps are near-irrelevant at this stage, and why the only samples
worth excluding are the ones carrying unusually many or unusually large references.

This script does not recompute the formula; it reads the recorded `packed["seq_len"]` out
of each cached `.pth`, which is the number the model will actually see.

WHY EXCLUSION IS THE ONLY LEVER

The trainer builds its loader with `collate_fn=lambda x: x[0]`, i.e. batch size 1. Peak
memory is therefore set by the single largest sample in the set, not by an average. No
amount of reordering, shuffling or length-bucketing changes a maximum -- dropping the
sample is the only thing that lowers it. Measured on the run this was written for:
excluding 17 of 963 samples at `seq_len >= 38080` took peak VRAM from 77.5 to 76.35 GiB
(2.79 GiB headroom) and turned OOMs at 7h42m and 3h10m into a 10h+ clean run.

The filter is applied at READ time by `UnifiedDataset` (see
`patches/diffsynth/site-packages/diffsynth_cache_exclude.diff`). It never writes, renames
or deletes anything under the cache, so the cache is NOT invalidated and reverting is
`unset DIFFSYNTH_CACHE_EXCLUDE_FILE`. It does change `len(dataset)`, so the tqdm total
drops by `excluded * dataset_repeat`.

PRIVACY

Cache entry keys can be media-derived filenames. This script writes them to `--out` and
never prints them: stdout is counts, thresholds and sequence-length quantiles only. It
refuses to write `--out` inside a git work tree so the list cannot be committed by
accident. `--hashes` adds opaque sha256 prefixes of the key text as a reproducibility
aid; a hash is not a filename and reveals nothing about content.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
from pathlib import Path

DEFAULT_THRESHOLD = 38080


def find_cached(cache: Path) -> list[Path]:
    # Same walk UnifiedDataset.search_for_cached_data_files does: recursive, *.pth.
    return sorted(p for p in cache.rglob("*.pth") if p.is_file())


def find_seq_len(obj, depth: int = 0):
    """Pull `packed["seq_len"]` out of a cached sample, whatever it is nested inside."""
    if depth > 6:
        return None
    if isinstance(obj, dict):
        packed = obj.get("packed")
        if isinstance(packed, dict) and "seq_len" in packed:
            return as_int(packed["seq_len"])
        if "seq_len" in obj:
            return as_int(obj["seq_len"])
        for value in obj.values():
            found = find_seq_len(value, depth + 1)
            if found is not None:
                return found
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            found = find_seq_len(value, depth + 1)
            if found is not None:
                return found
    return None


def as_int(value):
    try:
        if hasattr(value, "item"):
            return int(value.item())
        return int(value)
    except Exception:  # noqa: BLE001 - a shape we do not understand is "not found"
        return None


def load_seq_len(path: Path, torch):
    # mmap first: a cached sample is ~150 MB and we only need one scalar out of it, so
    # deserialising the whole tensor payload for 963 files would be an hour of pointless
    # IO. Not every torch version / save format supports it, hence the fallback.
    for kwargs in ({"mmap": True, "weights_only": False}, {"weights_only": False}, {}):
        try:
            return find_seq_len(torch.load(path, map_location="cpu", **kwargs))
        except TypeError:
            continue
        except Exception:  # noqa: BLE001
            continue
    return None


def inside_git_worktree(path: Path) -> bool:
    # Ask about the nearest EXISTING ancestor of the target. Falling back to the cwd
    # would flag every not-yet-created path just because this script was run from a repo.
    probe = path.absolute().parent
    while not probe.is_dir() and probe != probe.parent:
        probe = probe.parent
    try:
        done = subprocess.run(["git", "-C", str(probe), "rev-parse", "--is-inside-work-tree"],
                              capture_output=True, text=True, timeout=15)
    except Exception:  # noqa: BLE001 - no git, no repo to protect
        return False
    return done.returncode == 0 and done.stdout.strip() == "true"


def quantiles(values: list[int]) -> str:
    ordered = sorted(values)
    def at(fraction: float) -> int:
        return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]
    return (f"min={ordered[0]} p50={at(0.50)} p90={at(0.90)} p99={at(0.99)} "
            f"max={ordered[-1]}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Regenerate the DIFFSYNTH_CACHE_EXCLUDE_FILE list from the cache.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--cache", type=Path, required=True, help="stage-1 cache dir ($OUT/split-cache)")
    ap.add_argument("--out", type=Path, help="write the exclusion list here (omit for a dry run)")
    ap.add_argument("--threshold", type=int, default=DEFAULT_THRESHOLD,
                    help=f"exclude samples with packed seq_len >= this (default {DEFAULT_THRESHOLD})")
    ap.add_argument("--key", choices=("stem", "name", "path"), default="stem",
                    help="how to name an excluded sample; the filter matches all three")
    ap.add_argument("--hashes", action="store_true",
                    help="also print opaque sha256 prefixes of the keys (not filenames)")
    ap.add_argument("--allow-in-repo", action="store_true",
                    help="permit --out inside a git work tree (you almost certainly do not want this)")
    args = ap.parse_args()

    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"error: torch is required to read the cache ({exc})")

    if not args.cache.is_dir():
        raise SystemExit(f"error: no cache directory at {args.cache}")
    if args.out and not args.allow_in_repo and inside_git_worktree(args.out):
        raise SystemExit(
            f"error: refusing to write {args.out} inside a git work tree. The keys are "
            f"derived from dataset filenames and must never be committed. Point --out at "
            f"something like $OUT/cache_exclude.txt, or pass --allow-in-repo if you are "
            f"certain the location is ignored."
        )

    cached = find_cached(args.cache)
    if not cached:
        raise SystemExit(f"error: no .pth files under {args.cache}")
    print(f"cache      : {args.cache}")
    print(f"samples    : {len(cached)}")
    print(f"threshold  : seq_len >= {args.threshold}")
    print("reading packed seq_len from each sample (mmap, one scalar per file) ...")

    lengths: dict[Path, int] = {}
    unreadable: list[Path] = []
    for index, path in enumerate(cached, 1):
        seq_len = load_seq_len(path, torch)
        if seq_len is None:
            unreadable.append(path)
        else:
            lengths[path] = seq_len
        if index % 100 == 0 or index == len(cached):
            print(f"  {index}/{len(cached)}", flush=True)

    if not lengths:
        raise SystemExit("error: no sample yielded a packed seq_len; is this a stage-1 cache?")

    excluded = sorted(p for p, n in lengths.items() if n >= args.threshold)
    keep = len(lengths) - len(excluded)
    print()
    print(f"measured   : {len(lengths)}   unreadable: {len(unreadable)}")
    print(f"seq_len    : {quantiles(list(lengths.values()))}")
    print(f"excluded   : {len(excluded)}   remaining: {keep}")
    if excluded:
        over = [lengths[p] for p in excluded]
        print(f"  excluded seq_len range {min(over)}..{max(over)}")

    def key_of(path: Path) -> str:
        return {"stem": path.stem, "name": path.name, "path": str(path)}[args.key]

    if args.hashes:
        print("opaque key hashes (sha256/12, not filenames):")
        for path in excluded:
            print(f"  {hashlib.sha256(key_of(path).encode()).hexdigest()[:12]}  seq_len={lengths[path]}")

    if not args.out:
        print("\ndry run: pass --out to write the list.")
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(f"{key_of(p)}\n" for p in excluded)
    args.out.write_text(
        f"# regenerated by scripts/vast/regen_cache_exclusions.py\n"
        f"# cache={args.cache} threshold={args.threshold} key={args.key}\n"
        f"# {len(excluded)} of {len(lengths)} samples excluded; {keep} remain\n"
        f"# CONTAINS DATASET-DERIVED NAMES -- do not commit, do not paste into chat.\n"
        f"{body}", encoding="utf-8")
    print(f"\nwrote {len(excluded)} keys to {args.out}")
    print(f"next: export DIFFSYNTH_CACHE_EXCLUDE_FILE={args.out}   # then relaunch stage 2")
    print("the cache itself is untouched, so this does not invalidate it")
    return 0


if __name__ == "__main__":
    sys.exit(main())
