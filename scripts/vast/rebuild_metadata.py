#!/usr/bin/env python3
"""Rebuild DiffSynth Ref2VA metadata.json from a media + sidecar-caption folder.

Privacy: this script reads caption sidecars only to place them in the JSON. It never
prints caption text, never prints media file names, and never inspects pixels. Keep it
that way -- the summary at the end is counts only.

What it does, and why:

  * every video with a NON-EMPTY same-name .txt sidecar becomes one row
  * Ref2VA needs at least one reference image per row, so the video's own first frame
    is extracted to _train_refs/ with ffmpeg and used as the reference
  * every still with a non-empty sidecar is looped to a NUM_FRAMES clip in
    _train_clips/ at FPS, because the trainer's video path is the only path; H3 wants
    num_frames % 17 == 5 so a still cannot be fed as a single frame
  * reference stills already listed in an older metadata.json are carried forward
  * rows without a caption or without any usable reference are dropped, counted, and
    reported rather than silently emitted as broken rows

Writes atomically (tmp + replace) and re-reads the result to verify before exiting 0.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

SKIP_DIRS = {"_train_refs", "_train_clips", ".ipynb_checkpoints"}
VIDEO_EXT = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def log(msg: str) -> None:
    print(msg, flush=True)


def nonempty_sidecar(media: Path) -> Path | None:
    t = media.with_suffix(".txt")
    try:
        if t.is_file() and t.stat().st_size > 0:
            return t
    except OSError:
        return None
    return None


def read_prompt(sidecar: Path) -> str:
    # Read for JSON only. Callers must never print the return value.
    return sidecar.read_text(encoding="utf-8", errors="replace").strip()


def ffmpeg_ok(args: list[str]) -> bool:
    try:
        r = subprocess.run(
            args,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=120,
        )
        return r.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


def extract_first_frame(video: Path, dest: Path) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp.jpg")
    ok = ffmpeg_ok(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-ss", "0", "-i", str(video), "-frames:v", "1", "-q:v", "2", str(tmp)]
    )
    if ok and tmp.is_file() and tmp.stat().st_size > 0:
        tmp.replace(dest)
        return True
    tmp.unlink(missing_ok=True)
    return False


def still_to_clip(image: Path, dest: Path, num_frames: int, fps: int) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp.mp4")
    # Even-dimension scale only (yuv420p requirement); no content inspection.
    ok = ffmpeg_ok(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-loop", "1", "-framerate", str(fps), "-i", str(image),
         "-frames:v", str(num_frames),
         "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an", str(tmp)]
    )
    if ok and tmp.is_file() and tmp.stat().st_size > 0:
        tmp.replace(dest)
        return True
    tmp.unlink(missing_ok=True)
    return False


def walk_media(root: Path) -> tuple[list[Path], list[Path]]:
    videos: list[Path] = []
    images: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
        d = Path(dirpath)
        for fn in filenames:
            p = d / fn
            ext = p.suffix.lower()
            if ext in VIDEO_EXT:
                videos.append(p)
            elif ext in IMAGE_EXT:
                images.append(p)
    videos.sort()
    images.sort()
    return videos, images


def load_old_rows(meta: Path) -> dict[str, dict]:
    if not meta.is_file():
        return {}
    try:
        data = json.loads(meta.read_text())
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, list):
        return {}
    return {
        row["video"]: row
        for row in data
        if isinstance(row, dict) and isinstance(row.get("video"), str)
    }


def carried_stills(root: Path, refs_dirname: str, old_row: dict | None) -> list[dict]:
    """Reference stills from a previous metadata.json that still exist on disk."""
    out: list[dict] = []
    if not old_row:
        return out
    seen: set[str] = set()
    for item in old_row.get("references") or []:
        if not isinstance(item, dict):
            continue
        img = item.get("image")
        if not isinstance(img, str) or not img or img in seen:
            continue
        # Generated first frames are re-derived below; only carry hand-supplied stills.
        if img == refs_dirname or img.startswith(refs_dirname + "/"):
            continue
        p = root / img
        try:
            if not p.is_file() or p.stat().st_size <= 0:
                continue
        except OSError:
            continue
        seen.add(img)
        out.append({"type": "image", "image": img})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Rebuild Ref2VA metadata.json (counts only, no captions printed)")
    ap.add_argument("--dataset", type=Path, default=Path(os.environ.get("DATASET", "/workspace/dataset")))
    ap.add_argument("--num-frames", type=int, default=int(os.environ.get("NUM_FRAMES", 73)))
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    root: Path = args.dataset
    if not root.is_dir():
        log(f"error: dataset {root} does not exist")
        return 2
    meta = root / "metadata.json"
    refs_dirname, clips_dirname = "_train_refs", "_train_clips"
    refs, clips = root / refs_dirname, root / clips_dirname

    stats = dict.fromkeys(
        (
            "videos_captioned", "videos_included", "videos_skipped_empty_prompt",
            "videos_skipped_no_ref", "videos_skipped_missing",
            "images_captioned", "images_included", "images_skipped_empty_prompt",
            "images_skipped_ffmpeg", "images_skipped_missing",
            "first_frames_reused", "first_frames_extracted", "first_frames_failed",
            "old_rows", "new_rows",
        ),
        0,
    )

    old = load_old_rows(meta)
    stats["old_rows"] = len(old)
    videos, images = walk_media(root)
    refs.mkdir(parents=True, exist_ok=True)
    clips.mkdir(parents=True, exist_ok=True)

    def rel(p: Path) -> str:
        return str(p.relative_to(root))

    rows: list[dict] = []

    for vp in videos:
        if not vp.is_file() or vp.stat().st_size <= 0:
            stats["videos_skipped_missing"] += 1
            continue
        side = nonempty_sidecar(vp)
        if not side:
            continue
        stats["videos_captioned"] += 1
        prompt = read_prompt(side)
        if not prompt:
            stats["videos_skipped_empty_prompt"] += 1
            continue
        vrel = rel(vp)
        dest = refs / (vp.stem + ".jpg")
        if dest.is_file() and dest.stat().st_size > 0:
            stats["first_frames_reused"] += 1
            have_first = True
        else:
            have_first = extract_first_frame(vp, dest)
            stats["first_frames_extracted" if have_first else "first_frames_failed"] += 1
        row_refs: list[dict] = []
        if have_first and dest.is_file() and dest.stat().st_size > 0:
            row_refs.append({"type": "image", "image": rel(dest)})
        row_refs.extend(carried_stills(root, refs_dirname, old.get(vrel)))
        if not row_refs:
            stats["videos_skipped_no_ref"] += 1
            continue
        rows.append(
            {
                "video": vrel,
                "prompt": prompt,
                "input_audio": vrel,
                "references": row_refs,
                "frame_rate": args.fps,
            }
        )
        stats["videos_included"] += 1

    jobs: list[tuple[Path, Path, Path]] = []
    for ip in images:
        if not ip.is_file() or ip.stat().st_size <= 0:
            stats["images_skipped_missing"] += 1
            continue
        side = nonempty_sidecar(ip)
        if not side:
            continue
        stats["images_captioned"] += 1
        if not read_prompt(side):
            stats["images_skipped_empty_prompt"] += 1
            continue
        jobs.append((ip, side, clips / ip.relative_to(root).with_suffix(".mp4")))

    def convert(job: tuple[Path, Path, Path]) -> tuple[bool, Path, Path, Path]:
        ip, side, clip = job
        if clip.is_file() and clip.stat().st_size > 0:
            return True, ip, side, clip
        return still_to_clip(ip, clip, args.num_frames, args.fps), ip, side, clip

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for fut in as_completed([ex.submit(convert, j) for j in jobs]):
            ok, ip, side, clip = fut.result()
            if not ok or not clip.is_file() or clip.stat().st_size <= 0:
                stats["images_skipped_ffmpeg"] += 1
                continue
            prompt = read_prompt(side)
            if not prompt:
                stats["images_skipped_empty_prompt"] += 1
                continue
            crel = rel(clip)
            rows.append(
                {
                    "video": crel,
                    "prompt": prompt,
                    "input_audio": crel,
                    "references": [{"type": "image", "image": rel(ip)}],
                    "frame_rate": args.fps,
                }
            )
            stats["images_included"] += 1

    stats["new_rows"] = len(rows)
    tmp = meta.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(meta)

    verify = json.loads(meta.read_text())
    n = len(verify)
    prompt_ok = sum(
        1 for r in verify if isinstance(r.get("prompt"), str) and r["prompt"].strip()
    )
    ref_lens: dict[int, int] = {}
    for r in verify:
        k = len(r.get("references") or [])
        ref_lens[k] = ref_lens.get(k, 0) + 1
    log("REBUILD_STATS " + json.dumps(stats, sort_keys=True))
    log(
        "VERIFY "
        + json.dumps(
            {
                "rows": n,
                "prompt_nonempty": prompt_ok,
                "keys": sorted({k for r in verify for k in r}),
                "ref_lens": ref_lens,
                "meta_bytes": meta.stat().st_size,
            }
        )
    )
    if n == 0 or prompt_ok != n:
        log("ERROR metadata invalid")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
