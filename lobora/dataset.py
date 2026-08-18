"""Folder walk + DiffSynth-compatible Ref2VA JSON / FL2VA CSV."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from PIL import Image

from lobora.buckets import BucketKey, assign_buckets, bucket_histogram
from lobora.console import info, warn
from lobora.grid import DEFAULT_FPS, MIN_FRAMES

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm"}


@dataclass
class Reference:
    type: str
    path: str
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Sample:
    sample_id: str
    path: str
    caption: str
    kind: str  # video | image
    height: int
    width: int
    num_frames: int
    fps: int = DEFAULT_FPS
    audio_path: str = ""
    references: list[Reference] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def stem(self) -> str:
        return Path(self.path).stem


def _probe_image(path: Path) -> tuple[int, int]:
    with Image.open(path) as im:
        return im.size[1], im.size[0]  # H, W


def _probe_video(path: Path, default_frames: int) -> tuple[int, int, int]:
    """Best-effort ffprobe; fall back to defaults if ffmpeg is missing."""
    import json as _json
    import shutil
    import subprocess

    if not shutil.which("ffprobe"):
        return 832, 480, default_frames
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,nb_frames,avg_frame_rate",
        "-of",
        "json",
        str(path),
    ]
    try:
        out = subprocess.check_output(cmd, text=True, timeout=30)
        stream = (_json.loads(out).get("streams") or [{}])[0]
        width = int(stream.get("width") or 480)
        height = int(stream.get("height") or 832)
        frames = stream.get("nb_frames")
        if frames and str(frames).isdigit() and int(frames) > 0:
            num_frames = int(frames)
        else:
            rate = stream.get("avg_frame_rate") or "24/1"
            num, den = rate.split("/")
            fps = float(num) / max(float(den), 1e-6)
            # duration via format
            dur_cmd = [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "json",
                str(path),
            ]
            dur_out = subprocess.check_output(dur_cmd, text=True, timeout=30)
            duration = float((_json.loads(dur_out).get("format") or {}).get("duration") or 3.0)
            num_frames = max(MIN_FRAMES, int(duration * fps))
        return height, width, num_frames
    except Exception:
        return 832, 480, default_frames


def _read_caption(path: Path, caption_ext: str) -> str:
    cap = path.with_suffix(f".{caption_ext.lstrip('.')}")
    if not cap.is_file():
        return ""
    return cap.read_text(encoding="utf-8").strip()


def scan_folder(
    folder: Path,
    *,
    caption_ext: str = "txt",
    allow_image_samples: bool = True,
    default_frames: int = 73,
    fps: int = DEFAULT_FPS,
) -> tuple[list[Sample], list[dict[str, str]]]:
    folder = folder.resolve()
    samples: list[Sample] = []
    skipped: list[dict[str, str]] = []
    if not folder.is_dir():
        raise FileNotFoundError(f"dataset folder not found: {folder}")

    for path in sorted(folder.rglob("*")):
        if not path.is_file():
            continue
        ext = path.suffix.lower()
        if ext not in IMAGE_EXTS | VIDEO_EXTS:
            continue
        caption = _read_caption(path, caption_ext)
        if not caption:
            skipped.append({"path": str(path), "reason": "missing_or_empty_caption"})
            continue
        rel = str(path.relative_to(folder))
        if ext in IMAGE_EXTS:
            if not allow_image_samples:
                skipped.append({"path": str(path), "reason": "image_samples_disabled"})
                continue
            height, width = _probe_image(path)
            samples.append(
                Sample(
                    sample_id=rel,
                    path=str(path),
                    caption=caption,
                    kind="image",
                    height=height,
                    width=width,
                    num_frames=1,
                    fps=fps,
                )
            )
            continue
        height, width, num_frames = _probe_video(path, default_frames)
        audio_sidecar = path.with_suffix(".wav")
        samples.append(
            Sample(
                sample_id=rel,
                path=str(path),
                caption=caption,
                kind="video",
                height=height,
                width=width,
                num_frames=num_frames,
                fps=fps,
                audio_path=str(audio_sidecar) if audio_sidecar.is_file() else str(path),
            )
        )
    return samples, skipped


def load_ref2va_json(path: Path, *, default_frames: int, fps: int) -> list[Sample]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Ref2VA metadata.json must be a list of rows")
    base = path.parent
    samples: list[Sample] = []
    for i, row in enumerate(raw):
        video = row.get("video")
        prompt = (row.get("prompt") or "").strip()
        if not video or not prompt:
            continue
        media = (base / video).resolve()
        height, width, num_frames = _probe_video(media, default_frames)
        refs = []
        for ref in row.get("references") or []:
            refs.append(
                Reference(
                    type=str(ref.get("type") or "image"),
                    path=str((base / (ref.get("image") or ref.get("video") or ref.get("audio") or "")).resolve()),
                    extra={k: v for k, v in ref.items() if k not in {"type", "image", "video", "audio"}},
                )
            )
        samples.append(
            Sample(
                sample_id=row.get("id") or f"ref2va_{i:04d}",
                path=str(media),
                caption=prompt,
                kind="video",
                height=height,
                width=width,
                num_frames=int(row.get("num_frames") or num_frames),
                fps=int(row.get("frame_rate") or fps),
                audio_path=str((base / row["input_audio"]).resolve()) if row.get("input_audio") else str(media),
                references=refs,
                extras=row,
            )
        )
    return samples


def load_fl2va_csv(path: Path, *, default_frames: int, fps: int) -> list[Sample]:
    base = path.parent
    samples: list[Sample] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for i, row in enumerate(reader):
            video = (row.get("video") or "").strip()
            prompt = (row.get("prompt") or "").strip()
            if not video or not prompt:
                continue
            media = (base / video).resolve()
            height, width, num_frames = _probe_video(media, default_frames)
            samples.append(
                Sample(
                    sample_id=row.get("id") or f"fl2va_{i:04d}",
                    path=str(media),
                    caption=prompt,
                    kind="video",
                    height=height,
                    width=width,
                    num_frames=int(row.get("num_frames") or num_frames),
                    fps=int(row.get("frame_rate") or fps),
                    audio_path=str((base / row["input_audio"]).resolve()) if row.get("input_audio") else str(media),
                    extras=row,
                )
            )
    return samples


def load_dataset(
    folder: Path,
    *,
    caption_ext: str = "txt",
    metadata_path: str = "",
    allow_image_samples: bool = True,
    default_frames: int = 73,
    fps: int = DEFAULT_FPS,
    max_pixels: int | None = None,
    max_frames: int = 124,
    min_bucket_size: int = 1,
) -> tuple[list[Sample], dict[BucketKey, list[int]], list[dict[str, str]]]:
    skipped: list[dict[str, str]] = []
    samples: list[Sample] = []

    meta = Path(metadata_path) if metadata_path else None
    if meta and meta.is_file():
        suffix = meta.suffix.lower()
        if suffix == ".json":
            samples.extend(load_ref2va_json(meta, default_frames=default_frames, fps=fps))
        elif suffix == ".csv":
            samples.extend(load_fl2va_csv(meta, default_frames=default_frames, fps=fps))
        else:
            raise ValueError(f"Unsupported metadata file: {meta}")
        info(f"loaded {len(samples)} rows from {meta.name}")

    if folder.is_dir():
        folder_samples, folder_skipped = scan_folder(
            folder,
            caption_ext=caption_ext,
            allow_image_samples=allow_image_samples,
            default_frames=default_frames,
            fps=fps,
        )
        skipped.extend(folder_skipped)
        existing = {s.path for s in samples}
        for sample in folder_samples:
            if sample.path not in existing:
                samples.append(sample)

    if not samples:
        raise FileNotFoundError(f"No captioned media found under {folder}")

    _keys, groups = assign_buckets(
        samples,
        max_pixels=max_pixels,
        max_frames=max_frames,
        min_bucket_size=min_bucket_size,
    )
    hist = bucket_histogram(groups)
    info("bucket histogram:")
    for key, count in hist:
        info(f"  {key.kind} {key.width}x{key.height} Tlat={key.latent_t}: {count}")
    if skipped:
        warn(f"skipped {len(skipped)} files (see cache/manifest.json)")
    return samples, groups, skipped


class H3Dataset:
    """Index-only dataset; actual tensor loading happens in the cache/train stages."""

    def __init__(self, samples: list[Sample]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Sample:
        return self.samples[index]

    def __iter__(self) -> Iterator[Sample]:
        return iter(self.samples)
