"""Two-stage disk cache. Keys include model_rev (LensTrainer omitted this)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from lobora.buckets import BucketKey, sample_bucket
from lobora.console import info, ok, warn
from lobora.dataset import Sample


def cache_key(
    *,
    kind: str,
    path: str,
    caption: str,
    height: int,
    width: int,
    frames_or_latent_t: int,
    model_rev: str,
) -> str:
    payload = f"{kind}|{path}|{caption}|{height}x{width}x{frames_or_latent_t}|{model_rev}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sample_cache_key(sample: Sample, *, bucket: BucketKey, model_rev: str, kind: str) -> str:
    return cache_key(
        kind=kind,
        path=sample.path,
        caption=sample.caption,
        height=bucket.height,
        width=bucket.width,
        frames_or_latent_t=bucket.latent_t,
        model_rev=model_rev,
    )


@dataclass
class CacheManifest:
    samples: int
    skipped: list[dict[str, str]]
    keys: dict[str, str]


class DiskCache:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.latents = self.root / "latents"
        self.text = self.root / "text"
        self.latents.mkdir(parents=True, exist_ok=True)
        self.text.mkdir(parents=True, exist_ok=True)

    def path_for(self, kind: str, digest: str) -> Path:
        folder = self.text if kind == "text" else self.latents
        return folder / f"{digest}.pt"

    def has(self, kind: str, digest: str) -> bool:
        return self.path_for(kind, digest).is_file()

    def write_manifest(self, manifest: CacheManifest) -> None:
        path = self.root / "manifest.json"
        path.write_text(json.dumps(asdict(manifest), indent=2) + "\n", encoding="utf-8")

    def plan(
        self,
        samples: list[Sample],
        groups: dict[BucketKey, list[int]],
        *,
        model_rev: str,
    ) -> list[dict[str, Any]]:
        index_to_bucket = {}
        for key, idxs in groups.items():
            for i in idxs:
                index_to_bucket[i] = key
        planned = []
        for i, sample in enumerate(samples):
            bucket = index_to_bucket.get(i) or sample_bucket(
                sample, max_pixels=None, max_frames=124
            )
            lat_key = sample_cache_key(sample, bucket=bucket, model_rev=model_rev, kind="latents")
            txt_key = sample_cache_key(sample, bucket=bucket, model_rev=model_rev, kind="text")
            planned.append(
                {
                    "index": i,
                    "sample_id": sample.sample_id,
                    "bucket": bucket.as_tuple(),
                    "latents_key": lat_key,
                    "text_key": txt_key,
                    "latents_hit": self.has("latents", lat_key),
                    "text_hit": self.has("text", txt_key),
                }
            )
        return planned


def precompute_or_report(
    cache: DiskCache,
    samples: list[Sample],
    groups: dict,
    *,
    model_rev: str,
    skipped: list[dict[str, str]],
    encode_fn=None,
) -> list[dict[str, Any]]:
    """Stage-1 cache. ``encode_fn`` is optional so CPU tests can skip DiffSynth."""
    planned = cache.plan(samples, groups, model_rev=model_rev)
    missing_lat = sum(1 for row in planned if not row["latents_hit"])
    missing_txt = sum(1 for row in planned if not row["text_hit"])
    info(f"cache: {len(planned) - missing_lat}/{len(planned)} latents, {len(planned) - missing_txt}/{len(planned)} text")
    if encode_fn is None:
        if missing_lat or missing_txt:
            warn("encode_fn not provided — cache misses left on disk (DiffSynth not loaded)")
        cache.write_manifest(
            CacheManifest(
                samples=len(samples),
                skipped=skipped,
                keys={row["sample_id"]: row["latents_key"] for row in planned},
            )
        )
        return planned

    for row in planned:
        sample = samples[row["index"]]
        if row["latents_hit"] and row["text_hit"]:
            continue
        payload = encode_fn(sample, row)
        if payload.get("latents") is not None and not row["latents_hit"]:
            _torch_save(cache.path_for("latents", row["latents_key"]), payload["latents"])
        if payload.get("text") is not None and not row["text_hit"]:
            _torch_save(cache.path_for("text", row["text_key"]), payload["text"])
    cache.write_manifest(
        CacheManifest(
            samples=len(samples),
            skipped=skipped,
            keys={row["sample_id"]: row["latents_key"] for row in planned},
        )
    )
    ok("stage-1 cache complete")
    return planned


def _torch_save(path: Path, obj: Any) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj, path)
