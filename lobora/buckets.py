"""Aspect / duration bucketing so mixed sizes can use batch_size > 1."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterator, Sequence

import torch
from torch.utils.data import Sampler

from lobora.grid import SpatialSize, image_latent_time, latent_time, snap_frames_nearest, snap_spatial


@dataclass(frozen=True)
class BucketKey:
    height: int
    width: int
    latent_t: int
    kind: str  # "video" | "image"

    def as_tuple(self) -> tuple[int, int, int, str]:
        return (self.height, self.width, self.latent_t, self.kind)


def bucket_for_video(
    height: int,
    width: int,
    num_frames: int,
    *,
    max_pixels: int | None = None,
    max_frames: int = 124,
) -> BucketKey:
    size = snap_spatial(height, width, max_pixels=max_pixels)
    frames = snap_frames_nearest(num_frames, max_frames=max_frames)
    return BucketKey(
        height=size.height,
        width=size.width,
        latent_t=latent_time(frames),
        kind="video",
    )


def bucket_for_image(
    height: int,
    width: int,
    *,
    max_pixels: int | None = None,
) -> BucketKey:
    size = snap_spatial(height, width, max_pixels=max_pixels)
    return BucketKey(
        height=size.height,
        width=size.width,
        latent_t=image_latent_time(),
        kind="image",
    )


def assign_buckets(
    samples: Sequence[object],
    *,
    max_pixels: int | None,
    max_frames: int,
    min_bucket_size: int = 1,
) -> tuple[list[BucketKey], dict[BucketKey, list[int]]]:
    """Return per-sample keys and index lists. Tiny buckets fold into a neighbor."""
    keys: list[BucketKey] = []
    groups: dict[BucketKey, list[int]] = defaultdict(list)
    for i, sample in enumerate(samples):
        key = sample_bucket(sample, max_pixels=max_pixels, max_frames=max_frames)
        keys.append(key)
        groups[key].append(i)

    if min_bucket_size <= 1:
        return keys, dict(groups)

    folded = _fold_small_buckets(groups, min_bucket_size=min_bucket_size)
    remapped = [None] * len(keys)
    for key, idxs in folded.items():
        for i in idxs:
            remapped[i] = key
    return remapped, folded  # type: ignore[return-value]


def sample_bucket(
    sample: object,
    *,
    max_pixels: int | None,
    max_frames: int,
) -> BucketKey:
    kind = getattr(sample, "kind", "video")
    height = int(getattr(sample, "height"))
    width = int(getattr(sample, "width"))
    if kind == "image":
        return bucket_for_image(height, width, max_pixels=max_pixels)
    frames = int(getattr(sample, "num_frames", 22))
    return bucket_for_video(
        height, width, frames, max_pixels=max_pixels, max_frames=max_frames
    )


def _manhattan(a: BucketKey, b: BucketKey) -> int:
    return abs(a.height - b.height) + abs(a.width - b.width) + abs(a.latent_t - b.latent_t) * 16


def _fold_small_buckets(
    groups: dict[BucketKey, list[int]],
    *,
    min_bucket_size: int,
) -> dict[BucketKey, list[int]]:
    """Merge buckets smaller than min_bucket_size into the nearest same-kind neighbor."""
    groups = {k: list(v) for k, v in groups.items()}
    while True:
        small = [k for k, idxs in groups.items() if len(idxs) < min_bucket_size]
        if not small:
            break
        # If everything is small, stop — caller can still train with batch_size=1.
        large = [k for k in groups if k not in small]
        if not large:
            break
        donor = min(small, key=lambda k: len(groups[k]))
        same_kind = [k for k in large if k.kind == donor.kind]
        pool = same_kind or large
        target = min(pool, key=lambda k: _manhattan(donor, k))
        groups[target].extend(groups.pop(donor))
    return groups


def bucket_histogram(groups: dict[BucketKey, list[int]]) -> list[tuple[BucketKey, int]]:
    return sorted(
        ((k, len(v)) for k, v in groups.items()),
        key=lambda item: (-item[1], item[0].as_tuple()),
    )


class BucketBatchSampler(Sampler[list[int]]):
    """Yield homogeneous batches. Images never share a batch with videos."""

    def __init__(
        self,
        groups: dict[BucketKey, list[int]],
        *,
        batch_size: int,
        drop_last: bool = False,
        seed: int = 0,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self.groups = {k: list(v) for k, v in groups.items()}
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.seed = seed
        self._length = self._count_batches()

    def _count_batches(self) -> int:
        total = 0
        for idxs in self.groups.values():
            n = len(idxs) // self.batch_size
            if not self.drop_last and len(idxs) % self.batch_size:
                n += 1
            total += n
        return total

    def __len__(self) -> int:
        return self._length

    def __iter__(self) -> Iterator[list[int]]:
        g = torch.Generator()
        g.manual_seed(self.seed)
        keys = list(self.groups.keys())
        order = torch.randperm(len(keys), generator=g).tolist()
        batches: list[list[int]] = []
        for ki in order:
            key = keys[ki]
            idxs = list(self.groups[key])
            perm = torch.randperm(len(idxs), generator=g).tolist()
            shuffled = [idxs[i] for i in perm]
            for start in range(0, len(shuffled), self.batch_size):
                chunk = shuffled[start : start + self.batch_size]
                if len(chunk) < self.batch_size and self.drop_last:
                    continue
                batches.append(chunk)
        perm = torch.randperm(len(batches), generator=g).tolist()
        for i in perm:
            yield batches[i]
