from dataclasses import dataclass

from lobora.buckets import (
    BucketBatchSampler,
    BucketKey,
    assign_buckets,
    bucket_for_image,
    bucket_for_video,
)


@dataclass
class Fake:
    kind: str
    height: int
    width: int
    num_frames: int = 73


def test_video_and_image_never_share_bucket():
    v = bucket_for_video(832, 480, 73)
    i = bucket_for_image(832, 480)
    assert v.kind == "video"
    assert i.kind == "image"
    assert v.latent_t != i.latent_t


def test_fold_small_buckets():
    samples = [
        Fake("video", 832, 480, 73),
        Fake("video", 832, 480, 73),
        Fake("video", 640, 352, 39),  # singleton → fold
    ]
    keys, groups = assign_buckets(samples, max_pixels=None, max_frames=124, min_bucket_size=2)
    assert all(len(v) >= 2 for v in groups.values())
    assert sum(len(v) for v in groups.values()) == 3


def test_batch_sampler_homogeneous():
    samples = [Fake("video", 832, 480, 73) for _ in range(6)] + [
        Fake("image", 832, 480) for _ in range(4)
    ]
    _keys, groups = assign_buckets(samples, max_pixels=None, max_frames=124, min_bucket_size=1)
    sampler = BucketBatchSampler(groups, batch_size=2, drop_last=True, seed=0)
    index_kind = {i: s.kind for i, s in enumerate(samples)}
    batches = list(sampler)
    assert batches
    for batch in batches:
        kinds = {index_kind[i] for i in batch}
        assert len(kinds) == 1
        assert len(batch) == 2
