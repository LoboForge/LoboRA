from lobora.grid import (
    MIN_FRAMES,
    is_valid_frame_count,
    latent_time,
    snap_frames_down,
    snap_frames_nearest,
    snap_spatial,
)


def test_valid_frame_counts():
    for n in (22, 39, 56, 73, 90, 107, 124):
        assert is_valid_frame_count(n)
        assert latent_time(n) >= 2


def test_snap_frames_down():
    assert snap_frames_down(10) == MIN_FRAMES
    assert snap_frames_down(40) == 39
    assert snap_frames_down(73) == 73


def test_snap_frames_nearest():
    assert snap_frames_nearest(70) == 73
    assert snap_frames_nearest(30) == 22


def test_snap_spatial_div32():
    size = snap_spatial(833, 481)
    assert size.height % 32 == 0
    assert size.width % 32 == 0
    assert size.height <= 832
    assert size.width <= 480


def test_snap_spatial_max_pixels():
    size = snap_spatial(1080, 1920, max_pixels=480 * 832)
    assert size.height % 32 == 0
    assert size.width % 32 == 0
    assert size.pixels <= 480 * 832
