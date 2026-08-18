"""MiniMax-H3 spatial / temporal grid.

H3 video VAE requires:
- height and width divisible by 32
- num_frames ≡ 5 (mod 17), minimum 22 (22, 39, 56, 73, 90, 107, 124, …)
- latent T = ((num_frames - 5) // 17) * 5 + 2
"""

from __future__ import annotations

from dataclasses import dataclass

TIME_DIVISION = 17
TIME_REMAINDER = 5
MIN_FRAMES = 22
SPATIAL_DIVISION = 32
DEFAULT_FPS = 24
AUDIO_SAMPLE_RATE = 32000

# Common train lengths used by DiffSynth examples.
STANDARD_FRAME_COUNTS = (22, 39, 56, 73, 90, 107, 124)


@dataclass(frozen=True)
class SpatialSize:
    height: int
    width: int

    @property
    def pixels(self) -> int:
        return self.height * self.width


def is_valid_frame_count(num_frames: int) -> bool:
    return num_frames >= MIN_FRAMES and num_frames % TIME_DIVISION == TIME_REMAINDER


def latent_time(num_frames: int) -> int:
    """Video VAE temporal latent length for a pixel-frame count."""
    if num_frames < MIN_FRAMES:
        raise ValueError(f"num_frames must be >= {MIN_FRAMES}, got {num_frames}")
    if num_frames % TIME_DIVISION != TIME_REMAINDER:
        raise ValueError(
            f"num_frames must satisfy n % {TIME_DIVISION} == {TIME_REMAINDER}, got {num_frames}"
        )
    return ((num_frames - TIME_REMAINDER) // TIME_DIVISION) * 5 + 2


def snap_frames_down(num_frames: int) -> int:
    """Largest valid H3 frame count that is <= num_frames (or MIN_FRAMES)."""
    if num_frames < MIN_FRAMES:
        return MIN_FRAMES
    n = num_frames - ((num_frames - TIME_REMAINDER) % TIME_DIVISION)
    if n < MIN_FRAMES:
        return MIN_FRAMES
    return n


def snap_frames_nearest(num_frames: int, *, max_frames: int = 124) -> int:
    """Nearest valid frame count in [MIN_FRAMES, max_frames]."""
    candidates = [
        f for f in range(MIN_FRAMES, max_frames + 1) if f % TIME_DIVISION == TIME_REMAINDER
    ]
    if not candidates:
        return MIN_FRAMES
    return min(candidates, key=lambda f: (abs(f - num_frames), f))


def snap_spatial(height: int, width: int, *, max_pixels: int | None = None) -> SpatialSize:
    """Snap H/W down to multiples of 32, optionally shrinking to max_pixels."""
    h = max(SPATIAL_DIVISION, (height // SPATIAL_DIVISION) * SPATIAL_DIVISION)
    w = max(SPATIAL_DIVISION, (width // SPATIAL_DIVISION) * SPATIAL_DIVISION)
    if max_pixels and h * w > max_pixels:
        scale = (max_pixels / (h * w)) ** 0.5
        h = max(SPATIAL_DIVISION, int(h * scale) // SPATIAL_DIVISION * SPATIAL_DIVISION)
        w = max(SPATIAL_DIVISION, int(w * scale) // SPATIAL_DIVISION * SPATIAL_DIVISION)
        # If we overshot due to rounding, shrink the longer edge.
        while h * w > max_pixels and (h > SPATIAL_DIVISION or w > SPATIAL_DIVISION):
            if h >= w and h > SPATIAL_DIVISION:
                h -= SPATIAL_DIVISION
            elif w > SPATIAL_DIVISION:
                w -= SPATIAL_DIVISION
            else:
                break
    return SpatialSize(height=h, width=w)


def image_latent_time() -> int:
    """Single-frame stills encode as one latent frame (VAE process_image=True)."""
    return 1
