from pathlib import Path

from PIL import Image

from lobora.dataset import load_dataset, scan_folder


def _write_pair(folder: Path, name: str, size: tuple[int, int], caption: str) -> None:
    Image.new("RGB", size, color=(128, 40, 40)).save(folder / f"{name}.png")
    (folder / f"{name}.txt").write_text(caption, encoding="utf-8")


def test_scan_folder_images(tmp_path: Path):
    _write_pair(tmp_path, "a", (480, 832), "a red fox in a forest")
    _write_pair(tmp_path, "b", (640, 360), "wide landscape")
    (tmp_path / "orphan.png").write_bytes(b"")  # skipped: no caption
    samples, skipped = scan_folder(tmp_path, allow_image_samples=True)
    assert len(samples) == 2
    assert all(s.kind == "image" for s in samples)
    assert any(s["reason"] == "missing_or_empty_caption" for s in skipped)


def test_load_dataset_buckets(tmp_path: Path):
    for i in range(4):
        _write_pair(tmp_path, f"c{i}", (480, 832), f"clip {i} of the subject")
    samples, groups, _skipped = load_dataset(
        tmp_path,
        allow_image_samples=True,
        max_pixels=480 * 832,
        max_frames=124,
        min_bucket_size=1,
    )
    assert len(samples) == 4
    assert groups
    assert all(k.kind == "image" for k in groups)
