from __future__ import annotations
import re
from pathlib import Path
import numpy as np

# Common image extensions to probe when auto-detecting pattern.
_IMAGE_EXTS = ["jpg", "JPG", "jpeg", "JPEG", "png", "PNG", "tif", "TIF", "tiff", "TIFF", "bmp", "BMP"]


def parse_frame_number(filename: str) -> int:
    stem = Path(filename).stem
    m = re.search(r"\d+", stem)
    if m is None:
        raise ValueError(f"Cannot parse frame number from: {filename}")
    return int(m.group())


def detect_pattern(frames_dir: str | Path) -> str:
    """Return a glob pattern matching the most common image type in *frames_dir*.

    Tries each extension in ``_IMAGE_EXTS`` and returns ``*.EXT`` for the
    extension with the most matches.  Falls back to ``"Img*.jpg"`` when the
    directory is empty or contains no recognised images.
    """
    frames_dir = Path(frames_dir)
    best_ext, best_count = "jpg", 0
    for ext in _IMAGE_EXTS:
        n = sum(1 for _ in frames_dir.glob(f"*.{ext}"))
        if n > best_count:
            best_count, best_ext = n, ext
    return f"*.{best_ext}" if best_count > 0 else "Img*.jpg"


def scan_frames(frames_dir: str | Path, pattern: str = "Img*.jpg") -> list[Path]:
    frames_dir = Path(frames_dir)
    paths = list(frames_dir.glob(pattern))
    try:
        paths = sorted(paths, key=lambda p: parse_frame_number(p.name))
    except ValueError:
        paths = sorted(paths)
    return paths


def load_frame(path: str | Path) -> np.ndarray:
    from PIL import Image
    img = Image.open(path).convert("L")
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr
