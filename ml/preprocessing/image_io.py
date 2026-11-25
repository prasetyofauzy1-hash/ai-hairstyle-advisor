"""Image decoding. Images are kept in memory; nothing here logs pixel data or paths."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

MAX_BYTES = 10 * 1024 * 1024
MAX_SIDE = 2048


class ImageDecodeError(ValueError):
    pass


def decode_image(data: bytes) -> np.ndarray:
    """Decode JPEG/PNG/WebP bytes to an RGB uint8 array (EXIF orientation applied, metadata dropped)."""
    if len(data) > MAX_BYTES:
        raise ImageDecodeError("image_too_large")
    buf = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ImageDecodeError("unsupported_or_corrupt_image")
    h, w = bgr.shape[:2]
    scale = MAX_SIDE / max(h, w)
    if scale < 1:
        bgr = cv2.resize(bgr, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def load_image(path: str | Path) -> np.ndarray:
    return decode_image(Path(path).read_bytes())
