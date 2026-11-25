"""Consent-gated image storage: re-encoded JPEG (drops EXIF/GPS) under a random UUID name outside the web root."""

from __future__ import annotations

import io
import uuid
import warnings
from pathlib import Path

import cv2
import numpy as np

from PIL import Image

from app.config import settings

MAX_PIXELS = 40_000_000

ALLOWED_MAGIC = {b"\xff\xd8\xff": "jpeg", b"\x89PNG\r\n\x1a\n": "png", b"RIFF": "webp"}


def sniff_format(data: bytes) -> str | None:
    for magic, fmt in ALLOWED_MAGIC.items():
        if data.startswith(magic) and (fmt != "webp" or data[8:12] == b"WEBP"):
            return fmt
    return None


def pixel_count(data: bytes) -> int | None:
    """Width x height from the image header, read before any full decode allocates the pixels."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as im:
                w, h = im.size
    except Image.DecompressionBombError:
        return MAX_PIXELS + 1
    except Exception:
        return None
    return w * h


def store_image(image_rgb: np.ndarray) -> str:
    root = Path(settings.image_storage_dir)
    root.mkdir(parents=True, exist_ok=True)
    name = f"{uuid.uuid4()}.jpg"
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise RuntimeError("image re-encode failed")
    (root / name).write_bytes(buf.tobytes())
    return name


def image_file(name: str) -> Path:
    return Path(settings.image_storage_dir) / Path(name).name


def delete_image(name: str | None) -> None:
    if name:
        image_file(name).unlink(missing_ok=True)
