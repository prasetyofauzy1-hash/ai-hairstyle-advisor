import cv2
import numpy as np
import pytest

from ml.preprocessing.image_io import MAX_SIDE, ImageDecodeError, decode_image


def _jpeg(img_bgr, exif: bytes | None = None) -> bytes:
    data = cv2.imencode(".jpg", img_bgr)[1].tobytes()
    if exif is None:
        return data
    return data[:2] + exif + data[2:]


def _exif_orientation_segment(orientation: int) -> bytes:
    # Minimal APP1 Exif segment (big-endian TIFF) with a single Orientation tag.
    tiff = b"MM\x00\x2a\x00\x00\x00\x08" + b"\x00\x01" + b"\x01\x12\x00\x03\x00\x00\x00\x01" \
        + orientation.to_bytes(2, "big") + b"\x00\x00" + b"\x00\x00\x00\x00"
    payload = b"Exif\x00\x00" + tiff
    return b"\xff\xe1" + (len(payload) + 2).to_bytes(2, "big") + payload


def test_decodes_to_rgb():
    bgr = np.zeros((40, 60, 3), np.uint8)
    bgr[:] = (255, 0, 0)  # blue in BGR
    rgb = decode_image(cv2.imencode(".png", bgr)[1].tobytes())
    assert rgb.shape == (40, 60, 3)
    assert tuple(rgb[0, 0]) == (0, 0, 255)


def test_applies_exif_orientation():
    bgr = np.zeros((40, 80, 3), np.uint8)
    rotated = decode_image(_jpeg(bgr, _exif_orientation_segment(6)))  # 6 = rotate 90 CW
    assert rotated.shape[:2] == (80, 40)


def test_downscales_large_images():
    big = np.zeros((MAX_SIDE * 2, MAX_SIDE, 3), np.uint8)
    out = decode_image(cv2.imencode(".png", big)[1].tobytes())
    assert max(out.shape[:2]) == MAX_SIDE


def test_rejects_garbage():
    with pytest.raises(ImageDecodeError):
        decode_image(b"not an image")
