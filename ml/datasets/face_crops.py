"""Cached RGB face crops and aligned SFace descriptors for the image experiment."""

from __future__ import annotations

import hashlib
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from ml.datasets.identity import FaceEmbedder
from ml.feature_extraction.landmarks import FaceLandmarkDetector
from ml.models.download import ensure_model
from ml.preprocessing.image_io import decode_image

SIZE = 144
_detector = None
_embedder = None


def square_crop(rgb: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray:
    x0, y0, x1, y1 = box
    width = 1.25 * max(x1 - x0, y1 - y0)
    scale = SIZE / width
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    matrix = np.array([[scale, 0, SIZE / 2 - scale * cx], [0, scale, SIZE / 2 - scale * cy]])
    return cv2.warpAffine(rgb, matrix, (SIZE, SIZE), borderMode=cv2.BORDER_REPLICATE)


def _init(model: Path, sface: bool) -> None:
    global _detector, _embedder
    cv2.setNumThreads(1)
    _detector = FaceLandmarkDetector(model_path=model)
    _embedder = FaceEmbedder() if sface else None


def _one(args: tuple[Path, Path]) -> np.ndarray:
    photo, cache = args
    data = photo.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    representation = "sface-v1" if _embedder is not None else f"mesh-square-v1-{SIZE}"
    output = cache / f"{representation}-{sha}.npy"
    if output.is_file():
        return np.load(output, allow_pickle=False)
    rgb = decode_image(data)
    faces = _detector.detect(rgb)
    if not faces:
        raise ValueError(f"No face crop for a previously quality-passing photo: {photo}")
    result = square_crop(rgb, faces[0].bbox) if _embedder is None else _embedder.embed(rgb, faces[0].landmarks)
    pending = output.with_suffix(f".{os.getpid()}.npy")
    np.save(pending, result, allow_pickle=False)
    pending.replace(output)
    return result


def face_assets(photos: list[Path], cache: Path, workers: int, *, sface: bool = False) -> np.ndarray:
    cache.mkdir(parents=True, exist_ok=True)
    model = ensure_model("face_landmarker.task")
    if sface:
        ensure_model("face_recognition_sface_2021dec.onnx")
    with ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(model, sface),
                             mp_context=multiprocessing.get_context("spawn")) as pool:
        return np.stack(list(pool.map(_one, [(photo, cache) for photo in photos], chunksize=8)))
