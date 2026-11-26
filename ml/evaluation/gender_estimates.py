"""Estimate aggregate research cohorts with the InsightFace buffalo_s attribute model."""

from __future__ import annotations

import hashlib
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

MODEL_SHA256 = "4fde69b1c810857b88c64a335084f1c3fe8f01246c9a191b48c7bb756d6652fb"
_detector = None
_network = None


def _init(path: Path) -> None:
    import dlib

    global _detector, _network
    cv2.setNumThreads(1)
    _detector = dlib.get_frontal_face_detector()
    _network = cv2.dnn.readNetFromONNX(str(path))


def _estimate(path: Path) -> str:
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Cannot read photo for gender estimation: {path}")
    scale = min(1, 480 / max(image.shape[:2]))
    if scale < 1:
        image = cv2.resize(image, None, fx=scale, fy=scale)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = _detector(rgb, 0)
    if not len(faces):
        faces = _detector(rgb, 1)
    if len(faces) != 1:
        return "unknown"
    box = faces[0]
    cx, cy = (box.left() + box.right()) / 2, (box.top() + box.bottom()) / 2
    s = 96 / (1.5 * max(box.width(), box.height()))
    crop = cv2.warpAffine(image, np.array([[s, 0, 48 - s * cx], [0, s, 48 - s * cy]]), (96, 96))
    _network.setInput(cv2.dnn.blobFromImage(crop, size=(96, 96), swapRB=True))
    return ("female", "male")[int(np.argmax(_network.forward().ravel()[:2]))]


def estimates(paths: list[Path], model: Path, workers: int) -> list[str]:
    if hashlib.sha256(model.read_bytes()).hexdigest() != MODEL_SHA256:
        raise ValueError("Attribute model is not the pinned InsightFace buffalo_s genderage.onnx")
    try:
        import dlib  # noqa: F401
    except ImportError:
        raise RuntimeError("Install dlib (e.g. dlib-bin) to estimate image cohorts") from None
    with ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(model,),
                             mp_context=multiprocessing.get_context("spawn")) as pool:
        return list(pool.map(_estimate, paths, chunksize=12))
