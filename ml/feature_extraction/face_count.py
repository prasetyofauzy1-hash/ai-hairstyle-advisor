"""Count faces with OpenCV Zoo YuNet; the MediaPipe landmarker often misses all but one face in group photos."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ml.models.download import ensure_model


class FaceCounter:
    def __init__(self, model_path: str | Path | None = None, score_threshold: float = 0.8):
        self._path = str(model_path or ensure_model("face_detection_yunet_2023mar.onnx"))
        self._threshold = score_threshold

    def face_widths(self, image_rgb: np.ndarray) -> list[float]:
        h, w = image_rgb.shape[:2]
        det = cv2.FaceDetectorYN.create(self._path, "", (w, h), self._threshold, 0.3, 5000)
        _, faces = det.detect(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
        return [] if faces is None else sorted((float(f[2]) for f in faces), reverse=True)
