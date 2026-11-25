"""MediaPipe Face Landmarker wrapper: detects faces and returns the main face's landmarks and head pose."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from ml.models.download import ensure_model

# Face-oval ring of the MediaPipe face mesh, clockwise from the top (landmark 10).
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
             152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
LEFT_IRIS_CENTER = 468   # subject's right eye, image left in a non-mirrored photo
RIGHT_IRIS_CENTER = 473


@dataclass
class DetectedFace:
    landmarks: np.ndarray          # (478, 3): x, y in pixels, z in MediaPipe's relative units scaled by width
    yaw: float                     # degrees
    pitch: float
    roll: float

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        xs, ys = self.landmarks[:, 0], self.landmarks[:, 1]
        return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())

    @property
    def area(self) -> float:
        x0, y0, x1, y1 = self.bbox
        return (x1 - x0) * (y1 - y0)


def _euler_from_matrix(matrix: np.ndarray) -> tuple[float, float, float]:
    r = np.asarray(matrix)[:3, :3]
    pitch = math.degrees(math.atan2(r[2, 1], r[2, 2]))
    yaw = math.degrees(math.asin(max(-1.0, min(1.0, -r[2, 0]))))
    roll = math.degrees(math.atan2(r[1, 0], r[0, 0]))
    return yaw, pitch, roll


class FaceLandmarkDetector:
    def __init__(self, model_path: str | Path | None = None, max_faces: int = 3):
        model_path = model_path or ensure_model("face_landmarker.task")
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=max_faces,
            output_facial_transformation_matrixes=True,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)

    def detect(self, image_rgb: np.ndarray) -> list[DetectedFace]:
        h, w = image_rgb.shape[:2]
        result = self._landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(image_rgb)))
        faces = []
        for i, lms in enumerate(result.face_landmarks):
            pts = np.array([[p.x * w, p.y * h, p.z * w] for p in lms], dtype=np.float64)
            yaw = pitch = roll = float("nan")
            if result.facial_transformation_matrixes:
                yaw, pitch, roll = _euler_from_matrix(result.facial_transformation_matrixes[i])
            faces.append(DetectedFace(pts, yaw, pitch, roll))
        return sorted(faces, key=lambda f: f.area, reverse=True)

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
