"""Scale- and roll-invariant geometric face features from MediaPipe landmarks.

Landmark choices are documented in research/methodology/face-shape-features.md.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from ml.feature_extraction.landmarks import LEFT_IRIS_CENTER, RIGHT_IRIS_CENTER

# Measurement landmark pairs (left index, right index) on the MediaPipe face mesh.
FACE_TOP, CHIN = 10, 152
FOREHEAD = (54, 284)
CHEEKBONE = (234, 454)
JAW = (172, 397)
CHIN_SIDES = (176, 400)
JAW_ANGLE_LEFT = (132, 172, 148)    # (upper jaw, gonion, toward chin)
JAW_ANGLE_RIGHT = (361, 397, 377)

FEATURE_NAMES = [
    "length_to_width", "forehead_to_cheekbone", "jaw_to_cheekbone",
    "forehead_to_jaw", "chin_to_jaw", "jaw_angle_deg",
]


@dataclass
class FaceMeasurements:
    """Distances in pixels of the aligned image (only meaningful relative to each other)."""
    face_length: float
    forehead_width: float
    cheekbone_width: float
    jaw_width: float
    chin_width: float


@dataclass
class FaceFeatures:
    measurements: FaceMeasurements
    length_to_width: float
    forehead_to_cheekbone: float
    jaw_to_cheekbone: float
    forehead_to_jaw: float
    chin_to_jaw: float
    jaw_angle_deg: float

    def vector(self) -> np.ndarray:
        return np.array([getattr(self, n) for n in FEATURE_NAMES], dtype=np.float64)

    def to_dict(self) -> dict:
        return asdict(self)


def align_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """Rotate 2D landmarks about the eye midpoint so the eye line is horizontal (removes head roll)."""
    pts = landmarks[:, :2]
    left, right = pts[LEFT_IRIS_CENTER], pts[RIGHT_IRIS_CENTER]
    angle = math.atan2(right[1] - left[1], right[0] - left[0])
    c, s = math.cos(-angle), math.sin(-angle)
    rot = np.array([[c, -s], [s, c]])
    center = (left + right) / 2
    return (pts - center) @ rot.T + center


def _dist(pts: np.ndarray, a: int, b: int) -> float:
    return float(np.linalg.norm(pts[a] - pts[b]))


def _angle(pts: np.ndarray, a: int, vertex: int, b: int) -> float:
    v1, v2 = pts[a] - pts[vertex], pts[b] - pts[vertex]
    cos = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def extract_features(landmarks: np.ndarray) -> FaceFeatures:
    if landmarks.shape[0] < 478:
        raise ValueError("expected 478 MediaPipe landmarks (with iris)")
    pts = align_landmarks(landmarks)
    m = FaceMeasurements(
        face_length=_dist(pts, FACE_TOP, CHIN),
        forehead_width=_dist(pts, *FOREHEAD),
        cheekbone_width=_dist(pts, *CHEEKBONE),
        jaw_width=_dist(pts, *JAW),
        chin_width=_dist(pts, *CHIN_SIDES),
    )
    jaw_angle = (_angle(pts, *JAW_ANGLE_LEFT) + _angle(pts, *JAW_ANGLE_RIGHT)) / 2
    return FaceFeatures(
        measurements=m,
        length_to_width=m.face_length / m.cheekbone_width,
        forehead_to_cheekbone=m.forehead_width / m.cheekbone_width,
        jaw_to_cheekbone=m.jaw_width / m.cheekbone_width,
        forehead_to_jaw=m.forehead_width / m.jaw_width,
        chin_to_jaw=m.chin_width / m.jaw_width,
        jaw_angle_deg=jaw_angle,
    )
