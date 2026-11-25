"""Landmark and measurement overlays (labels in Bahasa Indonesia)."""

from __future__ import annotations

import cv2
import numpy as np

from ml.feature_extraction import geometry as g
from ml.feature_extraction.landmarks import FACE_OVAL

MEASUREMENT_LINES = [
    ("Panjang wajah", (g.FACE_TOP, g.CHIN), (255, 80, 80)),
    ("Lebar dahi", g.FOREHEAD, (80, 200, 255)),
    ("Lebar tulang pipi", g.CHEEKBONE, (120, 255, 120)),
    ("Lebar rahang", g.JAW, (255, 200, 60)),
    ("Lebar dagu", g.CHIN_SIDES, (220, 120, 255)),
]


def draw_analysis(image_rgb: np.ndarray, landmarks: np.ndarray, title_lines: list[str] | None = None) -> np.ndarray:
    """Draw the mesh, face oval, measurement lines, a legend and optional title lines on a copy of the image."""
    out = image_rgb.copy()
    scale = max(1.0, out.shape[1] / 640)
    thick = max(1, round(1.5 * scale))
    pts = np.round(landmarks[:, :2]).astype(np.int32)
    for p in pts[:468]:
        cv2.circle(out, tuple(int(v) for v in p), max(1, round(scale)), (230, 230, 230), -1, cv2.LINE_AA)
    cv2.polylines(out, [pts[FACE_OVAL]], True, (255, 255, 255), thick, cv2.LINE_AA)
    for i, (label, (a, b), color) in enumerate(MEASUREMENT_LINES):
        cv2.line(out, tuple(int(v) for v in pts[a]), tuple(int(v) for v in pts[b]), color, thick + 1, cv2.LINE_AA)
        for idx in (a, b):
            cv2.circle(out, tuple(int(v) for v in pts[idx]), thick + 3, color, -1, cv2.LINE_AA)
        put_text(out, label, (10, out.shape[0] - 12 - round(24 * scale) * i), color, scale)
    for i, line in enumerate(title_lines or []):
        put_text(out, line, (10, round(28 * scale) * (i + 1)), (255, 255, 255), scale)
    return out


def render_face_report(image_rgb: np.ndarray, landmarks: np.ndarray, title_lines: list[str] | None = None,
                       width: int = 640, margin: float = 0.6) -> np.ndarray:
    """Crop around the face, resize to `width` and draw the analysis overlay at that resolution."""
    h, w = image_rgb.shape[:2]
    (x0, y0), (x1, y1) = landmarks[:, :2].min(axis=0), landmarks[:, :2].max(axis=0)
    mx, my = (x1 - x0) * margin, (y1 - y0) * margin
    cx0, cy0 = max(0, int(x0 - mx)), max(0, int(y0 - my))
    cx1, cy1 = min(w, int(x1 + mx)), min(h, int(y1 + my))
    crop = image_rgb[cy0:cy1, cx0:cx1]
    s = width / crop.shape[1]
    crop = cv2.resize(crop, (width, round(crop.shape[0] * s)), interpolation=cv2.INTER_AREA)
    pts = (landmarks[:, :2] - [cx0, cy0]) * s
    return draw_analysis(crop, pts, title_lines)


def put_text(img: np.ndarray, text: str, org: tuple[int, int], color, scale: float = 1.0) -> None:
    fs = 0.6 * scale
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), max(2, round(4 * scale)), cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, fs, color, max(1, round(1.5 * scale)), cv2.LINE_AA)
