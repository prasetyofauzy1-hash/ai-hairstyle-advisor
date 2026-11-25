"""Image quality gates. Messages are user-facing and therefore in Bahasa Indonesia."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import cv2
import numpy as np

from ml.feature_extraction.landmarks import DetectedFace


@dataclass(frozen=True)
class QualityThresholds:
    min_image_side: int = 256
    min_face_width_px: float = 150.0      # cheekbone-to-cheekbone width
    min_sharpness: float = 15.0           # variance of Laplacian on the face crop (resized to 256 px wide)
    min_brightness: float = 50.0
    max_brightness: float = 215.0
    max_abs_yaw: float = 15.0
    max_abs_pitch: float = 15.0
    max_abs_roll: float = 25.0
    edge_margin_px: float = 2.0
    secondary_face_ratio: float = 0.5     # a 2nd face this large relative to the main face is ambiguous


MESSAGES = {
    "image_too_small": "Resolusi foto terlalu kecil. Gunakan foto minimal 256 piksel di sisi terpendek.",
    "no_face": "Wajah tidak terdeteksi. Pastikan wajah terlihat jelas dan menghadap kamera.",
    "multiple_faces": "Terdeteksi lebih dari satu wajah dengan ukuran serupa. Gunakan foto dengan satu wajah saja.",
    "face_too_small": "Wajah terlalu kecil di dalam foto. Dekatkan kamera atau potong foto di sekitar wajah.",
    "face_cut_off": "Sebagian wajah terpotong di tepi foto. Pastikan seluruh wajah, termasuk dagu, terlihat.",
    "blurry": "Foto terlalu buram. Gunakan foto yang lebih tajam.",
    "too_dark": "Foto terlalu gelap. Gunakan pencahayaan yang lebih terang dan merata.",
    "too_bright": "Foto terlalu terang. Hindari cahaya langsung yang berlebihan.",
    "head_turned": "Kepala terlalu menoleh ke samping. Hadapkan wajah lurus ke kamera.",
    "head_tilted_vertically": "Kepala terlalu menunduk atau mendongak. Posisikan wajah sejajar dengan kamera.",
    "head_tilted": "Kepala terlalu miring. Tegakkan posisi kepala.",
    "forehead_covered": "Dahi sebagian tertutup rambut, sehingga pengukuran dahi kurang akurat.",
}

BLOCKING = {"image_too_small", "no_face", "multiple_faces", "face_too_small", "face_cut_off", "blurry",
            "too_dark", "too_bright", "head_turned", "head_tilted_vertically", "head_tilted"}


@dataclass
class QualityIssue:
    code: str
    message: str
    blocking: bool


@dataclass
class QualityReport:
    issues: list[QualityIssue] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return not any(i.blocking for i in self.issues)

    def add(self, code: str) -> None:
        self.issues.append(QualityIssue(code, MESSAGES[code], code in BLOCKING))

    def to_dict(self) -> dict:
        return {"passed": self.passed, "issues": [asdict(i) for i in self.issues], "metrics": self.metrics}


def face_crop(image_rgb: np.ndarray, face: DetectedFace) -> np.ndarray:
    h, w = image_rgb.shape[:2]
    x0, y0, x1, y1 = face.bbox
    x0, y0 = max(0, int(x0)), max(0, int(y0))
    x1, y1 = min(w, int(x1)), min(h, int(y1))
    return image_rgb[y0:y1, x0:x1]


def sharpness(gray: np.ndarray) -> float:
    if gray.shape[1] != 256:
        gray = cv2.resize(gray, (256, max(1, round(gray.shape[0] * 256 / gray.shape[1]))), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def assess_quality(image_rgb: np.ndarray, faces: list[DetectedFace],
                   t: QualityThresholds = QualityThresholds()) -> QualityReport:
    report = QualityReport()
    h, w = image_rgb.shape[:2]
    if min(h, w) < t.min_image_side:
        report.add("image_too_small")
    if not faces:
        report.add("no_face")
        return report
    main = faces[0]
    report.metrics["face_count"] = float(len(faces))
    if len(faces) > 1 and faces[1].area >= t.secondary_face_ratio * main.area:
        report.add("multiple_faces")

    pts = main.landmarks
    face_width = float(np.linalg.norm(pts[234, :2] - pts[454, :2]))
    report.metrics["face_width_px"] = face_width
    if face_width < t.min_face_width_px:
        report.add("face_too_small")
    x0, y0, x1, y1 = main.bbox
    if x0 < t.edge_margin_px or y0 < t.edge_margin_px or x1 > w - t.edge_margin_px or y1 > h - t.edge_margin_px:
        report.add("face_cut_off")

    crop = face_crop(image_rgb, main)
    if crop.size:
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        report.metrics["sharpness"] = sharpness(gray)
        report.metrics["brightness"] = float(gray.mean())
        if report.metrics["sharpness"] < t.min_sharpness:
            report.add("blurry")
        if report.metrics["brightness"] < t.min_brightness:
            report.add("too_dark")
        elif report.metrics["brightness"] > t.max_brightness:
            report.add("too_bright")

    report.metrics.update(yaw_deg=main.yaw, pitch_deg=main.pitch, roll_deg=main.roll)
    if abs(main.yaw) > t.max_abs_yaw:
        report.add("head_turned")
    if abs(main.pitch) > t.max_abs_pitch:
        report.add("head_tilted_vertically")
    if abs(main.roll) > t.max_abs_roll:
        report.add("head_tilted")
    return report
