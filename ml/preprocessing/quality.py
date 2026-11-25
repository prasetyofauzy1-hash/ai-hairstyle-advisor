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
    max_clipped_fraction: float = 0.15    # share of face pixels at >= 250 (blown-out highlights)
    min_eye_brightness_ratio: float = 0.45  # eye region vs. whole face; opaque sunglasses fall far below
    max_abs_yaw: float = 15.0
    max_abs_pitch: float = 25.0           # MediaPipe reports ~18-19 deg on level frontal NASA portraits
    max_abs_roll: float = 25.0
    edge_margin_px: float = 2.0
    secondary_face_ratio: float = 0.5     # a 2nd face this large relative to the main face is ambiguous
    secondary_width_ratio: float = 0.7    # same rule for detector box widths (~sqrt of the area ratio)


MESSAGES = {
    "image_too_small": "Resolusi foto terlalu kecil. Sisi terpendek foto harus minimal 256 piksel.",
    "no_face": "Wajah tidak terdeteksi. Pastikan wajah terlihat jelas dan menghadap kamera.",
    "multiple_faces": "Terdeteksi lebih dari satu wajah dengan ukuran serupa. Gunakan foto dengan satu wajah saja.",
    "face_too_small": "Wajah terlalu kecil di dalam foto. Coba ambil foto lebih dekat, tetapi pastikan seluruh wajah terlihat.",
    "face_cut_off": "Sebagian wajah terpotong di tepi foto. Pastikan seluruh wajah, termasuk dagu, terlihat.",
    "blurry": "Foto terlalu buram. Gunakan foto yang lebih tajam.",
    "too_dark": "Foto terlalu gelap. Gunakan pencahayaan yang lebih terang dan merata.",
    "too_bright": "Foto terlalu terang. Hindari cahaya langsung yang berlebihan.",
    "head_turned": "Kepala terlalu menoleh ke samping. Hadapkan wajah lurus ke kamera.",
    "head_tilted_vertically": "Kepala terlalu menunduk atau mendongak. Sejajarkan kamera dengan mata dan lihat lurus ke depan.",
    "head_tilted": "Kepala terlalu miring. Tegakkan posisi kepala.",
    "eyes_covered": "Mata tampak tertutup, misalnya oleh kacamata hitam. Lepas kacamata hitam agar hasil lebih tepat.",
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


EYES = ([33, 160, 158, 133, 153, 144], [362, 385, 387, 263, 373, 380])


def sharpness(gray: np.ndarray) -> float:
    """Laplacian variance after scaling to mean 128, so a dark but sharp photo is not called blurry."""
    if gray.shape[1] != 256:
        gray = cv2.resize(gray, (256, max(1, round(gray.shape[0] * 256 / gray.shape[1]))), interpolation=cv2.INTER_AREA)
    g = gray.astype(np.float64) * (128.0 / max(float(gray.mean()), 1.0))
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def eye_brightness_ratio(image_rgb: np.ndarray, face: DetectedFace, face_mean: float) -> float:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    mask = np.zeros(gray.shape, np.uint8)
    for idx in EYES:
        pts = face.landmarks[idx, :2]
        c = pts.mean(0)
        cv2.fillPoly(mask, [((pts - c) * 1.8 + c).astype(np.int32)], 1)
    return float(gray[mask > 0].mean() / max(face_mean, 1.0)) if mask.any() else 1.0


def assess_quality(image_rgb: np.ndarray, faces: list[DetectedFace],
                   t: QualityThresholds = QualityThresholds(),
                   detector_face_widths: list[float] | None = None) -> QualityReport:
    report = QualityReport()
    h, w = image_rgb.shape[:2]
    if min(h, w) < t.min_image_side:
        report.add("image_too_small")
    if not faces:
        report.add("no_face")
        return report
    main = faces[0]
    report.metrics["face_count"] = float(len(faces))
    widths = detector_face_widths or []
    if (len(faces) > 1 and faces[1].area >= t.secondary_face_ratio * main.area) or (
            len(widths) > 1 and widths[1] >= t.secondary_width_ratio * widths[0]):
        report.add("multiple_faces")
        return report  # size/pose checks on one face of a group photo would only confuse

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
        report.metrics["clipped_fraction"] = float((crop.max(axis=2) >= 250).mean())
        if report.metrics["brightness"] < t.min_brightness:
            report.add("too_dark")
        elif report.metrics["brightness"] > t.max_brightness or report.metrics["clipped_fraction"] > t.max_clipped_fraction:
            report.add("too_bright")
        if report.metrics["sharpness"] < t.min_sharpness:
            report.add("blurry")
        report.metrics["eye_brightness_ratio"] = eye_brightness_ratio(image_rgb, main, report.metrics["brightness"])
        if report.metrics["eye_brightness_ratio"] < t.min_eye_brightness_ratio:
            report.add("eyes_covered")

    report.metrics.update(yaw_deg=main.yaw, pitch_deg=main.pitch, roll_deg=main.roll)
    if abs(main.yaw) > t.max_abs_yaw:
        report.add("head_turned")
    if abs(main.pitch) > t.max_abs_pitch:
        report.add("head_tilted_vertically")
    if abs(main.roll) > t.max_abs_roll:
        report.add("head_tilted")
    return report
