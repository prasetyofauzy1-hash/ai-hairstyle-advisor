import cv2
import numpy as np

from ml.feature_extraction.landmarks import DetectedFace
from ml.preprocessing.quality import BLOCKING, MESSAGES, assess_quality


def _codes(report):
    return {i.code for i in report.issues}


def test_every_message_is_defined():
    assert BLOCKING <= set(MESSAGES)


def test_frontal_portrait_passes(detector, frontal_image):
    report = assess_quality(frontal_image, detector.detect(frontal_image))
    assert report.passed, report.to_dict()


def test_no_face():
    blank = np.full((512, 512, 3), 128, np.uint8)
    report = assess_quality(blank, [])
    assert not report.passed and "no_face" in _codes(report)


def test_blurry(detector, frontal_image):
    blurred = cv2.GaussianBlur(frontal_image, (0, 0), 6)
    faces = detector.detect(blurred)
    if faces:  # detector may already fail on very blurry input, which is also a rejection
        assert "blurry" in _codes(assess_quality(blurred, faces))


def test_dark_and_bright(detector, frontal_image):
    faces = detector.detect(frontal_image)
    dark = (frontal_image * 0.25).astype(np.uint8)
    bright = np.clip(frontal_image.astype(np.int16) + 140, 0, 255).astype(np.uint8)
    assert "too_dark" in _codes(assess_quality(dark, faces))
    assert "too_bright" in _codes(assess_quality(bright, faces))


def test_small_face(detector, frontal_image):
    small = cv2.resize(frontal_image, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_AREA)
    faces = detector.detect(small)
    assert faces
    assert "face_too_small" in _codes(assess_quality(small, faces))


def test_face_cut_off(detector, frontal_image):
    faces = detector.detect(frontal_image)
    x0, y0, x1, y1 = faces[0].bbox
    cropped = frontal_image[: int((y0 + y1) / 2) + int((y1 - y0) * 0.3)]  # chin out of frame
    cut = detector.detect(cropped)
    if cut:
        assert "face_cut_off" in _codes(assess_quality(cropped, cut))


def test_multiple_similar_faces(detector, frontal_image):
    doubled = np.hstack([frontal_image, frontal_image])
    faces = detector.detect(doubled)
    assert len(faces) >= 2
    assert "multiple_faces" in _codes(assess_quality(doubled, faces))


def test_head_pose_gates():
    lm = np.zeros((478, 3))
    lm[:, 0] = np.linspace(100, 400, 478)
    lm[:, 1] = np.linspace(100, 400, 478)
    lm[454, 0], lm[234, 0] = 400, 100
    img = np.full((512, 512, 3), 128, np.uint8)
    report = assess_quality(img, [DetectedFace(lm, yaw=30, pitch=-30, roll=40)])
    assert {"head_turned", "head_tilted_vertically", "head_tilted"} <= _codes(report)


def test_level_frontal_pitch_passes():
    lm = np.zeros((478, 3))
    lm[:, 0] = np.linspace(100, 400, 478)
    lm[:, 1] = np.linspace(100, 400, 478)
    lm[454, 0], lm[234, 0] = 400, 100
    img = np.full((512, 512, 3), 128, np.uint8)
    assert "head_tilted_vertically" not in _codes(assess_quality(img, [DetectedFace(lm, yaw=0, pitch=19, roll=0)]))


def test_dark_but_sharp_is_dark_not_blurry(detector, frontal_image):
    faces = detector.detect(frontal_image)
    codes = _codes(assess_quality((frontal_image * 0.2).astype(np.uint8), faces))
    assert "too_dark" in codes and "blurry" not in codes


def test_blown_highlights_are_too_bright(detector, frontal_image):
    faces = detector.detect(frontal_image)
    over = np.clip(frontal_image.astype(np.float32) * 1.6, 0, 255).astype(np.uint8)
    assert "too_bright" in _codes(assess_quality(over, faces))


def test_sunglasses_warn(detector, frontal_image):
    face = detector.detect(frontal_image)[0]
    img = frontal_image.copy()
    x0, y0, x1, y1 = face.bbox
    for idx in ([33, 160, 158, 133, 153, 144], [362, 385, 387, 263, 373, 380]):
        c = face.landmarks[idx, :2].mean(0)
        cv2.ellipse(img, (int(c[0]), int(c[1])), (int(x1 - x0) // 6, int(y1 - y0) // 12), 0, 0, 360, (15, 15, 15), -1)
    report = assess_quality(img, [face])
    assert "eyes_covered" in _codes(report) and report.passed
    assert "eyes_covered" not in _codes(assess_quality(frontal_image, [face]))


def test_group_photo_reports_only_multiple_faces(detector, frontal_image):
    faces = detector.detect(frontal_image)
    report = assess_quality(frontal_image, faces, detector_face_widths=[160.0, 155.0, 151.0])
    assert _codes(report) == {"multiple_faces"}


def test_messages_are_indonesian():
    assert MESSAGES["no_face"].startswith("Wajah tidak terdeteksi")
