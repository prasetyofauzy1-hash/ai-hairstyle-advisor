"""Feature stability on the same face under small image-level pose changes.

Real multi-pose photos of one person are not available yet, so the changes are synthesised:
in-plane rotation (roll), rescaling, and a mild horizontal perspective warp approximating a few
degrees of yaw. Each variant goes through the full detector, so this measures landmark jitter as well
as geometry. Tolerances are engineering limits, not accuracy claims.
"""

import cv2
import numpy as np
import pytest

from ml.feature_extraction.geometry import FEATURE_NAMES, extract_features

TOLERANCE = {  # max absolute change allowed per feature
    "length_to_width": 0.04, "forehead_to_cheekbone": 0.03, "jaw_to_cheekbone": 0.03,
    "forehead_to_jaw": 0.04, "chin_to_jaw": 0.04, "jaw_angle_deg": 3.0,
}


def _rotate(img, deg):
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    return cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REPLICATE)


def _scale(img, s):
    return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)


def _yaw_warp(img, strength):
    h, w = img.shape[:2]
    d = strength * h
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[0, d], [w, -d], [w, h + d], [0, h - d]])
    return cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (w, h), borderMode=cv2.BORDER_REPLICATE)


VARIANTS = {
    "roll_+5": lambda i: _rotate(i, 5),
    "roll_-5": lambda i: _rotate(i, -5),
    "roll_+10": lambda i: _rotate(i, 10),
    "scale_0.7": lambda i: _scale(i, 0.7),
    "scale_1.3": lambda i: _scale(i, 1.3),
    "yaw_warp_+": lambda i: _yaw_warp(i, 0.02),
    "yaw_warp_-": lambda i: _yaw_warp(i, -0.02),
}


@pytest.fixture(scope="module")
def baseline(detector, frontal_image):
    return extract_features(detector.detect(frontal_image)[0].landmarks)


@pytest.mark.parametrize("name", VARIANTS)
def test_features_stable_under_small_pose_changes(detector, frontal_image, baseline, name):
    faces = detector.detect(VARIANTS[name](frontal_image))
    assert faces, f"face lost under {name}"
    changed = extract_features(faces[0].landmarks)
    diffs = {n: abs(getattr(changed, n) - getattr(baseline, n)) for n in FEATURE_NAMES}
    too_big = {n: round(d, 4) for n, d in diffs.items() if d > TOLERANCE[n]}
    assert not too_big, f"{name}: {too_big}"


def test_prediction_stable_under_roll(analyzer, frontal_image):
    base = analyzer.analyze(frontal_image).prediction
    rolled = analyzer.analyze(_rotate(frontal_image, 5)).prediction
    assert base is not None and rolled is not None
    assert rolled.shape == base.shape
