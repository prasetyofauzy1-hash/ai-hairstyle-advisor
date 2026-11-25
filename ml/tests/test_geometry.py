import math

import numpy as np
import pytest

from ml.feature_extraction.geometry import extract_features


@pytest.fixture(scope="module")
def landmarks(detector, frontal_image):
    return detector.detect(frontal_image)[0].landmarks


def _transform(pts, angle_deg=0.0, scale=1.0, shift=(0.0, 0.0)):
    a = math.radians(angle_deg)
    rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    out = pts.copy()
    out[:, :2] = (pts[:, :2] @ rot.T) * scale + shift
    return out


@pytest.mark.parametrize("angle,scale,shift", [(0, 2.5, (0, 0)), (20, 1.0, (0, 0)), (-35, 0.4, (100, -50))])
def test_features_invariant_to_similarity_transforms(landmarks, angle, scale, shift):
    base = extract_features(landmarks).vector()
    moved = extract_features(_transform(landmarks, angle, scale, shift)).vector()
    np.testing.assert_allclose(moved, base, rtol=1e-9, atol=1e-9)


def test_ratios_are_plausible(landmarks):
    f = extract_features(landmarks)
    assert 0.9 < f.length_to_width < 1.6
    assert 0.6 < f.jaw_to_cheekbone < 1.0
    assert 0.6 < f.forehead_to_cheekbone < 1.0
    assert 110 < f.jaw_angle_deg < 170


def test_requires_iris_landmarks():
    with pytest.raises(ValueError):
        extract_features(np.zeros((468, 3)))
