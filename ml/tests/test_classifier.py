import numpy as np
import pytest

from ml.classification.rule_based import PROTOTYPES, SHAPE_LABELS_ID, SHAPES, classify
from ml.feature_extraction.geometry import FaceFeatures, FaceMeasurements

DEFAULTS = {"length_to_width": 1.2, "forehead_to_cheekbone": 0.85, "jaw_to_cheekbone": 0.8,
            "forehead_to_jaw": 1.05, "chin_to_jaw": 0.4, "jaw_angle_deg": 148.0}


def _features(**values):
    v = {**DEFAULTS, **values}
    return FaceFeatures(measurements=FaceMeasurements(1, 1, 1, 1, 1), **v)


@pytest.mark.parametrize("shape", SHAPES)
def test_prototype_centre_is_classified_as_its_shape(shape):
    centre = {k: mu for k, (mu, _) in PROTOTYPES[shape].items()}
    pred = classify(_features(**centre))
    assert pred.shape == shape
    assert pred.label_id == SHAPE_LABELS_ID[shape]


def test_scores_are_a_distribution_and_marked_uncalibrated():
    pred = classify(_features())
    assert set(pred.scores) == set(SHAPES)
    assert np.isclose(sum(pred.scores.values()), 1.0)
    assert pred.calibrated is False and pred.method == "rule_based_v0"
    assert "belum dikalibrasi" in pred.explanation_id


def test_forehead_occlusion_drops_forehead_features():
    pred = classify(_features(forehead_to_jaw=2.0), forehead_occluded=True)
    for dev in pred.deviations.values():
        assert not any("forehead" in k for k in dev)


def test_longer_face_moves_towards_oblong():
    short = classify(_features(length_to_width=1.05)).scores["oblong"]
    long = classify(_features(length_to_width=1.40)).scores["oblong"]
    assert long > short
