import numpy as np
import pandas as pd
import pytest

from ml.training import train_face_shape_diamond as retrain


def test_cv_keeps_groups_together_with_each_shape_in_every_fold():
    y = np.array(["diamond"] * 3 + ["oval"] * 10)
    groups = np.array([0, 1, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7])
    _, folds = retrain.cv_folds(y, groups)
    assert len(folds) == 3
    for tr, va in folds:
        assert not set(groups[tr]) & set(groups[va])
        assert set(y[tr]) == set(y[va]) == {"diamond", "oval"}

    groups[:3] = 0
    with pytest.raises(ValueError, match="at least two independent training groups"):
        retrain.cv_folds(y, groups)


def test_weights_combine_shape_frequency_and_source_weight():
    y = np.array(["diamond", "oval", "oval", "oval", "oval"])
    ffhq = np.array([True, False, False, False, True])
    w = retrain.weights(y, ffhq, 0.5, 3.0)
    assert w.mean() == pytest.approx(1)
    assert w[0] / w[-1] == pytest.approx(2)
    assert w[-1] / w[1] == pytest.approx(3)


def test_cohort_averages_observed_shapes_and_keeps_unknown(monkeypatch):
    frame = pd.DataFrame({"label": ["oval", "oval", "diamond", "round"], "x": [1, 1, 0, 0]})
    gender = pd.Series(["male", "male", "female", "unknown"])
    monkeypatch.setattr(retrain, "predict_bundle", lambda model, X: np.where(X[:, 0], "oval", "diamond"))
    result = retrain.score_cohort(frame, gender, {"model": {"feature_names": ["x"]}})
    assert result["all"]["photos"] == 4
    assert result["male"]["models"]["model"]["macro_f1"] == 1.0
    assert result["female"]["models"]["model"]["macro_f1"] == 1.0
    assert result["unknown"]["models"]["model"]["macro_f1"] == 0.0
