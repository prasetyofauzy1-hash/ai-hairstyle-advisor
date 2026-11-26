import numpy as np
import pytest

from ml.evaluation import facebench


def test_common_identity_screen_excludes_candidates_before_any_model_scores(monkeypatch, tmp_path):
    rows = [
        {"label": "square", "passed": True},
        {"label": "heart", "passed": True},
        {"label": "oval", "passed": True},
        {"label": "oblong", "passed": False},
        {"label": "diamond", "passed": True, "forehead_occluded": True, "feature": 1},
        {"label": "round", "passed": True, "forehead_occluded": False, "feature": 2},
    ]
    embeddings = np.array([[1, 0, 0], [0, 1, 0], [np.nan, 0, 0], [1, 0, 0], [0, 0, 1], [0, 0, 1]])
    threshold = facebench.SAME_PERSON_COSINE
    refs = {
        "kaggle": np.array([[threshold, -np.sqrt(1 - threshold**2), 0], [np.nan, 0, 0]]),
        "ffhq_train": np.array([[1, 0, 0], [0, 1, 0]]),
    }
    artifact = tmp_path / "trained.joblib"
    artifact.write_bytes(b"artifact fixture")
    monkeypatch.setattr(facebench, "MODELS", {"learned": artifact})
    monkeypatch.setattr(facebench.joblib, "load", lambda path: {"feature_names": ["feature"]})

    def rules(frame):
        assert frame["forehead_occluded"].tolist() == [True, False]
        return ["diamond", "round"]

    def predict(bundle, x):
        assert x.tolist() == [[1.0], [2.0]]
        return ["oblong", "round"]

    monkeypatch.setattr(facebench, "phase2_rules", rules)
    monkeypatch.setattr(facebench, "predict_bundle", predict)
    report = facebench.score(rows, embeddings, refs)
    assert report["quality_passed"] == 5
    assert report["missing_identity_embedding"] == 1
    assert report["identity_matches_excluded"] == 2
    assert report["test_photos"] == 2
    assert report["references"] == {
        "kaggle": {"photos": 2, "embedded": 1, "matches": 1},
        "ffhq_train": {"photos": 2, "embedded": 2, "matches": 2},
    }
    assert report["class_counts"]["diamond"] == 1
    assert report["macro_labels"] == ["diamond", "round"]
    assert report["models"]["Phase 2 rules"]["macro_f1"] == pytest.approx(1.0)
    learned = report["models"]["learned"]
    assert learned["macro_f1"] == pytest.approx(0.5)
    assert learned["confusion_labels"] == facebench.LABELS
    assert learned["confusion_matrix"][facebench.LABELS.index("diamond")][facebench.LABELS.index("oblong")] == 1


def test_identity_screen_rejects_empty_reference_pool():
    with pytest.raises(ValueError, match="No usable identity embeddings in missing"):
        facebench.score([{"passed": True}], np.array([[1, 0]]), {"missing": np.array([[np.nan, np.nan]])})
