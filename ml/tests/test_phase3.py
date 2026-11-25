import csv

import numpy as np
import pandas as pd
import pytest

from ml.classification.learned import DEFAULT_MODEL, LearnedFaceShapeClassifier
from ml.datasets.agreement import agreement, cohen_kappa, fleiss_kappa
from ml.datasets.ffhq_select import ALLOWED_LICENSES, candidates
from ml.datasets.identity import identity_groups
from ml.feature_extraction.geometry import FEATURE_NAMES, extract_features
from ml.training.train_face_shape import holdout_split, run


def test_identity_groups_split_people_and_keep_nan_singletons():
    rng = np.random.default_rng(0)
    people = rng.normal(size=(3, 128))
    emb = np.concatenate([p + 0.05 * rng.normal(size=(4, 128)) for p in people])
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    emb = np.vstack([emb, np.full((2, 128), np.nan)])
    g = identity_groups(emb)
    assert len(set(g[:12])) == 3
    assert all(len(set(g[i * 4:(i + 1) * 4])) == 1 for i in range(3))
    assert len(set(g[12:]) | set(g[:12])) == 5


def test_kappa_values():
    assert cohen_kappa(list("aabb"), list("aabb")) == 1.0
    assert cohen_kappa(list("abab"), list("abba")) == pytest.approx(0.0)
    assert fleiss_kappa(np.array([[2, 0], [0, 2], [2, 0], [0, 2]])) == pytest.approx(1.0)


def test_consensus_drops_disagreement_and_unsure():
    labels = {"1": {"a": "oval", "b": "oval"}, "2": {"a": "round", "b": "oval"},
              "3": {"a": "ragu", "b": "ragu"}, "4": {"a": "oval"}}
    report, consensus = agreement(labels)
    assert report["images_labeled_by_all"] == 3
    assert [c["ffhq_id"] for c in consensus] == ["1"]


def test_ffhq_candidates_respect_licenses_and_seed():
    meta = {str(i): {"metadata": {"license": lic}} for i, lic in
            enumerate(["Attribution License", "Attribution-NonCommercial License", "Public Domain Mark"] * 5)}
    ids = candidates(meta)
    assert all(meta[str(i)]["metadata"]["license"] in ALLOWED_LICENSES for i in ids)
    assert len(ids) == 10 and ids == candidates(meta)


def _synthetic_features(tmp_path, n_people=60, per_person=5):
    rng = np.random.default_rng(1)
    rows = []
    shapes = ["heart", "oblong", "oval", "round", "square"]
    for p in range(n_people):
        label = shapes[p % 5]
        centre = np.array([1.1 + 0.05 * shapes.index(label), 0.85, 0.8, 1.05, 0.4, 145.0 + shapes.index(label)])
        for _ in range(per_person):
            x = centre + rng.normal(scale=0.01, size=6)
            rows.append({"image": f"{p}_{_}.jpg", "label": label, "source_split": "training_set", "passed": True,
                         "issues": "", **dict(zip(FEATURE_NAMES, x)), "identity_group": p})
    path = tmp_path / "features.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_holdout_is_identity_disjoint(tmp_path):
    df = pd.read_csv(_synthetic_features(tmp_path))
    tr, te = holdout_split(df["label"].to_numpy(), df["identity_group"].to_numpy())
    assert not set(df["identity_group"].iloc[tr]) & set(df["identity_group"].iloc[te])


def test_training_run_writes_report_and_model(tmp_path):
    report = run(_synthetic_features(tmp_path), tmp_path / "out", tmp_path / "model")
    assert (tmp_path / "out" / "metrics.json").exists() and (tmp_path / "out" / "confusion_matrices.png").exists()
    assert report["selected_model"] not in ("majority_class", "rule_based_v0")
    clf = LearnedFaceShapeClassifier(tmp_path / "model" / "face_shape_kaggle_niten19.joblib")
    assert set(clf.labels) == {"heart", "oblong", "oval", "round", "square"}


@pytest.mark.skipif(not DEFAULT_MODEL.exists(), reason="trained model not present")
def test_committed_model_predicts_on_sample(frontal_image, detector):
    features = extract_features(detector.detect(frontal_image)[0].landmarks)
    pred = LearnedFaceShapeClassifier().predict(features)
    assert pred.shape in pred.scores and pred.calibrated is False
    assert sum(pred.scores.values()) == pytest.approx(1.0)


def test_confusion_matrix_keeps_predictions_outside_labels():
    from ml.training.train_face_shape import metrics
    m = metrics(np.array(["oval", "round", "oval"]), np.array(["oval", "diamond", "round"]), ["oval", "round"])
    assert m["confusion_labels"] == ["oval", "round", "diamond"]
    assert np.sum(m["confusion_matrix"]) == 3


def _synthetic_ffhq(tmp_path, n_train=40, n_test=15):
    rng = np.random.default_rng(2)
    shapes = ["diamond", "oval", "round", "oblong"]
    rows = []
    for i in range(n_train + n_test):
        split = "train" if i < n_train else "test"
        label = shapes[i % 4] if split == "train" else shapes[1 + i % 3]
        centre = np.array([1.1 + 0.05 * shapes.index(label), 0.85, 0.8, 1.05, 0.4, 145.0 + shapes.index(label)])
        rows.append({"ffhq_id": f"{i:05d}", "image": f"{i:05d}.png", "label": label, "source_split": split, "passed": True,
                     "issues": "", **dict(zip(FEATURE_NAMES, centre + rng.normal(scale=0.01, size=6))),
                     "label_source": "ai_pseudo_label" if split == "train" else "human",
                     "identity_group": 0 if i in (0, n_train) else i})
    path = tmp_path / "ffhq.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_ffhq_run_never_trains_on_human_test_and_drops_identity_overlap(tmp_path):
    from ml.training.train_face_shape_ffhq import ARTIFACT, COMBINED, KAGGLE_ONLY
    from ml.training.train_face_shape_ffhq import run as run_ffhq

    kaggle = _synthetic_features(tmp_path)
    run(kaggle, tmp_path / "base", tmp_path / "model")
    report = run_ffhq(kaggle, _synthetic_ffhq(tmp_path), tmp_path / "out", tmp_path / "model",
                      tmp_path / "model" / "face_shape_kaggle_niten19.joblib")
    d = report["data"]
    assert d["ffhq_train_dropped_for_identity_overlap_with_test"] == 1 and d["ffhq_train_used"] == 39
    assert d["ffhq_human_test_used"] == 15
    assert report["configs"][COMBINED]["train_size"] == report["configs"][KAGGLE_ONLY]["train_size"] + 39
    assert "diamond" in report["configs"][COMBINED]["labels"]
    assert set(report["current_model"]["ffhq_human_test"]["per_class"]) == {"oblong", "oval", "round"}
    assert (tmp_path / "out" / "README.md").exists()
    assert (tmp_path / "out" / "confusion_matrices_ffhq_human_test.png").exists()
    assert "diamond" in LearnedFaceShapeClassifier(tmp_path / "model" / ARTIFACT).labels
