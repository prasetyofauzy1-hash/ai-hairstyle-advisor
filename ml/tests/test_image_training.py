import numpy as np
import pandas as pd
import pytest

from ml.datasets.face_crops import SIZE, square_crop
from ml.training.foundation_features import encode_foundation, fuse
from ml.training.train_face_shape_image import baseline_predictions, encode, select, subset_metrics


def test_crop_preserves_aspect_ratio():
    image = np.zeros((120, 120, 3), dtype=np.uint8)
    image[40:80, 50:70] = 255
    crop = square_crop(image, (50, 40, 70, 80))
    assert crop.shape == (SIZE, SIZE, 3)
    y, x = np.where(crop[:, :, 0] > 128)
    assert abs(np.ptp(y) / np.ptp(x) - 2) < 0.05


def test_image_selection_never_places_a_group_on_both_sides(monkeypatch):
    train = pd.DataFrame({"label": np.repeat(["oval", "round"], 12),
                          "identity_group": np.repeat(np.arange(12), 2)})
    features = train[["identity_group"]].to_numpy()

    def fit(config, training_image, training_frame):
        class Classifier:
            def predict(self, held_image):
                assert not (set(training_image.ravel()) & set(held_image.ravel()))
                return np.where(held_image.ravel() < 6, "oval", "round")

        return Classifier(), None

    monkeypatch.setattr("ml.training.train_face_shape_image.fit", fit)
    selected, report = select(train, {"test_backbone": features})
    assert set(selected) == {"direct_image", "image_plus_landmarks"}
    assert len(report["trials"]) == 8
    assert report["selected"]["direct_image"]["cv_macro_f1"] == 1.0


def test_subset_metrics_retains_guesses_outside_the_cohorts_classes():
    frame = pd.DataFrame({"label": ["oval", "round", "oval"]})
    results = subset_metrics(frame, np.array(["female", "male", "unknown"]),
                             {"model": np.array(["heart", "round", "oval"])})
    female = results["female"]["models"]["model"]
    assert female["accuracy"] == 0
    assert female["per_class"]["oval"]["support"] == 1
    assert female["confusion_labels"] == ["oval", "heart"]
    assert female["confusion_matrix"] == [[0, 1]]
    assert results["all"]["photos"] == 3


def test_face_specific_representation_does_not_load_imagenet_backbones(monkeypatch, tmp_path):
    def face_assets(photos, cache, workers, *, sface):
        assert sface
        assert photos == [tmp_path / "photo.jpg"]
        return np.ones((1, 128))

    monkeypatch.setattr("ml.training.train_face_shape_image.face_assets", face_assets)
    result = encode({"sface"}, [tmp_path / "photo.jpg"], tmp_path, 1)
    assert result["sface"].shape == (1, 128)


def test_app_baseline_is_not_scored_on_overlapping_niten19_holdout(monkeypatch):
    frame = pd.DataFrame({"feature": [1.0]})
    baselines = {name: {"feature_names": ["feature"], "prediction": ["oval"]}
                 for name in ("app_landmarks", "combined_landmarks")}
    calls = []

    def predict(bundle, features):
        calls.append(bundle)
        return np.array(bundle["prediction"])

    monkeypatch.setattr("ml.training.train_face_shape_image.phase2_rules", lambda frame: np.array(["oval"]))
    monkeypatch.setattr("ml.training.train_face_shape_image.predict_bundle", predict)
    assert set(baseline_predictions("niten19_holdout", frame, baselines)) == {"phase2_rules", "combined_landmarks"}
    assert len(calls) == 1 and calls[0] is baselines["combined_landmarks"]
    for benchmark in ("uzy", "facebench", "roboflow_holdout"):
        assert set(baseline_predictions(benchmark, frame, baselines)) == {"phase2_rules", *baselines}


def test_fusion_gives_equal_weight_to_unit_vectors_regardless_of_dimension():
    a = np.array([[1.0]])
    b = np.array([[0.6, 0.8]])
    combined = fuse([a, b])
    np.testing.assert_allclose(np.linalg.norm(combined, axis=1), 1.0)
    np.testing.assert_allclose(np.linalg.norm(combined[:, :1], axis=1), np.linalg.norm(combined[:, 1:], axis=1))


def test_foundation_cache_does_not_publish_a_partial_write(monkeypatch, tmp_path):
    features = np.array([[1.0, 0.0]])
    monkeypatch.setattr("ml.training.foundation_features.face_assets",
                        lambda *args: np.zeros((1, 2, 2, 3), dtype=np.uint8))
    monkeypatch.setattr("ml.training.foundation_features.frozen_features", lambda *args: features)
    photos = [tmp_path / "face.jpg"]

    def failed_save(file, *args, **kwargs):
        file.write(b"partial array")
        raise OSError("no space left")

    with monkeypatch.context() as patch:
        patch.setattr("ml.training.foundation_features.np.save", failed_save)
        with pytest.raises(OSError, match="no space left"):
            encode_foundation(["dino"], photos, tmp_path, 1)
    assert not list(tmp_path.iterdir())

    result = encode_foundation(["dino"], photos, tmp_path, 1)
    cached, = tmp_path.glob("*.npy")
    np.testing.assert_array_equal(np.load(cached), result["dino"])


def test_foundation_selection_keeps_families_and_locked_reference_separate(monkeypatch):
    train = pd.DataFrame({"label": np.repeat(["oval", "round"], 12),
                          "identity_group": np.repeat(np.arange(12), 2)})
    features = {key: train[["identity_group"]].to_numpy()
                for key in ("dino", "clip", "dino_sface", "clip_sface", "dino_clip_sface", "sface", "mobile_tail")}

    def fit(config, training_image, training_frame):
        class Classifier:
            def predict(self, held_image):
                assert not (set(training_image.ravel()) & set(held_image.ravel()))
                return np.where(held_image.ravel() < 6, "oval", "round")

        return Classifier(), None

    monkeypatch.setattr("ml.training.train_face_shape_image.fit", fit)
    selected, report = select(train, features, foundation=True)
    assert len(report["trials"]) == 22
    assert not selected["foundation_image"]["geometry"]
    assert selected["foundation_hybrid"]["geometry"]
    assert selected["foundation_ensemble"]["backbone"].endswith("_sface")
    assert selected["mobile_partial"]["backbone"] == "mobile_tail"
    assert selected["pr14_hybrid"] == {"backbone": "sface", "geometry": True, "C": 1.0, "power": 0.5}
