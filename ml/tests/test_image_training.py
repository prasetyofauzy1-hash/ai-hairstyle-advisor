import numpy as np
import pandas as pd

from ml.datasets.face_crops import SIZE, square_crop
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
