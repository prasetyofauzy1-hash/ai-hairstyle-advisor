"""Load the Phase 3 face-shape model trained by ml.training.train_face_shape and predict from FaceFeatures.

Trained on the Kaggle niten19 set (5 classes, no Diamond, unclear image provenance: research use only, TD-14).
Scores are the model's class probabilities and are NOT calibrated.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

from ml.classification.rule_based import SHAPE_LABELS_ID
from ml.feature_extraction.geometry import FaceFeatures

DEFAULT_MODEL = Path(__file__).resolve().parents[1] / "models" / "artifacts" / "face_shape_kaggle_niten19.joblib"


@dataclass
class LearnedPrediction:
    shape: str
    label_id: str
    scores: dict[str, float]
    method: str
    calibrated: bool = False

    def to_dict(self) -> dict:
        return {"shape": self.shape, "label_id": self.label_id, "scores": self.scores,
                "method": self.method, "calibrated": self.calibrated}


class LearnedFaceShapeClassifier:
    def __init__(self, path: str | Path = DEFAULT_MODEL):
        bundle = joblib.load(path)
        self.model = bundle["model"]
        self.encoder = bundle["label_encoder"]
        self.feature_names: list[str] = bundle["feature_names"]
        self.labels: list[str] = bundle["labels"]
        self.method = f"{bundle['model_name']}@{bundle['trained_on']}"

    def predict(self, features: FaceFeatures) -> LearnedPrediction:
        x = np.array([[getattr(features, n) for n in self.feature_names]], dtype=np.float64)
        proba = self.model.predict_proba(x)[0]
        classes = self.encoder.inverse_transform(np.arange(len(proba))) if self.encoder is not None else self.model.classes_
        scores = {str(c): float(p) for c, p in zip(classes, proba)}
        best = max(scores, key=scores.get)
        return LearnedPrediction(best, SHAPE_LABELS_ID[best], scores, self.method)
