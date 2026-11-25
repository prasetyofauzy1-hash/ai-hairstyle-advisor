"""Rule-based face-shape baseline.

Each class is described by a prototype (expected value and tolerance) for a few
geometric features, written from common styling-guide definitions of the
shapes. A class score is the negative mean squared z-distance to its prototype;
a softmax turns scores into relative scores that sum to 1.

The prototype values are provisional: they are NOT fitted to labeled data and
NOT validated. Phase 3 calibrates them on a held-out calibration split. The
output scores must be presented as uncalibrated.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ml.feature_extraction.geometry import FaceFeatures

SHAPES = ["oval", "round", "square", "oblong", "heart", "diamond"]

SHAPE_LABELS_ID = {
    "oval": "Oval",
    "round": "Bulat",
    "square": "Persegi",
    "oblong": "Lonjong",
    "heart": "Hati",
    "diamond": "Wajik (Diamond)",
}

# feature -> (expected value, tolerance). Ratio centres come from styling-guide shape definitions;
# jaw_angle_deg centres were shifted once to the range MediaPipe actually produces on frontal
# portraits (~144-154 deg on the unlabeled samples in ml/datasets/samples; no labels were used). Length is measured from MediaPipe landmark 10,
# which sits below the real hairline, so length_to_width is lower than "true" face length / width.
PROTOTYPES: dict[str, dict[str, tuple[float, float]]] = {
    "oval":    {"length_to_width": (1.22, 0.06), "forehead_to_jaw": (1.10, 0.08), "jaw_to_cheekbone": (0.78, 0.05), "jaw_angle_deg": (148, 7)},
    "round":   {"length_to_width": (1.08, 0.06), "forehead_to_jaw": (1.05, 0.08), "jaw_to_cheekbone": (0.82, 0.05), "jaw_angle_deg": (152, 7)},
    "square":  {"length_to_width": (1.10, 0.06), "forehead_to_jaw": (0.98, 0.07), "jaw_to_cheekbone": (0.88, 0.05), "jaw_angle_deg": (141, 7)},
    "oblong":  {"length_to_width": (1.36, 0.07), "forehead_to_jaw": (1.05, 0.08), "jaw_to_cheekbone": (0.82, 0.06), "jaw_angle_deg": (147, 8)},
    "heart":   {"length_to_width": (1.20, 0.07), "forehead_to_jaw": (1.25, 0.08), "jaw_to_cheekbone": (0.72, 0.05), "chin_to_jaw": (0.38, 0.06)},
    "diamond": {"length_to_width": (1.25, 0.07), "forehead_to_cheekbone": (0.80, 0.05), "forehead_to_jaw": (1.00, 0.08), "jaw_to_cheekbone": (0.75, 0.05)},
}

TEMPERATURE = 1.0

FEATURE_LABELS_ID = {
    "length_to_width": "rasio panjang terhadap lebar wajah",
    "forehead_to_jaw": "rasio lebar dahi terhadap rahang",
    "jaw_to_cheekbone": "rasio lebar rahang terhadap tulang pipi",
    "forehead_to_cheekbone": "rasio lebar dahi terhadap tulang pipi",
    "chin_to_jaw": "rasio lebar dagu terhadap rahang",
    "jaw_angle_deg": "sudut rahang",
}


@dataclass
class FaceShapePrediction:
    shape: str
    label_id: str
    scores: dict[str, float]          # softmax of rule scores, uncalibrated
    deviations: dict[str, dict[str, float]]   # per class, per feature z-distance
    explanation_id: str
    method: str = "rule_based_v0"
    calibrated: bool = False

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def _class_score(features: FaceFeatures, proto: dict[str, tuple[float, float]]) -> tuple[float, dict[str, float]]:
    z = {name: (getattr(features, name) - mu) / tol for name, (mu, tol) in proto.items()}
    return -float(np.mean([v * v for v in z.values()])), z


def _explain(shape: str, features: FaceFeatures, z: dict[str, float]) -> str:
    closest = sorted(z, key=lambda k: abs(z[k]))[:2]
    parts = [f"{FEATURE_LABELS_ID[k]} {getattr(features, k):.2f}" for k in closest]
    return (f"Bentuk wajah paling mendekati {SHAPE_LABELS_ID[shape]}, terutama karena "
            + " dan ".join(parts)
            + ". Ini estimasi berbasis aturan geometri yang belum dikalibrasi dengan data berlabel.")


def classify(features: FaceFeatures, forehead_occluded: bool = False) -> FaceShapePrediction:
    raw, deviations = {}, {}
    for shape in SHAPES:
        proto = PROTOTYPES[shape]
        if forehead_occluded:
            proto = {k: v for k, v in proto.items() if "forehead" not in k}
        raw[shape], deviations[shape] = _class_score(features, proto)
    logits = np.array([raw[s] for s in SHAPES]) / TEMPERATURE
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()
    scores = {s: float(p) for s, p in zip(SHAPES, probs)}
    best = max(scores, key=scores.get)
    return FaceShapePrediction(
        shape=best,
        label_id=SHAPE_LABELS_ID[best],
        scores=scores,
        deviations={s: {k: float(v) for k, v in d.items()} for s, d in deviations.items()},
        explanation_id=_explain(best, features, deviations[best]),
    )
