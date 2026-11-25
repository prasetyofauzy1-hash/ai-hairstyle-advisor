"""End-to-end face analysis: quality gates -> landmarks -> features -> rule-based face shape.

Only metadata (check codes, numeric metrics) is ever logged; never pixels, paths or landmarks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from ml.classification.rule_based import FaceShapePrediction, classify
from ml.feature_extraction.geometry import FaceFeatures, extract_features
from ml.feature_extraction.landmarks import DetectedFace, FaceLandmarkDetector
from ml.preprocessing.hair import HairSegmenter, forehead_hair_fraction
from ml.preprocessing.quality import QualityReport, assess_quality

log = logging.getLogger(__name__)

FOREHEAD_OCCLUSION_THRESHOLD = 0.3


@dataclass
class AnalysisResult:
    quality: QualityReport
    face: DetectedFace | None = None
    features: FaceFeatures | None = None
    forehead_hair_fraction: float | None = None
    prediction: FaceShapePrediction | None = None

    def to_dict(self) -> dict:
        return {
            "quality": self.quality.to_dict(),
            "features": self.features.to_dict() if self.features else None,
            "forehead_hair_fraction": self.forehead_hair_fraction,
            "forehead_visible": None if self.forehead_hair_fraction is None
            else self.forehead_hair_fraction < FOREHEAD_OCCLUSION_THRESHOLD,
            "prediction": self.prediction.to_dict() if self.prediction else None,
        }


class FaceAnalyzer:
    def __init__(self, detector: FaceLandmarkDetector | None = None, hair: HairSegmenter | None = None,
                 use_hair_segmenter: bool = True):
        self.detector = detector or FaceLandmarkDetector()
        self.hair = hair if hair is not None else (HairSegmenter() if use_hair_segmenter else None)

    def analyze(self, image_rgb: np.ndarray) -> AnalysisResult:
        faces = self.detector.detect(image_rgb)
        quality = assess_quality(image_rgb, faces)
        result = AnalysisResult(quality=quality, face=faces[0] if faces else None)
        if not quality.passed:
            log.info("analysis rejected by quality gates: %s", [i.code for i in quality.issues])
            return result
        result.features = extract_features(result.face.landmarks)
        occluded = False
        if self.hair is not None:
            result.forehead_hair_fraction = forehead_hair_fraction(self.hair.hair_mask(image_rgb), result.face.landmarks)
            occluded = result.forehead_hair_fraction >= FOREHEAD_OCCLUSION_THRESHOLD
            if occluded:
                quality.add("forehead_covered")
        result.prediction = classify(result.features, forehead_occluded=occluded)
        log.info("analysis done: shape=%s", result.prediction.shape)
        return result

    def close(self) -> None:
        self.detector.close()
        if self.hair is not None:
            self.hair.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
