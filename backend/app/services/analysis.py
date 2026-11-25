"""Wraps ml.FaceAnalyzer + the Phase 3 learned classifier. No CV/ML logic lives here."""

from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np

from app.config import settings
from ml.classification.learned import DEFAULT_MODEL, LearnedFaceShapeClassifier
from ml.pipeline import AnalysisResult, FaceAnalyzer

_lock = threading.Lock()
_analyzer: FaceAnalyzer | None = None
_learned: LearnedFaceShapeClassifier | None = None


def _models() -> tuple[FaceAnalyzer, LearnedFaceShapeClassifier | None]:
    global _analyzer, _learned
    if _analyzer is None:
        _analyzer = FaceAnalyzer(use_hair_segmenter=settings.use_hair_segmenter)
        _learned = LearnedFaceShapeClassifier() if DEFAULT_MODEL.exists() else None
    return _analyzer, _learned


@dataclass
class AnalysisOutput:
    result: AnalysisResult
    learned: dict | None


def analyze(image_rgb: np.ndarray) -> AnalysisOutput:
    with _lock:  # MediaPipe task objects are not thread-safe
        analyzer, learned = _models()
        result = analyzer.analyze(image_rgb)
        pred = learned.predict(result.features).to_dict() if (learned and result.features is not None) else None
    return AnalysisOutput(result, pred)
