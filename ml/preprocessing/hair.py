"""Forehead-occlusion estimate from the MediaPipe hair segmenter."""

from __future__ import annotations

from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from ml.models.download import ensure_model

# Forehead polygon: upper face-oval arc (temple to temple) closed along the top of the eyebrows.
FOREHEAD_POLYGON = [54, 103, 67, 109, 10, 338, 297, 332, 284, 300, 293, 334, 296, 336, 107, 66, 105, 63, 70]


class HairSegmenter:
    def __init__(self, model_path: str | Path | None = None):
        model_path = model_path or ensure_model("hair_segmenter.tflite")
        options = vision.ImageSegmenterOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE,
            output_category_mask=True,
            output_confidence_masks=False,
        )
        self._segmenter = vision.ImageSegmenter.create_from_options(options)

    def hair_mask(self, image_rgb: np.ndarray) -> np.ndarray:
        """Boolean mask, True where the model labels hair."""
        # The model expects 4 channels (RGBA) per the MediaPipe model card.
        rgba = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2RGBA)
        result = self._segmenter.segment(mp.Image(image_format=mp.ImageFormat.SRGBA, data=np.ascontiguousarray(rgba)))
        return result.category_mask.numpy_view().squeeze() == 1

    def close(self) -> None:
        self._segmenter.close()


def forehead_hair_fraction(hair_mask: np.ndarray, landmarks: np.ndarray) -> float:
    poly = np.round(landmarks[FOREHEAD_POLYGON, :2]).astype(np.int32)
    region = np.zeros(hair_mask.shape, dtype=np.uint8)
    cv2.fillPoly(region, [poly], 1)
    area = int(region.sum())
    if area == 0:
        return 0.0
    return float((hair_mask & region.astype(bool)).sum() / area)
