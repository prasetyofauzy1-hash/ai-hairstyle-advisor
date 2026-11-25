"""Group photos of the same person with SFace embeddings so splits never put one person in train and test.

The datasets have no identity IDs; embeddings are used only for this grouping and are never logged or committed.
"""

from __future__ import annotations

import cv2
import numpy as np

from ml.models.download import ensure_model

# OpenCV's published cosine threshold for "same identity" with SFace.
SAME_PERSON_COSINE = 0.363
# MediaPipe indices for SFace's 5-point template order (image-left eye, image-right eye, nose tip, mouth corners).
FIVE_POINTS = (468, 473, 1, 61, 291)


class FaceEmbedder:
    def __init__(self):
        self._model = cv2.FaceRecognizerSF.create(str(ensure_model("face_recognition_sface_2021dec.onnx")), "")

    def embed(self, image_rgb: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        pts = landmarks[list(FIVE_POINTS), :2]
        x0, y0 = landmarks[:, :2].min(axis=0)
        x1, y1 = landmarks[:, :2].max(axis=0)
        row = np.array([[x0, y0, x1 - x0, y1 - y0, *pts.ravel(), 1.0]], dtype=np.float32)
        crop = self._model.alignCrop(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR), row)
        vec = self._model.feature(crop).ravel().astype(np.float32)
        return vec / (np.linalg.norm(vec) + 1e-12)


def identity_groups(embeddings: np.ndarray, threshold: float = SAME_PERSON_COSINE) -> np.ndarray:
    """Average-linkage clusters cut at cosine distance 1 - threshold; rows with NaN get singleton groups.

    Average linkage avoids the chaining of single linkage (connected components), which merges a dataset with many
    look-alike celebrities into one giant group.
    """
    from scipy.cluster.hierarchy import fcluster, linkage

    n = len(embeddings)
    valid = np.flatnonzero(~np.isnan(embeddings).any(axis=1))
    groups = np.full(n, -1, dtype=np.int64)
    if len(valid) > 1:
        tree = linkage(embeddings[valid].astype(np.float64), method="average", metric="cosine")
        groups[valid] = fcluster(tree, t=1.0 - threshold, criterion="distance")
    elif len(valid) == 1:
        groups[valid] = 1
    missing = groups < 0
    groups[missing] = groups.max() + 1 + np.arange(missing.sum())
    _, groups = np.unique(groups, return_inverse=True)
    return groups
