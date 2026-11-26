"""Pinned, CPU-only vision feature extraction for the face-shape research experiment."""

from __future__ import annotations

import hashlib
import math
import os
import tempfile
from pathlib import Path

import numpy as np
from sklearn.preprocessing import normalize

from ml.datasets.face_crops import face_assets
from ml.training.mobile_tail import activations

MODELS = {
    "dino": ("facebook/dinov2-small", "ed25f3a31f01632728cabb09d1542f84ab7b0056",
             140, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    "clip": ("openai/clip-vit-base-patch32", "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268",
             224, (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
}
REPRESENTATIONS = {
    "dino": ("dino",),
    "clip": ("clip",),
    "dino_sface": ("dino", "sface"),
    "clip_sface": ("clip", "sface"),
    "dino_clip_sface": ("dino", "clip", "sface"),
    "sface": ("sface",),
    "mobile_tail": ("mobile_tail",),
}


def frozen_features(name: str, pixels: np.ndarray) -> np.ndarray:
    import torch
    from torch.nn.functional import interpolate
    from transformers import CLIPVisionModelWithProjection, Dinov2Model

    repo, revision, size, mean, std = MODELS[name]
    model_class = Dinov2Model if name == "dino" else CLIPVisionModelWithProjection
    model = model_class.from_pretrained(repo, revision=revision).eval()
    mean = torch.tensor(mean)[None, :, None, None]
    std = torch.tensor(std)[None, :, None, None]
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(pixels), 16):
            batch = torch.from_numpy(pixels[start:start + 16]).permute(0, 3, 1, 2).float() / 255
            batch = interpolate(batch, (size, size), mode="bicubic", align_corners=False, antialias=True)
            outputs = model((batch.clamp(0, 1) - mean) / std)
            if name == "dino":
                tokens = outputs.last_hidden_state
                vectors = torch.cat((tokens[:, 0], tokens[:, 1:].mean(dim=1)), dim=1)
            else:
                vectors = outputs.image_embeds
            chunks.append(vectors.numpy())
    return normalize(np.concatenate(chunks))


def fuse(blocks: list[np.ndarray]) -> np.ndarray:
    return np.column_stack(blocks) / math.sqrt(len(blocks)) if len(blocks) > 1 else blocks[0]


def _save_cache(output: Path, data: np.ndarray) -> None:
    with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as pending:
        try:
            np.save(pending, data, allow_pickle=False)
            pending.close()
            os.replace(pending.name, output)
        finally:
            Path(pending.name).unlink(missing_ok=True)


def encode_foundation(names, photos: list[Path], cache: Path, workers: int) -> dict[str, np.ndarray]:
    bases = set().union(*(REPRESENTATIONS[name] for name in names))
    features = {}
    crop_bases = bases - {"sface"}
    if crop_bases:
        pixels = face_assets(photos, cache, workers)
        digest = hashlib.sha256(pixels).hexdigest()
        for name in sorted(crop_bases):
            version = MODELS[name][1] if name in MODELS else "torchvision-047dcff4"
            output = cache / f"foundation-v1-{name}-{version}-{digest}.npy"
            if not output.is_file():
                print(f"Encoding {len(photos)} photos with {name}", flush=True)
                data = activations(pixels) if name == "mobile_tail" else frozen_features(name, pixels)
                _save_cache(output, data)
            features[name] = np.load(output, allow_pickle=False)
    if "sface" in bases:
        features["sface"] = face_assets(photos, cache, workers, sface=True)
    return {name: fuse([features[base] for base in REPRESENTATIONS[name]]) for name in names}
