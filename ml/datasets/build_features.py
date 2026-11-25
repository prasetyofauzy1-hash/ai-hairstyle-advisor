"""Run the Phase 2 pipeline over a folder-per-class dataset and write a feature table plus identity groups.

    python -m ml.datasets.build_features data/raw/niten19 --out data/processed/niten19

Expects <root>/<split>/<Class>/*.jpg (Kaggle niten19 layout). Writes features.csv (one row per image, rejected
ones included with their quality codes) and embeddings.npy (local only, gitignored). No pixels are written or logged.
"""

from __future__ import annotations

import argparse
import csv
import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from ml.datasets.identity import FaceEmbedder, identity_groups
from ml.feature_extraction.geometry import FEATURE_NAMES
from ml.models.download import MODELS, ensure_model
from ml.pipeline import FOREHEAD_OCCLUSION_THRESHOLD, FaceAnalyzer
from ml.preprocessing.image_io import ImageDecodeError, load_image

log = logging.getLogger(__name__)
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
_analyzer: FaceAnalyzer | None = None
_embedder: FaceEmbedder | None = None


def list_images(root: Path) -> list[tuple[str, str, str]]:
    items = []
    for split_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            for f in sorted(class_dir.iterdir()):
                if f.suffix.lower() in IMAGE_EXTS:
                    items.append((str(f.relative_to(root)), class_dir.name.lower(), split_dir.name))
    return items


def _init() -> None:
    global _analyzer, _embedder
    _analyzer = FaceAnalyzer()
    _embedder = FaceEmbedder()


def _process(args: tuple[Path, str, str, str]) -> tuple[dict, np.ndarray]:
    root, rel, label, split = args
    row = {"image": rel, "label": label, "source_split": split, "passed": False, "issues": ""}
    emb = np.full(128, np.nan, dtype=np.float32)
    try:
        image = load_image(root / rel)
    except (ImageDecodeError, OSError):
        row["issues"] = "decode_error"
        return row, emb
    result = _analyzer.analyze(image)
    row["passed"] = result.quality.passed
    row["issues"] = ";".join(i.code for i in result.quality.issues)
    for k in ("yaw", "pitch", "roll", "sharpness", "brightness", "face_width_px"):
        row[k] = result.quality.metrics.get(k)
    if result.face is not None:
        emb = _embedder.embed(image, result.face.landmarks)
    if result.features is not None:
        for name in FEATURE_NAMES:
            row[name] = getattr(result.features, name)
        row["forehead_hair_fraction"] = result.forehead_hair_fraction
        row["forehead_occluded"] = result.forehead_hair_fraction >= FOREHEAD_OCCLUSION_THRESHOLD
        row["rule_based_shape"] = result.prediction.shape
    return row, emb


def build(root: Path, out: Path, workers: int = 4) -> list[dict]:
    items = [(root, *it) for it in list_images(root)]
    for name in MODELS:  # download once here, so workers never race on the same temp file
        ensure_model(name)
    with ProcessPoolExecutor(max_workers=workers, initializer=_init,
                             mp_context=multiprocessing.get_context("spawn")) as pool:
        results = list(pool.map(_process, items, chunksize=16))
    rows = [r for r, _ in results]
    embeddings = np.stack([e for _, e in results])
    for row, g in zip(rows, identity_groups(embeddings)):
        row["identity_group"] = int(g)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "embeddings.npy", embeddings)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with (out / "features.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    log.info("wrote %d rows, %d passed quality gates", len(rows), sum(r["passed"] for r in rows))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("ml.pipeline").setLevel(logging.WARNING)
    build(args.root, args.out, args.workers)


if __name__ == "__main__":
    main()
