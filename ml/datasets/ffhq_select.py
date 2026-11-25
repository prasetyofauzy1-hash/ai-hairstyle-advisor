"""Select FFHQ images for human face-shape labeling (Phase 3, TD-14) and write a license manifest.

    python -m ml.datasets.ffhq_select data/raw/ffhq-dataset-v2.json --images data/raw/ffhq \
        --manifest ml/datasets/ffhq_labeling/manifest.csv --target 600

Candidates are a seeded random order of FFHQ images whose Flickr license allows attribution-only reuse or is public
domain (non-commercial licenses are skipped so the set stays shareable). Each image is downloaded from a Hugging
Face mirror, checked against NVIDIA's MD5, and kept only if it passes the Phase 2 quality gates. Images stay local
(gitignored); only the manifest (ids, licenses, attribution, checksums) is committed. No labels are produced here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import random
import urllib.request
from pathlib import Path

from ml.pipeline import FaceAnalyzer
from ml.preprocessing.image_io import load_image

log = logging.getLogger(__name__)
MIRROR = "https://huggingface.co/datasets/marcosv/ffhq-dataset/resolve/main/Part{part}/{idx:05d}.png"
ALLOWED_LICENSES = {"Attribution License", "Public Domain Mark", "Public Domain Dedication (CC0)",
                    "United States Government Work"}
SEED = 42
FIELDS = ["ffhq_id", "license", "license_url", "author", "photo_url", "photo_title", "file_md5", "local_file"]


def candidates(meta: dict, seed: int = SEED) -> list[int]:
    ids = [int(k) for k, v in meta.items() if v["metadata"]["license"] in ALLOWED_LICENSES]
    random.Random(seed).shuffle(ids)
    return ids


def fetch(idx: int, md5: str, images: Path) -> Path | None:
    path = images / f"{idx:05d}.png"
    if not path.exists():
        images.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(MIRROR.format(part=idx // 10000 + 1, idx=idx), tmp)
        tmp.replace(path)
    if hashlib.md5(path.read_bytes()).hexdigest() != md5:
        path.unlink()
        return None
    return path


def select(meta_path: Path, images: Path, manifest: Path, target: int) -> list[dict]:
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    rows, tried, rejected = [], 0, 0
    with FaceAnalyzer() as analyzer:
        for idx in candidates(meta):
            if len(rows) >= target:
                break
            entry = meta[str(idx)]
            tried += 1
            path = fetch(idx, entry["image"]["file_md5"], images)
            if path is None or not analyzer.analyze(load_image(path)).quality.passed:
                rejected += 1
                continue
            m = entry["metadata"]
            rows.append({"ffhq_id": f"{idx:05d}", "license": m["license"], "license_url": m["license_url"],
                         "author": m["author"], "photo_url": m["photo_url"], "photo_title": m["photo_title"],
                         "file_md5": entry["image"]["file_md5"], "local_file": path.name})
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    log.info("selected %d images from %d candidates (%d rejected or failed checksum)", len(rows), tried, rejected)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("metadata", type=Path, help="NVIDIA's ffhq-dataset-v2.json")
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--target", type=int, default=600)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("ml.pipeline").setLevel(logging.WARNING)
    select(args.metadata, args.images, args.manifest, args.target)


if __name__ == "__main__":
    main()
