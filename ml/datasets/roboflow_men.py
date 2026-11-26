"""Stage the pinned Roboflow Men's Face Shape v1 YOLOv8 export for local research."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd

from ml.datasets.build_features import analyze_items
from ml.feature_extraction.geometry import FEATURE_NAMES

SOURCE = "https://universe.roboflow.com/ahmer-irfan/men-s-face-shape/dataset/1"
EXPORT = "https://api.roboflow.com/ahmer-irfan/men-s-face-shape/1/yolov8"
ARCHIVE_SHA256 = "ed2dc70b1b81f13069ef582ea6ccc46e904e7048c47926a070ff65fa209fe9e2"
CLASSES = ("diamond", "oblong", "oval", "round", "square", "triangle")
IMAGE = re.compile(r"(train|valid|test)/images/[\w.-]+\.jpg", re.ASCII)
FIELDS = ("local_file", "source_split", "label", "box_count", "file_sha256", "annotation_sha256", "source_url")


def download(archive: Path) -> None:
    key = os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        raise ValueError("Set ROBOFLOW_API_KEY to download the pinned Roboflow export")
    try:
        for _ in range(12):
            with urllib.request.urlopen(EXPORT + "?" + urllib.parse.urlencode({"api_key": key}), timeout=45) as response:
                payload = json.load(response)
            link = payload.get("export", {}).get("link")
            if link:
                if urllib.parse.urlsplit(link).scheme != "https":
                    raise ValueError("Roboflow export link must use HTTPS")
                with urllib.request.urlopen(link, timeout=90) as response:
                    blob = response.read()
                if hashlib.sha256(blob).hexdigest() != ARCHIVE_SHA256:
                    raise ValueError("Roboflow export SHA-256 differs from the audited archive")
                archive.write_bytes(blob)
                return
            time.sleep(5)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Roboflow export request returned HTTP {exc.code}") from None
    except (urllib.error.URLError, TimeoutError):
        raise RuntimeError("Roboflow export request failed") from None
    raise RuntimeError("Roboflow export was not ready")


def stage(archive: Path, out: Path) -> list[dict]:
    if hashlib.sha256(archive.read_bytes()).hexdigest() != ARCHIVE_SHA256:
        raise ValueError("Roboflow export SHA-256 differs from the audited archive")
    images = out / "images"
    images.mkdir(parents=True, exist_ok=True)
    rows = []
    with zipfile.ZipFile(archive) as z:
        names = sorted(n for n in z.namelist() if IMAGE.fullmatch(n))
        if len(names) != 604 or len(set(names)) != len(names):
            raise ValueError("Expected 604 distinct Roboflow image members")
        for name in names:
            path = PurePosixPath(name)
            split = path.parts[0]
            annotation = z.read(f"{split}/labels/{path.stem}.txt")
            lines = annotation.splitlines()
            cls = int(lines[0].split()[0]) if len(lines) == 1 else -1
            if len(lines) == 1 and not 0 <= cls < len(CLASSES):
                raise ValueError("Unrecognized Roboflow class index")
            image = z.read(name)
            local = f"{split}_{path.name}"
            (images / local).write_bytes(image)
            rows.append(dict(local_file=local, source_split=split, label=CLASSES[cls] if cls >= 0 else "",
                             box_count=len(lines), file_sha256=hashlib.sha256(image).hexdigest(),
                             annotation_sha256=hashlib.sha256(annotation).hexdigest(), source_url=SOURCE))
    with (out / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def load_split(manifest: Path, features: Path, assignments: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    membership = pd.read_csv(assignments, dtype=str)
    if (membership["file_sha256"].duplicated().any()
            or not membership["file_sha256"].str.fullmatch(r"[0-9a-f]{64}", na=False).all()
            or set(membership["split"]) != {"train", "test"}):
        raise ValueError("Invalid or duplicate Roboflow split entries")
    photos = pd.read_csv(manifest, dtype={"file_sha256": str})
    rows = pd.read_csv(features)
    assigned = membership.merge(photos, on="file_sha256", how="left", validate="one_to_one")
    assigned = assigned.merge(
        rows, left_on=["local_file", "label", "source_split"], right_on=["image", "label", "source_split"],
        how="left", validate="one_to_one",
    )
    if (not assigned["passed"].eq(True).all() or not assigned["box_count"].eq(1).all()
            or not np.isfinite(assigned[FEATURE_NAMES].to_numpy(float)).all()
            or any(set(group["label"]) != set(CLASSES[:-1]) for _, group in assigned.groupby("split"))):
        raise ValueError("Roboflow split must contain verified single-face, quality-passing photos of all five supported labels")
    return tuple(assigned[assigned["split"] == split].reset_index(drop=True) for split in ("train", "test"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("ml.pipeline").setLevel(logging.WARNING)
    args.out.mkdir(parents=True, exist_ok=True)
    archive = args.out / "source.zip"
    if not archive.is_file():
        download(archive)
    rows = stage(archive, args.out)
    feature_rows, embeddings = analyze_items(
        [(args.out / "images", r["local_file"], r["label"], r["source_split"]) for r in rows], args.workers,
    )
    args.features.mkdir(parents=True, exist_ok=True)
    np.save(args.features / "embeddings.npy", embeddings)
    with (args.features / "features.csv").open("w", newline="", encoding="utf-8") as f:
        fields = list(dict.fromkeys(k for row in feature_rows for k in row))
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(feature_rows)
    print(f"Staged {len(rows)} photos; {sum(r['passed'] for r in feature_rows)} passed quality gates")


if __name__ == "__main__":
    main()
