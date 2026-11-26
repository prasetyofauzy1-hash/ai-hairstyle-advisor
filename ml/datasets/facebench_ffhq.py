"""Download the non-commercial FFHQ face-shape subset of the human-labeled FaceBench evaluation set."""

from __future__ import annotations

import argparse
import csv
import json
import re
import urllib.request
from pathlib import Path

from ml.datasets.ffhq_select import ALLOWED_LICENSES, FIELDS, fetch

LABELS_URL = (
    "https://huggingface.co/datasets/wxqlab/FaceBench/resolve/"
    "a4f72f7eba638af19476f62139462579e19df23a/test.jsonl"
)
EXISTING_MANIFEST = Path(__file__).with_name("ffhq_labeling") / "manifest.csv"
RESEARCH_LICENSES = ALLOWED_LICENSES | {"Attribution-NonCommercial License"}
LABELS = {
    "Oval face": "oval",
    "Round face": "round",
    "Square face": "square",
    "Oblong face": "oblong",
    "Heart-shaped face": "heart",
    "Diamond-shaped face": "diamond",
}
MANIFEST_FIELDS = [*FIELDS, "facebench_image_id", "label", "label_source", "label_url", "split"]


def download(metadata: Path, images: Path, manifest: Path) -> list[dict]:
    with EXISTING_MANIFEST.open(encoding="utf-8", newline="") as f:
        excluded = {int(row["ffhq_id"]) for row in csv.DictReader(f)}
    with metadata.open(encoding="utf-8") as f:
        ffhq = json.load(f)
    candidates = {}
    with urllib.request.urlopen(LABELS_URL, timeout=60) as response:
        for line in response:
            if not line.strip():
                continue
            row = json.loads(line)
            if (row["question_id"] != "skin_q2" or row["question_type"] != "SCQ"
                    or row["metadata"]["image_source"] != "FFHQ"):
                continue
            image_id = row["image_id"]
            match = re.fullmatch(r"test-FFHQ-(\d{5})\.png", image_id)
            if not match or row["gt_answer"] not in LABELS:
                raise ValueError(f"Unexpected FaceBench face-shape row: {image_id}")
            idx = int(match[1])
            label = LABELS[row["gt_answer"]]
            if idx in candidates and candidates[idx][1] != label:
                raise ValueError(f"Conflicting FaceBench face-shape labels: {image_id}")
            candidates[idx] = (image_id, label)
    if not candidates:
        raise ValueError("No FaceBench FFHQ face-shape rows found")

    rows = []
    for idx, (image_id, label) in sorted(candidates.items()):
        if idx in excluded:
            continue
        entry = ffhq[str(idx)]
        photo = entry["metadata"]
        if photo["license"] not in RESEARCH_LICENSES:
            continue
        md5 = entry["image"]["file_md5"]
        path = fetch(idx, md5, images)
        if path is None:
            raise ValueError(f"FFHQ image failed the published MD5 check: {idx:05d}")
        rows.append({
            "ffhq_id": f"{idx:05d}", "license": photo["license"], "license_url": photo["license_url"],
            "author": photo["author"], "photo_url": photo["photo_url"], "photo_title": photo["photo_title"],
            "file_md5": md5, "local_file": path.name, "facebench_image_id": image_id, "label": label,
            "label_source": "FaceBench", "label_url": LABELS_URL, "split": "test",
        })

    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metadata", type=Path, help="NVIDIA's ffhq-dataset-v2.json")
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    rows = download(args.metadata, args.images, args.manifest)
    print(f"Wrote {len(rows)} FaceBench FFHQ labels to {args.manifest}")


if __name__ == "__main__":
    main()
