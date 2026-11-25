"""Run the Phase 2 pipeline over the labeled FFHQ subset and write a feature table.

    python -m ml.datasets.ffhq_features --images data/raw/ffhq --out data/processed/ffhq \
        --kaggle-embeddings data/processed/niten19/embeddings.npy --download

`--download` fetches the manifest's photos (MD5-checked) without re-running `ffhq_select`, which would rebuild the
manifest under the current quality gates and could drop photos the split and labels refer to.

Train rows carry the AI pseudo-labels (`ragu` dropped), test rows Uzy's human labels; `label_source` keeps the two
apart. Identity groups are computed over all FFHQ rows so a person in both train and test can be found, and
`max_cosine_to_kaggle` flags faces that may also be in niten19. No pixels are written or logged.
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import numpy as np

from ml.datasets.agreement import UNSURE
from ml.datasets.build_features import analyze_items
from ml.datasets.ffhq_select import fetch
from ml.datasets.identity import identity_groups

SUBSET = Path(__file__).resolve().parent / "ffhq_labeling"
log = logging.getLogger(__name__)


def _read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def labeled_items(subset: Path = SUBSET) -> list[dict]:
    """Manifest rows with a usable label: AI pseudo-labels for train, human labels for test, never mixed."""
    split = {r["ffhq_id"]: r["split"] for r in _read(subset / "split.csv")}
    ai = {r["ffhq_id"]: r for r in _read(subset / "ai_pseudo_labels_train.csv")}
    human = {r["ffhq_id"]: r for r in _read(subset / "human_labels_test.csv")}
    if set(ai) != {k for k, v in split.items() if v == "train"} or set(human) != {k for k, v in split.items() if v == "test"}:
        raise ValueError("AI labels must cover exactly the train split and human labels exactly the test split")
    items = []
    for r in _read(subset / "manifest.csv"):
        label = (human if split[r["ffhq_id"]] == "test" else ai)[r["ffhq_id"]]
        if label["label"] == UNSURE:
            continue
        items.append({"ffhq_id": r["ffhq_id"], "local_file": r["local_file"], "label": label["label"],
                      "split": split[r["ffhq_id"]], "label_source": label["source"]})
    return items


def max_cosine(embeddings: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Highest cosine similarity of each (unit) embedding to any valid reference embedding; NaN rows stay NaN."""
    ref = reference[~np.isnan(reference).any(axis=1)]
    return (embeddings @ ref.T).max(axis=1) if len(ref) else np.full(len(embeddings), np.nan)


def download(images: Path, subset: Path = SUBSET) -> None:
    bad = [r["ffhq_id"] for r in _read(subset / "manifest.csv")
           if fetch(int(r["ffhq_id"]), r["file_md5"], images) is None]
    if bad:
        raise ValueError(f"{len(bad)} FFHQ photos failed the manifest checksum: {bad[:5]}")


def build(images: Path, out: Path, kaggle_embeddings: Path | None = None, workers: int = 4) -> list[dict]:
    items = labeled_items()
    rows, embeddings = analyze_items([(images, it["local_file"], it["label"], it["split"]) for it in items], workers)
    similar = max_cosine(embeddings, np.load(kaggle_embeddings)) if kaggle_embeddings else None
    for i, (row, it, g) in enumerate(zip(rows, items, identity_groups(embeddings))):
        row.update(ffhq_id=it["ffhq_id"], label_source=it["label_source"], identity_group=int(g))
        if similar is not None:
            row["max_cosine_to_kaggle"] = float(similar[i])
    out.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with (out / "features.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    log.info("wrote %d rows, %d passed quality gates", len(rows), sum(r["passed"] for r in rows))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--kaggle-embeddings", type=Path, help="niten19 embeddings.npy from ml.datasets.build_features")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--download", action="store_true", help="fetch missing manifest photos first")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("ml.pipeline").setLevel(logging.WARNING)
    if args.download:
        download(args.images)
    build(args.images, args.out, args.kaggle_embeddings, args.workers)


if __name__ == "__main__":
    main()
