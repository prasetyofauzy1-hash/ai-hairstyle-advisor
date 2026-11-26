"""Score the committed face-shape models on FaceBench FFHQ after training-identity screening."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from ml.classification.learned import DEFAULT_MODEL
from ml.classification.rule_based import SHAPES
from ml.datasets.build_features import analyze_items
from ml.datasets.ffhq_features import SUBSET, max_cosine
from ml.datasets.ffhq_select import fetch
from ml.datasets.identity import SAME_PERSON_COSINE
from ml.training.train_face_shape import metrics
from ml.training.train_face_shape_ffhq import ARTIFACT, phase2_rules, predict_bundle

MODELS = {
    "Kaggle-only MLP": DEFAULT_MODEL,
    "Kaggle + FFHQ AI pseudo-labels MLP": DEFAULT_MODEL.with_name(ARTIFACT),
}
LABELS = sorted(SHAPES)


def _read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def training_reference(images: Path, workers: int) -> np.ndarray:
    split = {r["ffhq_id"] for r in _read(SUBSET / "split.csv") if r["split"] == "train"}
    manifest = {r["ffhq_id"]: r for r in _read(SUBSET / "manifest.csv")}

    def item(ffhq_id: str) -> tuple[Path, str, str, str]:
        path = fetch(int(ffhq_id), manifest[ffhq_id]["file_md5"], images)
        if path is None:
            raise ValueError(f"Training reference failed the FFHQ checksum: {ffhq_id}")
        return images, path.name, "", "train"

    with ThreadPoolExecutor(max_workers=workers) as pool:
        items = list(pool.map(item, sorted(split)))
    _, embeddings = analyze_items(items, workers)
    return embeddings


def screened_frame(rows: list[dict], embeddings: np.ndarray, references: dict[str, np.ndarray],
                   identity_review: list[dict]) -> tuple[pd.DataFrame, dict]:
    quality = np.array([r["passed"] for r in rows], dtype=bool)
    valid = np.isfinite(embeddings).all(axis=1)
    screenable = quality & valid
    flagged = np.zeros(len(rows), dtype=bool)
    reference_counts = {}
    for name, ref in references.items():
        usable = ref[np.isfinite(ref).all(axis=1)]
        if not len(usable):
            raise ValueError(f"No usable identity embeddings in {name}")
        close = screenable & (max_cosine(embeddings, usable) >= SAME_PERSON_COSINE)
        flagged |= close
        reference_counts[name] = {"photos": len(ref), "embedded": len(usable), "matches": int(close.sum())}
    review = {r["facebench_file"]: r for r in identity_review}
    if len(review) != len(identity_review):
        raise ValueError("Duplicate FaceBench photos in identity review")
    files = np.array([r["image"] for r in rows])
    flagged_files = set(files[flagged])
    if (not set(review) <= set(files[screenable])
            or {name for name, row in review.items() if row["sface_flagged"] == "yes"} != flagged_files
            or any(row["sface_flagged"] not in {"yes", "no"} for row in review.values())):
        raise ValueError("Identity review does not cover the current FaceBench screen")
    if any(row["decision"] not in {"confirmed", "not_confirmed"} for row in review.values()):
        raise ValueError("Invalid FaceBench identity review decision")
    confirmed = {name for name, row in review.items() if row["decision"] == "confirmed"}
    matched = np.isin(files, list(confirmed))
    keep = screenable & ~matched
    if not keep.any():
        raise ValueError("No FaceBench photos remain after quality and identity screening")

    frame = pd.DataFrame([r for r, retain in zip(rows, keep, strict=True) if retain])
    truth = frame["label"].to_numpy()
    return frame, {
        "candidate_photos": len(rows),
        "quality_passed": int(quality.sum()),
        "missing_identity_embedding": int((quality & ~valid).sum()),
        "cosine_threshold": SAME_PERSON_COSINE,
        "references": reference_counts,
        "sface_flagged": int(flagged.sum()),
        "identity_reviewed": len(review),
        "sface_flagged_confirmed": len(flagged_files & confirmed),
        "identity_matches_excluded": int(matched.sum()),
        "test_photos": int(keep.sum()),
        "class_counts": {label: int((truth == label).sum()) for label in LABELS},
        "macro_labels": sorted(set(truth)),
    }


def score(rows: list[dict], embeddings: np.ndarray, references: dict[str, np.ndarray],
          identity_review: list[dict]) -> dict:
    frame, report = screened_frame(rows, embeddings, references, identity_review)
    truth = frame["label"].to_numpy()

    def evaluate(predictions: np.ndarray) -> dict:
        result = metrics(truth, predictions, report["macro_labels"])
        result["confusion_labels"] = LABELS
        result["confusion_matrix"] = confusion_matrix(truth, predictions, labels=LABELS).tolist()
        return result

    results = {"Phase 2 rules": evaluate(phase2_rules(frame))}
    for name, path in MODELS.items():
        bundle = joblib.load(path)
        result = evaluate(predict_bundle(bundle, frame[bundle["feature_names"]].to_numpy(float)))
        result["artifact"] = path.name
        result["artifact_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        results[name] = result
    return {**report, "models": results}


def markdown(report: dict) -> str:
    lines = [
        "# FaceBench FFHQ: Phase 3 external evaluation", "",
        "Generated by `python -m ml.evaluation.facebench`. See [reproduction and protocol]("
        "../../../../docs/datasets.md#follow-up-public-pre-labeled-evaluation-data-2026-11-26).", "",
        f"Of {report['candidate_photos']} FaceBench FFHQ photos, {report['quality_passed']} passed quality checks. "
        f"Of those, {report['missing_identity_embedding']} lacked an identity embedding and "
        f"{report['sface_flagged']} were flagged as possible matches at SFace cosine ≥ {report['cosine_threshold']:.3f}. "
        f"The [identity audit](identity_review.csv) confirmed {report['sface_flagged_confirmed']} of those flags. "
        f"Only the {report['identity_matches_excluded']} confirmed matches across "
        f"{report['identity_reviewed']} reviewed photos were excluded for identity. "
        f"**{report['test_photos']} photos were scored.**", "",
        "| Reference pool | Photos | Embedded | FaceBench photos flagged |", "|---|---:|---:|---:|",
    ]
    for name, ref in report["references"].items():
        lines.append(f"| {name} | {ref['photos']} | {ref['embedded']} | {ref['matches']} |")
    lines += [
        "", "The pools conservatively include all niten19 archive photos and the full FFHQ train split, "
        "not just the rows used to fit the artifacts. Unconfirmed is not proof of different identities. "
        "No photos, embeddings, names or per-photo predictions are published.", "",
        "| True shape | Photos |", "|---|---:|",
    ]
    lines.extend(f"| {label} | {count} |" for label, count in report["class_counts"].items())
    lines += [
        "", f"Macro-F1 equally weights the shapes with retained ground-truth examples "
        f"({', '.join(report['macro_labels'])}); predicting an unsupported shape still counts as an error. "
        "The [Uzy benchmark](../kaggle_plus_ffhq_ai/README.md) weights its own labeled classes, so the scores "
        "are not directly comparable. The small retained set cannot establish a model ranking or performance "
        "on unsupported classes.", "",
        "| Model | Macro-F1 | Accuracy |", "|---|---:|---:|",
    ]
    for name, model in report["models"].items():
        lines.append(f"| {name} | {model['macro_f1']:.3f} | {model['accuracy']:.3f} |")
    lines += ["", "## Confusion matrices", "", "Rows are FaceBench labels; columns are model predictions."]
    for name, model in report["models"].items():
        labels = model["confusion_labels"]
        lines += ["", f"### {name}", "", "| True / predicted | " + " | ".join(labels) + " |",
                  "|---|" + "---:|" * len(labels)]
        for label, counts in zip(labels, model["confusion_matrix"], strict=True):
            lines.append(f"| {label} | " + " | ".join(map(str, counts)) + " |")
    return "\n".join(lines) + "\n"


def benchmark_inputs(manifest: Path, images: Path, ffhq_train: Path, niten_embeddings: Path, workers: int
                     ) -> tuple[list[dict], np.ndarray, dict[str, np.ndarray]]:
    niten = np.load(niten_embeddings)
    if niten.shape != (5000, 128):
        raise ValueError("Expected embeddings from all 5,000 niten19 archive photos")
    candidates = _read(manifest)
    items = []
    for row in candidates:
        path = images / row["local_file"]
        if hashlib.md5(path.read_bytes()).hexdigest() != row["file_md5"]:
            raise ValueError(f"FaceBench photo failed the FFHQ checksum: {path.name}")
        items.append((images, row["local_file"], row["label"], "test"))
    rows, embeddings = analyze_items(items, workers)
    ffhq = training_reference(ffhq_train, workers)
    return rows, embeddings, {"niten19 archive": niten, "FFHQ train split": ffhq}


def run(manifest: Path, images: Path, ffhq_train: Path, niten_embeddings: Path, out: Path, workers: int,
        identity_review: Path) -> dict:
    report = score(*benchmark_inputs(manifest, images, ffhq_train, niten_embeddings, workers), _read(identity_review))
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (out / "README.md").write_text(markdown(report), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--images", type=Path, required=True, help="FaceBench FFHQ candidate photos")
    parser.add_argument("--ffhq-train", type=Path, required=True, help="Local cache for the existing FFHQ train photos")
    parser.add_argument("--niten-embeddings", type=Path, required=True, help="embeddings.npy from ml.datasets.build_features")
    parser.add_argument("--identity-review", type=Path, required=True, help="CSV with the audited identity decisions")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("ml.pipeline").setLevel(logging.WARNING)
    report = run(args.manifest, args.images, args.ffhq_train, args.niten_embeddings, args.out, args.workers,
                 args.identity_review)
    print(f"Scored {report['test_photos']} FaceBench photos; results in {args.out}")


if __name__ == "__main__":
    main()
