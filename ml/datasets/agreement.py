"""Inter-annotator agreement and consensus labels for the self-labeled FFHQ set.

    python -m ml.datasets.agreement labels_*.csv --out ml/datasets/ffhq_labeling/consensus.csv

Reads the CSVs exported by the labeling tool (ffhq_id, annotator, label). Reports Cohen's kappa for two annotators
and Fleiss' kappa for two or more (images labeled by every annotator only), and keeps an image in the consensus set
only if every annotator gave the same non-"ragu" label.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

UNSURE = "ragu"


def read_labels(paths: list[Path]) -> dict[str, dict[str, str]]:
    """{ffhq_id: {annotator: label}}"""
    out: dict[str, dict[str, str]] = defaultdict(dict)
    for p in paths:
        with p.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                out[r["ffhq_id"]][r["annotator"]] = r["label"]
    return dict(out)


def cohen_kappa(a: list[str], b: list[str]) -> float:
    cats = sorted(set(a) | set(b))
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[c] * cb[c] for c in cats) / (n * n)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def fleiss_kappa(table: np.ndarray) -> float:
    """table: (items, categories) counts, same number of raters per item."""
    n = table.sum(axis=1)[0]
    p_j = table.sum(axis=0) / table.sum()
    p_i = (np.square(table).sum(axis=1) - n) / (n * (n - 1))
    p_bar, pe = p_i.mean(), np.square(p_j).sum()
    return 1.0 if pe == 1 else float((p_bar - pe) / (1 - pe))


def agreement(labels: dict[str, dict[str, str]]) -> tuple[dict, list[dict]]:
    annotators = sorted({a for v in labels.values() for a in v})
    if len(annotators) < 2:
        raise ValueError("agreement needs at least two annotators")
    complete = {k: v for k, v in labels.items() if set(v) == set(annotators)}
    cats = sorted({l for v in complete.values() for l in v.values()})
    table = np.array([[list(v.values()).count(c) for c in cats] for v in complete.values()])
    report = {"annotators": annotators, "images_labeled_by_all": len(complete),
              "fleiss_kappa": fleiss_kappa(table) if len(complete) else None}
    if len(annotators) == 2 and complete:
        a, b = annotators
        report["cohen_kappa"] = cohen_kappa([v[a] for v in complete.values()], [v[b] for v in complete.values()])
    consensus = [{"ffhq_id": k, "label": next(iter(set(v.values())))} for k, v in sorted(complete.items())
                 if len(set(v.values())) == 1 and UNSURE not in v.values()]
    report["consensus_images"] = len(consensus)
    report["consensus_class_counts"] = dict(Counter(c["label"] for c in consensus))
    return report, consensus


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("labels", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report, consensus = agreement(read_labels(args.labels))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["ffhq_id", "label"])
        w.writeheader()
        w.writerows(consensus)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
