"""Prepare a local blinded face-shape review and score frozen predictions."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from sklearn.model_selection import train_test_split

from ml.classification.rule_based import SHAPES
from ml.training.kaggle_finetune import COHORTS, validate, write_json
from ml.training.kaggle_kernel import digest
from ml.training.train_face_shape_image import subset_metrics

HOLDOUTS = COHORTS[1:]
SAMPLE_SIZE = 120
SEED = 42
PAGE_SIZE = 20


def sample_rows(cohort, labels):
    rows = np.arange(len(labels))
    if cohort == "niten19_holdout":
        rows, _ = train_test_split(rows, train_size=SAMPLE_SIZE, stratify=labels, random_state=SEED)
    return np.sort(rows)


def archive(package, cohort):
    return package / cohort / f"{cohort}.npz"


def write_csv(path, fields, rows):
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(fields)
        writer.writerows(rows)


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def stage(package, workspace):
    validate(package)
    workspace.mkdir(mode=0o700)
    images, pages = workspace / "images", workspace / "pages"
    images.mkdir()
    pages.mkdir()
    items = []
    for cohort in HOLDOUTS:
        with np.load(archive(package, cohort), allow_pickle=False) as data:
            pixels = data["pixels"]
            items.extend((cohort, int(row), pixels[row]) for row in sample_rows(cohort, data["label"]))
    random.Random(SEED).shuffle(items)
    mapping = []
    for start in range(0, len(items), PAGE_SIZE):
        canvas = Image.new("RGB", (5 * 160, 4 * 181), "#eee")
        draw = ImageDraw.Draw(canvas)
        for offset, (cohort, row, pixels) in enumerate(items[start:start + PAGE_SIZE]):
            tile = start + offset + 1
            name = f"{uuid4().hex}.png"
            image = Image.fromarray(pixels)
            image.save(images / name)
            x, y = (offset % 5) * 160, (offset // 5) * 181
            canvas.paste(image, (x + 8, y))
            draw.text((x + 8, y + 148), f"{tile:03d}", fill="black")
            mapping.append((tile, cohort, row))
        canvas.save(pages / f"page-{start // PAGE_SIZE + 1:02d}.png")
    write_csv(workspace / "private-mapping.csv", ("tile", "cohort", "archive_row"), mapping)
    print(f"Staged {len(items)} blinded crops.")


def freeze(mapping, blind, output):
    keys = {int(row["tile"]): (row["cohort"], int(row["archive_row"])) for row in read_csv(mapping)}
    calls = read_csv(blind)
    decisions = {int(row["tile"]): row["prediction"] for row in calls}
    if len(decisions) != len(calls) or decisions.keys() != keys.keys() or set(decisions.values()) - set(SHAPES):
        raise ValueError("Every blinded tile needs exactly one valid prediction")
    write_csv(output, ("cohort", "archive_row", "prediction"),
              sorted((*keys[tile], prediction) for tile, prediction in decisions.items()))
    print(f"Froze {len(decisions)} predictions before scoring.")


def score(package, predictions, output):
    ledger, _ = validate(package)
    rows = read_csv(predictions)
    decisions = {(row["cohort"], int(row["archive_row"])): row["prediction"] for row in rows}
    if len(decisions) != len(rows) or set(decisions.values()) - set(SHAPES):
        raise ValueError("Invalid or duplicate prediction")
    benchmarks = {}
    expected = set()
    for cohort in HOLDOUTS:
        with np.load(archive(package, cohort), allow_pickle=False) as data:
            labels = data["label"]
        indices = sample_rows(cohort, labels)
        keys = [(cohort, int(row)) for row in indices]
        expected.update(keys)
        if not set(keys).issubset(decisions):
            raise ValueError("Predictions do not cover the frozen benchmark rows")
        benchmarks[cohort] = subset_metrics(pd.DataFrame({"label": labels[indices]}),
                                            np.asarray(ledger["gender"][cohort])[indices],
                                            {"vision_zero_shot": np.asarray([decisions[k] for k in keys])})
    if expected != decisions.keys():
        raise ValueError("Predictions contain unexpected benchmark rows")
    write_json(output, {"prediction_sha256": digest(predictions), "niten19_sample_seed": SEED,
                        "archive_sha256": {k: ledger["archives"][k]["sha256"] for k in HOLDOUTS},
                        "benchmarks": benchmarks})
    print(f"Scored {len(expected)} frozen predictions across {len(benchmarks)} holdouts.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    for action, args in {"stage": ("package", "workspace"), "freeze": ("mapping", "blind", "output"),
                         "score": ("package", "predictions", "output")}.items():
        sub = actions.add_parser(action)
        for arg in args:
            sub.add_argument(arg, type=Path)
    values = vars(parser.parse_args())
    {"stage": stage, "freeze": freeze, "score": score}[values.pop("action")](**values)


if __name__ == "__main__":
    main()
