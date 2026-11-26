import json

import numpy as np
import pytest
from PIL import Image

from ml.evaluation import vision_zero_shot as vision


def test_blind_workflow_stages_and_scores_exact_cohort_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(vision, "SAMPLE_SIZE", 2)
    labels = np.array(["round", "oval"] * 2)
    package = tmp_path / "package"
    gender = ["male", "female"] * 2
    ledger = {"gender": {}, "archives": {}}
    for cohort in vision.HOLDOUTS:
        path = vision.archive(package, cohort)
        path.parent.mkdir(parents=True)
        np.savez(path, label=labels, pixels=np.zeros((len(labels), 144, 144, 3), dtype=np.uint8))
        ledger["gender"][cohort] = gender
        ledger["archives"][cohort] = {"sha256": vision.digest(path)}
    monkeypatch.setattr(vision, "validate", lambda _: (ledger, {}))
    workspace = tmp_path / "review"
    vision.stage(package, workspace)
    images = list((workspace / "images").iterdir())
    assert len(images) == 14
    assert all(len(photo.stem) == 32 for photo in images)
    with Image.open(images[0]) as photo:
        assert not photo.info
    mapping = workspace / "private-mapping.csv"
    rows = vision.read_csv(mapping)
    assert [row["cohort"] for row in rows] != sorted(row["cohort"] for row in rows)
    assert set(rows[0]) == {"tile", "cohort", "archive_row"}
    blind = tmp_path / "blind.csv"
    predictions = tmp_path / "predictions.csv"
    calls = [(row["tile"], labels[int(row["archive_row"])]) for row in rows]
    vision.write_csv(blind, ("tile", "prediction"), calls[:-1] + calls[:1])
    with pytest.raises(ValueError, match="exactly one"):
        vision.freeze(mapping, blind, predictions)
    blind.unlink()
    vision.write_csv(blind, ("tile", "prediction"), calls)
    vision.freeze(mapping, blind, predictions)

    metrics = tmp_path / "metrics.json"
    vision.score(package, predictions, metrics)
    report = json.loads(metrics.read_text())
    assert report["prediction_sha256"] == vision.digest(predictions)
    results = report["benchmarks"]
    assert results["niten19_holdout"]["all"]["photos"] == 2
    for cohort in vision.HOLDOUTS:
        assert results[cohort]["male"]["models"]["vision_zero_shot"]["accuracy"] == 1.0
