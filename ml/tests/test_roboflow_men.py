import hashlib
import zipfile

import pandas as pd
import pytest

from ml.datasets import roboflow_men
from ml.feature_extraction.geometry import FEATURE_NAMES


def test_stage_verifies_archive_and_retains_box_provenance(monkeypatch, tmp_path):
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, mode="w") as data:
        for i in range(604):
            data.writestr(f"train/images/{i:03d}.jpg", str(i).encode())
            data.writestr(f"train/labels/{i:03d}.txt", "0 .5 .5 .5 .5\n" * (2 if i == 0 else 1))
    with pytest.raises(ValueError, match="SHA-256 differs"):
        roboflow_men.stage(archive, tmp_path)

    monkeypatch.setattr(roboflow_men, "ARCHIVE_SHA256", hashlib.sha256(archive.read_bytes()).hexdigest())
    rows = roboflow_men.stage(archive, tmp_path)
    assert (rows[0]["box_count"], rows[0]["label"]) == (2, "")
    assert (rows[1]["box_count"], rows[1]["label"]) == (1, "diamond")
    assert rows[1]["file_sha256"] == hashlib.sha256(b"1").hexdigest()
    assert (tmp_path / "images" / rows[1]["local_file"]).read_bytes() == b"1"


def test_load_split_keeps_verified_membership_and_rejects_repeated_hash(tmp_path):
    photos, features, memberships = [], [], []
    for i, (split, label) in enumerate((s, c) for s in ("train", "test") for c in roboflow_men.CLASSES[:-1]):
        photo, digest = f"{i}.jpg", f"{i:064x}"
        photos.append({"local_file": photo, "file_sha256": digest, "label": label, "box_count": 1, "source_split": "train"})
        features.append({"image": photo, "passed": True, "label": label, "source_split": "train",
                         **{name: 1.0 for name in FEATURE_NAMES}})
        memberships.append({"file_sha256": digest, "split": split})
    manifest, feature_csv, split_csv = [tmp_path / f"{name}.csv" for name in ("manifest", "features", "split")]
    for path, rows in zip((manifest, feature_csv, split_csv), (photos, features, memberships), strict=True):
        pd.DataFrame(rows).to_csv(path, index=False)
    train, test = roboflow_men.load_split(manifest, feature_csv, split_csv)
    assert len(train) == len(test) == 5
    assert not set(train["file_sha256"]) & set(test["file_sha256"])

    pd.DataFrame([*memberships, {**memberships[0], "split": "test"}]).to_csv(split_csv, index=False)
    with pytest.raises(ValueError, match="duplicate Roboflow split entries"):
        roboflow_men.load_split(manifest, feature_csv, split_csv)
