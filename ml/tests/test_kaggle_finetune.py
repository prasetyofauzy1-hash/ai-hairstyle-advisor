from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml.training import kaggle_finetune as workflow
from ml.training.kaggle_kernel import grouped_folds


@pytest.fixture
def staged(tmp_path, monkeypatch):
    source = tmp_path / "photos"
    source.mkdir()
    (tmp_path / "json").mkdir()
    frames = {}
    for key in workflow.COHORTS:
        photo = source / f"{key}.jpg"
        photo.write_bytes(key.encode())
        frames[key] = pd.DataFrame({"image": [photo.name], "label": ["oval"], "identity_group": [0]})
    monkeypatch.setattr(workflow, "training_cohorts", lambda _: ([(frames["train"], source)], None))
    def local_benchmarks(*_, allow_download):
        assert allow_download is False
        return {key: (frames[key], source, None) for key in workflow.COHORTS[1:]}

    monkeypatch.setattr(workflow, "benchmark_cohorts", local_benchmarks)
    monkeypatch.setattr(workflow, "benchmark_gender", lambda *_: np.array(["male"]))
    monkeypatch.setattr(workflow, "face_assets", lambda *_: np.zeros((1, 144, 144, 3), dtype=np.uint8))
    return Namespace(workspace=tmp_path / "pack", workers=1, owner="private-owner",
                     prefix="face-shape", cache=None, attributes=tmp_path)


def test_packages_separate_private_uploads(staged):
    workflow.package(staged)
    ledger, ids = workflow.validate(staged.workspace)
    assert ledger["gender"] == {key: ["male"] for key in workflow.COHORTS[1:]}
    assert len(set(ids["datasets"].values())) == 5
    for key in workflow.COHORTS:
        assert workflow.read_json(staged.workspace / key / "dataset-metadata.json")["isPrivate"] is True
        with np.load(staged.workspace / key / f"{key}.npz", allow_pickle=False) as archive:
            assert set(archive.files) == ({"pixels", "label", "group"} if key == "train" else {"pixels", "label"})
    path = staged.workspace / "kernel/kernel-metadata.json"
    metadata = workflow.read_json(path)
    assert metadata["dataset_sources"] == [ids["datasets"]["train"]]
    metadata["dataset_sources"].append(ids["datasets"]["uzy"])
    workflow.write_json(path, metadata)
    with pytest.raises(ValueError, match="attached only to the training dataset"):
        workflow.validate(staged.workspace)


def test_rejects_training_photo_in_holdout(staged):
    photos = staged.workspace.parent / "photos"
    (photos / "uzy.jpg").write_bytes((photos / "train.jpg").read_bytes())
    with pytest.raises(ValueError, match="Exact training-photo overlap with uzy"):
        workflow.package(staged)


def test_credentials_require_configured_owner(tmp_path, monkeypatch):
    monkeypatch.setenv("KAGGLE_CONFIG_DIR", str(tmp_path))
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    with pytest.raises(ValueError, match="configure kaggle.json"):
        workflow.credentials("right-owner")
    workflow.write_json(tmp_path / "kaggle.json", {"username": "right-owner", "key": "mock-key"})
    assert workflow.credentials("right-owner") == {"KAGGLE_USERNAME": "right-owner", "KAGGLE_KEY": "mock-key"}
    monkeypatch.setenv("KAGGLE_USERNAME", "wrong-owner")
    monkeypatch.setenv("KAGGLE_KEY", "other-mock-key")
    with pytest.raises(ValueError, match="do not match"):
        workflow.credentials("right-owner")


def test_submit_uses_only_verified_private_training_data(staged, monkeypatch):
    workflow.package(staged)
    staged.confirm_upload_rights = True
    staged.wait_minutes = 1
    staged.update_datasets = False
    staged.upload_holdouts = False
    monkeypatch.setenv("KAGGLE_USERNAME", staged.owner)
    monkeypatch.setenv("KAGGLE_KEY", "mock-key")
    monkeypatch.setenv("KAGGLE_API_TOKEN", "unused-mock-token")
    _, ids = workflow.validate(staged.workspace)
    version = ids["kernel"] + "/1"
    scored = []
    monkeypatch.setattr(workflow, "score", lambda *args: scored.append(args))
    responses = iter([
        (("datasets", "create", "-p", str(staged.workspace / "train"), "-q", "-t"),
         "Your private Dataset is being created."),
        (("datasets", "status", ids["datasets"]["train"], "--format", "json"),
         '{"status":"ready","current_version_number":1}'),
        (("datasets", "metadata", ids["datasets"]["train"]), ""),
        (("kernels", "push", "-p", str(staged.workspace / "kernel"), "--accelerator", "NvidiaTeslaT4"),
         "Kernel version 1 successfully pushed."),
        (("kernels", "status", version), 'Kernel has status "KernelWorkerStatus.COMPLETE"'),
        (("kernels", "output", version), ""),
    ])

    def fake_cli(environment, *command):
        assert "KAGGLE_API_TOKEN" not in environment
        assert environment["KAGGLE_KEY"] == "mock-key"
        expected, result = next(responses)
        assert command[:len(expected)] == expected
        if command[:2] == ("datasets", "metadata"):
            workflow.write_json(Path(command[-1]) / "dataset-metadata.json", {
                "isPrivate": True, "ownerUser": staged.owner, "datasetSlug": "face-shape-train",
            })
        return result

    monkeypatch.setattr(workflow, "cli", fake_cli)
    workflow.submit(staged)
    assert scored == [(staged.workspace, staged.workspace / "outputs")]
    assert list(responses) == []


def test_grouped_folds_keep_identity_together():
    labels = np.repeat([0, 1], 10)
    groups = np.repeat(np.arange(10), 2)
    _, folds = grouped_folds(labels, groups)
    assert len(folds) == 5
    for train, valid in folds:
        assert set(groups[train]).isdisjoint(groups[valid])
        assert set(labels[train]) == set(labels[valid]) == {0, 1}
