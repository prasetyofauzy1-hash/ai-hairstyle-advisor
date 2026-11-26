import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ml.training import kaggle_finetune as workflow, kaggle_kernel as kernel
from ml.training import kaggle_stronger as stronger
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


def test_stronger_stage_keeps_same_training_only_mount(staged, tmp_path):
    workflow.package(staged)
    before, ids = workflow.validate(staged.workspace)
    target = tmp_path / "stronger"

    stronger.stage(staged.workspace, target)

    after, staged_ids = workflow.validate(target)
    assert staged_ids == ids
    assert after["archives"] == before["archives"]
    assert after["kernel_sha256"] != before["kernel_sha256"]
    assert b'DEFAULT_EXPERIMENT = "stronger"' in (target / "kernel/kaggle_kernel.py").read_bytes()
    assert workflow.read_json(target / "kernel/kernel-metadata.json")["machine_shape"] == "NvidiaTeslaT4"


def test_fusion_weight_is_selected_on_all_grouped_validation_folds():
    classes = np.array(["oblong", "oval", "round"])
    labels = np.tile(classes, 3)
    groups = np.arange(len(labels))
    _, folds = grouped_folds(labels, groups, maximum=3)
    correct = np.tile(np.eye(3), (3, 1))
    neural = correct.copy()
    neural[::2] = (correct[::2] + 0.2 * np.roll(correct[::2], 1, axis=1)) / 1.2
    hybrid = correct.copy()
    hybrid[1::2] = np.roll(correct[1::2], 1, axis=1)

    report = stronger.choose_fusion(labels, classes, folds, neural, hybrid)

    assert report["selected"]["neural_weight"] == 0.75
    assert report["selected"]["fold_macro_f1"] == [1.0] * 3
    neural[0] = np.nan
    with pytest.raises(ValueError, match="incomplete class probabilities"):
        stronger.choose_fusion(labels, classes, folds, neural, hybrid)


def test_fusion_rejects_swapped_pixels_with_identical_labels(tmp_path, monkeypatch):
    archive = tmp_path / "train.npz"
    frame = pd.DataFrame({"label": ["oval", "oval"], "identity_group": [1, 2]})
    pixels = np.array([[[[0, 0, 0]]], [[[255, 0, 0]]]], dtype=np.uint8)
    np.savez(archive, label=frame["label"].to_numpy(dtype=str), group=frame["identity_group"], pixels=pixels)
    monkeypatch.setattr(stronger, "face_assets", lambda *args: pixels[::-1])

    with pytest.raises(ValueError, match="row order differ"):
        stronger.check_alignment(archive, frame, [], Namespace(cache=None, workers=1))


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
            workflow.write_json(Path(command[-1]) / "dataset-metadata.json", {"info": {
                "isPrivate": True, "ownerUser": staged.owner, "datasetSlug": "face-shape-train",
            }})
        return result

    monkeypatch.setattr(workflow, "cli", fake_cli)
    workflow.submit(staged)
    assert scored == [(staged.workspace, staged.workspace / "outputs")]
    assert list(responses) == []


@pytest.mark.parametrize("info", [
    None,
    {"isPrivate": False, "ownerUser": "private-owner", "datasetSlug": "private-data"},
    {"isPrivate": True, "ownerUser": "another-owner", "datasetSlug": "private-data"},
    {"isPrivate": True, "ownerUser": "private-owner", "datasetSlug": "private-data", "collaborators": ["other"]},
])
def test_remote_dataset_requires_confirmed_unshared_privacy(info):
    def fake_cli(*command):
        workflow.write_json(Path(command[-1]) / "dataset-metadata.json", {"info": info})

    with pytest.raises(ValueError, match="unshared private dataset"):
        workflow.check_remote_private(fake_cli, "private-owner/private-data")


def test_grouped_folds_keep_identity_together():
    labels = np.repeat([0, 1], 10)
    groups = np.repeat(np.arange(10), 2)
    _, folds = grouped_folds(labels, groups)
    assert len(folds) == 5
    for train, valid in folds:
        assert set(groups[train]).isdisjoint(groups[valid])
        assert set(labels[train]) == set(labels[valid]) == {0, 1}


@pytest.mark.parametrize("launcher", ["colab_kernel_launcher.py", "ipykernel_launcher.py"])
def test_kernel_editor_uses_kaggle_defaults(monkeypatch, launcher):
    archive = Path("/kaggle/input/private-dataset/train.npz")
    monkeypatch.setattr(sys, "argv", [launcher, "-f", "/runtime/kernel-1.json"])
    monkeypatch.setattr(Path, "rglob", lambda root, pattern: [archive] if root == Path("/kaggle/input") else [])
    calls = []
    monkeypatch.setattr(kernel, "train", lambda *args, **kwargs: calls.append((args, kwargs)))

    kernel.main()
    assert calls == [((archive, Path("/kaggle/working")), {"smoke": False})]


@pytest.mark.parametrize("launcher", ["kaggle_kernel.py", "colab_kernel_launcher.py"])
def test_kernel_rejects_unrecognized_cli_option(monkeypatch, capsys, launcher):
    monkeypatch.setattr(sys, "argv", [launcher, "--smkoe"])
    with pytest.raises(SystemExit, match="2"):
        kernel.main()
    assert "unrecognized arguments: --smkoe" in capsys.readouterr().err


def test_editor_source_checksum_matches_script(monkeypatch):
    expected = kernel.digest(Path(kernel.__file__))
    monkeypatch.delattr(kernel, "__file__")
    assert kernel.source_digest() == expected


def test_kernel_logs_gpu_diagnostics_before_cuda_guard(monkeypatch, capsys, tmp_path):
    torch = SimpleNamespace(
        __version__="test+cpu", version=SimpleNamespace(cuda=None),
        cuda=SimpleNamespace(is_available=lambda: False), device=lambda kind: kind,
    )
    monkeypatch.setitem(sys.modules, "torch", torch)

    def nvidia_smi(command, **kwargs):
        assert command[0] == "nvidia-smi"
        return SimpleNamespace(returncode=0, stdout="Tesla T4, 555.01\n", stderr="")

    monkeypatch.setattr(kernel.subprocess, "run", nvidia_smi)
    with pytest.raises(RuntimeError, match="A Kaggle CUDA GPU must be enabled"):
        kernel.train(tmp_path / "missing.npz", tmp_path)
    output = capsys.readouterr().out
    assert "torch.__version__=test+cpu; torch.version.cuda=None; torch.cuda.is_available()=False" in output
    assert "nvidia-smi exit 0: Tesla T4, 555.01" in output
