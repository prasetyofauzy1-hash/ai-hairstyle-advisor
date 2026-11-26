"""Prepare private Kaggle datasets, submit the training-only script, and score outputs locally."""

from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from ml.datasets.face_crops import face_assets
from ml.training.kaggle_kernel import digest, network, predict
from ml.training.train_face_shape_image import (
    add_cohort_arguments,
    benchmark_cohorts,
    benchmark_gender,
    paths,
    subset_metrics,
    training_cohorts,
)

COHORTS = ("train", "niten19_holdout", "uzy", "facebench", "roboflow_holdout")
ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "research/results/phase3"
OWNER = re.compile(r"[a-z0-9][a-z0-9_-]{1,39}")
SLUG = re.compile(r"[a-z0-9][a-z0-9-]{1,32}")
DATASET_PENDING = {"not_yet_persisted", "blobs_received", "blobs_decompressed", "blobs_copied_to_sds",
                   "individual_blobs_compressed", "reprocessing"}


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def stage_metadata(workspace: Path, owner: str, prefix: str) -> None:
    if not OWNER.fullmatch(owner) or not SLUG.fullmatch(prefix):
        raise ValueError("Use a lowercase Kaggle owner and URL-safe dataset prefix")
    for key in COHORTS:
        directory = workspace / key
        directory.mkdir()
        slug = f"{prefix}-{key.replace('_', '-')}"
        write_json(directory / "dataset-metadata.json", {
            "id": f"{owner}/{slug}", "title": slug.replace("-", " ").title(), "isPrivate": True,
            "licenses": [{"name": "other"}],
            "description": "Private non-commercial research only. No new license granted. Original image, "
                           "annotation, attribution and redistribution terms still apply.",
        })
    kernel = workspace / "kernel"
    kernel.mkdir()
    slug = f"{prefix}-fit-{secrets.token_hex(4)}"
    metadata = read_json(RESULTS / "kaggle_finetune/kernel-metadata.json")
    metadata.update(id=f"{owner}/{slug}", title=slug.replace("-", " ").title(),
                    dataset_sources=[f"{owner}/{prefix}-train"])
    shutil.copyfile(Path(__file__).with_name("kaggle_kernel.py"), kernel / metadata["code_file"])
    write_json(kernel / "kernel-metadata.json", metadata)


def validate(workspace: Path) -> tuple[dict, dict]:
    ledger = read_json(workspace / "local-ledger.json")
    if not OWNER.fullmatch(ledger["owner"]) or set(ledger["archives"]) != set(COHORTS):
        raise ValueError("Invalid owner or missing cohort in package")
    owner = ledger["owner"]
    ids = {}
    for key in COHORTS:
        directory = workspace / key
        meta = read_json(directory / "dataset-metadata.json")
        namespace, _, slug = meta["id"].partition("/")
        if (namespace != owner or not re.fullmatch(r"[a-z0-9-]+", slug) or meta["isPrivate"] is not True
                or meta["licenses"] != [{"name": "other"}] or meta["id"] in ids.values()):
            raise ValueError(f"Invalid private dataset metadata for {key}")
        ids[key] = meta["id"]
        archive = directory / f"{key}.npz"
        files = set(directory.iterdir())
        if files != {archive, directory / "dataset-metadata.json"} or any(p.is_symlink() for p in files):
            raise ValueError(f"Unexpected upload files in {key}")
        if digest(archive) != ledger["archives"][key]["sha256"]:
            raise ValueError(f"Archive checksum mismatch for {key}")
    kernel = workspace / "kernel"
    metadata = read_json(kernel / "kernel-metadata.json")
    kernel_owner, _, slug = metadata["id"].partition("/")
    if (kernel_owner != owner or not re.fullmatch(r"[a-z0-9-]+", slug)
            or metadata["is_private"] is not True or metadata["enable_gpu"] is not True
            or metadata["dataset_sources"] != [ids["train"]]
            or any(metadata[key] for key in ("competition_sources", "kernel_sources", "model_sources"))
            or metadata["code_file"] != "kaggle_kernel.py"):
        raise ValueError("Kernel must be private, GPU-enabled, and attached only to the training dataset")
    code = kernel / metadata["code_file"]
    files = set(kernel.iterdir())
    if (files != {code, kernel / "kernel-metadata.json"} or any(p.is_symlink() for p in files)
            or digest(code) != ledger["kernel_sha256"]):
        raise ValueError("Unexpected or modified files in kernel upload directory")
    return ledger, {"datasets": ids, "kernel": metadata["id"]}


def package(args: argparse.Namespace) -> None:
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if not (args.attributes / "json").is_dir():
        raise ValueError("--attributes must contain the DCGM FFHQ attributes json directory")
    training, holdouts = training_cohorts(args)
    cohorts = benchmark_cohorts(args, holdouts, allow_download=False)
    args.workspace.mkdir(mode=0o700)
    stage_metadata(args.workspace, args.owner, args.prefix)
    ledger = {"owner": args.owner, "archives": {}, "gender": {},
              "kernel_sha256": digest(args.workspace / "kernel/kaggle_kernel.py")}
    train = pd.concat([frame for frame, _ in training], ignore_index=True)
    train_paths = [photo for frame, root in training for photo in source_paths(frame, root)]
    train_hashes = {digest(photo) for photo in train_paths}
    entries = {"train": (train, train_paths), **{
        key: (frame, source_paths(frame, root)) for key, (frame, root, _) in cohorts.items()
    }}
    for key, (frame, photos) in entries.items():
        data = {"label": frame["label"].to_numpy(dtype=str)}
        if key == "train":
            data["group"] = train["identity_group"].to_numpy(dtype=np.int64)
        else:
            if any(digest(photo) in train_hashes for photo in photos):
                raise ValueError(f"Exact training-photo overlap with {key}")
            gender = benchmark_gender(args, key, photos, cohorts[key][2])
            ledger["gender"][key] = gender.tolist()
        data["pixels"] = face_assets(photos, args.cache, args.workers)
        archive = args.workspace / key / f"{key}.npz"
        np.savez_compressed(archive, **data)
        ledger["archives"][key] = {"sha256": digest(archive), "rows": len(frame)}
        print(f"Packaged {key}: {len(frame)} photos", flush=True)
    write_json(args.workspace / "local-ledger.json", ledger)
    validate(args.workspace)
    print("Validated all private metadata, file checksums, and the training-only kernel mount.")


def source_paths(frame: pd.DataFrame, root: Path) -> list[Path]:
    root = root.resolve(strict=True)
    photos = paths(frame, root)
    if any(not p.resolve(strict=True).is_relative_to(root) or not p.is_file() for p in photos):
        raise ValueError("Photo path must be a local file under its staged source root")
    return photos


def cli(environment: dict[str, str], *command: str) -> str:
    adjacent = Path(sys.executable).with_name("kaggle")
    executable = str(adjacent) if adjacent.is_file() else shutil.which("kaggle")
    if not executable:
        raise RuntimeError("Install the optional Kaggle CLI as described in the runbook")
    result = subprocess.run([executable, *command], env=environment, check=False,
                            capture_output=True, text=True, timeout=1800)
    if result.returncode:
        raise RuntimeError(f"Kaggle {command[0]} {command[1]} failed (exit {result.returncode}); "
                           "inspect the account or CLI privately")
    return result.stdout.strip()


def credentials(owner: str) -> dict[str, str]:
    actual, key = os.environ.get("KAGGLE_USERNAME"), os.environ.get("KAGGLE_KEY")
    if bool(actual) != bool(key):
        raise ValueError("Set both KAGGLE_USERNAME and KAGGLE_KEY or neither")
    if not actual:
        config = Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle")) / "kaggle.json"
        if not config.is_file():
            raise ValueError("Set KAGGLE_USERNAME and KAGGLE_KEY or configure kaggle.json")
        try:
            settings = read_json(config)
            actual, key = settings.get("username"), settings.get("key")
        except (AttributeError, ValueError):
            raise ValueError("Invalid Kaggle credentials file") from None
    if actual != owner:
        raise ValueError("Kaggle credentials do not match the prepared dataset owner")
    if not isinstance(key, str) or not key:
        raise ValueError("Missing Kaggle API key")
    return {"KAGGLE_USERNAME": owner, "KAGGLE_KEY": key}


def check_remote_private(kaggle, dataset: str) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        kaggle("datasets", "metadata", dataset, "-p", tmp)
        meta = read_json(Path(tmp) / "dataset-metadata.json").get("info")
        owner, slug = dataset.split("/")
        if (not isinstance(meta, dict) or meta.get("isPrivate") is not True or meta.get("ownerUser") != owner
                or meta.get("datasetSlug") != slug or meta.get("collaborators")):
            raise ValueError("Kaggle did not confirm an unshared private dataset in the configured account")


def until_ready(status, success: set[str], pending: set[str], deadline: float) -> None:
    while True:
        value = status()
        if value in success:
            return
        if value not in pending:
            raise RuntimeError("Kaggle did not report a successful terminal status; check the private job")
        if time.monotonic() >= deadline:
            raise TimeoutError("Kaggle wait expired; check the private job before attempting another run")
        time.sleep(30)


def submit(args: argparse.Namespace) -> None:
    ledger, ids = validate(args.workspace)
    if not args.confirm_upload_rights:
        raise ValueError("Review all applicable source and Kaggle terms before passing --confirm-upload-rights")
    if args.wait_minutes < 1:
        raise ValueError("--wait-minutes must be positive")
    login = credentials(ledger["owner"])
    with tempfile.TemporaryDirectory() as config:
        # Use only the checked account; an unrelated access token takes precedence in Kaggle's CLI.
        environment = {key: value for key, value in os.environ.items() if not key.startswith("KAGGLE_")}
        environment.update(login, KAGGLE_CONFIG_DIR=config, HOME=config)
        submit_private(args, ids, lambda *command: cli(environment, *command))


def dataset_status(kaggle, ref: str) -> dict:
    return json.loads(kaggle("datasets", "status", ref, "--format", "json"))


def submit_private(args: argparse.Namespace, ids: dict, kaggle) -> None:
    out = args.workspace / "outputs"
    out.mkdir()
    deadline = time.monotonic() + args.wait_minutes * 60
    for key in COHORTS if args.upload_holdouts else ("train",):
        ref = ids["datasets"][key]
        path = str(args.workspace / key)
        if args.update_datasets:
            check_remote_private(kaggle, ref)
            version_num = dataset_status(kaggle, ref)["current_version_number"] + 1
            message = kaggle("datasets", "version", "-p", path, "-m", "Private research update", "-q", "-t")
            if "dataset version is being created" not in message.lower():
                raise RuntimeError("Kaggle did not confirm the private dataset version")
        else:
            version_num = 1
            message = kaggle("datasets", "create", "-p", path, "-q", "-t")
            if "private dataset is being created" not in message.lower():
                raise RuntimeError("Kaggle did not confirm creation of a private dataset")
        def version_status():
            status = dataset_status(kaggle, ref)
            if status["status"] == "ready" and status["current_version_number"] < version_num:
                return "previous_version"
            return status["status"]

        until_ready(version_status, {"ready"}, DATASET_PENDING | {"previous_version"}, deadline)
        check_remote_private(kaggle, ref)

    message = kaggle("kernels", "push", "-p", str(args.workspace / "kernel"), "--accelerator", "NvidiaTeslaT4")
    version = re.search(r"Kernel version (\d+) successfully pushed", message)
    if not version or "not valid" in message:
        raise RuntimeError("Kaggle did not confirm a versioned kernel push")
    ref = f"{ids['kernel']}/{version[1]}"
    print(f"Submitted private kernel {ref}; waiting for completion.", flush=True)

    def kernel_status():
        message = kaggle("kernels", "status", ref)
        match = re.search(r'has status "([a-zA-Z_.]+)"', message)
        return match[1].lower().split(".")[-1] if match else "unknown"

    until_ready(kernel_status, {"complete"}, {"new_script", "queued", "running"}, deadline)
    kaggle("kernels", "output", ref, "-p", str(out), "-q", "--file-pattern", r"^(resnet18\.pt|selection\.json)$")
    score(args.workspace, out)


def score(workspace: Path, outputs: Path) -> None:
    import torch

    ledger, _ = validate(workspace)
    selection = read_json(outputs / "selection.json")
    if (selection["archive_sha256"] != ledger["archives"]["train"]["sha256"]
            or selection["kernel_sha256"] != ledger["kernel_sha256"]
            or selection["checkpoint_sha256"] != digest(outputs / "resnet18.pt")
            or selection["smoke"] or selection["device"] != "cuda"):
        raise ValueError("Refusing to score smoke, stale, or mismatched Kaggle outputs")
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    checkpoint = torch.load(outputs / "resnet18.pt", weights_only=True, map_location="cpu")
    classes = np.asarray(checkpoint["classes"])
    model = network(len(classes), pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    metrics = {}
    for key in COHORTS[1:]:
        with np.load(workspace / key / f"{key}.npz", allow_pickle=False) as data:
            pred = classes[predict(model, data["pixels"], "cpu")]
            frame = pd.DataFrame({"label": data["label"]})
        gender = np.asarray(ledger["gender"][key])
        metrics[key] = subset_metrics(frame, gender, {"kaggle_finetune": pred})
        print(f"Scored {key} locally: {len(frame)} photos", flush=True)
    write_json(outputs / "metrics.json", {"selection": selection, "benchmarks": metrics})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for action in ("pack", "validate", "submit", "score"):
        command = sub.add_parser(action)
        command.add_argument("workspace", type=Path)
        if action == "pack":
            add_cohort_arguments(command)
            command.add_argument("--owner", required=True)
            command.add_argument("--prefix", default="face-shape")
        elif action == "submit":
            command.add_argument("--confirm-upload-rights", action="store_true")
            command.add_argument("--upload-holdouts", action="store_true")
            command.add_argument("--update-datasets", action="store_true")
            command.add_argument("--wait-minutes", type=int, default=480)
        elif action == "score":
            command.add_argument("outputs", type=Path)
    args = parser.parse_args()
    if args.action == "pack":
        with threadpool_limits(1, user_api="blas"):
            package(args)
    elif args.action == "validate":
        validate(args.workspace)
        print("Validated all private metadata, file checksums, and the training-only kernel mount.")
    elif args.action == "submit":
        submit(args)
    else:
        score(args.workspace, args.outputs)


if __name__ == "__main__":
    main()
