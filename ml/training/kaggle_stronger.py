"""Stage the stronger private Kaggle run and score its late fusion locally."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from ml.datasets.face_crops import face_assets
from ml.training import kaggle_finetune as workflow, kaggle_kernel as kernel
from ml.training import train_face_shape_image as image

HYBRID = {"backbone": image.SFACE, "geometry": True, "C": 1.0, "power": 0.5}
FUSION_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)


def stronger_source() -> bytes:
    source = Path(kernel.__file__).read_bytes()
    marker = b'DEFAULT_EXPERIMENT = "resnet18"'
    if source.count(marker) != 1:
        raise ValueError("Could not locate the kernel experiment default")
    return source.replace(marker, b'DEFAULT_EXPERIMENT = "stronger"')


def stage(previous: Path, workspace: Path) -> None:
    ledger, _ = workflow.validate(previous)
    workspace.mkdir(mode=0o700)
    for key in workflow.COHORTS:
        shutil.copytree(previous / key, workspace / key)
    target = workspace / "kernel"
    target.mkdir()
    source = target / "kaggle_kernel.py"
    source.write_bytes(stronger_source())
    meta = workflow.read_json(previous / "kernel/kernel-metadata.json")
    meta.pop("docker_image", None)
    meta.update(machine_shape="NvidiaTeslaT4", docker_image_pinning_type="latest")
    workflow.write_json(target / "kernel-metadata.json", meta)
    ledger["kernel_sha256"] = kernel.digest(source)
    workflow.write_json(workspace / "local-ledger.json", ledger)
    workflow.validate(workspace)
    print("Staged stronger kernel with one private training dataset and local-only holdouts.", flush=True)


def checked_probabilities(values, rows, classes):
    if (values.shape != (rows, classes) or not np.isfinite(values).all()
            or np.any(values < 0) or not np.allclose(values.sum(axis=1), 1)):
        raise ValueError("Malformed or incomplete class probabilities")
    return values


def hybrid_margins(fitted, frame, embeddings, classes):
    model, scaler = fitted
    if not np.array_equal(model.classes_, classes):
        raise ValueError("Hybrid and neural classes do not align")
    return model.decision_function(image.full_features(embeddings, frame, scaler))


def fit_hybrid(frame, embeddings, classes):
    y, groups = frame["label"].to_numpy(), frame["identity_group"].to_numpy()
    _, inner = kernel.grouped_folds(y, groups, maximum=3)
    margins = np.full((len(y), len(classes)), np.nan)
    for tr, va in inner:
        model = image.fit(HYBRID, embeddings[tr], frame.iloc[tr])
        margins[va] = hybrid_margins(model, frame.iloc[va], embeddings[va], classes)
    calibrator = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(margins, y)
    return image.fit(HYBRID, embeddings, frame), calibrator


def hybrid_probabilities(fitted, frame, embeddings, classes):
    model, calibrator = fitted
    if not np.array_equal(calibrator.classes_, classes):
        raise ValueError("Calibrator and neural classes do not align")
    return calibrator.predict_proba(hybrid_margins(model, frame, embeddings, classes))


def fold_scores(labels, predicted, folds):
    return [float(f1_score(labels[va], predicted[va], average="macro", zero_division=0)) for _, va in folds]


def choose_fusion(labels, classes, folds, neural, hybrid):
    neural = checked_probabilities(neural, len(labels), len(classes))
    hybrid = checked_probabilities(hybrid, len(labels), len(classes))
    trials = []
    for weight in FUSION_WEIGHTS:
        pred = classes[(weight * neural + (1 - weight) * hybrid).argmax(axis=1)]
        scores = fold_scores(labels, pred, folds)
        trials.append({"neural_weight": weight, "fold_macro_f1": scores, "cv_macro_f1": float(np.mean(scores))})
    best = max(trials, key=lambda trial: (trial["cv_macro_f1"], -abs(trial["neural_weight"] - 0.5)))
    return {"selected": best, "trials": trials}


def load_outputs(workspace, outputs):
    import torch

    ledger, _ = workflow.validate(workspace)
    if (workspace / "kernel/kaggle_kernel.py").read_bytes() != stronger_source():
        raise ValueError("Local kernel implementation does not match the staged stronger kernel")
    selection = workflow.read_json(outputs / "stronger-selection.json")
    checkpoint, oof = outputs / "stronger.pt", outputs / "training-oof.npz"
    if (selection["archive_sha256"] != ledger["archives"]["train"]["sha256"]
            or selection["kernel_sha256"] != ledger["kernel_sha256"]
            or selection["checkpoint_sha256"] != kernel.digest(checkpoint)
            or selection["oof_sha256"] != kernel.digest(oof) or selection["smoke"] or selection["device"] != "cuda"):
        raise ValueError("Refusing to score smoke, stale, or mismatched stronger Kaggle outputs")
    model = torch.load(checkpoint, weights_only=True, map_location="cpu")
    if selection["selected"]["backbone"] != model["backbone"]:
        raise ValueError("Checkpoint and selected backbone do not match")
    with np.load(oof, allow_pickle=False) as data:
        probabilities = data["probabilities"].copy()
    return ledger, selection, model, probabilities


def check_alignment(archive, frame, photos, args):
    labels = frame["label"].to_numpy()
    with np.load(archive, allow_pickle=False) as data:
        if (not np.array_equal(data["label"], labels)
                or ("group" in data.files and not np.array_equal(data["group"], frame["identity_group"]))
                or not np.array_equal(data["pixels"], face_assets(photos, args.cache, args.workers))):
            raise ValueError("Local photos, group assignments or row order differ from the validated archive")


def score(args):
    import torch

    torch.set_num_threads(args.workers)
    workspace, outputs = args.workspace, args.outputs
    ledger, selection, checkpoint, neural_oof = load_outputs(workspace, outputs)
    training, holdouts = image.training_cohorts(args)
    train = pd.concat([frame for frame, _ in training], ignore_index=True)
    photos = [p for frame, root in training for p in image.paths(frame, root)]
    check_alignment(workspace / "train/train.npz", train, photos, args)
    labels, groups = train["label"].to_numpy(), train["identity_group"].to_numpy()
    classes = np.asarray(checkpoint["classes"])
    seed, folds = kernel.grouped_folds(labels, groups)
    if (not np.array_equal(np.unique(labels), classes)
            or (seed, len(folds), len(labels)) != (selection["seed"], selection["folds"], selection["training_rows"])):
        raise ValueError("Local class order, folds or rows do not match Kaggle training")
    neural_oof = checked_probabilities(neural_oof, len(labels), len(classes))
    if not np.allclose(fold_scores(labels, classes[neural_oof.argmax(axis=1)], folds),
                       selection["selected"]["fold_macro_f1"]):
        raise ValueError("Downloaded OOF predictions do not reproduce the selected backbone's fold scores")
    device = torch.device("cpu")
    name = checkpoint["backbone"]
    model = kernel.strong_network(name, len(classes), config=checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])
    sface = face_assets(photos, args.cache, args.workers, sface=True)
    hybrid_oof = np.full_like(neural_oof, np.nan)
    for tr, va in folds:
        fitted = fit_hybrid(train.iloc[tr], sface[tr], classes)
        hybrid_oof[va] = hybrid_probabilities(fitted, train.iloc[va], sface[va], classes)
    fusion = choose_fusion(labels, classes, folds, neural_oof, hybrid_oof)
    weight = fusion["selected"]["neural_weight"]
    print(f"Training-only fusion neural weight: {weight}", flush=True)
    fitted = fit_hybrid(train, sface, classes)
    metrics = {}
    for key, (frame, root, _) in image.benchmark_cohorts(args, holdouts, allow_download=False).items():
        archive = workspace / key / f"{key}.npz"
        photos = image.paths(frame, root)
        check_alignment(archive, frame, photos, args)
        with np.load(archive, allow_pickle=False) as data:
            neural = kernel.strong_probabilities(model, data["pixels"], device, name)
        hybrid = hybrid_probabilities(fitted, frame, face_assets(photos, args.cache, args.workers, sface=True), classes)
        metrics[key] = image.subset_metrics(frame, np.asarray(ledger["gender"][key]), {
            "stronger_finetune": classes[neural.argmax(axis=1)],
            "late_fusion": classes[(weight * neural + (1 - weight) * hybrid).argmax(axis=1)],
        })
        print(f"Scored {key} locally: {len(frame)} photos", flush=True)
    workflow.write_json(outputs / "stronger-metrics.json", {
        "selection": selection, "fusion": fusion, "benchmarks": metrics,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    staging = actions.add_parser("stage")
    staging.add_argument("previous", type=Path)
    staging.add_argument("workspace", type=Path)
    scoring = actions.add_parser("score")
    scoring.add_argument("workspace", type=Path)
    scoring.add_argument("outputs", type=Path)
    image.add_cohort_arguments(scoring)
    args = parser.parse_args()
    if args.action == "stage":
        stage(args.previous, args.workspace)
    else:
        if args.workers < 1:
            parser.error("--workers must be at least 1")
        with threadpool_limits(1, user_api="blas"):
            score(args)


if __name__ == "__main__":
    main()
