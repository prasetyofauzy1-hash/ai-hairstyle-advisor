"""Compare frozen pretrained image backbones with landmark classifiers on four fixed benchmarks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

from ml.classification.learned import DEFAULT_MODEL
from ml.datasets.face_crops import face_assets
from ml.datasets.roboflow_men import load_split
from ml.evaluation.facebench import _read, benchmark_inputs, screened_frame
from ml.evaluation.gender_estimates import estimates
from ml.feature_extraction.geometry import FEATURE_NAMES
from ml.training.train_face_shape import SEED, holdout_split, load, metrics
from ml.training.train_face_shape_diamond import cohort_estimates, cv_folds, weights
from ml.training.train_face_shape_ffhq import ARTIFACT, load_ffhq, phase2_rules, predict_bundle

BACKBONES = ("mobilenet_v3_small", "resnet18")
SFACE = "sface"


def embeddings(name: str, pixels: np.ndarray) -> np.ndarray:
    import torch
    from torchvision import models

    if name == "mobilenet_v3_small":
        backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        model = torch.nn.Sequential(backbone.features, backbone.avgpool, torch.nn.Flatten())
    elif name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        model.fc = torch.nn.Identity()
    else:
        raise ValueError(f"Unknown image backbone: {name}")
    model.eval()
    mean = torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]
    with torch.inference_mode():
        features = [model((torch.from_numpy(batch).permute(0, 3, 1, 2).float() / 255 - mean) / std).numpy()
                    for batch in np.array_split(pixels, (len(pixels) + 31) // 32)]
    return normalize(np.concatenate(features))


def full_features(image: np.ndarray, frame: pd.DataFrame, scaler: StandardScaler | None) -> np.ndarray:
    if scaler is None:
        return image
    geo = scaler.transform(frame[FEATURE_NAMES].to_numpy(float))
    return np.column_stack([image, normalize(geo)])


def fit(configuration: dict, image: np.ndarray, frame: pd.DataFrame):
    y = frame["label"].to_numpy()
    ffhq = frame["source_split"].eq("train").to_numpy() & frame["ffhq_id"].notna().to_numpy()
    sample_weight = weights(y, ffhq, configuration["power"], 1.0)
    if configuration["backbone"] == "mobile_tail":
        from ml.training.mobile_tail import train_tail

        return train_tail(image, y, sample_weight, configuration["lr"]), None
    scaler = StandardScaler().fit(frame[FEATURE_NAMES].to_numpy(float)) if configuration["geometry"] else None
    model = SVC(C=configuration["C"], gamma="scale", random_state=SEED)
    model.fit(full_features(image, frame, scaler), y, sample_weight=sample_weight)
    return model, scaler


def select(train: pd.DataFrame, features: dict[str, np.ndarray], *, foundation: bool = False) -> tuple[dict, dict]:
    y = train["label"].to_numpy()
    seed, folds = cv_folds(y, train["identity_group"].to_numpy())
    trials = []
    for name, image in features.items():
        if foundation and name == SFACE:
            continue
        if name == "mobile_tail":
            configs = [{"backbone": name, "geometry": False, "lr": lr, "power": 0.5} for lr in (0.0003, 0.001)]
        else:
            configs = [{"backbone": name, "geometry": geometry, "C": c, "power": power}
                       for geometry in (False, True)
                       for c in ((1.0, 3.0) if foundation else (1.0, 10.0))
                       for power in ((0.5,) if foundation else (0.0, 0.5))]
        for config in configs:
            scores = []
            for tr, va in folds:
                model, scaler = fit(config, image[tr], train.iloc[tr])
                pred = model.predict(full_features(image[va], train.iloc[va], scaler))
                scores.append(f1_score(y[va], pred, average="macro", zero_division=0))
            trials.append({**config, "cv_macro_f1": float(np.mean(scores))})
    if foundation:
        families = {
            "foundation_image": [t for t in trials if "_" not in t["backbone"] and not t["geometry"]],
            "foundation_hybrid": [t for t in trials if "_" not in t["backbone"] and t["geometry"]],
            "foundation_ensemble": [t for t in trials if t["backbone"].endswith("_sface")],
            "mobile_partial": [t for t in trials if t["backbone"] == "mobile_tail"],
        }
    else:
        families = {"direct_image": [t for t in trials if not t["geometry"]],
                    "image_plus_landmarks": [t for t in trials if t["geometry"]]}
    selected = {family: max(options, key=lambda trial: trial["cv_macro_f1"]) for family, options in families.items()}
    if foundation:
        selected["pr14_hybrid"] = {"backbone": SFACE, "geometry": True, "C": 1.0, "power": 0.5}
    return selected, {"cv_seed": seed, "cv_folds": len(folds), "selected": selected, "trials": trials}


def subset_metrics(frame: pd.DataFrame, gender: np.ndarray, predictions: dict[str, np.ndarray]) -> dict:
    if not np.isin(gender, ["male", "female"]).any():
        raise ValueError("No known gender estimates in benchmark")
    y = frame["label"].to_numpy()
    masks = {"all": np.ones(len(frame), dtype=bool), **{value: gender == value for value in sorted(set(gender))}}
    return {name: {"photos": int(mask.sum()), "models": {
        model: metrics(y[mask], pred[mask], sorted(set(y[mask]))) for model, pred in predictions.items()
    }} for name, mask in masks.items()}


def paths(frame: pd.DataFrame, root: Path) -> list[Path]:
    return [root / image for image in frame["image"]]


def baseline_predictions(benchmark: str, frame: pd.DataFrame, baselines: dict) -> dict[str, np.ndarray]:
    predictions = {"phase2_rules": phase2_rules(frame)}
    for name, bundle in baselines.items():
        if benchmark == "niten19_holdout" and name == "app_landmarks":
            continue
        predictions[name] = predict_bundle(bundle, frame[bundle["feature_names"]].to_numpy(float))
    return predictions


def encode(names, photos: list[Path], cache: Path, workers: int) -> dict[str, np.ndarray]:
    vision = sorted(set(names) - {SFACE})
    features = {}
    if vision:
        pixels = face_assets(photos, cache, workers)
        features.update({name: embeddings(name, pixels) for name in vision})
    if SFACE in names:
        features[SFACE] = face_assets(photos, cache, workers, sface=True)
    return features


def training_cohorts(args: argparse.Namespace):
    kaggle = load(args.kaggle_features)
    tr, te = holdout_split(kaggle["label"].to_numpy(), kaggle["identity_group"].to_numpy())
    _, ffhq, uzy = load_ffhq(args.ffhq_features)
    ffhq = ffhq.loc[~ffhq["identity_group"].isin(uzy["identity_group"])].copy()
    ffhq["identity_group"] += kaggle["identity_group"].max() + 1
    rf_root = args.roboflow_data / "images"
    rf_train, rf_test = load_split(args.roboflow_data / "manifest.csv", args.roboflow_features, args.roboflow_split)
    rf_train["identity_group"] = ffhq["identity_group"].max() + 1 + np.arange(len(rf_train))
    training = [(kaggle.iloc[tr], args.niten_images), (ffhq, args.ffhq_images), (rf_train, rf_root)]
    return training, (kaggle.iloc[te], uzy, rf_test)


def benchmark_cohorts(args: argparse.Namespace, holdouts, *, allow_download: bool = True) -> dict:
    niten19, uzy, rf_test = holdouts
    facebench, _ = screened_frame(
        *benchmark_inputs(args.facebench_manifest, args.facebench_images, args.ffhq_images,
                          args.niten_embeddings, args.workers, allow_download=allow_download),
        _read(args.identity_review),
    )
    return {
        "niten19_holdout": (niten19, args.niten_images, None),
        "uzy": (uzy, args.ffhq_images, uzy["ffhq_id"]),
        "facebench": (facebench, args.facebench_images, facebench["image"].map(lambda photo: Path(photo).stem)),
        "roboflow_holdout": (rf_test, args.roboflow_data / "images", None),
    }


def benchmark_gender(args: argparse.Namespace, key: str, photos: list[Path], ffhq_ids) -> np.ndarray:
    if ffhq_ids is None:
        return np.asarray(estimates(photos, args.gender_model, args.workers))
    return cohort_estimates(args.attributes, ffhq_ids, key).to_numpy()


def run(args: argparse.Namespace) -> dict:
    import torch

    torch.set_num_threads(args.workers)
    if not (args.attributes / "json").is_dir():
        raise ValueError("--attributes must contain the DCGM FFHQ attributes json directory")
    training, holdouts = training_cohorts(args)
    train = pd.concat([frame for frame, _ in training], ignore_index=True)
    encoder, names = encode, (*BACKBONES, SFACE)
    if args.foundation:
        from ml.training.foundation_features import REPRESENTATIONS, encode_foundation

        encoder, names = encode_foundation, tuple(REPRESENTATIONS)
    features = encoder(names, [p for frame, root in training for p in paths(frame, root)], args.cache, args.workers)
    selection, selection_report = select(train, features, foundation=args.foundation)
    fitted = {key: fit(config, features[config["backbone"]], train) for key, config in selection.items()}
    print(f"Training-only selection: {selection}", flush=True)

    benchmarks = benchmark_cohorts(args, holdouts)
    baselines = {"app_landmarks": joblib.load(DEFAULT_MODEL),
                 "combined_landmarks": joblib.load(DEFAULT_MODEL.with_name(ARTIFACT))}
    report = {"training": dict(zip(("niten19", "ffhq_ai", "roboflow"), [len(f) for f, _ in training], strict=True)),
              "selection": selection_report, "benchmarks": {}}
    for key, (frame, root, ffhq_ids) in benchmarks.items():
        photos = paths(frame, root)
        gender = benchmark_gender(args, key, photos, ffhq_ids)
        predictions = baseline_predictions(key, frame, baselines)
        test_features = encoder(sorted({c["backbone"] for c in selection.values()}), photos, args.cache, args.workers)
        for model_name, (model, scaler) in fitted.items():
            predictions[model_name] = model.predict(
                full_features(test_features[selection[model_name]["backbone"]], frame, scaler),
            )
        report["benchmarks"][key] = subset_metrics(frame, gender, predictions)
        print(f"Scored {key}: {len(frame)} photos", flush=True)
    return report


def add_cohort_arguments(parser) -> None:
    for name in ("kaggle-features", "ffhq-features", "ffhq-images", "niten-embeddings", "facebench-manifest",
                 "facebench-images", "attributes", "identity-review", "roboflow-data", "roboflow-features",
                 "roboflow-split", "niten-images", "gender-model", "cache"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_cohort_arguments(parser)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--foundation", action="store_true", help="run the pinned CPU DINOv2/CLIP experiment")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    with threadpool_limits(1, user_api="blas"):
        report = run(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
