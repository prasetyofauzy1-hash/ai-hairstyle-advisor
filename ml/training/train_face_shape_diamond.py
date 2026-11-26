"""Compare weighted retraining on niten19 + FFHQ AI labels against the prior combined model."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import ParameterGrid, StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ml.classification.learned import DEFAULT_MODEL
from ml.datasets.ffhq_features import SUBSET
from ml.evaluation.facebench import _read, benchmark_inputs, screened_frame
from ml.feature_extraction.geometry import FEATURE_NAMES
from ml.training.train_face_shape import SEED, holdout_split, load, metrics
from ml.training.train_face_shape_ffhq import ARTIFACT, load_ffhq, predict_bundle


def cv_folds(y: np.ndarray, groups: np.ndarray) -> tuple[int, list]:
    labels = set(y)
    n_folds = min(5, min(len(set(groups[y == label])) for label in labels))
    if n_folds < 2:
        raise ValueError("Each shape needs at least two independent training groups")
    for seed in range(SEED, SEED + 100):
        folds = list(StratifiedGroupKFold(n_folds, shuffle=True, random_state=seed).split(y, y, groups))
        if all(set(y[tr]) == set(y[va]) == labels for tr, va in folds):
            return seed, folds
    raise ValueError("No group-disjoint folds with all training shapes represented")


def weights(y: np.ndarray, ffhq: np.ndarray, label_power: float, ffhq_weight: float) -> np.ndarray:
    _, inverse, counts = np.unique(y, return_inverse=True, return_counts=True)
    w = counts[inverse].astype(float) ** -label_power * np.where(ffhq, ffhq_weight, 1.0)
    return w / w.mean()


def fit_candidate(X, y, ffhq, family, params, label_power, ffhq_weight):
    if family == "mlp":
        estimator = MLPClassifier(max_iter=1000, early_stopping=False, random_state=SEED, **params)
    else:
        estimator = SVC(random_state=SEED, **params)
    model = make_pipeline(StandardScaler(), estimator)
    model.fit(X, y, **{f"{model.steps[-1][0]}__sample_weight": weights(y, ffhq, label_power, ffhq_weight)})
    return model


def select(train: pd.DataFrame) -> tuple[dict, dict]:
    y = train["label"].to_numpy()
    groups = train["identity_group"].to_numpy()
    ffhq = train["source_split"].eq("train").to_numpy() & train["ffhq_id"].notna().to_numpy()
    X = train[FEATURE_NAMES].to_numpy(float)
    seed, folds = cv_folds(y, groups)
    labels = sorted(set(y))
    configurations = {
        "mlp": {"hidden_layer_sizes": [(32,), (64, 32)], "alpha": [0.001, 0.01]},
        "svm_rbf": {"C": [1, 3, 10], "gamma": ["scale"]},
    }
    trials = []
    for family, grid in configurations.items():
        for params in ParameterGrid(grid):
            for label_power, ffhq_weight in [(0.0, 1.0), (0.5, 1.0), (0.0, 3.0), (0.5, 3.0)]:
                scores = []
                for tr, va in folds:
                    model = fit_candidate(X[tr], y[tr], ffhq[tr], family, params, label_power, ffhq_weight)
                    scores.append(f1_score(y[va], model.predict(X[va]), labels=labels, average="macro", zero_division=0))
                trials.append({"family": family, "params": params, "label_power": label_power,
                               "ffhq_weight": ffhq_weight, "cv_macro_f1": float(np.mean(scores))})
    best = max(trials, key=lambda t: t["cv_macro_f1"])
    model = fit_candidate(X, y, ffhq, best["family"], best["params"], best["label_power"], best["ffhq_weight"])
    return {"model": model, "label_encoder": None, "feature_names": FEATURE_NAMES}, {
        "cv_folds": len(folds), "cv_seed": seed, "selected": best, "trials": trials,
    }


def estimate(attributes: Path, ffhq_id: str) -> str:
    path = attributes / "json" / f"{int(ffhq_id):05d}.json"
    if not path.is_file():
        return "unknown"
    detections = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(detections, list) or len(detections) != 1:
        return "unknown"
    value = detections[0].get("faceAttributes", {}).get("gender")
    return value if value in {"male", "female"} else "unknown"


def cohort_estimates(attributes: Path, photos: pd.Series, name: str) -> pd.Series:
    estimates = photos.map(lambda photo: estimate(attributes, photo))
    if not estimates.isin({"male", "female"}).any():
        raise ValueError(f"No known gender estimates for {name} in --attributes {attributes}")
    return estimates


def cohort(frame: pd.DataFrame, gender: pd.Series) -> dict:
    return pd.crosstab(gender, frame["label"]).to_dict(orient="index")


def score_cohort(frame: pd.DataFrame, gender: pd.Series, models: dict) -> dict:
    y = frame["label"].to_numpy()
    prediction = {name: predict_bundle(model, frame[model["feature_names"]].to_numpy(float))
                  for name, model in models.items()}
    subsets = {"all": np.ones(len(frame), dtype=bool)}
    subsets.update({value: gender.eq(value).to_numpy() for value in sorted(set(gender))})
    return {
        subset: {
            "photos": int(mask.sum()),
            "models": {name: metrics(y[mask], pred[mask], sorted(set(y[mask]))) for name, pred in prediction.items()},
        }
        for subset, mask in subsets.items()
    }


def run(args: argparse.Namespace) -> dict:
    if not (args.attributes / "json").is_dir():
        raise ValueError(f"--attributes must point to a DCGM FFHQ attributes checkout with a json directory: {args.attributes}")
    kaggle = load(args.kaggle_features)
    tr, te = holdout_split(kaggle["label"].to_numpy(), kaggle["identity_group"].to_numpy())
    _, ffhq_train, uzy = load_ffhq(args.ffhq_features)
    ffhq_train = ffhq_train[~ffhq_train["identity_group"].isin(uzy["identity_group"])].copy()
    train_gender = cohort_estimates(args.attributes, ffhq_train["ffhq_id"], "FFHQ AI training")
    uzy_gender = cohort_estimates(args.attributes, uzy["ffhq_id"], "Uzy")
    ffhq_train["identity_group"] += kaggle["identity_group"].max() + 1
    fit = pd.concat([kaggle.iloc[tr], ffhq_train], ignore_index=True)
    model, selection = select(fit)

    inputs = benchmark_inputs(args.facebench_manifest, args.facebench_images, args.ffhq_images,
                              args.niten_embeddings, args.workers)
    facebench, _ = screened_frame(*inputs, _read(args.identity_review))
    facebench_gender = cohort_estimates(args.attributes, facebench["image"].map(lambda photo: Path(photo).stem), "FaceBench")
    models = {"prior": joblib.load(DEFAULT_MODEL.with_name(ARTIFACT)), "retrained": model}
    return {
        "selection": selection,
        "training": {"niten19": len(tr), "ffhq_ai": len(ffhq_train), "ffhq_ai_cohort": cohort(ffhq_train, train_gender)},
        "uzy": score_cohort(uzy, uzy_gender, models),
        "facebench": score_cohort(facebench, facebench_gender, models),
        "niten19_holdout_retrained": metrics(
            kaggle.iloc[te]["label"].to_numpy(), predict_bundle(model, kaggle.iloc[te][FEATURE_NAMES].to_numpy(float)),
            sorted(set(kaggle.iloc[te]["label"])),
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("kaggle-features", "ffhq-features", "ffhq-images", "niten-embeddings", "facebench-manifest",
                 "facebench-images", "attributes", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--identity-review", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("ml.pipeline").setLevel(logging.WARNING)
    report = run(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Scored {report['uzy']['all']['photos']} Uzy and {report['facebench']['all']['photos']} FaceBench photos")


if __name__ == "__main__":
    main()
