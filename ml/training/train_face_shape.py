"""Phase 3: train and compare face-shape classifiers on a Phase 2 feature table.

    python -m ml.training.train_face_shape data/processed/niten19/features.csv \
        --out research/results/phase3/kaggle_niten19 --model-out ml/models/artifacts

Only images that passed the quality gates are used. One identity-aware stratified fold (~20%) is held out as the
test set; models are tuned with identity-aware stratified 5-fold CV on the rest. Every number written comes from
this run; nothing is copied from elsewhere.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from ml.classification.rule_based import PROTOTYPES
from ml.feature_extraction.geometry import FEATURE_NAMES

SEED = 42
N_FOLDS = 5


class RulePrototypes(BaseEstimator, ClassifierMixin):
    """The Phase 2 rule baseline as an estimator. fit() does nothing unless fitted=True, in which case each class
    prototype becomes the per-class training mean/std of every feature (prototype calibration)."""

    def __init__(self, fitted: bool = False):
        self.fitted = fitted

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        if self.fitted:
            self.prototypes_ = {c: {f: (X[y == c, i].mean(), X[y == c, i].std() + 1e-9)
                                    for i, f in enumerate(FEATURE_NAMES)} for c in self.classes_}
        else:
            self.prototypes_ = PROTOTYPES
        return self

    def predict(self, X):
        names = list(self.prototypes_)
        scores = np.stack([
            -np.mean([((X[:, FEATURE_NAMES.index(f)] - mu) / tol) ** 2 for f, (mu, tol) in self.prototypes_[c].items()], axis=0)
            for c in names], axis=1)
        return np.array(names)[scores.argmax(axis=1)]


@dataclass
class Candidate:
    name: str
    estimator: object
    grid: dict
    encode: bool = False   # XGBoost and MLP early stopping need integer labels


def candidates() -> list[Candidate]:
    return [
        Candidate("majority_class", DummyClassifier(strategy="most_frequent"), {}),
        Candidate("rule_based_v0", RulePrototypes(fitted=False), {}),
        Candidate("rule_prototypes_fitted", RulePrototypes(fitted=True), {}),
        Candidate("random_forest", RandomForestClassifier(random_state=SEED, n_jobs=-1),
                  {"n_estimators": [300], "max_depth": [None, 8], "min_samples_leaf": [1, 5]}),
        Candidate("svm_rbf", make_pipeline(StandardScaler(), SVC(kernel="rbf", probability=True, random_state=SEED)),
                  {"svc__C": [0.3, 1, 3, 10], "svc__gamma": ["scale", 0.1]}),
        Candidate("xgboost", XGBClassifier(random_state=SEED, n_jobs=4, tree_method="hist", eval_metric="mlogloss"),
                  {"n_estimators": [200, 400], "max_depth": [3, 5], "learning_rate": [0.05], "subsample": [0.8]},
                  encode=True),
        Candidate("mlp", make_pipeline(StandardScaler(), MLPClassifier(max_iter=2000, early_stopping=True, random_state=SEED)),
                  {"mlpclassifier__hidden_layer_sizes": [(32,), (64, 32)], "mlpclassifier__alpha": [1e-3, 1e-2]},
                  encode=True),
    ]


def metrics(y_true, y_pred, labels) -> dict:
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    pc, rc, fc, sc = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": p, "macro_recall": r, "macro_f1": f1,
        "per_class": {c: {"precision": pc[i], "recall": rc[i], "f1": fc[i], "support": int(sc[i])} for i, c in enumerate(labels)},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["passed"].astype(str) == "True"].dropna(subset=FEATURE_NAMES)
    return df.reset_index(drop=True)


def holdout_split(y, groups) -> tuple[np.ndarray, np.ndarray]:
    """Of the seeded identity-aware folds, hold out the one whose class mix is closest to the whole set.

    Large identity groups make folds uneven; choosing by labels only (never by model scores) keeps the test set
    representative without peeking at results."""
    outer = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    labels, overall = np.unique(y, return_counts=True)
    overall = overall / overall.sum()

    def distance(test_idx):
        counts = np.array([(y[test_idx] == c).sum() for c in labels]) / len(test_idx)
        return float(np.abs(counts - overall).sum())

    return min(outer.split(np.zeros(len(y)), y, groups), key=lambda split: distance(split[1]))


def bootstrap_ci(y_true, y_pred, labels, n: int = 1000, seed: int = SEED) -> dict:
    """95% percentile bootstrap intervals over test images (ignores identity clustering, so they are optimistic)."""
    rng = np.random.default_rng(seed)
    acc, f1 = [], []
    for _ in range(n):
        i = rng.integers(0, len(y_true), len(y_true))
        acc.append(accuracy_score(y_true[i], y_pred[i]))
        f1.append(precision_recall_fscore_support(y_true[i], y_pred[i], labels=labels, average="macro", zero_division=0)[2])
    return {"accuracy_95ci": np.percentile(acc, [2.5, 97.5]).tolist(), "macro_f1_95ci": np.percentile(f1, [2.5, 97.5]).tolist()}


def run(features_csv: Path, out: Path, model_out: Path | None) -> dict:
    raw = pd.read_csv(features_csv)
    df = load(features_csv)
    X, y, groups = df[FEATURE_NAMES].to_numpy(float), df["label"].to_numpy(), df["identity_group"].to_numpy()
    labels = sorted(np.unique(y))
    train_idx, test_idx = holdout_split(y, groups)
    assert not set(groups[train_idx]) & set(groups[test_idx])
    inner = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    enc = LabelEncoder().fit(labels)

    results, fitted = {}, {}
    for cand in candidates():
        y_tr = enc.transform(y[train_idx]) if cand.encode else y[train_idx]
        search = GridSearchCV(clone(cand.estimator), cand.grid or {}, cv=inner, scoring="f1_macro", n_jobs=1, refit=True)
        search.fit(X[train_idx], y_tr, groups=groups[train_idx])
        best = search.best_index_
        cv = search.cv_results_
        pred = search.predict(X[test_idx])
        if cand.encode:
            pred = enc.inverse_transform(pred)
        results[cand.name] = {
            "best_params": search.best_params_,
            "cv_macro_f1_mean": float(cv["mean_test_score"][best]),
            "cv_macro_f1_std": float(cv["std_test_score"][best]),
            "test": {**metrics(y[test_idx], pred, labels), **bootstrap_ci(y[test_idx], pred, labels)},
        }
        fitted[cand.name] = (search.best_estimator_, cand.encode)

    # Only the ML models are exported: they have predict_proba and pickle without this script's classes.
    trainable = [n for n in results if n in ("random_forest", "svm_rbf", "xgboost", "mlp")]
    best_name = max(trainable, key=lambda n: results[n]["cv_macro_f1_mean"])

    quality = {
        "images_total": int(len(raw)),
        "passed_quality_gates": int((raw["passed"].astype(str) == "True").sum()),
        "rejection_reasons": raw.loc[raw["passed"].astype(str) != "True", "issues"].fillna("").astype(str)
        .str.split(";").explode().loc[lambda s: s != ""].value_counts().to_dict(),
        "used_for_training": int(len(df)),
        "class_counts": df["label"].value_counts().sort_index().to_dict(),
        "test_class_counts": df["label"].iloc[test_idx].value_counts().sort_index().to_dict(),
        "identity_groups_with_more_than_one_label": int((df.groupby("identity_group")["label"].nunique() > 1).sum()),
        "images_in_groups_with_more_than_one_label": int(df.groupby("identity_group")["label"].transform("nunique").gt(1).sum()),
        "identity_groups": int(df["identity_group"].nunique()),
        "largest_identity_group": int(df["identity_group"].value_counts().max()),
    }
    # How much identity overlap the dataset's own train/test split has (why we don't use it).
    src_tr = set(df.loc[df["source_split"] == "training_set", "identity_group"])
    src_te = df[df["source_split"] == "testing_set"]
    quality["source_test_images_sharing_identity_with_source_train"] = int(src_te["identity_group"].isin(src_tr).sum())
    quality["source_test_images"] = int(len(src_te))

    report = {
        "dataset": features_csv.parent.name,
        "features_csv_sha256": hashlib.sha256(features_csv.read_bytes()).hexdigest(),
        "features": FEATURE_NAMES,
        "labels": labels,
        "seed": SEED,
        "protocol": f"identity-aware stratified hold-out (1 of {N_FOLDS} folds) + {N_FOLDS}-fold identity-aware stratified CV on the rest",
        "train_size": int(len(train_idx)), "test_size": int(len(test_idx)),
        "data": quality,
        "models": results,
        "selected_model": best_name,
        "selection_rule": "highest CV macro-F1 among RF, SVM, XGBoost and MLP; the test set was not used for selection",
        "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__, "xgboost": xgboost.__version__,
                     "numpy": np.__version__},
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(report, indent=2, default=float))
    _plots(out, report, df, labels)
    (out / "README.md").write_text(_markdown(report))

    if model_out is not None:
        est, encoded = fitted[best_name]
        model_out.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model": est, "label_encoder": enc if encoded else None, "feature_names": FEATURE_NAMES, "labels": labels,
            "model_name": best_name, "trained_on": report["dataset"], "train_size": report["train_size"],
            "features_csv_sha256": report["features_csv_sha256"], "seed": SEED, "versions": report["versions"],
            "test_macro_f1": results[best_name]["test"]["macro_f1"], "calibrated": False,
        }, model_out / "face_shape_kaggle_niten19.joblib", compress=3)
    return report


def _plots(out: Path, report: dict, df: pd.DataFrame, labels: list[str]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(report["models"])
    fig, axes = plt.subplots(2, (len(names) + 1) // 2, figsize=(4 * ((len(names) + 1) // 2), 8))
    for ax, name in zip(axes.ravel(), names):
        cm = np.array(report["models"][name]["test"]["confusion_matrix"])
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(labels)), labels, rotation=45, fontsize=8)
        ax.set_yticks(range(len(labels)), labels, fontsize=8)
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7)
        ax.set_title(f"{name}\nmacro-F1 {report['models'][name]['test']['macro_f1']:.3f}", fontsize=9)
        ax.set_xlabel("predicted", fontsize=8)
        ax.set_ylabel("true", fontsize=8)
    for ax in axes.ravel()[len(names):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "confusion_matrices.png", dpi=110)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(FEATURE_NAMES), figsize=(3.2 * len(FEATURE_NAMES), 3.4))
    for ax, f in zip(axes, FEATURE_NAMES):
        ax.boxplot([df.loc[df["label"] == c, f] for c in labels], showfliers=False)
        ax.set_xticks(range(1, len(labels) + 1), labels, rotation=45, fontsize=8)
        ax.set_title(f, fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "feature_distributions.png", dpi=110)
    plt.close(fig)


def _markdown(r: dict) -> str:
    d = r["data"]
    lines = [
        f"# Phase 3 results: {r['dataset']}", "",
        "Generated by `python -m ml.training.train_face_shape`; every number below comes from that run (`metrics.json`).", "",
        f"- Images: {d['images_total']}; passed quality gates: {d['passed_quality_gates']}; used: {d['used_for_training']}",
        f"- Rejection reasons (an image can have several): {d['rejection_reasons']}",
        f"- Class counts used: {d['class_counts']}",
        f"- Identity groups (SFace, cosine >= 0.363): {d['identity_groups']}; largest group: {d['largest_identity_group']} images",
        f"- Source test images sharing an identity group with source train: "
        f"{d['source_test_images_sharing_identity_with_source_train']} of {d['source_test_images']}",
        f"- Protocol: {r['protocol']}; seed {r['seed']}; train {r['train_size']}, test {r['test_size']}", "",
        f"- Test class counts: {d['test_class_counts']}",
        f"- Identity groups holding more than one face-shape label: {d['identity_groups_with_more_than_one_label']} "
        f"({d['images_in_groups_with_more_than_one_label']} images); a sign of label noise or of grouping errors", "",
        "| Model | CV macro-F1 (mean ± std) | Test accuracy | Test macro-P | Test macro-R | Test macro-F1 (95% CI) |",
        "|---|---|---|---|---|---|",
    ]
    for name, m in r["models"].items():
        t = m["test"]
        lines.append(f"| {name} | {m['cv_macro_f1_mean']:.3f} ± {m['cv_macro_f1_std']:.3f} | {t['accuracy']:.3f} | "
                     f"{t['macro_precision']:.3f} | {t['macro_recall']:.3f} | {t['macro_f1']:.3f} "
                     f"({t['macro_f1_95ci'][0]:.3f}-{t['macro_f1_95ci'][1]:.3f}) |")
    lines += ["", f"Selected model: **{r['selected_model']}** ({r['selection_rule']}).", "",
              "Bootstrap intervals resample test images, not people, so they understate uncertainty. "
              "Probabilities from these models are not calibrated.", "",
              "![Confusion matrices](confusion_matrices.png)", "", "![Feature distributions](feature_distributions.png)", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("features_csv", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model-out", type=Path, default=None)
    args = parser.parse_args()
    r = run(args.features_csv, args.out, args.model_out)
    for name, m in r["models"].items():
        print(f"{name:24s} cv_f1={m['cv_macro_f1_mean']:.3f}±{m['cv_macro_f1_std']:.3f} test_f1={m['test']['macro_f1']:.3f}")
    print("selected:", r["selected_model"])


if __name__ == "__main__":
    main()
