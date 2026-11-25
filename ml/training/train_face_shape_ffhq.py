"""Phase 3b: retrain on Kaggle niten19 plus the FFHQ AI pseudo-labels and test on Uzy's human-labeled FFHQ photos.

    python -m ml.training.train_face_shape_ffhq data/processed/niten19/features.csv data/processed/ffhq/features.csv \
        --out research/results/phase3/kaggle_plus_ffhq_ai --model-out ml/models/artifacts \
        --current-model ml/models/artifacts/face_shape_kaggle_niten19.joblib

Both training configurations share the identity-aware Kaggle hold-out of train_face_shape, so the only difference
between them is the added FFHQ rows. Models are tuned and selected by identity-aware 5-fold CV on each training set;
neither test set is used for tuning or selection. The FFHQ test photos carry the only human labels and are never
trained on. Every number written comes from this run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from types import SimpleNamespace
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder

from ml.classification.rule_based import classify
from ml.feature_extraction.geometry import FEATURE_NAMES
from ml.training.train_face_shape import N_FOLDS, SEED, bootstrap_ci, candidates, holdout_split, load, metrics

TRAINABLE = ("random_forest", "svm_rbf", "xgboost", "mlp")
ARTIFACT = "face_shape_kaggle_ffhq_ai.joblib"
KAGGLE_ONLY, COMBINED = "kaggle_only", "kaggle_plus_ffhq_ai"


def load_ffhq(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(path, dtype={"ffhq_id": str})
    usable = raw[raw["passed"].astype(str) == "True"].dropna(subset=FEATURE_NAMES)
    train = usable[(usable["source_split"] == "train") & (usable["label_source"] == "ai_pseudo_label")]
    test = usable[(usable["source_split"] == "test") & (usable["label_source"] == "human")]
    if set(train["ffhq_id"]) & set(test["ffhq_id"]):
        raise ValueError("an FFHQ photo is in both train and test")
    return raw, train.reset_index(drop=True), test.reset_index(drop=True)


def predict_bundle(bundle: dict, X: np.ndarray) -> np.ndarray:
    pred = bundle["model"].predict(X)
    return bundle["label_encoder"].inverse_transform(pred) if bundle["label_encoder"] is not None else pred


def phase2_rules(df: pd.DataFrame) -> np.ndarray:
    """The Phase 2 classifier as the app runs it: forehead features are dropped when hair covers the forehead."""
    occluded = df["forehead_occluded"].astype(str) == "True" if "forehead_occluded" in df else pd.Series(False, df.index)
    return np.array([classify(SimpleNamespace(**{n: row[n] for n in FEATURE_NAMES}), forehead_occluded=occ).shape
                     for (_, row), occ in zip(df.iterrows(), occluded)])


def evaluate(y_true, y_pred, labels) -> dict:
    return {**metrics(y_true, y_pred, labels), **bootstrap_ci(y_true, y_pred, labels)}


def run(kaggle_csv: Path, ffhq_csv: Path, out: Path, model_out: Path | None = None,
        current_model: Path | None = None) -> dict:
    kag = load(kaggle_csv)
    Xk, yk, gk = kag[FEATURE_NAMES].to_numpy(float), kag["label"].to_numpy(), kag["identity_group"].to_numpy()
    k_tr, k_te = holdout_split(yk, gk)
    assert not set(gk[k_tr]) & set(gk[k_te])

    ffhq_raw, ff_tr, ff_te = load_ffhq(ffhq_csv)
    # A person in both FFHQ train and test would leak; drop such training rows.
    leak = ff_tr["identity_group"].isin(set(ff_te["identity_group"]))
    ff_tr = ff_tr[~leak].reset_index(drop=True)
    Xf_tr, yf_tr = ff_tr[FEATURE_NAMES].to_numpy(float), ff_tr["label"].to_numpy()
    gf_tr = ff_tr["identity_group"].to_numpy() + gk.max() + 1
    Xf_te, yf_te = ff_te[FEATURE_NAMES].to_numpy(float), ff_te["label"].to_numpy()
    kaggle_labels, human_labels = sorted(np.unique(yk)), sorted(np.unique(yf_te))

    train_sets = {
        KAGGLE_ONLY: (Xk[k_tr], yk[k_tr], gk[k_tr]),
        COMBINED: (np.vstack([Xk[k_tr], Xf_tr]), np.concatenate([yk[k_tr], yf_tr]), np.concatenate([gk[k_tr], gf_tr])),
    }
    inner = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    configs, fitted = {}, {}
    for config, (X, y, g) in train_sets.items():
        labels = sorted(np.unique(y))
        enc = LabelEncoder().fit(labels)
        models = {}
        for cand in candidates():
            y_fit = enc.transform(y) if cand.encode else y
            search = GridSearchCV(clone(cand.estimator), cand.grid or {}, cv=inner, scoring="f1_macro", n_jobs=1,
                                  refit=True, error_score="raise")
            search.fit(X, y_fit, groups=g)

            def predict(Xt, search=search, cand=cand):
                p = search.predict(Xt)
                return enc.inverse_transform(p) if cand.encode else p

            models[cand.name] = {
                "best_params": search.best_params_,
                "cv_macro_f1_mean": float(search.cv_results_["mean_test_score"][search.best_index_]),
                "cv_macro_f1_std": float(search.cv_results_["std_test_score"][search.best_index_]),
                "kaggle_test": evaluate(yk[k_te], predict(Xk[k_te]), kaggle_labels),
                "ffhq_human_test": evaluate(yf_te, predict(Xf_te), human_labels),
            }
            fitted[(config, cand.name)] = (search.best_estimator_, enc if cand.encode else None)
        selected = max(TRAINABLE, key=lambda n: models[n]["cv_macro_f1_mean"])
        configs[config] = {"train_size": int(len(y)), "labels": labels,
                           "class_counts": pd.Series(y).value_counts().sort_index().to_dict(),
                           "selected_model": selected, "models": models}

    phase2 = {"kaggle_test": evaluate(yk[k_te], phase2_rules(kag.iloc[k_te]), kaggle_labels),
              "ffhq_human_test": evaluate(yf_te, phase2_rules(ff_te), human_labels),
              "ffhq_human_test_forehead_occluded": int((ff_te.get("forehead_occluded", pd.Series(dtype=str))
                                                        .astype(str) == "True").sum())}

    current = None
    if current_model is not None:
        bundle = joblib.load(current_model)
        current = {"file": current_model.name, "model_name": bundle["model_name"], "trained_on": bundle["trained_on"],
                   "train_size": bundle["train_size"], "features_csv_sha256": bundle["features_csv_sha256"],
                   # It was fit on an older hold-out split, so this hold-out overlaps its training data.
                   "kaggle_test": None,
                   "ffhq_human_test": evaluate(yf_te, predict_bundle(bundle, Xf_te), human_labels)}

    ffhq_passed = ffhq_raw["passed"].astype(str) == "True"
    report = {
        "kaggle_features_csv_sha256": hashlib.sha256(kaggle_csv.read_bytes()).hexdigest(),
        "ffhq_features_csv_sha256": hashlib.sha256(ffhq_csv.read_bytes()).hexdigest(),
        "features": FEATURE_NAMES, "seed": SEED,
        "protocol": (f"identity-aware stratified Kaggle hold-out (1 of {N_FOLDS} folds), shared by both configurations; "
                     f"{N_FOLDS}-fold identity-aware stratified CV on each training set; FFHQ human test never trained on"),
        "selection_rule": "highest CV macro-F1 among RF, SVM, XGBoost and MLP; no test set was used for selection",
        "macro_f1_on_ffhq_human_test": (f"unweighted mean F1 over the shapes in the human labels ({', '.join(human_labels)}); "
                                        "predictions of other shapes count as errors"),
        "data": {
            "kaggle_used": int(len(kag)), "kaggle_train": int(len(k_tr)), "kaggle_test": int(len(k_te)),
            "kaggle_test_class_counts": pd.Series(yk[k_te]).value_counts().sort_index().to_dict(),
            "ffhq_rows": int(len(ffhq_raw)),
            "ffhq_rows_by_split": ffhq_raw["source_split"].value_counts().sort_index().to_dict(),
            "ffhq_passed_quality_gates_by_split": ffhq_raw.loc[ffhq_passed, "source_split"].value_counts().sort_index().to_dict(),
            "ffhq_rejection_reasons": ffhq_raw.loc[~ffhq_passed, "issues"].fillna("").astype(str).str.split(";").explode()
            .loc[lambda s: s != ""].value_counts().to_dict(),
            "ffhq_train_dropped_for_identity_overlap_with_test": int(leak.sum()),
            "ffhq_train_used": int(len(ff_tr)),
            "ffhq_train_class_counts": pd.Series(yf_tr).value_counts().sort_index().to_dict(),
            "ffhq_human_test_used": int(len(ff_te)),
            "ffhq_human_test_class_counts": pd.Series(yf_te).value_counts().sort_index().to_dict(),
        },
        "phase2_rule_baseline": phase2,
        "current_model": current,
        "configs": configs,
        "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__, "xgboost": xgboost.__version__,
                     "numpy": np.__version__},
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(report, indent=2, default=float))
    _plot(out, report)
    (out / "README.md").write_text(_markdown(report))

    if model_out is not None:
        name = configs[COMBINED]["selected_model"]
        est, enc = fitted[(COMBINED, name)]
        model_out.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model": est, "label_encoder": enc, "feature_names": FEATURE_NAMES, "labels": configs[COMBINED]["labels"],
            "model_name": name, "trained_on": "niten19+ffhq_ai_pseudo_labels", "train_size": configs[COMBINED]["train_size"],
            "features_csv_sha256": report["kaggle_features_csv_sha256"],
            "ffhq_features_csv_sha256": report["ffhq_features_csv_sha256"], "seed": SEED, "versions": report["versions"],
            "test_macro_f1": configs[COMBINED]["models"][name]["kaggle_test"]["macro_f1"],
            "ffhq_human_test_macro_f1": configs[COMBINED]["models"][name]["ffhq_human_test"]["macro_f1"],
            "calibrated": False,
        }, model_out / ARTIFACT, compress=3)
    return report


def headline(report: dict) -> dict[str, dict]:
    """The three models the comparison is about: rule baseline, current Kaggle-only artifact, retrained model."""
    combined = report["configs"][COMBINED]
    rows = {"Rule baseline (Phase 2, rule_based_v0)": report["phase2_rule_baseline"]}
    if report["current_model"] is not None:
        rows[f"Current Kaggle-only model ({report['current_model']['model_name']})"] = report["current_model"]
    rows[f"Kaggle + FFHQ AI labels ({combined['selected_model']}, selected)"] = combined["models"][combined["selected_model"]]
    return rows


def _plot(out: Path, report: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = headline(report)
    fig, axes = plt.subplots(1, len(rows), figsize=(4.6 * len(rows), 3.8))
    for ax, (name, m) in zip(np.atleast_1d(axes), rows.items()):
        t = m["ffhq_human_test"]
        cm, cols, labels = np.array(t["confusion_matrix"]), t["confusion_labels"], list(t["per_class"])
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(cols)), cols, rotation=45, fontsize=8)
        ax.set_yticks(range(len(labels)), labels, fontsize=8)
        for i in range(len(labels)):
            for j in range(len(cols)):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8)
        ax.set_title(f"{name}\nmacro-F1 {t['macro_f1']:.3f}", fontsize=8)
        ax.set_xlabel("predicted", fontsize=8)
        ax.set_ylabel("human label (Uzy)", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "confusion_matrices_ffhq_human_test.png", dpi=110)
    plt.close(fig)


def _f1(t: dict | None) -> str:
    if t is None:
        return "n/a"
    return f"{t['macro_f1']:.3f} ({t['macro_f1_95ci'][0]:.3f}-{t['macro_f1_95ci'][1]:.3f})"


def _markdown(r: dict) -> str:
    d, cur = r["data"], r["current_model"]
    lines = [
        "# Phase 3b results: Kaggle niten19 + FFHQ AI pseudo-labels, tested on human FFHQ labels", "",
        "Generated by `python -m ml.training.train_face_shape_ffhq`; every number below comes from that run (`metrics.json`).", "",
        f"- Kaggle images used: {d['kaggle_used']} (train {d['kaggle_train']}, hold-out {d['kaggle_test']}); "
        f"hold-out class counts: {d['kaggle_test_class_counts']}",
        f"- FFHQ rows (usable labels only): {d['ffhq_rows_by_split']}; passed the current quality gates: "
        f"{d['ffhq_passed_quality_gates_by_split']}; rejection reasons: {d['ffhq_rejection_reasons']}",
        f"- FFHQ AI pseudo-labels used for training: {d['ffhq_train_used']} {d['ffhq_train_class_counts']} "
        f"({d['ffhq_train_dropped_for_identity_overlap_with_test']} dropped for sharing an identity group with a test photo)",
        f"- FFHQ human test (Uzy's labels, one annotator, never trained on): {d['ffhq_human_test_used']} "
        f"{d['ffhq_human_test_class_counts']}",
        f"- Protocol: {r['protocol']}; seed {r['seed']}",
        f"- Macro-F1 on the human test: {r['macro_f1_on_ffhq_human_test']}", "",
        "## Headline: FFHQ human test", "",
        "| Model | Trained on | FFHQ human macro-F1 (95% CI) | FFHQ accuracy | Kaggle hold-out macro-F1 (95% CI) |",
        "|---|---|---|---|---|",
    ]
    trained = {0: "nothing (hand-written)", 1: f"{cur['trained_on']} ({cur['train_size']} images)" if cur else "",
               2: f"Kaggle train + FFHQ AI ({r['configs'][COMBINED]['train_size']} images)"}
    if cur is None:
        trained[1] = trained.pop(2)
    for i, (name, m) in enumerate(headline(r).items()):
        lines.append(f"| {name} | {trained[i]} | {_f1(m['ffhq_human_test'])} | {m['ffhq_human_test']['accuracy']:.3f} | "
                     f"{_f1(m['kaggle_test'])} |")
    lines += ["", "![Confusion matrices on the FFHQ human test](confusion_matrices_ffhq_human_test.png)", "",
              "Confusion matrices (rows: Uzy's label, columns: prediction):", ""]
    for name, m in headline(r).items():
        t = m["ffhq_human_test"]
        lines += [f"**{name}**", "", "| true \\ predicted | " + " | ".join(t["confusion_labels"]) + " |",
                  "|---" * (len(t["confusion_labels"]) + 1) + "|"]
        lines += [f"| {lab} | " + " | ".join(str(v) for v in row) + " |" for lab, row in zip(t["per_class"], t["confusion_matrix"])]
        lines += ["", "Per-class F1: " + ", ".join(f"{c} {v['f1']:.3f}" for c, v in t["per_class"].items()), ""]
    lines += ["## Every candidate", "",
              "| Config | Model | CV macro-F1 (mean ± std) | Kaggle hold-out macro-F1 (95% CI) | FFHQ human macro-F1 (95% CI) |",
              "|---|---|---|---|---|"]
    for config, c in r["configs"].items():
        for name, m in c["models"].items():
            mark = " (selected)" if name == c["selected_model"] else ""
            lines.append(f"| {config} | {name}{mark} | {m['cv_macro_f1_mean']:.3f} ± {m['cv_macro_f1_std']:.3f} | "
                         f"{_f1(m['kaggle_test'])} | {_f1(m['ffhq_human_test'])} |")
    lines += ["", f"Selection: {r['selection_rule']}.", "",
              "The headline rule baseline is `ml.classification.rule_based.classify` as the app runs it, dropping the "
              f"forehead features when hair covers the forehead ({r['phase2_rule_baseline']['ffhq_human_test_forehead_occluded']} "
              "human test photos); the `rule_based_v0` rows above always use all six features, as in Phase 3.", "",
              "`kaggle_only` is retrained here on the current feature table so it differs from the Kaggle+FFHQ run only by "
              "the added FFHQ rows; the committed Kaggle-only artifact was trained on an older feature table (quality gates "
              "have changed since), so its numbers are reported separately. Its Kaggle hold-out score is n/a here because that older split differs and "
              "overlaps this hold-out; see `research/results/phase3/kaggle_niten19`.", "",
              "Bootstrap intervals resample test images, not people. The human test has a single annotator, so label "
              "reliability is unknown; the AI training labels are pseudo-labels, not ground truth. Probabilities are not "
              "calibrated.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("kaggle_features_csv", type=Path)
    parser.add_argument("ffhq_features_csv", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model-out", type=Path, default=None)
    parser.add_argument("--current-model", type=Path, default=None, help="Kaggle-only artifact to compare against")
    args = parser.parse_args()
    r = run(args.kaggle_features_csv, args.ffhq_features_csv, args.out, args.model_out, args.current_model)
    for name, m in headline(r).items():
        print(f"{name:55s} ffhq_f1={m['ffhq_human_test']['macro_f1']:.3f} kaggle_f1={_f1(m['kaggle_test'])}")


if __name__ == "__main__":
    main()
