"""Self-contained Kaggle script: end-to-end ResNet-18 training with no holdout inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

SEED = 42


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def grouped_folds(y: np.ndarray, groups: np.ndarray, maximum: int = 5):
    labels = set(y)
    count = min(maximum, min(len(set(groups[y == label])) for label in labels))
    if len(labels) < 2 or count < 2:
        raise ValueError("Training requires two or more identity groups per class")
    for seed in range(SEED, SEED + 100):
        folds = list(StratifiedGroupKFold(count, shuffle=True, random_state=seed).split(y, y, groups))
        if all(set(y[tr]) == set(y[va]) == labels for tr, va in folds):
            return seed, folds
    raise ValueError("Could not put all classes in each identity-disjoint fold")


def image_batch(pixels: np.ndarray, torch, device):
    x = torch.from_numpy(pixels).to(device).permute(0, 3, 1, 2).float() / 255
    mean = x.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
    std = x.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
    return (x - mean) / std


def network(classes: int, *, pretrained: bool):
    from torch import nn
    from torchvision.models import ResNet18_Weights, resnet18

    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, classes)
    return model


def fit(pixels, y, classes: int, *, lr: float, epochs: int, device, pretrained: bool):
    import torch

    torch.manual_seed(SEED)
    model = network(classes, pretrained=pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    weights = np.bincount(y, minlength=classes).astype(float) ** -0.5
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    rng = np.random.default_rng(SEED)
    model.train()
    for _ in range(epochs):
        for ix in np.array_split(rng.permutation(len(y)), (len(y) + 31) // 32):
            batch = image_batch(pixels[ix], torch, device)
            flip = torch.from_numpy(rng.random(len(ix)) < 0.5).to(device)
            batch[flip] = batch[flip].flip(-1)
            optimizer.zero_grad()
            loss_fn(model(batch), torch.from_numpy(y[ix]).to(device)).backward()
            optimizer.step()
    return model


def predict(model, pixels: np.ndarray, device) -> np.ndarray:
    import torch

    model.eval()
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(pixels), 32):
            batch = image_batch(pixels[start:start + 32], torch, device)
            predictions.append(((model(batch) + model(batch.flip(-1))) / 2).argmax(1).cpu().numpy())
    return np.concatenate(predictions)


def train(archive: Path, out: Path, *, smoke: bool = False) -> dict:
    import torch

    device = torch.device("cpu" if smoke else "cuda")
    if not smoke and not torch.cuda.is_available():
        raise RuntimeError("A Kaggle CUDA GPU must be enabled; use --smoke only for local CPU verification")
    with np.load(archive, allow_pickle=False) as data:
        if set(data.files) != {"pixels", "label", "group"}:
            raise ValueError("Training input must contain pixels, label and training-only group fields")
        pixels, label, groups = (data[key] for key in ("pixels", "label", "group"))
    if smoke:
        rows = np.array([np.flatnonzero((label == c) & (groups == g))[0]
                         for c in sorted(set(label)) for g in np.unique(groups[label == c])[:4]])
        pixels, label, groups = pixels[rows], label[rows], groups[rows]
        torch.set_num_threads(2)
    classes, y = np.unique(label, return_inverse=True)
    seed, folds = grouped_folds(y, groups, maximum=2 if smoke else 5)
    epochs = 1 if smoke else 5
    trials = []
    for lr in ((0.0001,) if smoke else (0.0001, 0.0003)):
        scores = []
        for tr, va in folds:
            model = fit(pixels[tr], y[tr], len(classes), lr=lr, epochs=epochs, device=device, pretrained=not smoke)
            pred = predict(model, pixels[va], device)
            scores.append(float(f1_score(y[va], pred, average="macro", zero_division=0)))
            del model
        trials.append({"lr": lr, "fold_macro_f1": scores, "cv_macro_f1": float(np.mean(scores))})
    best = max(trials, key=lambda trial: trial["cv_macro_f1"])
    model = fit(pixels, y, len(classes), lr=best["lr"], epochs=epochs, device=device, pretrained=not smoke)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.cpu().state_dict(), "classes": classes.tolist()}, out / "resnet18.pt")
    report = {"archive_sha256": digest(archive), "kernel_sha256": digest(Path(__file__)),
              "training_rows": len(y), "device": device.type,
              "smoke": smoke, "seed": seed, "folds": len(folds), "epochs": epochs, "selected": best, "trials": trials,
              "checkpoint_sha256": digest(out / "resnet18.pt")}
    (out / "selection.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Training-only selection: {len(y)} photos, {len(folds)} folds, {device.type}", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path)
    parser.add_argument("--out", type=Path, default=Path("/kaggle/working"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.train is None:
        candidates = list(Path("/kaggle/input").rglob("train.npz"))
        if len(candidates) != 1:
            raise ValueError("Attach exactly one private training dataset containing train.npz")
        args.train = candidates[0]
    train(args.train, args.out, smoke=args.smoke)


if __name__ == "__main__":
    main()
