"""Self-contained Kaggle script for end-to-end training with no holdout inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import linecache
import subprocess
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

SEED = 42
DEFAULT_EXPERIMENT = "resnet18"
STRONG_BACKBONES = {"convnext_tiny": (144, 5e-5), "dinov2_small": (140, 2e-5)}
DINO_REPO = "facebook/dinov2-small"
DINO_REVISION = "ed25f3a31f01632728cabb09d1542f84ab7b0056"


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def source_digest() -> str:
    if "__file__" in globals():
        return digest(Path(__file__))
    source = "".join(linecache.getlines(main.__code__.co_filename))
    if not source:
        raise RuntimeError("Save the kernel source to a .py file and run it as a script to record its checksum")
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


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


def cuda_diagnostics(torch) -> bool:
    available = torch.cuda.is_available()
    print(f"torch.__version__={torch.__version__}; torch.version.cuda={torch.version.cuda}; "
          f"torch.cuda.is_available()={available}", flush=True)
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                                capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"nvidia-smi unavailable: {exc}", flush=True)
    else:
        print(f"nvidia-smi exit {result.returncode}: {(result.stdout or result.stderr).strip()}", flush=True)
    return available


def train(archive: Path, out: Path, *, smoke: bool = False) -> dict:
    import torch

    device = torch.device("cpu" if smoke else "cuda")
    if not smoke and not cuda_diagnostics(torch):
        raise RuntimeError("A Kaggle CUDA GPU must be enabled; use --smoke only for local CPU verification")
    kernel_sha256 = source_digest()
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
    report = {"archive_sha256": digest(archive), "kernel_sha256": kernel_sha256,
              "training_rows": len(y), "device": device.type,
              "smoke": smoke, "seed": seed, "folds": len(folds), "epochs": epochs, "selected": best, "trials": trials,
              "checkpoint_sha256": digest(out / "resnet18.pt")}
    (out / "selection.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Training-only selection: {len(y)} photos, {len(folds)} folds, {device.type}", flush=True)
    return report


def strong_network(name, classes, *, pretrained=False, config=None):
    from torch import nn

    if name == "convnext_tiny":
        from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

        backbone = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None)
        dimensions = backbone.classifier[-1].in_features
        backbone.classifier[-1] = nn.Identity()
    elif name == "dinov2_small":
        from transformers import Dinov2Config, Dinov2Model

        backbone = (Dinov2Model.from_pretrained(DINO_REPO, revision=DINO_REVISION, use_safetensors=True)
                    if pretrained else Dinov2Model(Dinov2Config(**(config or {
                        "hidden_size": 384, "num_attention_heads": 6, "patch_size": 14,
                    }))))
        dimensions = backbone.config.hidden_size * 2
    else:
        raise ValueError(f"Unknown backbone: {name}")

    class Classifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone
            self.head = nn.Linear(dimensions, classes)

        def forward(self, x):
            features = self.backbone(x)
            if name == "dinov2_small":
                import torch

                tokens = features.last_hidden_state
                features = torch.cat((tokens[:, 0], tokens[:, 1:].mean(dim=1)), dim=1)
            return self.head(features)

    return Classifier()


def strong_batch(pixels, torch, device, name, augment=None):
    from torch.nn.functional import interpolate

    batch = torch.from_numpy(pixels).to(device).permute(0, 3, 1, 2).float() / 255
    size = STRONG_BACKBONES[name][0]
    if augment is not None:
        batch = augment(batch)
    elif batch.shape[-2:] != (size, size):
        batch = interpolate(batch, (size, size), mode="bilinear", align_corners=False, antialias=True)
    mean = batch.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
    std = batch.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
    return (batch - mean) / std


def fit_strong(pixels, y, classes, *, name, epochs, device, pretrained):
    import torch
    from torchvision.transforms import v2

    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    model = strong_network(name, classes, pretrained=pretrained).to(device)
    size, lr = STRONG_BACKBONES[name]
    optimizer = torch.optim.AdamW([{"params": model.backbone.parameters(), "lr": lr},
                                   {"params": model.head.parameters(), "lr": lr * 10}])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    weights = np.bincount(y, minlength=classes).astype(float) ** -0.5
    weights /= weights.mean()
    loss = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device),
                                     label_smoothing=0.05)
    augment = v2.Compose([v2.RandomResizedCrop((size, size), scale=(0.85, 1.0), ratio=(0.95, 1.05)),
                          v2.RandomHorizontalFlip(), v2.RandomRotation(8),
                          v2.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.12)])
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    model.train()
    for _ in range(epochs):
        indices = rng.permutation(len(y))
        for start in range(0, len(y), 32):
            ix = indices[start:start + 32]
            batch = strong_batch(pixels[ix], torch, device, name, augment)
            targets = torch.from_numpy(y[ix]).to(device)
            order = torch.randperm(len(ix), device=device)
            mix = float(rng.beta(0.2, 0.2))
            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                logits = model(batch * mix + batch[order] * (1 - mix))
                value = mix * loss(logits, targets) + (1 - mix) * loss(logits, targets[order])
            scaler.scale(value).backward()
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()
    return model


def strong_probabilities(model, pixels, device, name):
    import torch

    model.eval()
    probabilities = []
    with torch.inference_mode():
        for start in range(0, len(pixels), 32):
            batch = strong_batch(pixels[start:start + 32], torch, device, name)
            probabilities.append(((model(batch) + model(batch.flip(-1))) / 2).softmax(1).cpu().numpy())
    return np.concatenate(probabilities)


def train_strong(archive: Path, out: Path, *, smoke: bool = False) -> dict:
    import torch

    device = torch.device("cpu" if smoke else "cuda")
    if not smoke and not cuda_diagnostics(torch):
        raise RuntimeError("A Kaggle CUDA GPU must be enabled; use --smoke only for local CPU verification")
    with np.load(archive, allow_pickle=False) as data:
        if set(data.files) != {"pixels", "label", "group"}:
            raise ValueError("Training input must contain pixels, label and training-only group fields")
        pixels, label, groups = (data[key] for key in ("pixels", "label", "group"))
    if smoke:
        rows = np.array([np.flatnonzero((label == c) & (groups == g))[0]
                         for c in sorted(set(label)) for g in np.unique(groups[label == c])[:2]])
        pixels, label, groups = pixels[rows], label[rows], groups[rows]
        torch.set_num_threads(2)
    classes, y = np.unique(label, return_inverse=True)
    seed, folds = grouped_folds(y, groups, maximum=2 if smoke else 5)
    epochs = 1 if smoke else 8
    trials = []
    best, best_probabilities = None, None
    for name in (("convnext_tiny",) if smoke else STRONG_BACKBONES):
        probabilities = np.full((len(y), len(classes)), np.nan, dtype=np.float32)
        scores = []
        for tr, va in folds:
            model = fit_strong(pixels[tr], y[tr], len(classes), name=name, epochs=epochs,
                               device=device, pretrained=not smoke)
            probabilities[va] = strong_probabilities(model, pixels[va], device, name)
            scores.append(float(f1_score(y[va], probabilities[va].argmax(1), average="macro", zero_division=0)))
            print(f"{name} fold {len(scores)}/{len(folds)} macro-F1={scores[-1]:.4f}", flush=True)
            del model
        trial = {"backbone": name, "backbone_lr": STRONG_BACKBONES[name][1], "fold_macro_f1": scores,
                 "cv_macro_f1": float(np.mean(scores))}
        trials.append(trial)
        if best is None or trial["cv_macro_f1"] > best["cv_macro_f1"]:
            best, best_probabilities = trial, probabilities
    name = best["backbone"]
    model = fit_strong(pixels, y, len(classes), name=name, epochs=epochs, device=device, pretrained=not smoke)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint, oof = out / "stronger.pt", out / "training-oof.npz"
    config = json.loads(model.backbone.config.to_json_string(use_diff=False)) if name == "dinov2_small" else None
    torch.save({"state_dict": model.cpu().state_dict(), "classes": classes.tolist(), "backbone": name,
                "config": config}, checkpoint)
    np.savez_compressed(oof, probabilities=best_probabilities)
    report = {"archive_sha256": digest(archive), "kernel_sha256": source_digest(),
              "checkpoint_sha256": digest(checkpoint), "oof_sha256": digest(oof), "training_rows": len(y),
              "device": device.type, "smoke": smoke, "seed": seed, "folds": len(folds), "epochs": epochs,
              "selected": best, "trials": trials}
    (out / "stronger-selection.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Training-only selection: {len(y)} photos, {len(folds)} folds, {device.type}; {name}", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path)
    parser.add_argument("--out", type=Path, default=Path("/kaggle/working"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--experiment", choices=("resnet18", "stronger"), default=DEFAULT_EXPERIMENT)
    if Path(sys.argv[0]).stem in {"colab_kernel_launcher", "ipykernel_launcher"}:
        args, unknown = parser.parse_known_args()
        if unknown and not (len(unknown) == 2 and unknown[0] == "-f" and unknown[1].endswith(".json")):
            parser.error(f"unrecognized arguments: {' '.join(unknown)}")
    else:
        args = parser.parse_args()
    if args.train is None:
        candidates = list(Path("/kaggle/input").rglob("train.npz"))
        if len(candidates) != 1:
            raise ValueError("Attach exactly one private training dataset containing train.npz")
        args.train = candidates[0]
    trainer = train_strong if args.experiment == "stronger" else train
    trainer(args.train, args.out, smoke=args.smoke)


if __name__ == "__main__":
    main()
