"""CPU fine-tuning of a MobileNetV3 tail with cached frozen activations."""

from __future__ import annotations

import numpy as np

from ml.training.train_face_shape import SEED


def mobilenet():
    from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

    return mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)


def activations(pixels: np.ndarray) -> np.ndarray:
    import torch

    prefix = mobilenet().features[:-2].eval()
    mean = torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]
    output = []
    with torch.inference_mode():
        for start in range(0, len(pixels), 32):
            x = torch.from_numpy(pixels[start:start + 32]).permute(0, 3, 1, 2).float() / 255
            x = (x - mean) / std
            output.append(torch.stack((prefix(x), prefix(x.flip(-1))), dim=1).numpy())
    return np.concatenate(output)


class Tail:
    def __init__(self, net, classes: np.ndarray):
        self.net = net
        self.classes = classes

    def predict(self, x: np.ndarray) -> np.ndarray:
        import torch

        predictions = []
        self.net.eval()
        with torch.inference_mode():
            for start in range(0, len(x), 64):
                pair = torch.from_numpy(x[start:start + 64])
                logits = self.net(pair.flatten(0, 1)).reshape(len(pair), 2, -1)
                predictions.append(logits.mean(dim=1).argmax(dim=1).numpy())
        return self.classes[np.concatenate(predictions)]


def train_tail(x: np.ndarray, y: np.ndarray, sample_weight: np.ndarray, lr: float) -> Tail:
    import torch
    from torch import nn

    torch.manual_seed(SEED)
    classes, encoded = np.unique(y, return_inverse=True)
    model = mobilenet()
    classifier = model.classifier
    classifier[-1] = nn.Linear(classifier[-1].in_features, len(classes))
    net = nn.Sequential(model.features[-2:], model.avgpool, nn.Flatten(), classifier)
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(reduction="none")
    rng = np.random.default_rng(SEED)
    for _ in range(4):
        order = rng.permutation(len(y))
        view = rng.integers(0, 2, len(y))
        for start in range(0, len(y), 64):
            rows = order[start:start + 64]
            inputs = torch.from_numpy(x[rows, view[rows]])
            target = torch.from_numpy(encoded[rows])
            weight = torch.from_numpy(sample_weight[rows].astype(np.float32))
            optimizer.zero_grad()
            loss = (criterion(net(inputs), target) * weight).mean()
            loss.backward()
            optimizer.step()
    return Tail(net, classes)
