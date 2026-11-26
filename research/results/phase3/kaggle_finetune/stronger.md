# Stronger Kaggle backbones and late fusion

Research only: this experiment does not update the app model. It uses the fixed cohorts, rights
constraints and private training-only dataset from the [ResNet-18 experiment](README.md).

The kernel compares ImageNet V1 ConvNeXt-Tiny at 144 px with DINOv2 ViT-S/14 at 140 px
(`facebook/dinov2-small`, revision `ed25f3a31f01632728cabb09d1542f84ab7b0056`). Both are
fine-tuned end to end for eight epochs with AdamW and cosine decay. Backbone learning rates are
`5e-5` and `2e-5` respectively, with a 10× higher rate for the new classifier. Training uses random
resized crops, horizontal flips, small rotations and color jitter, plus mixup (alpha 0.2) and
label smoothing (0.05). Inference averages original and flipped logits before softmax.

Mean macro-F1 across the same five training-only identity-grouped folds selects the backbone.
The kernel refits it on all training rows and returns only a checkpoint, an aggregate selection
report, and private out-of-fold class probabilities in training archive order. The local scorer
checks hashes and verifies local crop, label and group alignment to the validated archive.

The fusion branch uses the PR #14 SFace + landmark SVC recipe. Multinomial logistic calibration
uses standardized out-of-fold SVC margins from a separate set of identity-grouped folds within each outer
training fold. Both models predict the same outer validation rows without training on those
rows or their identity groups. Mean outer-fold macro-F1 selects a neural probability weight from
0, 0.25, 0.5, 0.75 and 1; equal scores favor the weight closest to 0.5. The published PR #14
baseline remains the original, uncalibrated classifier. Neither the four local benchmark labels
nor their predictions determine the backbone or fusion weight.

## Reproduce

From the repository root, stage a validated private workspace for the stronger kernel:

```sh
python -m ml.training.kaggle_stronger stage "$previous_workspace" "$workspace"
```

Confirm that the existing notebook and its single training dataset are private and that the
dataset version is unchanged. With credentials supplied privately as in the
[original runbook](README.md#submit-once-fauzys-account-and-upload-permissions-are-confirmed),
submit only the staged kernel. Do not upload or attach the holdouts.

```sh
kaggle kernels push -p "$workspace/kernel" --accelerator NvidiaTeslaT4
```

After completion, download `stronger.pt`, `stronger-selection.json`, and `training-oof.npz`
from the exact completed version into a private output directory. Do not publish those files.
Score locally with the same source-path and cache arguments as the original packaging command;
if DINOv2 wins, use the Transformers version that produced its checkpoint. Version 6 used
`transformers==5.20.2`; Transformers 4.57.6 uses different attention parameter names and
cannot load that checkpoint. Scoring reconstructs the architecture without downloading weights.

```sh
python -m ml.training.kaggle_stronger score "$workspace" "$workspace/outputs" \
    [same cohort and cache arguments as the original packaging command]
```

Only publish the aggregate `stronger-metrics.json` from the scorer, not the checkpoint or per-row
outputs.

## Version 6 results

The private notebook reused the existing training dataset and completed on a Tesla T4
(PyTorch `2.11.0+cu130`, CUDA `13.0`, driver `580.178.04`). It used 2,913 training crops and five
identity-grouped folds at seed 59. DINOv2 won on training-only mean macro-F1 (0.387 vs 0.373
for ConvNeXt-Tiny). Training-only OOF scoring selected a 50/50 fusion (0.408 macro-F1, vs 0.387
for the standalone DINOv2 and 0.375 for the calibrated hybrid).

Each cell is **accuracy / macro-F1**. Subgroups use estimated gender; the six-photo niten19
male subgroup and one-photo unknown subgroup are too small for meaningful comparisons.

| Local holdout | Cohort (photos) | DINOv2 fine-tune | Late fusion | v5 ResNet-18 | PR #14 hybrid | App landmarks |
|---|---|---:|---:|---:|---:|---:|
| niten19 | All (682) | 47.7% / .472 | 55.7% / .551 | 48.1% / .489 | 63.6% / .638 | n/a |
| | Female (675) | 47.9% / .474 | 55.4% / .548 | 48.0% / .488 | 63.4% / .636 | n/a |
| | Male (6) | 16.7% / .100 | 83.3% / .833 | 50.0% / .458 | 83.3% / .833 | n/a |
| | Unknown (1) | 100.0% / 1.000 | 100.0% / 1.000 | 100.0% / 1.000 | 100.0% / 1.000 | n/a |
| Uzy | All (101) | 33.7% / .372 | 36.6% / .401 | 38.6% / .392 | 41.6% / .416 | 22.8% / .321 |
| | Female (51) | 39.2% / .450 | 39.2% / .443 | 51.0% / .522 | 45.1% / .413 | 21.6% / .342 |
| | Male (50) | 28.0% / .306 | 34.0% / .366 | 26.0% / .263 | 38.0% / .403 | 24.0% / .292 |
| FaceBench | All (59) | 39.0% / .284 | 55.9% / .363 | 62.7% / .471 | 50.8% / .349 | 40.7% / .251 |
| | Female (31) | 48.4% / .421 | 61.3% / .436 | 67.7% / .577 | 58.1% / .500 | 41.9% / .319 |
| | Male (28) | 28.6% / .164 | 50.0% / .428 | 57.1% / .507 | 42.9% / .230 | 39.3% / .248 |
| Roboflow | All, estimated male (22) | 54.5% / .441 | 59.1% / .447 | 59.1% / .513 | 63.6% / .481 | 31.8% / .244 |

Per-class precision, recall, F1, support and confusion matrices are in the [version 6 metrics](stronger-metrics.json),
[version 5 metrics](metrics.json) and [PR #14 reference metrics](../image_backbones/metrics.json).
The app model was not scored on the overlapping niten19 holdout. Both new models missed the
single Diamond example in each of FaceBench and Roboflow.

Fusion improved on standalone DINOv2 on every holdout, but it did not beat the strongest
reference on either human benchmark, or beat the PR #14 hybrid on both. The best new accuracy
is 43.4 percentage points below 80% on Uzy and 24.1 below on FaceBench. Prior bests remain
41.6% on Uzy among the learned models and 62.7% on FaceBench; the [prior label-agreement audit](../image_backbones/README.md)
found too few independently co-labeled identities to estimate an accuracy ceiling.
