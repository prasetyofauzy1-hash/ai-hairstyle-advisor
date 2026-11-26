# Additional AI-labeled FFHQ training photos

**Research only. The app model is unchanged.** This experiment expands the training data, not
the model architectures or the four frozen holdouts.

## Selection and labels

The source pool was 870 checksum-verified FFHQ photos (650 selected using the DCGM estimated-male
flag and 220 other). We used seed 103, the existing attribution/public-domain photo-license allowlist,
and omitted the original 600 FFHQ labeling IDs and all 81 FaceBench IDs. NVIDIA's dataset license and
the original Flickr photo terms still apply. See the [hosting and rights notes](../kaggle_finetune/README.md#before-any-cloud-upload).

Quality gates and finite SFace features retained 560 photos, and 555 also had exactly one dlib-detected
face. A local reference bundle contained 6,285 photo hashes from the original training and holdout
sources, including the raw source pools. Every benchmark reference had a SHA-256 and at least one
identity embedding: niten19 682/682, Uzy 104/104, FaceBench 81/81, and Roboflow 22/22. Seven other
raw references had hashes but neither embedding. We removed exact hash matches and pairs with SFace
cosine at least .56, dlib distance below .54, or both SFace cosine at least .363 and dlib distance
below .60. This retained 268 candidates, reduced to 253 by the same identity and exact duplication
checks within the candidate pool. These are approximate identity controls, not proof of disjoint
identities; no identity-pair evidence or embeddings are published.

For review ordering and within-candidate deduplication only, we prioritized v5 ResNet-18 Diamond
probability with a +.20 ranking bonus for InsightFace-estimated male. Devin labeled the 256×256
contact-sheet tiles by visual review, using only the photo and ID on the tile, not its predicted label
or score. Following the [existing labeling protocol](../../../../ml/datasets/ffhq_labeling/README.md),
obscured, distorted, or ambiguous faces were marked `ragu` and excluded from training. The labels
are `source=ai_pseudo_label`, not human ground truth. No photos, crops, model weights, or contact
sheets are committed. Source attribution and labels are in the separate
[expansion manifest and labels](../../../../ml/datasets/ffhq_labeling/expansion/).

| Reviewed label | Estimated male | Estimated female | Total |
|---|---:|---:|---:|
| Oval | 24 | 4 | 28 |
| Round | 17 | 4 | 21 |
| Square | 17 | 1 | 18 |
| Oblong | 24 | 2 | 26 |
| Heart | 5 | 3 | 8 |
| Diamond | 12 | 5 | 17 |
| **Training only** | **99** | **19** | **118** |
| Abstained (`ragu`) | 114 | 21 | 135 |
| **Total reviewed** | **213** | **40** | **253** |

All 118 accepted photos passed feature extraction again. The expansion added only 118 usable
examples, short of a few hundred, rather than forcing labels onto ambiguous faces.

## Training and scoring

Both retrains used 3,031 training photos: 2,562 niten19, 288 original FFHQ AI-labeled, 63 Roboflow,
and 118 new FFHQ AI-labeled. Expansion candidates screened as distinct received fresh training
identity groups. Before GPU submission we checked that the new training archive preserved every
original row and appended exactly the 118 expansion rows with noncolliding group IDs, and that
all arrays and gender assignments in the four local holdouts matched the v5 packages. Only the
training crop archive was uploaded, to an owner-matched, unshared private Kaggle dataset.

The local PR #14 SFace + landmark retrain swept the same SVM C and class weight settings under
training-only identity-grouped CV. It used five folds at seed 42 and selected C=1 and
inverse-square-root class weighting (mean CV macro-F1 .430). Because the groups and chosen fold
seed can differ from the original run, use the frozen holdouts rather than training CV to
compare runs. The private GPU retrain uses the unchanged [v5 ResNet-18 recipe](../kaggle_finetune/README.md).
The deployed app baseline is not scored on niten19 because its training overlaps that holdout.

Kaggle notebook version 7 completed in the private GPU environment on a Tesla T4, driver 580.178.04.
The log reported PyTorch `2.11.0+cu130`, CUDA 13.0, and `torch.cuda.is_available()=True`.
The ResNet used five grouped folds at seed 42 and selected learning rate .0001 (.378 mean validation
macro-F1 vs .363 for .0003). The checkpoint was downloaded and scored locally after hash validation.

## Results

Each cell is **accuracy / macro-F1**. Gender is an estimate, not a model input or self-identification.
See [hybrid metrics](hybrid-metrics.json) and [ResNet metrics](resnet-metrics.json) for per-class precision,
recall, F1, support and confusion matrices.

| Holdout | Cohort (photos) | Expanded hybrid | Expanded ResNet | PR #14 hybrid | v5 ResNet | App landmarks |
|---|---|---:|---:|---:|---:|---:|
| niten19 | All (682) | 59.1% / .592 | 50.3% / .516 | 63.6% / .638 | 48.1% / .489 | n/a |
| | Female (675) | 59.0% / .591 | 50.2% / .516 | 63.4% / .636 | 48.0% / .488 | n/a |
| | Male (6) | 66.7% / .708 | 50.0% / .458 | 83.3% / .833 | 50.0% / .458 | n/a |
| | Unknown (1) | 100.0% / 1.000 | 100.0% / 1.000 | 100.0% / 1.000 | 100.0% / 1.000 | n/a |
| Uzy | All (101) | 39.6% / .391 | 34.7% / .383 | 41.6% / .416 | 38.6% / .392 | 22.8% / .321 |
| | Female (51) | 47.1% / .433 | 47.1% / .546 | 45.1% / .413 | 51.0% / .522 | 21.6% / .342 |
| | Male (50) | 32.0% / .332 | 22.0% / .240 | 38.0% / .403 | 26.0% / .263 | 24.0% / .292 |
| FaceBench | All (59) | 52.5% / .360 | 50.8% / .432 | 50.8% / .349 | 62.7% / .471 | 40.7% / .251 |
| | Female (31) | 58.1% / .494 | 51.6% / .412 | 58.1% / .500 | 67.7% / .577 | 41.9% / .319 |
| | Male (28) | 46.4% / .258 | 50.0% / .404 | 42.9% / .230 | 57.1% / .507 | 39.3% / .248 |
| Roboflow | All and male (22) | 68.2% / .537 | 45.5% / .353 | 63.6% / .481 | 59.1% / .513 | 31.8% / .244 |

The expanded ResNet correctly classified FaceBench's only Diamond and made one false-positive Diamond
prediction (precision .500, recall 1.000, F1 .667, support 1). The expanded hybrid called that example
Oblong. Both new models called Roboflow's sole Diamond Oblong (precision, recall and F1 all zero
for Diamond). Uzy and niten19 have no Diamond ground truth. One hit on one example is not evidence
of reliable Diamond classification.

The expanded hybrid gained one correct photo each on FaceBench and Roboflow versus its own prior run,
but lost two on Uzy. Neither retrain improved on the current bests on both independently human-labeled
holdouts, and Uzy's estimated-male results fell versus each model's own prior scores. The strongest new
accuracy was 39.6% on Uzy and 52.5% on FaceBench, **40.4 and 27.5 percentage points short of 80%**.
Roboflow's 68.2% is 15/22, not evidence of reaching 80% across sources. Subjective class boundaries, sparse
Diamond ground truth and limited repeated labels prevent estimating a universal accuracy ceiling.
Additional AI labels avoid human training-label work, but cannot replace independent test labels.

## Reproduce

Start with the local data from the [PR #14 runbook](../image_backbones/README.md#reproduce) and
the original v5 checkpoint. The optional screening dependencies are CPU PyTorch/torchvision,
`dlib-bin`, dlib's five-point shape predictor and ResNet v1 recognition weights, and the pinned
InsightFace `genderage.onnx`. Reuse the exact reviewed cohort when reproducing the scores; do not
regenerate or replace its labels based on evaluation results.

```sh
workspace=data/processed/ffhq_ai_expansion
labels=ml/datasets/ffhq_labeling/expansion
python -m ml.datasets.ffhq_expand download data/raw/ffhq-dataset-v2.json "$workspace" \
    --attributes data/raw/ffhq-attributes --facebench-manifest data/raw/facebench_ffhq/manifest.csv
python -m ml.datasets.ffhq_expand features --manifest "$labels/manifest.csv" \
    --labels "$labels/ai_pseudo_labels_train.csv" --images "$workspace/images" \
    --out "$workspace/features.csv"
```

For a new screen, create a private newline-delimited list of all locally staged reference photo
paths from niten19, the original FFHQ labeling subset (including Uzy), FaceBench and Roboflow,
including all train, holdout and audit photos. Use `ffhq_expand references` to assemble a local
archive with both identity models. Confirm that the archive contains every holdout path with a hash
and at least one embedding. Use `ffhq_expand screen` with that archive, the same dlib weights,
InsightFace checkpoint, v5 ResNet-18 checkpoint, and a private crop cache. Contact sheets and
model ranking remain local. The labels for this run are already provided.

Append `--ffhq-expansion-features "$workspace/features.csv" --ffhq-expansion-images "$workspace/images"`
to both the normal `train_face_shape_image` command and the `kaggle_finetune pack` command from the
prior runbooks. Use a fresh package path and private dataset prefix, validate the package, and
follow the Kaggle runbook using only the training dataset. Download the exact version's
`resnet18.pt` and `selection.json`; `kaggle_finetune score` verifies the training archive, kernel,
and checkpoint hashes before evaluating the four local holdouts. Do not upload holdouts.
