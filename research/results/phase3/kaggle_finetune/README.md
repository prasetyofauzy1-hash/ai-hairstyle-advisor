# Private Kaggle GPU fine-tuning

**Research only; the app model is unchanged.** The end-to-end fine-tune improved both metrics on
FaceBench but fell short of the PR #14 hybrid on Uzy.

The private script fully fine-tunes an ImageNet V1 ResNet-18, including the backbone. It compares AdamW
learning rates 0.0001 and 0.0003 at five epochs using mean validation macro-F1 on training-only
identity-grouped folds. It starts each fit from pretrained weights, applies inverse square-root
class-frequency loss weighting and random horizontal flips, then averages original and flipped logits
at inference. It refits the winning setting on all training rows. The notebook sees only the training crop
archive. It has internet enabled to fetch the torchvision checkpoint and does not send inputs to a model API.

The [prior protocol](../image_backbones/README.md) defines the cohort membership: grouped niten19 split,
FFHQ AI-label groups separated from Uzy, screened FaceBench FFHQ with committed exclusions, and the committed
Roboflow split. Packaging also rejects byte-exact train/holdout matches. These controls reduce identity
leakage but do not prove that different photographs always depict different people. Report the four holdouts
separately. Compare with the [published references](../cpu_foundation/README.md) only when using those same
cohorts; the deployed app baseline is not valid on the overlapping niten19 holdout. Do not select settings from
holdout scores. Only improvements on both Uzy and FaceBench count toward reconsidering the app model.

## Before any cloud upload

Private visibility is not permission to upload or a new license. Crops are still identifiable source-photo
derivatives. Check the Kaggle terms for hosted content and confirm each applicable photo and annotation license,
any required attribution/share-alike, and permission to host the material there. Keep attribution and original
terms with the local source manifests.

| Source | Constraint |
|---|---|
| niten19 | The archive declares CC0 but contains scraped celebrity photos with unclear rights. The copy used for local research is not cleared for re-upload by that declaration. Applies to training and the niten19 holdout. |
| FFHQ AI and Uzy | Check the [FFHQ per-photo terms](https://github.com/NVlabs/ffhq-dataset#licenses), original creator attribution, dataset CC BY-NC-SA 4.0, and the terms or permission for derived labels. |
| FaceBench FFHQ | The FaceBench annotation card says CC BY 4.0; FFHQ source-photo and dataset terms still apply to the images and crops. |
| Roboflow v1 | The export declares CC BY 4.0 but does not establish underlying photo rights. API access and permission for local research are not clearance for Kaggle hosting. |

**Do not submit the mixed training archive without clearance for all three training sources.** Keep the test
sets local unless there is a reason to upload them and they are also cleared. The driver does not upload
holdouts by default; `--upload-holdouts` is optional and creates four separate private datasets not attached
to the kernel. The generated metadata says `other` rather than offering a blanket license on source photos.
Only the checkpoint and a selection report are downloaded by the driver. Do not publish the checkpoint
without reviewing terms and privacy, or share any of the Kaggle resources, crops, or local caches.

## Rehearse locally without credentials

Use the ML environment and staged inputs in the [image experiment instructions](../image_backbones/README.md#reproduce).
Run this from the repo root, using a new gitignored directory (or a private directory outside the repo).
Replace `local-rehearsal` with Fauzy's actual Kaggle username when preparing the package for submission.

```sh
workspace=data/processed/kaggle-finetune
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m ml.training.kaggle_finetune pack "$workspace" \
    --owner local-rehearsal \
    --kaggle-features data/processed/niten19/features.csv \
    --ffhq-features data/processed/ffhq/features.csv \
    --ffhq-images data/raw/ffhq --niten-embeddings data/processed/niten19/embeddings.npy \
    --facebench-manifest data/raw/facebench_ffhq/manifest.csv \
    --facebench-images data/raw/facebench_ffhq/images --attributes data/raw/ffhq-attributes \
    --identity-review research/results/phase3/facebench_ffhq/identity_review.csv \
    --roboflow-data data/raw/roboflow_men --roboflow-features data/processed/roboflow_men/features.csv \
    --roboflow-split research/results/phase3/diamond_retrain/roboflow_split.csv \
    --niten-images "data/raw/niten19/FaceShape Dataset" --gender-model data/raw/genderage.onnx \
    --cache data/processed/image_backbones/crops
python -m ml.training.kaggle_finetune validate "$workspace"
python "$workspace/kernel/kaggle_kernel.py" --train "$workspace/train/train.npz" \
    --out "$workspace/smoke" --smoke
```

The smoke run picks at most four groups per class, uses random initial weights and runs two CPU folds without
a checkpoint download. Its outputs cannot be used for benchmark scoring.

In the Kaggle editor, the script accepts the notebook launcher's injected connection-file argument.
Without explicit paths, it finds the single attached `train.npz` and writes to `/kaggle/working`.
For a scoreable checksum, run the unchanged staged `.py` file. Before training, the log records the PyTorch
version, CUDA build and availability, and the GPU name and driver reported by `nvidia-smi`.

## Submit once Fauzy's account and upload permissions are confirmed

Install the optional client (`python -m pip install kaggle==2.2.5 six==1.17.0`) in the research environment.
Supply Fauzy's `KAGGLE_USERNAME` and `KAGGLE_KEY` via the secrets environment or use a private `kaggle.json`
under `KAGGLE_CONFIG_DIR` (default `~/.kaggle`). Never put credentials in commands, metadata, or Git.
Check that the account has Kaggle GPU access and quota. After the rights check above:

```sh
python -m ml.training.kaggle_finetune submit "$workspace" --confirm-upload-rights
```

The driver creates private datasets or, with `--update-datasets`, versions existing datasets only after
checking their owner and unshared private visibility. It then pushes the private GPU script, waits for the
exact kernel version, and scores its downloaded checkpoint on the local holdouts. Do not run concurrent
submissions against the same dataset IDs. Output lives in `"$workspace/outputs"`; `metrics.json` contains
accuracy, macro-F1, per-class reports, estimated-gender subgroups, and confusion matrices. If waiting fails
after submission, inspect the exact version in Kaggle and download that version's two outputs into the
workspace before running `python -m ml.training.kaggle_finetune score "$workspace" "$workspace/outputs"`.
The scorer rejects checkpoints that do not match the training archive and script. Never publish the archives,
individually attributable metadata, or identity-screening evidence.

## GPU run and local results

Private notebook version 5 reused the existing private training dataset. No holdouts were uploaded.
The submission requested T4 and the latest environment instead of specifying the prior CPU image.
Kaggle reported `gcr.io/kaggle-private-byod/python@sha256:fc720c15545eb096a510e1f085b17fc118387a007599424d953a4f327b71cacf`.
The completed run logged PyTorch `2.11.0+cu130`, CUDA `13.0`, `torch.cuda.is_available()=True`,
and `Tesla T4, 580.178.04` from `nvidia-smi`. Kaggle's prior CPU-image runs did not train;
the successful run does not isolate which environment or account change resolved GPU access.

Training used 2,913 crops and five identity-disjoint folds at seed 59. The 0.0001 learning rate
won on training-only CV (mean macro-F1 0.386 vs 0.370 for 0.0003). The local scorer checked
the training archive, staged kernel and downloaded checkpoint hashes before scoring.

Each cell is **accuracy / macro-F1**. Gender cohorts use estimates, not self-reported labels.

| Held-out benchmark | Cohort (photos) | Kaggle fine-tune | PR #14 hybrid | App landmarks |
|---|---|---:|---:|---:|
| niten19 | All (682) | 48.1% / .489 | 63.6% / .638 | n/a |
| | Female (675) | 48.0% / .488 | 63.4% / .636 | n/a |
| | Male (6) | 50.0% / .458 | 83.3% / .833 | n/a |
| | Unknown (1) | 100.0% / 1.000 | 100.0% / 1.000 | n/a |
| Uzy | All (101) | 38.6% / .392 | 41.6% / .416 | 22.8% / .321 |
| | Female (51) | 51.0% / .522 | 45.1% / .413 | 21.6% / .342 |
| | Male (50) | 26.0% / .263 | 38.0% / .403 | 24.0% / .292 |
| FaceBench | All (59) | 62.7% / .471 | 50.8% / .349 | 40.7% / .251 |
| | Female (31) | 67.7% / .577 | 58.1% / .500 | 41.9% / .319 |
| | Male (28) | 57.1% / .507 | 42.9% / .230 | 39.3% / .248 |
| Roboflow | All, estimated male (22) | 59.1% / .513 | 63.6% / .481 | 31.8% / .244 |

Per-class precision, recall, F1, support and confusion matrices for every cohort are in the
[fine-tune metrics](metrics.json) and the [PR #14 reference metrics](../image_backbones/metrics.json)
(`image_plus_landmarks` and `app_landmarks`). The fine-tune missed the only Diamond example
in each of FaceBench and Roboflow. Its accuracy is 41.4 percentage points short of 80% on Uzy
and 17.3 points short on FaceBench. These results measure performance on the existing labels,
not a ceiling for the task. As in the [prior label-agreement audit](../image_backbones/README.md),
there are too few independently co-labeled identities to estimate a general ceiling.
