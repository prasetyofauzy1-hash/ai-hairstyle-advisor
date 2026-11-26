# Pretrained image backbones for face shape (research only)

This experiment does not replace the app model or change the test sets. It compares two ways of learning from
face photos with the Phase 2 rules, the deployed niten19 landmark model, and the published niten19 + FFHQ AI landmark
model. It uses the same fixed cohorts as the [Roboflow experiment](../diamond_retrain/roboflow.md).

## Protocol

- The trainer uses the niten19 identity-grouped training split, the quality-passing FFHQ AI training rows after
  removing groups overlapping Uzy, and the screened Roboflow training split. Roboflow retained one representative
  per identity and was screened against the other training and benchmark pools. Group IDs from different sources
  are not considered equivalent; existing within-source groups and cross-source audits are approximate and
  do not prove that every identity is distinct.
- MediaPipe finds the largest face on each previously quality-passing photo. A square around its mesh bounding
  box includes 25% padding relative to the longest side, preserves aspect ratio and is resized to 144 × 144;
  borders are replicated. No pose warp or left/right averaging is applied. This cannot undo the source's 512 × 512
  aspect distortion on Roboflow.
- With no CUDA device, MobileNetV3-Small and ResNet-18 are frozen ImageNet V1 feature extractors. Inputs are RGB
  with ImageNet channel normalization. The models' globally pooled vectors are L2-normalized; the hybrid also
  appends separately normalized landmark features standardized using that fold's training rows only. A follow-up
  round adds the repository's SFace descriptor, a separate 128-dimensional, pretrained face-recognition network
  run on a five-point-aligned crop rather than the square crop. This is transfer learning via a trained head,
  not end-to-end fine-tuning.
- Separate group-disjoint CV sweeps for image-only and image-plus-landmark heads use RBF SVM C ∈ {1, 10}
  and inverse class-frequency exponent ∈ {0, 0.5}, choosing the highest mean validation macro-F1 in each family.
  As in the [previous experiment](../diamond_retrain/README.md), fold construction starts at seed 42 and requires
  each shape in every training and validation fold. Candidates are fixed and refitted on all training rows before
  benchmark inference. No benchmark data or scores inform head or backbone selection.
- Macro-F1 averages the classes with ground truth on each cohort or subgroup; guesses outside those classes
  still count as errors. Reported gender is an estimate, not self-identification or a model input: DCGM FFHQ
  estimates for Uzy and FaceBench; the pinned non-commercial InsightFace gender model for niten19 and Roboflow.
  Unknowns are retained, and very small support (including one Diamond on each of FaceBench and Roboflow)
  makes subgroup and class scores noisy.

## Results

Training: 2,562 niten19, 288 FFHQ AI-label and 63 screened Roboflow photos. With the ImageNet candidates alone,
training CV selected MobileNetV3-Small for the image-only head (macro-F1 0.299) and ResNet-18 for the hybrid
(0.381). Adding SFace raised these training CV scores to 0.353 and 0.422; both selected SFace with C = 1 and
class-frequency exponent 0.5. All 24 configurations used the same five grouped folds (seed 59).

Each cell is **accuracy / macro-F1**, not a pooled score. See [metrics.json](metrics.json) for all configurations,
per-class precision/recall/F1 and support, the full gender breakdown including unknowns, and confusion matrices.

| Held-out benchmark | Photos | Phase 2 rules | App landmarks | Combined landmarks | SFace only | SFace + landmarks |
|---|---:|---:|---:|---:|---:|---:|
| niten19 | 682 | 0.340 / 0.325 | 0.578 / 0.562 | 0.522 / 0.517 | 0.607 / 0.593 | **0.636 / 0.638** |
| Uzy | 101 | **0.426** / 0.391 | 0.228 / 0.321 | 0.277 / 0.340 | 0.376 / 0.319 | 0.416 / **0.416** |
| FaceBench | 59 | 0.475 / 0.306 | 0.407 / 0.251 | 0.458 / 0.324 | 0.424 / 0.246 | **0.508 / 0.349** |
| Roboflow | 22 | 0.364 / 0.256 | 0.318 / 0.244 | 0.364 / 0.257 | 0.455 / 0.366 | **0.636 / 0.481** |

The hybrid improved both metrics over the app and combined landmark artifacts on **both** independent human
benchmarks. The photo-only model did not: both external macro-F1 scores were lower than the app artifact.
On Uzy the hybrid correctly classified one fewer photo than the Phase 2 rules despite the higher macro-F1.
The [earlier Diamond-weighted landmark run](../diamond_retrain/README.md) reached 0.337 / 0.410 on Uzy and
0.424 / 0.274 on FaceBench. Neither photo-based model recognized the sole Diamond in either holdout; the
source and subgroup coverage is too limited to justify an app model swap.

| Held-out benchmark | Estimated cohort | Photos | App landmarks | SFace only | SFace + landmarks |
|---|---|---:|---:|---:|---:|
| niten19 | Female | 675 | 0.579 / 0.563 | 0.606 / 0.592 | 0.634 / 0.636 |
| niten19 | Male | 6 | 0.333 / 0.292 | 0.667 / 0.625 | 0.833 / 0.833 |
| Uzy | Female | 51 | 0.216 / 0.342 | 0.392 / 0.358 | 0.451 / 0.413 |
| Uzy | Male | 50 | 0.240 / 0.292 | 0.360 / 0.276 | 0.380 / 0.403 |
| FaceBench | Female | 31 | 0.419 / 0.319 | 0.387 / 0.270 | 0.581 / 0.500 |
| FaceBench | Male | 28 | 0.393 / 0.248 | 0.464 / 0.225 | 0.429 / 0.230 |
| Roboflow | Male | 22 | 0.318 / 0.244 | 0.455 / 0.366 | 0.636 / 0.481 |

## Label agreement and the 80% target

The original Uzy labels (104), FFHQ AI training labels (496), and FaceBench FFHQ labels (81) have **zero shared
FFHQ IDs** pairwise, so a same-photo human–human or human–AI agreement rate cannot be estimated from them.
A local exploratory dlib screen of Uzy vs. FaceBench followed by review of the five pairs at distance below 0.40
corroborated one same-person pair of different photos. The two labels agreed (Oblong); this is one pair, not a
representative agreement rate. The two previously corroborated FaceBench/FFHQ training-pool matches had an AI
abstention (`ragu`) on the training side, so they supply no forced-choice agreement labels.

As a noise check, three visually corroborated near-duplicate pairs in the previous Roboflow audit, all selected
by perceptual hash within Hamming distance 4, agreed on only one label. For those **three pairs only**, a predictor
forced to give each pair one consistent shape could match at most four of six assigned labels (67%).
In the quality-passing niten19 data, 49 of 150 multi-photo SFace groups had conflicting labels; using each
group's majority shape would match 3,139/3,244 labels (96.8%) across all groups, including singletons. Approximate
groups can include look-alikes, and different poses and crops can change perceived shape. Neither calculation is
a bound on any benchmark's accuracy. No pairing evidence or per-photo annotations are published here.

The best observed accuracy among the new models on the independent human benchmarks is **41.6% on Uzy**
and **50.8% on FaceBench**, respectively **38.4** and **29.2 percentage points short of 80%**. As a
near-term estimate for this training recipe and these cohorts, the hybrid's nominal 95% Wilson intervals are
32.5–51.3% and 38.4–63.2% (assuming independent photos; identity clustering can make these intervals too
narrow). The best niten19 and Roboflow point estimates were both 63.6%; those are different populations,
not evidence of approaching 80% on independent human labels. A universal accuracy ceiling is not
estimable from the available repeat labels. The most informative next experiment is a larger, rights-cleared
test with independently repeated labels, adjudication, and more male and Diamond examples.

## Reproduce

Start with the staged inputs in the [Roboflow reproduction instructions](../diamond_retrain/roboflow.md#reproduce).
Use the existing ML environment, then install CPU PyTorch and torchvision
(`torch==2.15.0`, `torchvision==0.29.2` in this run, from the PyTorch CPU index).
The first run downloads the torchvision ImageNet V1 weights. Weight and source-photo terms are separate;
do not distribute photos, crops, or research-only attribute models. The repository's
[dataset audit](../../../../docs/datasets.md) describes the source terms and rights limitations.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m ml.training.train_face_shape_image \
    --kaggle-features data/processed/niten19/features.csv \
    --ffhq-features data/processed/ffhq/features.csv \
    --ffhq-images data/raw/ffhq --niten-embeddings data/processed/niten19/embeddings.npy \
    --facebench-manifest data/raw/facebench_ffhq/manifest.csv \
    --facebench-images data/raw/facebench_ffhq/images --attributes data/raw/ffhq-attributes \
    --identity-review research/results/phase3/facebench_ffhq/identity_review.csv \
    --roboflow-data data/raw/roboflow_men --roboflow-features data/processed/roboflow_men/features.csv \
    --roboflow-split research/results/phase3/diamond_retrain/roboflow_split.csv \
    --niten-images "data/raw/niten19/FaceShape Dataset" --gender-model data/raw/genderage.onnx \
    --cache data/processed/image_backbones/crops --out data/processed/image_backbones/metrics.json
```
