# Adding Roboflow Men's Face Shape v1

Adding the screened Roboflow training photos did **not** improve the held-out results overall. The same niten19 holdout score was essentially unchanged, and Uzy, FaceBench and the new Roboflow holdout scored lower than the existing combined artifact. Neither model correctly predicted the single Diamond in either Diamond-containing holdout. Keep the candidate as a research experiment; the app model is unchanged.

## Test results

Macro-F1 averages only the shapes represented in that cohort; predictions of other shapes still count as errors. The full [metrics JSON](roboflow_metrics.json) includes per-class precision, recall, F1, support and confusion matrices for each cohort and group. Subgroup values are descriptive only, especially with six estimated-male niten19 photos and a single Diamond per relevant benchmark.

| Held-out cohort | Photos | Prior macro-F1 | With Roboflow |
|---|---:|---:|---:|
| Uzy — all | 101 | 0.340 | 0.335 |
| Uzy — estimated male | 50 | 0.342 | 0.316 |
| Uzy — estimated female | 51 | 0.336 | 0.355 |
| FaceBench — all | 59 | 0.324 | 0.265 |
| FaceBench — estimated male | 28 | 0.221 | 0.202 |
| FaceBench — estimated female | 31 | 0.442 | 0.329 |
| Roboflow — all; all estimated male | 22 | 0.257 | 0.222 |
| niten19 — all | 682 | 0.5169 | 0.5171 |
| niten19 — estimated male | 6 | 0.292 | 0.292 |
| niten19 — estimated female | 675 | 0.5194 | 0.5197 |
| niten19 — unknown | 1 | 0.000 | 0.000 |

| Shape | Uzy (n; F1 prior → new) | FaceBench | Roboflow | niten19 |
|---|---|---|---|---|
| Diamond | — | 1; 0.000 → 0.000 | 1; 0.000 → 0.000 | — |
| Heart | — | 5; 0.400 → 0.364 | — | 154; 0.576 → 0.515 |
| Oblong | 11; 0.412 → 0.444 | 7; 0.471 → 0.333 | 5; 0.000 → 0.000 | 117; 0.440 → 0.416 |
| Oval | 62; 0.145 → 0.143 | 12; 0.000 → 0.000 | 5; 0.333 → 0.444 | 129; 0.380 → 0.390 |
| Round | 28; 0.464 → 0.417 | 22; 0.609 → 0.511 | 3; 0.200 → 0.000 | 193; 0.580 → 0.635 |
| Square | — | 12; 0.467 → 0.385 | 8; 0.750 → 0.667 | 89; 0.608 → 0.629 |

## Training and data quality

- Fit on the same 2,562 niten19 and 288 FFHQ AI training rows as the [previous experiment](README.md), plus 63 distinct screened Roboflow representatives, two labeled Diamond. See the [source and overlap audit](../../../../docs/datasets.md#follow-up-diamond-and-male-trainingevaluation-coverage). The prior in the table is the committed niten19 + FFHQ AI model, not the intermediate weighted experiment.
- Roboflow train counts: Diamond 2, Oblong 15, Oval 16, Round 10, Square 20. Of these, 61 were estimated male (including both Diamond), one female and one unknown. Triangle is not in the model taxonomy and was excluded. The original version stretches the photos to 512×512, so the data may distort shape ratios; the experiment cannot isolate resizing from the other source differences.
- The training-only search reused the 28 MLP/SVM configurations from the previous experiment, with five stratified identity-group folds (seed 59, the first seed in the search with all classes in every fold). The chosen one-layer, 32-unit MLP used alpha 0.01, inverse square-root class-frequency weights and 3× FFHQ AI weight; training CV macro-F1 was 0.392. No held-out predictions informed the choice.
- FFHQ gender estimates come from the pinned DCGM attributes as in the original report. For niten19 and Roboflow we used InsightFace buffalo_s `genderage.onnx` with dlib HOG single-face boxes and the 96×96 preprocessing documented in the original report. These binary image estimates are not self-reported identity and were not features or training weights.

## Reproduce

Prepare the same niten19 and FFHQ inputs as in the [original experiment](README.md#reproduce-the-ffhq-benchmark-comparison). For the additional cohort estimates, install dlib (the local run used `dlib-bin==20.0.1.post1`) and extract `genderage.onnx` from the [official buffalo_s v0.7 archive](https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_s.zip). Supply `ROBOFLOW_API_KEY` through the environment; the download needs it only if the pinned archive is not already cached. Source photos and the manifest stay in gitignored paths.

```sh
python -m ml.datasets.roboflow_men --out data/raw/roboflow_men \
    --features data/processed/roboflow_men

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m ml.training.train_face_shape_diamond \
    --kaggle-features data/processed/niten19/features.csv \
    --ffhq-features data/processed/ffhq/features.csv \
    --ffhq-images data/raw/ffhq --niten-embeddings data/processed/niten19/embeddings.npy \
    --facebench-manifest data/raw/facebench_ffhq/manifest.csv \
    --facebench-images data/raw/facebench_ffhq/images --attributes data/raw/ffhq-attributes \
    --identity-review research/results/phase3/facebench_ffhq/identity_review.csv \
    --roboflow-data data/raw/roboflow_men --roboflow-features data/processed/roboflow_men/features.csv \
    --roboflow-split research/results/phase3/diamond_retrain/roboflow_split.csv \
    --niten-images "data/raw/niten19/FaceShape Dataset" --gender-model data/raw/genderage.onnx \
    --out data/processed/diamond_retrain/roboflow_metrics.json
```
