# Diamond and male coverage: weighted retraining

This is the experiment **before** the authenticated Roboflow export was available. See the [Roboflow follow-up](roboflow.md) for the expanded training run and new holdout.

The candidate improved macro-F1 on Uzy's labels and on both male-estimate subsets, but regressed on FaceBench overall and still missed its only Diamond example. No additional photo source cleared the [dataset audit](../../../../docs/datasets.md#follow-up-diamond-and-male-trainingevaluation-coverage). This experiment does not change either test set or replace the app's model.

## Test results

“Prior” is the existing combined niten19 + FFHQ AI model. Each macro-F1 averages the classes represented in that row's ground truth; a prediction of any other class still counts as an error. Subgroup values are descriptive; in particular, FaceBench's male-estimate subgroup has only one Diamond and one Heart, and the female-estimate subgroup has no Diamond.

| Test cohort | Photos | Prior macro-F1 | Retrained macro-F1 |
|---|---:|---:|---:|
| Uzy — all | 101 | 0.340 | **0.410** |
| Uzy — male estimate | 50 | 0.342 | **0.386** |
| Uzy — female estimate | 51 | 0.336 | **0.433** |
| FaceBench — all | 59 | **0.324** | 0.274 |
| FaceBench — male estimate | 28 | 0.221 | **0.231** |
| FaceBench — female estimate | 31 | **0.442** | 0.338 |
| niten19 holdout | 682 | **0.517** | 0.512 |

The prior niten19 holdout value is from the [original evaluation](../kaggle_plus_ffhq_ai/README.md); this run uses the same feature file and split. The [metrics JSON](metrics.json) contains precision, recall, F1, support, and confusion matrices for both FFHQ tests and their subgroups.

| Shape | Uzy photos | Uzy F1 prior → new | FaceBench photos | FaceBench F1 prior → new |
|---|---:|---:|---:|---:|
| Diamond | 0 | — | 1 | 0.000 → 0.000 |
| Heart | 0 | — | 5 | 0.400 → 0.200 |
| Oblong | 11 | 0.412 → 0.514 | 7 | 0.471 → 0.353 |
| Oval | 62 | 0.145 → 0.247 | 12 | 0.000 → 0.000 |
| Round | 28 | 0.464 → 0.471 | 22 | 0.609 → 0.591 |
| Square | 0 | — | 12 | 0.467 → 0.500 |

## Training and cohort coverage

Both sources used the existing quality gates and within-source SFace identity groups. The fit used 2,562 niten19 training rows and 288 FFHQ AI-labeled rows, after excluding FFHQ training groups that also occur in the Uzy test. The existing FaceBench identity audit still applies because no training source was added. Identity grouping is approximate; this run did not establish identity disjointness between niten19 and the FFHQ subsets.

| Cohort used | Male estimate | Female estimate | Unknown |
|---|---:|---:|---:|
| niten19 fit | 14 | 2,542 | 6 |
| FFHQ AI fit | 130 | 155 | 3 |
| niten19 holdout | 6 | 675 | 1 |
| Uzy test | 50 | 51 | 0 |
| FaceBench test | 28 | 31 | 0 |

All three training Diamond examples were male-estimate FFHQ photos with AI pseudo-labels, not new human-labeled examples. To avoid putting multiple photos from an identity group into different folds, the search used stratified group CV and required every fold to contain every shape; the three Diamond groups limited CV to three folds. The splitter chose the first seed from 42 onward satisfying that requirement (48). Of 28 MLP and RBF-SVM configurations, the best training-only mean macro-F1 was **0.416**: one 32-unit MLP layer, alpha 0.01, inverse square-root shape-frequency weights, and no extra FFHQ weight. The benchmarks were scored after selection. Gender estimates were used only to report cohorts, not as input features or fit weights.

FFHQ gender estimates come from the [DCGM FFHQ attributes](https://github.com/DCGM/ffhq-features-dataset/tree/9c4c7ef5132ddbdea686d83c63f4f29a5b01d46a) (Microsoft Face estimates); missing or non-single-face records are unknown. The niten19 counts are a separate local audit using [InsightFace buffalo_s](https://github.com/deepinsight/insightface/tree/master/model_zoo)'s `genderage.onnx` (research-only; SHA-256 `4fde69b1c810857b88c64a335084f1c3fe8f01246c9a191b48c7bb756d6652fb`). It used dlib's HOG detector on images downscaled to at most 480 px on the longer side (retrying once with upsampling if no face was found). It took the model's binary argmax on a centered, 1.5× padded square single-face crop warped to 96×96 RGB, and counted all other photos as unknown. Neither source measures self-identified gender, and the two estimators are not directly comparable. No per-photo gender estimates are published.

## Reproduce the FFHQ benchmark comparison

Use the [existing FaceBench preparation](../../../../docs/datasets.md#follow-up-public-pre-labeled-evaluation-data-2026-11-26), and check out the DCGM attributes at the linked revision locally:

```sh
python -m ml.datasets.ffhq_features --images data/raw/ffhq --out data/processed/ffhq \
    --kaggle-embeddings data/processed/niten19/embeddings.npy --download
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m ml.training.train_face_shape_diamond \
    --kaggle-features data/processed/niten19/features.csv \
    --ffhq-features data/processed/ffhq/features.csv \
    --ffhq-images data/raw/ffhq --niten-embeddings data/processed/niten19/embeddings.npy \
    --facebench-manifest data/raw/facebench_ffhq/manifest.csv \
    --facebench-images data/raw/facebench_ffhq/images --attributes data/raw/ffhq-attributes \
    --identity-review research/results/phase3/facebench_ffhq/identity_review.csv \
    --out data/processed/diamond_retrain/metrics.json
```
