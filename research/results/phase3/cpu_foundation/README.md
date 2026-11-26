# CPU-only foundation models and partial fine-tuning

This research experiment uses the four fixed cohorts from the [previous image experiment](../image_backbones/README.md).
It does not use a GPU, paid compute, new training data, or new labels. The app and its model are unchanged.

## Protocol

- **Frozen image features:** run [DINOv2 ViT-S/14](https://huggingface.co/facebook/dinov2-small) at 140 × 140 pixels
  and [CLIP ViT-B/32](https://huggingface.co/openai/clip-vit-base-patch32) at 224 × 224 on the same square face crops.
  DINOv2 concatenates the CLS token and the mean of the patch tokens; CLIP uses its projected image embedding,
  not text prompts or zero-shot labels. The code pins checkpoint revisions and uses each model's RGB normalization.
- **Hybrid and feature ensembles:** L2-normalize each backbone's vector and test DINOv2 or CLIP alone and
  with the repository's six landmark measurements. Also test DINOv2 + SFace, CLIP + SFace, and the three-backbone
  combination, each with and without landmark measurements. Equal-weight fusion concatenates normalized image vectors
  and scales them to unit norm; a single RBF SVM fits the resulting features. Landmarks are standardized on that
  fold's training rows only, then normalized. This is feature fusion, not voting across model predictions.
- **Partial CNN fine-tuning:** cache original and flipped MobileNetV3-Small ImageNet V1 activations from the frozen
  prefix. Train the final two feature modules and classification head for four epochs (AdamW, batch size 64).
  Randomly switch between original and horizontal-flip views in training; average the two logits in validation
  and test. No holdout statistics, activations or labels are used in training.
- **Model selection:** compare 22 configurations by mean validation macro-F1 on the same five identity-grouped
  training folds as PR #14. RBF SVM C ∈ {1, 3}; CNN learning rate ∈ {0.0003, 0.001}; inverse
  class-frequency exponent 0.5. Retrain each family's training-CV winner before benchmark inference, and
  refit the previous SFace + landmark hybrid with its fixed PR #14 settings for comparison. Model and
  hyperparameter choices were not informed by the holdout scores.
- **Benchmarking:** Uzy and FaceBench are independent human-labeled benchmarks; the experiment does not count
  gains restricted to niten19 or Roboflow as an improvement. The deployed niten19 landmark model is still
  omitted on the overlapping niten19 holdout. Estimated gender, supported classes, and known identity-screen
  limitations are unchanged from the previous experiment.

## Results

The same 2,913 images supplied training data (2,562 niten19, 288 FFHQ AI labels, 63 screened Roboflow). The
five grouped folds used seed 59. Training CV chose CLIP alone (macro-F1 .379), CLIP + landmarks (.416),
fused DINOv2 + CLIP + SFace + landmarks (.443), and MobileNetV3-Small at learning rate .0003 (.321).
The earlier SFace + landmarks reference scored .422 in its training CV.

Each cell is **accuracy / macro-F1**. The [full JSON](metrics.json) includes all 22 CV trials, the existing
rules and landmark baselines, per-class precision/recall/F1 and support, all estimated-gender results, and
confusion matrices. Each benchmark remains separate.

| Held-out benchmark | Photos | PR #14 hybrid | CLIP | CLIP + landmarks | Feature ensemble + landmarks | Partial MobileNet |
|---|---:|---:|---:|---:|---:|---:|
| niten19 | 682 | .636 / .638 | .532 / .522 | .565 / .558 | **.645 / .644** | .375 / .361 |
| Uzy | 101 | **.416 / .416** | .376 / .399 | .396 / .402 | .386 / .374 | .218 / .281 |
| FaceBench | 59 | **.508 / .349** | .441 / .229 | .424 / .224 | .458 / .240 | .373 / .233 |
| Roboflow | 22 | **.636 / .481** | .545 / .362 | .500 / .324 | .591 / .434 | .500 / .396 |

The ensemble improved on the previous hybrid on niten19 but regressed on both independent human benchmarks.
No new model improved on the reference on either human benchmark overall. None recognized the single Diamond
on FaceBench or Roboflow. The best new accuracy was **39.6% on Uzy** and **45.8% on FaceBench**, respectively
**40.4 and 34.2 percentage points short of 80%**; these are different new models, not a combined result.
The previous hybrid remains the better choice (41.6% and 50.8%). The
[prior agreement audit](../image_backbones/README.md#label-agreement-and-the-80-target) could not establish
a general accuracy ceiling, and this model comparison does not add evidence for one.

| Held-out benchmark | Estimated cohort | Photos | PR #14 hybrid | Feature ensemble + landmarks |
|---|---|---:|---:|---:|
| niten19 | Female | 675 | .634 / .636 | .644 / .644 |
| niten19 | Male | 6 | .833 / .833 | .667 / .708 |
| Uzy | Female | 51 | .451 / .413 | .471 / .430 |
| Uzy | Male | 50 | .380 / .403 | .300 / .299 |
| FaceBench | Female | 31 | .581 / .500 | .452 / .203 |
| FaceBench | Male | 28 | .429 / .230 | .464 / .269 |
| Roboflow | Male | 22 | .636 / .481 | .591 / .434 |

The male-estimated FaceBench subset improved with the ensemble, but the male-estimated Uzy and Roboflow subsets
did not. These estimates are not self-reported gender. A larger, rights-cleared, independently relabeled
benchmark with more Diamond examples would help distinguish label disagreement from model error before paying
for longer training.

## Reproduce

Follow the [previous image experiment instructions](../image_backbones/README.md#reproduce) for the four staged
sources, landmark tables, attribute models and CPU torchvision/torch. Install `transformers==4.57.6` in the
research environment, then add `--foundation` to the previous command. Use a gitignored cache
and a different `--out` (for example, `data/processed/cpu_foundation/metrics.json`). Pinned foundation weights
download on the first run; use `HF_HUB_OFFLINE=1` on subsequent runs if those weights are already cached.
Do not publish crops, cached features, source photos, or identity evidence.

If more CPU time is needed, a larger multicore CPU-only cloud VM could parallelize trials or run longer
fine-tuning without using anyone's GPU. A paid hosted CPU embedding endpoint can offload frozen inference,
but should not receive source photos unless the photo terms and privacy requirements allow it. Neither option
was used here.
