# Direct zero-shot AI vision for face shape

**Research only. No app-model change.** One pass of Devin's built-in vision did not beat the prior
best trained models. No external model API or API key, cloud training, or user's GPU was used.

## Protocol

- Reused the frozen [PR #18 holdouts](../ffhq_ai_expansion/README.md): Uzy (101), screened FaceBench
  (59), Roboflow (22), and niten19. For niten19 only, selected 120 of the 682 holdout rows by
  class-stratified sampling without replacement (seed 42) before the visual pass. Older deployed
  app training overlaps this holdout; the deployed app is not scored here. The sample has no
  estimated-male faces, and niten19 has no Diamond ground truth.
- Displayed the same 144 × 144 face crops used by the ResNet scoring pipeline, in 16 local contact
  sheets. The crops were staged under random opaque filenames without metadata and shuffled across
  sources (seed 42). The only annotation added to each tile was an ordinal in the shuffled review.
  Neither the source mapping nor ground-truth labels was opened during prediction. Two existing
  source images (one niten19, one Roboflow) visibly contained face-outline graphics, so blinding
  was not perfect at the pixel level. Both predictions were incorrect.
- Devin viewed all 302 crops with the [fixed rubric](rubric.md) and made one of six forced choices
  for each crop, with no examples, retries, or feedback. The completed blinded CSV was frozen
  before scoring (SHA-256
  `8e2b3743ed7f65f7768c412125e32c9b5dc90ffd16e64495bfea990203ec7484`).
  [Published predictions](predictions.csv) are keyed only by cohort and zero-based row position in
  its frozen, hash-checked archive. No photos, sheets, source IDs, or private mapping are published.

## Results

Each cell is **accuracy / macro-F1**. Macro-F1 averages the classes present in the cohort's ground
truth; predictions outside those classes still count as errors. See [metrics.json](metrics.json)
for per-class precision, recall, F1 and support, and confusion matrices.

| Holdout | Devin vision | Prior reference |
|---|---:|---:|
| Uzy (101) | 30.7% / .354 | 41.6% / .416 ([PR #14 hybrid](../image_backbones/README.md)) |
| FaceBench (59) | 52.5% / .404 | 62.7% / .471 ([PR #16 ResNet](../kaggle_finetune/README.md)) |
| Roboflow (22) | 40.9% / .267 | 68.2% / .537 ([PR #18 expanded hybrid](../ffhq_ai_expansion/README.md)) |
| niten19 sample (120) | 38.3% / .441 | 63.3% / .642 (PR #14 hybrid on the same 120 rows) |

For the [matched niten19 reference](niten19-reference.json), refit the original hybrid with its
published SFace + landmark configuration on its original training groups, without a new sweep.
The ordered labels and crops matched the frozen local archive, and the refit reproduced all
published niten19 metrics (63.6% / .638 on 682 photos) before restricting its predictions to the
same 120 sampled rows.

| Holdout | Estimated cohort | n | Devin vision |
|---|---|---:|---:|
| Uzy | Female | 51 | 45.1% / .471 |
| Uzy | Male | 50 | 16.0% / .209 |
| FaceBench | Female | 31 | 61.3% / .544 |
| FaceBench | Male | 28 | 42.9% / .260 |
| Roboflow | Male | 22 | 40.9% / .267 |
| niten19 sample | Female | 120 | 38.3% / .441 |

Gender estimates are inherited from the local benchmark package, not inferred by Devin or
self-reported: DCGM FFHQ attributes on Uzy and FaceBench, InsightFace on Roboflow and niten19.
The best prior models' estimated-male results on Uzy, FaceBench and Roboflow were, respectively,
38.0% / .403, 57.1% / .507, and 68.2% / .537. The vision pass called the single Diamond in each
of FaceBench and Roboflow **Oblong**, for **0/2** Diamond hits. It also made two false-positive
Diamond calls on each of those holdouts; precision, recall, and F1 were all zero on both.
Neither Uzy nor niten19 has Diamond ground truth.

Vision accuracy was **49.3 points short of 80% on Uzy** and **27.5 points short on FaceBench**.
These numbers characterize this one pass with this rubric and crop format. Blinding example IDs
does not rule out prior exposure of a large model to public images during pretraining. Crop
distortion, occlusion, subjective class boundaries, and scarce Diamond examples limit what the
results say about the model's potential or an accuracy ceiling. The scoring is reproducible from
the saved calls; a fresh vision pass may not reproduce those calls.

## Reproduce

Use the local, hash-validated PR #18 package from the
[expansion runbook](../ffhq_ai_expansion/README.md). It is not necessary to upload anything.
To score the frozen predictions:

```sh
python -m ml.evaluation.vision_zero_shot score "$package" \
    research/results/phase3/vision_zero_shot/predictions.csv "$private/metrics.json"
```

For a separate repeat, stage a **new** local workspace with
`python -m ml.evaluation.vision_zero_shot stage "$package" "$workspace"`. Provide the reviewer
only the pages and rubric. Collect one `tile,prediction` row per tile with lowercase shape names.
After the reviewer freezes that CSV, run
`python -m ml.evaluation.vision_zero_shot freeze "$workspace/private-mapping.csv" "$blind_csv" "$predictions"`,
then score this new file separately. Keep the crops, pages, and mapping out of Git.
