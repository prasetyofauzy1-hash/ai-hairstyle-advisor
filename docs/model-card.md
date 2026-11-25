# Model card: face-shape classifier (Kaggle niten19, Phase 3)

All numbers come from `research/results/phase3/kaggle_niten19/metrics.json` (seed 42), produced by
`python -m ml.training.train_face_shape`.

| | |
|---|---|
| File | `ml/models/artifacts/face_shape_kaggle_niten19.joblib`, loaded with `ml.classification.learned.LearnedFaceShapeClassifier` |
| Model | scikit-learn MLP (hidden layers 64-32, alpha 0.01) on standardized features, chosen by CV macro-F1 |
| Inputs | The 6 Phase 2 features: `length_to_width`, `forehead_to_cheekbone`, `jaw_to_cheekbone`, `forehead_to_jaw`, `chin_to_jaw`, `jaw_angle_deg` |
| Outputs | One of heart, oblong, oval, round, square, with class probabilities that are **not calibrated** |
| Training data | Kaggle niten19: 2,977 of 5,000 images pass the quality gates; 2,660 train, 317 test; 250 SFace identity groups, none shared between train and test |
| Licence and use | Non-commercial research only (TD-14). niten19 photos are scraped celebrity images with unclear provenance |

## Results on the held-out test set (317 images)

| Model | CV macro-F1 | Test accuracy | Test macro-F1 (95% bootstrap CI) |
|---|---|---|---|
| Majority class | 0.057 | 0.199 | 0.066 |
| Rule baseline (Phase 2, hand-written) | 0.333 | 0.379 | 0.298 (0.243-0.346) |
| Rule prototypes fitted on train | 0.495 | 0.413 | 0.389 (0.336-0.444) |
| Random forest | 0.512 | 0.470 | 0.444 (0.386-0.499) |
| SVM (RBF) | 0.512 | 0.476 | 0.447 (0.389-0.500) |
| XGBoost | 0.508 | 0.495 | 0.476 (0.415-0.531) |
| **MLP (selected)** | 0.521 | 0.498 | 0.476 (0.417-0.530) |

Per-class test F1 of the MLP: square 0.646, heart 0.491, oval 0.485, round 0.415, oblong 0.344.

The trained models beat the hand-written rules and the majority baseline, but differ from each other by less than
their CV spread, so the choice among them is not meaningful. Roughly half of the test faces are misclassified.

## Results on human-labeled FFHQ photos (Phase 3b)

From `research/results/phase3/kaggle_plus_ffhq_ai/metrics.json` (seed 42), produced by
`python -m ml.training.train_face_shape_ffhq`. A second model, `ml/models/artifacts/face_shape_kaggle_ffhq_ai.joblib`
(MLP, chosen by CV macro-F1), was trained on the Kaggle train split plus FFHQ AI pseudo-labels. The test set is the
104 FFHQ test photos labeled by one person (Uzy); 101 pass the quality gates (oval 62, round 28, oblong 11; no heart,
square or diamond), and none of them, nor anyone grouped with them by SFace, was used for training.

| Model | FFHQ accuracy | FFHQ macro-F1 (95% bootstrap CI) |
|---|---|---|
| Rule baseline (Phase 2, hand-written) | 0.436 | **0.400 (0.295-0.510)** |
| Current Kaggle-only MLP (`face_shape_kaggle_niten19.joblib`) | 0.228 | 0.321 (0.212-0.407) |
| Kaggle + FFHQ AI pseudo-labels MLP (`face_shape_kaggle_ffhq_ai.joblib`) | 0.277 | 0.340 (0.240-0.432) |

Macro-F1 averages the three shapes in the human labels; predicting any other shape counts as an error. Confusion
matrices (rows: Uzy's label) are in the results README. Both learned models call most oval faces round, heart or
square (oval F1 0.061 and 0.145), while the rule baseline gets oval F1 0.544. Adding the pseudo-labels moved
macro-F1 from 0.321 to 0.340, well inside the intervals, so it is not a measurable improvement, and neither learned
model beats the hand-written rules on these photos. The intervals of all three overlap.

Training data for the new model: 2,562 Kaggle images (the identity-aware train split of the current feature table,
where 3,244 of 5,000 pass the gates) plus 288 FFHQ AI pseudo-labels (round 100, oval 84, oblong 51, square 32,
heart 18, diamond 3). Of the 364 usable pseudo-labels, 21 failed the gates and 55 were dropped because SFace grouped
the face with a test photo; that grouping is conservative (one group holds 22 photos), so it may drop more than needed.

## Limitations

- **No Diamond class**, and female celebrities only: the model does not cover the app's six shapes or men.
- **Label noise**: 46 identity groups (983 images) contain one person labeled with more than one shape. Part of this
  may be grouping errors, but it suggests the labels are inconsistent, which caps any model's score.
- Bootstrap intervals resample images, not people, so they understate uncertainty.
- Faces had to pass the quality gates; 40 % of niten19 was rejected (mostly small faces and head tilt), so results do
  not describe low-resolution or non-frontal photos.
- **The FFHQ human test has one annotator**, so no agreement can be measured and label reliability is unknown. It
  has no heart, square or diamond faces, so it says nothing about those classes, and with 101 photos the intervals are
  wide.
- **FFHQ training labels are AI pseudo-labels**, not ground truth; Diamond has only 3 usable examples, so the new
  model cannot be expected to learn it.
- The Kaggle hold-out score of the new model (0.517) is not comparable with the 0.476 above: the feature table now
  passes 3,244 images instead of 2,977, so the split differs. For the same reason the committed Kaggle-only model is
  not scored on the new hold-out, which overlaps its training data.
- 6 geometric features only; landmark 10 sits below the hairline, so face length is underestimated.

## Intended use

Research and portfolio comparison against the rule baseline. Show the result to users as an estimate with the
uncalibrated label, never as a diagnosis. The app keeps loading the Kaggle-only model; neither learned model beats the rule baseline on human-labeled FFHQ
photos, so this is not a reason to switch. Revisit once the FFHQ set has labels from more than one person.
