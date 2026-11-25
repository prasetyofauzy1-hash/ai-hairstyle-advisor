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

## Limitations

- **No Diamond class**, and female celebrities only: the model does not cover the app's six shapes or men.
- **Label noise**: 46 identity groups (983 images) contain one person labeled with more than one shape. Part of this
  may be grouping errors, but it suggests the labels are inconsistent, which caps any model's score.
- Bootstrap intervals resample images, not people, so they understate uncertainty.
- Faces had to pass the quality gates; 40 % of niten19 was rejected (mostly small faces and head tilt), so results do
  not describe low-resolution or non-frontal photos.
- 6 geometric features only; landmark 10 sits below the hairline, so face length is underestimated.

## Intended use

Research and portfolio comparison against the rule baseline. Show the result to users as an estimate with the
uncalibrated label, never as a diagnosis. Replace or retrain once the FFHQ set has human consensus labels.
