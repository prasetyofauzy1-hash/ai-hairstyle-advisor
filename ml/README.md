# ML layer

Framework-independent Python package used by the backend.

| Folder | Content | Phase |
|---|---|---|
| datasets/ | dataset manifests and download scripts; `samples/` holds 7 public-domain test portraits | 2-3 |
| notebooks/ | exploration only; logic lives in modules | 2-3 |
| preprocessing/ | image quality checks, face crop/alignment | 2 |
| feature_extraction/ | MediaPipe landmarks -> geometric features | 2 |
| classification/ | rule-based baseline, RF / SVM / XGBoost / MLP | 2-3 |
| recommendation/ | catalog filtering, scoring, ranking | 4 |
| evaluation/ | landmark/measurement overlays (2); metrics, confusion matrices, CV reports (3) | 2-3 |
| models/ | MediaPipe model download + checksums (2); serialized models (git-ignored) + model cards (3) | 2-3 |

## Phase 2 pipeline (`ml.pipeline.FaceAnalyzer`)

1. `preprocessing.image_io.decode_image`: bytes → RGB, EXIF orientation applied, longest side ≤ 2048 px.
2. `feature_extraction.landmarks.FaceLandmarkDetector`: up to 3 faces, largest first, with yaw/pitch/roll.
3. `preprocessing.quality.assess_quality`: blocking gates (resolution, no/multiple faces, face size, cut-off, blur,
   exposure, head pose) with Bahasa Indonesia messages. Thresholds live in `QualityThresholds` and are initial guesses.
4. `feature_extraction.geometry.extract_features`: roll-aligned measurements → ratios + jaw angle.
5. `preprocessing.hair`: forehead hair fraction; ≥ 0.3 marks the forehead as covered.
6. `classification.rule_based.classify`: uncalibrated shape scores + Indonesian explanation.
7. `evaluation.visualization.render_face_report`: overlay image. CLI: `python -m ml.analyze <images> --out <dir>`.

Models download to `~/.cache/hairstyle-advisor/models` (override with `HAIRSTYLE_MODEL_DIR`).
