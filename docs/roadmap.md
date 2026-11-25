# Development Roadmap

Each phase ends with a review gate: the owner approves before the next phase starts. Durations assume focused work and are rough.

| Phase | Scope | Exit criteria (all must be real, measured) | Est. |
|---|---|---|---|
| 1 Research & architecture | Datasets, models, risks, architecture, structure | This repo's docs approved; dataset decision made | done |
| 2 Computer vision | Quality gates, MediaPipe landmarks, alignment, features, overlay visualization, rule-based classifier | `ml` package + pytest; landmark overlays on sample images; feature stability test (same face, small pose changes) | done |
| 3 Machine learning | Dataset prep (approved sources), label validation, feature selection, RF/SVM/XGBoost/MLP, stratified/group CV, tuning, serialization | Evaluation report with accuracy, macro P/R/F1, confusion matrices vs rule baseline; model card; reproducible with one command | Kaggle part done; FFHQ waits on annotators |
| 4 Recommendation | Curated catalog (male + female, ≥ 40 styles), filter + transparent scoring + ranking + explanations | Unit tests per rule; sanity review of top-5 for each shape × hair type | 3-4 d |
| 5 Backend & web | FastAPI, Postgres schema + Alembic, consent & deletion, Next.js pages in Bahasa Indonesia | End-to-end analysis → recommendations in browser; OpenAPI docs; integration tests | 6-9 d |
| 6 Generative AI | GPU benchmark, hair/face masks, SD1.5 inpainting (+ControlNet/IP-Adapter), async worker, evaluation | Benchmark + evaluation report (ArcFace similarity, LPIPS/FID-KID, boundary checks) on a fixed test set | 5-8 d |
| 7 Android | Expo app: camera/gallery, analysis, recommendations, try-on | APK build talking to the same API | 4-6 d |
| 8 Testing & deployment | Coverage of critical paths, performance, Docker Compose (CPU + GPU profiles), deployment docs | Fresh-machine install works from README | 3-4 d |

## Phase 2 status

Done: `ml/preprocessing` (decode, EXIF orientation, quality gates), `ml/feature_extraction` (Face Landmarker, roll alignment,
features), forehead-occlusion estimate from the MediaPipe hair segmenter, `ml/classification/rule_based.py`, overlay renderer,
pytest suite (`ml/tests`, CI in `.github/workflows/ml-tests.yml`), overlays in `research/results/phase2/`.

Known gaps carried into Phase 3: stability is tested with synthetic roll/scale/perspective changes of one photo, not real
multi-pose photos of one person; rule prototypes are uncalibrated and their accuracy is unknown until a labeled split exists.

## Phase 3 status

Done on Kaggle niten19 (research only, TD-14): feature table from the Phase 2 pipeline with quality-gate rejection
counts, SFace identity groups (TD-21), identity-aware hold-out + 5-fold CV comparing the rule baseline, fitted prototypes,
RF, SVM, XGBoost and MLP (`ml/training/train_face_shape.py`), results in
[`research/results/phase3/kaggle_niten19/`](../research/results/phase3/kaggle_niten19/README.md), model card in
[`docs/model-card.md`](model-card.md), serialized model in `ml/models/artifacts/`.

Prepared but **not done**: the FFHQ labeling set (`ml/datasets/ffhq_labeling/`), labeling tool, Indonesian annotation guide
and kappa script. FFHQ labels need at least two human annotators, none are named yet, so no FFHQ result and no Diamond
class exist. The trained model covers 5 shapes only.

## Phase 4 status

Done: catalog of 55 styles (30 for `pria`, 29 for `wanita`, some for both) with Indonesian text, filters, transparent rule
scoring, ranking and top-5 explanations in `ml/recommendation/` ([rules and limitations](recommendation-rules.md)). Unit
tests cover each rule and filter; the sanity review of top-5 per shape × hair type is in
[`research/results/phase4/sanity_review.md`](../research/results/phase4/sanity_review.md). Not done: reference images
(none license-clear yet), any validation with users or stylists, ML ranking (no preference data).

## Phase 5 status

FastAPI backend (`backend/`) with PostgreSQL schema and Alembic migration, consent-gated image storage, token-protected
read and deletion, OpenAPI docs at `/docs`, and a Next.js web app in Bahasa Indonesia. Integration tests run the real
pipeline against PostgreSQL in CI, plus a Playwright browser test of upload → analysis → recommendations → deletion.
Not done: user accounts and history across analyses (TD-30), try-on and Android (later phases).

## Next steps

1. FFHQ: annotators label the subset, run `ml.datasets.agreement`, add the consensus set to training and evaluation.
2. Phase 6 (generative try-on) and Phase 7 (Android) after the owner's go-ahead.
