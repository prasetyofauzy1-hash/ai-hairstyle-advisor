# Development Roadmap

Each phase ends with a review gate: the owner approves before the next phase starts. Durations assume focused work and are rough.

| Phase | Scope | Exit criteria (all must be real, measured) | Est. |
|---|---|---|---|
| 1 Research & architecture | Datasets, models, risks, architecture, structure | This repo's docs approved; dataset decision made | done |
| 2 Computer vision | Quality gates, MediaPipe landmarks, alignment, features, overlay visualization, rule-based classifier | `ml` package + pytest; landmark overlays on sample images; feature stability test (same face, small pose changes) | 3-5 d |
| 3 Machine learning | Dataset prep (approved sources), label validation, feature selection, RF/SVM/XGBoost/MLP, stratified/group CV, tuning, serialization | Evaluation report with accuracy, macro P/R/F1, confusion matrices vs rule baseline; model card; reproducible with one command | 4-7 d (+ labeling time) |
| 4 Recommendation | Curated catalog (male + female, ≥ 40 styles), filter + transparent scoring + ranking + explanations | Unit tests per rule; sanity review of top-5 for each shape × hair type | 3-4 d |
| 5 Backend & web | FastAPI, Postgres schema + Alembic, consent & deletion, Next.js pages in Bahasa Indonesia | End-to-end analysis → recommendations in browser; OpenAPI docs; integration tests | 6-9 d |
| 6 Generative AI | GPU benchmark, hair/face masks, SD1.5 inpainting (+ControlNet/IP-Adapter), async worker, evaluation | Benchmark + evaluation report (ArcFace similarity, LPIPS/FID-KID, boundary checks) on a fixed test set | 5-8 d |
| 7 Android | Expo app: camera/gallery, analysis, recommendations, try-on | APK build talking to the same API | 4-6 d |
| 8 Testing & deployment | Coverage of critical paths, performance, Docker Compose (CPU + GPU profiles), deployment docs | Fresh-machine install works from README | 3-4 d |

## Immediate next steps (Phase 2, after approval)

1. `ml/preprocessing`: image loading, EXIF strip, quality gates.
2. `ml/feature_extraction`: Face Landmarker wrapper, eye alignment, feature vector + `forehead_visible`.
3. Landmark/measurement overlay renderer.
4. Rule-based classifier with soft scores.
5. Tests on a handful of license-clear sample images (FFHQ, attributed).
