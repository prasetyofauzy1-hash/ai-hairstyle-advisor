# Technical Decisions (Phase 1)

| ID | Decision | Reason | Alternatives rejected |
|---|---|---|---|
| TD-01 | Monorepo per brief §11 | One place for code, research and docs | Multi-repo |
| TD-02 | `ml/` and `generative_ai/` as standalone Python packages | Inference independent from backend; testable without API | ML inside FastAPI services |
| TD-03 | MediaPipe Face Landmarker (478 pts) | Apache-2.0, CPU real-time, dense contour | dlib 68 pts (sparser jaw/forehead), training own model |
| TD-04 | Rule-based classifier before ML | No reliable dataset; gives a defensible baseline | Training directly on scraped labels |
| TD-05 | Geometric features → RF/SVM/XGBoost (CNN only as optional comparison) | Interpretable, small-data friendly, matches research goal | End-to-end CNN as primary |
| TD-06 | Stratified (Group)K-Fold, fixed seeds, experiments logged to `model_experiments` + `research/experiments` | Reproducibility, leakage control | Single random split |
| TD-07 | Transparent rule-weighted recommendation score | Explainable; no invented percentages | Opaque ML ranking before preference data exists |
| TD-08 | SD1.5 inpainting fp16 default, SDXL optional | 8 GB VRAM | SDXL default; training diffusion |
| TD-09 | Try-on as async GPU worker with Postgres job table | Isolation, no extra infra | Synchronous endpoint; Redis from day 1 |
| TD-10 | PostgreSQL 16 + SQLAlchemy 2 + Alembic | Required stack; migrations | Raw SQL |
| TD-11 | Images: consent-gated storage, UUID names, EXIF stripped, never logged | Biometric privacy, UU PDP | Store everything by default |
| TD-12 | Scope: non-commercial research/portfolio | Licenses of InsightFace, IP-Adapter-FaceID, CelebAMask-HQ | — |
| TD-13 | UI text in Bahasa Indonesia; code/docs in English | Brief §10; portfolio reach | — |
