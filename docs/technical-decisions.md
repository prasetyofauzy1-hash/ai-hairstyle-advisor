# Technical Decisions

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
| TD-14 | Phase 3 data: Kaggle niten19 + self-labeled FFHQ, **non-commercial research only** (owner's decision) | Only large labeled source; scraped-photo provenance unclear, so no commercial use; FFHQ adds both genders and Diamond | Kaggle only; delaying until a clean commercial dataset exists |

## Phase 2 decisions

| ID | Decision | Reason | Alternatives rejected |
|---|---|---|---|
| TD-15 | Face Landmarker (Tasks API, 478 pts incl. iris, `num_faces=3`); largest face is the subject, a second face ≥ 50 % of its area rejects the photo | Main-face selection without a separate detector; ambiguity is surfaced instead of guessed | BlazeFace + crop + mesh (extra model, same result) |
| TD-16 | Features are ratios and the jaw angle after rotating landmarks so the iris line is horizontal | Invariant to scale, translation and roll (tested exactly) | Raw pixel distances; full 3D normalization (yaw from a single image is unreliable) |
| TD-17 | Forehead visibility = share of the forehead polygon the MediaPipe hair segmenter marks as hair; ≥ 0.3 drops forehead features from the classifier and warns the user | Landmark 10 is below the hairline and bangs cover it; this keeps hair from skewing the forehead width | Ignoring occlusion; asking users to pull hair back as a hard gate |
| TD-18 | Rule baseline = distance to per-shape prototypes, softmax to relative scores, flagged `calibrated: false` | Transparent, gives a ranked list for recommendations; Phase 3 evaluates and calibrates it | Hand-written if/else thresholds (brittle, no ranking) |
| TD-19 | Quality-gate messages and explanations in Bahasa Indonesia; logs carry only check codes and the predicted shape | Brief requirements; faces and image data never in logs (tested) | — |
| TD-20 | Public-domain NASA portraits as test fixtures (`ml/datasets/samples/SOURCES.md`) | License-clear, real faces for CI; carry no labels, so no accuracy is claimed from them | FFHQ samples (CC BY-NC-SA, would need attribution handling in CI) |
