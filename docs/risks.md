# Technical Risks and Dependencies

| # | Risk | Impact | Mitigation |
|---|---|---|---|
| 1 | No license-clear labeled face-shape dataset with 6 classes, both genders, identity IDs | ML results not defensible | Rule baseline first; use Kaggle set only with approval and caveats; self-label a small FFHQ set with κ agreement |
| 2 | Face-shape labels are subjective | Accuracy ceiling, noisy evaluation | Report inter-annotator agreement; macro-F1 + confusion matrix; treat adjacent-class confusions (Oval/Oblong, Round/Square) explicitly |
| 3 | Forehead/hairline not observable (MediaPipe has no hairline point; bangs) | Heart/Oblong errors | `forehead_visible` flag from hair mask; features that don't need forehead; lower confidence |
| 4 | Pose / selfie distortion changes ratios | Unstable predictions | Pose gate (yaw/pitch), guidance in UI for photo capture, test-retest stability metric |
| 5 | Identity leakage in CV | Inflated accuracy | Group-K-fold by identity when IDs exist; Kaggle set has none → report as limitation |
| 6 | Diffusion inpainting alters face identity | Try-on looks like someone else | Protect face-skin mask, IP-Adapter-FaceID/ControlNet, measure ArcFace similarity, reject outputs below threshold |
| 7 | 8 GB VRAM | SDXL slow/unstable | SD1.5 fp16 as default; SDXL with offload as optional; one GPU job at a time |
| 8 | Non-commercial licenses (InsightFace, IP-Adapter-FaceID, CelebAMask-HQ, Kaggle provenance) | Can't commercialize | Project scoped as research/portfolio; documented swap list |
| 9 | Biometric data (faces) privacy, incl. Indonesia PDP Law (UU 27/2022) | Legal / trust | Explicit consent before storage, deletion endpoint, no images in logs, in-memory processing by default |
| 10 | Hairstyle catalog quality and reference image rights | Weak recommendations, copyright | Curated metadata with sources; CC-licensed or generated reference images with attribution |
| 11 | Scope (web + Android + CV + ML + GenAI) | Slips | Strict phase gates; Android last, reuses API |

## External dependencies

MediaPipe (Apache-2.0), OpenCV, scikit-learn, XGBoost, PyTorch + CUDA 12.x, diffusers/transformers, FastAPI, SQLAlchemy/Alembic, PostgreSQL 16, Next.js, Expo, Docker (with NVIDIA Container Toolkit for the GPU worker).
