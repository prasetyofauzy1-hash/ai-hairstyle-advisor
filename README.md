# AI Hairstyle Advisor

Research-oriented application that analyzes facial geometry, predicts face shape (Oval, Round, Square, Oblong, Heart, Diamond), recommends hairstyles with explanations, and offers a generative virtual try-on. Web (Next.js) and Android (Expo) clients, FastAPI backend, PostgreSQL, OpenCV + MediaPipe, scikit-learn / XGBoost, Hugging Face Diffusers.

> **Status: Phase 2 (computer vision pipeline).** The `ml` package detects the main face, checks image quality, extracts MediaPipe landmarks and normalized geometric features, and gives a rule-based face-shape estimate with landmark overlays. No trained model or evaluation metric exists yet; the rule-based scores are uncalibrated. Nothing in this repository reports a metric that was not produced by a run.

## Documents

| Doc | Content |
|---|---|
| [Architecture](docs/architecture.md) | Components, flows, data model, privacy |
| [Datasets](docs/datasets.md) | Dataset research, licenses, limitations, decision needed |
| [Pretrained models](docs/pretrained-models.md) | CV / GenAI model survey and licenses |
| [Compute requirements](docs/compute-requirements.md) | RTX 3060 Ti 8 GB estimates and trade-offs |
| [Risks](docs/risks.md) | Technical risks and dependencies |
| [Technical decisions](docs/technical-decisions.md) | Decision log |
| [Roadmap](docs/roadmap.md) | Phases 1-8 with exit criteria |
| [Face-shape features](research/methodology/face-shape-features.md) | Measurement method draft |

## Structure

```
frontend/web        Next.js app (Phase 5)
frontend/mobile     Expo Android app (Phase 7)
backend/app         FastAPI: api, services, schemas, database (Phase 5)
ml/                 CV + ML package: preprocessing, features, classification, recommendation, evaluation
generative_ai/      Hair segmentation, inpainting, virtual try-on (Phase 6)
research/           Methodology, experiments, results
docs/               Project documentation
```

## Local setup (so far)

```bash
cp .env.example .env    # then set a strong POSTGRES_PASSWORD (and in DATABASE_URL)
docker compose up -d db   # PostgreSQL 16, bound to 127.0.0.1 only
```

## Scope and ethics

Non-commercial research/portfolio project (several pretrained models and datasets are research-only, see docs). Facial images are stored only with explicit consent, can be deleted, and are never written to logs.

## Phase 2: run the CV pipeline

```bash
python3.10 -m venv .venv && . .venv/bin/activate
pip install -r ml/requirements.txt
python -m ml.models.download            # MediaPipe models, SHA-256 verified
python -m pytest ml/tests -q
python -m ml.analyze ml/datasets/samples/*.jpg --out research/results/phase2
```

Sample overlays and their JSON output are in `research/results/phase2/`.
