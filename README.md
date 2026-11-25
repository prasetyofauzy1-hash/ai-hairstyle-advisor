# AI Hairstyle Advisor

Research-oriented application that analyzes facial geometry, predicts face shape (Oval, Round, Square, Oblong, Heart, Diamond), recommends hairstyles with explanations, and offers a generative virtual try-on. Web (Next.js) and Android (Expo) clients, FastAPI backend, PostgreSQL, OpenCV + MediaPipe, scikit-learn / XGBoost, Hugging Face Diffusers.

> **Status: Phase 5 (backend & web).** A FastAPI backend with PostgreSQL and a Next.js web app in Bahasa Indonesia run the full flow in a browser: upload (stored only with consent), face-shape analysis, recommendations, deletion ([backend](backend/README.md), [web](frontend/web/README.md)). The `ml` package detects the main face, checks image quality, extracts MediaPipe landmarks and normalized geometric features, and classifies face shape with a rule baseline or a model trained on Kaggle niten19 (5 shapes, research use only). Results are in [`research/results/phase3/kaggle_niten19`](research/results/phase3/kaggle_niten19/README.md) and the [model card](docs/model-card.md). The FFHQ labeling set is prepared but has no human labels yet. A rule-based recommender ranks 55 catalog hairstyles with Indonesian explanations ([rules](docs/recommendation-rules.md)); it is not validated with users. Every metric in this repository comes from a run.

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
frontend/web        Next.js app
frontend/mobile     Expo Android app (Phase 7)
backend/            FastAPI: api, services, schemas, database, Alembic migrations
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

## Phase 5: run the web app

```bash
docker compose up -d db
pip install -r backend/requirements.txt && python -m ml.models.download
(cd backend && PYTHONPATH=..:. alembic upgrade head && PYTHONPATH=..:. uvicorn app.main:app --port 8000) &
cd frontend/web && npm ci && npm run build && npm run start   # http://localhost:3000, API docs at /docs
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

## Phase 3: train and compare face-shape models

Download Kaggle niten19 (e.g. the Hugging Face mirror `bkprocovid19/face_shape`, `archive.zip`) and unpack it so
`data/raw/niten19/{training_set,testing_set}/<Class>/*.jpg` exists, then:

```bash
python -m ml.datasets.build_features data/raw/niten19 --out data/processed/niten19
python -m ml.training.train_face_shape data/processed/niten19/features.csv \
    --out research/results/phase3/kaggle_niten19 --model-out ml/models/artifacts
```

FFHQ labeling: `ml.datasets.ffhq_select` → `ml.datasets.labeling_tool` → `ml.datasets.agreement` (see
[`docs/annotation-guide-id.md`](docs/annotation-guide-id.md)).

## Phase 4: hairstyle recommendations

```python
from ml.recommendation import UserProfile, recommend
recommend(UserProfile("round", "wavy", "thin", gender="wanita", preferred_length="medium"))
```

Returns the top 5 styles with rule-point scores, a per-rule breakdown, explanation, advantages and tips in Bahasa Indonesia.
Rules and limitations: [`docs/recommendation-rules.md`](docs/recommendation-rules.md). Sanity review:
`python -m ml.recommendation.review`.
