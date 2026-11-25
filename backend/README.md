# Backend (FastAPI)

Phase 5 API. It wraps the `ml` package (`FaceAnalyzer`, the Phase 3 learned classifier and the Phase 4 recommender);
no CV or ML logic is re-implemented here.

```
app/api        routers: analyses (upload, read, image, delete, recommendations), hairstyles
app/services   analysis.py (ml wrapper), storage.py (consent-gated image files), catalog_sync.py
app/schemas    Pydantic request/response models (drive the OpenAPI docs)
app/database   SQLAlchemy 2 models, session
alembic/       migrations (0001: face_analyses, facial_features, hairstyles, recommendations)
```

## Run

```bash
docker compose up -d db                      # from the repo root, after setting POSTGRES_PASSWORD in .env
pip install -r backend/requirements.txt
python -m ml.models.download
cd backend
export PYTHONPATH=..:. DATABASE_URL=postgresql+psycopg://hairstyle:<password>@localhost:5432/hairstyle_advisor
alembic upgrade head
uvicorn app.main:app --port 8000             # OpenAPI docs at http://localhost:8000/docs
```

Tests: `python -m pytest backend/tests -q` (uses `DATABASE_URL` if set, otherwise a local SQLite file; CI runs them on
PostgreSQL 16). They run the real MediaPipe pipeline and model on a public-domain sample photo.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/analyses` | multipart `image` + `consent` (default `false`). Returns quality, features, prediction and a one-time `access_token` |
| GET | `/api/analyses/{id}` | read an analysis (`X-Analysis-Token` header) |
| GET | `/api/analyses/{id}/image` | stored photo, only if consent was given (`Cache-Control: no-store`) |
| DELETE | `/api/analyses/{id}/image` | delete the stored photo, keep the analysis |
| DELETE | `/api/analyses/{id}` | delete the analysis, photo file, features and recommendations |
| POST | `/api/analyses/{id}/recommendations` | hair profile → top 5 from the rule engine; `face_shape` overrides the prediction |
| GET | `/api/hairstyles` | catalog (`?gender=pria|wanita`) |

## Privacy

- Uploads: max 8 MB, JPEG/PNG/WebP checked by magic bytes, then decoded; anything else is rejected with an Indonesian message.
- Without consent the photo is analyzed in memory only and never written to disk. With consent it is re-encoded as JPEG
  (drops EXIF/GPS) and saved as `<uuid4>.jpg` in `IMAGE_STORAGE_DIR`, outside any web root.
- No accounts yet: each analysis has a random token (only its SHA-256 is stored). Without it, an analysis looks like 404.
- Deleting an analysis removes the file and cascades to features and recommendations.
- Logs carry only booleans and check codes. `app/logging_filter.py` additionally drops records with raw bytes and redacts
  base64 blobs and stored image paths.

The predicted face shape comes from the Phase 3 MLP (5 shapes, no Diamond, female faces only, uncalibrated scores); each
response carries these limitations in Indonesian.
