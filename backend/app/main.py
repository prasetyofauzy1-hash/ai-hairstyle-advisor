from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import logging_filter
from app.body_limit import BodyLimitMiddleware
from app.api import analyses, hairstyles
from app.config import settings
from app.database.session import SessionLocal
from app.services.catalog_sync import sync_catalog

logging.basicConfig(level=logging.INFO)
logging_filter.install()


@asynccontextmanager
async def lifespan(_: FastAPI):
    logging_filter.install()  # again, after uvicorn has attached its handlers
    with SessionLocal() as db:
        sync_catalog(db)
    yield


app = FastAPI(title="AI Hairstyle Advisor API", version="0.5.0", lifespan=lifespan,
              description="Face-shape analysis and rule-based hairstyle recommendations. Research use only.")
app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["*"], allow_headers=["*"])
app.add_middleware(BodyLimitMiddleware, max_bytes=settings.max_upload_bytes)
app.include_router(analyses.router)
app.include_router(hairstyles.router)


@app.get("/api/health", tags=["health"])
def health():
    return {"status": "ok"}
