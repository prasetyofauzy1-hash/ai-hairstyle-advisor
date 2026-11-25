from __future__ import annotations

import hashlib
import hmac
import uuid

from fastapi import Depends, Header, HTTPException
from sqlalchemy.orm import Session

from app.database.models import FaceAnalysis
from app.database.session import get_db


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def get_analysis(analysis_id: uuid.UUID, x_analysis_token: str = Header(...), db: Session = Depends(get_db)) -> FaceAnalysis:
    a = db.get(FaceAnalysis, analysis_id)
    if a is None or not hmac.compare_digest(a.token_hash, hash_token(x_analysis_token)):
        raise HTTPException(404, "Analisis tidak ditemukan.")
    return a
