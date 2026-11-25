from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database.models import Hairstyle
from app.database.session import get_db
from app.schemas import HairstyleOut

router = APIRouter(prefix="/api/hairstyles", tags=["hairstyles"])


@router.get("", response_model=list[HairstyleOut])
def list_hairstyles(gender: Literal["pria", "wanita"] | None = None, db: Session = Depends(get_db)):
    rows = db.scalars(select(Hairstyle).order_by(Hairstyle.slug)).all()
    return [h for h in rows if gender is None or gender in h.gender_category]
