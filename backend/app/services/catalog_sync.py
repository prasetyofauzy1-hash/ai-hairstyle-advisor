from __future__ import annotations

from dataclasses import asdict

from sqlalchemy.orm import Session

from app.database.models import Hairstyle
from ml.recommendation.catalog import load_catalog


def sync_catalog(db: Session) -> int:
    """Upsert the Phase 4 catalog (ml/recommendation/catalog.json is the source of truth) into `hairstyles`."""
    styles = load_catalog()
    for s in styles:
        attrs = asdict(s)
        for k in ("id", "name", "gender", "length", "reference_image"):
            attrs.pop(k)
        db.merge(Hairstyle(slug=s.id, name=s.name, gender_category=list(s.gender), length=s.length, attributes=attrs,
                           reference_image_path=None, image_license=None))
    db.commit()
    return len(styles)
