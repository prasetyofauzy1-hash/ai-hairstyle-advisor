from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Index, Integer, String, Text, Uuid
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

JSONType = JSON().with_variant(JSONB(), "postgresql")


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class FaceAnalysis(Base):
    __tablename__ = "face_analyses"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    token_hash: Mapped[str] = mapped_column(String(64))
    image_path: Mapped[str | None] = mapped_column(String(255), nullable=True)
    consent_given_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    quality_passed: Mapped[bool] = mapped_column(Boolean)
    quality: Mapped[dict] = mapped_column(JSONType)
    predicted_shape: Mapped[str | None] = mapped_column(String(16), nullable=True)
    class_probabilities: Mapped[dict | None] = mapped_column(JSONType, nullable=True)
    model_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    rule_baseline: Mapped[dict | None] = mapped_column(JSONType, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    features: Mapped[FacialFeatures | None] = relationship(back_populates="analysis", cascade="all, delete-orphan",
                                                           passive_deletes=True, uselist=False)
    recommendations: Mapped[list[Recommendation]] = relationship(back_populates="analysis", cascade="all, delete-orphan",
                                                                 passive_deletes=True, order_by="Recommendation.rank")

    __table_args__ = (Index("ix_face_analyses_created_at", "created_at"),)


class FacialFeatures(Base):
    __tablename__ = "facial_features"

    analysis_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("face_analyses.id", ondelete="CASCADE"), primary_key=True)
    values: Mapped[dict] = mapped_column(JSONType)
    forehead_hair_fraction: Mapped[float | None] = mapped_column(Float, nullable=True)
    forehead_visible: Mapped[bool | None] = mapped_column(Boolean, nullable=True)

    analysis: Mapped[FaceAnalysis] = relationship(back_populates="features")


class Hairstyle(Base):
    __tablename__ = "hairstyles"

    slug: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(128))
    gender_category: Mapped[list] = mapped_column(JSONType)
    length: Mapped[str] = mapped_column(String(16))
    attributes: Mapped[dict] = mapped_column(JSONType)
    reference_image_path: Mapped[str | None] = mapped_column(String(255), nullable=True)
    image_license: Mapped[str | None] = mapped_column(String(128), nullable=True)

    __table_args__ = (Index("ix_hairstyles_length", "length"),)


class Recommendation(Base):
    __tablename__ = "recommendations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    analysis_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("face_analyses.id", ondelete="CASCADE"), index=True)
    hairstyle_slug: Mapped[str] = mapped_column(ForeignKey("hairstyles.slug"), index=True)
    rank: Mapped[int] = mapped_column(Integer)
    score: Mapped[int] = mapped_column(Integer)
    score_breakdown: Mapped[list] = mapped_column(JSONType)
    explanation: Mapped[str] = mapped_column(Text)
    profile: Mapped[dict] = mapped_column(JSONType)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    analysis: Mapped[FaceAnalysis] = relationship(back_populates="recommendations")
