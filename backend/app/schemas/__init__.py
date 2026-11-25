from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

FaceShape = Literal["oval", "round", "square", "oblong", "heart", "diamond"]


class QualityIssueOut(BaseModel):
    code: str
    message: str
    blocking: bool


class QualityOut(BaseModel):
    passed: bool
    issues: list[QualityIssueOut]
    metrics: dict[str, float]


class PredictionOut(BaseModel):
    shape: str
    label_id: str
    scores: dict[str, float] = Field(description="Model class probabilities; NOT calibrated")
    method: str
    calibrated: bool = False


class AnalysisOut(BaseModel):
    id: uuid.UUID
    created_at: datetime
    image_stored: bool
    consent_given_at: datetime | None
    quality: QualityOut
    features: dict | None
    forehead_visible: bool | None
    prediction: PredictionOut | None = Field(description="Phase 3 learned model (5 shapes, no Diamond)")
    rule_baseline: dict | None = Field(description="Phase 2 rule classifier (6 shapes, uncalibrated)")
    limitations: list[str]


class AnalysisCreated(AnalysisOut):
    access_token: str = Field(description="Shown once. Send as X-Analysis-Token to read or delete this analysis.")


class RecommendationRequest(BaseModel):
    hair_type: Literal["straight", "wavy", "curly", "coily"]
    thickness: Literal["thin", "medium", "thick"]
    gender: Literal["pria", "wanita"] | None = None
    preferred_length: Literal["very_short", "short", "medium", "long"] | None = None
    max_maintenance: Literal["low", "medium", "high"] | None = None
    avoid_bangs: bool = False
    formal: bool = False
    face_shape: FaceShape | None = Field(None, description="Override; defaults to the analysis prediction")


class ScoreTerm(BaseModel):
    rule: str
    points: int
    reason: str


class RecommendationOut(BaseModel):
    rank: int
    id: str
    name: str
    description: str
    reference_image: dict | None
    reference_image_note: str | None
    score: int
    score_unit: str
    score_breakdown: list[ScoreTerm]
    explanation: str
    advantages: list[str]
    styling_tips: list[str]
    length: str
    maintenance: str


class RecommendationsOut(BaseModel):
    analysis_id: uuid.UUID
    face_shape: str
    face_shape_label: str
    face_shape_source: Literal["model", "user"]
    method: str
    recommendations: list[RecommendationOut]
    notes: list[str]


class HairstyleOut(BaseModel):
    slug: str
    name: str
    gender_category: list[str]
    length: str
    attributes: dict
    reference_image_path: str | None
    image_license: str | None
