from __future__ import annotations

import logging
import secrets
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, File, Form, HTTPException, Response, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.api.deps import get_analysis, hash_token
from app.config import settings
from app.database.models import FaceAnalysis, FacialFeatures, Recommendation
from app.database.session import get_db
from app.schemas import AnalysisCreated, AnalysisOut, RecommendationRequest, RecommendationsOut
from app.services import analysis as svc
from app.services import storage
from ml.preprocessing.image_io import ImageDecodeError, decode_image
from ml.recommendation import UserProfile, recommend

log = logging.getLogger("app.analyses")
router = APIRouter(prefix="/api/analyses", tags=["analyses"])

LIMITATIONS = [
    "Model hanya bisa memperkirakan lima bentuk: oval, bulat, persegi, lonjong, dan hati. Bentuk wajik (diamond) "
    "belum bisa dikenali otomatis, tetapi bisa Anda pilih sendiri.",
    "Model dilatih dan diuji hanya dengan foto wajah perempuan (data Kaggle niten19). Ketepatan hasil untuk wajah "
    "laki-laki belum diketahui.",
    "Pada data uji, sekitar 5 dari 10 tebakan bentuk wajah benar. Skor macro-F1, yang memberi bobot sama pada tiap "
    "bentuk wajah, adalah 0,476 dari 1. Angka ini tidak menunjukkan apakah hasil untuk foto Anda benar.",
    "Skor belum terkalibrasi, jadi tidak bisa digunakan untuk mengetahui peluang hasil ini benar.",
    "Model ini untuk riset nonkomersial.",
]
UNREADABLE = "Kami tidak bisa membuka foto ini. Coba pilih foto lain berformat JPEG, PNG, atau WebP."


def to_out(a: FaceAnalysis) -> dict:
    f = a.features
    return {
        "id": a.id, "created_at": a.created_at, "image_stored": a.image_path is not None,
        "consent_given_at": a.consent_given_at, "quality": a.quality,
        "features": f.values if f else None, "forehead_visible": f.forehead_visible if f else None,
        "prediction": a.class_probabilities and {
            "shape": a.predicted_shape, "label_id": a.class_probabilities["label_id"],
            "scores": a.class_probabilities["scores"], "method": a.model_version, "calibrated": False},
        "rule_baseline": a.rule_baseline, "limitations": LIMITATIONS,
    }


@router.post("", response_model=AnalysisCreated, status_code=201,
             summary="Analyze a face photo; the image is stored only when consent=true")
async def create_analysis(image: UploadFile = File(...), consent: bool = Form(False), db: Session = Depends(get_db)):
    data = await image.read(settings.max_upload_bytes + 1)
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(413, f"Ukuran foto maksimal {settings.max_upload_bytes // (1024 * 1024)} MB.")
    if storage.sniff_format(data) is None:
        raise HTTPException(415, UNREADABLE)
    pixels = storage.pixel_count(data)
    if pixels is None:
        raise HTTPException(422, UNREADABLE)
    if pixels > storage.MAX_PIXELS:
        raise HTTPException(413, "Resolusi foto terlalu besar (maksimal 40 megapiksel).")
    try:
        rgb = decode_image(data)
    except ImageDecodeError:
        raise HTTPException(422, UNREADABLE)
    out = svc.analyze(rgb)
    r = out.result
    token = secrets.token_urlsafe(32)
    a = FaceAnalysis(token_hash=hash_token(token), quality_passed=r.quality.passed, quality=r.quality.to_dict(),
                     rule_baseline=r.prediction.to_dict() if r.prediction else None)
    if out.learned:
        a.predicted_shape = out.learned["shape"]
        a.class_probabilities = {"scores": out.learned["scores"], "label_id": out.learned["label_id"]}
        a.model_version = out.learned["method"]
    if r.features is not None:
        d = r.to_dict()
        a.features = FacialFeatures(values=d["features"], forehead_hair_fraction=d["forehead_hair_fraction"],
                                    forehead_visible=d["forehead_visible"])
    if consent and r.quality.passed:
        a.consent_given_at = datetime.now(timezone.utc)
        a.image_path = storage.store_image(rgb)
    try:
        db.add(a)
        db.commit()
    except Exception:
        storage.delete_image(a.image_path)
        raise
    db.refresh(a)
    log.info("analysis created: passed=%s stored=%s", a.quality_passed, a.image_path is not None)
    return {**to_out(a), "access_token": token}


@router.get("/{analysis_id}", response_model=AnalysisOut)
def read_analysis(a: FaceAnalysis = Depends(get_analysis)):
    return to_out(a)


@router.get("/{analysis_id}/image", responses={200: {"content": {"image/jpeg": {}}}})
def read_image(a: FaceAnalysis = Depends(get_analysis)):
    if a.image_path is None:
        raise HTTPException(404, "Foto tidak disimpan untuk analisis ini.")
    return FileResponse(storage.image_file(a.image_path), media_type="image/jpeg",
                        headers={"Cache-Control": "no-store"})


@router.delete("/{analysis_id}/image", status_code=204, summary="Delete the stored photo, keep the analysis")
def delete_image(a: FaceAnalysis = Depends(get_analysis), db: Session = Depends(get_db)):
    storage.delete_image(a.image_path)
    a.image_path = None
    a.consent_given_at = None
    db.merge(a)
    db.commit()
    return Response(status_code=204)


@router.delete("/{analysis_id}", status_code=204, summary="Delete the analysis, its photo, features and recommendations")
def delete_analysis(a: FaceAnalysis = Depends(get_analysis), db: Session = Depends(get_db)):
    storage.delete_image(a.image_path)
    db.delete(db.merge(a))
    db.commit()
    return Response(status_code=204)


@router.post("/{analysis_id}/recommendations", response_model=RecommendationsOut)
def create_recommendations(req: RecommendationRequest, a: FaceAnalysis = Depends(get_analysis),
                           db: Session = Depends(get_db)):
    shape, source, secondary = req.face_shape, "user", None
    if shape is None:
        if a.predicted_shape is None:
            raise HTTPException(422, "Aplikasi belum bisa memperkirakan bentuk wajah. Silakan pilih sendiri.")
        shape, source = a.predicted_shape, "model"
        ranked = sorted(a.class_probabilities["scores"].items(), key=lambda kv: -kv[1])
        secondary = ranked[1][0] if len(ranked) > 1 else None
    profile = UserProfile(face_shape=shape, hair_type=req.hair_type, thickness=req.thickness, gender=req.gender,
                          preferred_length=req.preferred_length, max_maintenance=req.max_maintenance,
                          avoid_bangs=req.avoid_bangs, formal=req.formal, secondary_face_shape=secondary)
    res = recommend(profile)
    a = db.merge(a)
    a.recommendations.clear()
    for r in res["recommendations"]:
        a.recommendations.append(Recommendation(
            hairstyle_slug=r["id"], rank=r["rank"], score=r["score"], score_breakdown=r["score_breakdown"],
            explanation=r["explanation"], profile=req.model_dump() | {"face_shape": shape}))
    db.commit()
    return {**res, "analysis_id": a.id, "face_shape_source": source}
