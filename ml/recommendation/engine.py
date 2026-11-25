"""Rule-based hairstyle recommendation (Phase 4): filter by metadata -> score with named rules -> rank -> top 5.

The score is a sum of rule points, not a percentage or probability. User-facing text is Bahasa Indonesia.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ml.classification.rule_based import SHAPE_LABELS_ID, SHAPES
from ml.recommendation import rules as R
from ml.recommendation.catalog import (GENDERS, HAIR_TYPES, LABELS_ID, LENGTHS, MAINTENANCE, THICKNESSES, Hairstyle,
                                       load_catalog)

SCORE_NOTE = "Poin mengikuti panduan penataan rambut, bukan persentase kecocokan atau jaminan hasil."
NO_IMAGE_NOTE = "Gambar referensi belum tersedia."


@dataclass(frozen=True)
class UserProfile:
    face_shape: str
    hair_type: str
    thickness: str
    gender: str | None = None             # "pria" / "wanita"; None shows every style
    preferred_length: str | None = None
    max_maintenance: str | None = None
    avoid_bangs: bool = False
    formal: bool = False
    secondary_face_shape: str | None = None

    def __post_init__(self):
        if self.face_shape not in SHAPES:
            raise ValueError(f"unknown face_shape {self.face_shape!r}")
        if self.hair_type not in HAIR_TYPES or self.thickness not in THICKNESSES:
            raise ValueError("unknown hair_type or thickness")
        if self.gender is not None and self.gender not in GENDERS:
            raise ValueError(f"unknown gender {self.gender!r}")
        if self.preferred_length is not None and self.preferred_length not in LENGTHS:
            raise ValueError(f"unknown preferred_length {self.preferred_length!r}")
        if self.max_maintenance is not None and self.max_maintenance not in MAINTENANCE:
            raise ValueError(f"unknown max_maintenance {self.max_maintenance!r}")
        if self.secondary_face_shape is not None and self.secondary_face_shape not in SHAPES:
            raise ValueError(f"unknown secondary_face_shape {self.secondary_face_shape!r}")


@dataclass
class Recommendation:
    style: Hairstyle
    score: int
    terms: list[tuple[str, int, str]] = field(default_factory=list)
    length_distance: int = 0

    def explanation(self) -> str:
        good = [r for _, p, r in self.terms if p > 0]
        bad = [r for _, p, r in self.terms if p < 0]
        text = " ".join(good) if good else "Gaya ini sesuai dengan pilihan yang Anda masukkan, tetapi panduan kami tidak memberi alasan kecocokan tambahan."
        if bad:
            text += " Perlu diperhatikan: " + " ".join(bad)
        return text

    def to_dict(self, rank: int) -> dict:
        return {
            "rank": rank, "id": self.style.id, "name": self.style.name, "description": self.style.description,
            "reference_image": self.style.reference_image,
            "reference_image_note": None if self.style.reference_image else NO_IMAGE_NOTE,
            "score": self.score, "score_unit": "poin",
            "score_breakdown": [{"rule": n, "points": p, "reason": r} for n, p, r in self.terms],
            "explanation": self.explanation(),
            "advantages": list(self.style.advantages), "styling_tips": list(self.style.styling_tips),
            "length": LABELS_ID[self.style.length], "maintenance": LABELS_ID.get(self.style.maintenance, "sedang"),
        }


def passes_filters(style: Hairstyle, p: UserProfile) -> bool:
    """Step 1: drop styles whose metadata makes them incompatible."""
    if p.gender is not None and p.gender not in style.gender:
        return False
    if p.hair_type not in style.hair_types:
        return False
    if p.max_maintenance is not None and MAINTENANCE.index(style.maintenance) > MAINTENANCE.index(p.max_maintenance):
        return False
    if p.avoid_bangs and style.has_bangs:
        return False
    return R.shape_points(p.face_shape, style.effects) >= 0


def score(style: Hairstyle, p: UserProfile, length_distance: int = 0) -> Recommendation:
    """Step 2: sum of named rule terms."""
    terms = list(R.shape_terms(p.face_shape, style.effects))
    if style.hair_types[p.hair_type] == "ideal":
        terms.append(("jenis_rambut", R.HAIR_TYPE_IDEAL[0], R.HAIR_TYPE_IDEAL[1].format(hair=LABELS_ID[p.hair_type])))
    thick = LABELS_ID.get(p.thickness, "sedang")
    if style.thickness[p.thickness] == "ideal":
        terms.append(("ketebalan", R.THICKNESS_IDEAL[0], R.THICKNESS_IDEAL[1].format(thick=thick)))
    elif style.thickness[p.thickness] == "avoid":
        terms.append(("ketebalan", R.THICKNESS_AVOID[0], R.THICKNESS_AVOID[1].format(thick=thick)))
    if p.preferred_length is not None:
        rule = (R.LENGTH_MATCH, R.LENGTH_ADJACENT)[length_distance] if length_distance < 2 else R.LENGTH_OTHER
        terms.append(("panjang", rule[0], rule[1].format(length=LABELS_ID[p.preferred_length])))
    if p.formal and style.formal:
        terms.append(("formal", *R.FORMAL_MATCH))
    if p.secondary_face_shape is not None and p.secondary_face_shape != p.face_shape:
        pts = R.shape_points(p.secondary_face_shape, style.effects)
        name = SHAPE_LABELS_ID[p.secondary_face_shape]
        if pts > 0:
            terms.append(("bentuk_wajah_kedua", R.SECONDARY_SHAPE_FIT[0], R.SECONDARY_SHAPE_FIT[1].format(shape=name)))
        elif pts < 0:
            terms.append(("bentuk_wajah_kedua", R.SECONDARY_SHAPE_MISFIT[0], R.SECONDARY_SHAPE_MISFIT[1].format(shape=name)))
    return Recommendation(style, sum(t[1] for t in terms), terms, length_distance)


def recommend(p: UserProfile, top_k: int = 5, catalog=None) -> dict:
    styles = [s for s in (catalog or load_catalog()) if passes_filters(s, p)]
    notes = [SCORE_NOTE]
    if p.face_shape == "diamond":
        notes.append("Model terlatih belum bisa mengenali bentuk Wajik (diamond), jadi bentuk ini berasal dari pilihan Anda atau "
                     "aturan ukuran wajah, bukan dari perkiraan model. Perlakukan hasilnya dengan hati-hati.")
    if p.preferred_length is None:
        recs = [score(s, p) for s in styles]
    else:
        i = LENGTHS.index(p.preferred_length)
        recs = []
        # Widen the length band only while fewer than top_k styles qualify; each step away costs points.
        for d in range(len(LENGTHS)):
            band = [s for s in styles if abs(LENGTHS.index(s.length) - i) == d]
            recs += [score(s, p, length_distance=d) for s in band]
            if len(recs) >= top_k:
                break
        if any(r.style.length != p.preferred_length for r in recs):
            notes.append("Gaya dengan panjang yang Anda pilih kurang dari lima, jadi kami juga menampilkan gaya dengan "
                         "panjang lain (poinnya dikurangi).")
    # Step 3: preferred length first (fallback styles only fill open slots), then score, face-shape points, id.
    recs.sort(key=lambda r: (r.length_distance, -r.score, -R.shape_points(p.face_shape, r.style.effects), r.style.id))
    top = recs[:top_k]
    if len(top) < top_k:
        notes.append(f"Hanya {len(top)} gaya yang sesuai dengan pilihan Anda.")
    return {
        "face_shape": p.face_shape, "face_shape_label": SHAPE_LABELS_ID[p.face_shape],
        "method": "rule_based_v1", "recommendations": [r.to_dict(k + 1) for k, r in enumerate(top)], "notes": notes,
    }
