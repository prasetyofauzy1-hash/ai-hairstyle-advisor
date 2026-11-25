"""Hairstyle catalog (Phase 4): structured metadata curated for this project, loaded and validated from catalog.json.

Entries are written from widely repeated styling-guide heuristics, not validated by stylists or user ratings.
Reference images are not included yet (see docs/recommendation-rules.md).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

CATALOG_PATH = Path(__file__).with_name("catalog.json")

GENDERS = ("pria", "wanita")
LENGTHS = ("very_short", "short", "medium", "long")
HAIR_TYPES = ("straight", "wavy", "curly", "coily")
THICKNESSES = ("thin", "medium", "thick")
MAINTENANCE = ("low", "medium", "high")
EFFECTS = ("volume_top", "volume_sides", "volume_jaw", "soft_layers", "textured", "side_part", "bangs_full",
           "bangs_side", "length_past_chin", "sharp_lines", "face_exposed", "tight_sides")
BANGS = ("bangs_full", "bangs_side")

LABELS_ID = {
    "very_short": "sangat pendek", "short": "pendek", "medium": "sedang", "long": "panjang",
    "straight": "lurus", "wavy": "bergelombang", "curly": "keriting", "coily": "keriting rapat",
    "thin": "tipis", "thick": "tebal", "low": "rendah", "high": "tinggi",
}


@dataclass(frozen=True)
class Hairstyle:
    id: str
    name: str
    gender: tuple[str, ...]
    length: str
    hair_types: dict[str, str]
    thickness: dict[str, str]
    effects: tuple[str, ...]
    maintenance: str
    formal: bool
    description: str
    advantages: tuple[str, ...]
    styling_tips: tuple[str, ...]
    reference_image: dict | None = None

    @property
    def has_bangs(self) -> bool:
        return any(e in BANGS for e in self.effects)


def _validate(s: dict) -> None:
    sid = s.get("id", "?")
    assert set(s["gender"]) <= set(GENDERS) and s["gender"], sid
    assert s["length"] in LENGTHS, sid
    assert s["hair_types"] and set(s["hair_types"]) <= set(HAIR_TYPES), sid
    assert set(s["hair_types"].values()) <= {"ideal", "ok"}, sid
    assert set(s["thickness"]) == set(THICKNESSES) and set(s["thickness"].values()) <= {"ideal", "ok", "avoid"}, sid
    assert s["effects"] and set(s["effects"]) <= set(EFFECTS), sid
    assert s["maintenance"] in MAINTENANCE and isinstance(s["formal"], bool), sid
    assert s["description"] and len(s["advantages"]) >= 2 and len(s["styling_tips"]) >= 2, sid


def parse(data: dict) -> list[Hairstyle]:
    styles = []
    for s in data["styles"]:
        _validate(s)
        styles.append(Hairstyle(
            id=s["id"], name=s["name"], gender=tuple(s["gender"]), length=s["length"], hair_types=dict(s["hair_types"]),
            thickness=dict(s["thickness"]), effects=tuple(s["effects"]), maintenance=s["maintenance"],
            formal=s["formal"], description=s["description"], advantages=tuple(s["advantages"]),
            styling_tips=tuple(s["styling_tips"]), reference_image=s.get("reference_image")))
    ids = [s.id for s in styles]
    assert len(ids) == len(set(ids)), "duplicate hairstyle id"
    return styles


@lru_cache(maxsize=1)
def load_catalog(path: Path = CATALOG_PATH) -> tuple[Hairstyle, ...]:
    return tuple(parse(json.loads(path.read_text(encoding="utf-8"))))
