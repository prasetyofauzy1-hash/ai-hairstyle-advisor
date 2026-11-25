import copy
import itertools
import json

import pytest

from ml.classification.rule_based import SHAPES
from ml.recommendation import rules as R
from ml.recommendation.catalog import CATALOG_PATH, HAIR_TYPES, LENGTHS, THICKNESSES, load_catalog, parse
from ml.recommendation.engine import SCORE_NOTE, UserProfile, passes_filters, recommend, score

CATALOG = load_catalog()
BY_ID = {s.id: s for s in CATALOG}
RAW = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))


def rule_names(rec):
    return [b["rule"] for b in rec["score_breakdown"]]


# ---- catalog

def test_catalog_size_and_gender_coverage():
    assert len(CATALOG) >= 40
    assert sum("pria" in s.gender for s in CATALOG) >= 20
    assert sum("wanita" in s.gender for s in CATALOG) >= 20


@pytest.mark.parametrize("hair", HAIR_TYPES)
def test_every_hair_type_has_styles_for_both_genders(hair):
    for g in ("pria", "wanita"):
        assert sum(hair in s.hair_types and g in s.gender for s in CATALOG) >= 5


def test_catalog_rejects_bad_entries():
    for field, bad in [("length", "huge"), ("hair_types", {"bald": "ideal"}), ("effects", ["glitter"]),
                       ("gender", ["other"]), ("thickness", {"thin": "ideal"}), ("advantages", ["one"])]:
        data = copy.deepcopy(RAW)
        data["styles"][0][field] = bad
        with pytest.raises(AssertionError):
            parse(data)
    data = copy.deepcopy(RAW)
    data["styles"].append(copy.deepcopy(data["styles"][0]))
    with pytest.raises(AssertionError):
        parse(data)


def test_every_shape_has_rules_and_reasons_are_indonesian():
    assert set(R.SHAPE_EFFECT_RULES) == set(SHAPES)
    for table in R.SHAPE_EFFECT_RULES.values():
        for pts, reason in table.values():
            assert pts in (-1, 1) and reason and "the " not in reason.lower()


# ---- filters

def test_gender_filter():
    style = BY_ID["pompadour"]
    assert passes_filters(style, UserProfile("oval", "straight", "medium", gender="pria"))
    assert not passes_filters(style, UserProfile("oval", "straight", "medium", gender="wanita"))
    assert passes_filters(style, UserProfile("oval", "straight", "medium"))


def test_hair_type_filter():
    assert not passes_filters(BY_ID["pompadour"], UserProfile("oval", "coily", "medium"))
    assert passes_filters(BY_ID["short_afro"], UserProfile("oval", "coily", "medium"))


def test_maintenance_and_bangs_filters():
    high = next(s for s in CATALOG if s.maintenance == "high")
    assert not passes_filters(high, UserProfile("oval", next(iter(high.hair_types)), "medium", max_maintenance="low"))
    bangs = next(s for s in CATALOG if s.has_bangs and R.shape_points("oval", s.effects) >= 0)
    assert not passes_filters(bangs, UserProfile("oval", next(iter(bangs.hair_types)), "medium", avoid_bangs=True))


def test_negative_face_shape_points_filtered():
    # volume_sides (-1) + volume_top (-1) for diamond: Short Afro must not be recommended to a diamond face
    assert R.shape_points("diamond", BY_ID["short_afro"].effects) < 0
    assert not passes_filters(BY_ID["short_afro"], UserProfile("diamond", "coily", "medium"))


# ---- scoring rules

def test_shape_rule_terms():
    rec = score(BY_ID["lob"], UserProfile("round", "straight", "medium"))
    assert {"bentuk_wajah:round:length_past_chin", "bentuk_wajah:round:side_part"} <= {t[0] for t in rec.terms}


def test_hair_type_ideal_vs_ok():
    style = BY_ID["pixie_cut"]
    assert style.hair_types["straight"] == "ideal" and style.hair_types["curly"] == "ok"
    ideal = score(style, UserProfile("oval", "straight", "medium"))
    ok = score(style, UserProfile("oval", "curly", "medium"))
    assert ("jenis_rambut", 2) in [(t[0], t[1]) for t in ideal.terms]
    assert "jenis_rambut" not in [t[0] for t in ok.terms]
    assert ideal.score - ok.score == R.HAIR_TYPE_IDEAL[0]


@pytest.mark.parametrize("thick", THICKNESSES)
def test_thickness_rule(thick):
    for style in CATALOG:
        terms = dict((t[0], t[1]) for t in score(style, UserProfile("oval", next(iter(style.hair_types)), thick)).terms)
        expected = {"ideal": R.THICKNESS_IDEAL[0], "avoid": R.THICKNESS_AVOID[0], "ok": None}[style.thickness[thick]]
        assert terms.get("ketebalan") == expected


def test_length_match_and_relaxation():
    res = recommend(UserProfile("oval", "straight", "medium", gender="wanita", preferred_length="long"))
    recs = res["recommendations"]
    assert len(recs) == 5
    for r in recs:
        pts = {b["rule"]: b["points"] for b in r["score_breakdown"]}["panjang"]
        assert pts == (R.LENGTH_MATCH[0] if BY_ID[r["id"]].length == "long" else R.LENGTH_ADJACENT[0])
    # fallback styles carry their length penalty in the score, and results stay in descending score order
    for prof in [UserProfile("square", "straight", "thin", gender="wanita", preferred_length="short"),
                 UserProfile("oblong", "coily", "thick", gender="pria", preferred_length="long")]:
        scores = [r["score"] for r in recommend(prof)["recommendations"]]
        assert scores == sorted(scores, reverse=True) and min(scores) >= 0
    sparse = recommend(UserProfile("oblong", "coily", "thick", gender="pria", preferred_length="long"))
    assert "long_locs" in [r["id"] for r in sparse["recommendations"]]
    assert any("panjang lain" in n for n in sparse["notes"])


def test_formal_rule():
    style = next(s for s in CATALOG if s.formal)
    p = UserProfile("oval", next(iter(style.hair_types)), "medium", formal=True)
    assert ("formal", R.FORMAL_MATCH[0]) in [(t[0], t[1]) for t in score(style, p).terms]
    informal = next(s for s in CATALOG if not s.formal)
    assert "formal" not in [t[0] for t in score(informal, UserProfile("oval", next(iter(informal.hair_types)), "medium", formal=True)).terms]


def test_secondary_shape_rule():
    style = BY_ID["lob"]  # round: +2 ; oblong: length_past_chin -1
    up = score(style, UserProfile("oval", "straight", "medium", secondary_face_shape="round"))
    down = score(style, UserProfile("oval", "straight", "medium", secondary_face_shape="oblong"))
    assert ("bentuk_wajah_kedua", 1) in [(t[0], t[1]) for t in up.terms]
    assert ("bentuk_wajah_kedua", -1) in [(t[0], t[1]) for t in down.terms]


# ---- engine output

def test_invalid_input_rejected():
    for kwargs in [dict(face_shape="triangle"), dict(hair_type="bald"), dict(thickness="huge"), dict(gender="x"),
                   dict(preferred_length="xl"), dict(max_maintenance="none"), dict(secondary_face_shape="star")]:
        base = dict(face_shape="oval", hair_type="straight", thickness="medium") | kwargs
        with pytest.raises(ValueError):
            UserProfile(**base)


def test_output_fields_score_and_determinism():
    p = UserProfile("heart", "wavy", "thin", gender="wanita")
    a, b = recommend(p), recommend(p)
    assert a == b
    assert a["face_shape_label"] == "Hati" and SCORE_NOTE in a["notes"]
    recs = a["recommendations"]
    assert [r["rank"] for r in recs] == [1, 2, 3, 4, 5]
    assert [r["score"] for r in recs] == sorted((r["score"] for r in recs), reverse=True)  # no length preference
    for r in recs:
        assert r["score"] == sum(x["points"] for x in r["score_breakdown"])
        assert r["score_unit"] == "poin" and isinstance(r["score"], int)
        assert "%" not in r["explanation"]
        assert r["reference_image"] is None and r["reference_image_note"]
        assert r["advantages"] and r["styling_tips"] and r["explanation"]


def test_diamond_note():
    assert any("Wajik" in n for n in recommend(UserProfile("diamond", "straight", "medium"))["notes"])
    assert not any("Wajik" in n for n in recommend(UserProfile("oval", "straight", "medium"))["notes"])


def test_top_k():
    assert len(recommend(UserProfile("oval", "straight", "medium"), top_k=3)["recommendations"]) == 3
    assert len(recommend(UserProfile("oval", "straight", "medium"))["recommendations"]) == 5


@pytest.mark.parametrize("shape,hair", list(itertools.product(SHAPES, HAIR_TYPES)))
def test_sanity_top5_every_shape_and_hair_type(shape, hair):
    for thick, gender in itertools.product(THICKNESSES, (None, "pria", "wanita")):
        res = recommend(UserProfile(shape, hair, thick, gender=gender))
        recs = res["recommendations"]
        assert len(recs) >= 4, (shape, hair, thick, gender)
        if len(recs) < 5:
            assert any(n.startswith("Hanya") for n in res["notes"])
        ids = [r["id"] for r in recs]
        assert len(ids) == len(set(ids))
        for r in recs:
            s = BY_ID[r["id"]]
            assert hair in s.hair_types and (gender is None or gender in s.gender)
            assert R.shape_points(shape, s.effects) >= 0


def test_formal_and_thickness_avoid_are_filters():
    for r in recommend(UserProfile("square", "straight", "thin", formal=True))["recommendations"]:
        assert BY_ID[r["id"]].formal and BY_ID[r["id"]].thickness["thin"] != "avoid"


def test_no_negative_scores_and_limiting_choice_named():
    res = recommend(UserProfile("oval", "coily", "medium", gender="wanita", max_maintenance="low"))
    assert all(r["score"] >= 0 for r in res["recommendations"])
    if len(res["recommendations"]) < 5:
        assert any("Pilihan yang paling membatasi" in n for n in res["notes"])


def test_secondary_shape_note():
    res = recommend(UserProfile("oblong", "straight", "thin", secondary_face_shape="oval"))
    assert any("skor kedua tertinggi" in n for n in res["notes"])
    assert not any("skor kedua tertinggi" in n for n in recommend(UserProfile("oblong", "straight", "thin"))["notes"])
