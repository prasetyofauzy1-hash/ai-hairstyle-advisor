# Phase 4 recommendation rules (rule_based_v1)

Code: `ml/recommendation/` (`catalog.json`, `catalog.py`, `rules.py`, `engine.py`). Every point comes from a named rule in
`rules.py`; this file lists the same rules. User-facing text (names, reasons, advantages, tips, notes) is Bahasa Indonesia.

## Input

`UserProfile(face_shape, hair_type, thickness, gender=None, preferred_length=None, max_maintenance=None, avoid_bangs=False,
formal=False, secondary_face_shape=None)`

- `face_shape`: `oval, round, square, oblong, heart, diamond` (same IDs as `ml/classification/rule_based.py`).
- `hair_type`: `straight, wavy, curly, coily`. `thickness`: `thin, medium, thick`.
- `gender`: style category `pria` / `wanita`; `None` shows every style. `preferred_length`: `very_short, short, medium, long`.
- `max_maintenance`: `low, medium, high`. `secondary_face_shape`: the classifier's second-most-likely shape, if wanted.

Invalid values raise `ValueError`.

## Step 1: filters (style removed)

| Filter | Removes a style when |
|---|---|
| Gender | `gender` is set and not in the style's `gender` list |
| Hair type | the user's hair type is not listed in the style's `hair_types` (not `ideal` or `ok`) |
| Maintenance | the style needs more upkeep than `max_maintenance` |
| Bangs | `avoid_bangs` and the style has `bangs_full` or `bangs_side` |
| Formal | `formal=True` and the style is not office-appropriate |
| Thickness | the user's thickness is `avoid` for the style |
| Negative score | the summed rule points (step 2) are below 0 |
| Face shape | the face-shape rules below sum to < 0 for this style |
| Length | with `preferred_length`, only that length is kept; if fewer than 5 styles remain, the band widens one step at a time (see the length rule) |

## Step 2: score (sum of rule points)

| Rule (`score_breakdown.rule`) | Points | When |
|---|---:|---|
| `bentuk_wajah:<shape>:<effect>` | ±1 each | per style effect in the table below |
| `jenis_rambut` | +2 | the user's hair type is `ideal` for the style (`ok` gives 0) |
| `ketebalan` | +1 | thickness is `ideal` for the style (`ok` gives 0; `avoid` is filtered out) |
| `panjang` | +1 / -1 / -2 | length equals the preference / one step away / two or more steps away (only shown when fewer than 5 exact matches) |
| `bentuk_wajah_kedua` | +1 / -1 | face-shape points for `secondary_face_shape` are > 0 / < 0 |

### Face-shape × style-effect rules

Styles carry effect tags (`volume_top`, `volume_sides`, `volume_jaw`, `soft_layers`, `textured`, `side_part`,
`bangs_full`, `bangs_side`, `length_past_chin`, `sharp_lines`, `face_exposed`, `tight_sides`). Pairs not listed score 0.

| Shape | Effect | Points | Reason shown to the user |
|---|---|---:|---|
| oval (Oval) | `face_exposed` | +1 | Wajah oval proporsional, jadi gaya yang memperlihatkan wajah menonjolkan keseimbangan itu. |
| round (Bulat) | `volume_top` | +1 | Volume di bagian atas membuat wajah bulat tampak lebih panjang. |
| round (Bulat) | `side_part` | +1 | Belahan samping memecah kesan bulat dengan garis diagonal. |
| round (Bulat) | `length_past_chin` | +1 | Panjang di bawah dagu menarik mata ke bawah sehingga wajah tampak lebih ramping. |
| round (Bulat) | `tight_sides` | +1 | Sisi yang tipis tidak menambah lebar pada pipi. |
| round (Bulat) | `volume_sides` | -1 | Volume di samping menambah lebar pada wajah bulat. |
| round (Bulat) | `bangs_full` | -1 | Poni rata memendekkan wajah bulat. |
| round (Bulat) | `volume_jaw` | -1 | Ujung yang menggembung di garis dagu menegaskan bentuk bulat. |
| square (Persegi) | `soft_layers` | +1 | Lapisan lembut melunakkan sudut rahang yang tegas. |
| square (Persegi) | `textured` | +1 | Tekstur memecah garis wajah yang kotak. |
| square (Persegi) | `side_part` | +1 | Belahan samping mengimbangi garis rahang yang lurus. |
| square (Persegi) | `bangs_side` | +1 | Poni samping menambah garis lengkung di sekitar dahi. |
| square (Persegi) | `sharp_lines` | -1 | Garis potongan yang tegas mengulang sudut rahang persegi. |
| square (Persegi) | `volume_jaw` | -1 | Ujung rambut tepat di rahang menarik perhatian ke rahang yang lebar. |
| oblong (Lonjong) | `volume_sides` | +1 | Volume di samping menambah lebar pada wajah lonjong. |
| oblong (Lonjong) | `bangs_full` | +1 | Poni rata memendekkan kesan wajah yang panjang. |
| oblong (Lonjong) | `bangs_side` | +1 | Poni samping memotong panjang wajah secara visual. |
| oblong (Lonjong) | `volume_jaw` | +1 | Volume di sekitar rahang menyeimbangkan wajah yang panjang. |
| oblong (Lonjong) | `volume_top` | -1 | Volume tinggi di atas membuat wajah lonjong tampak makin panjang. |
| oblong (Lonjong) | `length_past_chin` | -1 | Rambut panjang lurus ke bawah menambah kesan panjang. |
| oblong (Lonjong) | `tight_sides` | -1 | Sisi yang tipis membuat wajah lonjong tampak lebih sempit. |
| heart (Hati) | `volume_jaw` | +1 | Volume di sekitar dagu menyeimbangkan dahi yang lebar. |
| heart (Hati) | `bangs_side` | +1 | Poni samping mempersempit kesan dahi yang lebar. |
| heart (Hati) | `side_part` | +1 | Belahan samping mengurangi kesan dahi lebar. |
| heart (Hati) | `volume_top` | -1 | Volume di atas memperlebar bagian dahi. |
| heart (Hati) | `face_exposed` | -1 | Rambut yang disisir ke belakang memperlihatkan seluruh lebar dahi. |
| diamond (Wajik (Diamond)) | `bangs_full` | +1 | Poni menambah lebar visual pada dahi yang sempit. |
| diamond (Wajik (Diamond)) | `bangs_side` | +1 | Poni samping melembutkan tulang pipi yang menonjol. |
| diamond (Wajik (Diamond)) | `volume_jaw` | +1 | Volume di sekitar rahang menyeimbangkan tulang pipi yang lebar. |
| diamond (Wajik (Diamond)) | `volume_sides` | -1 | Volume di samping menambah lebar di tulang pipi. |
| diamond (Wajik (Diamond)) | `volume_top` | -1 | Volume di atas membuat wajah wajik tampak lebih runcing. |

## Step 3-4: rank and return

Sort by score (so shown points always descend; fallback lengths already carry their penalty), then face-shape points,
then length distance, then style `id` (so ties are deterministic but alphabetical, not meaningful).
`notes` also say when fewer than 5 styles remain and which single choice limits the result most (with the count
without it), when the ranking used the model's second-highest shape, and when no returned style has a face-shape rule.
Return the top 5 with name, description, `reference_image` (currently `null` plus a note), `score` with
`score_unit: "poin"`, `score_breakdown`, an explanation built from the rules that fired (positive reasons first,
negatives after "Perlu diperhatikan:"), advantages, styling tips, and `notes` (always includes that the score is rule
points, not a percentage or probability).

## Limitations

- The rules and weights are an initial hand-written encoding of common styling-guide advice (e.g. "add width for long
  faces, height for round faces"). They are not tuned or validated on preference data, user ratings or stylist review;
  none exists in this project yet. No recommendation-quality metric is reported.
- The score is an ordinal sum of rule points, not a compatibility percentage or probability. Equal scores are common;
  their order is alphabetical.
- Catalog metadata (hair types, thickness, effects, maintenance) was written for this project and has not been reviewed
  by a stylist. Coverage is uneven: for `pria` + `coily`, oblong and heart faces get 4 results instead of 5
  (see `research/results/phase4/sanity_review.md`).
- No reference images yet: images need clear licenses, and none were added rather than using unlicensed photos.
- Face shape comes from Phase 3, whose model covers 5 shapes (no Diamond) with test macro-F1 0.476 and has no male faces
  in training. Diamond is accepted here only as manual or rule-baseline input. A wrong face shape changes the face-shape
  terms; hair type, thickness, gender and length terms do not depend on it.
- Gender is a style category filter only; `None` shows all styles.
- ML ranking is deferred until preference or human-rating data exists (original brief).

## Sanity review

`python -m ml.recommendation.review` writes `research/results/phase4/sanity_review.md`: top 5 for each face shape ×
hair type × gender at medium thickness, and automated checks over all 216 shape × hair type × thickness × gender
profiles (filters respected, score equals breakdown, result count).
