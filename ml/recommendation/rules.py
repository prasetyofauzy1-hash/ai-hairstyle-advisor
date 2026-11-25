"""Transparent scoring rules. Every point a hairstyle gets comes from one named rule below; docs/recommendation-rules.md
lists the same tables. Weights are initial choices, not tuned on any preference or rating data (none exists yet).
"""

from __future__ import annotations

# (face shape, style effect) -> (points, Indonesian reason). Missing pairs score 0.
SHAPE_EFFECT_RULES: dict[str, dict[str, tuple[int, str]]] = {
    "oval": {
        "face_exposed": (1, "Wajah oval proporsional, jadi gaya yang memperlihatkan wajah menonjolkan keseimbangan itu."),
    },
    "round": {
        "volume_top": (1, "Volume di bagian atas membuat wajah bulat tampak lebih panjang."),
        "side_part": (1, "Belahan samping memecah kesan bulat dengan garis diagonal."),
        "length_past_chin": (1, "Panjang di bawah dagu menarik mata ke bawah sehingga wajah tampak lebih ramping."),
        "tight_sides": (1, "Sisi yang tipis tidak menambah lebar pada pipi."),
        "volume_sides": (-1, "Volume di samping menambah lebar pada wajah bulat."),
        "bangs_full": (-1, "Poni rata memendekkan wajah bulat."),
        "volume_jaw": (-1, "Ujung yang menggembung di garis dagu menegaskan bentuk bulat."),
    },
    "square": {
        "soft_layers": (1, "Lapisan lembut melunakkan sudut rahang yang tegas."),
        "textured": (1, "Tekstur memecah garis wajah yang kotak."),
        "side_part": (1, "Belahan samping mengimbangi garis rahang yang lurus."),
        "bangs_side": (1, "Poni samping menambah garis lengkung di sekitar dahi."),
        "sharp_lines": (-1, "Garis potongan yang tegas mengulang sudut rahang persegi."),
        "volume_jaw": (-1, "Ujung rambut tepat di rahang menarik perhatian ke rahang yang lebar."),
    },
    "oblong": {
        "volume_sides": (1, "Volume di samping menambah lebar pada wajah lonjong."),
        "bangs_full": (1, "Poni rata memendekkan kesan wajah yang panjang."),
        "bangs_side": (1, "Poni samping memotong panjang wajah secara visual."),
        "volume_jaw": (1, "Volume di sekitar rahang menyeimbangkan wajah yang panjang."),
        "volume_top": (-1, "Volume tinggi di atas membuat wajah lonjong tampak makin panjang."),
        "length_past_chin": (-1, "Rambut panjang lurus ke bawah menambah kesan panjang."),
        "tight_sides": (-1, "Sisi yang tipis membuat wajah lonjong tampak lebih sempit."),
    },
    "heart": {
        "volume_jaw": (1, "Volume di sekitar dagu menyeimbangkan dahi yang lebar."),
        "bangs_side": (1, "Poni samping mempersempit kesan dahi yang lebar."),
        "side_part": (1, "Belahan samping mengurangi kesan dahi lebar."),
        "volume_top": (-1, "Volume di atas memperlebar bagian dahi."),
        "face_exposed": (-1, "Rambut yang disisir ke belakang memperlihatkan seluruh lebar dahi."),
    },
    "diamond": {
        "bangs_full": (1, "Poni menambah lebar visual pada dahi yang sempit."),
        "bangs_side": (1, "Poni samping melembutkan tulang pipi yang menonjol."),
        "volume_jaw": (1, "Volume di sekitar rahang menyeimbangkan tulang pipi yang lebar."),
        "volume_sides": (-1, "Volume di samping menambah lebar di tulang pipi."),
        "volume_top": (-1, "Volume di atas membuat wajah wajik tampak lebih runcing."),
    },
}

HAIR_TYPE_IDEAL = (2, "Potongan ini paling cocok untuk rambut {hair}.")
THICKNESS_IDEAL = (1, "Potongan ini cocok untuk rambut {thick}.")
THICKNESS_AVOID = (-2, "Kurang ideal untuk rambut {thick}; hasilnya bisa kurang bervolume atau sulit diatur.")
LENGTH_MATCH = (1, "Panjangnya sesuai dengan yang Anda inginkan ({length}).")
LENGTH_ADJACENT = (-1, "Panjangnya satu tingkat dari pilihan Anda ({length}), ditampilkan karena pilihan yang sama terlalu sedikit.")
LENGTH_OTHER = (-2, "Panjangnya berbeda dari pilihan Anda ({length}), ditampilkan karena gaya yang cocok terlalu sedikit.")
FORMAL_MATCH = (1, "Cocok untuk suasana kantor atau acara formal.")
SECONDARY_SHAPE_FIT = (1, "Juga cocok untuk bentuk wajah {shape}, kemungkinan kedua dari analisis.")
SECONDARY_SHAPE_MISFIT = (-1, "Kurang cocok untuk bentuk wajah {shape}, kemungkinan kedua dari analisis.")


def shape_terms(shape: str, effects) -> list[tuple[str, int, str]]:
    """[(rule name, points, reason)] for each style effect that has a rule for this face shape."""
    table = SHAPE_EFFECT_RULES[shape]
    return [(f"bentuk_wajah:{shape}:{e}", *table[e]) for e in effects if e in table]


def shape_points(shape: str, effects) -> int:
    return sum(p for _, p, _ in shape_terms(shape, effects))
