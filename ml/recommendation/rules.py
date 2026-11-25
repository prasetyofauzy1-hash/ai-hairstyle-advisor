"""Transparent scoring rules. Every point a hairstyle gets comes from one named rule below; docs/recommendation-rules.md
lists the same tables. Weights are initial choices, not tuned on any preference or rating data (none exists yet).
"""

from __future__ import annotations

# (face shape, style effect) -> (points, Indonesian reason). Missing pairs score 0.
SHAPE_EFFECT_RULES: dict[str, dict[str, tuple[int, str]]] = {
    "oval": {
        "face_exposed": (1, "Wajah oval cenderung seimbang, jadi gaya yang memperlihatkan wajah menonjolkan keseimbangan itu."),
    },
    "round": {
        "volume_top": (1, "Volume di bagian atas membuat wajah bulat tampak lebih panjang."),
        "side_part": (1, "Belahan samping memberi garis miring yang membuat wajah tampak tidak terlalu bulat."),
        "length_past_chin": (1, "Rambut yang melewati dagu bisa membuat wajah tampak lebih ramping."),
        "tight_sides": (1, "Rambut samping yang tipis tidak membuat pipi tampak lebih lebar."),
        "volume_sides": (-1, "Volume di samping bisa membuat wajah bulat tampak lebih lebar."),
        "bangs_full": (-1, "Poni rata bisa membuat wajah bulat tampak lebih pendek."),
        "volume_jaw": (-1, "Ujung rambut yang mengembang di garis dagu bisa membuat wajah tampak lebih bulat."),
    },
    "square": {
        "soft_layers": (1, "Potongan berlapis membuat garis rahang tampak lebih lembut."),
        "textured": (1, "Rambut bertekstur membuat garis wajah tampak tidak terlalu kaku."),
        "side_part": (1, "Belahan samping bisa mengimbangi garis rahang yang lurus."),
        "bangs_side": (1, "Poni samping memberi garis melengkung di sekitar dahi."),
        "sharp_lines": (-1, "Garis potongan yang tegas bisa membuat rahang tampak makin bersudut."),
        "volume_jaw": (-1, "Ujung rambut tepat di rahang bisa membuat rahang tampak lebih lebar."),
    },
    "oblong": {
        "volume_sides": (1, "Volume di samping bisa membuat wajah lonjong tampak lebih lebar."),
        "bangs_full": (1, "Poni rata bisa membuat wajah terlihat tidak terlalu panjang."),
        "bangs_side": (1, "Poni samping bisa membuat wajah terlihat tidak terlalu panjang."),
        "volume_jaw": (1, "Volume di sekitar rahang bisa membuat wajah panjang tampak lebih seimbang."),
        "volume_top": (-1, "Volume tinggi di atas bisa membuat wajah lonjong tampak makin panjang."),
        "length_past_chin": (-1, "Rambut yang jatuh panjang melewati dagu bisa membuat wajah tampak lebih panjang."),
        "tight_sides": (-1, "Rambut samping yang tipis bisa membuat wajah lonjong tampak lebih sempit."),
    },
    "heart": {
        "volume_jaw": (1, "Volume di sekitar dagu bisa mengimbangi dahi yang lebar."),
        "bangs_side": (1, "Poni samping bisa membuat dahi terlihat tidak terlalu lebar."),
        "side_part": (1, "Belahan samping bisa membuat dahi tampak tidak terlalu lebar."),
        "volume_top": (-1, "Volume di atas bisa membuat dahi tampak lebih lebar."),
        "face_exposed": (-1, "Rambut yang disisir ke belakang memperlihatkan seluruh lebar dahi."),
    },
    "diamond": {
        "bangs_full": (1, "Poni bisa membuat dahi terlihat lebih lebar."),
        "bangs_side": (1, "Poni samping bisa membuat tulang pipi yang menonjol tampak lebih lembut."),
        "volume_jaw": (1, "Volume di sekitar rahang bisa mengimbangi tulang pipi yang lebar."),
        "volume_sides": (-1, "Volume di samping bisa membuat tulang pipi tampak lebih lebar."),
        "volume_top": (-1, "Volume di atas bisa membuat wajah wajik tampak lebih runcing."),
    },
}

HAIR_TYPE_IDEAL = (2, "Panduan ini menyarankan potongan ini untuk rambut {hair}.")
THICKNESS_IDEAL = (1, "Panduan ini menyarankan potongan ini untuk rambut dengan ketebalan {thick}.")
THICKNESS_AVOID = (-2, "Kurang disarankan untuk rambut dengan ketebalan {thick}; hasilnya bisa kurang bervolume atau sulit diatur.")
LENGTH_MATCH = (1, "Panjangnya sesuai dengan yang Anda inginkan ({length}).")
LENGTH_ADJACENT = (-1, "Gaya ini sedikit lebih pendek atau panjang dari pilihan Anda ({length}). Kami menampilkannya karena pilihan dengan panjang tersebut terbatas.")
LENGTH_OTHER = (-2, "Panjang gaya ini berbeda dari pilihan Anda ({length}). Kami menampilkannya karena pilihan dengan panjang tersebut terbatas.")
SECONDARY_SHAPE_FIT = (1, "Gaya ini juga disarankan untuk wajah {shape}, bentuk dengan skor kedua tertinggi.")
SECONDARY_SHAPE_MISFIT = (-1, "Gaya ini kurang disarankan untuk wajah {shape}, bentuk dengan skor kedua tertinggi.")


def shape_terms(shape: str, effects) -> list[tuple[str, int, str]]:
    """[(rule name, points, reason)] for each style effect that has a rule for this face shape."""
    table = SHAPE_EFFECT_RULES[shape]
    return [(f"bentuk_wajah:{shape}:{e}", *table[e]) for e in effects if e in table]


def shape_points(shape: str, effects) -> int:
    return sum(p for _, p, _ in shape_terms(shape, effects))
