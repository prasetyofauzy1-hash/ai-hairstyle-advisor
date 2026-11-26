import re
from pathlib import Path

from ml.recommendation.catalog import load_catalog

SPECS = Path(__file__).resolve().parents[2] / "frontend/web/src/components/style-illustration.tsx"


def test_every_catalog_style_has_an_illustration():
    ids = set(re.findall(r"^\s+(\w+): s\(", SPECS.read_text(), re.M))
    assert {st.id for st in load_catalog()} == ids
