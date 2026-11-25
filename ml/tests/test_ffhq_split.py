import csv
import json
from collections import Counter
from pathlib import Path

import pytest

from ml.datasets import labeling_tool
from ml.datasets.ffhq_split import write_split

SUBSET = Path(__file__).resolve().parents[1] / "datasets" / "ffhq_labeling"


def read_rows(path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def test_committed_ffhq_split_is_reproducible_and_stratified(tmp_path):
    regenerated = tmp_path / "split.csv"
    write_split(SUBSET / "manifest.csv", regenerated)
    assert regenerated.read_bytes() == (SUBSET / "split.csv").read_bytes()

    split = read_rows(regenerated)
    licenses = {r["ffhq_id"]: r["license"] for r in read_rows(SUBSET / "manifest.csv")}
    assert len(split) == len(licenses) == len({r["ffhq_id"] for r in split}) == 600
    assert {r["ffhq_id"] for r in split} == set(licenses)
    assert Counter(r["split"] for r in split) == {"test": 200, "train": 400}
    assert Counter(licenses[r["ffhq_id"]] for r in split if r["split"] == "test") == {
        "Attribution License": 174,
        "Public Domain Mark": 21,
        "Public Domain Dedication (CC0)": 4,
        "United States Government Work": 1,
    }


def test_labeling_page_only_loads_test_images_and_isolates_storage(monkeypatch):
    def fake_thumbnail(path):
        return path.name

    monkeypatch.setattr(labeling_tool, "thumbnail_data_uri", fake_thumbnail)
    manifest = SUBSET / "manifest.csv"
    split = SUBSET / "split.csv"
    full = labeling_tool.build_page(manifest, SUBSET)
    test = labeling_tool.build_page(manifest, SUBSET, test_split=split)

    def items(page):
        return json.loads(page.split("const ITEMS=", 1)[1].split(", CHOICES=", 1)[0])

    test_ids = {r["ffhq_id"] for r in read_rows(split) if r["split"] == "test"}
    assert len(items(full)) == 600
    assert items(test) == [item for item in items(full) if item["id"] in test_ids]
    assert "<title>Pelabelan Bentuk Wajah — Set Uji (200 Foto)</title>" in test
    assert 'const KEY="faceshape-labels";' in full
    assert 'const KEY="faceshape-labels-test-' in test
    assert 'a.download=`labels_${name()}.csv`' in test


def test_labeling_page_rejects_incomplete_split(tmp_path):
    split = tmp_path / "split.csv"
    split.write_text("ffhq_id,split\n40566,test\n23628,train\n", encoding="utf-8")
    with pytest.raises(ValueError, match="each manifest ID exactly once"):
        labeling_tool.build_page(SUBSET / "manifest.csv", tmp_path, test_split=split)
