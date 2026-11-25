import csv
import json
from pathlib import Path

import pytest

from ml.datasets import labeling_tool
from ml.datasets.ffhq_split import write_split

SUBSET = Path(__file__).resolve().parents[1] / "datasets" / "ffhq_labeling"


def read_rows(path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def test_committed_ffhq_split_matches_first_104_manifest_rows(tmp_path):
    regenerated = tmp_path / "split.csv"
    write_split(SUBSET / "manifest.csv", regenerated)
    assert regenerated.read_bytes() == (SUBSET / "split.csv").read_bytes()

    split = read_rows(regenerated)
    manifest_ids = [r["ffhq_id"] for r in read_rows(SUBSET / "manifest.csv")]
    assert len(manifest_ids) == len(set(manifest_ids)) == 600
    assert [r["ffhq_id"] for r in split] == manifest_ids
    assert [r["split"] for r in split] == ["test"] * 104 + ["train"] * 496


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

    assert len(items(full)) == 600
    assert items(test) == items(full)[:104]
    assert "<title>Pelabelan Bentuk Wajah — Set Uji (104 Foto)</title>" in test
    assert 'const KEY="faceshape-labels";' in full
    assert 'const KEY="faceshape-labels-test-' in test
    assert 'a.download=`labels_${name()}.csv`' in test


def test_labeling_page_rejects_incomplete_split(tmp_path):
    split = tmp_path / "split.csv"
    split.write_text("ffhq_id,split\n40566,test\n23628,train\n", encoding="utf-8")
    with pytest.raises(ValueError, match="each manifest ID exactly once"):
        labeling_tool.build_page(SUBSET / "manifest.csv", tmp_path, test_split=split)
