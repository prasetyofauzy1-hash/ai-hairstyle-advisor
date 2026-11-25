import csv
import json
from pathlib import Path

import pytest

from ml.classification.rule_based import SHAPES
from ml.datasets import labeling_tool
from ml.datasets.agreement import UNSURE
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


def test_ai_pseudo_labels_cover_only_the_training_split():
    split = read_rows(SUBSET / "split.csv")
    train_ids = [r["ffhq_id"] for r in split if r["split"] == "train"]
    test_ids = {r["ffhq_id"] for r in split if r["split"] == "test"}
    labels = read_rows(SUBSET / "ai_pseudo_labels_train.csv")
    labeled_ids = [r["ffhq_id"] for r in labels]

    assert len(labeled_ids) == len(set(labeled_ids)) == 496
    assert labeled_ids == train_ids
    assert test_ids.isdisjoint(labeled_ids)
    assert {r["label"] for r in labels} <= {*SHAPES, UNSURE}
    assert {r["source"] for r in labels} == {"ai_pseudo_label"}
    assert {r["annotator"] for r in labels} == {"devin-ai-vision"}


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


def test_human_test_labels_cover_only_the_test_split():
    split = read_rows(SUBSET / "split.csv")
    test_ids = [r["ffhq_id"] for r in split if r["split"] == "test"]
    labels = read_rows(SUBSET / "human_labels_test.csv")
    assert [r["ffhq_id"] for r in labels] == test_ids
    assert {r["ffhq_id"] for r in labels}.isdisjoint(r["ffhq_id"] for r in read_rows(SUBSET / "ai_pseudo_labels_train.csv"))
    assert {r["label"] for r in labels} <= {*SHAPES, UNSURE}
    assert {r["source"] for r in labels} == {"human"}
    assert {r["annotator"] for r in labels} == {"Uzy"}


def test_ffhq_feature_items_keep_human_and_ai_labels_apart():
    from ml.datasets.ffhq_features import labeled_items

    items = labeled_items()
    by_source = {s: [it for it in items if it["label_source"] == s] for s in ("human", "ai_pseudo_label")}
    assert len(by_source["human"]) == 104 and {it["split"] for it in by_source["human"]} == {"test"}
    assert len(by_source["ai_pseudo_label"]) == 364 and {it["split"] for it in by_source["ai_pseudo_label"]} == {"train"}
    assert UNSURE not in {it["label"] for it in items}
