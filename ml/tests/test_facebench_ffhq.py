import csv
import io
import json

import pytest

from ml.datasets import facebench_ffhq


def prepare(tmp_path, monkeypatch):
    def answer(idx, label, source="FFHQ", question="skin_q2"):
        return {
            "question_id": question, "question_type": "SCQ", "image_id": f"test-{source}-{idx:05d}.png",
            "gt_answer": label, "metadata": {"image_source": source},
        }

    annotations = [
        answer(1, "Diamond-shaped face"), answer(2, "Oval face"), answer(3, "Round face", "CelebA-HQ"),
        answer(4, "Square face", question="skin_q1"), answer(5, "Heart-shaped face"),
        answer(6, "Oblong face"), answer(1, "Diamond-shaped face"),
    ]
    payload = "\n\n".join(json.dumps(row) for row in annotations).encode()
    monkeypatch.setattr(facebench_ffhq.urllib.request, "urlopen", lambda *a, **kw: io.BytesIO(payload))
    excluded = tmp_path / "existing.csv"
    excluded.write_text("ffhq_id\n00002\n", encoding="utf-8")
    monkeypatch.setattr(facebench_ffhq, "EXISTING_MANIFEST", excluded)

    def meta(license_name):
        return {
            "image": {"file_md5": "published-md5"},
            "metadata": {
                "license": license_name, "license_url": "https://example.com/license", "author": "Photographer",
                "photo_url": "https://example.com/photo", "photo_title": "Portrait",
            },
        }

    metadata = tmp_path / "metadata.json"
    metadata.write_text(json.dumps({
        "1": meta("Attribution-NonCommercial License"), "5": meta("Unverified"), "6": meta("Attribution License"),
    }), encoding="utf-8")
    return metadata


def test_download_uses_only_licensed_ffhq_shape_rows_outside_existing_manifest(tmp_path, monkeypatch):
    metadata = prepare(tmp_path, monkeypatch)
    images = tmp_path / "images"
    manifest = tmp_path / "output" / "facebench.csv"
    calls = []

    def fetch(idx, md5, destination):
        calls.append((idx, md5, destination))
        return destination / f"{idx:05d}.png"

    monkeypatch.setattr(facebench_ffhq, "fetch", fetch)
    rows = facebench_ffhq.download(metadata, images, manifest)

    assert calls == [(1, "published-md5", images), (6, "published-md5", images)]
    assert [(row["ffhq_id"], row["label"]) for row in rows] == [("00001", "diamond"), ("00006", "oblong")]
    assert rows[0]["license"] == "Attribution-NonCommercial License"
    assert rows[0]["author"] == "Photographer"
    assert rows[0]["facebench_image_id"] == "test-FFHQ-00001.png"
    assert rows[0]["label_url"] == facebench_ffhq.LABELS_URL
    assert all(row["split"] == "test" for row in rows)
    with manifest.open(encoding="utf-8", newline="") as f:
        assert list(csv.DictReader(f)) == rows


def test_checksum_failure_does_not_write_manifest(tmp_path, monkeypatch):
    metadata = prepare(tmp_path, monkeypatch)
    monkeypatch.setattr(facebench_ffhq, "fetch", lambda *a: None)
    manifest = tmp_path / "facebench.csv"

    with pytest.raises(ValueError, match="failed the published MD5 check"):
        facebench_ffhq.download(metadata, tmp_path / "images", manifest)
    assert not manifest.exists()
