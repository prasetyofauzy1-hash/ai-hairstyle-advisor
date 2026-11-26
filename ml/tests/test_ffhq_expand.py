import csv
from types import SimpleNamespace

import numpy as np
import pytest

from ml.datasets import ffhq_expand
from ml.datasets.ffhq_expand import identity_flags
from ml.feature_extraction.geometry import FEATURE_NAMES
from ml.training.train_face_shape_image import expansion_cohort


def test_reference_bundle_keeps_hashes_and_indices_when_decoding_fails(tmp_path, monkeypatch, capsys):
    corrupt = tmp_path / "corrupt.jpg"
    corrupt.write_bytes(b"")
    other = tmp_path / "other.jpg"
    other.write_bytes(b"other")
    paths = tmp_path / "photos.txt"
    paths.write_text(f"{corrupt}\n{other}\n")
    archive = tmp_path / "references.npz"
    monkeypatch.setattr(ffhq_expand, "analyze_items", lambda *_: ([], np.zeros((2, 128))))
    monkeypatch.setattr(ffhq_expand, "dlib_embeddings",
                        lambda photos, _: [ffhq_expand.dlib_faces(photos[0]), [np.ones(128)]])

    ffhq_expand.reference_bundle(SimpleNamespace(photos=paths, references=archive, workers=1))

    with np.load(archive) as data:
        assert data["sha256"].tolist() == [ffhq_expand.kernel.digest(p) for p in (corrupt, other)]
        assert data["dlib_index"].tolist() == [1]
    assert "1 unreadable reference photos" in capsys.readouterr().out


def test_identity_screen_rejects_joint_and_single_strong_matches():
    sface = np.array([[1, 0]], dtype=np.float32)
    dlib = np.array([[0, 0]], dtype=np.float32)
    refs = np.array([[0.4, 0], [0.4, 0], [0.7, 0], [np.nan, np.nan], [0.2, 0]], dtype=np.float32)
    faces = np.array([[0.58, 0], [0.65, 0], [0.8, 0], [0.5, 0], [0.2, 0]], dtype=np.float32)

    rejected = identity_flags(sface, dlib, refs, faces, np.array([0, 1, 2, 3, 4]))

    assert rejected.tolist() == [[True, False, True, True, True]]


def test_expansion_uses_fresh_identity_groups_and_rejects_facebench(tmp_path):
    features = tmp_path / "training.csv"
    facebench = tmp_path / "facebench.csv"
    facebench.write_text("ffhq_id\n69999\n")
    row = {"image": "69998.png", "ffhq_id": "69998", "source_split": "train",
           "label_source": "ai_pseudo_label", "label": "diamond", "passed": True,
           **dict.fromkeys(FEATURE_NAMES, 1.0)}
    with features.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)

    assert expansion_cohort(features, facebench, first_group=200)["identity_group"].tolist() == [200]
    facebench.write_text("ffhq_id\n69998\n")
    with pytest.raises(ValueError, match="training-only FFHQ pseudo-labels"):
        expansion_cohort(features, facebench, first_group=200)
