"""Integration tests: real MediaPipe pipeline, real classifier, real database (Postgres in CI, SQLite locally)."""

import io
import logging
from pathlib import Path

import cv2
import numpy as np

from app.config import settings

SAMPLE = Path(__file__).resolve().parents[2] / "ml/datasets/samples/mae_jemison.jpg"


def upload(client, consent, data=None, name="face.jpg"):
    data = data if data is not None else SAMPLE.read_bytes()
    return client.post("/api/analyses", files={"image": (name, data, "image/jpeg")}, data={"consent": str(consent).lower()})


def stored_files():
    return list(Path(settings.image_storage_dir).glob("*"))


def test_health_and_openapi(client):
    assert client.get("/api/health").json() == {"status": "ok"}
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/api/analyses", "/api/analyses/{analysis_id}", "/api/analyses/{analysis_id}/recommendations",
            "/api/hairstyles"} <= set(paths)


def test_hairstyles_synced_from_catalog(client):
    from ml.recommendation.catalog import load_catalog
    assert len(client.get("/api/hairstyles").json()) == len(load_catalog())
    assert all("pria" in h["gender_category"] for h in client.get("/api/hairstyles?gender=pria").json())


def test_no_consent_stores_no_image(client):
    before = len(stored_files())
    r = upload(client, consent=False)
    assert r.status_code == 201
    body = r.json()
    assert body["image_stored"] is False and body["consent_given_at"] is None
    assert len(stored_files()) == before
    h = {"X-Analysis-Token": body["access_token"]}
    assert client.get(f"/api/analyses/{body['id']}/image", headers=h).status_code == 404


def test_full_flow_with_consent_and_deletion(client):
    r = upload(client, consent=True)
    body = r.json()
    assert r.status_code == 201 and body["image_stored"] and body["quality"]["passed"]
    pred = body["prediction"]
    assert pred["calibrated"] is False and pred["shape"] in pred["scores"] and "diamond" not in pred["scores"]
    assert abs(sum(pred["scores"].values()) - 1) < 1e-6
    assert body["limitations"]
    aid, h = body["id"], {"X-Analysis-Token": body["access_token"]}

    img = client.get(f"/api/analyses/{aid}/image", headers=h)
    assert img.status_code == 200 and img.content[:3] == b"\xff\xd8\xff"
    assert b"Exif" not in img.content[:64]

    rec = client.post(f"/api/analyses/{aid}/recommendations", headers=h,
                      json={"hair_type": "wavy", "thickness": "medium", "gender": "wanita"}).json()
    assert rec["face_shape"] == pred["shape"] and rec["face_shape_source"] == "model"
    assert len(rec["recommendations"]) == 5
    assert all(r["score"] == sum(t["points"] for t in r["score_breakdown"]) for r in rec["recommendations"])

    manual = client.post(f"/api/analyses/{aid}/recommendations", headers=h,
                         json={"hair_type": "wavy", "thickness": "medium", "face_shape": "diamond"}).json()
    assert manual["face_shape_source"] == "user" and any("Wajik" in n for n in manual["notes"])

    assert client.delete(f"/api/analyses/{aid}/image", headers=h).status_code == 204
    assert client.get(f"/api/analyses/{aid}", headers=h).json()["image_stored"] is False
    assert client.delete(f"/api/analyses/{aid}", headers=h).status_code == 204
    assert client.get(f"/api/analyses/{aid}", headers=h).status_code == 404


def test_delete_removes_file(client):
    body = upload(client, consent=True).json()
    before = len(stored_files())
    assert client.delete(f"/api/analyses/{body['id']}", headers={"X-Analysis-Token": body["access_token"]}).status_code == 204
    assert len(stored_files()) == before - 1


def test_token_required(client):
    body = upload(client, consent=False).json()
    assert client.get(f"/api/analyses/{body['id']}", headers={"X-Analysis-Token": "x"}).status_code == 404
    assert client.get(f"/api/analyses/{body['id']}").status_code == 422
    assert client.delete(f"/api/analyses/{body['id']}", headers={"X-Analysis-Token": "x"}).status_code == 404


def test_rejects_bad_uploads(client):
    assert upload(client, False, data=b"not an image").status_code == 415
    assert upload(client, False, data=b"\xff\xd8\xff" + b"0" * 100).status_code == 422
    assert upload(client, False, data=b"\xff\xd8\xff" + b"0" * (settings.max_upload_bytes + 1)).status_code == 413


def test_upload_kept_in_memory(client, monkeypatch):
    import tempfile
    spilled = []
    real = tempfile.SpooledTemporaryFile.rollover
    monkeypatch.setattr(tempfile.SpooledTemporaryFile, "rollover", lambda self: (spilled.append(1), real(self))[1])
    big = cv2.imencode(".png", np.random.default_rng(0).integers(0, 255, (900, 900, 3), np.uint8))[1].tobytes()
    assert 1024 * 1024 < len(big) < settings.max_upload_bytes
    assert upload(client, False, data=big, name="big.png").status_code == 201
    assert not spilled


def test_rejects_huge_pixel_count(client):
    from PIL import Image
    buf = io.BytesIO()
    Image.new("L", (12000, 12000)).save(buf, "PNG")
    assert len(buf.getvalue()) < settings.max_upload_bytes
    assert upload(client, False, data=buf.getvalue(), name="huge.png").status_code == 413


def test_quality_failure_has_no_prediction(client):
    blank = cv2.imencode(".jpg", np.full((480, 480, 3), 200, np.uint8))[1].tobytes()
    body = upload(client, consent=True, data=blank).json()
    assert body["quality"]["passed"] is False and body["prediction"] is None
    assert body["image_stored"] is False and not any(Path(settings.image_storage_dir).glob("*"))
    assert any(i["blocking"] for i in body["quality"]["issues"])
    r = client.post(f"/api/analyses/{body['id']}/recommendations", headers={"X-Analysis-Token": body["access_token"]},
                    json={"hair_type": "wavy", "thickness": "medium"})
    assert r.status_code == 422


def test_logs_never_contain_image_data(client, caplog):
    from app.logging_filter import NoFaceDataFilter
    caplog.handler.addFilter(NoFaceDataFilter())
    with caplog.at_level(logging.DEBUG):
        body = upload(client, consent=True).json()
        logging.getLogger("app.test").info("bytes %s", SAMPLE.read_bytes()[:50])
        logging.getLogger("app.test").info("b64 %s", "A" * 300)
        logging.getLogger("app.test").info("path /x/%s.jpg", body["id"])
    text = caplog.text
    assert "AAAA" * 10 not in text and body["id"] + ".jpg" not in text and "\\xff\\xd8" not in text
    stored = stored_files()
    assert all(p.name not in text for p in stored)


def test_chunked_oversized_upload_gets_413(client):
    boundary = "xyz"
    head = (f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="a.jpg"\r\n'
            "Content-Type: image/jpeg\r\n\r\n").encode()

    def body():
        yield head
        for _ in range(settings.max_upload_bytes // 65536 + 4):
            yield b"\xff" * 65536
        yield f"\r\n--{boundary}--\r\n".encode()

    r = client.post("/api/analyses", content=body(), headers={"content-type": f"multipart/form-data; boundary={boundary}"})
    assert r.status_code == 413 and "MB" in r.json()["detail"]
