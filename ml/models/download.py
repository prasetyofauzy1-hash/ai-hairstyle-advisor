"""Download and verify the pretrained model files (MediaPipe, OpenCV Zoo) used by the pipeline."""

from __future__ import annotations

import hashlib
import os
import urllib.request
from pathlib import Path

MODELS = {
    "face_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
        "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff",
    ),
    "hair_segmenter.tflite": (
        "https://storage.googleapis.com/mediapipe-models/image_segmenter/hair_segmenter/float32/latest/hair_segmenter.tflite",
        "2628cf3ce5f695f604cbea2841e00befcaa3624bf80caf3664bef2656d59bf84",
    ),
    # OpenCV Zoo SFace (Apache-2.0): face embeddings, used only to group same-person photos for leakage-free splits.
    "face_recognition_sface_2021dec.onnx": (
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
    ),
}


def model_dir() -> Path:
    return Path(os.environ.get("HAIRSTYLE_MODEL_DIR", Path.home() / ".cache" / "hairstyle-advisor" / "models"))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_model(name: str, directory: Path | None = None) -> Path:
    url, sha256 = MODELS[name]
    directory = directory or model_dir()
    path = directory / name
    if path.exists() and _sha256(path) == sha256:
        return path
    directory.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    if _sha256(tmp) != sha256:
        tmp.unlink()
        raise RuntimeError(
            f"Checksum mismatch for {name}; the upstream 'latest' model may have changed. "
            "Verify the new file and update MODELS."
        )
    tmp.replace(path)
    return path


if __name__ == "__main__":
    for model_name in MODELS:
        print(ensure_model(model_name))
