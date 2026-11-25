"""CLI: analyze portraits and write landmark/measurement overlays plus a JSON summary.

    python -m ml.analyze ml/datasets/samples/*.jpg --out research/results/phase2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from ml.evaluation.visualization import render_face_report
from ml.pipeline import FaceAnalyzer
from ml.preprocessing.image_io import load_image


def _round(obj):
    if isinstance(obj, float):
        return round(obj, 4)
    if isinstance(obj, dict):
        return {k: _round(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round(v) for v in obj]
    return obj


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    summary = {}
    with FaceAnalyzer() as analyzer:
        for path in args.images:
            image = load_image(path)
            result = analyzer.analyze(image)
            summary[path.name] = _round(result.to_dict())
            if result.face is None:
                continue
            if result.prediction:
                p = result.prediction
                lines = [f"{p.label_id} (skor {p.scores[p.shape]:.2f}, belum dikalibrasi)"]
            else:
                lines = ["Ditolak: " + ", ".join(i.code for i in result.quality.issues if i.blocking)]
            if result.quality.issues and result.prediction:
                lines.append("Catatan: " + ", ".join(i.code for i in result.quality.issues))
            vis = render_face_report(image, result.face.landmarks, lines)
            cv2.imwrite(str(args.out / f"{path.stem}_analysis.jpg"), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
