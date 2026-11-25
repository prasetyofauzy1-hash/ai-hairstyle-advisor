import json
import logging

import numpy as np

from ml.evaluation.visualization import render_face_report


def test_all_samples_produce_landmarks_and_valid_output(analyzer, sample_paths):
    from ml.preprocessing.image_io import load_image
    assert len(sample_paths) >= 5
    for path in sample_paths:
        result = analyzer.analyze(load_image(path))
        assert result.face is not None, path.name
        assert result.face.landmarks.shape == (478, 3)
        d = result.to_dict()
        json.dumps(d)  # serialisable for the API
        if d["quality"]["passed"]:
            assert d["prediction"]["shape"] in d["prediction"]["scores"]
        else:
            assert d["prediction"] is None


def test_rejected_image_has_no_prediction(analyzer):
    result = analyzer.analyze(np.full((512, 512, 3), 128, np.uint8))
    assert not result.quality.passed and result.prediction is None


def test_logs_contain_no_image_data(analyzer, frontal_image, caplog):
    with caplog.at_level(logging.DEBUG):
        analyzer.analyze(frontal_image)
    text = " ".join(r.getMessage() for r in caplog.records)
    assert len(text) < 500 and "[[" not in text and ".jpg" not in text


def test_visualization_renders(analyzer, frontal_image):
    result = analyzer.analyze(frontal_image)
    vis = render_face_report(frontal_image, result.face.landmarks, ["Uji"])
    assert vis.shape[1] == 640 and vis.dtype == np.uint8
