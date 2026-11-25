from pathlib import Path

import pytest

from ml.preprocessing.image_io import load_image

SAMPLES = Path(__file__).resolve().parents[1] / "datasets" / "samples"


@pytest.fixture(scope="session")
def detector():
    from ml.feature_extraction.landmarks import FaceLandmarkDetector
    with FaceLandmarkDetector() as d:
        yield d


@pytest.fixture(scope="session")
def analyzer():
    from ml.pipeline import FaceAnalyzer
    with FaceAnalyzer() as a:
        yield a


@pytest.fixture(scope="session")
def sample_paths():
    return sorted(SAMPLES.glob("*.jpg"))


@pytest.fixture(scope="session")
def frontal_image():
    return load_image(SAMPLES / "jonny_kim.jpg")
