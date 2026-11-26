import pytest

from ml.evaluation import gender_estimates


@pytest.mark.parametrize("exists", [False, True], ids=["missing", "unreadable"])
def test_gender_estimation_rejects_unreadable_photo(tmp_path, exists):
    photo = tmp_path / "image.jpg"
    if exists:
        photo.write_bytes(b"not an image")
    with pytest.raises(ValueError, match="Cannot read photo for gender estimation:.*image.jpg"):
        gender_estimates._estimate(photo)
