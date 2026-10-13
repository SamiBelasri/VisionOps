from pathlib import Path

from PIL import Image

from visionops.data.validate import validate_imagefolder


def test_validate_imagefolder_returns_class_counts(tmp_path: Path):
    for class_name in ("class_a", "class_b"):
        class_dir = tmp_path / class_name
        class_dir.mkdir()
        Image.new("RGB", (8, 8), color="white").save(class_dir / "sample.png")

    report = validate_imagefolder(tmp_path)

    assert report["valid"] is True
    assert report["num_classes"] == 2
    assert report["total_images"] == 2
    assert report["images_per_class"] == {"class_a": 1, "class_b": 1}
