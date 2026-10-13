"""Quality checks for ImageFolder-compatible datasets."""

from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def validate_imagefolder(root: str | Path) -> dict[str, Any]:
    """Validate an ImageFolder directory and return a serializable report."""

    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {root}")

    classes = sorted(path for path in root.iterdir() if path.is_dir())
    if not classes:
        raise ValueError(f"No class directories found in: {root}")

    counts: dict[str, int] = {}
    invalid_files: list[str] = []
    total_images = 0

    for class_dir in classes:
        valid_count = 0
        for path in sorted(class_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            try:
                with Image.open(path) as image:
                    image.verify()
                valid_count += 1
            except (UnidentifiedImageError, OSError):
                invalid_files.append(str(path))
        counts[class_dir.name] = valid_count
        total_images += valid_count

    return {
        "root": str(root),
        "num_classes": len(classes),
        "classes": [path.name for path in classes],
        "images_per_class": counts,
        "total_images": total_images,
        "invalid_files": invalid_files,
        "valid": not invalid_files and total_images > 0,
    }
