"""Dataset construction for the classification MVP."""

from pathlib import Path
from typing import Any

from .registry import get_dataset_spec


def build_dataset(
    name: str,
    root: str | Path,
    *,
    train: bool = True,
    image_size: int = 224,
    download: bool = False,
) -> Any:
    """Build a supported classification dataset.

    Intel Image Classification is expected in ImageFolder-compatible layout.
    EuroSAT uses torchvision's built-in dataset and is not downloaded unless
    ``download=True`` is explicitly provided.
    """

    spec = get_dataset_spec(name)
    root = Path(root)

    try:
        from torchvision import datasets, transforms
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "torchvision is required to build VisionOps datasets. "
            "Install it with: python -m pip install torchvision"
        ) from exc

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )

    if spec.name == "intel":
        split = "seg_train" if train else "seg_test"
        # Accept both the extracted Kaggle layout (root/seg_train/seg_train)
        # and a direct ImageFolder split supplied by the caller.
        split_root = root / split
        if not split_root.exists() and any(path.is_dir() for path in root.glob("*")):
            split_root = root
        if not split_root.exists():
            raise FileNotFoundError(
                f"Intel dataset split not found: {split_root}. "
                "Expected ImageFolder-compatible directories."
            )
        return datasets.ImageFolder(split_root, transform=transform)

    if spec.name == "eurosat":
        # EuroSAT has no official train/test split in torchvision. The full
        # dataset is returned here; deterministic splitting is a separate step.
        return datasets.EuroSAT(root=root, download=download, transform=transform)

    raise ValueError(f"Dataset '{name}' is registered but has no loader.")
