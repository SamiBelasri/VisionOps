"""Dataset registry for the classification pipeline.

The registry deliberately separates dataset selection from training logic.
Dataset download and parsing will be added once the contracts are tested.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetSpec:
    """Metadata required by the classification pipeline."""

    name: str
    num_classes: int
    description: str


DATASETS: dict[str, DatasetSpec] = {
    "eurosat": DatasetSpec(
        name="eurosat",
        num_classes=10,
        description="Satellite image classification dataset.",
    ),
    "intel": DatasetSpec(
        name="intel",
        num_classes=6,
        description="Natural scene image classification dataset.",
    ),
}


def get_dataset_spec(name: str) -> DatasetSpec:
    """Return a registered dataset specification.

    Raises:
        ValueError: if ``name`` is not supported.
    """

    key = name.strip().lower()
    try:
        return DATASETS[key]
    except KeyError as exc:
        supported = ", ".join(sorted(DATASETS))
        raise ValueError(f"Unknown dataset '{name}'. Supported datasets: {supported}") from exc
