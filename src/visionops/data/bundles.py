"""Dataset bundle construction from deterministic split indices."""

from dataclasses import dataclass
from typing import Any

from torch.utils.data import Dataset, Subset

from .splits import stratified_split_indices


@dataclass(frozen=True)
class ClassificationBundle:
    """The three dataset views used by a supervised classifier."""

    train: Dataset
    validation: Dataset
    test: Dataset
    class_names: tuple[str, ...]


def build_classification_bundle(
    dataset: Dataset,
    *,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    class_names: tuple[str, ...] = (),
) -> ClassificationBundle:
    """Create train/validation/test views over one labelled dataset.

    The dataset must expose a ``targets`` sequence. The returned subsets share
    the original data and only store index lists; image files are not copied.
    """

    targets: Any = getattr(dataset, "targets", None)
    if targets is None:
        raise AttributeError("dataset must expose a 'targets' sequence")

    train_indices, validation_indices, test_indices = stratified_split_indices(
        targets,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    return ClassificationBundle(
        train=Subset(dataset, train_indices),
        validation=Subset(dataset, validation_indices),
        test=Subset(dataset, test_indices),
        class_names=class_names,
    )
