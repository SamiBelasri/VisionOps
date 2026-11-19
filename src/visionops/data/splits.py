"""Reproducible, class-balanced dataset splitting utilities."""

from collections import defaultdict
from collections.abc import Sequence
import random


def stratified_split_indices(
    targets: Sequence[int],
    *,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list[int], list[int], list[int]]:
    """Split sample indices while approximately preserving class proportions.

    The function does not inspect or copy images. It only returns indices, so
    the same underlying dataset can be wrapped by several dataset subsets.
    ``test_ratio`` and ``validation_ratio`` are fractions of the full dataset.
    """

    if not 0 <= validation_ratio < 1:
        raise ValueError("validation_ratio must be in [0, 1).")
    if not 0 <= test_ratio < 1:
        raise ValueError("test_ratio must be in [0, 1).")
    if validation_ratio + test_ratio >= 1:
        raise ValueError("validation_ratio + test_ratio must be smaller than 1.")
    if not targets:
        raise ValueError("Cannot split an empty target sequence.")

    by_class: dict[int, list[int]] = defaultdict(list)
    for index, target in enumerate(targets):
        by_class[int(target)].append(index)

    rng = random.Random(seed)
    train_indices: list[int] = []
    validation_indices: list[int] = []
    test_indices: list[int] = []

    for class_indices in by_class.values():
        shuffled = list(class_indices)
        rng.shuffle(shuffled)
        n_test = round(len(shuffled) * test_ratio)
        n_validation = round(len(shuffled) * validation_ratio)
        test_indices.extend(shuffled[:n_test])
        validation_indices.extend(shuffled[n_test : n_test + n_validation])
        train_indices.extend(shuffled[n_test + n_validation :])

    rng.shuffle(train_indices)
    rng.shuffle(validation_indices)
    rng.shuffle(test_indices)
    return train_indices, validation_indices, test_indices
