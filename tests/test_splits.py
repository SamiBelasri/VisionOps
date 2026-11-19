import pytest

from visionops.data.splits import stratified_split_indices


def test_split_is_reproducible_and_disjoint():
    targets = [0] * 10 + [1] * 10
    first = stratified_split_indices(targets, validation_ratio=0.2, test_ratio=0.2, seed=7)
    second = stratified_split_indices(targets, validation_ratio=0.2, test_ratio=0.2, seed=7)

    assert first == second
    train, validation, test = map(set, first)
    assert not train & validation
    assert not train & test
    assert not validation & test
    assert len(train | validation | test) == len(targets)


def test_split_preserves_classes():
    targets = [0] * 10 + [1] * 10
    train, validation, test = stratified_split_indices(
        targets, validation_ratio=0.2, test_ratio=0.2, seed=42
    )

    for indices in (train, validation, test):
        assert {targets[index] for index in indices} == {0, 1}


def test_invalid_ratios_are_rejected():
    with pytest.raises(ValueError):
        stratified_split_indices([0, 1], validation_ratio=0.8, test_ratio=0.3)
