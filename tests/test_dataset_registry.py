import pytest

from visionops.data.registry import get_dataset_spec


def test_registered_datasets():
    assert get_dataset_spec("EuroSAT").num_classes == 10
    assert get_dataset_spec("intel").num_classes == 6


def test_unknown_dataset_has_actionable_error():
    with pytest.raises(ValueError, match="Supported datasets"):
        get_dataset_spec("unknown")
