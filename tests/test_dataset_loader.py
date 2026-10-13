from pathlib import Path

import pytest

from visionops.data.datasets import build_dataset


def test_intel_loader_reports_missing_split(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="Intel dataset split not found"):
        build_dataset("intel", tmp_path)


def test_unknown_loader_fails_before_data_access(tmp_path: Path):
    with pytest.raises(ValueError, match="Supported datasets"):
        build_dataset("unknown", tmp_path)
