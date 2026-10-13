"""Dataset interfaces and validation."""

from .registry import DATASETS, DatasetSpec, get_dataset_spec
from .datasets import build_dataset

__all__ = ["DATASETS", "DatasetSpec", "build_dataset", "get_dataset_spec"]
