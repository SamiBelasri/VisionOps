"""Dataset interfaces and validation."""

from .registry import DATASETS, DatasetSpec, get_dataset_spec
from .datasets import build_dataset
from .validate import validate_imagefolder
from .splits import stratified_split_indices

__all__ = [
    "DATASETS",
    "DatasetSpec",
    "build_dataset",
    "get_dataset_spec",
    "stratified_split_indices",
    "validate_imagefolder",
]
