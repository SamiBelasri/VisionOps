import pytest
from torch.utils.data import Dataset

from visionops.data.bundles import build_classification_bundle


class TinyDataset(Dataset):
    def __init__(self):
        self.targets = [0] * 10 + [1] * 10

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return index, self.targets[index]


def test_bundle_contains_three_disjoint_views():
    dataset = TinyDataset()
    bundle = build_classification_bundle(dataset, validation_ratio=0.2, test_ratio=0.2)

    assert len(bundle.train) == 12
    assert len(bundle.validation) == 4
    assert len(bundle.test) == 4
    assert bundle.class_names == ()


def test_bundle_requires_targets():
    class UnlabelledDataset(Dataset):
        def __len__(self):
            return 1

        def __getitem__(self, index):
            return index

    with pytest.raises(AttributeError, match="targets"):
        build_classification_bundle(UnlabelledDataset())
