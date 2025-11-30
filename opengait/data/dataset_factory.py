# datasets/dataset_factory.py

from typing import Dict, Type

from .dataset import DataSet as StandardDataset
from .base_dataset import BaseDataset
from .clean_gallery_dataset import CleanGalleryDataset
from .perturbed_gallery_dataset import PerturbedGalleryDataset
from .augmented_dataset import AugmentedDataset


_DATASET_REGISTRY: Dict[str, Type[BaseDataset]] = {
    "Standard": StandardDataset,         # behaves like original OpenGait
    "CleanGallery": CleanGalleryDataset,
    "PerturbedGallery": PerturbedGalleryDataset,
    "Augmented": AugmentedDataset,
    # you can add: "AugmentedV2": AugmentedDatasetV2, etc.
}


def build_dataset(data_cfg: dict, training: bool):
    """
    Factory for building datasets.

    Expects `data_cfg["dataset_type"]` to be one of:
        - "Standard"
        - "CleanGallery"
        - "PerturbedGallery"
        - "Augmented"
    Defaults to "Standard" if missing.
    """
    dataset_type = data_cfg.get("dataset_type", "Standard")
    cls = _DATASET_REGISTRY.get(dataset_type, StandardDataset)
    return cls(data_cfg, training)
