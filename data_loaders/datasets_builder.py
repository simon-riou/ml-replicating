"""
Config parser to build datasets.
"""

from typing import Union
import argparse
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from data_loaders.builders.classification_data import *

def build_dataset(
    config: Union[dict, argparse.Namespace],
    split: str = 'train'
) -> Dataset:
    """
    Build a classification dataset from configuration.

    Args:
        config: Configuration dict or argparse.Namespace containing dataset parameters.
                Expected structure:
                - dataset.type: str, type of dataset (e.g., "MNIST")
                - dataset.root: str, root directory for dataset storage
                - dataset.download: bool, whether to download if not present
        split: str, dataset split to load ('train', 'test', or 'val').
               Default is 'train'.

    Returns:
        Dataset: A torch.utils.data.Dataset instance ready to be used with DataLoader.

    Raises:
        ValueError: If dataset type is not supported or config is invalid.
    """

    if hasattr(config, 'dataset'):
        dataset_config = config.dataset
    elif isinstance(config, dict) and 'dataset' in config:
        dataset_config = config['dataset']
    else:
        raise ValueError("No dataset provided in config.")

    dataset_type = dataset_config.get('type', 'MNIST')
    root = dataset_config.get('root', './data')
    download = dataset_config.get('download', True)

    is_train = (split == 'train')

    # TODO: Future - load transforms from config or presets
    # For now, use basic transforms: ToTensor + Normalize

    if dataset_type == 'MNIST':
        # MNIST: 28x28 grayscale images, 10 classes (digits 0-9)
        dataset = build_MNIST(root, is_train, download)

    elif dataset_type == 'CIFAR10':
        # CIFAR10: 32x32 RGB images, 10 classes
        dataset = build_CIFAR10(root, is_train, download)

    elif dataset_type == 'CIFAR100':
        # CIFAR100: 32x32 RGB images, 100 classes
        dataset = build_CIFAR100(root, is_train, download)

    elif dataset_type == 'ImageNet':
        dataset = build_ImageNet_HF(root, is_train)

    # TODO: Add more datasets

    else:
        raise ValueError(
            f"Unsupported dataset type: '{dataset_type}'. "
            f"Currently supported: ['MNIST']. "
            f"To add more datasets, extend the build_dataset() function in "
            f"datasets/classification_data.py"
        )

    return dataset
