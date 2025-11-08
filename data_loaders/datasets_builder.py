"""
Config parser to build datasets.
"""

from typing import Union
import argparse
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from data_loaders.builders.classification_data import *
from data_loaders.subset_wrapper import SubsetDataset
from data_loaders.builders.transforms_builder import build_transforms


def _get_config_value(config, key, default=None):
    """
    Get value from config, handling both dict and namespace objects.

    Args:
        config: Configuration dict or namespace
        key: Key to retrieve
        default: Default value if key not found

    Returns:
        Value from config or default
    """
    if isinstance(config, dict):
        return config.get(key, default)
    else:
        return getattr(config, key, default)


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

    # Build transforms from config
    # Support both split-specific keys (train_transforms/test_transforms)
    # and generic 'transforms' key
    transform_key = f'{split}_transforms' if split in ['train', 'test'] else 'transforms'
    transforms_config = dataset_config.get(transform_key, None)

    # Fallback to generic 'transforms' if split-specific not found
    if transforms_config is None and split in ['train', 'test']:
        transforms_config = dataset_config.get('transforms', None)

    # Build the transform pipeline (returns None if no config provided)
    transform = build_transforms(transforms_config, split=split)

    if dataset_type == 'MNIST':
        dataset = build_MNIST(dataset_config, split, transform=transform)

    elif dataset_type == 'CIFAR10':
        dataset = build_CIFAR10(dataset_config, split, transform=transform)

    elif dataset_type == 'CIFAR100':
        dataset = build_CIFAR100(dataset_config, split, transform=transform)

    elif dataset_type == 'Food101':
        dataset = build_Food101(dataset_config, split, transform=transform)

    elif dataset_type == 'ImageNet':
        dataset = build_ImageNet_HF(dataset_config, split, transform=transform)

    # TODO: Add more datasets

    else:
        raise ValueError(
            f"Unsupported dataset type: '{dataset_type}'. "
            f"Currently supported: ['MNIST', 'CIFAR10', 'CIFAR100', 'Food101', 'ImageNet']. "
            f"To add more datasets, extend the build_dataset() function in "
            f"datasets/classification_data.py"
        )

    # Apply subset filtering if specified
    class_subset = dataset_config.get('class_subset', None)
    sample_subset = dataset_config.get('sample_subset', None)
    download_subset = dataset_config.get('download_subset', False)
    load_subset = dataset_config.get('load_subset', False)

    if class_subset is not None or sample_subset is not None:
        print("[*] Creating subset of the dataset")
        dataset = SubsetDataset(
            dataset=dataset,
            class_subset=class_subset,
            sample_subset=sample_subset,
            download_subset=download_subset,
            load_subset=load_subset,
            split=split
        )

        # Print subset information
        subset_info = dataset.get_subset_info()
        print(f"\n{'='*60}")
        print(f"Dataset Subset Configuration ({split} split)")
        print(f"{'='*60}")
        print(f"Original dataset size: {subset_info['original_dataset_size']}")
        print(f"Subset dataset size: {subset_info['total_samples']}")
        print(f"Download subset mode: {subset_info['download_subset']}")
        print(f"Load subset mode: {subset_info['load_subset']}")

        if class_subset is not None:
            print(f"Selected classes: {subset_info['selected_classes']}")
            print(f"Number of classes: {subset_info['num_classes']}")

        if sample_subset is not None:
            print(f"Sample subset parameter: {subset_info['sample_subset']}")

        if 'samples_per_class' in subset_info:
            print(f"\nSamples per class:")
            for class_idx, count in sorted(subset_info['samples_per_class'].items()):
                print(f"  Class {class_idx}: {count} samples")
        print(f"{'='*60}\n")

    return dataset
