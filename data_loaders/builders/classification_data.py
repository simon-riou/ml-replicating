"""
Classification dataset builder.
"""

from torchvision import datasets, transforms
from typing import Optional, Union
import os
from pathlib import Path
from datasets import load_dataset
from PIL import Image
import torch
import numpy as np
import albumentations as A


class AlbumentationsWrapper:
    """
    Wrapper to use Albumentations transforms with torchvision datasets.

    Torchvision datasets return PIL images, but Albumentations expects numpy arrays.
    This wrapper handles the conversion: PIL -> numpy -> Albumentations -> tensor.
    """

    def __init__(self, albumentations_transform: A.Compose):
        """
        Args:
            albumentations_transform: Albumentations Compose object
        """
        self.transform = albumentations_transform

    def __call__(self, image):
        """
        Apply Albumentations transform to a PIL image.

        Args:
            image: PIL Image

        Returns:
            Transformed image (tensor if ToTensorV2 is in pipeline)
        """
        # Convert PIL to numpy array (RGB format)
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image

        # Apply Albumentations transform
        # Albumentations expects dict with 'image' key
        transformed = self.transform(image=image_np)

        return transformed['image']


def _create_default_torchvision_transform(dataset_type: str, split: str) -> transforms.Compose:
    """
    Create default torchvision transforms as fallback.

    Args:
        dataset_type: Type of dataset (e.g., 'MNIST', 'CIFAR10')
        split: Dataset split ('train' or 'test')

    Returns:
        torchvision Compose transform
    """
    if dataset_type == 'MNIST':
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.1307,), std=(0.3081,))
        ])

    elif dataset_type in ['CIFAR10', 'CIFAR100']:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    elif dataset_type == 'Food101':
        return transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    elif dataset_type == 'ImageNet':
        if split == 'train':
            return transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            return transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

    else:
        # Generic default
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])


def build_MNIST(config, split, transform: Optional[A.Compose] = None):
    """
    Build MNIST dataset with Albumentations support.

    Args:
        config: Dataset configuration
        split: Dataset split ('train' or 'test')
        transform: Optional Albumentations Compose. If None, uses default torchvision transforms.

    Returns:
        MNIST dataset
    """
    # Use Albumentations if provided, otherwise fallback to default torchvision
    if transform is not None:
        final_transform = AlbumentationsWrapper(transform)
    else:
        final_transform = _create_default_torchvision_transform('MNIST', split)

    dataset = datasets.MNIST(
        root=config.get('root', './data/mnist'),
        train= split == 'train',
        transform=final_transform,
        download=config.get('download', True)
    )

    return dataset

def build_CIFAR10(config, split, transform: Optional[A.Compose] = None):
    """
    Build CIFAR-10 dataset with Albumentations support.

    Args:
        config: Dataset configuration
        split: Dataset split ('train' or 'test')
        transform: Optional Albumentations Compose. If None, uses default torchvision transforms.

    Returns:
        CIFAR-10 dataset
    """
    # Use Albumentations if provided, otherwise fallback to default torchvision
    if transform is not None:
        final_transform = AlbumentationsWrapper(transform)
    else:
        final_transform = _create_default_torchvision_transform('CIFAR10', split)

    dataset = datasets.CIFAR10(
        root=config.get('root', './data/cifar10'),
        train= split == 'train',
        transform=final_transform,
        download=config.get('download', True)
    )

    return dataset

def build_CIFAR100(config, split, transform: Optional[A.Compose] = None):
    """
    Build CIFAR-100 dataset with Albumentations support.

    Args:
        config: Dataset configuration
        split: Dataset split ('train' or 'test')
        transform: Optional Albumentations Compose. If None, uses default torchvision transforms.

    Returns:
        CIFAR-100 dataset
    """
    # Use Albumentations if provided, otherwise fallback to default torchvision
    if transform is not None:
        final_transform = AlbumentationsWrapper(transform)
    else:
        final_transform = _create_default_torchvision_transform('CIFAR100', split)

    dataset = datasets.CIFAR100(
        root=config.get('root', './data/cifar100'),
        train= split == 'train',
        transform=final_transform,
        download=config.get('download', True)
    )

    return dataset

def build_Food101(config, split, transform: Optional[A.Compose] = None):
    """
    Build Food-101 dataset with Albumentations support.

    Args:
        config: Dataset configuration
        split: Dataset split ('train' or 'test')
        transform: Optional Albumentations Compose. If None, uses default torchvision transforms.

    Returns:
        Food-101 dataset
    """
    # Use Albumentations if provided, otherwise fallback to default torchvision
    if transform is not None:
        final_transform = AlbumentationsWrapper(transform)
    else:
        final_transform = _create_default_torchvision_transform('Food101', split)

    dataset = datasets.Food101(
        root=config.get('root', './data/food101'),
        split=split,
        transform=final_transform,
        download=config.get('download', True)
    )

    return dataset

def build_ImageNet_HF(
    config,
    split,
    transform: Optional[A.Compose] = None
):
    """
    Build ImageNet dataset from Hugging Face with Albumentations support.

    Args:
        config: Dataset configuration
        split: Dataset split ('train' or 'test')
        transform: Optional Albumentations Compose. If None, uses default torchvision transforms.

    Returns:
        Dataset: A PyTorch-compatible dataset object

    Notes:
        - ImageNet requires authentication on Hugging Face Hub
        - You must accept the terms at: https://huggingface.co/datasets/imagenet-1k
        - Run `huggingface-cli login` before using this function
    """

    # Use Albumentations if provided, otherwise fallback to default torchvision
    if transform is not None:
        final_transform = AlbumentationsWrapper(transform)
    else:
        final_transform = _create_default_torchvision_transform('ImageNet', split)

    split = "train" if (split == 'train') else "validation"

    cache_dir = Path(config.get('root', './data/imagenet')) / "huggingface_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading ImageNet (imagenet-1k) - split: {split}")
    print(f"Cache directory: {cache_dir}")

    if config.get('streaming', False):
        print("Using streaming mode (no download, loads on-the-fly)")
    else:
        print("Downloading dataset... This may take a while for ImageNet.")

    try:
        hf_dataset = load_dataset(
            'imagenet-1k',
            split=split,
            cache_dir=str(cache_dir),
            streaming=config.get('streaming', False)
        )
    except Exception as e:
        if "authentication" in str(e).lower() or "gated" in str(e).lower():
            raise ValueError(
                f"Authentication required for imagenet-1k. "
                "Please follow these steps:\n"
                "1. Create a Hugging Face account at https://huggingface.co/join\n"
                f"2. Accept the dataset terms at https://huggingface.co/datasets/imagenet-1k\n"
                "3. Run: hf auth login\n"
                "4. Enter your access token from https://huggingface.co/settings/tokens"
            )
        else:
            raise e

    class HFImageNetDataset(torch.utils.data.Dataset):
        def __init__(self, hf_dataset, transform):
            self.hf_dataset = hf_dataset
            self.transform = transform

            if config.get('streaming', False):
                self._length = None
            else:
                self._length = len(hf_dataset)

        def __len__(self):
            if self._length is None:
                raise TypeError(
                    "Streaming datasets don't have a length. "
                    "Use iter() or disable streaming."
                )
            return self._length

        def __getitem__(self, idx):
            item = self.hf_dataset[idx]

            # HF ImageNet format: {'image': PIL.Image, 'label': int}
            image = item['image']
            label = item['label']

            if image.mode != 'RGB':
                image = image.convert('RGB')

            if self.transform:
                image = self.transform(image)

            return image, label

    dataset = HFImageNetDataset(hf_dataset, final_transform)

    print(f"ImageNet dataset loaded successfully!")
    if not config.get('streaming', False):
        print(f"Dataset size: {len(dataset)} images")

    return dataset