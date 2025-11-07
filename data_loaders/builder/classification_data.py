"""
Classification dataset builder.
"""

from torchvision import datasets, transforms
from typing import Optional
import os
from pathlib import Path
from datasets import load_dataset
from PIL import Image
import torch

def build_MNIST(root, is_train, download):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    ])

    dataset = datasets.MNIST(
        root=root,
        train=is_train,
        transform=transform,
        download=download
    )

    return dataset

def build_CIFAR10(root, is_train, download):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = datasets.CIFAR10(
        root=root,
        train=is_train,
        transform=transform,
        download=download
    )

    return dataset

def build_CIFAR100(root, is_train, download):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    ])

    dataset = datasets.CIFAR100(
        root=root,
        train=is_train,
        transform=transform,
        download=download
    )

    return dataset

def build_ImageNet_HF(
    root: str,
    is_train: bool,
    hf_dataset_name: str = "imagenet-1k",
    streaming: bool = False,
    transform: Optional[transforms.Compose] = None
):
    """
    Build ImageNet dataset from Hugging Face.

    Args:
        root (str): Root directory where dataset will be cached
        is_train (bool): If True, loads training split; otherwise validation split
        download (bool): If True, downloads the dataset (always True for HF datasets)
        hf_dataset_name (str): Hugging Face dataset identifier
            Options:
            - "imagenet-1k" (ILSVRC2012, 1000 classes) - Requires authentication
            - "imagenet-1k-mini" (smaller subset for testing)
        streaming (bool): If True, uses streaming mode (no download, loads on-the-fly)
        transform (transforms.Compose, optional): Custom transforms to apply.
            If None, uses default ImageNet transforms.

    Returns:
        Dataset: A PyTorch-compatible dataset object

    Notes:
        - ImageNet requires authentication on Hugging Face Hub
        - You must accept the terms at: https://huggingface.co/datasets/imagenet-1k
        - Run `huggingface-cli login` before using this function
    """

    # Default transforms
    if transform is None:
        if is_train:
            transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

    split = "train" if is_train else "validation"

    cache_dir = Path(root) / "huggingface_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading ImageNet ({hf_dataset_name}) - split: {split}")
    print(f"Cache directory: {cache_dir}")

    if streaming:
        print("Using streaming mode (no download, loads on-the-fly)")
    else:
        print("Downloading dataset... This may take a while for ImageNet.")

    try:
        hf_dataset = load_dataset(
            hf_dataset_name,
            split=split,
            cache_dir=str(cache_dir),
            streaming=streaming
        )
    except Exception as e:
        if "authentication" in str(e).lower() or "gated" in str(e).lower():
            raise ValueError(
                f"Authentication required for {hf_dataset_name}. "
                "Please follow these steps:\n"
                "1. Create a Hugging Face account at https://huggingface.co/join\n"
                f"2. Accept the dataset terms at https://huggingface.co/datasets/{hf_dataset_name}\n"
                "3. Run: huggingface-cli login\n"
                "4. Enter your access token from https://huggingface.co/settings/tokens"
            )
        else:
            raise e

    class HFImageNetDataset(torch.utils.data.Dataset):
        def __init__(self, hf_dataset, transform):
            self.hf_dataset = hf_dataset
            self.transform = transform

            if streaming:
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

    dataset = HFImageNetDataset(hf_dataset, transform)

    print(f"ImageNet dataset loaded successfully!")
    if not streaming:
        print(f"Dataset size: {len(dataset)} images")

    return dataset