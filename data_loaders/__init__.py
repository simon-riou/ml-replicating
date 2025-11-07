"""
Datasets module for loading and managing datasets.

This module provides builders for various types of datasets:
- classification_data: For classification tasks (MNIST, CIFAR10, ImageFolder, etc.)
- diffusion_data: For diffusion models (TODO: future implementation)
"""

from . import datasets
from .datasets import build_dataset


__all__ = [
    'datasets',
    'build_dataset',
]