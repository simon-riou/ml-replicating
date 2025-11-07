"""
Datasets module for loading and managing datasets.

This module provides builders for various types of datasets:
- classification_data: For classification tasks (MNIST, CIFAR10, ImageFolder, etc.)
- diffusion_data: For diffusion models (TODO: future implementation)
"""

from . import datasets_builder
from .datasets_builder import build_dataset


__all__ = [
    'datasets_builder',
    'build_dataset',
]