"""
Predefined transform presets for common datasets using Albumentations.

Each preset is a list of transform configurations that can be passed to
build_transforms_from_list(). Presets follow naming convention:
- {dataset}_basic: Simple transforms (ToTensor + Normalize)
- {dataset}_augmented: Training transforms with data augmentation
"""

from typing import Dict, List, Any


# ==================== MNIST Presets ====================

MNIST_BASIC: List[Dict[str, Any]] = [
    {'type': 'Normalize', 'mean': [0.1307], 'std': [0.3081], 'max_pixel_value': 255.0},
    {'type': 'ToTensorV2'},
]

MNIST_AUGMENTED: List[Dict[str, Any]] = [
    {'type': 'Rotate', 'limit': 15, 'p': 0.5},
    {'type': 'Affine', 'translate_percent': {'x': (-0.1, 0.1), 'y': (-0.1, 0.1)}, 'p': 0.5},
    {'type': 'Normalize', 'mean': [0.1307], 'std': [0.3081], 'max_pixel_value': 255.0},
    {'type': 'ToTensorV2'},
]


# ==================== CIFAR-10 / CIFAR-100 Presets ====================

CIFAR_BASIC: List[Dict[str, Any]] = [
    {'type': 'Normalize', 'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5], 'max_pixel_value': 255.0},
    {'type': 'ToTensorV2'},
]

CIFAR_AUGMENTED: List[Dict[str, Any]] = [
    {'type': 'RandomCrop', 'height': 32, 'width': 32, 'padding': 4},
    {'type': 'HorizontalFlip', 'p': 0.5},
    {'type': 'Normalize', 'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5], 'max_pixel_value': 255.0},
    {'type': 'ToTensorV2'},
]

# CIFAR-10 with accurate statistics
CIFAR10_BASIC: List[Dict[str, Any]] = [
    {'type': 'Normalize',
     'mean': [0.4914, 0.4822, 0.4465],
     'std': [0.2470, 0.2435, 0.2616],
     'max_pixel_value': 255.0},
    {'type': 'ToTensorV2'},
]

CIFAR10_AUGMENTED: List[Dict[str, Any]] = [
    {'type': 'RandomCrop', 'height': 32, 'width': 32, 'padding': 4},
    {'type': 'HorizontalFlip', 'p': 0.5},
    {'type': 'Normalize',
     'mean': [0.4914, 0.4822, 0.4465],
     'std': [0.2470, 0.2435, 0.2616],
     'max_pixel_value': 255.0},
    {'type': 'ToTensorV2'},
]

# CIFAR-100 with accurate statistics
CIFAR100_BASIC: List[Dict[str, Any]] = [
    {'type': 'Normalize',
     'mean': [0.5071, 0.4867, 0.4408],
     'std': [0.2675, 0.2565, 0.2761],
     'max_pixel_value': 255.0},
    {'type': 'ToTensorV2'},
]

CIFAR100_AUGMENTED: List[Dict[str, Any]] = [
    {'type': 'RandomCrop', 'height': 32, 'width': 32, 'padding': 4},
    {'type': 'HorizontalFlip', 'p': 0.5},
    {'type': 'Normalize',
     'mean': [0.5071, 0.4867, 0.4408],
     'std': [0.2675, 0.2565, 0.2761],
     'max_pixel_value': 255.0},
    {'type': 'ToTensorV2'},
]


# ==================== ImageNet Presets ====================

IMAGENET_BASIC: List[Dict[str, Any]] = [
    {'type': 'Resize', 'height': 256, 'width': 256},
    {'type': 'CenterCrop', 'height': 224, 'width': 224},
    {'type': 'Normalize',
     'mean': [0.485, 0.456, 0.406],
     'std': [0.229, 0.224, 0.225],
     'max_pixel_value': 255.0},
    {'type': 'ToTensorV2'},
]

IMAGENET_AUGMENTED: List[Dict[str, Any]] = [
    {'type': 'RandomResizedCrop', 'height': 224, 'width': 224, 'scale': (0.08, 1.0)},
    {'type': 'HorizontalFlip', 'p': 0.5},
    {'type': 'ColorJitter', 'brightness': 0.4, 'contrast': 0.4, 'saturation': 0.4, 'hue': 0.1, 'p': 0.8},
    {'type': 'Normalize',
     'mean': [0.485, 0.456, 0.406],
     'std': [0.229, 0.224, 0.225],
     'max_pixel_value': 255.0},
    {'type': 'ToTensorV2'},
]


# ==================== Food-101 Presets ====================

FOOD101_BASIC: List[Dict[str, Any]] = [
    {'type': 'Resize', 'height': 512, 'width': 512},
    {'type': 'Normalize',
     'mean': [0.5, 0.5, 0.5],
     'std': [0.5, 0.5, 0.5],
     'max_pixel_value': 255.0},
    {'type': 'ToTensorV2'},
]

FOOD101_AUGMENTED: List[Dict[str, Any]] = [
    {'type': 'RandomResizedCrop', 'scale': (512, 512), 'scale': (0.8, 1.0)},
    {'type': 'HorizontalFlip', 'p': 0.5},
    {'type': 'Rotate', 'limit': 15, 'p': 0.5},
    {'type': 'ColorJitter', 'brightness': 0.2, 'contrast': 0.2, 'saturation': 0.2, 'hue': 0.1, 'p': 0.5},
    {'type': 'Normalize',
     'mean': [0.5, 0.5, 0.5],
     'std': [0.5, 0.5, 0.5],
     'max_pixel_value': 255.0},
    {'type': 'ToTensorV2'},
]


# ==================== Generic Presets ====================

GENERIC_BASIC: List[Dict[str, Any]] = [
    {'type': 'Normalize', 'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5], 'max_pixel_value': 255.0},
    {'type': 'ToTensorV2'},
]

GENERIC_AUGMENTED: List[Dict[str, Any]] = [
    {'type': 'HorizontalFlip', 'p': 0.5},
    {'type': 'ShiftScaleRotate', 'shift_limit': 0.1, 'scale_limit': 0.1, 'rotate_limit': 15, 'p': 0.5},
    {'type': 'ColorJitter', 'brightness': 0.2, 'contrast': 0.2, 'saturation': 0.2, 'hue': 0.1, 'p': 0.5},
    {'type': 'Normalize', 'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5], 'max_pixel_value': 255.0},
    {'type': 'ToTensorV2'},
]


# ==================== Master Preset Dictionary ====================

TRANSFORM_PRESETS: Dict[str, List[Dict[str, Any]]] = {
    # MNIST
    'mnist_basic': MNIST_BASIC,
    'mnist_augmented': MNIST_AUGMENTED,

    # CIFAR (generic)
    'cifar_basic': CIFAR_BASIC,
    'cifar_augmented': CIFAR_AUGMENTED,

    # CIFAR-10 (specific)
    'cifar10_basic': CIFAR10_BASIC,
    'cifar10_augmented': CIFAR10_AUGMENTED,

    # CIFAR-100 (specific)
    'cifar100_basic': CIFAR100_BASIC,
    'cifar100_augmented': CIFAR100_AUGMENTED,

    # ImageNet
    'imagenet_basic': IMAGENET_BASIC,
    'imagenet_augmented': IMAGENET_AUGMENTED,
    'imagenet_train': IMAGENET_AUGMENTED,  # Alias
    'imagenet_test': IMAGENET_BASIC,        # Alias

    # Food-101
    'food101_basic': FOOD101_BASIC,
    'food101_augmented': FOOD101_AUGMENTED,

    # Generic
    'generic_basic': GENERIC_BASIC,
    'generic_augmented': GENERIC_AUGMENTED,
}


def get_default_preset_for_dataset(dataset_type: str, split: str = 'train') -> str:
    """
    Get the default preset name for a given dataset type and split.

    Args:
        dataset_type: Dataset type (e.g., 'MNIST', 'CIFAR10', 'ImageNet')
        split: Dataset split ('train' or 'test')

    Returns:
        Preset name (e.g., 'cifar10_augmented', 'imagenet_basic')

    Examples:
        >>> get_default_preset_for_dataset('CIFAR10', 'train')
        'cifar10_augmented'
        >>> get_default_preset_for_dataset('CIFAR10', 'test')
        'cifar10_basic'
    """
    dataset_type = dataset_type.lower()
    suffix = 'augmented' if split == 'train' else 'basic'

    # Map dataset types to preset prefixes
    preset_map = {
        'mnist': 'mnist',
        'cifar10': 'cifar10',
        'cifar100': 'cifar100',
        'imagenet': 'imagenet',
        'food101': 'food101',
    }

    preset_prefix = preset_map.get(dataset_type, 'generic')
    return f"{preset_prefix}_{suffix}"
