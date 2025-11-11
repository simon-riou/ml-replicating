"""
Transform builder for dataset augmentation using Albumentations and Torchvision.

This module provides functionality to build data transforms from YAML configuration,
supporting both presets and custom transform lists. It supports both albumentations
and torchvision transforms (e.g., RandAugment).
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Any, Dict, List, Union, Optional
import torchvision.transforms.v2 as transforms_v2
import numpy as np
from PIL import Image


class TorchvisionTransformWrapper(A.ImageOnlyTransform):
    """
    Wrapper to use torchvision transforms within albumentations pipeline.

    This wrapper converts images between numpy arrays (used by albumentations)
    and PIL Images (used by torchvision), applies the torchvision transform,
    and converts back.
    """

    def __init__(self, transform, p=1.0):
        super().__init__(p=p)
        self.transform = transform

    def apply(self, img, **params):
        """Apply torchvision transform to numpy image."""
        # Convert numpy array (HWC) to PIL Image
        pil_img = Image.fromarray(img)
        # Apply torchvision transform
        transformed = self.transform(pil_img)
        # Convert back to numpy array
        if isinstance(transformed, Image.Image):
            return np.array(transformed)
        return transformed

    def get_transform_init_args_names(self):
        return ("transform",)


def build_single_transform(transform_config: Dict[str, Any]) -> A.BasicTransform:
    """
    Build a single transform from configuration (Albumentations or Torchvision).

    Args:
        transform_config: Dictionary with 'type' and optional parameters
                         Example: {'type': 'HorizontalFlip', 'p': 0.5}
                         For torchvision: {'type': 'RandAugment', 'num_ops': 2, 'magnitude': 9}

    Returns:
        Transform instance (Albumentations or wrapped Torchvision)

    Raises:
        ValueError: If transform type is not found or invalid
    """
    if not isinstance(transform_config, dict):
        raise ValueError(f"Transform config must be a dict, got {type(transform_config)}")

    if 'type' not in transform_config:
        raise ValueError(f"Transform config must contain 'type' key, got {transform_config}")

    transform_type = transform_config['type']
    transform_params = {k: v for k, v in transform_config.items() if k != 'type'}

    # List of torchvision-specific transforms
    TORCHVISION_TRANSFORMS = ['RandAugment', 'TrivialAugmentWide', 'AutoAugment']

    # Check if it's a torchvision transform
    if transform_type in TORCHVISION_TRANSFORMS:
        if hasattr(transforms_v2, transform_type):
            torchvision_class = getattr(transforms_v2, transform_type)
            try:
                torchvision_transform = torchvision_class(**transform_params)
                # Wrap it to work with albumentations
                return TorchvisionTransformWrapper(torchvision_transform)
            except Exception as e:
                raise ValueError(f"Error creating torchvision transform {transform_type} with params {transform_params}: {e}")
        else:
            raise ValueError(f"Transform type '{transform_type}' not found in torchvision.transforms.v2")

    # Get the transform class from Albumentations
    if hasattr(A, transform_type):
        transform_class = getattr(A, transform_type)
    elif transform_type == 'ToTensorV2':
        transform_class = ToTensorV2
    else:
        raise ValueError(f"Transform type '{transform_type}' not found in albumentations or torchvision")

    try:
        return transform_class(**transform_params)
    except Exception as e:
        raise ValueError(f"Error creating transform {transform_type} with params {transform_params}: {e}")


def build_transforms_from_list(transforms_list: List[Dict[str, Any]]) -> A.Compose:
    """
    Build Albumentations Compose from a list of transform configurations.

    Args:
        transforms_list: List of transform config dicts
                        Example: [{'type': 'HorizontalFlip', 'p': 0.5},
                                 {'type': 'Normalize'},
                                 {'type': 'ToTensorV2'}]

    Returns:
        Albumentations Compose object
    """
    if not isinstance(transforms_list, list):
        raise ValueError(f"transforms_list must be a list, got {type(transforms_list)}")

    transforms = []
    for i, transform_config in enumerate(transforms_list):
        try:
            transforms.append(build_single_transform(transform_config))
        except Exception as e:
            raise ValueError(f"Error building transform at index {i}: {e}")

    return A.Compose(transforms)


def build_transforms_from_preset(
    preset_name: str,
    overrides: Optional[Dict[str, Any]] = None,
    additional: Optional[List[Dict[str, Any]]] = None
) -> A.Compose:
    """
    Build transforms from a preset with optional overrides and additional transforms.

    Args:
        preset_name: Name of the preset (e.g., 'cifar_basic', 'imagenet_train')
        overrides: Optional dict to override specific transform parameters
                  Example: {'normalize': {'mean': [0.5, 0.5, 0.5]}}
        additional: Optional list of additional transforms to append

    Returns:
        Albumentations Compose object

    Raises:
        ValueError: If preset name is not found
    """
    from .transform_presets import TRANSFORM_PRESETS

    if preset_name not in TRANSFORM_PRESETS:
        available = ', '.join(TRANSFORM_PRESETS.keys())
        raise ValueError(f"Preset '{preset_name}' not found. Available presets: {available}")

    # Get base preset transforms list
    preset_transforms = TRANSFORM_PRESETS[preset_name].copy()

    # Apply overrides if provided
    if overrides:
        preset_transforms = _apply_overrides(preset_transforms, overrides)

    # Add additional transforms if provided
    if additional:
        preset_transforms.extend(additional)

    return build_transforms_from_list(preset_transforms)


def _apply_overrides(
    transforms_list: List[Dict[str, Any]],
    overrides: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Apply parameter overrides to a transforms list.

    Args:
        transforms_list: Base list of transform configs
        overrides: Dict mapping transform names (lowercase) to parameter overrides
                  Example: {'normalize': {'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5]}}

    Returns:
        Modified transforms list with overrides applied
    """
    result = []
    for transform_config in transforms_list:
        transform_config = transform_config.copy()
        transform_type = transform_config.get('type', '').lower()

        # Check if this transform has overrides
        if transform_type in overrides:
            # Merge override params into transform config
            override_params = overrides[transform_type]
            if isinstance(override_params, dict):
                transform_config.update(override_params)

        result.append(transform_config)

    return result


def build_transforms(
    transforms_config: Union[str, List[Dict[str, Any]], Dict[str, Any], None],
    split: str = 'train'
) -> Optional[A.Compose]:
    """
    Main entry point to build transforms from various config formats.

    Supports three formats:
    1. String preset: "cifar_augmented"
    2. Dict with preset and overrides: {'preset': 'cifar_augmented', 'override': {...}}
    3. List of transforms: [{'type': 'HorizontalFlip', 'p': 0.5}, ...]

    Args:
        transforms_config: Transform configuration (string, dict, or list)
        split: Dataset split ('train' or 'test'), used for automatic preset selection

    Returns:
        Albumentations Compose object, or None if config is None

    Examples:
        >>> # Preset only
        >>> build_transforms("cifar_augmented")

        >>> # Preset with overrides
        >>> build_transforms({
        ...     'preset': 'cifar_augmented',
        ...     'override': {'normalize': {'mean': [0.5, 0.5, 0.5]}}
        ... })

        >>> # Custom list
        >>> build_transforms([
        ...     {'type': 'HorizontalFlip', 'p': 0.5},
        ...     {'type': 'Normalize'},
        ...     {'type': 'ToTensorV2'}
        ... ])
    """
    # Return None if no config provided (will use defaults)
    if transforms_config is None:
        return None

    # Format 1: String preset
    if isinstance(transforms_config, str):
        return build_transforms_from_preset(transforms_config)

    # Format 2: Dict with preset/override/additional
    if isinstance(transforms_config, dict):
        # Check if it's a preset-based config
        if 'preset' in transforms_config:
            preset_name = transforms_config['preset']
            overrides = transforms_config.get('override', None)
            additional = transforms_config.get('additional', None)
            return build_transforms_from_preset(preset_name, overrides, additional)

        # If it has 'type', treat as single transform wrapped in list
        if 'type' in transforms_config:
            return build_transforms_from_list([transforms_config])

        raise ValueError(
            f"Dict config must contain 'preset' or 'type' key, got {transforms_config.keys()}"
        )

    # Format 3: List of transforms
    if isinstance(transforms_config, list):
        return build_transforms_from_list(transforms_config)

    raise ValueError(
        f"transforms_config must be str, dict, list, or None. Got {type(transforms_config)}"
    )


def get_available_presets() -> List[str]:
    """
    Get list of all available transform presets.

    Returns:
        List of preset names
    """
    from .transform_presets import TRANSFORM_PRESETS
    return list(TRANSFORM_PRESETS.keys())
