"""
Dataset wrapper for creating class and sample subsets.
"""

from torch.utils.data import Dataset
import numpy as np
from typing import Optional, Union, List
import torch
import os
import hashlib
import json
from tqdm import tqdm


class SubsetDataset(Dataset):
    """
    Wrapper dataset that creates subsets based on classes and/or samples.

    Args:
        dataset: Original PyTorch dataset with (image, label) format
        class_subset: Specification of which classes to keep:
                     - List[int]: specific class indices to keep (e.g., [0, 1, 2])
                     - int: keep the first N classes (e.g., 3 keeps classes 0, 1, 2)
                     - float (0.0-1.0): keep that percentage of classes (e.g., 0.5 keeps first 50% of classes)
                     - None: keep all classes
        sample_subset: Number or percentage of samples to keep per class:
                      - int: keep exactly that many samples per class
                      - float (0.0-1.0): keep that percentage of samples per class
                      - None: keep all samples
        download_subset: If True, saves the subset indices to disk cache after creation
        load_subset: If True, tries to load the subset indices from disk cache if available
        split: Dataset split name ('train', 'test', 'val') - used to create unique cache files
    """

    def __init__(
        self,
        dataset: Dataset,
        class_subset: Optional[Union[List[int], int, float]] = None,
        sample_subset: Optional[Union[int, float]] = None,
        download_subset: bool = False,
        load_subset: bool = False,
        split: str = 'train'
    ):
        self.dataset = dataset
        self.original_class_subset = class_subset
        self.sample_subset = sample_subset
        self.download_subset = download_subset
        self.load_subset = load_subset
        self.split = split

        # Try to load from cache if load_subset is enabled
        cache_loaded = False
        if self.load_subset:
            cache_loaded = self._load_from_cache()

        if not cache_loaded:
            # Convert class_subset to list of class indices (needs full scan)
            self.class_subset = self._process_class_subset(class_subset)

            # Build index mapping
            self.indices = self._build_indices()

            # Save to cache if download_subset is enabled
            if self.download_subset:
                self._save_to_cache()

        # Update class mapping if needed
        if self.class_subset is not None:
            self.class_to_new_idx = {old_idx: new_idx
                                     for new_idx, old_idx in enumerate(sorted(self.class_subset))}
        else:
            self.class_to_new_idx = None

    def _process_class_subset(self, class_subset) -> Optional[List[int]]:
        """
        Convert class_subset parameter to a list of class indices.

        Args:
            class_subset: Can be None, list, int, or float

        Returns:
            List of class indices or None
        """
        if class_subset is None:
            return None

        # Get all unique classes in the dataset
        all_classes = set()
        for idx in tqdm(range(len(self.dataset)), desc="Scanning dataset classes", leave=False):
            _, label = self.dataset[idx]
            all_classes.add(label)
        all_classes = sorted(list(all_classes))
        num_classes = len(all_classes)

        if isinstance(class_subset, list):
            # Already a list of class indices
            return class_subset

        elif isinstance(class_subset, int):
            # Keep the first N classes
            if class_subset <= 0:
                raise ValueError(f"class_subset as int must be positive, got {class_subset}")
            num_to_keep = min(class_subset, num_classes)
            return all_classes[:num_to_keep]

        elif isinstance(class_subset, float):
            # Keep a percentage of classes
            if not (0.0 < class_subset <= 1.0):
                raise ValueError(
                    f"class_subset as float must be in (0.0, 1.0], got {class_subset}"
                )
            num_to_keep = max(1, int(num_classes * class_subset))
            return all_classes[:num_to_keep]

        else:
            raise TypeError(
                f"class_subset must be None, list, int, or float, got {type(class_subset)}"
            )

    def _build_indices(self) -> List[int]:
        """
        Build list of valid indices based on class_subset and sample_subset.

        Returns:
            List of indices into the original dataset
        """
        # First, organize all indices by class
        class_to_indices = {}

        for idx in tqdm(range(len(self.dataset)), desc="Building subset indices", leave=False):
            _, label = self.dataset[idx]
            if label not in class_to_indices:
                class_to_indices[label] = []
            class_to_indices[label].append(idx)

        # Filter by class_subset if specified
        if self.class_subset is not None:
            class_to_indices = {k: v for k, v in class_to_indices.items()
                               if k in self.class_subset}

        # Apply sample_subset if specified
        if self.sample_subset is not None:
            for class_idx in class_to_indices:
                indices = class_to_indices[class_idx]

                if isinstance(self.sample_subset, float):
                    # Percentage
                    if not (0.0 < self.sample_subset <= 1.0):
                        raise ValueError(
                            f"sample_subset as float must be in (0.0, 1.0], got {self.sample_subset}"
                        )
                    num_samples = max(1, int(len(indices) * self.sample_subset))
                elif isinstance(self.sample_subset, int):
                    # Absolute number
                    if self.sample_subset <= 0:
                        raise ValueError(
                            f"sample_subset as int must be positive, got {self.sample_subset}"
                        )
                    num_samples = min(self.sample_subset, len(indices))
                else:
                    raise TypeError(
                        f"sample_subset must be int or float, got {type(self.sample_subset)}"
                    )

                # Randomly select samples : TODO: add arg for fixed randomness if assigned
                np.random.seed(42)
                class_to_indices[class_idx] = np.random.choice(
                    indices,
                    size=num_samples,
                    replace=False
                ).tolist()

        # Flatten all indices into a single list
        all_indices = []
        for class_idx in sorted(class_to_indices.keys()):
            all_indices.extend(class_to_indices[class_idx])

        return all_indices

    def _get_cache_path(self) -> str:
        """
        Generate a unique cache file path based on dataset configuration.

        Returns:
            Path to the cache file
        """
        # Create a hash of the configuration to ensure uniqueness
        # Use original_class_subset (not processed) to ensure hash is consistent
        # before and after class_subset processing
        config_str = json.dumps({
            'dataset_type': type(self.dataset).__name__,
            'dataset_len': len(self.dataset),
            'class_subset': self.original_class_subset,
            'sample_subset': self.sample_subset,
            'split': self.split,  # Include split to avoid conflicts between train/test/val
        }, sort_keys=True)

        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]

        # Try to get a root directory from the dataset
        cache_dir = './.cache/subsets'
        if hasattr(self.dataset, 'root'):
            dataset_root = self.dataset.root
            cache_dir = os.path.join(dataset_root, '.subset_cache')

        os.makedirs(cache_dir, exist_ok=True)

        cache_filename = f'subset_{self.split}_{config_hash}.pt'
        return os.path.join(cache_dir, cache_filename)

    def _save_to_cache(self):
        """
        Save the subset indices to a cache file.
        """
        try:
            cache_path = self._get_cache_path()
            cache_data = {
                'indices': self.indices,
                'class_subset': self.class_subset,
                'sample_subset': self.sample_subset,
                'dataset_type': type(self.dataset).__name__,
                'dataset_len': len(self.dataset),
                'split': self.split,
            }
            torch.save(cache_data, cache_path)
            print(f"Subset cached to: {cache_path}")
        except Exception as e:
            print(f"Warning: Could not save subset cache: {e}")

    def _load_from_cache(self) -> bool:
        """
        Try to load subset indices from cache.

        Returns:
            True if cache was loaded successfully, False otherwise
        """
        try:
            cache_path = self._get_cache_path()

            if not os.path.exists(cache_path):
                return False

            cache_data = torch.load(cache_path, weights_only=False)

            # Verify cache validity
            # Note: Need to process class_subset to compare it properly
            # For now, compare the cached class_subset directly
            if (cache_data['dataset_type'] != type(self.dataset).__name__ or
                cache_data['dataset_len'] != len(self.dataset) or
                cache_data.get('split') != self.split):
                print("Cache invalid, rebuilding subset...")
                return False

            # Load cached data
            self.indices = cache_data['indices']
            self.class_subset = cache_data['class_subset']
            print(f"Subset loaded from cache: {cache_path}")
            return True

        except Exception as e:
            print(f"Warning: Could not load subset cache: {e}")
            return False

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Map to original dataset index
        original_idx = self.indices[idx]
        image, label = self.dataset[original_idx]

        # Remap label if we're using a class subset
        if self.class_to_new_idx is not None:
            label = self.class_to_new_idx[label]

        return image, label

    def get_subset_info(self):
        """
        Return information about the subset configuration.

        Returns:
            dict with subset statistics
        """
        info = {
            'total_samples': len(self),
            'original_dataset_size': len(self.dataset),
            'download_subset': self.download_subset,
            'load_subset': self.load_subset,
        }

        if self.class_subset is not None:
            info['num_classes'] = len(self.class_subset)
            info['selected_classes'] = sorted(self.class_subset)
            info['class_subset_spec'] = self.original_class_subset

        if self.sample_subset is not None:
            info['sample_subset'] = self.sample_subset

        # Count samples per class
        class_counts = {}
        for idx in self.indices:
            _, label = self.dataset[idx]
            if self.class_to_new_idx is not None:
                label = self.class_to_new_idx[label]
            class_counts[label] = class_counts.get(label, 0) + 1

        info['samples_per_class'] = class_counts

        return info
