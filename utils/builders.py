"""
Builders for creating model, optimizer, criterion and scheduler from configuration.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Iterator


def build_model(config: Any) -> nn.Module:
    """
    Build model from configuration.

    Args:
        config: Configuration object with model settings

    Returns:
        nn.Module: Configured model

    Raises:
        ValueError: If model type is not specified or not supported
    """
    if not hasattr(config, 'model'):
        raise ValueError("Configuration must contain 'model' section")

    model_config = config.model

    # Get model type and params
    if isinstance(model_config, dict):
        model_type = model_config.get('type')
        if model_type is None:
            raise ValueError("Model type must be specified in config")
        # Remove 'type' from params to pass to model constructor
        model_params = {k: v for k, v in model_config.items() if k != 'type'}
    else:
        if not hasattr(model_config, 'type'):
            raise ValueError("Model type must be specified in config")
        model_type = model_config.type
        model_params = {k: v for k, v in vars(model_config).items() if k != 'type'}

    # Build model based on type
    if model_type == 'ViT':
        from models.ViT import ViT
        return ViT(**model_params)
    elif model_type == 'AlexNet':
        from models.AlexNet import AlexNet
        return AlexNet(**model_params)
    elif model_type == 'MLP':
        from models.MLP import MLP
        return MLP(**model_params)
    elif model_type == 'DDPM':
        from models.DDPM import DDPM
        return DDPM(**model_params)
    else:
        raise ValueError(f"Unsupported model type: {model_type}. "
                         f"Supported types: ViT, AlexNet, MLP, DDPM")


def build_optimizer(config: Any, model_params: Iterator[torch.nn.Parameter]) -> torch.optim.Optimizer:
    """
    Build optimizer from configuration.

    Args:
        config: Configuration object with optimizer settings
        model_params: Model parameters to optimize

    Returns:
        torch.optim.Optimizer: Configured optimizer

    Raises:
        ValueError: If optimizer type is not supported
    """
    if hasattr(config, 'optimizer'):
        opt_config = config.optimizer
    else:
        # Default settings if no optimizer in config
        opt_config = {
            'type': 'SGD',
            'lr': config.lr,
            'momentum': config.momentum if hasattr(config, 'momentum') else 0.9,
            'weight_decay': 0.0
        }

    # Get optimizer type
    if isinstance(opt_config, dict):
        opt_type = opt_config.get('type', 'SGD')
        # Remove 'type' from params to pass to optimizer
        opt_params = {k: v for k, v in opt_config.items() if k != 'type'}
    else:
        opt_type = opt_config.type if hasattr(opt_config, 'type') else 'SGD'
        opt_params = {k: v for k, v in vars(opt_config).items() if k != 'type'}

    # Build optimizer based on type
    if opt_type == 'SGD':
        return torch.optim.SGD(model_params, **opt_params)
    elif opt_type == 'Adam':
        return torch.optim.Adam(model_params, **opt_params)
    elif opt_type == 'AdamW':
        return torch.optim.AdamW(model_params, **opt_params)
    elif opt_type == 'RMSprop':
        return torch.optim.RMSprop(model_params, **opt_params)
    else:
        raise ValueError(f"Unsupported optimizer type: {opt_type}. "
                         f"Supported types: SGD, Adam, AdamW, RMSprop")


def build_criterion(config: Any) -> nn.Module:
    """
    Build loss criterion from configuration.

    Args:
        config: Configuration object with criterion settings

    Returns:
        nn.Module: Configured loss criterion

    Raises:
        ValueError: If criterion type is not supported
    """
    if hasattr(config, 'criterion'):
        crit_config = config.criterion
    else:
        # Default settings if no criterion in config
        crit_config = {'type': 'CrossEntropyLoss'}

    # Get criterion type and params
    if isinstance(crit_config, dict):
        crit_type = crit_config.get('type', 'CrossEntropyLoss')
        # Remove 'type' from params to pass to criterion
        crit_params = {k: v for k, v in crit_config.items() if k != 'type'}
    else:
        crit_type = crit_config.type if hasattr(crit_config, 'type') else 'CrossEntropyLoss'
        crit_params = {k: v for k, v in vars(crit_config).items() if k != 'type'}

    # Build criterion based on type
    if crit_type == 'CrossEntropyLoss':
        return nn.CrossEntropyLoss(**crit_params)
    else:
        raise ValueError(f"Unsupported criterion type: {crit_type}. "
                         f"Supported types: CrossEntropyLoss")

def build_scheduler(config: Any, optimizer: torch.optim.Optimizer):
    """
    Build learning rate scheduler from configuration.

    Supports both single scheduler and SequentialLR with multiple schedulers.
    When using scheduler_list, creates a SequentialLR that chains multiple schedulers.

    Args:
        config: Configuration object with scheduler settings
        optimizer: The optimizer to schedule

    Returns:
        torch.optim.lr_scheduler: Configured scheduler or None

    Raises:
        ValueError: If scheduler type is not supported
    """
    # Check for scheduler_list (for SequentialLR)
    if hasattr(config, 'scheduler_list'):
        scheduler_configs = config.scheduler_list

        if scheduler_configs is None or len(scheduler_configs) == 0:
            return None

        # Build individual schedulers
        schedulers = []
        milestones = []
        cumulative_iters = 0

        for sched_config in scheduler_configs:
            # Get scheduler type and params
            if isinstance(sched_config, dict):
                sched_type = sched_config.get('type')
                if sched_type is None:
                    raise ValueError("Scheduler type must be specified in scheduler_list")
                sched_params = {k: v for k, v in sched_config.items() if k != 'type'}
            else:
                sched_type = sched_config.type if hasattr(sched_config, 'type') else None
                if sched_type is None:
                    raise ValueError("Scheduler type must be specified in scheduler_list")
                sched_params = {k: v for k, v in vars(sched_config).items() if k != 'type'}

            # Build individual scheduler
            scheduler = _build_single_scheduler(sched_type, optimizer, sched_params)
            schedulers.append(scheduler)

        # Check if custom milestones provided in config
        if not hasattr(config, 'milestones'):
            raise ValueError("Milestones list must be specified in scheduler_list")
        milestones = config.milestones

        # Return SequentialLR if multiple schedulers, otherwise return single scheduler
        if len(schedulers) > 1:
            return torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=schedulers,
                milestones=milestones
            )
        else:
            return schedulers[0] if schedulers else None

    # Legacy support: single scheduler config
    elif hasattr(config, 'scheduler'):
        sched_config = config.scheduler

        # Get scheduler type and params
        if isinstance(sched_config, dict):
            sched_type = sched_config.get('type', 'LinearLR')
            sched_params = {k: v for k, v in sched_config.items() if k != 'type'}
        else:
            sched_type = sched_config.type if hasattr(sched_config, 'type') else 'LinearLR'
            sched_params = {k: v for k, v in vars(sched_config).items() if k != 'type'}

        return _build_single_scheduler(sched_type, optimizer, sched_params)

    return None


def _build_single_scheduler(sched_type: str, optimizer: torch.optim.Optimizer, sched_params: Dict[str, Any]):
    """
    Build a single learning rate scheduler.

    Args:
        sched_type: Type of scheduler (e.g., 'LinearLR', 'CosineAnnealingLR')
        optimizer: The optimizer to schedule
        sched_params: Parameters for the scheduler

    Returns:
        torch.optim.lr_scheduler: Configured scheduler

    Raises:
        ValueError: If scheduler type is not supported
    """
    # Build scheduler based on type
    if sched_type == 'LinearLR':
        return torch.optim.lr_scheduler.LinearLR(optimizer, **sched_params)
    elif sched_type == 'CosineAnnealingLR':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **sched_params)
    elif sched_type == 'StepLR':
        return torch.optim.lr_scheduler.StepLR(optimizer, **sched_params)
    elif sched_type == 'ExponentialLR':
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, **sched_params)
    elif sched_type == 'CosineAnnealingWarmRestarts':
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, **sched_params)
    elif sched_type == 'MultiStepLR':
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, **sched_params)
    else:
        raise ValueError(f"Unsupported scheduler type: {sched_type}. "
                         f"Supported types: LinearLR, CosineAnnealingLR, StepLR, "
                         f"ExponentialLR, CosineAnnealingWarmRestarts, MultiStepLR")


def build_trainer(config: Any):
    """
    Build trainer from configuration based on model type.

    Routes to the appropriate trainer class based on the model type:
    - Classification models (ViT, AlexNet, MLP) -> ClassificationTrainer
    - Diffusion models (DDPM) -> DiffusionTrainer

    Args:
        config: Configuration object with model settings

    Returns:
        Appropriate trainer instance (ClassificationTrainer or DiffusionTrainer)

    Raises:
        ValueError: If model type is not specified or trainer not found
    """
    if not hasattr(config, 'model'):
        raise ValueError("Configuration must contain 'model' section")

    model_config = config.model

    # Get model type
    if isinstance(model_config, dict):
        model_type = model_config.get('type')
    else:
        model_type = model_config.type if hasattr(model_config, 'type') else None

    if model_type is None:
        raise ValueError("Model type must be specified in config")

    # Route to appropriate trainer
    if model_type in ['ViT', 'AlexNet', 'MLP']:
        # Classification models
        from training.classification_trainer import ClassificationTrainer
        return ClassificationTrainer(config)
    elif model_type == 'DDPM':
        # Diffusion models
        from training.diffusion_trainer import DiffusionTrainer
        return DiffusionTrainer(config)
    else:
        raise ValueError(f"No trainer found for model type: {model_type}. "
                         f"Supported types: ViT, AlexNet, MLP, DDPM")