import datetime
import os
from pathlib import Path
from abc import ABC, abstractmethod
import random
import numpy as np

import torch
import torch.utils.data
from torch import nn
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter

from utils.checkpoints import save_checkpoint, load_checkpoint, save_config_and_args
from data_loaders import build_dataset


class BaseTrainer(ABC):
    def __init__(self, args):
        self.args = args
        self.device = None
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.lr_scheduler = None
        self.writer = None
        self.train_loader = None
        self.test_loader = None
        self.start_epoch = 0
        self.start_n_iter = 0
        self.n_iter = 0

    def setup_device(self):
        """Setup device (CUDA/CPU) and deterministic mode if needed."""
        self.device = self.args.device if torch.cuda.is_available() else "cpu"

        # Set deterministic mode
        worker_init_fn = None
        if self.args.use_deterministic_algorithms is not None:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True)

            torch.manual_seed(self.args.use_deterministic_algorithms)
            np.random.seed(self.args.use_deterministic_algorithms)
            torch.cuda.manual_seed_all(self.args.use_deterministic_algorithms)

            def worker_init_fn(worker_id):
                torch.manual_seed(self.args.use_deterministic_algorithms)
                torch.cuda.manual_seed(self.args.use_deterministic_algorithms)
                torch.cuda.manual_seed_all(self.args.use_deterministic_algorithms)
                np.random.seed(self.args.use_deterministic_algorithms)
                random.seed(self.args.use_deterministic_algorithms)
                return
        else:
            torch.backends.cudnn.benchmark = True

        return worker_init_fn

    def setup_dataloaders(self, worker_init_fn=None):
        """
        Setup train and test dataloaders.

        Args:
            worker_init_fn: Optional worker initialization function for deterministic mode
        """
        # Load datasets from config
        train_dataset = build_dataset(self.args, split='train')
        self.train_loader = data.DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            drop_last=True,
            worker_init_fn=worker_init_fn,
            num_workers=self.args.num_workers
        )

        test_dataset = build_dataset(self.args, split='test')
        self.test_loader = data.DataLoader(
            test_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            worker_init_fn=worker_init_fn,
            num_workers=self.args.num_workers
        )

    def setup_tensorboard(self):
        """Setup TensorBoard writer for logging."""
        self.writer = SummaryWriter(log_dir=self.args.tb_dir)

    def setup_run_directory(self):
        """Create directory for saving checkpoints and configs."""
        if not self.args.no_save and self.start_epoch < self.args.epochs:
            timestamp = datetime.datetime.now().strftime("%Y_%m_%d-%Hh%Mm%Ss")
            self.args.run_dir = Path(self.args.run_dir, f"{self.model.__class__.__name__}_{timestamp}")
            os.makedirs(self.args.run_dir, exist_ok=True)

    def load_checkpoint_if_needed(self):
        """Load checkpoint if resume path is provided."""
        if self.args.resume:
            ckpt = load_checkpoint(self.args.resume)
            self.model.load_state_dict(ckpt['net'])
            self.model.to(self.device)
            self.start_epoch = ckpt['epoch']
            self.start_n_iter = ckpt['n_iter']
            self.optimizer.load_state_dict(ckpt['optim'])

            # Load scheduler state if it exists in checkpoint
            if self.lr_scheduler is not None and 'scheduler' in ckpt:
                self.lr_scheduler.load_state_dict(ckpt['scheduler'])
                print("Scheduler state restored")

            print("Last checkpoint restored\n")

    def save_checkpoint_with_metrics(self, epoch, metrics, is_best=False):
        """
        Save checkpoint with metrics.

        Args:
            epoch: Current epoch number
            metrics: Dictionary of metrics to save
            is_best: Whether this is the best model so far
        """
        if self.args.no_save:
            return

        # Save config and args on first checkpoint save
        config_file = self.args.run_dir / "config.json"
        if not config_file.exists():
            save_config_and_args(self.args, self.args.run_dir)

        # Prepare checkpoint
        cpkt = {
            'net': self.model.state_dict(),
            'epoch': epoch,
            'n_iter': self.n_iter,
            'optim': self.optimizer.state_dict()
        }

        # Save scheduler state if it exists
        if self.lr_scheduler is not None:
            cpkt['scheduler'] = self.lr_scheduler.state_dict()

        # Add metrics with additional info
        metrics_with_info = {
            **metrics,
            'lr': self.optimizer.param_groups[0]['lr'],
            'epoch': epoch,
            'n_iter': self.n_iter
        }

        save_path = self.args.run_dir / f"model_ckpt_epoch_{epoch}_iter_{self.n_iter}.ckpt"
        save_checkpoint(cpkt, save_path, is_best=is_best, max_keep=self.args.max_keep,
                       metrics=metrics_with_info)

    @abstractmethod
    def build_model(self):
        """
        Build and return the model.
        """
        pass

    @abstractmethod
    def build_criterion(self):
        """
        Build and return the loss criterion.
        """
        pass

    @abstractmethod
    def train_one_epoch(self, epoch):
        """
        Train for one epoch.

        Args:
            epoch: Current epoch number
        """
        pass

    @abstractmethod
    def evaluate(self, epoch):
        """
        Evaluate the model.

        Args:
            epoch: Current epoch number

        Returns:
            Main metric value
        """
        pass

    def train(self):
        """
        Main training loop - orchestrates the entire training process.
        This method is common across all trainers.
        """
        # Setup everything
        worker_init_fn = self.setup_device()
        self.setup_dataloaders(worker_init_fn)
        self.setup_tensorboard()

        # Build model, criterion, optimizer, scheduler
        self.model = self.build_model()
        self.criterion = self.build_criterion()
        self.optimizer = self.build_optimizer()
        self.lr_scheduler = self.build_scheduler()

        self.load_checkpoint_if_needed()

        self.setup_run_directory()

        self.model.to(self.device)

        # Warning for resume training
        if self.start_epoch >= self.args.epochs:
            print(' [!] Model trained on max epoch already! Update the --epochs argument.')
            return

        # Initialize iteration counter
        self.n_iter = self.start_n_iter

        # Training loop
        for epoch in range(self.start_epoch, self.args.epochs):
            # Train one epoch
            self.train_one_epoch(epoch + 1)

            # Evaluate
            self.evaluate(epoch + 1)

            # Step scheduler
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        # Close tensorboard writer
        self.writer.close()

    def build_optimizer(self):
        """
        Build optimizer from config.
        Can be overridden by subclasses if needed.
        """
        from utils.builders import build_optimizer
        return build_optimizer(self.args, self.model.parameters())

    def build_scheduler(self):
        """
        Build learning rate scheduler from config.
        Can be overridden by subclasses if needed.
        """
        from utils.builders import build_scheduler
        return build_scheduler(self.args, self.optimizer)
