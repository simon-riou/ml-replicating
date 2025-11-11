import time
import os
from pathlib import Path
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
from torchvision.utils import save_image, make_grid

from training.base_trainer import BaseTrainer
from utils.builders import build_model


class DiffusionTrainer(BaseTrainer):
    def __init__(self, args):
        super().__init__(args)
        self.best_loss = float('inf')  # Track best loss

        # Diffusion-specific parameters
        self.num_sample_images = getattr(args, 'num_sample_images', 64)
        self.sample_every_n_epochs = getattr(args, 'sample_every_n_epochs', 1)

    def build_model(self):
        return build_model(self.args)

    def build_criterion(self):
        # Loss is computed inside the model for simplicity
        return None
    
    def _normalize(self, image):
        min_val = image.min()
        max_val = image.max()
        return 2 * (image - min_val) / (max_val - min_val) - 1
    
    def train_one_epoch(self, epoch):
        self.model.train()

        pbar = tqdm(
            enumerate(self.train_loader),
            total=len(self.train_loader),
            desc=f"Epoch {epoch}/{self.args.epochs} [Train]"
        )

        # Initialize metrics
        total_loss = 0.0
        total_samples = 0

        start_time = time.time()

        for i, (image, _) in pbar:  # TODO: take LABELS into accout for conditionnal DDPM
            
            image = image.to(self.device, non_blocking=True)
            image = self._normalize(image)

            prepare_time = time.time() - start_time

            # Compute loss
            loss = self.model.compute_loss(image, self.device)

            self.optimizer.zero_grad()
            loss.backward()

            if self.args.clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad_norm)

            self.optimizer.step()

            # Update metrics
            batch_size = image.shape[0]
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            # Update tensorboard
            self.writer.add_scalar('train/loss', loss.item(), self.n_iter)
            self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]["lr"], self.n_iter)

            # Compute computation time/ efficiency
            process_time = time.time() - start_time - prepare_time
            compute_efficiency = process_time / (process_time + prepare_time)

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{total_loss/total_samples:.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}',
                'eff': f'{compute_efficiency:.2%}'
            })

            start_time = time.time()
            self.n_iter += 1

    def evaluate(self, epoch):
        self.model.eval()

        total_loss = 0.0
        total_samples = 0

        header = f"Epoch {epoch}/{self.args.epochs} [Val]"
        pbar = tqdm(self.test_loader, desc=header)

        with torch.no_grad():
            for image, _ in pbar: # TODO: take LABELS into accout for conditionnal DDPM

                image = image.to(self.device, non_blocking=True)
                image = self._normalize(image)

                # Compute loss
                loss = self.model.compute_loss(image, self.device)

                batch_size = image.shape[0]
                total_loss += loss.item() * batch_size
                total_samples += batch_size

                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{total_loss/total_samples:.4f}'
                })

        avg_loss = total_loss / total_samples

        # Update tensorboard
        if self.writer:
            self.writer.add_scalar('eval/loss', avg_loss, epoch)

        # Generate samples
        if epoch % self.sample_every_n_epochs == 0:
            self.generate_and_save_samples(epoch)

        # is best model ?
        is_best = avg_loss < self.best_loss

        # Save checkpoint
        metrics = {
            'loss': avg_loss,
        }
        self.save_checkpoint_with_metrics(epoch, metrics, is_best=is_best)

        # Update best loss
        if is_best:
            self.best_loss = avg_loss

        return avg_loss

    @torch.no_grad()
    def generate_and_save_samples(self, epoch):
        self.model.eval()

        print(f"\nGenerating {self.num_sample_images} samples...")

        if hasattr(self.args.dataset, 'image_size'):
            image_size = self.args.dataset.image_size
        else:
            raise ValueError('[!] Config must specify \'image_size\' in config in dataset section.')

        if hasattr(self.args.dataset, 'in_channels'):
            in_channels = self.args.dataset.in_channels
        else:
            raise ValueError('[!] Config must specify \'in_channels\' in config in dataset section.')

        # Generate samples
        samples = self.model.sample(
            batch_size=self.num_sample_images,
            image_size=image_size,
            channels=in_channels,
            device=self.device
        )

        # Denormalize from [-1, 1] to [0, 1]
        samples = (samples + 1.0) / 2.0
        samples = torch.clamp(samples, 0.0, 1.0)

        # Create samples directory
        samples_dir = Path(self.args.run_dir) / 'samples'
        os.makedirs(samples_dir, exist_ok=True)

        # Save individual samples grid
        grid = make_grid(samples, nrow=8, normalize=False)
        save_path = samples_dir / f'samples_epoch_{epoch:04d}.png'
        save_image(grid, save_path)

        # Log to tensorboard
        self.writer.add_image('samples/generated', grid, epoch)

        print(f"Samples saved to {save_path}")

        self.model.train()
