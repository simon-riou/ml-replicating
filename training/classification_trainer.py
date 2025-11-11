import time
from tqdm.auto import tqdm

import torch
from torch import nn

import utils
from training.base_trainer import BaseTrainer
from utils.builders import build_model, build_criterion


class ClassificationTrainer(BaseTrainer):
    def __init__(self, args):
        super().__init__(args)
        self.best_acc1 = 0.0

    def build_model(self):
        return build_model(self.args)

    def build_criterion(self):
        return build_criterion(self.args)

    def train_one_epoch(self, epoch):
        self.model.train()

        pbar = tqdm(
            enumerate(self.train_loader),
            total=len(self.train_loader),
            desc=f"Epoch {epoch}/{self.args.epochs} [Train]"
        )

        # Initialize metrics
        total_loss = 0.0
        total_correct_1 = 0.0
        total_correct_5 = 0.0
        total_samples = 0

        start_time = time.time()

        for i, (image, target) in pbar:
            # Data preparation
            image, target = image.to(self.device), target.to(self.device)

            # Compute preparation (if it's too long ==> problem in the dataloaders)
            prepare_time = time.time() - start_time

            # Forward and backward pass
            output = self.model(image)
            loss = self.criterion(output, target)

            self.optimizer.zero_grad()
            loss.backward()

            if self.args.clip_grad_norm is not None:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad_norm)

            self.optimizer.step()

            # Compute metrics
            acc1, acc5 = utils.metrics.accuracy(output, target, topk=(1, 5))

            # Update metrics
            batch_size = image.shape[0]
            total_loss += loss.item() * batch_size
            total_correct_1 += acc1.item() * batch_size / 100.0
            total_correct_5 += acc5.item() * batch_size / 100.0
            total_samples += batch_size

            # Update tensorboard
            self.writer.add_scalar('train/loss', loss.item(), self.n_iter)
            self.writer.add_scalar('train/acc1', acc1.item(), self.n_iter)
            self.writer.add_scalar('train/acc5', acc5.item(), self.n_iter)
            self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]["lr"], self.n_iter)

            # Compute computation time/ efficiency
            process_time = time.time() - start_time - prepare_time
            compute_efficiency = process_time / (process_time + prepare_time)

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{total_loss/total_samples:.4f}',
                'acc@1': f'{total_correct_1/total_samples*100:.2f}%',
                'acc@5': f'{total_correct_5/total_samples*100:.2f}%',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}',
                'eff': f'{compute_efficiency:.2%}'
            })

            start_time = time.time()
            self.n_iter += 1

    def evaluate(self, epoch):
        self.model.eval()

        total_loss = 0.0
        correct_1 = 0
        correct_5 = 0
        total_samples = 0

        header = f"Epoch {epoch}/{self.args.epochs} [Val]"
        pbar = tqdm(self.test_loader, desc=header)

        with torch.no_grad():
            for image, target in pbar:
                # Data preparation
                image = image.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)

                output = self.model(image)
                loss = self.criterion(output, target)

                # Compute metrics
                acc1, acc5 = utils.metrics.accuracy(output, target, topk=(1, 5))

                batch_size = image.shape[0]
                total_loss += loss.item() * batch_size
                correct_1 += acc1.item() * batch_size / 100.0
                correct_5 += acc5.item() * batch_size / 100.0
                total_samples += batch_size

                # Update progress
                pbar.set_postfix({
                    'loss': f'{total_loss/total_samples:.4f}',
                    'acc@1': f'{correct_1/total_samples*100:.2f}%',
                    'acc@5': f'{correct_5/total_samples*100:.2f}%'
                })

        avg_loss = total_loss / total_samples
        avg_acc1 = correct_1 / total_samples * 100.0
        avg_acc5 = correct_5 / total_samples * 100.0

        # Update tensorboard
        if self.writer:
            self.writer.add_scalar('eval/loss', avg_loss, epoch)
            self.writer.add_scalar('eval/acc1', avg_acc1, epoch)
            self.writer.add_scalar('eval/acc5', avg_acc5, epoch)

        # is best model ?
        is_best = avg_acc1 > self.best_acc1

        # Save checkpoint
        metrics = {
            'loss': avg_loss,
            'acc1': avg_acc1,
            'acc5': avg_acc5,
        }
        self.save_checkpoint_with_metrics(epoch, metrics, is_best=is_best)

        # Update best accuracy
        if is_best:
            self.best_acc1 = avg_acc1

        return avg_acc1
