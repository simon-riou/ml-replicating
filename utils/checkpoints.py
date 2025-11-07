import os
import shutil
import numpy as np
import torch
from datetime import datetime
from typing import Dict, Optional


def log_metrics(
    save_dir: str,
    epoch: int,
    n_iter: int,
    metrics: Dict[str, float],
    is_best: bool = False
):
    """
    Log training metrics to training_log.txt file.

    Args:
        save_dir (str): Directory where to save the log file
        epoch (int): Current epoch number
        n_iter (int): Current iteration number
        metrics (dict): Dictionary containing metrics to log
            Expected keys: 'loss', 'acc1', 'acc5', 'lr'
        is_best (bool): If True, prepends "[BEST]" to the log line
    """
    log_path = os.path.join(save_dir, 'training_log.txt')

    best_marker = "[BEST] " if is_best else ""
    log_line = (
        f"{best_marker}Epoch {epoch:03d} | "
        f"Iter {n_iter:06d} | "
        f"Loss: {metrics.get('loss', 0.0):.4f} | "
        f"Acc@1: {metrics.get('acc1', 0.0):.2f}% | "
        f"Acc@5: {metrics.get('acc5', 0.0):.2f}% | "
        f"LR: {metrics.get('lr', 0.0):.6f}\n"
    )

    with open(log_path, 'a') as f:
        f.write(log_line)


def save_best_model_info(
    save_dir: str,
    epoch: int,
    n_iter: int,
    metrics: Dict[str, float],
    checkpoint_name: str
):
    """
    Save information about the best model to best_model_info.txt.

    Args:
        save_dir (str): Directory where to save the info file
        epoch (int): Epoch number of the best model
        n_iter (int): Iteration number of the best model
        metrics (dict): Dictionary containing metrics
            Expected keys: 'loss', 'acc1', 'acc5', 'lr'
        checkpoint_name (str): Name of the checkpoint file
    """
    info_path = os.path.join(save_dir, 'best_model_info.txt')

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    info_content = f"""Best Model Information
=====================
Epoch: {epoch}
Iteration: {n_iter}
Top-1 Accuracy: {metrics.get('acc1', 0.0):.2f}%
Top-5 Accuracy: {metrics.get('acc5', 0.0):.2f}%
Loss: {metrics.get('loss', 0.0):.4f}
Learning Rate: {metrics.get('lr', 0.0):.6f}
Checkpoint: {checkpoint_name}
Saved: {timestamp}
"""

    with open(info_path, 'w') as f:
        f.write(info_content)


def save_checkpoint(
    state,
    save_path: str,
    is_best: bool = False,
    max_keep: int = None,
    metrics: Optional[Dict[str, float]] = None
):
    """Saves torch model to checkpoint file.

    Args:
        state (torch model state): State of a torch Neural Network
        save_path (str): Destination path for saving checkpoint
        is_best (bool): If ``True`` creates additional copy
            ``best_model.ckpt``
        max_keep (int): Specifies the max amount of checkpoints to keep
        metrics (dict, optional): Dictionary containing metrics to log
            Expected keys: 'loss', 'acc1', 'acc5', 'lr', 'epoch', 'n_iter'
            If provided, logs metrics to training_log.txt and
            saves best model info if is_best=True
    """
    # save checkpoint
    torch.save(state, save_path)

    # deal with max_keep
    save_dir = os.path.dirname(save_path)
    list_path = os.path.join(save_dir, 'latest_checkpoint.txt')

    # Keep full path for best model copy before converting to basename
    full_save_path = save_path
    save_path = os.path.basename(save_path)
    if os.path.exists(list_path):
        with open(list_path) as f:
            ckpt_list = f.readlines()
            ckpt_list = [save_path + '\n'] + ckpt_list
    else:
        ckpt_list = [save_path + '\n']

    if max_keep is not None:
        for ckpt in ckpt_list[max_keep:]:
            ckpt = os.path.join(save_dir, ckpt[:-1])
            if os.path.exists(ckpt):
                os.remove(ckpt)
        ckpt_list[max_keep:] = []

    with open(list_path, 'w') as f:
        f.writelines(ckpt_list)

    # copy best
    if is_best:
        shutil.copyfile(full_save_path, os.path.join(save_dir, 'best_model.ckpt'))

    # log metrics if provided
    if metrics is not None:
        epoch = metrics.get('epoch', 0)
        n_iter = metrics.get('n_iter', 0)

        # Log to training_log.txt
        log_metrics(save_dir, epoch, n_iter, metrics, is_best)

        # Save best model info if this is the best model
        if is_best:
            save_best_model_info(save_dir, epoch, n_iter, metrics, save_path)


def load_checkpoint(ckpt_dir_or_file: str, map_location=None, load_best=False):
    """Loads torch model from checkpoint file.

    Args:
        ckpt_dir_or_file (str): Path to checkpoint directory or filename
        map_location: Can be used to directly load to specific device
        load_best (bool): If True loads ``best_model.ckpt`` if exists.
    """
    if os.path.isdir(ckpt_dir_or_file):
        if load_best:
            ckpt_path = os.path.join(ckpt_dir_or_file, 'best_model.ckpt')
        else:
            with open(os.path.join(ckpt_dir_or_file, 'latest_checkpoint.txt')) as f:
                ckpt_path = os.path.join(ckpt_dir_or_file, f.readline()[:-1])
    else:
        ckpt_path = ckpt_dir_or_file
    ckpt = torch.load(ckpt_path, map_location=map_location)
    print(' [*] Loading checkpoint from %s succeed!' % ckpt_path)
    return ckpt