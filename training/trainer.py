# FIXME: Finish all the TODOs

import datetime
import os
from pathlib import Path
import time
import warnings
import random

import torch
import torch.utils.data
import torchvision
import torchvision.transforms

from torch import nn
from torch.utils import data
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

import utils
from utils.checkpoints import *
from utils.metrics import *
from utils.builders import build_optimizer, build_criterion, build_scheduler

import models

def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, args, writer, n_iter):
    model.train()
    
    pbar = tqdm(enumerate(data_loader), total=len(data_loader), desc=f"Epoch {epoch} [Train]")

    start_time = time.time()

    for i, (image, target) in pbar:
        # data preparation
        image, target = image.to(device), target.to(device)
        
        # Compute preparation time to find any issues in dataloader
        prepare_time = start_time-time.time()
        
        # forward and backward pass
        output = model(image)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()

        if args.clip_grad_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            
        optimizer.step()
        
        # Compute metrics
        acc1, acc5 = utils.metrics.accuracy(output, target, topk=(1, 5))

        # Update progress bar
        pbar.set_postfix(loss=f'{loss.item():.4f}', acc1=f'{acc1.item():.2f}%', lr=f'{optimizer.param_groups[0]["lr"]:.6f}')

        # udpate tensorboardX
        writer.add_scalar('train/loss', loss.item(), n_iter)
        writer.add_scalar('train/acc1', acc1.item(), n_iter)
        writer.add_scalar('train/lr', optimizer.param_groups[0]["lr"], n_iter)
        
        # compute computation time and *compute_efficiency*
        process_time = start_time-time.time()-prepare_time
        compute_efficiency = process_time/(process_time+prepare_time)
        pbar.set_description(
            f'Compute efficiency: {compute_efficiency:.2f}, ' 
            f'loss: {loss.item():.2f},  epoch: {epoch}/{args.epochs}')
        start_time = time.time()

        n_iter += 1
    
    return n_iter


def evaluate(model, criterion, optimizer, data_loader, device, args, writer=None, epoch=0, n_iter=0, log_suffix=""):
    model.eval()

    total_loss = 0.0
    correct_1 = 0
    correct_5 = 0
    total_samples = 0

    header = f"Test: {log_suffix}"
    pbar = tqdm(data_loader, desc=header)

    with torch.no_grad():
        for image, target in pbar:
            # data preparation
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            
            output = model(image)
            loss = criterion(output, target)
            
            # Compute metrics
            acc1, acc5 = utils.metrics.accuracy(output, target, topk=(1, 5))
        
            batch_size = image.shape[0]
            total_loss += loss.item() * batch_size
            correct_1 += acc1.item() * batch_size / 100.0
            correct_5 += acc5.item() * batch_size / 100.0
            total_samples += batch_size

    avg_loss = total_loss / total_samples
    avg_acc1 = correct_1 / total_samples * 100.0
    avg_acc5 = correct_5 / total_samples * 100.0

    print(f"{header} Acc@1 {avg_acc1:.3f} Acc@5 {avg_acc5:.3f} Loss {avg_loss:.4f}")
    
    # udpate tensorboardX
    if writer:
        writer.add_scalar(f'eval/loss{log_suffix}', avg_loss, epoch)
        writer.add_scalar(f'eval/acc1{log_suffix}', avg_acc1, epoch)

    if not args.no_save:
        # save checkpoint if needed
        cpkt = {
            'net': model.state_dict(),
            'epoch': epoch,
            'n_iter': n_iter,
            'optim': optimizer.state_dict()
        }

        save_path = args.run_dir / f"model_ckpt_epoch_{epoch}_iter_{n_iter}.ckpt"
        save_checkpoint(cpkt, save_path, is_best=False, max_keep=args.max_keep)

    return avg_acc1



def train(args):
    # Get device
    device = args.device if torch.cuda.is_available() else "cpu"

    # Set deterministic mode
    worker_init_fn = None
    if args.use_deterministic_algorithms is not None:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)

        torch.manual_seed(args.use_deterministic_algorithms)
        np.random.seed(args.use_deterministic_algorithms)
        torch.cuda.manual_seed_all(args.use_deterministic_algorithms)

        def worker_init_fn(worker_id):                                                                                                                                                                                                                                                                         
            torch.manual_seed(args.use_deterministic_algorithms)                                                                                                                                   
            torch.cuda.manual_seed(args.use_deterministic_algorithms)                                                                                                                              
            torch.cuda.manual_seed_all(args.use_deterministic_algorithms)                                                                                          
            np.random.seed(args.use_deterministic_algorithms)                                                                                                             
            random.seed(args.use_deterministic_algorithms)                                                                                                       
            torch.manual_seed(args.use_deterministic_algorithms)                                                                                                                                   
            return
    else:
        torch.backends.cudnn.benchmark = True

    # ===================================================
    # ===================================================
    # TODO: Move the dataset definition elsewhere
    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = datasets.CIFAR10(root=args.data_path,
                                     train=True, 
                                     transform=train_transforms,
                                     download=True)
    train_data_loader = data.DataLoader(train_dataset, 
                                        batch_size=args.batch_size, 
                                        shuffle=True,
                                        drop_last=True,
                                        worker_init_fn=worker_init_fn if worker_init_fn is not None else None,
                                        num_workers=args.num_workers)

    test_dataset = datasets.CIFAR10(root=args.data_path,
                                     train=False, 
                                     transform=test_transforms, 
                                     download=True)
    test_data_loader = data.DataLoader(test_dataset, 
                                        batch_size=args.batch_size, 
                                        shuffle=False,
                                        worker_init_fn=worker_init_fn if worker_init_fn is not None else None,
                                        num_workers=args.num_workers)
    # ===================================================
    # ===================================================
    

    writer = SummaryWriter(log_dir=args.tb_dir)
    

    # ===================================================
    # ===================================================
    # TODO: Let the config choose
    model = models.ViT.ViT(
        img_size=28,
        in_channels=3,
        patch_size=7,
        nb_blocks=12,
        embed_dim=128, # 768
        num_heads=16, # 12
        out_classes=10
    )
    # ===================================================
    # ===================================================
    
    optimizer = build_optimizer(args, model.parameters())
    criterion = build_criterion(args)
    lr_scheduler = build_scheduler(args, optimizer)

     # load checkpoint if needed/ wanted
    start_n_iter = 0
    start_epoch = 0
    if args.resume:
        ckpt = load_checkpoint(args.resume) # custom method for loading last checkpoint
        model.load_state_dict(ckpt['net'])
        model.to(device)
        start_epoch = ckpt['epoch']
        start_n_iter = ckpt['n_iter']
        optimizer.load_state_dict(ckpt['optim'])
        print("last checkpoint restored\n")

    # Create the folder to save models
    if not args.no_save and start_epoch < args.epochs:
        timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        args.run_dir = Path(args.run_dir, f"{model.__class__.__name__}_{timestamp}" )
        os.makedirs(args.run_dir, exist_ok=True)

    model.to(device)

    # Warning for --resume training
    if start_epoch >= args.epochs:
        print(' [!] Model trained on max epoch already ! Update the --epochs argument to increase the number of epochs.')

    n_iter = start_n_iter # Global iterator
    for epoch in range(start_epoch, args.epochs):
        # Training
        n_iter = train_one_epoch(model, criterion, optimizer, train_data_loader, device, epoch+1, args, writer, n_iter)
        
        # Evaluating
        evaluate(model, criterion, optimizer, test_data_loader, device, args, writer, epoch+1, n_iter)

        if lr_scheduler is not None:
            lr_scheduler.step()


    writer.close()