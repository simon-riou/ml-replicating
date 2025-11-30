# DenseNet Configuration Files

This directory contains configuration files for training DenseNet models on various datasets.

### Key Components

- **DenseLayer**: BN-ReLU-Conv(1×1)-BN-ReLU-Conv(3×3) bottleneck structure
- **DenseBlock**: Multiple dense layers with concatenated connections
- **TransitionLayer**: BN-ReLU-Conv(1×1)-AvgPool(2×2) for downsampling and compression
- **Growth Rate (k)**: Number of feature maps added per layer (default: 32)
- **Compression Factor (θ)**: Reduction rate in transition layers (default: 0.5)

## Available Configurations

### DenseNet-121 (default)
- **Architecture**: [6, 12, 24, 16] layers in 4 dense blocks
- **Parameters**: ~7M for ImageNet (1000 classes)
- **Files**:
  - `densenet_cifar10.yaml` - CIFAR-10 training
  - `densenet_imagenet.yaml` - ImageNet training

### DenseNet-169
- **Architecture**: [6, 12, 32, 32] layers in 4 dense blocks
- **Parameters**: ~14M for ImageNet
- **File**: `densenet169_imagenet.yaml`

### DenseNet-201
- **Architecture**: [6, 12, 48, 32] layers in 4 dense blocks
- **Parameters**: ~20M for ImageNet
- **File**: `densenet201_imagenet.yaml`

### DenseNet-264
- **Architecture**: [6, 12, 64, 48] layers in 4 dense blocks
- **Parameters**: ~33M for ImageNet
- **File**: `densenet264_imagenet.yaml`

## Usage

Train DenseNet-121 on CIFAR-10:
```bash
python train.py --config configs/DenseNet/densenet_cifar10.yaml
```

Train DenseNet-121 on ImageNet:
```bash
python train.py --config configs/DenseNet/densenet_imagenet.yaml
```

Train larger variants:
```bash
python train.py --config configs/DenseNet/densenet169_imagenet.yaml
python train.py --config configs/DenseNet/densenet201_imagenet.yaml
python train.py --config configs/DenseNet/densenet264_imagenet.yaml
```

## Model Parameters

Customize your DenseNet by modifying these parameters in the config:

```yaml
model:
  type: "DenseNet"
  in_channels: 3                    # Input channels (3 for RGB)
  num_classes: 1000                 # Number of output classes
  growth_rate: 32                   # Growth rate k
  teta: 0.5                         # Compression factor θ
  dense_blocks: [6, 12, 24, 16]    # Layers per block
```

## Training Hyperparameters

### CIFAR-10
- **Batch size**: 64
- **Epochs**: 300
- **Optimizer**: SGD with Nesterov momentum (lr=0.1, momentum=0.9, weight_decay=1e-4)
- **Scheduler**: MultiStepLR (decay at epochs 150 and 225)

### ImageNet
- **Batch size**: 256
- **Epochs**: 90
- **Optimizer**: SGD with Nesterov momentum (lr=0.1, momentum=0.9, weight_decay=1e-4)
- **Scheduler**: Linear warmup (5 epochs) + Cosine annealing

## Reference

Huang, G., Liu, Z., Van Der Maaten, L., & Weinberger, K. Q. (2017).
Densely connected convolutional networks.
*CVPR 2017*.
