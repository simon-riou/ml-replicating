# ResNeXt Configurations

This directory contains training configurations for ResNeXt models based on the paper:
**"Aggregated Residual Transformations for Deep Neural Networks"** by Saining Xie et al. (CVPR 2017)

## Available Configurations

### CIFAR-10
- `resnext50_cifar10.yaml` - ResNeXt-50 for CIFAR-10

**Training Specifications (from paper):**
- Batch size: 128
- Epochs: 300
- Optimizer: SGD (momentum=0.9, weight_decay=5e-4)
- Learning rate: 0.1, divided by 10 at epochs 150 and 225
- Data augmentation: Random crop with 4-pixel padding + horizontal flip

### ImageNet
- `resnext50_imagenet.yaml` - ResNeXt-50 (32×4d architecture)
- `resnext101_imagenet.yaml` - ResNeXt-101 (32×4d architecture)

**Training Specifications (from paper):**
- Batch size: 256 (distributed across 8 GPUs)
- Epochs: 90
- Optimizer: SGD (momentum=0.9, weight_decay=1e-4)
- Learning rate: 0.1, divided by 10 every 30 epochs
- Data augmentation: RandomResizedCrop, horizontal flip, color jitter

## ResNeXt Architecture

ResNeXt introduces **cardinality** (the size of the set of transformations) as an essential dimension:
- **C (cardinality)**: Number of paths/groups in the aggregated transformation
- **d (width)**: Number of channels per path

Common notations:
- ResNeXt-50, 32×4d: 50 layers, cardinality=32, width=4
- ResNeXt-101, 32×4d: 101 layers, cardinality=32, width=4
- ResNeXt-101, 64×4d: 101 layers, cardinality=64, width=4

## Usage

### Training on CIFAR-10
```bash
python train.py --config configs/ResNeXt/resnext50_cifar10.yaml
```

### Training on ImageNet
```bash
python train.py --config configs/ResNeXt/resnext101_imagenet.yaml
```

## Model Variants Supported

The framework supports the following ResNeXt variants:
- ResNeXt <-- custom it as you like
- ResNeXt18
- ResNeXt34
- ResNeXt50
- ResNeXt101
- ResNeXt152

To use a different variant, change the `type` field in the config:
```yaml
model:
  type: "ResNeXt101"  # or ResNeXt18, ResNeXt34, ResNeXt50, ResNeXt152
  in_channels: 3
  num_classes: 1000
```

## References

- Paper: https://arxiv.org/abs/1611.05431
- Official implementation: https://github.com/facebookresearch/ResNeXt
