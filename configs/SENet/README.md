# SENet (Squeeze-and-Excitation Networks) Configurations

This directory contains configuration files for training SENet models, based on the paper:

**"Squeeze-and-Excitation Networks"** by Jie Hu, Li Shen, and Gang Sun (CVPR 2018)
- Paper: https://arxiv.org/abs/1709.01507
- Official Repository: https://github.com/hujie-frank/SENet

## Architecture Overview

### SE Block (Squeeze-and-Excitation Block)

The SE block is a channel attention mechanism that consists of three operations:

1. **Squeeze**: Global average pooling to aggregate spatial information
   ```
   z_c = (1/H×W) × Σ u_c(i,j)  for all i,j
   ```

2. **Excitation**: Two fully-connected layers with bottleneck (reduction ratio r=16)
   ```
   s = σ(W₂ · ReLU(W₁ · z))
   ```
   - First FC: C → C/r (with ReLU activation)
   - Second FC: C/r → C (with Sigmoid activation)

3. **Scale**: Element-wise multiplication with input
   ```
   x̃_c = s_c × u_c
   ```

### SENet-154 Architecture

SENet-154 is constructed by incorporating SE blocks into a modified version of the 64×4d ResNeXt-152.

**Modifications from standard ResNeXt (Appendix A):**

(a) Halved channels -> First 1×1 conv channels in bottleneck blocks are halved
(b) Stem replacement -> 7×7 conv replaced with three 3×3 convs
(c) 3×3 downsampling -> 1×1 stride-2 projection replaced with 3×3 stride-2
(d) Dropout -> Dropout (0.2) added before classification layer

**Architecture Details:**

```
Input (3×32×32)
    ↓
Three 3×3 Conv layers (64 channels each)
    ↓
Stage 1: 3 blocks  (64→256 channels, width_by_channel=4)
    ↓
Stage 2: 8 blocks  (256→512 channels, width_by_channel=8)
    ↓
Stage 3: 36 blocks (512→1024 channels, width_by_channel=16)
    ↓
Stage 4: 3 blocks  (1024→2048 channels, width_by_channel=32)
    ↓
Global Average Pooling + Dropout(0.2)
    ↓
Fully Connected (2048→num_classes)
```

**Bottleneck Block Structure:**
```
Input
  ├─ 1×1 conv (reduce channels by half)
  ├─ 3×3 grouped conv (cardinality C)
  ├─ 1×1 conv (expand channels)
  ├─ SE block (r=16)
  └─ Add with identity → ReLU
```

## Configuration Files

### senet154_cifar10.yaml

Training configuration for CIFAR-10 dataset following the original paper's specifications.

**Key Hyperparameters:**
- **Optimizer**: SGD with momentum 0.9
- **Learning Rate**: 0.1 (initial), step decay at epochs 80, 120, 160
- **Weight Decay**: 1e-4 (L2 regularization)
- **Batch Size**: 128
- **Epochs**: 200
- **SE Reduction Ratio**: 16

**Data Augmentation:**
- Random crop 32×32 with 4-pixel padding
- Random horizontal flip
- Normalization: mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]

**Training Command:**
```bash
python train.py --config configs/SENet/senet154_cifar10.yaml
```

## Training Details

### Learning Rate Schedule

The learning rate starts at 0.1 and is divided by 10 at specific milestones:

- **Epochs 1-80**: lr = 0.1
- **Epochs 81-120**: lr = 0.01
- **Epochs 121-160**: lr = 0.001
- **Epochs 161-200**: lr = 0.0001

This schedule follows the ResNet baseline paper specifications, scaled to 200 epochs.

### Computational Cost

The SE block adds minimal computational overhead:
- **Parameters**: ~10% increase compared to baseline
- **FLOPs**: ~1% increase in total computation
- **Training Time**: ~10% longer per epoch

## Implementation Notes

### SE Block Details

The SE block is implemented with:
- `bias=False` in both FC layers (as per paper)
- ReLU activation after first FC layer
- Sigmoid activation after second FC layer
- Proper tensor reshaping for broadcasting: [B, C, H, W] → [B, C] → [B, C, 1, 1]

### Bottleneck Architecture

Each bottleneck block follows the modified ResNeXt design:
1. 1×1 conv for channel reduction (halved compared to standard ResNeXt)
2. 3×3 grouped convolution (cardinality C=64)
3. 1×1 conv for channel expansion
4. SE block applied before residual addition
5. ReLU activation after addition

## References

1. **Squeeze-and-Excitation Networks**
   Jie Hu, Li Shen, Gang Sun
   CVPR 2018
   https://arxiv.org/abs/1709.01507

2. **Deep Residual Learning for Image Recognition**
   Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
   CVPR 2016
   https://arxiv.org/abs/1512.03385

3. **Aggregated Residual Transformations for Deep Neural Networks**
   Saining Xie, Ross Girshick, Piotr Dollár, Zhuowen Tu, Kaiming He
   CVPR 2017
   https://arxiv.org/abs/1611.05431
