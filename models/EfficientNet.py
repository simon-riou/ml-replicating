import torch
from torch import nn
import torch.nn.functional as F
from torchvision.ops import StochasticDepth
import math

# Modification from the SENet file to take a computed ratio of kept channels instead of a compute with r
class SEBlock(nn.Module):

    def __init__(self, input_channels, reduced_dim):
        super(SEBlock, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Squeeze
            nn.Conv2d(input_channels, reduced_dim, 1),
            nn.SiLU(), # ReLU ?
            nn.Conv2d(reduced_dim, input_channels, 1),
            nn.Sigmoid()              # Excitation
        )

    def forward(self, x):
        return x * self.se(x)

class MBConvBlock(nn.Module):
    """
    Bloc Mobile Inverted Bottleneck (MBConv).
    Structure: Expansion -> Depthwise Conv -> SE -> Projection.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio, se_ratio=0.25, drop_connect_rate=0.2):
        super(MBConvBlock, self).__init__()
        self.use_residual = (stride == 1 and in_channels == out_channels)
        expanded_dim = in_channels * expand_ratio

        layers = []

        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, expanded_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(expanded_dim),
                nn.SiLU()
            ])

        layers.extend([
            nn.Conv2d(expanded_dim, expanded_dim, kernel_size=kernel_size, 
                      stride=stride, padding=kernel_size//2, groups=expanded_dim, bias=False),
            nn.BatchNorm2d(expanded_dim),
            nn.SiLU()
        ])

        if se_ratio > 0:
            se_channels = max(1, int(in_channels * se_ratio))
            layers.append(SEBlock(expanded_dim, se_channels))

        layers.extend([
            nn.Conv2d(expanded_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
            # No activation func (Linear Bottleneck)
        ])

        self.block = nn.Sequential(*layers)
        
        self.stochastic_depth = StochasticDepth(drop_connect_rate, mode="row")

    def forward(self, x):
        result = self.block(x)
        if self.use_residual:
            result = self.stochastic_depth(result)
            result += x
        return result



class EfficientNet(nn.Module):
    def __init__(self, in_channels, width_coeff, depth_coeff, dropout_rate=0.2, num_classes=1000):
        super().__init__()
        
        # Format: [expand_ratio, channels, repeats, stride, kernel_size]
        self.base_config = [
            # Stage 2
            [1,  16, 1, 1, 3],
            # Stage 3
            [6,  24, 2, 2, 3],
            # Stage 4
            [6,  40, 2, 2, 5],
            # Stage 5
            [6,  80, 3, 2, 3],
            # Stage 6
            [6, 112, 3, 1, 5],
            # Stage 7
            [6, 192, 4, 2, 5],
            # Stage 8
            [6, 320, 1, 1, 3],
        ]

        # 1. Stem
        out_channels = int(32 * width_coeff)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU()
        )

        # 2. Base blocks
        layers = []
        in_channels = out_channels
        
        for expand_ratio, channels, repeats, stride, kernel_size in self.base_config:
            # Width scaling (num channels)
            out_channels = int(channels * width_coeff)
            
            # Depth scaling (num layers)
            num_layers = int(math.ceil(repeats * depth_coeff))
            
            for i in range(num_layers):
                current_stride = stride if i == 0 else 1
                
                inp = in_channels if i == 0 else out_channels
                
                layers.append(
                    MBConvBlock(
                        in_channels=inp,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=current_stride,
                        expand_ratio=expand_ratio
                    )
                )
            
            in_channels = out_channels
            
        self.blocks = nn.Sequential(*layers)

        # 3. Head (Classification)
        head_channels = int(1280 * width_coeff)
        
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, head_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(head_channels),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(head_channels, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x

# Format: (width_coeff, depth_coeff, resolution, dropout)
efficientnet_params = {
    'b0': (1.0, 1.0, 224, 0.2),
    'b1': (1.0, 1.1, 240, 0.2),
    'b2': (1.1, 1.2, 260, 0.3),
    'b3': (1.2, 1.4, 300, 0.3),
    'b4': (1.4, 1.8, 380, 0.4),
    'b5': (1.6, 2.2, 456, 0.4),
    'b6': (1.8, 2.6, 528, 0.5),
    'b7': (2.0, 3.1, 600, 0.5),
}

def get_efficientnet(version='b0', in_channels=3, num_classes=1000):
    params = efficientnet_params[version]
    model = EfficientNet(
        in_channels,
        width_coeff=params[0],
        depth_coeff=params[1],
        dropout_rate=params[3],
        num_classes=num_classes
    )
    return model

if __name__ == "__main__":
    from torchsummary import summary
    model = get_efficientnet('b1')
    summary(model, (3, 224, 224))