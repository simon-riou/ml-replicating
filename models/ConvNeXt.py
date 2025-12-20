import torch
from torch import nn
import torchvision
from torchvision.ops import StochasticDepth
import torch.nn.functional as F

class Block(nn.Module):
    def __init__(self, C, p, downsample=False, layer_scale_init_value=1e-6):
        super(Block, self).__init__()

        self.downsample = downsample
        if downsample:
            self.downsample = nn.Conv2d(C, C * 2, 2, 2, 0)
            C = C * 2

        self.conv_7x7 = nn.Conv2d(C, C, 7, 1, 3, groups=C)
        self.conv_upscaling = nn.Conv2d(C, C * 4, 1, 1, 0)
        self.conv_downscaling = nn.Conv2d(C * 4, C, 1, 1, 0)

        self.ln = nn.LayerNorm(C)
        self.gelu = nn.GELU()
        self.stochastic_depth = StochasticDepth(p, "batch")
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((C)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None

    def forward(self, x):
        if self.downsample:
            x = self.downsample(x)

        identity = x

        x = self.conv_7x7(x)
        x = torch.permute(x, (0, 2, 3, 1)) # B, C, H, W -> B, H, W, C
        x = self.ln(x)
        x = torch.permute(x, (0, 3, 1, 2)) # B, H, W, C -> B, C, H, W
        x = self.conv_upscaling(x)
        x = self.gelu(x)
        x = self.conv_downscaling(x)
        if self.gamma is not None:
            # Reshape gamma from (C,) to (1, C, 1, 1) for broadcasting
            x = self.gamma.view(1, -1, 1, 1) * x

        return self.stochastic_depth(x) + identity
    
class ConvNeXt(nn.Module):
    def __init__(self, in_channels, num_classes, C, B=[3, 3, 9, 3], p=0.1):
        super(ConvNeXt, self).__init__()

        self.stem = nn.Conv2d(in_channels, C, 4, 4, 0)

        self.ln_res2 = nn.LayerNorm(C)
        self.res2 = nn.ModuleList([Block(C, p) for _ in range(B[0])])

        self.ln_res3 = nn.LayerNorm(C)
        self.res3 = nn.ModuleList([Block(C, p, downsample=True)] + [Block(C * 2, p) for _ in range(B[1] - 1)])

        self.ln_res4 = nn.LayerNorm(C*2)
        self.res4 = nn.ModuleList([Block(C * 2, p, downsample=True)] + [Block(C * 4, p) for _ in range(B[2] - 1)])

        self.ln_res5 = nn.LayerNorm(C*4)
        self.res5 = nn.ModuleList([Block(C * 4, p, downsample=True)] + [Block(C * 8, p) for _ in range(B[3] - 1)])

        self.classification = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.LayerNorm(C * 8),
            nn.Linear(C * 8, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)

        x = torch.permute(x, (0, 2, 3, 1)) # B, C, H, W -> B, H, W, C
        x = self.ln_res2(x)
        x = torch.permute(x, (0, 3, 1, 2)) # B, H, W, C -> B, C, H, W
        for block in self.res2:
            x = block(x)

        x = torch.permute(x, (0, 2, 3, 1)) # B, C, H, W -> B, H, W, C
        x = self.ln_res3(x)
        x = torch.permute(x, (0, 3, 1, 2)) # B, H, W, C -> B, C, H, W
        for block in self.res3:
            x = block(x)

        x = torch.permute(x, (0, 2, 3, 1)) # B, C, H, W -> B, H, W, C
        x = self.ln_res4(x)
        x = torch.permute(x, (0, 3, 1, 2)) # B, H, W, C -> B, C, H, W
        for block in self.res4:
            x = block(x)

        x = torch.permute(x, (0, 2, 3, 1)) # B, C, H, W -> B, H, W, C
        x = self.ln_res5(x)
        x = torch.permute(x, (0, 3, 1, 2)) # B, H, W, C -> B, C, H, W
        for block in self.res5:
            x = block(x)

        return self.classification(x)
    
def ConvNeXtT(in_channels=3, num_classes=1000):
    """ConvNeXtT model"""
    return ConvNeXt(in_channels, num_classes, 96, [3, 3, 9, 3], p=0.1)


def ConvNeXtS(in_channels=3, num_classes=1000):
    """ConvNeXtS model"""
    return ConvNeXt(in_channels, num_classes, 96, [3, 3, 27, 3], p=0.2)


def ConvNeXtB(in_channels=3, num_classes=1000):
    """ConvNeXtB model"""
    return ConvNeXt(in_channels, num_classes, 128, [3, 3, 27, 3], p=0.4)


def ConvNeXtL(in_channels=3, num_classes=1000):
    """ConvNeXtL model"""
    return ConvNeXt(in_channels, num_classes, 192, [3, 3, 27, 3], p=0.4)


def ConvNeXtXL(in_channels=3, num_classes=1000):
    """ConvNeXtXL model"""
    return ConvNeXt(in_channels, num_classes, 256, [3, 3, 27, 3], p=0.6)

if __name__ == "__main__":
    from torchsummary import summary
    model = ConvNeXtT()
    summary(model, (3, 224, 224))
        