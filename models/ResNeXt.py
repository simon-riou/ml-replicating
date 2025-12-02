import torch
from torch import nn
import torch.nn.functional as F

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, width_by_channel, C, downsample=True):
        super(Bottleneck, self).__init__()

        self.downsample = downsample

        if downsample:
            self.conv1 = nn.Conv2d(in_channels, C * width_by_channel, 1, 2, 0, bias=False)
        else:
            self.conv1 = nn.Conv2d(in_channels, C * width_by_channel, 1, 1, 0, bias=False)

        self.proj = None
        self.bn = None
        if in_channels != out_channels:
          if downsample:
            self.proj = nn.Conv2d(in_channels, out_channels, 1, 2, 0, bias=False)
          else:
            self.proj = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)
          self.bn = nn.BatchNorm2d(out_channels)


        self.bn1 = nn.BatchNorm2d(C * width_by_channel)

        self.conv2 = nn.Conv2d(C * width_by_channel, C * width_by_channel, 3, 1, 1, groups=C, bias=False)
        self.bn2 = nn.BatchNorm2d(C * width_by_channel)

        self.conv3 = nn.Conv2d(C * width_by_channel, out_channels, 1, 1, 0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)


    def forward(self, x):
        if self.proj is not None:
            identity = self.bn(self.proj(x))
        else:
            identity = x

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        return F.relu(x + identity)

class ResNeXt(nn.Module):
    def __init__(self, layers, in_channels, num_classes, C=32):
        super(ResNeXt, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 64, 7, 2, 3)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.ModuleList([nn.MaxPool2d(3, 2, 1)] + [Bottleneck(64, 256, 4, C, False)] + [Bottleneck(256, 256, 4, C, False) for i in range(layers[0] - 1)])
        self.conv3 = nn.ModuleList([Bottleneck(256, 512, 8, C)] + [Bottleneck(512, 512, 8, C, False) for i in range(layers[1] - 1)])
        self.conv4 = nn.ModuleList([Bottleneck(512, 1024, 16, C)] + [Bottleneck(1024, 1024, 16, C, False) for i in range(layers[2] - 1)])
        self.conv5 = nn.ModuleList([Bottleneck(1024, 2048, 32, C)] + [Bottleneck(2048, 2048, 32, C, False) for i in range(layers[3] - 1)])

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(2048, num_classes)
        )

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        for block in self.conv2:
          x = block(x)
        for block in self.conv3:
          x = block(x)
        for block in self.conv4:
          x = block(x)
        for block in self.conv5:
          x = block(x)
        return self.classifier(x)

def ResNeXt18(in_channels=3, num_classes=1000):
    """ResNeXt-18 model"""
    return ResNeXt([2, 2, 2, 2], in_channels, num_classes)


def ResNeXt34(in_channels=3, num_classes=1000):
    """ResNeXt-34 model"""
    return ResNeXt([3, 4, 6, 3], in_channels, num_classes)


def ResNeXt50(in_channels=3, num_classes=1000):
    """ResNeXt-50 model"""
    return ResNeXt([3, 4, 6, 3], in_channels, num_classes)


def ResNeXt101(in_channels=3, num_classes=1000):
    """ResNeXt-101 model"""
    return ResNeXt([3, 4, 23, 3], in_channels, num_classes)


def ResNeXt152(in_channels=3, num_classes=1000):
    """ResNeXt-152 model"""
    return ResNeXt([3, 8, 36, 3], in_channels, num_classes)

if __name__ == "__main__":
    from torchsummary import summary
    model = ResNeXt152()
    summary(model, (3, 224, 224))