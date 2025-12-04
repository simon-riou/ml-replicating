import torch
from torch import nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, in_channels, r):
        super(SEBlock, self).__init__()

        # Squeeze
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))

        # Excitate
        self.fc1 = nn.Linear(in_channels, in_channels // r, bias=False)
        self.fc2 = nn.Linear(in_channels // r, in_channels, bias=False)

    def forward(self, x):
        # Squeeze
        batch_size, channels, _, _ = x.size()
        z_c = self.avg_pool(x).view(batch_size, channels)

        # Excitate
        s = self.fc1(z_c)
        s = F.relu(s)
        s = self.fc2(s)
        s = torch.sigmoid(s).view(batch_size, channels, 1, 1)

        return s * x


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, width_by_channel, C, downsample=True):
        super(Bottleneck, self).__init__()

        self.downsample = downsample

        if downsample:
            self.conv1 = nn.Conv2d(in_channels, C * width_by_channel // 2, 1, 2, 0, bias=False)
        else:
            self.conv1 = nn.Conv2d(in_channels, C * width_by_channel // 2, 1, 1, 0, bias=False)

        self.proj = None
        self.bn = None
        if in_channels != out_channels:
          if downsample:
            self.proj = nn.Conv2d(in_channels, out_channels, 3, 2, 1, bias=False)
          else:
            self.proj = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)
          self.bn = nn.BatchNorm2d(out_channels)


        self.bn1 = nn.BatchNorm2d(C * width_by_channel // 2)

        self.conv2 = nn.Conv2d(C * width_by_channel // 2, C * width_by_channel, 3, 1, 1, groups=C, bias=False)
        self.bn2 = nn.BatchNorm2d(C * width_by_channel)

        self.conv3 = nn.Conv2d(C * width_by_channel, out_channels, 1, 1, 0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.se = SEBlock(out_channels, 16)


    def forward(self, x):
        if self.proj is not None:
            identity = self.bn(self.proj(x))
        else:
            identity = x

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))

        x = self.se(x)

        return F.relu(x + identity)

class SENet(nn.Module):
    """
    SENet-154 is constructed by incorporating SE blocks 
    into a modified version of the 64×4d ResNeXt-152
    """
    def __init__(self, in_channels, num_classes):
        super(SENet, self).__init__()

        self.conv1 = nn.Sequential(
           nn.Conv2d(in_channels, 64, 3, 1, 1),
           nn.BatchNorm2d(64),
           nn.ReLU(),
           nn.Conv2d(64, 64, 3, 1, 1),
           nn.BatchNorm2d(64),
           nn.ReLU(),
           nn.Conv2d(64, 64, 3, 1, 1),
           nn.BatchNorm2d(64),
           nn.ReLU()
        )

        self.conv2 = nn.ModuleList([nn.MaxPool2d(3, 2, 1)] + [Bottleneck(64, 256, 4, 64, False)] + [Bottleneck(256, 256, 4, 64, False) for i in range(3 - 1)])
        self.conv3 = nn.ModuleList([Bottleneck(256, 512, 8, 64)] + [Bottleneck(512, 512, 8, 64, False) for i in range(8 - 1)])
        self.conv4 = nn.ModuleList([Bottleneck(512, 1024, 16, 64)] + [Bottleneck(1024, 1024, 16, 64, False) for i in range(36 - 1)])
        self.conv5 = nn.ModuleList([Bottleneck(1024, 2048, 32, 64)] + [Bottleneck(2048, 2048, 32, 64, False) for i in range(3 - 1)])

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(2048, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        for block in self.conv2:
          x = block(x)
        for block in self.conv3:
          x = block(x)
        for block in self.conv4:
          x = block(x)
        for block in self.conv5:
          x = block(x)
        return self.classifier(x)