import torch
from torch import nn
import torchvision
from torchvision.transforms.functional import center_crop 

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3,3), stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3,3), stride=1, padding='same'),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.block(x)


class Unet(nn.Module):
    def __init__(self, in_channels):
        super(Unet, self).__init__()

        self.features_maps = [in_channels, 64, 128, 256, 512]

        self.encoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(2)
        for i in range(len(self.features_maps) - 1):
            self.encoder.append(ConvBlock(self.features_maps[i], self.features_maps[i + 1]))

        self.bottle_neck = ConvBlock(self.features_maps[-1], self.features_maps[-1] * 2)

        self.decoder = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for i in range(len(self.features_maps) - 1, 0, -1):
            self.upsamples.append(
                nn.ConvTranspose2d(self.features_maps[i] * 2, self.features_maps[i], 
                                   kernel_size=2, stride=2)
            )
            self.decoder.append(ConvBlock(in_channels=self.features_maps[i] * 2, out_channels=self.features_maps[i]))

        self.last_conv = nn.Conv2d(in_channels=64, out_channels=in_channels, kernel_size=1, stride=1, padding='same')
        

    def forward(self, x):
        skip_connections = []

        for block in self.encoder:
            x = block(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottle_neck(x)

        for i, (upsample, block) in enumerate(zip(self.upsamples, self.decoder)):
            x = upsample(x)
            skip = skip_connections[::-1][i]
            x = torch.cat([center_crop(skip, x.shape[2:]), x], dim=1)
            x = block(x)

        
        return self.last_conv(x)
    
model = Unet(3)
print(model(torch.randn(4, 3, 32, 32)).shape)