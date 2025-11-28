import torch
from torch import nn

class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels_1x1, out_channels_3x3, out_channels_5x5, pool_proj, reduce_3x3, reduce_5x5):
        super(InceptionModule, self).__init__()

        self.conv_1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels_1x1, 1, 1, 0),
            nn.ReLU()
        )

        self.conv_3x3 = nn.Sequential(
            nn.Conv2d(in_channels, reduce_3x3, 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(reduce_3x3, out_channels_3x3, 3, 1, 'same'),
            nn.ReLU()
        )

        self.conv_5x5 = nn.Sequential(
            nn.Conv2d(in_channels, reduce_5x5, 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(reduce_5x5, out_channels_5x5, 5, 1, 'same'),
            nn.ReLU()
        )

        self.maxpool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, 1, 1, 0),
            nn.ReLU()
        )

    def forward(self, x):
        res_conv1x1 = self.conv_1x1(x)
        res_conv3x3 = self.conv_3x3(x)
        res_conv5x5 = self.conv_5x5(x)
        res_maxpool = self.maxpool(x)

        return torch.cat((res_conv1x1, res_conv3x3, res_conv5x5, res_maxpool), dim=1)
    
class GoogleLeNet(nn.Module):
    def __init__(self, in_channels, out_classes):
        super(GoogleLeNet, self).__init__()

        self.entry = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, 2, 3),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=1), # Remplaced in Inception v2, v3 by BatchNorm (better)
            nn.MaxPool2d(3, 2, 1),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=1), # Remplaced in Inception v2, v3 by BatchNorm (better)
            nn.Conv2d(64, 64, 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(64, 192, 3, 1, 'same'),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, 1)
        )

        self.maxpool = nn.MaxPool2d(3, 2, 1)
            
        self.inception_3a = InceptionModule(192, 64, 128, 32, 32, 96, 16)
        self.inception_3b = InceptionModule(256, 128, 192, 96, 64, 128, 32)

        self.inception_4a = InceptionModule(480, 192, 208, 48, 64, 96, 16)
        self.inception_4b = InceptionModule(512, 160, 224, 64, 64, 112, 24)
        self.inception_4c = InceptionModule(512, 128, 256, 64, 64, 128, 24)
        self.inception_4d = InceptionModule(512, 112, 288, 64, 64, 144, 32)
        self.inception_4e = InceptionModule(528, 256, 320, 128, 128, 160, 32)

        self.inception_5a = InceptionModule(832, 256, 320, 128, 128, 160, 32)
        self.inception_5b = InceptionModule(832, 384, 384, 128, 128, 192, 48)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), # AdaptiveAvgPool2d instead of AvgPool2d to take any input size
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(1024, out_classes)
        )

        self.extra_classifier_1 = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),  # AdaptiveAvgPool2d instead of AvgPool2d to take any input size
            nn.Conv2d(512, 128, 1, 1, 0),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2048, 1024),  # 2048 = 4x4x128
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(1024, out_classes)
        )

        self.extra_classifier_2 = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),  # AdaptiveAvgPool2d instead of AvgPool2d to take any input size
            nn.Conv2d(528, 128, 1, 1, 0),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2048, 1024),  # 2048 = 4x4x128
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(1024, out_classes)
        )

    def forward(self, x):
        x = self.entry(x)

        x = self.inception_3b(self.inception_3a(x))
        x = self.maxpool(x)

        inception_4a = self.inception_4a(x)
        inception_4d = self.inception_4d(self.inception_4c(self.inception_4b(inception_4a)))
        x = self.inception_4e(inception_4d)
        x = self.maxpool(x)

        x = self.inception_5b(self.inception_5a(x))

        main_output = self.classifier(x)

        # Si en mode training, retourner les classifiers auxiliaires
        if self.training:
            aux1_output = self.extra_classifier_1(inception_4a)
            aux2_output = self.extra_classifier_2(inception_4d)
            return main_output, aux1_output, aux2_output
        else:
            # En mode eval, retourner seulement le classifier principal
            return main_output
    
if __name__ == "__main__":
    from torchsummary import summary
    model = GoogleLeNet(in_channels=3, out_classes=1000)
    summary(model, (3, 224, 224))