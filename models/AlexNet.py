import torch
from torch import nn

class AlexNet(nn.Module):
    def __init__(self, in_channels, out_classes):
        super(AlexNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=96, kernel_size=11, stride=4, padding=2)
        self.norm1 = nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2)
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2)
        self.norm2 = nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1)


        self.pooling = nn.MaxPool2d(kernel_size=3, stride=2)

        self.flatten = nn.Flatten()

        self.classifier = nn.Sequential(
            nn.Linear(in_features=6*6*256, out_features=4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=4096, out_features=out_classes),
        )

    def forward(self, x):
        x = self.pooling(self.norm1(torch.relu(self.conv1(x))))
        x = self.pooling(self.norm2(torch.relu(self.conv2(x))))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = self.pooling(torch.relu(self.conv5(x)))
        x = self.flatten(x)

        return self.classifier(x)

if __name__ == "__main__":
    from torchsummary import summary
    model = AlexNet(3, 1000)
    summary(model, (3, 224, 224))