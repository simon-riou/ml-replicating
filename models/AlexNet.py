from torch import nn

class AlexNet(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channel, 32, (5, 5), stride=1, padding='same', bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=2, padding=0),
            nn.Conv2d(32, 64, (5, 5), stride=1, padding='same', bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=2, padding=0),
            nn.Conv2d(64, 64, (5, 5), stride=1, padding='same', bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=2, padding=0),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, 1000),
            nn.Dropout(p=0.8),
            nn.ReLU(),
            nn.Linear(1000, out_channel)
        )

    def forward(self, x):
        return self.classifier(self.features(x))