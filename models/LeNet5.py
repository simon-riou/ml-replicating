import torch
import torch.nn as nn

class LeNet5(nn.Module):
    def __init__(self, in_channels, out_classes):
        super(LeNet5, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=6, kernel_size=(5,5), stride=1, padding=2),
            nn.Sigmoid(),
            nn.AvgPool2d(2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5,5), stride=1, padding='valid'),
            nn.Sigmoid(),
            nn.AvgPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(400, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, out_classes)
        )

    def forward(self, x):
        assert(x.shape[2] == 28 and x.shape[3] == 28)

        return self.classifier(self.features(x))
    
if __name__ == "__main__":
    from torchsummary import summary
    model = LeNet5(in_channels=3, out_classes=10)
    summary(model, (3, 28, 28))