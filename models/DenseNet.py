import torch
from torch import nn

class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, dropout_rate=0.0):
        super(DenseLayer, self).__init__()

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels, 4 * growth_rate, 1, 1, 0, bias=False)
        self.dropout1 = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, 3, 1, 1, bias=False)
        self.dropout2 = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

    def forward(self, x):
        bottleneck = self.conv1(self.relu1(self.bn1(x)))
        if self.dropout1 is not None:
            bottleneck = self.dropout1(bottleneck)

        out = self.conv2(self.relu2(self.bn2(bottleneck)))
        if self.dropout2 is not None:
            out = self.dropout2(out)
        return out

class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.0):
        super(TransitionLayer, self).__init__()

        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
        self.pool = nn.AvgPool2d(2)

    def forward(self, x):
        x = self.conv(self.relu(self.bn(x)))
        if self.dropout is not None:
            x = self.dropout(x)
        return self.pool(x)

class DenseBlock(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate, dropout_rate=0.0):
        super(DenseBlock, self).__init__()

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer_in_channel = in_channels + i * growth_rate
            self.layers.append(DenseLayer(layer_in_channel, growth_rate, dropout_rate))

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_features = layer(torch.cat(features, 1))
            features.append(new_features)
        return torch.cat(features, 1)

class DenseNet(nn.Module):
    def __init__(self, in_channels, num_classes, growth_rate=32, teta=0.5, dense_blocks=[6, 12, 24, 16], dropout_rate=0.2):
        super(DenseNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, growth_rate * 2, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(growth_rate * 2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(3, 2, 1)

        num_features = growth_rate * 2

        self.dense1 = DenseBlock(dense_blocks[0], num_features, growth_rate, dropout_rate)
        num_features += dense_blocks[0] * growth_rate
        self.trans1 = TransitionLayer(num_features, int(num_features * teta), dropout_rate)
        num_features = int(num_features * teta)

        self.dense2 = DenseBlock(dense_blocks[1], num_features, growth_rate, dropout_rate)
        num_features += dense_blocks[1] * growth_rate
        self.trans2 = TransitionLayer(num_features, int(num_features * teta), dropout_rate)
        num_features = int(num_features * teta)

        self.dense3 = DenseBlock(dense_blocks[2], num_features, growth_rate, dropout_rate)
        num_features += dense_blocks[2] * growth_rate
        self.trans3 = TransitionLayer(num_features, int(num_features * teta), dropout_rate)
        num_features = int(num_features * teta)

        self.dense4 = DenseBlock(dense_blocks[3], num_features, growth_rate, dropout_rate)
        num_features += dense_blocks[3] * growth_rate

        self.classifier = nn.Sequential(
            nn.BatchNorm2d(num_features),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(num_features, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.trans1(self.dense1(x))
        x = self.trans2(self.dense2(x))
        x = self.trans3(self.dense3(x))
        x = self.dense4(x)

        return self.classifier(x)

if __name__ == "__main__":
    from torchsummary import summary
    model = DenseNet(3, 1000)
    summary(model, (3, 224, 224))