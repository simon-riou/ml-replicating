import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, num_layers, size_layers):
        super().__init__()
        self.num_layers = num_layers
        self.size_layers = size_layers

        if num_layers == 1:
            hidden_features = out_features

        self.first_layer = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU()
        )
        self.hidden_layer = nn.Sequential(
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU()
        )
        self.final_layer = nn.Sequential(
            nn.Linear(hidden_features, out_features),
            nn.ReLU()
        )

    def forward(self, x):
        if self.num_layers == 1:
            return self.first_layer(x)
        
        for _ in range(self.num_layers - 2):
            x = self.hidden_layer(x)

        return self.final_layer(x)