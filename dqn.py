import torch.nn as nn
import torch.nn.functional as F


class FCLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FCLayer, self).__init__()
        self.fc = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        return F.relu(self.fc(x), inplace=True)


class DQN(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_layers):
        super(DQN, self).__init__()
        layers = []
        prev_channels = in_channels
        for channels in hidden_layers + [out_channels]:
            layers.append(FCLayer(prev_channels, channels))
            prev_channels = channels
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
