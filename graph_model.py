import torch
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import (
    Dataset,
    DataLoader,
    RandomSampler,
    SequentialSampler,
    random_split,
)
from torch import nn


class BlockModel(nn.Module):
    def __init__(self, num_features, num_classes, num_layers, num_nodes):
        super().__init__()
        layers = []


        layers = [] 
        for i in range(num_layers):
          is_first = i == 0 
          layer_dim_in = num_features if is_first else num_nodes
          layers.append(nn.Linear(layer_dim_in, num_nodes))
          layers.append(nn.ReLU())
        
        layers = layers[0:len(layers) - 1]
        self.net = nn.Sequential(*layers)

        self.model_name = "block"

    def forward(self, x):
        x = self.net(x)
        return x


class UnSqueeze(nn.Module):
    def __init__(self, num_features=2, num_classes=2, max_throughput_multiplier=256):
        super().__init__()
        layers = []
        max_throughput = num_features * max_throughput_multiplier
        while num_features < max_throughput:        
            layers.append(nn.Linear(num_features, num_features * 2))
            layers.append(nn.ReLU())
            num_features *= 2
        while num_features > num_classes * 2:
            layers.append(nn.Linear(num_features, num_features // 2))
            num_features = num_features // 2

        layers.append(nn.Linear(num_features, num_features // 2))
        
        print("Unsqueeze final layer num_features ", num_features)
        print("Unsqueeze final layer num_features // 2 ", num_features // 2)
        self.net = nn.Sequential(*layers)

        self.model_name = "unsqueeze"

    def forward(self, x):
        x = self.net(x)

        return x