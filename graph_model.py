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

# TODO(ltang) -- automatically design/relate Width with "small-worldness" of the network
class WideNet(nn.Module):
    # Width should certainly depend on number of nodes in graph, also "small-worldness"
    def __init__(self, num_features=2, num_classes=2, width=100):
        super().__init__()
        self.model_name = "widenet"
        self.width = width

        # TODO(ltang) -- try periodic activation function
        layers = []
        layers.append(nn.Linear(num_features, width))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(width, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = self.net(x)
        return x
    
# Pray that this works; pass in two one-hot encodings instead of two numbers
# Could also envision passing in the "one_hot" param into any of the other models
class OneHotNet(nn.Module):
    def __init__(self, num_features, num_classes=2, width=100):
        super().__init__()
        self.model_name = "onehotnet"
        # TODO(ltang) -- try periodic activation function
        layers = []
        layers.append(nn.Linear(num_features, num_features * 2))
        layers.append(nn.GELU())
        layers.append(nn.Linear(num_features * 2, num_features * 2))
        layers.append(nn.GELU())
        layers.append(nn.Linear(num_features * 2, num_features * 2))
        layers.append(nn.GELU())
        layers.append(nn.Linear(num_features * 2, num_features * 2))
        layers.append(nn.GELU())
        layers.append(nn.Linear(num_features * 2, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = self.net(x)
        return x