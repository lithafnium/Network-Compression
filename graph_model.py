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

class GraphModel(nn.Module):
    def __init__(self, num_features, num_classes, num_layers, num_nodes):
        super().__init__()
        layers = [] 
        for i in range(num_layers):
          is_first = i == 0 
          layer_dim_in = num_features if is_first else num_nodes
          layers.append(nn.Linear(layer_dim_in, num_nodes))
          layers.append(nn.ReLU())

        self.net = nn.Sequential(*layers)
        self.output = nn.Linear(num_nodes, num_classes)

    def forward(self, x):
        x = self.net(x)
        x = self.output(x)

        return x