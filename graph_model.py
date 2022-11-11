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
        layers.append(nn.Linear(num_features, 10000))
        layers.append(nn.Linear(10000, num_classes))
        # layers.append(nn.ReLU())
        # layers.append(nn.Linear(8, 16))
        # layers.append(nn.ReLU())
        # layers.append(nn.Linear(16, 32))
        # layers.append(nn.ReLU())
        # layers.append(nn.Linear(32, 64))
        # layers.append(nn.ReLU())
        # layers.append(nn.Linear(64, 32))
        # layers.append(nn.ReLU())
        # layers.append(nn.Linear(32, 16))
        # layers.append(nn.ReLU())
        # layers.append(nn.Linear(16, 8))
        # layers.append(nn.ReLU())
        # layers.append(nn.Linear(8, num_classes))
        self.net = nn.Sequential(*layers)
        # # diff = max_nodes - min_nodes
        # # for i in range(num_layers):
        # #     if i < num_layers / 2:
        # #         if i == 0:
        # #             layer_dim_in = num_features
        # #             layer_dim_out = min_nodes
        # #         else:
        # #             layer_dim_in
        # #     # is_first = i == 0
        # #     # layer_dim_in = num_features if is_first else num_nodes
        # #     # layer_dim_in = min_nodes if is_first else
        # #     layers.append(nn.Linear(layer_dim_in, num_nodes))
        # #     layers.append(nn.LeakyReLU())

        # self.net = nn.Sequential(*layers)
        # self.output = nn.Linear(num_nodes, num_classes)

    def forward(self, x):
        x = self.net(x)
        # x = self.output(x)

        return x
