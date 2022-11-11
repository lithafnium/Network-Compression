from torch import nn
import torch.optim as optim
import torch
from torch.utils.data import (
    Dataset,
    DataLoader,
    RandomSampler,
    SequentialSampler,
    random_split,
)

import pandas as pd
from collections import defaultdict

from graph_model import GraphModel
from size_estimator import SizeEstimator
import tqdm
import numpy as np
from scipy.io import mmread
import json
import matplotlib.pyplot as plt

dtype = torch.float32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_tensor_type(
    "torch.cuda.FloatTensor" if torch.cuda.is_available() else "torch.FloatTensor"
)


class EdgeDataset(Dataset):
    def __init__(self, edges, labels):
        self.edges = edges
        self.labels = labels

    def __len__(self):
        return len(self.edges)

    def __getitem__(self, i):
        return torch.Tensor(self.edges[i]), self.labels[i]


class Trainer:
    def __init__(
        self,
        lr=1e-3,
        print_freq=1,
        min_layers=4,
        max_layers=6,
        max_nodes=512,
        min_nodes=16,
    ):
        self.criterion = nn.CrossEntropyLoss()
        self.graph_sizes = [100, 1000]
        self.graph_densities = [0.25, 0.5, 0.75]
        self.min_layers = min_layers
        self.max_layers = max_layers
        self.max_nodes = max_nodes
        self.min_nodes = min_nodes

        self.model_info = {}

    def save_plot(self, path, loss_values, num_epochs):
        plt.clf()
        plt.plot([i + 1 for i in range(num_epochs)], loss_values)
        plt.savefig(f"plots/{path}.png")

    def get_data(self, path):
        a = mmread(path)
        labels = []
        edges = []

        for i in range(len(a)):
            for j in range(len(a)):
                edges.append([i, j])
                if a[i][j] == 1:
                    labels.append(1)
                else:
                    labels.append(0)

        labels = np.array(labels)
        edges = np.array(edges)

        dataset = EdgeDataset(edges, labels)
        return dataset

    def eval(self, model: GraphModel, val_dataloader: DataLoader, path: str):
        print("Evaluating...")
        model.eval()
        val_acc = 0
        for X_val_batch, y_val_batch in val_dataloader:
            y_val_pred = model(X_val_batch)
            y_pred_softmax = torch.log_softmax(y_val_pred, dim=1)
            _, y_pred_tags = torch.max(y_pred_softmax, dim=1)

            correct_pred = (y_pred_tags == y_val_batch).float()
            acc = correct_pred.sum() / len(correct_pred)
            acc = torch.round(acc * 100)

            val_acc += acc

        print("Model accuracy: {:.3f}%".format((val_acc / len(val_dataloader)).item()))
        # estimator = SizeEstimator(model, input_size=(2,))
        self.model_info[path] = {
            "accuracy": (val_acc / len(val_dataloader)).item(),
        }
        with open(f"accuracy/{path}-accuracy.json", "w") as f:
            out = json.dumps(self.model_info, indent=4)
            f.write(out)

        torch.save(model.state_dict(), f"{path}.pt")

    def train(
        self,
        model: GraphModel,
        train_dataloader,
        val_dataloader,
        epochs,
        path,
        batch_size=16,
    ):
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        loss = []
        print(f"training model {path}")
        for epoch_i in tqdm.trange(epochs):
            # print(f"Beginning epoch {epoch_i + 1} of {epochs}")

            train_epoch_loss = 0
            model.train()

            for x_train_batch, y_train_batch in train_dataloader:
                b_nodes = x_train_batch.to(device)
                b_labels = y_train_batch.to(device)

                optimizer.zero_grad()

                y_train_pred = model(b_nodes)

                train_loss = self.criterion(y_train_pred, b_labels)
                train_loss.backward()
                optimizer.step()

                train_epoch_loss += train_loss.item() * x_train_batch.size(0)
            loss.append(train_epoch_loss)
            # print(
            #     f"Epoch {epoch_i + 1}: | Train Loss: {train_epoch_loss/len(train_dataloader.sampler):.5f} "
            # )
        self.save_plot(path, loss, epochs)
        self.eval(model, val_dataloader, path)

        # torch.save(model.state_dict(), f"./models/{path}.pt")

    def train_and_eval(self, batch_size=16, epochs=1000):
        for graph_size in self.graph_sizes:
            for graph_density in self.graph_densities:
                print(f"Grabbing graph-{graph_size}-{graph_density}.mtx")
                dataset = self.get_data(f"data/graph-{graph_size}-{graph_density}.mtx")

                train_dataloader = DataLoader(
                    dataset,
                    sampler=RandomSampler(dataset),  # Sampling for training is random
                    batch_size=batch_size,
                )

                evaluation_dataloader = DataLoader(
                    dataset,
                    sampler=SequentialSampler(
                        dataset
                    ),  # Sampling for validation is sequential as the order doesn't matter.
                    batch_size=batch_size,
                )

                for num_layers in range(self.min_layers, self.max_layers + 1):
                    num_nodes = self.min_nodes
                    while num_nodes <= self.max_nodes:
                        model = GraphModel(
                            2, 2, num_layers=num_layers, num_nodes=num_nodes
                        )
                        self.train(
                            model,
                            train_dataloader,
                            evaluation_dataloader,
                            epochs=epochs,
                            path=f"squeeze-model-{num_layers}-{num_nodes}-{graph_size}-{graph_density}",
                        )
                        num_nodes *= 2

        print("Saving output loss and size estimates...")

        with open("loss_sizes.json", "w") as f:
            out = json.dumps(self.model_info, indent=4)
            f.write(out)
