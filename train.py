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

class Trainer():
  def __init__(self, lr=1e-3, print_freq=1, max_layers=6, max_nodes=64, min_nodes=12):
    self.criterion = nn.CrossEntropyLoss()
    self.graph_sizes = [100, 1000]
    self.graph_densities = [0.05, 0.1, 0.25, 0.6, 0.85]

    self.max_layers = max_layers 
    self.max_nodes = max_nodes
    self.min_nodes = min_nodes

    self.model_info = {}

  def get_data(self, path):
    df = pd.read_csv(path)
    labels = df["label"]
    edges = df[["id_1", "id_2"]].to_numpy() 

    dataset = EdgeDataset(edges, labels)
    return dataset
  
  def eval(self, model: GraphModel, val_dataloader: DataLoader, path: str):
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
    estimator = SizeEstimator(model, input_size=(2,))
    self.model_info[path] = {
      "accuracy": (val_acc / len(val_dataloader)).item(),
      "model_size_estimate": estimator.estimate_size()
    }

  def train(self, model: GraphModel, train_dataloader, val_dataloader, epochs, path, batch_size=16):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    for epoch_i in range(0, epochs):
        print(f"Beginning epoch {epoch_i + 1} of {epochs}")

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
        print(
            f"Epoch {epoch_i + 1}: | Train Loss: {train_epoch_loss/len(train_dataloader.sampler):.5f} "
        )

    eval(model, val_dataloader, path)

    # torch.save(model.state_dict(), f"./models/{path}.pt")
  
  def train_and_eval(self, batch_size=16, epochs=1000):
    for graph_size in self.graph_sizes:
      for graph_density in self.graph_densities:
        print(f"Grabbing graph-{graph_size}-{graph_density}.csv")
        dataset = self.get_data(f"data/graph-{graph_size}-{graph_density}.csv")

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

        for num_layers in range(1, self.max_layers + 1):
          for num_nodes in range(self.min_nodes, self.max_nodes + 1):
            model = GraphModel(2, 2, num_layers=num_layers, num_nodes=num_nodes)
            self.train(
              model, 
              train_dataloader, 
              evaluation_dataloader, 
              epochs=epochs, 
              path=f"model-{num_layers}-{num_nodes}-{graph_size}-{graph_density}"
            )



