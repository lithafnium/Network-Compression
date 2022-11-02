import argparse
from random import randrange

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
from scipy.io import mmread
import numpy as np
import pandas as pd


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


class ToyModel(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()

        self.input = nn.Linear(num_features, 15126)
        self.output = nn.Linear(15126, num_classes)

    def forward(self, x):
        x = self.input(x)
        x = self.output(x)

        return x


def train(
    model: ToyModel, train_dataloader, val_dataloader, epochs, batch_size=64, lr=0.01
):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
    for epoch_i in range(0, epochs):
        print(f"Beginning epoch {epoch_i + 1} of {epochs}")

        train_epoch_loss = 0
        model.train()

        for x_train_batch, y_train_batch in train_dataloader:
            b_nodes = x_train_batch.to(device)
            b_labels = y_train_batch.to(device)

            optimizer.zero_grad()

            y_train_pred = model(b_nodes)

            train_loss = criterion(y_train_pred, b_labels)
            train_loss.backward()
            optimizer.step()

            train_epoch_loss += train_loss.item() * x_train_batch.size(0)
        print(
            f"Epoch {epoch_i + 1}: | Train Loss: {train_epoch_loss/len(train_dataloader.sampler):.5f} "
        )

    eval(model, val_dataloader)

    torch.save(model.state_dict(), "./model.pt")


def load_model(path):
    model = ToyModel(2, 2)
    model.load_state_dict(torch.load(path))

    return model


def eval(model: ToyModel, val_dataloader):
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


def read_data(path):
    a = mmread(path).toarray()
    # adj = np.zeros((max_id, max_id), dtype=int)
    data = []
    print(len(a))
    for i in range(len(a)):
        for j in range(len(a[i])):
            if a[i][j] == 1:
                data.append([i, j, 1])

            if len(data) > 1000:
                break
        if len(data) > 1000:
            break
    print("edges: ", len(data))
    ones = len(data)

    while len(data) < 130 * ones:
        i = randrange(len(a))
        j = randrange(len(a))
        while a[i][j] == 1 or [i, j, 0] in data:
            i = randrange(len(a))
            j = randrange(len(a))
        data.append([i, j, 0])
        if len(data) % 10000 == 0:
            print(len(data))
    df = pd.DataFrame(data, columns=["id_1", "id_2", "label"])
    return df


def main(
    train_model: bool = False, evaluate_model: bool = False, get_data: bool = False
):
    bs = 64
    if get_data:
        print("Reading data...")
        df = read_data()
        df.to_csv("dataset-mtx.csv", index=False)
    else:
        df = pd.read_csv("dataset-mtx.csv")

    labels = df["label"]
    edges = df[["id_1", "id_2"]].to_numpy()

    print("Creating dataset...")
    dataset = EdgeDataset(edges, labels)
    # train_size = int(0.6 * len(dataset))
    # eval_size = len(dataset) - train_size
    # train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])

    model = ToyModel(2, 2).to(device)
    print("Creating data loaders...")
    train_dataloader = DataLoader(
        dataset,
        sampler=RandomSampler(dataset),  # Sampling for training is random
        batch_size=bs,
    )

    evaluation_dataloader = DataLoader(
        dataset,
        sampler=SequentialSampler(
            dataset
        ),  # Sampling for validation is sequential as the order doesn't matter.
        batch_size=bs,
    )
    if train_model:
        print("Starting training...")
        train(model, train_dataloader, evaluation_dataloader, 1000)
    if evaluate_model:
        model = load_model("./model.pt")
        print("Starting evaluation...")
        eval(model, evaluation_dataloader)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "trains or runs edge detection classification model"
    )
    parser.add_argument("--train-model", action="store_true")

    parser.add_argument("--evaluate-model", action="store_true")
    parser.add_argument("--get-data", action="store_true")

    args = parser.parse_args()

    main(args.train_model, args.evaluate_model, args.get_data)
