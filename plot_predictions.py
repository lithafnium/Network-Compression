import argparse
from graph_model import BlockModel, WideNet, UnSqueeze, OneHotNet
import torch
import matplotlib.pyplot as plt
import numpy as np


def plot_predictions(path):
    model = BlockModel(
        num_features=2, num_classes=2, num_layers=6, num_nodes=256)

    model.load_state_dict(torch.load(path))

    model.eval()
    X = []
    y = []

    for i in range(100):
        for j in range(100):
            input = torch.Tensor([i, j])
            y_val_pred = model(input)
            y_pred_softmax = torch.log_softmax(y_val_pred, dim=0)
            _, y_pred_tags = torch.max(y_pred_softmax, dim=0)
            y.append(y_pred_tags.item())
            X.append([i, j])
    y = np.array(y)
    X = np.array(X)

    print(X.shape)
    print(y.shape)

    fig = plt.figure()
    colors = ["blue" if y_ == 1 else "red" for y_ in y]

    plt.figure(figsize=(15, 15), dpi=160)
    plt.scatter(X[:, 0], X[:, 1], c=colors)
    plt.xticks(np.arange(0, 101, 5))
    plt.yticks(np.arange(0, 101, 5))
    plt.grid()
    plt.savefig('plot.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", "-m", type=str, required=True)
    args = parser.parse_args()
    plot_predictions(args.model_path)
