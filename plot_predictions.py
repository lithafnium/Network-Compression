import argparse
from graph_model import BlockModel, WideNet, UnSqueeze, OneHotNet
import torch
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def plot(model_args, save_path, coords, model_path="", model=None):

    if model_path != "" and model == None:
        model = BlockModel(**model_args)
        model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    X = []
    y = []

    # for i in range(100):
    #     for j in range(100):
    #         input = torch.Tensor([i, j])
    #         input.to(device)
    #         y_val_pred = model(input)
    #         y_pred_softmax = torch.log_softmax(y_val_pred, dim=0)
    #         _, y_pred_tags = torch.max(y_pred_softmax, dim=0)
    #         y.append(y_pred_tags.item())
    #         X.append([i, j])

    for i, c in enumerate(coords): 
        input = coords[i].to(device) 
        y_val_pred = model(input)
        y_pred_softmax = torch.log_softmax(y_val_pred, dim=0)
        _, y_pred_tags = torch.max(y_pred_softmax, dim=0)

        y.append(y_pred_tags.item())
        X.append([input[0].cpu(), input[1].cpu()])

    y = np.array(y)
    X = np.array(X)

    print(X.shape)
    print(y.shape)

    fig = plt.figure()
    colors = ["blue" if y_ == 1 else "red" for y_ in y]

    plt.figure(figsize=(15, 15), dpi=160)
    plt.scatter(X[:, 0], X[:, 1], c=colors)
    # plt.xticks(np.arange(0, 101, 5))
    # plt.yticks(np.arange(0, 101, 5))
    plt.grid()
    plt.savefig(f"eval_image_plots/{save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", "-m", type=str, required=True)
    args = parser.parse_args()
    plot(args.model_path)
