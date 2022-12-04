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

from collections import defaultdict
from graph_model import BlockModel, UnSqueeze, WideNet, OneHotNet
from coin.siren import Siren
from size_estimator import SizeEstimator
import tqdm
import math
import multiprocessing as mp
import numpy as np
import json
from scipy import sparse
import matplotlib.pyplot as plt
import random
import copy
import wandb
from time import time
from scipy.io import mmread
from sklearn.metrics import auc, roc_curve
import networkx as nx
# import multiprocess as mp
from itertools import combinations


dtype = torch.float32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SMALL_WORLD = "small-world"
ERDOS_RENYI = "erdos-renyi"


class EdgeDataset(Dataset):
    def __init__(self, edges, labels):
        self.edges = edges
        self.labels = labels

    def __len__(self):
        return len(self.edges)

    def __getitem__(self, i):
        return self.edges[i], self.labels[i]


class Trainer:
    def __init__(
        self,
        lr=1e-3,
        batch_size=16,
        epochs=100,
        oversample=False,
        data_type=SMALL_WORLD,
        print_freq=1,
        min_layers=6,
        max_layers=6,
        max_nodes=64,
        min_nodes=64,
        num_workers=32,
        one_hot=False,
        order="none"
    ):
        # self.criterion = nn.CrossEntropyLoss()
        self.criterion = nn.MSELoss()
        # self.graph_sizes = [100]
        self.graph_sizes = [1000]
        # self.graph_densities = [0.050, 0.100, 0.250, 0.501]
        # self.graph_densities = [0.100, 0.250, 0.501]
        self.graph_densities = [0.404]
        self.small_world_p = [0.5]
        self.order = "degree150"
        # self.order = "none"
        self.min_layers = min_layers
        self.max_layers = max_layers
        self.max_nodes = max_nodes
        self.min_nodes = min_nodes

        self.lr = lr
        # Pulled momentum and WD values from PixMix
        self.momentum = 0.9
        self.weight_decay = 5e-4

        self.epochs = epochs
        self.batch_size = batch_size
        self.oversample = oversample
        self.data_type = data_type

        self.one_hot = one_hot

        self.num_workers = num_workers
        self.model_info = {}

    def save_plot(self, path, loss_values, num_epochs):
        plt.clf()
        plt.plot([i + 1 for i in range(num_epochs)], loss_values)
        plt.savefig(f"plots/{path}.png")

    def find_optimal_num_workers(self, dataset):
        best_workers, min_time = None, math.inf
        print(f"Num workers range: {2} to {mp.cpu_count()}")
        for num_workers in range(2, mp.cpu_count(), 2):
            print(f"Testing with {num_workers}")
            train_loader = DataLoader(dataset,
                                      shuffle=True,
                                      num_workers=num_workers,
                                      batch_size=self.batch_size,
                                      pin_memory=True
                                      )
            start = time()
            for i, data in enumerate(train_loader, 0):
                pass
            end = time()
            print("Finish with:{} second, num_workers={}".format(
                end - start, num_workers))
            if end - start < min_time:
                best_workers, min_time = num_workers, end - start

        print("Done testing for optimal num_workers")
        return best_workers

    def get_data(self, path, oversample_p=1.0):
        """ 
        Read and load a mtx file as EdgeDataset 
        """
        a = mmread(path)
        # Sparse to dense
        if isinstance(a, sparse.coo_matrix):
            a = a.toarray()
        labels = []
        edges = []

        G = nx.from_numpy_array(a)
        print("***** Graph Properties *****")
        print("Size: ", G.number_of_nodes())
        print("Density: ", nx.density(G))
        print("# Connected Comps: ", nx.number_connected_components(G))
        a = torch.tensor(a).float()

        coords = torch.ones(a.shape).nonzero(as_tuple=False).float() 
        coords = coords / (a.shape[0] - 1) - 0.5
        # Convert to range [-1, 1]
        coords *= 2

        print("coords?", coords)
        labels = a.reshape(1, -1).T
        # labels *= 255
        print("labels", labels)
        print(coords.shape, labels.shape)
        print(coords[0], labels[0])

        train_dataset = EdgeDataset(coords, labels)
        val_dataset = EdgeDataset(coords, labels)
        # val_dataset = EdgeDataset(og_edges, og_labels)
        return train_dataset, val_dataset

    def eval(self, model, val_dataloader: DataLoader, path: str):
        print("Evaluating...")
        model.eval()
        val_acc = 0
        val_epoch_loss = 0

        auc_roc = 0
        true_labels = []
        all_labels = []
        for X_val_batch, y_val_batch in val_dataloader:
            b_nodes = X_val_batch.to(device)
            b_labels = y_val_batch.to(device)

            y_val_pred = model(b_nodes)
            y_pred_softmax = torch.log_softmax(y_val_pred, dim=1)
            _, y_pred_tags = torch.max(y_pred_softmax, dim=1)

            true_labels.extend(b_labels.detach().cpu().numpy())
            all_labels.extend(y_pred_softmax.detach().cpu().numpy())

            correct_pred = (y_pred_tags == b_labels).float()
            acc = correct_pred.sum() / len(correct_pred)
            acc = torch.round(acc * 100)

            val_acc += acc

            val_loss = self.criterion(y_val_pred, b_labels.long())
            val_epoch_loss += val_loss.item() * y_val_batch.size(0)

        avg_loss = val_epoch_loss / len(val_dataloader.sampler)
        print("Test Loss: {:.5f}".format(avg_loss))

        fpr, tpr, thresholds = roc_curve(
            np.array(true_labels), np.array(all_labels)[:, 1])
        auc_roc = auc(fpr, tpr)

        print("Test Model Accuracy: {:.3f}%".format(
            (val_acc / len(val_dataloader)).item()))
        print("Test Model auc_roc: {:.3f}".format(auc_roc))
        # estimator = SizeEstimator(model, input_size=(2,))
        self.model_info[path] = {
            "accuracy": (val_acc / len(val_dataloader)).item(),
        }
        # wandb.log({"loss": avg_loss, "accuracy": (val_acc / len(val_dataloader)).item()})

        # with open(f"accuracy/{path}-accuracy.json", "w") as f:
        #     out = json.dumps(self.model_info, indent=4)
        #     f.write(out)

        # torch.save(model.state_dict(), f"models/{path}.pt")

    def train(
        self,
        model,
        train_dataloader,
        val_dataloader,
        epochs,
        path,
    ):
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        # optimizer = optim.SGD(
        #     model.parameters(),
        #     self.lr,
        #     momentum=self.momentum,
        #     weight_decay=self.weight_decay,
        #     nesterov=True
        # )
        loss = []
        print(f"Data path: {path}")
        # for epoch_i in tqdm.trange(epochs):
        for epoch_i in range(epochs):
            train_epoch_loss = 0
            model.train()

            train_acc = 0
            for j, data in enumerate(train_dataloader):
                x_train_batch, y_train_batch = data
                b_nodes = x_train_batch.to(device)
                b_labels = y_train_batch.to(device)
                # print(b_nodes.shape, b_labels.shape)
                # print("b_nodes", b_nodes)
                # print("b_labels", b_labels)
                optimizer.zero_grad()
                y_train_pred = model(b_nodes)
                # print(y_train_pred.size(), b_labels.size())
                # print(y_train_pred.dtype, b_labels.dtype)
                train_loss = self.criterion(y_train_pred, b_labels)
                train_loss.backward() 
                optimizer.step()

                train_epoch_loss += train_loss.item() * x_train_batch.size(0)
                # print("y_train_pred", y_train_pred)
                # _, y_pred_tags = torch.max(y_train_pred, dim=1)
                # print("b_labels", b_labels)
                # print("y_pred_tags", y_pred_tags)
                # print("y_train_pred", y_train_pred)

                # one_pred = 0
                # for pred in y_pred_tags:
                #     if pred.item() == 1:
                #         one_pred += 1

                # print("one Pred: ", one_pred)
                # print("percent pred ones: ", one_pred / len(y_pred_tags))

                # train_loss = self.criterion(y_train_pred, b_labels.long())
                # if j % 500 == 0:
                #     print("y_train_pred", y_train_pred)
                #     print("train_loss", train_loss)
                # train_loss.backward()
                # optimizer.step()

                # correct_pred = (y_pred_tags == b_labels).float()

                # for i in range(len(y_pred_tags)):
                #     if y_pred_tags[i] != b_labels[i]:
                #         print("misclassified on input", x_train_batch[i])
                #         print("predicted", y_pred_tags[i])
                # input()

                # acc = correct_pred.sum() / len(correct_pred)
                # acc = torch.round(acc * 100)

                # train_acc += acc
                # train_epoch_loss += train_loss.item() * x_train_batch.size(0)

            # Sanity check, especially if oversampling and/or doing weird things to data
            # if epoch_i > 0 and epoch_i % 10 == 0:
            #     self.eval(model, val_dataloader, path)

            avg_loss = train_epoch_loss / len(train_dataloader.sampler)
            loss.append(avg_loss)
            print(
                f"Epoch {epoch_i + 1}: | Train Loss: {avg_loss:.5f} "
            )
            # print("Model accuracy: {:.3f}%".format(
            #     (train_acc / len(train_dataloader)).item()))
            # wandb.log({"loss": avg_loss, "accuracy": (
            #     train_acc / len(train_dataloader)).item()})

        # TODO(leonard): fix this hacky af trick
        plot_path = path.strip(".mtx")
        print("plot_path", plot_path)
        self.save_plot(plot_path, loss, epochs)
        self.eval(model, val_dataloader, path)
        torch.save(model.state_dict(), f"models/{plot_path}.pt")
        # wandb.finish()
        # torch.save(model.state_dict(), f"./models/{path}.pt")

    def train_and_eval_single_graph_with_model(self, model, data_path):
        train_dataset, val_dataset = self.get_data(data_path)
        train_dataloader = DataLoader(
            train_dataset,
            # sampler=RandomSampler(train_dataset),  # Sampling for training is random
            shuffle=True,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=self.num_workers,
        )

        evaluation_dataloader = DataLoader(
            val_dataset,
            # sampler=SequentialSampler(
            #     val_dataset
            # ),  # Sampling for validation is sequential as the order doesn't matter.
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=self.num_workers,
        )

        # TODO(leonard): fix this hackkyy ass code
        data_path = data_path.strip("data/")
        self.train(
            model,
            train_dataloader,
            evaluation_dataloader,
            epochs=self.epochs,
            path=f"{model.model_name}-oversample{self.oversample}-{data_path}"
        )

    # TODO(leonard): def this out to run single experiments instead of things in loops
    # Def out graph and model layers; specify model first, then data

    def train_and_eval_all_graphs_and_models(self):
        """
        Run experiments across all types of graph data and models
        """

        # model = BlockModel(num_features=2, num_classes=2,
        #                    num_layers=18, num_nodes=256)
        model = Siren(
            dim_in=2,
            dim_hidden=28,
            dim_out=1,
            num_layers=10,
            final_activation=torch.nn.Identity(),
            w0=30.0,
            w0_initial=30.0
        )
        model.to(device)
        # data_path = '/data/leonardtang/cs222proj/data/mtx_graphs/socfb-Harvard1.mtx'
        data_path = 'data/graph-1000-0.501-small-world-Ordernone-p-0.5.mtx'
        self.train_and_eval_single_graph_with_model(model, data_path)
        # Temp return
        return

        for graph_size in self.graph_sizes:
            for graph_density in self.graph_densities:
                # Keeping this so it aligns with the .3f file names, fix in the future
                graph_density = format(graph_density, '.3f')

                for p in self.small_world_p:
                    # TODO(ltang): fix this hacky ass shit:
                    # In the future, generate these on the fly instead of saving files then loading
                    if self.data_type == ERDOS_RENYI:
                        data_path = f"data/graph-{graph_size}-{graph_density}-Erdos-Renyi-Order{self.order}-p-{p}.mtx"
                    elif self.data_type == SMALL_WORLD:
                        data_path = f"data/graph-{graph_size}-{graph_density}-small-world-Order{self.order}-p-{p}.mtx"
                    else:
                        raise Exception('Data type not handled')

                    print(f"Grabbing {data_path}")

                    # model = UnSqueeze(max_throughput_multiplier=1024)
                    # model = WideNet(width=10000)
                    # model = Siren(dim_in=2, dim_hidden=128, dim_out=2, num_layers=12)
                    model = BlockModel(
                        num_features=2, num_classes=2, num_layers=6, num_nodes=256)
                    model.to(device)

                    # wandb.init(project="training-runs", entity="cs222", config={
                    #     "learning_rate": self.lr,
                    #     "epochs": self.epochs,
                    #     "batch_size": self.batch_size,
                    #     "graph_size": graph_size,
                    #     "graph_density": graph_density,
                    #     "graph_file": data_path,
                    #     "model_name": model.model_name,
                    #     "oversampling": self.oversample,
                    #     "graph_type": self.data_type,
                    # })
                    # if model.model_name == "widenet":
                    #     wandb.run.name = f"{model.model_name}-{model.width}-oversample{self.oversample}-{data_path}"
                    # elif model.model_name == "block":
                    #     wandb.run.name = f"{model.model_name}-{6}-{256}-oversample{self.oversample}-{data_path}"
                    # else:
                    #     wandb.run.name = f"{model.model_name}-oversample{self.oversample}-{data_path}"
                    # # wandb.run.name = f"40layers-{model.model_name}-oversample{self.oversample}-{data_path}"
                    # wandb.run.save()
                    # model = torch.nn.DataParallel(model)
                    self.train_and_eval_single_graph_with_model(
                        model, data_path)

        print("Saving output loss and size estimates...")

        with open("loss_sizes.json", "w") as f:
            out = json.dumps(self.model_info, indent=4)
            f.write(out)
