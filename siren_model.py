import argparse
import getpass
import json
import os
import random
import torch
from siren import Siren
from torch.utils.data import (
    Dataset,
    DataLoader,
    RandomSampler,
    SequentialSampler,
    random_split,
)
from torchvision import transforms
from torchvision.utils import save_image
from training import Trainer

import pandas as pd
# Set up torch and cuda
dtype = torch.float32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_tensor_type('torch.cuda.FloatTensor' if torch.cuda.is_available() else 'torch.FloatTensor')


class EdgeDataset(Dataset):
    def __init__(self, edges, labels):
        self.edges = edges
        self.labels = labels

    def __len__(self):
        return len(self.edges)

    def __getitem__(self, i):
        return torch.Tensor(self.edges[i]), self.labels[i]

def train(args):
  return 

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("-t", "--train", help="train model", action="store_true")
  parser.add_argument("-ld", "--logdir", help="Path to save logs", default=f"/tmp/{getpass.getuser()}")
  parser.add_argument("-ni", "--num_iters", help="Number of iterations to train for", type=int, default=1)
  parser.add_argument("-lr", "--learning_rate", help="Learning rate", type=float, default=2e-4)
  parser.add_argument("-se", "--seed", help="Random seed", type=int, default=random.randint(1, int(1e6)))
  parser.add_argument("-fd", "--full_dataset", help="Whether to use full dataset", action='store_true')
  parser.add_argument("-iid", "--image_id", help="Image ID to train on, if not the full dataset", type=int, default=15)
  parser.add_argument("-lss", "--layer_size", help="Layer sizes as list of ints", type=int, default=28)
  parser.add_argument("-nl", "--num_layers", help="Number of layers", type=int, default=10)
  parser.add_argument("-w0", "--w0", help="w0 parameter for SIREN model.", type=float, default=30.0)
  parser.add_argument("-w0i", "--w0_initial", help="w0 parameter for first layer of SIREN model.", type=float, default=30.0)

  args = parser.parse_args()
  # Set random seeds
  torch.manual_seed(args.seed)
  torch.cuda.manual_seed_all(args.seed)
  print("creating model...")
  func_rep = Siren(
        dim_in=2,
        dim_hidden=args.layer_size,
        dim_out=1,
        num_layers=args.num_layers,
        final_activation=torch.nn.Identity(),
        w0_initial=args.w0_initial,
        w0=args.w0
    ).to(device)
  
  print("loading dataset...")
  df = pd.read_csv("dataset-mtx.csv")
  labels = df["label"]
  edges = df[["id_1", "id_2"]].to_numpy()

  print("creating dataset...")
  labels = torch.Tensor(labels)
  labels = torch.unsqueeze(labels, dim=-1)
  edges = torch.Tensor(edges)

  if args.train:
      trainer = Trainer(func_rep, lr=args.learning_rate)

      trainer.train(edges, labels, num_iters=args.num_iters)
      torch.save(trainer.representation.state_dict(), "./siren-model.pt")
      