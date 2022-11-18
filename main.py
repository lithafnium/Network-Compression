import argparse
from train import Trainer

def main():
  # TODO(leonard) -- set up argparsing to control archs, training procedure, training params, data type
  t = Trainer(lr=1e-2, batch_size=512, epochs=100, oversample=True) 
  t.train_and_eval_all_graphs_and_models()

if __name__ == "__main__":
  main()
