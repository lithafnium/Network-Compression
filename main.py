import argparse
from train import Trainer, SMALL_WORLD, ERDOS_RENYI

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--data-type", "-d", type=str, required=True, choices=[SMALL_WORLD, ERDOS_RENYI], default=SMALL_WORLD)
  # TODO(leonard) -- set up argparsing to control archs, training procedure, training params
  args = parser.parse_args()

  t = Trainer(lr=1e-3, batch_size=64, epochs=100, oversample=False, data_type=args.data_type) 
  t.train_and_eval_all_graphs_and_models()

if __name__ == "__main__":
  main()
