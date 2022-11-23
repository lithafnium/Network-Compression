import argparse
from train import Trainer, SMALL_WORLD, ERDOS_RENYI

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--data-type", "-d", type=str, required=True, choices=[SMALL_WORLD, ERDOS_RENYI], default=SMALL_WORLD)
  # TODO(leonard) -- set up argparsing to control archs, training procedure, training params
  parser.add_argument("--oversample", "-o", action="store_true")
  args = parser.parse_args()

  t = Trainer(
      lr=1e-2, 
      batch_size=512, 
      epochs=100, 
      oversample=args.oversample,
      data_type=args.data_type, 
      num_workers=18,
      one_hot=False,
    ) 
  t.train_and_eval_all_graphs_and_models()

if __name__ == "__main__":
  main()
