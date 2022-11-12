from train import Trainer

if __name__ == "__main__":
  t = Trainer(lr=1e-4) 
  t.train_and_eval(batch_size=128, epochs=100)
