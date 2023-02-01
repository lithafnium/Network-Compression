import torch
import utils.configuration as cf
from utils.datasets import get_data
import graph_model
import train
import argparse
import json



def run_clip(data_file, model):
    conf = cf.clip_example(data_file, use_cuda=True, download=False)
    train_loader, valid_loader, test_loader = get_data(data_file)
    
    max_throughput = 2
    for i in range(4, 12):
        max_throughput = pow(2, i)
        print("max throughput: ", max_throughput)
        model = graph_model.UnSqueeze(
            num_features=2, num_classes=2, max_throughput_multiplier=max_throughput)
        model.to("cuda")

        best_model = train.best_model(
            graph_model.UnSqueeze(
                num_features=2, num_classes=2, max_throughput_multiplier=max_throughput).to(conf.device), goal_acc=conf.goal_acc)
        # opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        opt = torch.optim.Adam(model.parameters(), lr=2e-4)

        lamda_scheduler = train.lamda_scheduler(
            conf, warmup=5, warmup_lamda=0.0, cooldown=1)

        tracked = ['train_loss', 'train_acc',
                'train_lip_loss', 'val_loss', 'val_acc']
        history = {key: [] for key in tracked}
        # -----------------------------------------------------------------------------------
        # cache for the lipschitz update
        cache = {'counter': 0}
        print(conf.device)

        print("Train model: {}".format(conf.model))
        for i in range(conf.epochs):
            # print(25*"<>")
            # print(50*"|")
            # print(25*"<>")
            print('Epoch', i)

            # train_step
            train_data = train.train_step(
                conf, model, opt, train_loader, valid_loader, cache)
            
            if i % 150 == 0 and opt.param_groups[0]['lr'] > 1e-6:
                opt.param_groups[0]['lr'] *= 0.75
                print("==== new lr", opt.param_groups[0]['lr'])
            # ------------------------------------------------------------------------
            # validation step

            # ------------------------------------------------------------------------
            # update history
            for key in tracked:
                if key in train_data:
                    history[key].append(train_data[key])

            # ------------------------------------------------------------------------
            lamda_scheduler(conf, train_data['train_acc'])
            best_model(train_data['train_acc'], 0, model=model)

        # -----------------------------------------------------------------------------------
        # Test the model afterwards
        # -----------------------------------------------------------------------------------
        conf.attack.attack_iters = 100
        test_data = train.test_step(
            conf, best_model.best_model, test_loader, attack=conf.attack)

        history["file"] = data_file
        history["test_data"] = test_data
        history["max_throughput"] = max_throughput
        history["model"] = model.model_name
        print(history)
        with open(f"{max_throughput}-{file_name[8:]}-json_output.json", "w") as f:
            json_object = json.dumps(history, indent=4)
            f.write(json_object)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", "-f", type=str, required=True)
    parser.add_argument("--model", "-m", type=str, default="unsqueeze")
    args = parser.parse_args()

    model = args.model
    file_name = args.file
    run_clip(file_name, model)
