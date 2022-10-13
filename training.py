import torch
import tqdm
from collections import OrderedDict


class Trainer():
    def __init__(self, representation, lr=1e-3, print_freq=1):
        """Model to learn a representation of a single datapoint.
        Args:
            representation (siren.Siren): Neural net representation of image to
                be trained.
            lr (float): Learning rate to be used in Adam optimizer.
            print_freq (int): Frequency with which to print losses.
        """
        self.representation = representation
        self.optimizer = torch.optim.Adam(self.representation.parameters(), lr=lr)
        self.print_freq = print_freq
        self.steps = 0  # Number of steps taken in training
        self.loss_func = torch.nn.MSELoss()
        self.best_vals = {'loss': 1e8}
        self.logs = {'loss': []}
        # Store parameters of best model (in terms of highest PSNR achieved)
        self.best_model = OrderedDict((k, v.detach().clone()) for k, v in self.representation.state_dict().items())

    def train(self, coordinates, features, num_iters):
        """Fit neural net to image.
        Args:
            coordinates (torch.Tensor): Tensor of coordinates.
                Shape (num_points, coordinate_dim).
            features (torch.Tensor): Tensor of features. Shape (num_points, feature_dim).
            num_iters (int): Number of iterations to train for.
        """
        with tqdm.trange(num_iters, ncols=100) as t:
            for i in t:
                # Update model
                self.optimizer.zero_grad()
                predicted = self.representation(coordinates)
                # print(predicted.size())
                # print(features.size())
                loss = self.loss_func(predicted, features)
                loss.backward()
                self.optimizer.step()

                # Calculate psnr
                # psnr = get_clamped_psnr(predicted, features)

                # Print results and update logs
                log_dict = {'loss': loss.item()}
                t.set_postfix(**log_dict)
                for key in ['loss']:
                    self.logs[key].append(log_dict[key])
                # Update best values
                if loss.item() < self.best_vals['loss']:
                    
                    self.best_vals['loss'] = loss.item()
                # predicted = torch.squeeze(predicted)
                # features = torch.squeeze(features)
                with open("out.txt", "a") as f:
                    f.write(str(predicted[:100]))
                    f.write(str(features[:100]))

                    correct_pred = (predicted == features).float()
                    acc = correct_pred.sum() / len(correct_pred)
                    acc = torch.round(acc * 100) / len(predicted)
                    f.write("Loss: " + str(loss.item()) + " | " + "Accuracy: " + str(acc.item()) +  "\n")
                # if psnr > self.best_vals['psnr']:
                #     self.best_vals['psnr'] = psnr
                #     # If model achieves best PSNR seen during training, update
                #     # model
                #     if i > int(num_iters / 2.):
                #         for k, v in self.representation.state_dict().items():
                #             self.best_model[k].copy_(v)