import numpy as np
from .quat_torch import *

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class CustomLoss(nn.Module):
    def __init__(self, base_loss_func=nn.MSELoss(), weight_base=0.5, weight_a=0.5):
        """
        Custom loss that includes a component to penalize deviation of a_z from 1.

        Args:
        - base_loss_func: The base loss function (e.g., MSE for quaternion prediction).
        - weight_az: Weighting factor for the a_z deviation component of the loss.
        """
        super(CustomLoss, self).__init__()
        self.base_loss_func = base_loss_func
        self.weight_base = weight_base
        self.weight_a = weight_a

    def forward(self, outputs, targets, inputs, epoch):
        """
        Calculate the custom loss.

        Args:
        - outputs: The predicted outputs (e.g., quaternions).
        - targets: The target outputs (e.g., ground truth quaternions).
        - inputs: The input data, where inputs[:, 2] should be the a_z data.

        Returns:
        - The total loss as a PyTorch scalar.
        """
        quaternion_diff = 2*qlog_pytorch(qmult_pytorch(qinverse_pytorch(outputs), targets))
        quaternion_diff = torch.norm(quaternion_diff, dim=1)
        base_loss = self.weight_base * F.mse_loss(quaternion_diff, torch.zeros_like(quaternion_diff), reduction='mean')

        g = torch.tensor([0., 0., 0., 1.], device=outputs.device, dtype=outputs.dtype).expand(outputs.size(0), 4)
        q_inv_outputs = qinverse_pytorch(outputs)
        a_outputs = qmult_pytorch(qmult_pytorch(q_inv_outputs, g), outputs)[:, 1:]
        a_diff = self.weight_a * F.mse_loss(inputs[:, 0:3], a_outputs, reduction='mean')

        total_loss = base_loss + a_diff
        print(f'Epoch {epoch+1}, Base Loss: {base_loss.item():.3f}, A Diff: {a_diff.item():.3f}')

        return total_loss

class CustomDataset(Dataset):
    def __init__(self, inputs, targets):
        """
        Args:
            inputs (numpy.ndarray): The input data, shape (N, 6), N is the number of timesteps,
                                    6 for 3-axis accelerometer and 3-axis gyroscope readings.
            targets (numpy.ndarray): The target data, shape (N, 4), N is the number of timesteps,
                                     4 for quaternion components.
        """
        assert inputs.shape[0] == targets.shape[0], "Inputs and targets must have the same number of timesteps"

        self.inputs = torch.tensor(inputs, dtype=torch.float)
        self.targets = torch.tensor(targets, dtype=torch.float)

    def __len__(self):
        # Return the number of samples (i.e., timesteps in this context)
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        # Retrieve the sample at index `idx` and its corresponding target
        input_sample = self.inputs[idx]
        target_sample = self.targets[idx]

        return input_sample, target_sample

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate=0.5):
        super(NeuralNet, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.layers = nn.ModuleList([nn.Linear(input_size, hidden_size)])
        self.layers.extend([nn.Linear(hidden_size, hidden_size) for _ in range(num_layers - 2)])
        self.layers.append(nn.Linear(hidden_size, output_size))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:  # Apply ReLU and dropout except for the output layer
                x = F.relu(x)
                x = self.dropout(x)
        # normalize because quaternions have unit norm
        x = F.normalize(x, p=2, dim=1)
        return x

class DeepLearning:
    def __init__(self, training_parameters, input_size=6, hidden_size=256, num_layers=5, output_size=4):
        self.model = NeuralNet(input_size, hidden_size, num_layers, output_size)
        self.epochs = training_parameters['epochs']
        self.learning_rate = training_parameters['learning_rate']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = CustomLoss(weight_a=0.7)

    def train(self, datasets):
        self.model.train()
        for dataset in datasets:
            train_dataset = CustomDataset(dataset[0], dataset[1])
            train_loader = DataLoader(train_dataset, batch_size=dataset[0].shape[0], shuffle=True)
            print(f'Training on dataset with {len(train_dataset)} samples')
            for epoch in range(self.epochs):
                for x, y in train_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    self.optimizer.zero_grad()
                    outputs = self.model(x)
                    loss = self.criterion(outputs, y, x, epoch)
                    loss.backward()
                    self.optimizer.step()
                if loss.item() < 1e-3:
                    break
            print("")

    def predict(self, test_dataset):
        test_dataset = torch.tensor(test_dataset, dtype=torch.float)
        self.model.eval()
        test_loader = DataLoader(test_dataset, batch_size=test_dataset.shape[0], shuffle=False)
        outputs = None
        with torch.no_grad():
            for x in test_loader:
                x = x.to(self.device)
                outputs = self.model(x)
        return outputs.cpu().numpy()

    def save_model(self, path):
        """
        Saves the model's state dictionary to the specified path.

        Parameters:
        - path (str): The file path where the model state dictionary should be saved.
        """
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        """
        Loads the model's state dictionary from the specified path.

        Parameters:
        - path (str): The file path to the saved model state dictionary.
        """
        self.model.load_state_dict(torch.load(path))
        self.model.to(self.device)
