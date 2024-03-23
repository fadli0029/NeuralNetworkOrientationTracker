import numpy as np

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F

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

# class NeuralNet(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, output_size):
#         super(NeuralNet, self).__init__()
#         self.layers = nn.ModuleList()
#         # Input layer
#         self.layers.append(nn.Linear(input_size, hidden_size))
#         # Hidden layers
#         for _ in range(num_layers - 2):
#             self.layers.append(nn.Linear(hidden_size, hidden_size))
#         # Output layer
#         self.layers.append(nn.Linear(hidden_size, output_size))

#     def forward(self, x):
#         for i, layer in enumerate(self.layers):
#             x = layer(x)
#             if i < len(self.layers) - 1:  # Apply activation function except for the output layer
#                 x = F.relu(x)
#         return x

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
        return x

class DeepLearning:
    def __init__(self, training_parameters, input_size=6, hidden_size=256, num_layers=5, output_size=4):
        self.model = NeuralNet(input_size, hidden_size, num_layers, output_size)
        # self.epochs = training_parameters['epochs']
        self.epochs = 100
        self.learning_rate = training_parameters['learning_rate']
        self.batch_size = training_parameters['batch_size']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        # self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

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
                    loss = self.criterion(outputs, y)
                    loss.backward()
                    self.optimizer.step()
                print(f'Epoch {epoch+1}, Loss: {loss.item()}')
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
