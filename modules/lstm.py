import numpy as np

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

class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super(TCNBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=(kernel_size-1) * dilation, dilation=dilation)
        self.relu = nn.ReLU()
        self.net = nn.Sequential(self.conv, self.relu)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.init_weights()

    def init_weights(self):
        self.conv.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCN(nn.Module):
    def __init__(self, input_size=6, num_channels=[64, 64, 128], kernel_size=3, output_size=4):
        super(TCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TCNBlock(in_channels, out_channels, kernel_size, dilation=dilation_size)]

        self.tcn = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        # TCN expects input shape as (batch_size, num_channels, length),
        # so permute your input accordingly
        x = x.permute(0, 2, 1)
        y = self.tcn(x)
        o = self.linear(y[:, :, -1])
        return o

class OrientationTrackerLSTM(nn.Module):
    def __init__(self, input_size=6, hidden_size=128, num_layers=5, output_size=4):
        super(OrientationTrackerLSTM, self).__init__()

        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)  # Maps to quaternion output

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # lstm_out shape: (batch_size, sequence_length, hidden_size)
        y_pred = self.fc(lstm_out)  # Apply the FC layer at every timestep
        # normalize the quaternion since rotation is a unit quaternion
        y_pred = y_pred / torch.norm(y_pred, p=2, dim=-1, keepdim=True)
        return y_pred

class LSTM:
    def __init__(self, training_parameters, input_size=6, hidden_size=128, num_layers=2, output_size=4):
        self.model = OrientationTrackerLSTM(input_size, hidden_size, num_layers, output_size)
    # def __init__(self, training_parameters, input_size=6, num_channels=[64, 64, 128], kernel_size=3, output_size=4):
    #     self.model = TCN(input_size, num_channels, kernel_size, output_size)
        self.epochs = training_parameters['epochs']
        self.learning_rate = training_parameters['learning_rate']
        self.batch_size = training_parameters['batch_size']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def train(self, datasets):
        self.model.train()
        for dataset in datasets:
            train_dataset = CustomDataset(dataset[0], dataset[1])
            train_loader = DataLoader(train_dataset, batch_size=dataset[0].shape[0], shuffle=False)
            for epoch in range(self.epochs):
                for x, y in train_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    self.optimizer.zero_grad()
                    outputs = self.model(x)
                    loss = self.criterion(outputs, y)
                    loss.backward()
                    self.optimizer.step()
                print(f'Epoch {epoch+1}/{self.epochs}, Loss: {loss.item()}')

    # def train(self, train_dataset):
    #     self.model.train()
    #     print("train_dataset[0].shape:", train_dataset[0].shape)
    #     train_dataset = CustomDataset(train_dataset[0], train_dataset[1])
    #     train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
    #     print("train_dataset[0].shape:", train_dataset[0].shape)
    #     for epoch in range(self.epochs):
    #         for x, y in train_loader:
    #             x, y = x.to(self.device), y.to(self.device)
    #             self.optimizer.zero_grad()
    #             print("x.shape:", x.shape)
    #             outputs = self.model(x)
    #             loss = self.criterion(outputs, y)
    #             loss.backward()
    #             self.optimizer.step()
    #         print(f'Epoch {epoch+1}/{self.epochs}, Loss: {loss.item()}')

    def predict(self, test_dataset):
        test_dataset = torch.tensor(test_dataset, dtype=torch.float)
        self.model.eval()
        test_loader = DataLoader(test_dataset, batch_size=test_dataset.shape[0], shuffle=False)
        outputs = None
        with torch.no_grad():
            for x in test_loader:
                print("x.shape:", x.shape)
                x = x.to(self.device)
                outputs = self.model(x)
                print("outputs.shape:", outputs.shape)
        return outputs.cpu().numpy()

    # def predict(self, test_dataset):
    #     test_dataset = torch.tensor(test_dataset, dtype=torch.float)
    #     self.model.eval()
    #     test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    #     predictions = []
    #     with torch.no_grad():
    #         for x in test_loader:
    #             x = x.to(self.device)
    #             outputs = self.model(x)
    #             predictions.append(outputs.cpu().numpy())
    #     return np.concatenate(predictions, axis=0)

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
