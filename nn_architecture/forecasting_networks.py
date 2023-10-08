'''This file contains network architectures for forecasting tasks'''

import torch
import torch.nn as nn
from torch import Tensor


class LSTMForecasting(nn.Module):
    '''LSTM network for forecasting tasks'''

    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.fc_in = nn.Linear(input_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, x: Tensor) -> Tensor:
        '''Forward pass'''
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, self.hidden_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, self.hidden_size, self.hidden_size).to(x.device)

        # Encode input
        x = self.fc_in(x)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Decode the hidden state of the last time step
        out = self.fc_out(out[:, -1, :])
        return out