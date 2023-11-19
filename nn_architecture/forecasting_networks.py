'''This file contains network architectures for forecasting tasks'''

from enum import auto
from turtle import forward
import torch
import torch.nn as nn
from torch import Tensor

from nn_architecture.ae_networks import Autoencoder


class LSTMForecasting(nn.Module):
    '''LSTM network for forecasting tasks'''

    def __init__(
        self, 
        n_channels: int, 
        hidden_dim: int, 
        sequence_length: int,
        n_channels_out: int = None, 
        num_layers: int = 1, 
        dropout: float = 0.0,
        ):
        super().__init__()
        
        # network parameters
        self.hidden_size = hidden_dim
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        self.n_channels_out = n_channels_out if n_channels_out is not None else n_channels
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # network layers
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True)
        self.fc_in = nn.Linear(n_channels, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, self.n_channels_out)

    def forward(self, x: Tensor) -> Tensor:
        '''Forward pass'''
        # Encode input
        x = self.fc_in(x)

        # Forward propagate LSTM
        out = self.lstm(x)[0][:, -1, :].unsqueeze(1)

        # Decode the hidden state of the last time step
        out = self.fc_out(out)
        return out
    
    
class AEForecasting(LSTMForecasting):
    '''AE-Wrapper for forecasting networks'''
    
    def __init__(
        self, 
        autoencoder: Autoencoder,
        hidden_dim: int, 
        num_layers: int = 1, 
        dropout: float = 0.0,
        encode_output: bool = False,
        fit_autoencoder = False,
    ):
        super().__init__(
            n_channels=autoencoder.output_dim,
            hidden_dim=hidden_dim,
            sequence_length=autoencoder.output_dim_2,
            n_channels_out=autoencoder.input_dim,
            num_layers=num_layers,
            dropout=dropout,
            )
        
        self.encode_output = encode_output
        
        # freeze autoencoder if already trained
        if not fit_autoencoder:
            autoencoder.eval()
            for param in autoencoder.parameters():
                param.requires_grad = False
            # for param in autoencoder.params:
            #     param.requires_grad = False
        self.autoencoder = autoencoder
        
    def forward(self, input):
        # encode input x
        x = self.autoencoder.encode(input)
        
        # access forward pass of parent class
        x = super(AEForecasting, self).forward(x)
        
        if self.encode_output:
            # concatenate input and output
            x = torch.cat((input[:, 1:, :], x), dim=1)
            
            # encode output
            x = self.autoencoder.encode(x)
        
        return x
        