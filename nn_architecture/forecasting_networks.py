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
    

class TradeNet(nn.Module):
    '''LSTM network for trading'''

    def __init__(
        self, 
        n_channels: int, 
        hidden_dim: int, 
        sequence_length: int,
        n_channels_out: int = None, 
        num_layers: int = 1, 
        dropout: float = 0.0,
        n_portfolio: int = 0,
        # cash: bool = True,
        ):
        super().__init__()
        
        # network parameters
        self.hidden_size = hidden_dim
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        self.n_channels_out = n_channels_out if n_channels_out is not None else n_channels
        self.portfolio = n_portfolio > 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # network layers
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, dropout=0.0, batch_first=True)
        if n_portfolio == 0:
            self.fc_in = nn.Linear(n_channels, hidden_dim)
        else:
            # intermediate step while encoding incoming stocks to encode also portfolio information into array
            self.fc_in1 = nn.Sequential(nn.Linear(n_channels, n_portfolio), nn.Sigmoid())
            self.fc_in2 = nn.Linear(n_portfolio, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, self.n_channels_out)
        
        # activation
        self.act_orders = nn.Tanh()
        self.act_softmax = nn.Softmax(dim=-1)
        self.act_buy = nn.ReLU()
        self.act_sell = lambda x: torch.where(x < 0, x, torch.zeros_like(x))

    def forward(self, stocks: Tensor, portfolio: Tensor=None) -> Tensor:
        '''Forward pass'''
        # Encode input
        if portfolio is None:
            stocks = self.fc_in(stocks)
        else:
            # mark each entry of portfolio > 0 with 1 and rest with 0
            portfolio = torch.where(portfolio > 0, torch.ones_like(portfolio), torch.zeros_like(portfolio))
            stocks = self.fc_in1(stocks)
            stocks = stocks + portfolio
            stocks = self.fc_in2(stocks)

        # Forward propagate LSTM
        stocks = self.lstm(stocks)[0][:, -1, :]

        # Decode the hidden state of the last time step
        orders_all = self.act_orders(self.fc_out(stocks))
        
        # split orders into buy and sell orders
        # index_buy = torch.where(orders_all > 1)        
        # orders_all[index_buy] = self.activation_orders(orders_all[index_buy])
        orders_buy = self.act_buy(orders_all)
        # orders_buy = self.act_softmax(orders_buy)
        # scale all orders to sum up to 1 but keep 0s
        orders_buy = orders_buy / torch.sum(orders_buy, dim=-1, keepdim=True)
        orders_sell = self.act_sell(orders_all)
        if portfolio is not None:
            orders_sell = orders_sell * portfolio.squeeze(1)
        
        return orders_buy, orders_sell
    
    
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
        