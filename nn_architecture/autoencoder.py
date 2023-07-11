import os
import random
from typing import Optional

import pandas as pd
import torch
import torch.nn as nn

# from utils.get_filter import moving_average as filter

class Autoencoder(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers=3, dropout=0.1):
        super(Autoencoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

    def encode(self, data):
        raise NotImplementedError

    def decode(self, encoded):
        raise NotImplementedError


class TransformerAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout, hidden_dec=256, **kwargs):
        super(TransformerAutoencoder, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=5, dim_feedforward=hidden_dim,
                                                        dropout=dropout)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        self.linear_enc = nn.Linear(input_dim, output_dim)

        self.decoder_layer = nn.TransformerEncoderLayer(d_model=output_dim, nhead=5, dim_feedforward=hidden_dim,
                                                        dropout=dropout)
        self.decoder = nn.TransformerEncoder(self.decoder_layer, num_layers=num_layers)
        self.linear_dec = nn.Linear(output_dim, input_dim)

        self.tanh = nn.Tanh()

    def forward(self, data):
        x = self.encode(data.to(self.device))
        x = self.decode(x)
        return x

    def encode(self, data):
        x = self.encoder(data.to(self.device))
        x = self.linear_enc(x)
        x = self.tanh(x)
        return x

    def decode(self, encoded):
        x = self.decoder(encoded)
        x = self.linear_dec(x)
        x = self.tanh(x)
        return x

    def save(self, path):
        path = '../trained_ae'
        file = f'ae_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.pth'
        # torch.save(save, os.path.join(path, file))


class TransformerAutoencoder_v0(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout, hidden_dec=256, **kwargs):
        super(TransformerAutoencoder_v0, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=5, dim_feedforward=hidden_dim,
                                                        dropout=dropout)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        self.linear_enc = nn.Linear(input_dim, output_dim)

        self.linear_dec = nn.Linear(output_dim, input_dim)

        self.tanh = nn.Tanh()

    def forward(self, data):
        x = self.encode(data.to(self.device))
        x = self.decode(x)
        return x

    def encode(self, data):
        x = self.encoder(data.to(self.device))
        x = self.linear_enc(x)
        return x

    def decode(self, encoded):
        x = self.linear_dec(encoded)
        return x

    def save(self, path):
        path = '../trained_ae'
        file = f'ae_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.pth'
        # torch.save(save, os.path.join(path, file))


class LSTMAutoencoder(Autoencoder):
    def __init__(self, input_dim, output_dim, sequence_length, hidden_dim=256, num_layers=3, dropout=0.1, activation=nn.Sigmoid(), **kwargs):
        super(LSTMAutoencoder, self).__init__(input_dim, output_dim, hidden_dim, num_layers, dropout)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.activation = activation
        self.sequence_length = sequence_length

        # encoder block
        self.batchnorm = nn.BatchNorm1d(self.input_dim)
        self.enc_lin_in = nn.Linear(self.input_dim, self.hidden_dim)
        self.enc_lstm = nn.LSTM(self.hidden_dim, self.output_dim, num_layers=self.num_layers, dropout=self.dropout, batch_first=True)
        self.enc_lin_out = nn.Linear(self.output_dim, self.output_dim)
        self.enc_dropout = nn.Dropout(self.dropout)

        # decoder block
        decoder_block = nn.ModuleList()
        if self.num_layers > 1:
            decoder_block.append(nn.Linear(self.output_dim, hidden_dim))
            decoder_block.append(self.activation)
            decoder_block.append(nn.Dropout(self.dropout))
        if self.num_layers > 2:
            for _ in range(self.num_layers-2):
                decoder_block.append(nn.Linear(self.hidden_dim, self.hidden_dim))
                decoder_block.append(self.activation)
                decoder_block.append(nn.Dropout(self.dropout))
        if self.num_layers == 1:
            decoder_block.append(nn.Linear(self.output_dim, self.input_dim*self.sequence_length))
        else:
            decoder_block.append(nn.Linear(self.hidden_dim, self.input_dim*self.sequence_length))
        decoder_block.append(self.activation)
        self.decoder = nn.Sequential(*decoder_block)

    def forward(self, data):
        return self.decode(self.encode(data))

    def encode(self, data):
        x = self.enc_lin_in(data)
        x = self.activation(x)
        x = self.enc_dropout(x)
        x = self.enc_lstm(x)[0][:, -1]
        x = self.enc_lin_out(x)
        x = self.activation(x)
        return x

    def decode(self, encoded):
        return self.decoder(encoded).reshape(-1, self.sequence_length, self.input_dim)

    def save(self, path):
        path = '../trained_ae'
        file = f'ae_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.pth'
        # torch.save(save, os.path.join(path, file))


def train_model(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    for batch in dataloader:
        optimizer.zero_grad()
        inputs = nn.BatchNorm1d(batch.shape[-1])(batch.float().permute(0, 2, 1)).permute(0, 2, 1)
        # inputs = filter(inputs.detach().cpu().numpy(), win_len=random.randint(29, 50), dtype=torch.Tensor)
        outputs = model(inputs.to(model.device))
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def test_model(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            inputs = nn.BatchNorm1d(batch.shape[-1])(batch.float().permute(0, 2, 1)).permute(0, 2, 1)
            outputs = model(inputs.to(model.device))
            loss = criterion(outputs, inputs)
            total_loss += loss.item()
    return total_loss / len(dataloader)


def train(num_epochs, model, train_dataloader, test_dataloader, optimizer, criterion, configuration: Optional[dict] = None):
    try:
        train_losses = []
        test_losses = []
        for epoch in range(num_epochs):
            train_loss = train_model(model, train_dataloader, optimizer, criterion)
            test_loss = test_model(model, test_dataloader, criterion)
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            print(f"Epoch {epoch + 1}/{num_epochs}: train_loss={train_loss:.4f}, test_loss={test_loss:.4f}")
        return train_losses, test_losses, model
    except KeyboardInterrupt:
        # save model at KeyboardInterrupt
        print("keyboard interrupt detected.")
        if configuration is not None:
            print("Configuration found.")
            configuration["model"]["state_dict"] = model.state_dict()  # update model's state dict
            save(configuration, configuration["general"]["default_save_path"])


def save(configuration, path):
    torch.save(configuration, path)
    print("Saved model and configuration to " + path)
