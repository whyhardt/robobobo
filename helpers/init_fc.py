import os

import torch

from helpers.init_ae import init_ae
from nn_architecture.forecasting_networks import AEForecasting, LSTMForecasting


def init_fc(
    n_channels, 
    sequence_length, 
    device, 
    hidden_dim=256, 
    num_layers=3, 
    dropout=0.3,
    load_checkpoint=False, 
    path_checkpoint='',
    path_autoencoder='',
    **kwargs
) -> torch.nn.Module:
    
    # Initiate autoencoder
    model_dict = None
    if load_checkpoint and os.path.isfile(path_checkpoint):
        model_dict = torch.load(path_checkpoint)
        
    # init forecasting network
    if path_autoencoder == '':
        # init basic LSTMForecasting Unit
        model = LSTMForecasting(n_channels, hidden_dim, sequence_length, num_layers, dropout=0.3)
    else:
        if not os.path.isfile(path_autoencoder):
            raise ValueError(f"Autoencoder file '{path_autoencoder}' not found.")
        
        ae_dict = torch.load(path_autoencoder, map_location=torch.device('cpu'))
        # autoencoder = init_ae(n_channels, sequence_length, load_checkpoint=True, path_checkpoint=path_autoencoder, **ae_dict['configuration'])
        ae_dict['configuration']['device'] = device
        autoencoder = init_ae(n_channels, sequence_length-1, **ae_dict['configuration'])
        autoencoder.load_state_dict(ae_dict['model'])
        model = AEForecasting(autoencoder, hidden_dim, num_layers, dropout).to(device)
        
    return model