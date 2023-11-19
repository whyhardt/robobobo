
import os
import torch
from nn_architecture.ae_networks import Autoencoder, TransformerAutoencoder, TransformerDoubleAutoencoder


def init_ae(
    n_channels, 
    sequence_length, 
    device, 
    target='full', 
    channels_out=10, 
    timeseries_out=10, 
    hidden_dim=256, 
    num_layers=3, 
    num_heads=4, 
    load_checkpoint=False, 
    path_checkpoint='',
    **kwargs) -> Autoencoder:
    
    # ------------------------------------------------------------------------------------------------------------------
    # Initiate and train autoencoder
    # ------------------------------------------------------------------------------------------------------------------

    # Initiate autoencoder
    model_dict = None
    if load_checkpoint and os.path.isfile(path_checkpoint):
        model_dict = torch.load(path_checkpoint)
        # model_state = model_dict['state_dict

        target_old = target
        channels_out_old = channels_out
        timeseries_out_old = timeseries_out

        target = model_dict['configuration']['target']
        channels_out = model_dict['configuration']['channels_out']
        timeseries_out = model_dict['configuration']['timeseries_out']
        
        # Report changes to user
        print(f"Loading model {path_checkpoint}.\n\nInhereting the following parameters:")
        print("parameter:\t\told value -> new value")
        print(f"target:\t\t\t{target_old} -> {target}")
        print(f"channels_out:\t{channels_out_old} -> {channels_out}")
        print(f"timeseries_out:\t{timeseries_out_old} -> {timeseries_out}")
        print('-----------------------------------\n')
        # print(f"Target: {target}")
        # if (target == 'channels') | (target == 'full'):
        #     print(f"channels_out: {channels_out}")
        # if (target == 'timeseries') | (target == 'full'):
        #     print(f"timeseries_out: {timeseries_out}")
        #     print('-----------------------------------\n')

    elif load_checkpoint and not os.path.isfile(path_checkpoint):
        raise FileNotFoundError(f"Checkpoint file {path_checkpoint} not found.")
    
    # Add parameters for tracking
    input_dim = n_channels if target in ['channels', 'full'] else sequence_length
    output_dim = channels_out if target in ['channels', 'full'] else n_channels
    output_dim_2 = sequence_length if target in ['channels'] else timeseries_out
    
    if target == 'channels':
        model = TransformerAutoencoder(input_dim=n_channels,
                                       output_dim=channels_out,
                                       output_dim_2=sequence_length,
                                       target=TransformerAutoencoder.TARGET_CHANNELS,
                                       hidden_dim=hidden_dim,
                                       num_layers=num_layers,
                                       num_heads=num_heads,).to(device)
    elif target == 'time':
        model = TransformerAutoencoder(input_dim=sequence_length,
                                       output_dim=timeseries_out,
                                       output_dim_2=n_channels,
                                       target=TransformerAutoencoder.TARGET_TIMESERIES,
                                       hidden_dim=hidden_dim,
                                       num_layers=num_layers,
                                       num_heads=num_heads,).to(device)
    elif target == 'full':
        model = TransformerDoubleAutoencoder(input_dim=n_channels,
                                             output_dim=output_dim,
                                             output_dim_2=output_dim_2,
                                             sequence_length=sequence_length,
                                             hidden_dim=hidden_dim,
                                             num_layers=num_layers,
                                             num_heads=num_heads,).to(device)
    else:
        raise ValueError(f"Encode target '{target}' not recognized, options are 'channels', 'time', or 'full'.")
    
    return model