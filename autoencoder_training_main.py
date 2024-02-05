# train an autoencoder with attention mechanism for multivariate time series
import sys
import os
import time
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

from nn_architecture.ae_networks import TransformerAutoencoder, train, save, TransformerDoubleAutoencoder
from helpers import system_inputs
from helpers.trainer import AETrainer
from helpers.ddp_training import AEDDPTrainer, run
from helpers.get_master import find_free_port
from helpers.init_ae import init_ae


def main():

    # ------------------------------------------------------------------------------------------------------------------
    # Configure training parameters
    # ------------------------------------------------------------------------------------------------------------------

    default_args = system_inputs.parse_arguments(sys.argv, file='autoencoder_training_main.py')
    print('-----------------------------------------\n')

    # User inputs
    opt = {
        'path_dataset': default_args['path_dataset'],
        'path_checkpoint': default_args['path_checkpoint'],
        'load_checkpoint': default_args['load_checkpoint'],
        'save_name': default_args['save_name'],
        'target': default_args['target'],
        'sample_interval': default_args['sample_interval'],
        # 'conditions': default_args['conditions'],
        'channel_label': default_args['channel_label'],
        'channels_out': default_args['channels_out'],
        'timeseries_out': default_args['timeseries_out'],
        'n_epochs': default_args['n_epochs'],
        'batch_size': default_args['batch_size'],
        'train_ratio': default_args['train_ratio'],
        'learning_rate': default_args['learning_rate'],
        'hidden_dim': default_args['hidden_dim'],
        'num_heads': default_args['num_heads'],
        'num_layers': default_args['num_layers'],
        'activation': default_args['activation'],
        'ddp': default_args['ddp'],
        'ddp_backend': default_args['ddp_backend'],
        'lr_scheduler': default_args['lr_scheduler'],
        # 'n_conditions': len(default_args['conditions']) if default_args['conditions'][0] != '' else 0,
        'norm_data': True,
        'std_data': False,
        'diff_data': False,
        'kw_timestep': default_args['kw_timestep'],
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        'world_size': torch.cuda.device_count() if torch.cuda.is_available() else mp.cpu_count(),
        'history': None,
        'trained_epochs': 0,
        'sequence_length': 16,
        'n_channels': 0,
        'set_auto_zero': False,
    }

    # ----------------------------------------------------------------------------------------------------------------------
    # Load, process, and split data
    # ----------------------------------------------------------------------------------------------------------------------

    # Scale function -> Not necessary; already in dataloader -> param: norm_data=True
    def scale(dataset):
        x_min, x_max = dataset.min(axis=1), dataset.max(axis=1)
        x_min = x_min.reshape(-1, 1, dataset.shape[-1])
        x_max = x_max.reshape(-1, 1, dataset.shape[-1])
        x_max[np.where(x_max==x_min)] = 1e-9
        return (dataset-x_min)/(x_max-x_min)

    # Split data function
    def split_data(dataset, train_size=.8):
        num_samples = dataset.shape[0]
        shuffle_index = np.arange(num_samples)
        np.random.shuffle(shuffle_index)
        
        cutoff_index = int(num_samples*train_size)
        train = dataset[shuffle_index[:cutoff_index]]
        test = dataset[shuffle_index[cutoff_index:]]

        return test, train

    dataset = pd.read_csv(opt['path_dataset'], index_col=0, header=0).to_numpy()
    # create windows of length default_args['sequence_length'] and stack as 3D tensor
    dataset = np.stack([dataset[i:i+opt['sequence_length']] for i in range(dataset.shape[0]-opt['sequence_length'])])
    dataset = scale(dataset)

    # Determine n_channels, output_dim, and seq_length
    opt['n_channels'] = dataset.shape[-1]

    # Split dataset and convert to pytorch dataloader class
    test_dataset, train_dataset = split_data(dataset, opt['train_ratio'])
    test_dataloader = DataLoader(test_dataset, batch_size=opt['batch_size'], shuffle=True)
    train_dataloader = DataLoader(train_dataset, batch_size=opt['batch_size'], shuffle=True)

    # ------------------------------------------------------------------------------------------------------------------
    # Initiate and train autoencoder
    # ------------------------------------------------------------------------------------------------------------------

    model = init_ae(**opt)
    
    # # Populate model configuration    
    # history = {}
    # for key in opt.keys():
    #     if (not key == 'history') | (not key == 'trained_epochs'):
    #         history[key] = [opt[key]]
    # history['trained_epochs'] = []

    # if default_args['load_checkpoint']:
    #     model_dict = torch.load(opt['path_checkpoint'])
    # else:
    #     model_dict = None
        
    # if model_dict is not None:
    #     # update history
    #     for key in history.keys():
    #         history[key] = model_dict['configuration']['history'][key] + history[key]

    # opt['history'] = history

    if opt['ddp']:
        trainer = AEDDPTrainer(model, opt)
        if default_args['load_checkpoint']:
            trainer.load_checkpoint(default_args['path_checkpoint'])
        mp.spawn(run, args=(opt['world_size'], find_free_port(), opt['ddp_backend'], trainer, opt),
                 nprocs=opt['world_size'], join=True)
    else:
        trainer = AETrainer(model, opt)
        if default_args['load_checkpoint']:
            trainer.load_checkpoint(default_args['path_checkpoint'])
        samples = trainer.training(train_dataloader, test_dataloader)
        model = trainer.model
        print("Training finished.")

    # ----------------------------------------------------------------------------------------------------------------------
    # Save autoencoder
    # ----------------------------------------------------------------------------------------------------------------------

    # Save model
    # model_dict = dict(state_dict=model.state_dict(), config=model.config)
    if opt['save_name'] is None:
        fn = opt['path_dataset'].split('/')[-1].split('.csv')[0]
        opt['save_name'] = os.path.join("trained_ae", f"ae_{fn}_{str(time.time()).split('.')[0]}.pt")
    # save(model_dict, save_name)

    trainer.save_checkpoint(opt['save_name'], update_history=True, samples=samples)
    print(f"Model and configuration saved in {opt['save_name']}")
    
    # ----------------------------------------------------------------------------------------------------------------------
    # Plot results
    # ----------------------------------------------------------------------------------------------------------------------
    
    # create plot dataset
    plot_dataset = test_dataset[np.random.choice(test_dataset.shape[0], 1)]
    plot_dataset = torch.tensor(plot_dataset, dtype=torch.float32).to(opt['device'])
    
    # process with model
    model.eval()
    with torch.no_grad():
        plot_dataset_reconstructed = model(plot_dataset)
    # get index of all 0s in plot_dataset and set all values in plot_dataset_reconstructed to 0 at this index
    # plot_dataset_reconstructed[torch.where(plot_dataset == 0)] = 0 
    
    # plot
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(10, 1, figsize=(10, 10))
    for n, i in enumerate(np.random.choice(plot_dataset.shape[-1], 10)):
        axs[n].plot(plot_dataset[0, :, i].cpu().numpy(), label='original')
        axs[n].plot(plot_dataset_reconstructed[0, :, i].cpu().numpy(), label='reconstructed')
    axs[n].legend()
    plt.show()
        

if __name__ == "__main__":
    main()
