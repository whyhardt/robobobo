# plotter to visualize data from dataset, trained GAN or trained RL agent
import os

import matplotlib.pyplot as plt
import numpy as np

import torch

from utils import ae_dataloader
from nn_architecture import transformer_autoencoder


class plotter:
    def __init__(self):
        pass


if __name__ == '__main__':
    print('test')

    # -----------------
    # configuration
    # -----------------

    cfg = {
        'file': 'trained_gan/gan_100ep.pt',
        # 'file': 'stock_data/stocks_sp500_2010_2020.csv',
        'dataset': False,
        'checkpoint_gan': True,
        'checkpoint_rl': False,
        'generated_gan': False,
        'autoencoder': True,
        'plot_loss': True,          # only for checkpoint file
        'seq_len': 40,              # if dataset is loaded, seq_len is needed (-1: take sequence as a whole)
        'plotted_samples': 10,
        'inv_std': True,
        'inv_diff': True,
        'path_autoencoder': 'trained_ae/ae_kagglev1.pth',
    }

    scaler = None     # scaler for standardization

    # -----------------
    # load data
    # -----------------
    if cfg['dataset'] + cfg['checkpoint_gan'] + cfg['checkpoint_rl'] + cfg['generated_gan'] != 1:
        raise ValueError('Exactly one of dataset, checkpoint_gan, checkpoint_rl, generated_gan must be True')

    # check if loaded data is from dataset, trained GAN or trained RL agent
    if cfg['dataset']:
        if not cfg ['autoencoder']:
            cfg['inv_std'] = False   # already defined in create_dataloader
            cfg['inv_diff'] = False  # already defined in create_dataloader
        train_data, test_data, scaler = ae_dataloader.create_dataloader(cfg['file'], seq_len=cfg['seq_len'], batch_size=cfg['plotted_samples'], train_ratio=1.0, standardize=cfg['inv_std'], differentiate=cfg['inv_diff'])
        data = train_data.dataset.data

        # load autoencoder
        if cfg['autoencoder'] and cfg['path_autoencoder']:
            ae = torch.load(cfg['path_autoencoder'], map_location=torch.device('cpu'))

    elif cfg['checkpoint_gan']:
        checkpoint = torch.load(cfg['file'], map_location=torch.device('cpu'))
        if cfg['plot_loss']:
            data = checkpoint['loss']
        else:
            data = checkpoint['generated_samples']
        data = np.array(data)

    if cfg['autoencoder']:
        ae_dict = torch.load(cfg['path_autoencoder'], map_location=torch.device('cpu'))
        ae = transformer_autoencoder.TransformerAutoencoder(**ae_dict['model'])
        ae.load_state_dict(ae_dict['model']['state_dict'])

    # ------------
    # process data
    # ------------

    # draw samples from data linearly along first dimension
    if len(data.shape) == 3:
        data = np.array([np.array(data[i]) for i in range(0, len(data), len(data)//cfg['plotted_samples'])])
        if len(data) > cfg['plotted_samples']:
            data = data[:cfg['plotted_samples']]
    if len(data.shape) == 2:
        data = data.reshape((1, data.shape[0], data.shape[1]))

    # decode with ae
    data = ae.decode(torch.tensor(data).float())

    # invert standardization
    if cfg['inv_std']:
        if scaler is None:
            path_dataset = checkpoint['configuration']['path_dataset']
            try:
                _, _, scaler = ae_dataloader.create_dataloader(path_dataset, seq_len=checkpoint['configuration']['sequence_length'], batch_size=cfg['plotted_samples'], train_ratio=1.0, differentiate=False)
            except FileNotFoundError as e:
                print(e)
                print('Take path_dataset from checkpoint and search in directory stock_data')
                path_dataset = os.path.join('../stock_data', checkpoint['configuration']['path_dataset'].split('/')[-1])
                _, _, scaler = ae_dataloader.create_dataloader(path_dataset, seq_len=checkpoint['configuration']['sequence_length'], batch_size=cfg['plotted_samples'], train_ratio=1.0, differentiate=False)

        data = np.array([scaler.inverse_transform(data[i]) for i in range(len(data))])

    # invert differentiation
    if cfg['inv_diff']:
        data = np.cumsum(data, axis=1)

    # if more than one feature, plot only one randomly chosen
    if data.shape[-1] > 1:
        data = data[:, :, np.random.randint(data.shape[-1])].reshape(-1, data.shape[1], 1)

    # ------------
    # plot data
    # ------------

    # plot data
    fig, axs = plt.subplots(cfg['plotted_samples'], 1, squeeze=True, sharex=True)
    for i, curve in enumerate(data):
        # draw samples from data linearly along first dimension
        axs[i].plot(curve, label='sample {}'.format(i))
    plt.legend()
    plt.show()
