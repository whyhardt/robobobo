import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn

from nn_architecture import models, transformer_autoencoder
from utils.ae_dataloader import create_dataloader

if __name__=='__main__':

    cfg = {
        'file_data': '../stock_data/stocks_sp499_2010_2020_mvgavg50_downsampled10.csv',
        'file_ae': '../trained_ae/mvgavg50_int_ae.pt',
        'file_gan': '../trained_gan/mvgavg_gan_10k.pt',
        'use_ae': False,
        'compare': True,
        'n_samples': 10,
        'integrate': True,
        'loss': True,
    }

    # load ae
    if cfg['use_ae']:
        ae_dict = torch.load(cfg['file_ae'], map_location=torch.device('cpu'))
        ae = transformer_autoencoder.TransformerAutoencoder(**ae_dict["model"])
        ae.load_state_dict(ae_dict['model']['state_dict'])
        decoder = nn.Sequential(
            ae.decoder,
            ae.linear_dec,
            nn.Tanh(),
        )

    # load gan
    gan_dict = torch.load(cfg['file_gan'], map_location=torch.device('cpu'))
    # gan = models.TransformerGenerator(latent_dim=gan_dict['configuration']['latent_dim'] + gan_dict['configuration']['n_conditions'],
    #                                   channels=ae_dict['model']['output_dim'],
    #                                   seq_len=gan_dict['configuration']['sequence_length_generated'],
    #                                   decoder=decoder,)#ae.decoder if cfg['use_ae'] else None,)

    # load data for conditions
    train_dataloader, test_dataloader, scaler = create_dataloader(training_data=cfg['file_data'],
                                                                  seq_len=gan_dict['configuration']['sequence_length'],# - gan_dict['configuration']['sequence_length_generated'],
                                                                  batch_size=8,
                                                                  train_ratio=.8,
                                                                  standardize=True,
                                                                  differentiate=True,
                                                                  start_zero=False,)

    gan = models.TransformerGenerator2(
        latent_dim=gan_dict['configuration']['latent_dim'] + train_dataloader.dataset.data.shape[-1],
        channels=train_dataloader.dataset.data.shape[-1],
        seq_len=gan_dict['configuration']['sequence_length_generated'],
        hidden_dim=int(gan_dict['configuration']['hidden_dim']*4),
        num_layers=4,
        decoder=None)
    gan.load_state_dict(gan_dict['generator'])

    # generate samples
    # draw random samples from dataset as conditions for generator
    full_samples = next(iter(test_dataloader))
    # full_samples = next(iter(train_dataloader))

    if full_samples.shape[0] > cfg['n_samples']:
        full_samples = full_samples[np.random.randint(0, train_dataloader.batch_size - 1, size=cfg['n_samples'])]
    else:
        cfg['n_samples'] = full_samples.shape[0]

    # if compare, get conditions from full_samples
    if cfg['compare']:
        conditions_real = full_samples[:, :gan_dict['configuration']['sequence_length']-gan_dict['configuration']['sequence_length_generated'], :]
    else:
        conditions_real = full_samples

    # encode conditions
    if cfg['use_ae']:
        conditions = ae.encode(conditions_real)
        full_samples_encoded = ae.encode(full_samples)
    else:
        conditions = conditions_real

    # format conditions into vector
    # conditions = conditions.reshape(cfg['n_samples'], -1)
    # get latent space
    latent_space = torch.randn(cfg['n_samples'], gan_dict['configuration']['sequence_length']-gan_dict['configuration']['sequence_length_generated'], gan_dict['configuration']['latent_dim'])
    # concatenate latent space and conditions
    conditions = torch.cat((latent_space, conditions), dim=-1)
    # generate samples
    generated_samples = gan(conditions)#.squeeze(2).transpose(2, 1)
    # decode samples
    if cfg['use_ae']:
        full_samples_decoded = ae.decode(full_samples_encoded)
        if generated_samples.shape[2] != conditions_real.shape[2]:
            generated_samples = ae.decode(generated_samples)

    # concatenate conditions and generated samples
    final_sample = torch.cat((conditions_real, generated_samples), dim=1)

    # plot samples in subplots
    if cfg['integrate']:
        fig, axs = plt.subplots(cfg['n_samples'], 2, figsize=(10, 10), sharex=True)
        for i in range(cfg['n_samples']):
            axs[i, 0].plot(full_samples[i, :, 0].detach().numpy(), label='real')
            # axs[i, 0].plot(torch.cat((full_samples[i, :conditions_real.shape[1], 0], full_samples_decoded[i, conditions_real.shape[1]:, 0])).detach().numpy(), label='real_ae')
            axs[i, 0].plot(final_sample[i, :, 0].detach().numpy(), label='generated')
            axs[i, 1].plot(np.cumsum(full_samples[i, :, 0].detach().numpy()), label='real')
            # axs[i, 1].plot(np.cumsum(torch.cat((full_samples[i, :conditions_real.shape[1], 0],full_samples_decoded[i, conditions_real.shape[1]:, 0])).detach().numpy()),label='real_ae')
            axs[i, 1].plot(np.cumsum(final_sample[i, :, 0].detach().numpy()), label='generated')
        plt.legend()
        plt.show()
    else:
        # plot samples in subplots
        fig, axs = plt.subplots(cfg['n_samples'], 1, figsize=(10, 10), sharex=True)
        for i in range(cfg['n_samples']):
            axs[i].plot(full_samples[i, :, 0].detach().numpy(), label='real')
            # axs[i].plot(torch.cat((full_samples[i, :conditions_real.shape[1], 0], full_samples_decoded[i, conditions_real.shape[1]:, 0])).detach().numpy(), label='real_ae')
            axs[i].plot(final_sample[i, :, 0].detach().numpy(), label='generated')
        plt.legend()
        plt.show()

    if cfg['loss']:
        plt.plot(gan_dict['generator_loss'], label='generator')
        plt.plot(gan_dict['discriminator_loss'], label='discriminator')
        plt.plot(np.convolve(gan_dict['discriminator_loss'], np.ones(100)/100), label='mvg avg D')
        plt.plot(np.convolve(gan_dict['generator_loss'], np.ones(100) / 100), label='mvg avg G')
        # plt.ylim([0, 10])
        plt.legend()
        plt.show()

