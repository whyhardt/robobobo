import numpy as np
import torch
import matplotlib.pyplot as plt

from nn_architecture import models, transformer_autoencoder
from utils.ae_dataloader import create_dataloader

if __name__=='__main__':

    cfg = {
        'file_data': '../stock_data/stocks_sp500_2010_2020.csv',
        'file_ae': '../trained_ae/ae_kagglev1.pth',
        'file_gan': '../trained_gan/gantrans_15000ep.pt',
        'use_ae': True,
        'compare': True,
        'n_samples': 10,
    }

    # load ae
    if cfg['use_ae']:
        ae_dict = torch.load(cfg['file_ae'], map_location=torch.device('cpu'))
        ae = transformer_autoencoder.TransformerAutoencoder(**ae_dict["model"])
        ae.load_state_dict(ae_dict['model']['state_dict'])

    # load gan
    gan_dict = torch.load(cfg['file_gan'], map_location=torch.device('cpu'))
    gan = models.TransformerGenerator(latent_dim=gan_dict['configuration']['latent_dim'] + gan_dict['configuration']['n_conditions'],
                                      channels=ae_dict['model']['output_dim'],
                                      seq_len=gan_dict['configuration']['sequence_length_generated'],
                                      decoder=None,)#ae.decoder if cfg['use_ae'] else None,)
    gan.load_state_dict(gan_dict['generator'])

    # load data for conditions
    train_dataloader, test_dataloader, scaler = create_dataloader(training_data=cfg['file_data'],
                                                                  seq_len=gan_dict['configuration']['sequence_length'],# - gan_dict['configuration']['sequence_length_generated'],
                                                                  batch_size=128,
                                                                  train_ratio=1.0,
                                                                  standardize=True,
                                                                  differentiate=True,)

    # generate samples
    # draw random samples from dataset as conditions for generator
    full_samples = next(iter(train_dataloader))[np.random.randint(0, train_dataloader.batch_size - 1, size=cfg['n_samples'])]

    # if compare, get conditions from full_samples
    if cfg['compare']:
        conditions_real = full_samples[:, :gan_dict['configuration']['sequence_length']-gan_dict['configuration']['sequence_length_generated'], :]
    else:
        conditions_real = full_samples

    # encode conditions
    if cfg['use_ae']:
        conditions = ae.encode(conditions_real)
        full_samples_encoded = ae.encode(full_samples)

    # format conditions into vector
    conditions = conditions.reshape(cfg['n_samples'], -1)
    # get latent space
    latent_space = torch.randn(cfg['n_samples'], gan_dict['configuration']['latent_dim'])
    # concatenate latent space and conditions
    conditions = torch.cat((latent_space, conditions), dim=1)
    # generate samples
    generated_samples = gan(conditions).squeeze(2).transpose(2, 1)
    # decode samples
    if cfg['use_ae']:
        full_samples_decoded = ae.decode(full_samples_encoded)
        if generated_samples.shape[2] != conditions_real.shape[2]:
            generated_samples = ae.decode(generated_samples)

    # concatenate conditions and generated samples
    final_sample = torch.cat((conditions_real, generated_samples), dim=1)

    # plot samples in subplots
    fig, axs = plt.subplots(cfg['n_samples'], 1, figsize=(10, 10), sharex=True)
    for i in range(cfg['n_samples']):
        axs[i].plot(full_samples[i, :, 0].detach().numpy(), label='real')
        axs[i].plot(torch.cat((full_samples[i, :conditions_real.shape[1], 0], full_samples_decoded[i, conditions_real.shape[1]:, 0])).detach().numpy(), label='real_ae')
        axs[i].plot(final_sample[i, :, 0].detach().numpy(), label='generated')
        plt.legend()
    plt.show()

