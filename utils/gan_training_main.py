import os
import sys
import warnings
from datetime import datetime
import torch
import torch.multiprocessing as mp

from helpers.trainer import Trainer
from helpers.get_master import find_free_port
from helpers.ddp_training import run, DDPTrainer
from nn_architecture.models import TransformerGenerator, TransformerDiscriminator, TtsDiscriminator
from nn_architecture.transformer_autoencoder import TransformerAutoencoder
from utils.ae_dataloader import create_dataloader
from helpers import system_inputs

"""Implementation of the training process of a GAN for the generation of synthetic sequential data.

Instructions to start the training:
  - set the filename of the dataset to load
      - the shape of the dataset should be (n_samples, n_conditions + n_features)
      - the dataset should be a csv file
      - the first columns contain the conditions 
      - the remaining columns contain the time-series data
  - set the configuration parameters (Training configuration; Data configuration; GAN configuration)"""


if __name__ == '__main__':
    """Main function of the training process."""

    # TODO: check out train_test_split and shuffle in DataLoader
    #  --> Data = (train_ae, train_gan, train_rl, test_rl)
    #             (        ----------test_ae-------------)
    #             (                   ------test_gan-----)
    #             (                             -test_rl-)
    #  --> data efficient approach; conservative approach would be to separate also each test set as well
    #  train framework is divided in stages:
    #  1. train ae
    #  2. train gan; use trained ae --> knows already data from trained_ae
    #  3. train rl; use trained ae and gan --> knows already data from trained_ae and trained_gan
    #  Best way might be to shuffle data and split then by hand into train_ae, train_gan, train_rl, test_rl

    sys.argv = ["sequence_length=40",
                "seq_len_generated=10",
                "patch_size=20",
                "n_epochs=2"]#, "load_checkpoint"]
    default_args = system_inputs.parse_arguments(sys.argv, file='gan_training_main.py')

    # ----------------------------------------------------------------------------------------------------------------------
    # Configure training parameters and load data
    # ----------------------------------------------------------------------------------------------------------------------

    # Training configuration
    ddp = default_args['ddp']
    ddp_backend = default_args['ddp_backend']
    load_checkpoint = default_args['load_checkpoint']
    path_checkpoint = default_args['path_checkpoint']
    train_gan = default_args['train_gan']
    filter_generator = default_args['filter_generator']

    # trained_embedding = False       # Use an existing embedding
    # use_embedding = False           # Train the embedding in the optimization process

    # Data configuration
    windows_slices = default_args['windows_slices']
    diff_data = True               # Differentiate data
    std_data = True                # Standardize data
    norm_data = False                # Normalize data

    # raise warning if no normalization and standardization is used at the same time
    if std_data and norm_data:
        raise Warning("Standardization and normalization are used at the same time.")

    if (default_args['seq_len_generated'] == -1 or default_args['sequence_length'] == -1) and windows_slices:
        raise ValueError('If window slices are used, the keywords "sequence_length" and "seq_len_generated" must be greater than 0.')

    if load_checkpoint:
        print(f'Resuming training from checkpoint {path_checkpoint}.')

    # Look for cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if not ddp else torch.device("cpu")
    world_size = torch.cuda.device_count() if torch.cuda.is_available() else mp.cpu_count()

    # GAN configuration
    opt = {
        'n_epochs': default_args['n_epochs'],
        'sequence_length': default_args['sequence_length'],
        'seq_len_generated': default_args['seq_len_generated'],
        'load_checkpoint': default_args['load_checkpoint'],
        'path_checkpoint': default_args['path_checkpoint'],
        'path_dataset': default_args['path_dataset'],
        'batch_size': default_args['batch_size'],
        'learning_rate': default_args['learning_rate'],
        'sample_interval': default_args['sample_interval'],
        'n_conditions': len(default_args['conditions']) if default_args['conditions'][0] != "None" else 0,
        'patch_size': default_args['patch_size'],
        'kw_timestep': default_args['kw_timestep_dataset'],
        'conditions': default_args['conditions'],
        'lambda_gp': 10,
        'hidden_dim': 128,          # Dimension of hidden layers in discriminator and generator
        'latent_dim': 16,           # Dimension of the latent space
        'critic_iterations': 5,     # number of iterations of the critic per generator iteration for Wasserstein GAN
        'n_lstm': 2,                # number of lstm layers for lstm GAN
        'world_size': world_size,   # number of processes for distributed training
        'path_autoencoder': default_args['path_autoencoder'], # use autoencoder before generator
    }

    # Load dataset as tensor
    train_dataloader, test_dataloader, scaler = create_dataloader(training_data=opt['path_dataset'],
                                                                  seq_len=opt['sequence_length'],
                                                                  batch_size=opt['batch_size'],
                                                                  train_ratio=0.8,
                                                                  standardize=std_data,
                                                                  diff_data=diff_data,)
    opt['n_samples'] = train_dataloader._dataset.data.shape[0]
    opt['sequence_length'] = train_dataloader._dataset.data.shape[1]
    opt['n_features'] = train_dataloader._dataset.data.shape[2]
    opt['n_features_generator'] = opt['n_features']

    if opt['sequence_length'] % opt['patch_size'] != 0:
        warnings.warn(f"Sequence length ({opt['sequence_length']}) must be a multiple of patch size ({default_args['patch_size']}).\n"
                      f"The sequence length is padded with zeros to fit the condition.")
        padding = 0
        while (opt['sequence_length'] + padding) % default_args['patch_size'] != 0:
            padding += 1
        padding = torch.zeros((train_dataloader._dataset.data.shape[0], padding, train_dataloader._dataset.data.shape[2]))
        train_dataloader._dataset.data = torch.cat((train_dataloader._dataset.data, padding), dim=1)
        opt['sequence_length'] = train_dataloader._dataset.data.shape[1]
    if opt['seq_len_generated'] == -1:
        opt['seq_len_generated'] = opt['sequence_length']

    # Initialize trained autoencoder and set n_conditions accordingly to encoder output dimension
    autoencoder = None
    if default_args['path_autoencoder'] != 'None':
        ae_dict = torch.load(default_args['path_autoencoder'])
        autoencoder = TransformerAutoencoder(**ae_dict["model"])
        autoencoder.load_state_dict(ae_dict["model"]["state_dict"])
        # calculate number of conditions for generator from autoencoders output dimension and input sequence length
        opt['n_conditions'] += ae_dict["model"]["output_dim"]*(opt['sequence_length'] - opt['seq_len_generated'])
        opt['n_features_generator'] = ae_dict["model"]["output_dim"]
    else:
        opt['n_conditions'] += opt['sequence_length'] - opt['seq_len_generated']

    # Initialize generator, discriminator and trainer
    generator = TransformerGenerator(latent_dim=opt['latent_dim'] + opt['n_conditions'],
                                     channels=opt['n_features_generator'],
                                     seq_len=opt['seq_len_generated'],
                                     decoder=autoencoder.linear_dec if autoencoder is not None else None)
    discriminator = TtsDiscriminator(seq_length=opt['sequence_length'],
                                     patch_size=opt['patch_size'],
                                     in_channels=opt['n_features'],
                                     n_classes=opt['n_features'],
                                     emb_size=250)
    # discriminator = TransformerDiscriminator(channels_in=opt['n_features'],
    #                                          channels_out=opt['n_features'],
    #                                          seq_len=opt['sequence_length'])

    print("Generator and discriminator initialized.")

    # ----------------------------------------------------------------------------------------------------------------------
    # Start training process
    # ----------------------------------------------------------------------------------------------------------------------

    if train_gan:
        # GAN-Training
        print('\n-----------------------------------------')
        print("Training GAN...")
        print('-----------------------------------------\n')
        if ddp:
            trainer = DDPTrainer(generator, discriminator, opt, autoencoder)
            if default_args['load_checkpoint']:
                trainer.load_checkpoint(default_args['path_checkpoint'])
            mp.spawn(run, args=(world_size, find_free_port(), ddp_backend, trainer, opt),
                     nprocs=world_size, join=True)
        else:
            trainer = Trainer(generator, discriminator, opt, autoencoder)
            if default_args['load_checkpoint']:
                trainer.load_checkpoint(default_args['path_checkpoint'])
            gen_samples = trainer.training(train_dataloader)

            # save final models, optimizer states, generated samples, losses and configuration as final result
            path = '../trained_gan'
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'gan_{trainer.epochs}ep_' + timestamp + '.pt'
            trainer.save_checkpoint(path_checkpoint=os.path.join(path, filename), generated_samples=gen_samples)

        print("GAN training finished.")
        print("Generated samples saved to file.")
        print("Model states saved to file.")
    else:
        print("GAN not trained.")
    