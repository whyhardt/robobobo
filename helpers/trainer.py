import os

import torch
import numpy as np

from nn_architecture import losses, models
from nn_architecture.losses import WassersteinGradientPenaltyLoss as Loss

# https://machinelearningmastery.com/how-to-implement-wasserstein-loss-for-generative-adversarial-networks/
# For implementation of Wasserstein-GAN see link above


class Trainer:
    """Trainer for conditional Wasserstein-GAN with gradient penalty.
    Source: https://arxiv.org/pdf/1704.00028.pdf"""

    def __init__(self, generator, discriminator, opt, autoencoder=None):
        # training configuration
        self.device = opt['device'] if 'device' in opt else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.sequence_length = opt['sequence_length'] if 'sequence_length' in opt else None
        self.sequence_length_generated = opt['seq_len_generated'] if 'seq_len_generated' in opt else None
        self.batch_size = opt['batch_size'] if 'batch_size' in opt else 32
        self.epochs = opt['n_epochs'] if 'n_epochs' in opt else 10
        self.use_checkpoint = opt['load_checkpoint'] if 'load_checkpoint' in opt else False
        self.path_checkpoint = opt['path_checkpoint'] if 'path_checkpoint' in opt else None
        self.latent_dim = opt['latent_dim'] if 'latent_dim' in opt else 10
        self.critic_iterations = opt['critic_iterations'] if 'critic_iterations' in opt else 5
        self.lambda_gp = opt['lambda_gp'] if 'lambda_gp' in opt else 10
        self.sample_interval = opt['sample_interval'] if 'sample_interval' in opt else 100
        self.learning_rate = opt['learning_rate'] if 'learning_rate' in opt else 0.0001
        self.n_conditions = opt['n_conditions'] if 'n_conditions' in opt else 0
        self.n_features = opt['n_features'] if 'n_features' in opt else 1
        self.b1 = 0  # .5
        self.b2 = 0.9  # .999
        self.rank = 0  # Device: cuda:0, cuda:1, ... --> Device: cuda:rank

        self.autoencoder = autoencoder

        self.generator = generator
        self.discriminator = discriminator

        self.generator.to(self.device)
        self.discriminator.to(self.device)

        self.generator_optimizer = torch.optim.Adam(self.generator.parameters(),
                                                    lr=self.learning_rate, betas=(self.b1, self.b2))
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(),
                                                        lr=self.learning_rate, betas=(self.b1, self.b2))

        self.loss = Loss()
        if isinstance(self.loss, losses.WassersteinGradientPenaltyLoss):
            self.loss.set_lambda_gp(self.lambda_gp)

        self.prev_g_loss = 0
        self.configuration = {
            'device': self.device,
            'generator': str(self.generator.__class__.__name__),
            'discriminator': str(self.discriminator.__class__.__name__),
            'sequence_length': self.sequence_length,
            'sequence_length_generated': self.sequence_length_generated,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'sample_interval': self.sample_interval,
            'learning_rate': self.learning_rate,
            'n_conditions': self.n_conditions,
            'latent_dim': self.latent_dim,
            'critic_iterations': self.critic_iterations,
            'lambda_gp': self.lambda_gp,
            'patch_size': opt['patch_size'] if 'patch_size' in opt else None,
            'b1': self.b1,
            'b2': self.b2,
            'path_dataset': opt['path_dataset'] if 'path_dataset' in opt else None,
        }

        self.d_losses = []
        self.g_losses = []

        # # load checkpoint
        # try:
        #     if self.use_checkpoint:
        #         self.load_checkpoint(self.path_checkpoint)
        #         self.use_checkpoint = False
        # except RuntimeError:
        #     Warning("Could not load checkpoint. If DDP was used while saving and is used now for loading the checkpoint will be loaded in a following step.")

    def training(self, dataloader):
        """Batch training of the conditional Wasserstein-GAN with GP."""

        self.generator.train()
        self.discriminator.train()
        if self.autoencoder is not None:
            self.autoencoder.eval()

        gen_samples = []
        # num_batches = int(np.ceil(dataset.shape[0] / self.batch_size))

        # checkpoint file settings; toggle between two checkpoints to avoid corrupted file if training is interrupted
        path_checkpoint = os.path.dirname(self.path_checkpoint)
        trigger_checkpoint_01 = True
        checkpoint_01_file = 'checkpoint_01.pt'
        checkpoint_02_file = 'checkpoint_02.pt'
        train_generator_iter = 0
        for epoch in range(self.epochs):
            for batch in dataloader:
                batch_size = batch.shape[0]

                # draw batch_size samples from sessions
                # data = dataset[i:i + batch_size, self.n_conditions:].to(self.device)
                # data_labels = dataset[i:i + batch_size, :self.n_conditions].to(self.device)

                # update generator every n iterations as suggested in paper
                if train_generator_iter % self.critic_iterations == 0:
                    train_generator = True
                else:
                    train_generator = False
                train_generator_iter += 1

                d_loss, g_loss, gen_imgs = self.batch_train(batch, None, train_generator)

                self.d_losses.append(d_loss)
                self.g_losses.append(g_loss)

            # Save a checkpoint of the trained GAN and the generated samples every sample interval
            if epoch % self.sample_interval == 0:
                gen_samples.append(gen_imgs[np.random.randint(0, batch_size)].detach().cpu().numpy())
                # save models and optimizer states as checkpoints
                # toggle between checkpoint files to avoid corrupted file during training
                if trigger_checkpoint_01:
                    self.save_checkpoint(os.path.join(path_checkpoint, checkpoint_01_file), generated_samples=gen_samples)
                    trigger_checkpoint_01 = False
                else:
                    self.save_checkpoint(os.path.join(path_checkpoint, checkpoint_02_file), generated_samples=gen_samples)
                    trigger_checkpoint_01 = True

            self.print_log(epoch + 1, d_loss, g_loss)

        self.manage_checkpoints(path_checkpoint, [checkpoint_01_file, checkpoint_02_file])

        return gen_samples

    def batch_train(self, data, data_labels, train_generator):
        """Trains the GAN-Model on one batch of data.
        No further batch-processing. Give batch as to-be-used."""
        batch_size = data.shape[0]

        seq_length = data.shape[1] if isinstance(self.generator, models.CondLstmGenerator) else 1

        # get conditional data for generator only --> only if generator has to forecast
        gen_cond_data = data[:, :self.sequence_length-self.sequence_length_generated].to(self.device)
        if self.autoencoder is not None:
            gen_cond_data_encoded = self.autoencoder.encode(gen_cond_data)
            gen_cond_data_encoded = gen_cond_data_encoded.view(-1, gen_cond_data_encoded.shape[1]*gen_cond_data_encoded.shape[2])
        else:
            gen_cond_data_encoded = gen_cond_data.view(-1, gen_cond_data.shape[1]*gen_cond_data.shape[2])

        # get conditional data for discriminator and generator
        if data_labels is not None:
            gen_labels = torch.cat((data_labels, gen_cond_data_encoded), dim=1).to(self.device)
            fake_labels = data_labels.view(-1, data_labels.shape[1], 1, 1).repeat(1, 1, 1, self.sequence_length).to(self.device)
            real_labels = data_labels.view(-1, data_labels.shape[1], 1, 1).repeat(1, 1, 1, self.sequence_length).to(self.device)
        else:
            gen_labels = gen_cond_data_encoded
            fake_labels = None
            real_labels = None

        # ------------------------------------------
        #  Generate Synthetic Data
        # ------------------------------------------

        # Sample noise and labels as generator input
        z = self.sample_latent_variable(batch_size=batch_size, latent_dim=self.latent_dim,
                                        device=self.device, sequence_length=seq_length)

        # Generate a batch of samples
        z = torch.cat((z, gen_labels), dim=1)
        gen_imgs = self.generator(z)

        if gen_imgs.shape[1] != self.n_features and self.autoencoder is not None:
            # if generator produces encoded output, then decode with autoencoder
            gen_imgs = self.autoencoder.decode(gen_imgs.squeeze(2).transpose(2, 1)).view(-1, self.n_features, 1, self.sequence_length_generated)

        fake_data = torch.cat((gen_cond_data.view(batch_size, self.n_features, 1, -1), gen_imgs), dim=-1).to(
            self.device)
        if fake_labels is not None:
            fake_data = torch.cat((fake_data, fake_labels), dim=1).to(self.device)

        # ---------------------
        #  Train Discriminator
        # ---------------------

        self.discriminator_optimizer.zero_grad()
        # if fake_data.dim() > 3 and fake_data.shape[2] == 1:
        #     fake_data = fake_data.squeeze(2)
        # if fake_data.shape[1] == self.n_features and fake_data.shape[2] == self.sequence_length:
        #     fake_data = fake_data.transpose(2, 1)
        validity_fake = self.discriminator(fake_data)

        # Loss for real images
        real_data = data.view(-1, self.n_features, 1, data.shape[1]).to(self.device)
        if real_labels is not None:
            real_data = torch.cat((real_data, real_labels), dim=1).to(self.device)

        validity_real = self.discriminator(real_data)

        # Total discriminator loss and update
        if isinstance(self.loss, losses.WassersteinGradientPenaltyLoss):
            # discriminator, real_images, fake_images, real_labels, fake_labels
            d_loss = self.loss.discriminator(validity_real, validity_fake, self.discriminator, real_data, fake_data)
        else:
            d_loss = self.loss.discriminator(validity_real, validity_fake)
        d_loss.backward(retain_graph=True)
        self.discriminator_optimizer.step()

        if train_generator:
            # -----------------
            #  Train Generator
            # -----------------

            self.generator_optimizer.zero_grad()
            validity = self.discriminator(fake_data)
            g_loss = self.loss.generator(validity)
            g_loss.backward()
            self.generator_optimizer.step()

            g_loss = g_loss.item()
            self.prev_g_loss = g_loss
        else:
            g_loss = self.prev_g_loss

        return d_loss.item(), g_loss, fake_data.squeeze(2).transpose(2, 1)

    def save_checkpoint(self, path_checkpoint=None, generated_samples=None, generator=None, discriminator=None):
        if path_checkpoint is None:
            path_checkpoint = 'trained_models'+os.path.sep+'checkpoint.pt'
        if generator is None:
            generator = self.generator
        if discriminator is None:
            discriminator = self.discriminator

        torch.save({
            'generator': generator.state_dict(),
            'discriminator': discriminator.state_dict(),
            'generator_optimizer': self.generator_optimizer.state_dict(),
            'discriminator_optimizer': self.discriminator_optimizer.state_dict(),
            'discriminator_loss': self.d_losses,
            'generator_loss': self.g_losses,
            'generated_samples': generated_samples,
            'configuration': self.configuration,
        }, path_checkpoint)

    def load_checkpoint(self, path_checkpoint):
        if os.path.isfile(path_checkpoint):
            # load state_dicts
            state_dict = torch.load(path_checkpoint, map_location=self.device)
            self.generator.load_state_dict(state_dict['generator'])
            self.discriminator.load_state_dict(state_dict['discriminator'])
            self.generator_optimizer.load_state_dict(state_dict['generator_optimizer'])
            self.discriminator_optimizer.load_state_dict(state_dict['discriminator_optimizer'])
            print(f"Device {self.device}:{self.rank}: Using pretrained GAN.")
        else:
            Warning("No checkpoint-file found. Using random initialization.")

    def manage_checkpoints(self, path_checkpoint: str, checkpoint_files: list, generator=None, discriminator=None):
        """if training was successful delete the sub-checkpoint files and save the most current state as checkpoint,
        but without generated samples to keep memory usage low. Checkpoint should be used for further training only.
        Therefore, there's no need for the saved samples."""

        print("Managing checkpoints...")
        # save current model as checkpoint.pt
        self.save_checkpoint(path_checkpoint=os.path.join(path_checkpoint, 'checkpoint.pt'), generator=generator, discriminator=discriminator)

        for f in checkpoint_files:
            if os.path.exists(os.path.join(path_checkpoint, f)):
                os.remove(os.path.join(path_checkpoint, f))

    def print_log(self, current_epoch, d_loss, g_loss):
        print(
            "[Epoch %d/%d] [D loss: %f] [G loss: %f]"
            % (current_epoch, self.epochs,
               d_loss, g_loss)
        )

    def set_optimizer_state(self, optimizer, g_or_d='G'):
        if g_or_d == 'G':
            self.generator_optimizer.load_state_dict(optimizer)
            print('Generator optimizer state loaded successfully.')
        elif g_or_d == 'D':
            self.discriminator_optimizer.load_state_dict(optimizer)
            print('Discriminator optimizer state loaded successfully.')
        else:
            raise ValueError('G_or_D must be either "G" (Generator) or "D" (Discriminator)')

    @staticmethod
    def sample_latent_variable(sequence_length=1, batch_size=1, latent_dim=1, device=torch.device('cpu')):
        """samples a latent variable from a normal distribution
        as a tensor of shape (batch_size, (sequence_length), latent_dim) on the given device"""
        if sequence_length > 1:
            # sample a sequence of latent variables
            # only used for RNN/LSTM generator
            return torch.randn((batch_size, sequence_length, latent_dim), device=device).float()
        else:
            return torch.randn((batch_size, latent_dim), device=device).float()
