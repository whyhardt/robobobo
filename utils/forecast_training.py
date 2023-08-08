# train a forecasting model for stock data
# use transformer generator as forecasting model
# use stock data as input
# load data with ae_dataloader

# %%
# import packages
import os
import numpy as np
import torch
from matplotlib import pyplot as plt

from nn_architecture.gan_networks import TransformerGenerator
from nn_architecture.ae_networks import TransformerAutoencoder
from utils.ae_dataloader import create_dataloader
from utils.get_filter import moving_average


if __name__ == '__main__':
    # %%
    # configuration
    cfg = {
        "checkpoints": {
            'load_checkpoint': False,
            'file_encoder': os.path.join('..', 'trained_ae', 'ae_deep.pt'),
            'file_model': os.path.join('..', 'trained_fc', 'bp_fc_1000ep.pt'),
        },
        "general": {
            "seq_len":      40,     # for dataloader; how many time steps per total sequence
            "scaler":       None,
            "training_data": os.path.join('..', 'stock_data', 'stocks_sp500_2010_2020_bandpass_downsampled10.csv'),
            "batch_size":   128,
            "train_ratio":  .8,
            "standardize":  True,
            "differentiate": True,
            "default_save_path": os.path.join('..', 'trained_fc', 'transformer_ae.pt'),
        },
        "model": {
            'state_dict': None,
            'state_dict_optimizer': None,
            'num_layers': 2,
            'hidden_dim': 2048,
            'dropout': 0.1,
            'seq_len': 10,       # how many time steps to forecast (split from total sequence)
            'latent_dim': 10,   # features * time steps to consider
            'channels': 1,      # features given by the autoencoder
        },
        "training": {
            'num_epochs': 1,
            'learning_rate': 0.0001,
        }
    }

    if cfg['checkpoints']['load_checkpoint']:
        print("Restoring configuration...")
        cfg['model'] = torch.load(cfg['checkpoints']['file_model'], map_location=torch.device('cpu'))['model']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load data
    print('Loading data...')
    train_dl, test_dl, scaler = create_dataloader(**cfg["general"])

    # load autoencoder
    print('Loading autoencoder...')
    encoder_dict = torch.load(cfg['checkpoints']['file_encoder'], map_location=torch.device('cpu'))
    encoder = TransformerAutoencoder(**encoder_dict["model"]).to(device)
    encoder.load_state_dict(encoder_dict['model']['state_dict'])
    encoder.eval()

    # set channels and seq_len of model
    cfg['model']['channels'] = encoder_dict['model']['output_dim']
    cfg['model']['latent_dim'] = (cfg['general']['seq_len']-cfg['model']['seq_len'])*cfg['model']['channels']

    # create model
    print('Initializing forecast model...')
    model = TransformerGenerator(**cfg["model"], decoder=encoder.decoder).to(device)
    loss = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['training']['learning_rate'])
    # load model
    if cfg['checkpoints']['load_checkpoint']:
        print("Loading model checkpoint...")
        model.load_state_dict(cfg['model']['state_dict'])
        optimizer.load_state_dict(cfg['model']['state_dict_optimizer'])

    # %%
    # train model
    losses = np.zeros((cfg['training']['num_epochs'], 2))
    print('Training model...')
    for epoch in range(cfg['training']['num_epochs']):
        for batch in train_dl:
            model.train()
            y = batch[:, -cfg['model']['seq_len']:, :]
            x = batch[:, :-cfg['model']['seq_len'], :]

            # encode x
            x = encoder.encode(x)

            # forward pass
            y_hat = model(x.reshape(-1, x.shape[1]*x.shape[2])).squeeze(2).permute(0, 2, 1)
            loss_train = loss(y_hat, y)

            # backward pass
            loss_train.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Test the model
            test_data = next(iter(test_dl))
            y = test_data[:, -cfg['model']['seq_len']:, :]
            x = test_data[:, :-cfg['model']['seq_len'], :]
            x = encoder.encode(x)
            model.eval()
            with torch.no_grad():
                y_hat_test = model(x.reshape(-1, x.shape[1]*x.shape[2])).squeeze(2).permute(0, 2, 1)
                loss_test = loss(y_hat_test, y)

        # print loss
        print(f"Epoch: {epoch}, Loss: {loss_train.item()}, Test Loss: {loss_test.item()}")
        losses[epoch, :] = [loss_train.item(), loss_test.item()]
    print("Finished training!")

    # %%
    # save model
    cfg['model']['state_dict'] = model.state_dict()
    cfg['model']['state_dict_optimizer'] = optimizer.state_dict()
    torch.save(cfg, cfg['general']['default_save_path'])
    print(f"Model saved to {cfg['general']['default_save_path']}!")

    # %%
    # plot losses
    plt.plot(losses[:, 0], label='train')
    plt.plot(losses[:, 1], label='test')
    plt.legend()
    plt.show()

    # %%
    # plot predictions for test data
    print('Plotting predictions...')
    model.eval()
    with torch.no_grad():
        test_data = next(iter(test_dl))
        y = test_data[:, -cfg['model']['seq_len']:, :]
        x = test_data[:, :-cfg['model']['seq_len'], :]
        x_enc = encoder.encode(x)
        y_hat_test = model(x_enc.reshape(-1, x_enc.shape[1]*x_enc.shape[2])).squeeze(2).permute(0, 2, 1)
    num_plots = 5
    fig, axs = plt.subplots(num_plots, 1, figsize=(10, 10), sharex=True)
    for i in range(num_plots):
        random_idx = np.random.randint(0, y.shape[2])
        axs[i].plot(torch.concat((x[i, :, random_idx], y[i, :, random_idx]), dim=0))
        axs[i].plot(torch.concat((x[i, :, random_idx], y_hat_test[i, :, random_idx]), dim=0))
    plt.legend(['true', 'pred'])
    plt.show()
