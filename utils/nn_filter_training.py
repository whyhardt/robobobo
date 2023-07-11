# train an autoencoder with attention mechanism for multivariate time series
import os.path

import numpy as np
from matplotlib import pyplot as plt
import torch.nn as nn
import torch

from utils.ae_dataloader import create_dataloader
from nn_architecture.autoencoder import LSTMAutoencoder, save, train

if __name__ == '__main__':
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # get parameters from saved model
    load_model = False
    training = True

    model_dict = None
    model_name = 'transformer_ae.pt'
    model_dir = '../trained_filter'

    data_dir = '../stock_data'
    data_file = 'stocks_sp500_2010_2020.csv'  # path to the csv file
    filtered_data_file = 'stocks_sp500_2010_2020_bandpass.csv'

    # configuration
    cfg = {
        "model": {
            "state_dict":   None,
            "input_dim":    None,
            "hidden_dim":   512,
            "output_dim":   10,
            "num_layers":   2,
            "dropout":      .3,
        },
        "training": {
            "lr":           1e-4,
            "epochs":       2,
        },
        "general": {
            "seq_len":      500,
            "scaler":       None,
            "training_data": os.path.join(data_dir, data_file),
            "batch_size":   32,
            "train_ratio":  .8,
            "standardize":  True,
            "differentiate": True,
            "default_save_path": os.path.join(model_dir, 'checkpoint_interrupted.pt'),
        }
    }

    # load model
    if load_model:
        cfg["model"] = torch.load(os.path.join(model_dir, model_name), map_location=torch.device('cpu'))["model"]
        print("adapted configuration from saved file " + os.path.join(model_dir, model_name))

    # load data from csv file as DataLoader
    cfg["general"]["training_data"] = os.path.join(data_dir, data_file)
    train_dataloader, test_dataloader, scaler = create_dataloader(**cfg["general"], shuffle=False)
    cfg["general"]["scaler"] = scaler

    # load filtered data
    cfg["general"]["training_data"] = os.path.join(data_dir, filtered_data_file)
    train_dataloader_filt, test_dataloader_filt, scaler_filt = create_dataloader(**cfg["general"], shuffle=False)

    # create the model
    if cfg["model"]["input_dim"] is None:
        cfg["model"]["input_dim"] = train_dataloader._dataset.data.shape[2]
    model = LSTMAutoencoder(**cfg["model"])

    # create the optimizer and criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["lr"])
    criterion = nn.MSELoss()

    if training:
        # train the model
        # train_losses, test_losses, model = train(num_epochs=cfg["training"]["epochs"], model=model, train_dataloader=(train_dataloader, train_dataloader_filt),
        #                                          test_dataloader=(test_dataloader, test_dataloader_filt), optimizer=optimizer, criterion=criterion, configuration=cfg)

        train_losses = []
        test_losses = []
        num_epochs = cfg["training"]["epochs"]
        for epoch in range(num_epochs):

            # get array with random indices
            indices = np.random.permutation(len(train_dataloader))

            # train model
            model.train()
            total_loss = 0
            for i in indices:
                optimizer.zero_grad()
                # inputs = batch.float()
                inputs = train_dataloader._dataset.data[indices[i] * train_dataloader.batch_size:indices[i] * train_dataloader.batch_size + train_dataloader.batch_size].float()
                outputs = model(inputs)
                loss = criterion(outputs, train_dataloader_filt._dataset.data[indices[i] * train_dataloader.batch_size:indices[i] * train_dataloader.batch_size + train_dataloader.batch_size].float())
                # loss.backward()
                # optimizer.step()
                total_loss += loss.item()
                train_loss = total_loss / len(train_dataloader)

            # test model
            indices = np.random.permutation(len(test_dataloader))
            model.eval()
            total_loss = 0
            with torch.no_grad():
                for i in indices:
                    inputs = test_dataloader._dataset.data[indices[i] * test_dataloader.batch_size:indices[i] * test_dataloader.batch_size + test_dataloader.batch_size].float()
                    outputs = model(inputs)
                    loss = criterion(outputs, test_dataloader_filt._dataset.data[indices[i] * test_dataloader.batch_size:indices[i] * test_dataloader.batch_size + test_dataloader.batch_size].float())
                    total_loss += loss.item()
                    test_loss = total_loss / len(test_dataloader)

            train_losses.append(train_loss)
            test_losses.append(test_loss)
            print(f"Epoch {epoch + 1}/{num_epochs}: train_loss={train_loss:.6f}, test_loss={test_loss:.6f}")

        # save model and training history under file with name model_CURRENTDATETIME.pth
        cfg["model"]["state_dict"] = model.state_dict()
        # get filename as ae_ + timestampe + pth
        save(cfg, os.path.join("..", model_dir, model_name))

        # plot the training and test losses
        plt.plot(train_losses)
        plt.plot(test_losses)
        plt.legend(['train', 'test'])
        plt.show()

    # encode a batch of sequences
    # batch = next(iter(test_dataloader))
    inputs = test_dataloader._dataset.data[0].float()
    outputs = model.encode(inputs)

    # decode a batch of sequences, rescale it with scaler and plot them
    outputs = model.decode(outputs)
    # outputs = scaler.inverse_transform(outputs.detach().cpu().numpy())
    # inputs = scaler.inverse_transform(inputs.detach().cpu().numpy())
    fig, axs = plt.subplots(10, 1, figsize=(10, 10), sharex=True)
    for i in range(10):
        stock = np.random.randint(0, inputs.shape[2])
        out = scaler.inverse_transform(outputs[i, :].detach().cpu().numpy())
        inp = scaler.inverse_transform(inputs[i, :].detach().cpu().numpy())
        axs[i].plot(inp[:, stock], label='Original')
        axs[i].plot(out[:, stock], label='Reconstructed')
    plt.legend()
    plt.show()
