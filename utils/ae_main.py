# train an autoencoder with attention mechanism for multivariate time series
import os.path

import numpy as np
from matplotlib import pyplot as plt
import torch.nn as nn
import torch

from utils.ae_dataloader import create_dataloader
from utils.get_filter import moving_average as filter
from nn_architecture.ae_networks import LSTMDoubleAutoencoder, TransformerAutoencoder, save, train, \
    LSTMTransformerAutoencoder, TransformerDoubleAutoencoder

if __name__ == '__main__':

    # get parameters from saved model
    load_model = False
    training = True

    model_dict = None
    model_name = 'transformer_ae.pt'
    model_dir = '../trained_ae'

    data_dir = '../stock_data'
    data_file = 'portfolio_custom140_2008_2022.csv'  # path to the csv file

    # configuration
    cfg = {
        "model": {
            "state_dict":   None,
            "input_dim":    None,
            "hidden_dim":   50,
            "output_dim":   50,
            "output_dim_2": 10,
            "num_layers":   3,
            "dropout":      .1,
            "activation":   nn.Tanh(),
        },
        "training": {
            "lr":           1e-4,
            "epochs":       10,
        },
        "general": {
            "seq_len":          20,
            "scaler":           None,
            "training_data":    os.path.join(data_dir, data_file),
            "batch_size":       32,
            "train_ratio":      .8,
            "standardize":      False,
            "differentiate":    False,
            "normalize":        True,
            "start_zero":       True,
            "default_save_path": os.path.join('../trained_ae', 'transformer_ae.pt'),
        }
    }

    # load model
    if load_model:
        cfg["model"] = torch.load(os.path.join(model_dir, model_name), map_location=torch.device('cpu'))["model"]
        print("adapted configuration from saved file " + os.path.join(model_dir, model_name))

    # load data from csv file as DataLoader
    cfg["general"]["training_data"] = os.path.join(data_dir, data_file)
    train_dataloader, test_dataloader, scaler = create_dataloader(**cfg["general"])
    cfg["general"]["scaler"] = scaler

    # create the model
    if cfg["model"]["input_dim"] is None:
        cfg["model"]["input_dim"] = train_dataloader.dataset.data.shape[-1]
    # model = LSTMDoubleAutoencoder(**cfg["model"], sequence_length=cfg["general"]["seq_len"])
    # model = TransformerDoubleAutoencoder(**cfg["model"], sequence_length=cfg["general"]["seq_len"])
    model = TransformerAutoencoder(**cfg["model"])
    if cfg["model"]["state_dict"] is not None:
        model.load_state_dict(cfg["model"]["state_dict"])
        print("Loaded model state dict!")

    # create the optimizer and criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["lr"])
    criterion = nn.MSELoss()

    if training:
        # train the model
        train_losses, test_losses, model = train(num_epochs=cfg["training"]["epochs"], model=model, train_dataloader=train_dataloader,
                                                 test_dataloader=test_dataloader, optimizer=optimizer, criterion=criterion, configuration=cfg)

        # save model and training history under file with name model_CURRENTDATETIME.pth
        cfg["model"]["state_dict"] = model.state_dict()
        # get filename as ae_ + timestampe + pth
        save(cfg, os.path.join("../trained_ae", model_name))

        # plot the training and test losses
        plt.plot(train_losses)
        plt.plot(test_losses)
        plt.legend(['train', 'test'])
        plt.show()

    # encode a batch of sequences
    batch = next(iter(test_dataloader))
    # inputs = nn.BatchNorm1d(batch.shape[-1])(batch.float().permute(0, 2, 1)).permute(0, 2, 1)
    inputs = batch.float()
    # win_lens = np.random.randint(29, 50, size=batch.shape[-1])
    # inputs = batch.float()
    # inputs_filtered = torch.zeros_like(inputs)
    # for i in range(batch.shape[-1]):
    #     inputs_filtered[:, i] = filter((inputs[:, i]-inputs[0, i]).detach().cpu().numpy(), win_len=win_lens[i], dtype=torch.Tensor)

    # decode a batch of sequences, rescale it with scaler and plot them
    outputs = model.decode(model.encode(inputs.to(model.device)))
    # outputs = scaler.inverse_transform(outputs.detach().cpu().numpy())
    # inputs = scaler.inverse_transform(inputs.detach().cpu().numpy())
    fig, axs = plt.subplots(10, 1, figsize=(10, 10), sharex=True)
    batch_num = np.random.randint(0, batch.shape[0])
    for i in range(10):
        feature = np.random.randint(0, inputs.shape[-1])
        # out = scaler.inverse_transform(outputs[i, :].detach().cpu().numpy())
        # inp = scaler.inverse_transform(inputs[i, :].detach().cpu().numpy())
        axs[i].plot(inputs[batch_num, :, feature].detach().cpu().numpy(), label='Original')
        # axs[i].plot(inputs_filtered[:, stock].detach().cpu().numpy(), label='Filter')
        axs[i].plot(outputs[batch_num, :, feature].detach().cpu().numpy(), label='Reconstructed')
        # axs[i, 1].plot(np.cumsum(inputs[:, stock].detach().cpu().numpy()), label='Original')
        # axs[i, 1].plot(np.cumsum(inputs_filtered[:, stock].detach().cpu().numpy()), label='Filter')
        # axs[i, 1].plot(np.cumsum(outputs[:, stock].detach().cpu().numpy()), label='Reconstructed')
    plt.legend()
    plt.show()

