"""Reinforcement learning (RL) framework for developing an agent which can trade with stocks in the stock market
The framework uses a deep reinforcement learning algorithm called Soft-Actor-Critic (SAC) to train the agent
The agent is trained to maximize the cumulative reward
The agent is trained on a dataset of stock market data
The environment simulates a real-time stock market with fees, slippage, and other factors
The prediction of the stock price is done by a generative adversarial network (GAN) generator
The GAN generator is trained on a dataset of stock market data which is not used for training the agent
Future implementations will include a sentiment analysis model based on the news headlines to predict the stock price"""

import os
import warnings
from datetime import datetime

import numpy as np

import torch
from matplotlib import pyplot as plt
from torch import nn

from nn_architecture.models import TransformerGenerator2, TtsDiscriminator
from nn_architecture.transformer_autoencoder import TransformerAutoencoder, TransformerAutoencoder_v0
from utils.ae_dataloader import create_dataloader
from sac_workingV import train_sac, test_sac, DataProcessor, RoboBobo
from environment import Environment


if __name__ == '__main__':
    """main file for training and testing the SAC-agent on the environment"""

    # warnings off
    warnings.filterwarnings("ignore")
    cfg = {
        # general
        'load_checkpoint': False,
        'file_checkpoint': os.path.join('trained_rl', 'deepsac_390ep.pt'),
        'file_data': os.path.join('stock_data', 'stocks_sp499_2010_2020.csv'),
        'file_gan_1': os.path.join('trained_gan', 'real_gan_1k.pt'),
        'file_gan_2': os.path.join('trained_gan', 'mvgavg_gan_10k.pt'),
        # 'file_encoder': os.path.join('trained_ae', 'ae_kagglev1.pth'),
        # 'file_encoder_bp': os.path.join('trained_ae', 'mvgavg50_down10_ae.pt'),
        # 'file_filter_bp': None,#os.path.join('filter', 'stocks_sp500_2010_2020_filter_dict.pt'),
        'checkpoint_interval': 10,

        # rl training
        'train': True,
        'max_episodes': 3,
        'batch_size': 8,
        'num_random_actions': 8,
        'train_test_split': 0.8,
        'replay_buffer_size': 10000,
        'hidden_dim': 1024,
        'num_layers': 6,
        'temperature': 1,

        # environment
        'cash_init': 10000,
        'commission': .001,
        'observation_length': 16,
    }


    # get stock data from yahoo finance and store in open price in dataframe with stock name as header and datetime as index
    # stock_ls = ['AAPL', 'AMZN', 'GOOG', 'MSFT', 'TSLA']
    # df = pd.DataFrame()
    # for s in stock_ls:
    #     # data = yf.download(s, start='2023-04-11', end='2023-04-18', interval="1m", auto_adjust=True)
    #     data = yf.download(s, start='2010-01-01', end='2020-12-31', auto_adjust=True)
    #     df[s] = data['Open']
    #     df[s] = df[s].mask(df[s].isna(), 0)  # mask all nan values with -1

    print('Initializing framework...')

    # --------------------------------------------
    # load stock data
    # --------------------------------------------

    # get stock prices from csv file in dir stock_data
    train_dl, test_dl, _ = create_dataloader(cfg['file_data'], seq_len=-1, batch_size=cfg['batch_size'], standardize=False, train_ratio=cfg['train_test_split'])
    _, _, scaler = create_dataloader(cfg['file_data'], seq_len=-1, batch_size=cfg['batch_size'], standardize=False, differentiate=True, train_ratio=cfg['train_test_split'])


    # --------------------------------------------
    # initialize framework
    # --------------------------------------------

    # Initialize encoder model
    # encoder_dict = torch.load(cfg['file_encoder'], map_location=torch.device('cpu'))
    # scaler = encoder_dict['general']['scaler']
    # encoder = TransformerAutoencoder_v0(**encoder_dict["model"])
    # encoder.load_state_dict(encoder_dict['model']['state_dict'])
    # encoder.eval()

    # Initialize bandpass encoder model
    # encoder_dict_bp = torch.load(cfg['file_encoder_bp'], map_location=torch.device('cpu'))
    # scaler_bp = encoder_dict['general']['scaler']
    # encoder_bp = TransformerAutoencoder(**encoder_dict_bp["model"])
    # encoder_bp.load_state_dict(encoder_dict_bp['model']['state_dict'])
    # encoder_bp.eval()

    # Initialize prediction model for short-term prediction (with unfiltered data)
    gan_1_dict = torch.load(cfg['file_gan_1'], map_location=torch.device('cpu'))
    generator_1 = TransformerGenerator2(**gan_1_dict['configuration']['generator'])
    generator_1.load_state_dict(gan_1_dict['generator'])
    generator_1.eval()
    generator_1.to(generator_1.device)

    # Initialize prediction model for long-term prediction (low-pass filtered data)
    gan_2_dict = torch.load(cfg['file_gan_2'], map_location=torch.device('cpu'))
    generator_2 = TransformerGenerator2(**gan_2_dict['configuration']['generator'])
    # delete all keys from state_dict which carry the name 'decoder'
    # for key in list(gan_2_dict['generator'].keys()):
    #     if 'decoder' in key:
    #         del fc_bp_dict['generator'][key]
    generator_2.load_state_dict(gan_2_dict['generator'])
    generator_2.eval()
    generator_2.to(generator_2.device)

    # Initialize DataProcessor
    data_processor = DataProcessor()
    # add short-term prediction
    data_processor.add_processor("short", predictor=generator_1, scaler=scaler, differentiate=True)
    # add long-term prediction
    # TODO: change scaler to scaler_mvgavg
    data_processor.add_processor("long", predictor=generator_2, mvg_avg=50, downsampling_rate=10, scaler=scaler, differentiate=True)

    # get state dimension from number of processors and their feature dimension
    keys_data_processor = data_processor.processor.keys()
    # default state_dim is len(portfolio)
    state_dim = 1  # train_dl.dataset.data.shape[2]
    # add predictions (features*seq_len) of each processor to state_dim
    for k in keys_data_processor:
        # get number of observed features (can be different for each processor due to autoencoder)
        if data_processor.processor[k]["autoencoder"]:
            features = data_processor.processor[k]["autoencoder"].output_dim
        else:
            # if no autoencoder is used, use number of features of original data
            features = train_dl.dataset.data.shape[-1]

        # get sequence length of processor
        if data_processor.processor[k]["predictor"]:
            seq_len = data_processor.processor[k]["predictor"].seq_len
        else:
            # if no predictor is used, use observation length
            seq_len = cfg['observation_length']
        state_dim += features * seq_len

    # get action dimension from number of stocks or from autoencoder output
    if "short" in data_processor.processor.keys() and data_processor.processor["short"]["autoencoder"] and False:
        action_dim = data_processor.processor["short"]["autoencoder"].output_dim
    else:
        action_dim = train_dl.dataset.data.shape[-1]

    # Initialize environment and agent
    env = Environment(train_dl.dataset.data.squeeze(0), commission=cfg['commission'], cash=cfg['cash_init'], observation_length=cfg['observation_length'])

    agent = RoboBobo(env, state_dim=state_dim, action_dim=action_dim, temperature=cfg['temperature'], hidden_dim=cfg['hidden_dim'], num_layers=cfg['num_layers'], replay_buffer_size=cfg['replay_buffer_size'])
    # create policy sub networks based on the data_processor predictor outputs
    policy_sub_networks = nn.ModuleList()
    for i, k in enumerate(keys_data_processor):
        if data_processor.processor[k]["predictor"]:
            seq_len = data_processor.processor[k]["predictor"].seq_len
            policy_sub_networks.append(agent.create_policy_sub_network(data_processor.processor[k]["predictor"].channels, 50, seq_len=seq_len, lstm=True, num_layers=2, dropout=0.25))
    agent.adjust_nets(policy_sub_networks=policy_sub_networks)

    if cfg['load_checkpoint']:
        agent.load_checkpoint(cfg['file_checkpoint'])

    # --------------------------------------------
    # train RL framework
    # --------------------------------------------

    if cfg['train']:
        # Start training
        total_equity_final, agent \
            = train_sac(env, agent, data_processor,
                        max_episodes=cfg['max_episodes'],
                        batch_size=cfg['batch_size'],
                        parameter_update_interval=1,
                        path_checkpoint=cfg['file_checkpoint'],
                        checkpoint_interval=cfg['checkpoint_interval'],
                        num_random_actions=cfg['num_random_actions'],)

        path_save = os.path.join('trained_rl', 'sac_' + datetime.now().strftime("%Y%m%d-%H%M%S") + '.pt')
        agent.save_checkpoint(path_save, reward=total_equity_final)

        plt.plot(total_equity_final, label='Total final equity [$]')
        plt.plot(np.convolve(total_equity_final, np.ones(10)/10, mode='valid'), label='Avg. total final equity [$]')
        plt.ylabel('Total final equity [$]')
        plt.xlabel('Episode')
        plt.title('Total final equity after each episode in [$]')
        plt.legend()
        plt.show()

    # test trained agent on test data
    print('Testing agent on test data')
    env_test = Environment(test_dl.dataset.data.squeeze(0), cash=cfg['cash_init'], observation_length=cfg['observation_length'], commission=cfg['commission'])
    test_sac(env_test, agent, data_processor, test=True, plot_reference=False)
    test_sac(env_test, agent, data_processor, test=False, plot_reference=True)