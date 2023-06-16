"""Reinforcement learning (RL) framework for developing an agent which can trade with stocks in the stock market
The framework uses a deep reinforcement learning algorithm called Soft-Actor-Critic (SAC) to train the agent
The agent is trained to maximize the cumulative reward
The agent is trained on a dataset of stock market data
The environment simulates a real-time stock market with fees, slippage, and other factors
The prediction of the stock price is done by a generative adversarial network (GAN) generator
The GAN generator is trained on a dataset of stock market data which is not used for training the agent
Future implementations will include a sentiment analysis model based on the news headlines to predict the stock price"""
import copy
import os
import warnings
from datetime import datetime

import numpy as np

import torch
from matplotlib import pyplot as plt
from torch import nn

from nn_architecture.agents import SACAgent, DDPGAgent, TD3Agent
from nn_architecture.models import TransformerGenerator2, TtsDiscriminator
from nn_architecture.transformer_autoencoder import TransformerAutoencoder, TransformerAutoencoder_v0
from utils.ae_dataloader import create_dataloader
from training import simple_train, simple_test
from environment import Environment

import gymnasium as gym


if __name__ == '__main__':
    """main file for training and testing the SAC-agent on the environment"""

    # warnings off
    warnings.filterwarnings("ignore")
    cfg = {
        # general
        'load_checkpoint': False,
        'file_checkpoint': os.path.join('trained_rl', 'td3_pend2.pt'),
        'file_data': os.path.join('stock_data', 'stocks_sp1_2010_2020.csv'),
        'file_predictor': [None, None],  # ['trained_gan/real_gan_1k.pt', 'trained_gan/mvgavg_gan_10k.pt',],
        'checkpoint_interval': 10,

        # rl training
        'train': False,
        'agent': 'ddpg',
        'env_id': "Pendulum-v1",
        'max_episodes': 1e0,
        'batch_size': 32,
        'num_random_actions': 1e3,
        'train_test_split': 0.8,
        'replay_buffer_size': 1e5,
        'hidden_dim': 64,
        'num_layers': 3,
        'num_layers_sub': 4,
        'temperature': 10,
        'learning_rate': 1e-4,
        'init_w': 3e-3,
        'reward_scaling': 1e-4,

        # environment
        'cash_init': 10000,
        'commission': .001,
        'observation_length': 16,
    }

    list_valid_agents = ['sac', 'ddpg', 'td3']
    assert cfg['agent'] in list_valid_agents, f"Agent must be one of: {list_valid_agents}"

    print('Initializing framework...')

    # --------------------------------------------
    # load stock data
    # --------------------------------------------

    # get stock prices from csv file in dir stock_data
    # train_dl, test_dl, _ = create_dataloader(cfg['file_data'], seq_len=-1, batch_size=cfg['batch_size'], standardize=False, train_ratio=cfg['train_test_split'])
    # _, _, scaler = create_dataloader(cfg['file_data'], seq_len=-1, batch_size=cfg['batch_size'], standardize=False, differentiate=True, train_ratio=cfg['train_test_split'])

    # --------------------------------------------
    # initialize framework
    # --------------------------------------------

    # Initialize prediction models
    # predictors = []
    # for i, file in enumerate(cfg['file_predictor']):
    #     if file:
    #         predict_dict = torch.load(file, map_location=torch.device('cpu'))
    #         predictor = TransformerGenerator2(**predict_dict['configuration']['generator'])
    #         predictor.load_state_dict(predict_dict['generator'])
    #         predictor.eval()
    #         predictor.to(predictor.device)
    #         predictors.append(copy.deepcopy(predictor))
    #     else:
    #         predictors.append(None)
    #
    # # Initialize DataProcessor
    # data_processor = DataProcessor()
    # processor_cfg = {
    #     'ptype': ("short", "long"),
    #     'scaler': (scaler, scaler),
    #     'differentiate': (True, True),
    #     'downsampling_rate': (None, 10),
    #     'mvg_avg': (None, 50),
    #     'predictor': predictors,
    # }

    # add data processors
    # for i in range(len(predictors)):
    #     processor_dict = {}
    #     for key in processor_cfg.keys():
    #         processor_dict[key] = processor_cfg[key][i]
    #     data_processor.add_processor(**processor_dict)

    # get state dimension from number of processors and their feature dimension
    # keys_data_processor = data_processor.processor.keys()
    # default state_dim is len(portfolio)

    # add predictions (features*seq_len) of each processor to state_dim
    # for k in keys_data_processor:
    #     # get number of observed features (can be different for each processor due to autoencoder)
    #     if data_processor.processor[k]["autoencoder"]:
    #         features = data_processor.processor[k]["autoencoder"].output_dim
    #     else:
    #         # if no autoencoder is used, use number of features of original data
    #         features = train_dl.dataset.data.shape[-1]
    #
    #     # get sequence length of processor
    #     if data_processor.processor[k]["predictor"]:
    #         seq_len = data_processor.processor[k]["predictor"].seq_len
    #     else:
    #         # if no predictor is used, use observation length
    #         seq_len = cfg['observation_length']
    #     state_dim += features * seq_len

    # get action dimension from number of stocks or from autoencoder output
    # if "short" in data_processor.processor.keys() and data_processor.processor["short"]["autoencoder"] and False:
    #     action_dim = data_processor.processor["short"]["autoencoder"].output_dim
    # else:
    #     action_dim = train_dl.dataset.data.shape[-1]

    # Initialize environment and agent
    # env = Environment(train_dl.dataset.data.squeeze(0).numpy(),
    #                   commission=cfg['commission'], cash=cfg['cash_init'], reward_scaling=cfg['reward_scaling'],
    #                   observation_length=cfg['observation_length'])
    # env = gym.make("LunarLander-v2", render_mode="human")
    # env = gym.make("MountainCarContinuous-v0", render_mode="human")  # "LunarLander-v2"
    env = gym.make(cfg['env_id'], render_mode="human")

    state_dim = env.observation_space.shape[0]  # 1 + cfg['observation_length'] + train_dl.dataset.data.shape[-1]  # train_dl.dataset.data.shape[2]
    action_dim = env.action_space.shape[0]  # train_dl.dataset.data.shape[-1]
    if cfg['agent'] == 'ddpg':
        agent = DDPGAgent(state_dim=state_dim, action_dim=action_dim, hidden_dim=cfg['hidden_dim'],
                          num_layers=cfg['num_layers'], learning_rate=cfg['learning_rate'],
                          init_w=cfg['init_w'], replay_buffer_size=cfg['replay_buffer_size'],
                          limit_low=env.action_space.low, limit_high=env.action_space.high)
    elif cfg['agent'] == 'td3':
        agent = TD3Agent(state_dim=state_dim, action_dim=action_dim, hidden_dim=cfg['hidden_dim'],
                         num_layers=cfg['num_layers'], learning_rate=cfg['learning_rate'],
                         init_w=cfg['init_w'], replay_buffer_size=cfg['replay_buffer_size'],
                         limit_low=env.action_space.low, limit_high=env.action_space.high)
    elif cfg['agent'] == 'sac':
        agent = SACAgent(state_dim=state_dim, action_dim=action_dim, hidden_dim=cfg['hidden_dim'],
                         num_layers=cfg['num_layers'], learning_rate=cfg['learning_rate'],
                         init_w=cfg['init_w'], replay_buffer_size=cfg['replay_buffer_size'],
                         limit_low=env.action_space.low, limit_high=env.action_space.high)
    else:
        raise NotImplementedError(f"Agent of type {cfg['agent']} is not implemented.")

    # if isinstance(agent, RoboBoboDDPG) or isinstance(agent, RoboBoboSAC):
    #     # create policy sub networks based on the data_processor predictor outputs
    #     policy_sub_networks = nn.ModuleList()
    #     for i, k in enumerate(keys_data_processor):
    #         if data_processor.processor[k]["predictor"]:
    #             seq_len = data_processor.processor[k]["predictor"].seq_len
    #             input_dim = data_processor.processor[k]["predictor"].channels
    #         else:
    #             seq_len = cfg['observation_length']
    #             input_dim = train_dl.dataset.data.shape[-1]
    #         hidden_dim = np.min((50, train_dl.dataset.data.shape[-1]))
    #         policy_sub_networks.append(
    #             agent.create_policy_sub_network(input_dim=input_dim, hidden_dim=hidden_dim, seq_len=seq_len,
    #                                             lstm=True, num_layers=cfg['num_layers_sub'], dropout=0.25))
    #     agent.adjust_nets(policy_sub_networks=policy_sub_networks)

    if cfg['load_checkpoint']:
        agent.load_checkpoint(cfg['file_checkpoint'])

    # --------------------------------------------
    # train RL framework
    # --------------------------------------------

    if cfg['train']:
        # Start training
        episode_rewards, agent = simple_train(env, agent, #data_processor,
                                              max_episodes=cfg['max_episodes'],
                                              batch_size=cfg['batch_size'],
                                              parameter_update_interval=1,
                                              path_checkpoint=cfg['file_checkpoint'],
                                              checkpoint_interval=cfg['checkpoint_interval'],
                                              num_random_actions=cfg['num_random_actions'],)

        path_save = os.path.join('trained_rl', 'sac_' + datetime.now().strftime("%Y%m%d-%H%M%S") + '.pt')
        agent.save_checkpoint(path_save)
        print('Saved checkpoint to {}'.format(path_save))

        plt.plot(episode_rewards, label='episode_rewards')
        plt.plot(np.convolve(episode_rewards, np.ones(10)/10, mode='valid'), label='Avg. rewards')
        plt.ylabel('Reward')
        plt.xlabel('Episode')
        plt.title('Reward after each episode')
        plt.legend()
        plt.show()

    # test trained agent on test data
    print('Testing agent on test data')
    # env_test = Environment(test_dl.dataset.data.squeeze(0).numpy(), cash=cfg['cash_init'], observation_length=cfg['observation_length'], commission=cfg['commission'])
    simple_test(gym.make(cfg['env_id'], render_mode="human"), agent, test=False, plot=True, plot_reference=False)
    # simple_test_ddpg(env_test, agent, test=False, plot_reference=False)