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
import pandas as pd

import torch
from matplotlib import pyplot as plt
from torch import nn

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.evaluation import evaluate_policy

# from nn_architecture.agents import SACAgent, DDPGAgent, TD3Agent
from utils.ae_dataloader import create_dataloader
from training import simple_train, simple_test
from environment import Environment

import gymnasium as gym
from gymnasium.wrappers import TimeLimit


if __name__ == '__main__':
    """main file for training and testing the SAC-agent on the environment"""

    # warnings off
    warnings.filterwarnings("ignore")
    cfg = {
        # general parameters
        'load_checkpoint': False,
        'file_checkpoint': os.path.join('trained_rl', 'sac_pend_1000ep.pt'),
        'file_data': os.path.join('stock_data', 'stocks_sp20_2010_2020.csv'),
        'file_predictor': [None, None],  # ['trained_gan/real_gan_1k.pt', 'trained_gan/mvgavg_gan_10k.pt',],
        'checkpoint_interval': 10,

        # training parameters
        'train': True,
        'agent': 'ppo',
        'env_id': "MountainCarContinuous-v0",  # Pendulum-v1, MountainCarContinuous-v0, LunarLander-v2
        'num_actions': 1e5,
        'num_random_actions': 5e2,
        'eval_interval': 1,
        'batch_size': 32,
        'learning_rate': 3e-4,
        'temperature': 0.0001,
        'train_test_split': 0.8,
        'replay_buffer_size': 1e6,
        'parameter_update_interval': 50,
        'polyak': 0.995,
        'gamma': 0.99,

        # network parameters
        'hidden_dim': 256,
        'num_layers': 3,
        'num_layers_sub': 4,
        'init_w': None,
        'dropout': 0.0,

        # environment
        'time_limit': 365,
        'cash_init': 10000,
        'commission': .001,
        'observation_length': 16,
        'reward_scaling': 1e-4,
    }

    list_valid_agents = ['sac', 'ddpg', 'td3', 'ppo']
    assert cfg['agent'] in list_valid_agents, f"Agent must be one of: {list_valid_agents}"

    print('Initializing framework...')

    # env = gym.make(cfg['env_id'], render_mode="human")
    training_data = pd.read_csv(cfg['file_data'], index_col=0, header=0).to_numpy(dtype=np.float32)
    test_data = training_data[int(cfg['train_test_split']*len(training_data)):]
    training_data = training_data[:int(cfg['train_test_split']*len(training_data))]
    env = Environment(training_data, cfg['cash_init'], cfg['observation_length'], time_limit=cfg['time_limit'])
    # env = TimeLimit(env, max_episode_steps=cfg['time_limit'])
    agent = PPO('MlpPolicy', env)

    # --------------------------------------------
    # train RL framework
    # --------------------------------------------

    if cfg['train']:
        avg_rewards, avg_stds = [], []
        num_epochs = int(cfg['num_actions'] / (cfg['time_limit']*cfg['eval_interval']))
        for i in range(num_epochs):
            agent.learn(total_timesteps=cfg['time_limit']*cfg['eval_interval'], log_interval=10)
            avg_reward, avg_std = evaluate_policy(agent, env, n_eval_episodes=5)
            avg_rewards.append(avg_reward)
            avg_stds.append(avg_std)
            print(f"Epoch {i}/{num_epochs}: avg_reward={avg_reward:.2f} +/- {avg_std:.2f}")

        # --------------------------------------------
        # plot results
        # --------------------------------------------

        plt.figure()
        plt.plot(avg_rewards)
        plt.fill_between(range(len(avg_rewards)), np.array(avg_rewards)-np.array(avg_stds), np.array(avg_rewards)+np.array(avg_stds), alpha=0.2)
        plt.xlabel('Epoch')
        plt.ylabel('Average reward')
        plt.show()

    # --------------------------------------------
    # test RL framework
    # --------------------------------------------
    env = Environment(test_data, cfg['cash_init'], cfg['observation_length'], time_limit=-1,)
    # rewards, std = evaluate_policy(agent, env, n_eval_episodes=1, return_episode_rewards=True)
    # print(f"Test reward: {rewards[0]} +/- {std[0]}")
    # plt.figure()
    # plt.plot(rewards[0])
    simple_test(env, agent, test=True, plot_reference=True)