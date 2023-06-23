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
from matplotlib import pyplot as plt

from stable_baselines3 import PPO, SAC, DDPG, TD3
from stable_baselines3.common.evaluation import evaluate_policy

# from nn_architecture.agents import SACAgent, DDPGAgent, TD3Agent
from utils.ae_dataloader import create_dataloader
from training import simple_train, simple_test
from environment import Environment

import gym
from gymnasium.wrappers import TimeLimit


if __name__ == '__main__':
    """main file for training and testing the SAC-agent on the environment"""

    # warnings off
    warnings.filterwarnings("ignore")
    cfg = {
        # general parameters
        'load_checkpoint': False,
        'file_checkpoint': 'trained_rl/checkpoint',
        'file_data': os.path.join('stock_data', 'stocks_sp20_2010_2020.csv'),
        'file_predictor': [None, None],  # ['trained_gan/real_gan_1k.pt', 'trained_gan/mvgavg_gan_10k.pt',],
        'checkpoint_interval': 10,

        # training parameters
        'train': True,
        'agent': 'ppo',
        'env_id': "Custom",  # Custom, Pendulum-v1, MountainCarContinuous-v0, LunarLander-v2
        'num_epochs': 5,
        'num_actions_per_epoch': 1e3,
        'num_random_actions': 5e2,
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
    # agent_dict structure:
    # key: agent name
    # value: (agent constructor, agent load function, bool: does agent support discrete actions?)
    agent_dict = {'sac': (lambda policy, env: SAC(policy, env, verbose=1),
                          lambda path: SAC.load(path),
                          False),
                  'ddpg': (lambda policy, env: DDPG(policy, env, verbose=1),
                           lambda path: DDPG.load(path),
                           False),
                  'td3': (lambda policy, env: TD3(policy, env, verbose=1),
                          lambda path: TD3.load(path),
                          False),
                  'ppo': (lambda policy, env: PPO(policy, env, verbose=1),
                          lambda path: PPO.load(path),
                          True),
                    }

    print('Initializing framework...')

    # load data
    training_data = pd.read_csv(cfg['file_data'], index_col=0, header=0).to_numpy(dtype=np.float32)
    test_data = training_data[int(cfg['train_test_split']*len(training_data)):]
    training_data = training_data[:int(cfg['train_test_split']*len(training_data))]

    # load environment
    if cfg['env_id'] == 'Custom':
        env = Environment(training_data, cfg['cash_init'], cfg['observation_length'], time_limit=cfg['time_limit'], discrete_actions=agent_dict[cfg['agent']][2])
        # env = TimeLimit(env, max_episode_steps=cfg['time_limit'])
    else:
        env = gym.make(cfg['env_id'], render_mode="human")

    # load agent
    if not cfg['load_checkpoint']:
        agent = agent_dict[cfg['agent']][0]('MlpPolicy', env)
        print(f"Agent {cfg['agent']} initialized!")
    else:
        agent = agent_dict[cfg['agent']][1](cfg['file_checkpoint'])
        print(f"Agent {cfg['agent']} from path {cfg['file_checkpoint']} loaded!")

    # --------------------------------------------
    # train RL framework
    # --------------------------------------------

    if cfg['train']:
        avg_rewards, avg_stds = [], []
        for i in range(int(cfg['num_epochs'])):
            agent.learn(total_timesteps=cfg['num_actions_per_epoch'], log_interval=10)
            avg_reward, avg_std = evaluate_policy(agent, env, n_eval_episodes=5)
            avg_rewards.append(avg_reward)
            avg_stds.append(avg_std)
            print(f"Epoch {i}/{int(cfg['num_epochs'])}: avg_reward={avg_reward:.2f} +/- {avg_std:.2f}")

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
    # save agent
    # --------------------------------------------

    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    # path = os.path.join('trained_rl', f'{cfg["agent"]}_{cfg["env_id"]}_{current_time}')
    path = os.path.join('trained_rl', f'checkpoint.pt')
    agent.save(path)
    print(f"Agent saved to {path}")

    # --------------------------------------------
    # test RL framework
    # --------------------------------------------
    # load environment
    if cfg['env_id'] == 'Custom':
        env = Environment(test_data, cfg['cash_init'], cfg['observation_length'], time_limit=-1,discrete_actions=agent_dict[cfg['agent']][2])
        # env = TimeLimit(env, max_episode_steps=cfg['time_limit'])
    else:
        env = gym.make(cfg['env_id'], render_mode="human")    # rewards, std = evaluate_policy(agent, env, n_eval_episodes=1, return_episode_rewards=True)
    simple_test(env, agent, deterministic=True)