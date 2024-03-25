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
from helpers.init_ae import init_ae

from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO, SAC, DDPG, TD3
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env

# from nn_architecture.agents import SACAgent, DDPGAgent, TD3Agent
from utils.ae_dataloader import create_dataloader
from training import simple_train, test
from environment import Environment
from nn_architecture.rl_networks import *

import gymnasium as gym


if __name__ == '__main__':
    """main file for training and testing the SAC-agent on the environment"""

    # warnings off
    warnings.filterwarnings("ignore")
    cfg = {
        # general parameters
        'load_checkpoint': False,
        'file_checkpoint': 'trained_rl/checkpoint.pt',
        'file_data': os.path.join('stock_data', 'portfolio_custom129_SHORT.csv'),
        'file_predictor': [None, None],  # ['trained_gan/real_gan_1k.pt', 'trained_gan/mvgavg_gan_10k.pt',],
        'file_ae': 'trained_ae/ae129.pt',
        'checkpoint_interval': 10,

        # rl setup parameters
        'train': True,
        'agent': 'sac',
        'env_id': 'Custom',  # Custom, Pendulum-v1, MountainCarContinuous-v0, LunarLander-v2
        'policy': 'MlpPolicy',  # MlpPolicy, Attn, AttnLstm
        'recurrent': True,

        # training parameters
        'num_epochs': 1e2,
        'num_actions_per_epoch': 1e3,
        'num_random_actions': 5e2,
        'batch_size': 32,
        'learning_rate': 3e-4,
        'train_test_split': 0.8,
        'replay_buffer_size': int(1e4),
        'parameter_update_interval': 50,

        # network parameters
        'hidden_dim': 32,
        'num_layers': 3,
        'num_layers_sub': 4,
        'init_w': None,

        # environment
        'time_limit': 365,
        'cash_init': 10000,
        'commission': .001,
        'observation_length': 16,
        'reward_scaling': 1e-4,
    }

    # make checks
    valid_policies = ['MlpPolicy', 'Attn', 'AttnLstm']
    assert cfg['policy'] in valid_policies, f"Policy must be one of: {valid_policies}"

    if not cfg['recurrent'] and (cfg['policy'] == 'Attn' or cfg['policy'] == 'AttnLstm'):
        cfg['recurrent'] = True
        print('Recurrent policy selected, setting recurrent to True')

    list_valid_agents = ['sac', 'ddpg', 'td3', 'ppo_cont', 'ppo_disc', 'rppo']
    assert cfg['agent'] in list_valid_agents, f"Agent must be one of: {list_valid_agents}"

    # agent_dict structure:
    # key: agent name
    # value: (agent constructor, agent load function, bool: does agent support discrete actions?)
    agent_dict = {'sac': (lambda policy, env, policy_kwargs: SAC(policy, env,
                                                                 policy_kwargs=policy_kwargs,
                                                                 buffer_size=cfg['replay_buffer_size'],
                                                                 train_freq=cfg['parameter_update_interval'],
                                                                 gradient_steps=cfg['parameter_update_interval'],),
                          lambda path, env: SAC.load(path, env),
                          False),
                  'ddpg': (lambda policy, env, policy_kwargs: DDPG(policy, env, policy_kwargs=policy_kwargs),
                           lambda path, env: DDPG.load(path, env),
                           False),
                  'td3': (lambda policy, env, policy_kwargs: TD3(policy, env, policy_kwargs=policy_kwargs),
                          lambda path, env: TD3.load(path, env),
                          False),
                  'ppo_cont': (lambda policy, env, policy_kwargs: PPO(policy, env, policy_kwargs=policy_kwargs),
                               lambda path, env: PPO.load(path, env, print_system_info=True),
                               False),
                  'ppo_disc': (lambda policy, env, policy_kwargs: PPO(policy, env, policy_kwargs=policy_kwargs),
                               lambda path, env: PPO.load(path, env, print_system_info=True),
                               True),
                  'rppo': (lambda policy, env, policy_kwargs: RecurrentPPO("MlpLstmPolicy", env, policy_kwargs=policy_kwargs),
                           lambda path, env: RecurrentPPO.load(path, env, print_system_info=True),
                           False),
                 }

    print('Initializing framework...')

    # load data
    training_data = pd.read_csv(cfg['file_data'], index_col=0, header=0)
    portfolio_names = training_data.columns
    training_data = training_data.to_numpy(dtype=np.float32)
    test_data = training_data[int(cfg['train_test_split']*len(training_data)):]
    training_data = training_data[:int(cfg['train_test_split']*len(training_data))]

    # load environment
    if cfg['env_id'] == 'Custom':
        if cfg['file_ae'] is not None and cfg['file_ae'] != '':
            state_dict = torch.load(cfg['file_ae'], map_location=torch.device('cpu'))
            encoder = init_ae(**state_dict['configuration'], sequence_length=16)
            encoder.load_state_dict(state_dict['model'])
            encoder.to(torch.device('cpu'))
            encoder.eval()
        else:
            encoder = None
        env = Environment(training_data, portfolio_names, cfg['cash_init'], cfg['observation_length'],
                          time_limit=cfg['time_limit'], discrete_actions=agent_dict[cfg['agent']][2],
                          recurrent=cfg["recurrent"], encoder=encoder)
        # env = TimeLimit(env, max_episode_steps=cfg['time_limit'])
        # It will check your custom environment and output additional warnings if needed
        check_env(env)
    else:
        env = gym.make(cfg['env_id'], render_mode="human")

    # custom network architecture and features extractor
    if cfg['policy'] == 'Attn':
        policy = AttnActorCriticPolicyOn
    elif cfg['policy'] == 'AttnLstm':
        policy = AttnLstmActorCriticPolicyOn
    else:
        # if cfg['policy'] == 'MlpPolicy'
        policy = 'MlpPolicy'

    if cfg['policy'] == 'MlpPolicy' and cfg['recurrent']:
        feature_extractor = LSTMFeatureExtractor
    elif (cfg['policy'] == 'Attn' or cfg['policy'] == 'AttnLstm') and cfg['recurrent']:
        feature_extractor = BasicFeatureExtractor
    else:
        feature_extractor = None

    net_arch = [cfg['hidden_dim'] for _ in range(cfg['num_layers'])]
    if cfg['agent'] == 'sac':
        net_arch = dict(pi=net_arch, qf=net_arch)
    else:
        net_arch = dict(pi=net_arch, vf=net_arch)

    if feature_extractor is not None:
        policy_kwargs = dict(
            features_extractor_class=feature_extractor,
            features_extractor_kwargs=dict(feature_dim=cfg['hidden_dim']),
        )
    else:
        # policy_kwargs = None
        policy_kwargs = dict(net_arch=net_arch)

    # load agent
    if not cfg['load_checkpoint']:
        agent = agent_dict[cfg['agent']][0](policy, env, policy_kwargs)
        print(f"Configuration:\n\tAgent:\t\t{cfg['agent']}\n\tPolicy:\t\t{cfg['policy']}\n\tRecurrent:\t{cfg['recurrent']}\n\tEnvironment:\t{cfg['env_id']}\n\tEncoder:{cfg['file_ae']}")
    else:
        agent = agent_dict[cfg['agent']][1](cfg['file_checkpoint'], env)
        # load replay buffer if agent is off-policy and replay buffer is available
        if cfg['agent'] in ['sac', 'ddpg', 'td3']:
            if os.path.exists(os.path.join('trained_rl', 'replay_buffer_checkpoint.pt')):
                agent.load_replay_buffer(os.path.join('trained_rl', 'replay_buffer_checkpoint.pt'))
                print(f"Replay buffer loaded from path {os.path.join('trained_rl', 'replay_buffer_checkpoint.pt')}!")
        print(f"Agent {cfg['agent']} loaded from path {cfg['file_checkpoint']}!")

    # --------------------------------------------
    # train RL framework
    # --------------------------------------------

    if cfg['train']:
        avg_rewards, avg_stds, best_reward = [], [], 0
        try:
            for i in range(int(cfg['num_epochs'])):
                agent.learn(total_timesteps=cfg['num_actions_per_epoch'])
                avg_reward, avg_std = evaluate_policy(agent, env, n_eval_episodes=5)
                avg_rewards.append(avg_reward)
                avg_stds.append(avg_std)
                if avg_reward > best_reward:
                    best_reward = avg_reward
                    agent.save("best_checkpoint.pt")
                if cfg['checkpoint_interval'] is not None and i % cfg['checkpoint_interval'] == 0:
                    agent.save("checkpoint.pt")
                print(f"Epoch {i+1}/{int(cfg['num_epochs'])}: avg_reward={avg_reward:.2f} +/- {avg_std:.2f}")
        except KeyboardInterrupt:
            print("Training interrupted by user. Continuing with next operations...")
            
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

        # save replay buffer if agent is off-policy
        if cfg['agent'] in ['sac', 'ddpg', 'td3']:
            agent.save_replay_buffer(os.path.join('trained_rl', 'replay_buffer_checkpoint.pt'))

        print(f"Agent saved to {path}")

    # --------------------------------------------
    # test RL framework
    # --------------------------------------------

    # load environment
    if cfg['env_id'] == 'Custom':
        # test_data = np.concatenate((training_data, test_data), axis=0)
        env = Environment(test_data, portfolio_names, cfg['cash_init'], cfg['observation_length'], time_limit=-1, test=True,
                          discrete_actions=agent_dict[cfg['agent']][2], recurrent=cfg["recurrent"], encoder=encoder)        # env = TimeLimit(env, max_episode_steps=cfg['time_limit'])
    else:
        env = gym.make(cfg['env_id'], render_mode="human")    # rewards, std = evaluate_policy(agent, env, n_eval_episodes=1, return_episode_rewards=True)
    mean, std = evaluate_policy(agent, env, n_eval_episodes=1, deterministic=True)
    print(f"Average reward over {1} episodes: {mean:.2f} +/- {std:.2f}")
    # test(env, agent, deterministic=True, title=cfg['file_checkpoint'].split('/')[-1].split('.')[0])