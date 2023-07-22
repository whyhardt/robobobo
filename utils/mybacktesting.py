# backtesting a trained agent in a verified backtesting environment
import os
import argparse
from abc import ABC
from typing import Union

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from backtesting import Backtest, Strategy
from stable_baselines3 import PPO, SAC, A2C, DDPG, TD3

from environment import Environment
from nn_architecture.ae_networks import TransformerAutoencoder


def collect_actions(agent: PPO, environment: Environment):
    """
    Collects actions from the agent for the given environment
    :param agent: the trained agent
    :param environment: the environment
    :return: a 2D-array of actions
    """

    actions_collected = []
    amounts_collected = []
    obs = environment.reset()[0]
    done = False
    truncated = False
    while not done and not truncated:
        action, _ = agent.predict(obs, deterministic=True)
        actions_collected.append(action)
        buy_amounts, _, index_buy = environment.buy_amounts(action)
        sell_amounts, _, index_sell = environment.sell_amounts(action)
        amounts = np.zeros(len(action))
        amounts[index_buy] = buy_amounts
        amounts[index_sell] = -sell_amounts
        amounts_collected.append(amounts)
        obs, _, done, truncated, _ = environment.step(action)
    return np.array(actions_collected), np.array(amounts_collected)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Backtest a trained agent')
    parser.add_argument('--data', type=str, default='../stock_data/portfolio_custom140_datetime',
                        help='Path to the data file. If it is a directory, all files in it will be used (only for OHCLV data)')
    parser.add_argument('--agent', type=str, default='../trained_rl/FINAL_ppo140_renv_4e6.pt',
                        help='Path to the trained agent')
    parser.add_argument('--encoder', type=str, default='../trained_ae/transformer_ae_800.pt',
                        help='Path to the trained encoder')
    parser.add_argument('--cash_init', type=float, default=1e4,
                        help='Initial cash amount')
    parser.add_argument('--commission', type=float, default=0.01,
                        help='Initial cash amount')
    parser.add_argument('--observation_length', type=int, default=16,
                        help='Length of the observation')
    parser.add_argument('--train_test_split', type=float, default=0.8,
                        help='train test split')
    parser.add_argument('--data_env', type=str, default='../stock_data/portfolio_custom140_2008_2022.csv',
                        help='Path to the data for the environment')
    args = parser.parse_args()

    # set start and end date in format '%Y-%m-%d %H:%M:%S'
    yf_start = pd.Timestamp('2008-01-01')
    yf_end = pd.Timestamp('2022-12-31')

    # load encoder, environment and trained agent
    print('Loading encoder, environment and trained agent...')
    if args.encoder is not None and args.encoder != '':
        state_dict = torch.load(args.encoder, map_location=torch.device('cpu'))
        encoder = TransformerAutoencoder(**state_dict['model'], seq_len=state_dict['general']['seq_len'])
        encoder.load_state_dict(state_dict['model']['state_dict'])
        encoder.eval()
    else:
        encoder = None
    env_data = pd.read_csv(args.data_env, index_col=0, header=0)
    columns = env_data.columns
    env_data = env_data.to_numpy()
    index_start = int(args.train_test_split * len(env_data))
    env_data = env_data[index_start:]
    env = Environment(env_data, args.cash_init, args.observation_length, encoder=encoder,
                      commission_buy=args.commission, commission_sell=args.commission)
    agent = PPO.load(args.agent, env)

    # collect actions from the agent
    print('Collecting actions from the agent...')
    actions, amounts = collect_actions(agent, env)

    # load data and setup backtesting environment for each file in the directory
    print('Loading data and setting up backtesting environment...')
    final_diff = []
    equity = 0
    if os.path.isdir(args.data):
        # for i, file in enumerate(os.listdir(args.data)):
        for i, file in enumerate(columns):
            # print(f'Adding {columns[i]} to backtesting environment...')

            # load csv with pandas datetime index
            df = pd.read_csv(os.path.join(args.data, file+'.csv'), index_col=0, header=0)[index_start:]
            df.index = pd.to_datetime(df.index)

            class MyStrategy(Strategy):
                t = 0
                observation_length = args.observation_length
                amounts = amounts[:, i]

                def init(self):
                    # print('\n----------------------------------------------------')
                    print(f'Running backtest on stock {columns[i]}')
                    # print('----------------------------------------------------\n')
                    # self.amounts = np.concatenate((np.zeros(self.observation_length-1), self.amounts))

                def next(self):
                    if  self.observation_length - 1 <= self.t < len(self.amounts):
                        if self.amounts[self.t] > 0:
                            self.buy(size=self.amounts[self.t])
                        elif self.amounts[self.t] < 0:
                            self.sell(size=-self.amounts[self.t])
                        # print(f'Time: {self.t+1} / {len(data)} - '
                        #       f'Action: {self.amounts[self.t]} - '
                        #       f'Equity: {self.equity} - ')

                    self.t += 1

            cash = 1e9
            bt = Backtest(df, MyStrategy, cash=cash, commission=0.001)

            results = bt.run()
            final_diff.append(results['Equity Final [$]'] - cash)

    # sum of all final differences
    print('Sum of all final differences: ', sum(final_diff))
