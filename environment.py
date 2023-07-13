# Description: Environment for the stock trading agent. Inherits from gym.Env.
from copy import deepcopy
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
from matplotlib import pyplot as plt

from stable_baselines3.common.env_checker import check_env

from nn_architecture.ae_networks import Autoencoder


class Environment(gym.Env):
    def __init__(self,
                 stock_data,
                 cash: int,
                 observation_length: int,
                 commission_buy=0.01,
                 commission_sell=0.005,
                 reward_scaling=1e-4,
                 random_splits=True,
                 time_limit=-1,
                 discrete_actions=True,
                 recurrent=False,
                 # optional parameters
                 encoder: Optional[Autoencoder] = None,
                 ):

        # set encoder first, since it defines the observation space
        self.encoder = encoder

        # set action and observation space
        # shape of action space: ([sell (=0) or buy (=1)]*[stock1, stock2, ...])
        if discrete_actions:
            self.action_space = gym.spaces.MultiBinary(stock_data.shape[-1])
        else:
            self.action_space = gym.spaces.Box(low=-1, high=1, shape=(stock_data.shape[-1],), dtype=np.float32)

        # shape of observation space: (cash, portfolio, stock_prices)
        if recurrent:
            self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(observation_length + 2, stock_data.shape[-1]), dtype=np.float32)
        else:
            if self.encoder is not None:
                obs_shape = (1, 1+stock_data.shape[-1]+self.encoder.output_dim)
            else:
                obs_shape = (1, 1+stock_data.shape[-1]+stock_data.shape[-1]*observation_length)
            self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32)

        # set data
        self._dataset = stock_data  # whole training dataset
        self.stock_data = self._dataset[:time_limit]  # used as a split of the whole dataset for one episode
        self._random_splits = random_splits

        # internal parameters
        self.t = observation_length
        self.cash = cash
        self.portfolio = np.zeros((stock_data.shape[1],))
        self.observation_length = observation_length

        self._time_limit = time_limit
        self._commission_sell = commission_sell
        self._commission_buy = commission_buy
        self._reward_scaling = reward_scaling
        self._portfolio_t_1 = np.zeros((1, stock_data.shape[1]))
        self._cash_t_1 = cash
        self._cash_init = cash
        self._discrete_actions = discrete_actions
        self._termination_reward = -100
        self._cash_threshold = 1e-2
        self._recurrent = recurrent

        # mapping from binary action space ([0,1]) to real action space ([-1,1])
        # easier to switch between discrete and continuous action space
        self._transform_binary_action = {
            0: -1,
            1: 1,
        }

    def step(self, action: np.ndarray):
        if self._discrete_actions:
            action = np.array([self._transform_binary_action[a] for a in action])

        self.t += 1

        self._cash_t_1 = self.total_equity()
        self._portfolio_t_1 = deepcopy(self.portfolio)

        if self.cash > 0:
            self._buy(action)
            self._sell(action)

        if self.cash < self._cash_threshold:
            self.cash = 0.0

        return self._get_obs(), self.reward(reward_scaling=True), self._terminated(), self._truncated(), {}

    def _buy(self, action):
        index_buy = (action > 0) * (self.stock_data[self.t] > 0)
        buy_orders = action[index_buy]

        # preprocessing of buy orders
        # check if buy orders <= cash (sum of buy orders surpasses 1) and compute softmax if necessary
        if np.sum(buy_orders) > 1:
            # rescale buy_orders so they sum up to one if not enough cash
            buy_orders /= np.sum(buy_orders)
        buy_amounts = buy_orders * self.cash
        buy_amounts -= buy_amounts*self._commission_buy
        # reserve some cash for later sell orders
        buy_amounts -= self._commission_sell * buy_amounts*2
        buy_amounts = np.clip(buy_amounts, 0, None)
        buy_amounts[buy_amounts < self._cash_threshold] = 0

        # compute buy amounts in shares as integers
        buy_amounts_share = buy_amounts / self.stock_data[self.t, index_buy]  # Unit check: [$ / ($/share) = share]
        buy_amounts_share = buy_amounts_share.astype(int)

        # compute buy amounts in dollars for share amounts as integers
        buy_amounts = buy_amounts_share * self.stock_data[self.t, index_buy]  # Unit check: [share * ($/share) = $]

        # check if portfolio would change insanely
        # if np.sum(buy_amounts_share) > 1e4:
        #     print("Portfolio would change insanely!")
        # update cash and portfolio after buy orders
        if self.cash >= np.sum(buy_amounts) + np.sum(buy_amounts*self._commission_buy):
            self.cash -= np.sum(buy_amounts) - np.sum(buy_amounts*self._commission_buy)
            self.portfolio[index_buy] += buy_amounts_share  # Unit check: [$ / ($/share) = share]

    def _sell(self, action):
        index_sell = (action < 0) * (self.stock_data[self.t] > 0)
        sell_orders = -action[index_sell]  # from negative sign (as actions are defined) to positive sign

        # get share amounts as integers and compute sell orders in first iteration
        sell_amounts_share = self.portfolio[index_sell] * sell_orders  # Unit check: [share * unit_less_fraction = share]
        sell_amounts_share = sell_amounts_share.astype(int)
        sell_amounts = sell_amounts_share * self.stock_data[self.t, index_sell]  # Unit check: [share * ($/share) = $]
        if np.sum(sell_amounts * self._commission_sell) > self.cash:
            # reduce sell_amounts if not enough cash for all sell orders
            sell_amounts *= self.cash / np.sum((sell_amounts + 1) * self._commission_sell)
            sell_amounts = np.clip(sell_amounts, 0, None)

            # compute sell amounts in shares as integers in second interation
            sell_amounts_share = sell_amounts / self.stock_data[self.t, index_sell]  # Unit check: [$ / ($/share) = share]
            sell_amounts_share = sell_amounts_share.astype(int)

            # compute sell amounts in dollars for share amounts as integers in second iteration
            sell_amounts = sell_amounts_share * self.stock_data[self.t, index_sell]  # Unit check: [share * ($/share) = $]

        # update cash and portfolio if enough cash for all sell order commissions
        if self.cash >= np.sum(sell_amounts * self._commission_sell):
            # Update cash: sum of all sells - percentual commission for each sell
            self.cash += np.sum(sell_amounts) - np.sum(sell_amounts * self._commission_sell)
            # update portfolio after sell orders
            self.portfolio[index_sell] -= sell_amounts_share  # Unit check: [$ / ($/share) = share]
        else:
            print('not enough cash to sell anymore')

    def _terminated(self):
        total_equity_low = self.total_equity().item() <= 0.001*self._cash_init
        # cash_low = self.cash <= 0.0001*self._cash_init
        # return bool(total_equity_low or cash_low)
        return bool(total_equity_low)

    def _truncated(self):
        return self.t == len(self.stock_data) - 1

    def _get_obs(self):
        cash = deepcopy(np.array([self.cash/self._cash_init]))
        portfolio = deepcopy(self.portfolio/100)
        stock_prices = deepcopy(self.stock_data[self.t - self.observation_length:self.t])
        stock_prices -= self.stock_data[self.t - self.observation_length]
        stock_prices /= np.max(np.abs(stock_prices))
        stock_prices[np.isnan(stock_prices)] = 0
        if not self._recurrent:
            stock_prices = np.reshape(stock_prices, (-1,))
            obs = np.concatenate((cash, portfolio, stock_prices), dtype=np.float32).reshape(1, -1)
        else:
            # obs_space = np.concatenate((portfolio, stock_prices), axis=1)
            cash = np.tile(cash, (1, stock_prices.shape[-1]))
            # portfolio = np.tile(portfolio, (self.observation_length, 1))
            obs = np.concatenate((cash, portfolio.reshape(1, -1), stock_prices), axis=0, dtype=np.float32)

        if self.encoder is not None:
            obs_stocks = obs[:, 1 + self.stock_data.shape[-1]:].reshape(1, self.observation_length, self.stock_data.shape[-1])
            obs_stocks = self.encode(torch.tensor(obs_stocks, dtype=torch.float32)).detach().numpy()
            obs = np.concatenate((obs[:, :1 + self.stock_data.shape[-1]], obs_stocks), axis=1)

        return obs

    def reward(self, reward_scaling=False):
        # Calculate reward for the current action
        r = self.total_equity() - self._cash_t_1
        if reward_scaling:
            r *= self._reward_scaling
        return r

    def reset(self, **kwargs):
        self.cash = self._cash_init
        self.t = self.observation_length
        self.portfolio = np.zeros((self.stock_data.shape[1],))
        # if random_split get random junk of total stock data and set as episode's stock data
        if self._time_limit > 0:
            if self._time_limit < self.observation_length:
                raise ValueError('Episode time limit has to be larger than observation length')
            else:
                start = np.random.randint(0, len(self._dataset) - self._time_limit)
                end = start + self._time_limit
            self.stock_data = deepcopy(self._dataset[start:end])
        else:
            self.stock_data = deepcopy(self._dataset)
        return self._get_obs(), {}

    def render(self):
        # Render the environment to the screen
        pass

    def close(self):
        # Close the environment
        pass

    def total_equity(self):
        # Calculate total equity
        # Total equity is defined as the sum of cash and the value of all shares
        return self.cash + np.sum(self.portfolio * self.stock_data[self.t])

    def encode(self, x):
        return self.encoder.encode(x)
