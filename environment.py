# Description: Environment for the stock trading agent. Inherits from gym.Env.
from copy import deepcopy
from typing import Optional

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch

from nn_architecture.ae_networks import Autoencoder


class Environment(gym.Env):
    def __init__(self,
                 stock_data,
                 portfolio_names,
                 cash: int,
                 observation_length: int,
                 commission_buy=0.01,
                 commission_sell=0.005,
                 reward_scaling=1e-4,
                 threshold_terminated=0.1,
                 random_splits=True,
                 time_limit=-1,
                 discrete_actions=False,
                 recurrent=False,
                 # optional parameters
                 encoder: Optional[Autoencoder] = None,
                 test = False,
                 ):
        
        # set encoder first, since it defines the observation space
        self.encoder = encoder

        # set action and observation space
        # shape of action space: ([sell (=0) or buy (=1)]*[stock1, stock2, ...])
        action_dimension = stock_data.shape[-1] if encoder is None else encoder.output_dim
        if discrete_actions:
            self.action_space = gym.spaces.MultiBinary(action_dimension)
        else:
            self.action_space = gym.spaces.Box(low=-1, high=1, shape=(action_dimension,), dtype=np.float32)

        # shape of observation space: (cash, portfolio, stock_prices)
        if recurrent:
            self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(encoder.output_dim_2, encoder.output_dim), dtype=np.float32)
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

        self._time_limit = time_limit #if time_limit > 0 else len(stock_data)-1
        self._commission_sell = commission_sell
        self._commission_buy = commission_buy
        self._reward_scaling = reward_scaling
        self._portfolio_t_1 = np.zeros((1, stock_data.shape[1]))
        self._cash_t_1 = cash
        self._cash_init = cash
        self._discrete_actions = discrete_actions
        self._termination_reward = -100
        self._cash_threshold = 1e-2
        self._threshold_terminated = threshold_terminated
        self._recurrent = recurrent
        self._test = test
        self._total_equity_episode = []
        self._actions_taken_episode = []
        self._portfolio_episode = []
        self._cash_episode = []
        self._portfolio_average_episode = np.mean(self.stock_data[observation_length - 1:], axis=1) / np.mean(self.stock_data[observation_length - 1]) * self._cash_init
        self._portfolio_names = portfolio_names
        self._forex_tag = '=X'
        self._crypto_tag = '-USD'

        # mapping from binary action space ([0,1]) to real action space ([-1,1])
        # easier to switch between discrete and continuous action space
        self._transform_binary_action = {
            0: -1,
            1: 1,
        }

    def step(self, action: np.ndarray):

        if self._discrete_actions:
            action = np.array([self._transform_binary_action[a] for a in action])
        
        if self.encoder is not None:
            # train agent to act on encoded space 
            # --> Since AE discovers underlying, simplified structure (i.e. dependencies between stocks)
            # --> Agent should also be able to act on this simplified structure
            # decode to original space to place orders
            # for decoding:
            # Original decoder space is [0, 1]
            # Target space is [-1, 1]
            with torch.no_grad():
                action = np.tanh(self.encoder.decode(torch.tensor(action.reshape(1, -1), dtype=torch.float32)).numpy()).reshape(-1)

        self._cash_t_1 = self.total_equity()
        self._portfolio_t_1 = deepcopy(self.portfolio)

        if self.cash > 0:
            self._buy(action)
            if self._test:
                # collect cash after buy orders
                cash_after_buy = deepcopy(self.cash)
            self._sell(action)
            if self._test:
                # collect cash after sell orders
                cash_after_sell = deepcopy(self.cash)
        if self._test and len(self._cash_episode) < 100:
            print(f'Cash after buy: {cash_after_buy}, Cash after sell: {cash_after_sell}, Total Equity: {self.total_equity()}')

        if self.cash < self._cash_threshold:
            self.cash = 0.0

        self.t += 1

        terminated = self._terminated()
        truncated = self._truncated()
        reward = self.reward(reward_scaling=True) if not terminated else -1000

        if self._test:
            # collect total equity for plot
            self._total_equity_episode.append(self.total_equity())
            self._actions_taken_episode.append(action)
            self._cash_episode.append(self.cash)
            self._portfolio_episode.append(self.portfolio)

        if (terminated or truncated) and self._test:
            print(f'Test done. Total Equity: {self.total_equity()}')
            self.plot_results()
            print(f'Portfolio: {self.portfolio}')

            # plot actions taken


        return self._get_obs(), reward, terminated, truncated, {}

    def buy_amounts(self, action):
        index_buy = (action > 0) * (self.stock_data[self.t] > 0)
        buy_orders = action[index_buy]

        # preprocessing of buy orders
        # check if buy orders <= cash (sum of buy orders surpasses 1) and rescale if necessary
        if np.sum(buy_orders) > 1:
            buy_orders /= np.sum(buy_orders)
        buy_amounts = buy_orders * self.cash
        # remove buy commission from cash-to-invest
        buy_amounts -= buy_amounts * self._commission_buy
        # reserve some cash for later sell orders
        # buy_amounts -= self._commission_sell * buy_amounts * 2
        # make sure that buy orders are positive
        buy_amounts = np.clip(buy_amounts, 0, None)
        # make sure that buy orders are above cash threshold - otherwise set to zero
        buy_amounts[buy_amounts < self._cash_threshold] = 0

        # compute buy amounts in shares
        index_buy_num = np.where(index_buy)[0]
        buy_amounts_share = buy_amounts / self.stock_data[self.t, index_buy]  # Unit check: [$ / ($/share) = share]

        # compute buy amounts in shares as integers - except forex or crypto
        index_int = [not (self._forex_tag in name or self._crypto_tag in name) for name in self._portfolio_names] * index_buy
        index_int_num = np.where(index_int)[0]
        index_int = np.zeros_like(index_buy_num, dtype=bool)
        for i, n in enumerate(index_int_num):
            index_int[i] = True if n in index_buy_num else False

        buy_amounts_share[index_int] = buy_amounts_share[index_int].astype(int)
        # compute buy amounts in dollars for share amounts as integers in second iteration
        buy_amounts = buy_amounts_share * self.stock_data[self.t, index_buy]  # Unit check: [share * ($/share) = $]

        return buy_amounts_share, buy_amounts, index_buy

    def sell_amounts(self, action):
        # index_sell: sell action * stock is already in stock market * stock is held in portfolio
        index_sell = (action < 0) * (self.stock_data[self.t] > 0) * (self.portfolio > 0)
        index_sell_num = np.where(index_sell)[0]

        # get share amounts as integers and compute sell orders in first iteration
        sell_amounts_share = self.portfolio[index_sell] * -action[index_sell]  # Unit check: [share * unit_less_fraction = share]

        # compute buy amounts in shares as integers - except forex or crypto
        index_int = [not (self._forex_tag in name or self._crypto_tag in name) for name in self._portfolio_names] * index_sell
        index_int_num = np.where(index_int)[0]

        index_int = np.zeros_like(index_sell_num, dtype=bool)
        for i, n in enumerate(index_int_num):
            index_int[i] = True if n in index_sell_num else False

        # set sell amounts of stocks to integers -> no fractions of stocks
        sell_amounts_share[index_int] = sell_amounts_share[index_int].astype(int)
        # compute buy amounts in dollars for share amounts as integers in second iteration
        sell_amounts = sell_amounts_share * self.stock_data[self.t, index_sell]  # Unit check: [share * ($/share) = $]

        if np.sum(sell_amounts * self._commission_sell) > self.cash:
            # reduce sell_amounts if not enough cash for all sell orders
            sell_amounts *= self.cash / np.sum((sell_amounts + 1) * self._commission_sell)

            # compute sell amounts in shares as integers in second interation
            sell_amounts_share = sell_amounts / self.stock_data[self.t, index_sell]  # Unit check: [$ / ($/share) = share]

            # compute buy amounts in shares as integers - except forex or crypto
            index_int = [not (self._forex_tag in name or self._crypto_tag in name) for name in self._portfolio_names] * index_sell
            index_int_num = np.where(index_int)[0]
            index_int = np.zeros_like(index_sell_num, dtype=bool)
            for i, n in enumerate(index_int_num):
                index_int[i] = True if n in index_sell_num else False

            # set sell amounts of stocks to integers -> no fractions of stocks
            sell_amounts_share[index_int] = sell_amounts_share[index_int].astype(int)
            # compute buy amounts in dollars for share amounts as integers in second iteration
            sell_amounts = sell_amounts_share * self.stock_data[self.t, index_sell]  # Unit check: [share * ($/share) = $]

        return sell_amounts_share, sell_amounts, index_sell

    def _buy(self, action):

        buy_amounts_share, buy_amounts, index_buy = self.buy_amounts(action)

        # check if portfolio would change insanely
        # if np.sum(buy_amounts_share) > 1e4:
        #     print("Portfolio would change insanely!")
        # update cash and portfolio after buy orders
        if self.cash >= np.sum(buy_amounts) + np.sum(buy_amounts*self._commission_buy):
            self.cash -= np.sum(buy_amounts) - np.sum(buy_amounts*self._commission_buy)
            self.portfolio[index_buy] += buy_amounts_share  # Unit check: [$ / ($/share) = share]

    def _sell(self, action):

        sell_amounts_share, sell_amounts, index_sell = self.sell_amounts(action)

        # update cash and portfolio if enough cash for all sell order commissions
        if self.cash >= np.sum(sell_amounts * self._commission_sell):
            # Update cash: sum of all sells - percentual commission for each sell
            self.cash += np.sum(sell_amounts) - np.sum(sell_amounts * self._commission_sell)
            # update portfolio after sell orders
            self.portfolio[index_sell] -= sell_amounts_share  # Unit check: [$ / ($/share) = share]
        else:
            print('not enough cash to sell anymore')

    def _terminated(self):
        total_equity_low = self.total_equity().item() <= self._threshold_terminated*self._cash_init
        # cash_low = self.cash <= 0.0001*self._cash_init
        # return bool(total_equity_low or cash_low)
        return bool(total_equity_low)

    def _truncated(self):
        return self.t == len(self.stock_data) - 1

    def _get_obs(self):
        cash = np.array([self.cash/self._cash_init])
        # portfolio = self.portfolio/np.max(self.portfolio) if np.max(self.portfolio) != 0 else deepcopy(self.portfolio)  # old: /100
        # get value of each portfolio element instead of amount of shares
        portfolio_value = np.zeros_like(self.portfolio)
        for i in range(len(self.portfolio)):
            if self.portfolio[i] == 0 or self.stock_data[self.t, i] != 0:
                portfolio_value[i] = 0
            else:
                portfolio_value[i] = portfolio_value[i]*self.stock_data[self.t, i]
        portfolio = portfolio_value/np.max(portfolio_value) if np.max(portfolio_value) != 0 else deepcopy(portfolio_value)

        stock_prices = deepcopy(self.stock_data[self.t-self.observation_length+1:self.t+1])
        stock_prices -= stock_prices[0]
        stock_prices /= np.max(np.abs(stock_prices))
        stock_prices[np.isnan(stock_prices)] = 0

        if self.encoder is not None:
            with torch.no_grad():
                stock_prices = self.encode(torch.tensor(stock_prices.reshape([1] + list(stock_prices.shape)), dtype=torch.float32)).numpy()
                stock_prices = np.reshape(stock_prices, (*stock_prices.shape[1:],))

        if not self._recurrent:
            stock_prices = np.reshape(stock_prices, (-1,))
            obs = np.concatenate((cash, portfolio, stock_prices), dtype=np.float32).reshape(1, -1)
        else:
            # obs_space = np.concatenate((portfolio, stock_prices), axis=1)
            # cash = np.tile(cash, (1, stock_prices.shape[-1]))
            # portfolio = np.tile(portfolio, (self.observation_length, 1))
            # obs = np.concatenate((cash, portfolio.reshape(1, -1), stock_prices), axis=0, dtype=np.float32)
            obs = stock_prices

        return obs

    def reward(self, reward_scaling=False):
        # Calculate reward for the current action
        r = self.total_equity() - self._cash_t_1
        if reward_scaling:
            r *= self._reward_scaling
        return r

    def reset(self, **kwargs):
        self.cash = self._cash_init
        self.t = self.observation_length - 1
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

        self._total_equity_episode = []
        self._actions_taken_episode = []
        self._portfolio_episode = []
        self._cash_episode = []
        self._portfolio_average_episode = np.mean(self.stock_data[self.observation_length - 1:], axis=1) / np.mean(self.stock_data[self.observation_length - 1]) * self._cash_init

        return self._get_obs(), {}

    def render(self):
        # Render the environment to the screen
        pass

    def plot_results(self):
        fig, axs = plt.subplots(4, 1, sharex=True)

        # plot the average of all stock prices
        axs[0].plot(self._portfolio_average_episode, '--', label='Portfolio average')
        axs[0].plot(self._total_equity_episode, label='Total Equity')
        axs[0].set_ylabel('rel. price')
        axs[0].grid()

        actions_mean = np.mean(self._actions_taken_episode, axis=1)
        actions_std = np.std(self._actions_taken_episode, axis=1)
        axs[1].plot(actions_mean, label='actions')
        axs[1].fill_between(np.arange(len(self._actions_taken_episode)), actions_mean - actions_std, actions_mean + actions_std, alpha=0.2)
        axs[1].set_ylabel('actions')
        axs[1].grid()

        portfolio_mean = np.mean(self._portfolio_episode, axis=1)
        portfolio_std = np.std(self._portfolio_episode, axis=1)
        axs[2].plot(portfolio_mean, label='portfolio')
        axs[2].fill_between(np.arange(len(portfolio_mean)), portfolio_mean - portfolio_std, portfolio_mean + portfolio_std, alpha=0.2)
        axs[2].set_ylabel('portfolio')
        axs[2].grid()

        axs[3].plot(self._cash_episode, label='cash')
        axs[3].set_ylabel('cash')
        axs[3].set_xlabel('time steps (days)')
        axs[3].set_xticks(np.arange(0, len(self._cash_episode), len(self._cash_episode) // 30))
        # set orientation of x labels
        for tick in axs[3].get_xticklabels():
            tick.set_rotation(90)
        # set x labels to every 5th tick
        axs[3].grid()

        fig.suptitle('Episode results; Total Equity: {:.2f}'.format(self.total_equity()))
        plt.show()

    def close(self):
        # Close the environment
        pass

    def total_equity(self):
        # Calculate total equity
        # Total equity is defined as the sum of cash and the value of all shares
        return self.cash + np.sum(self.portfolio * self.stock_data[self.t])

    def encode(self, x):
        return self.encoder.encode(x)
