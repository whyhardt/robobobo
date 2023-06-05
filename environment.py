# This file includes the environment class
# The environment class is used to simulate the stock market
# Inputs:
# - Array where each element corresponds to a stock
# - Each element contains information about the action for the designated stock
# - The actions can be either: buy n values (int > 0), sell n values (int < 0), or hold action (= 0)
import copy

import numpy as np
import torch
from matplotlib import pyplot as plt
from backtesting import Backtest, Strategy


class ObservationSpace:
    def __init__(self, cash: int, num_stocks: int, observation_length: int):
        # What do I need in the observation space?
        # - current cash --> float; stop buy-orders if surpass remaining cash
        # - array with the last n values of the portfolio --> tensor(float); measure agent's performance
        # - one hot encoded array of the stocks I own with the amount of stocks I own --> tensor(int); have an overview over portfolio and to stop sell-orders if I don't own enough stocks
        # - 2D-array with last m stock prices --> tensor(float); base for agent's calculations on the stock market
        self.cash = torch.zeros((1, 1))  # + cash
        # self.total_equity_diff = torch.zeros(1, 1)
        self.portfolio = torch.zeros((1, num_stocks))  # Maybe switch to portfolio value instead of number of stocks
        self.stock_prices = torch.zeros((observation_length, num_stocks))

        # internal parameters
        self._init_cash = cash
        self._observation_length = observation_length
        self.num_stocks = num_stocks

        # auxiliary parameters
        self.shape = (2, 1)
        # observation space dimension is defined as each single feature dimension summed up
        # 1 for cash, num_stocks*observation_length for portfolio, num_stocks*observation_length for stock_prices
        self.dim = num_stocks# + num_stocks*observation_length + 1

    def __call__(self, normalized=True, dtype=tuple):
        """Makes the class instance callable.
        Returns a tuple of observable variables.
        If normalized=True: Returned variables get normalized"""
        # if normalized:
        #     os = [self.cash/self._init_cash if self._init_cash != 0 else self.cash,
        #           self.portfolio/torch.sum(self.portfolio) if torch.sum(self.portfolio) != 0 else self.portfolio,
        #           self.stock_prices/torch.max(self.stock_prices) if torch.max(self.stock_prices) != 0 else self.stock_prices]
        # else:
        #     os = [self.cash, self.portfolio, self.stock_prices]

        if normalized:
            os = [
                copy.deepcopy((self.cash  / self._init_cash)),
                # copy.deepcopy(self.portfolio / torch.max(torch.abs(self.portfolio)) if torch.max(
                #     torch.abs(self.portfolio)) != 0 else self.portfolio),
                # copy.deepcopy(self.stock_prices / torch.max(self.stock_prices) if torch.max(
                #     self.stock_prices) != 0 else self.stock_prices)
            ]
        else:
            # os = [copy.deepcopy(self.portfolio), copy.deepcopy(self.stock_prices)]
            os = [copy.deepcopy(self.cash)]

        if dtype == tuple:
            return tuple(os)
        elif dtype == torch.Tensor:
            for i, e in enumerate(os):
                os[i] = e.reshape(1, -1)
            return torch.cat(os, dim=1)

    def set(self,
            cash=None,
            portfolio=None,
            stock_prices=None):
        """Set observation space according to input.
        If input=None: use current observation space variable
        Observation space variables are:
        portfolio: Gives difference between current and previous portfolio
        stock_prices: are the last n stock prices (n: observation_length)"""
        # self.portfolio = [portfolio if portfolio is not None else self.portfolio][0]
        # self.portfolio = [self.portfolio - portfolio if portfolio is not None else torch.zeros_like(self.portfolio)][0]
        self.cash = cash if cash is not None else self.cash
        self.portfolio = portfolio if portfolio is not None else self.portfolio
        self.stock_prices = stock_prices if stock_prices is not None else self.stock_prices


class ActionSpace(object):
    """ActionSpace for cognitive environment.
    Can be discrete or continuous.
    If input parameter steps in __init__-method is None: Continuous action space.
    Otherwise: Discrete action space"""

    def __init__(self, action_low: float, action_high: float, action_dim: int):
        self.low = action_low
        self.high = action_high

        # internal parameters
        self._action_space = np.array(self.low).reshape(1, 1)
        self._action_dim = action_dim

        self._set_action_space()

        self.shape = self._action_space.shape

    def __call__(self):
        # Continuous action space
        return {"Lower bound": self.low, "Upper bound": self.high}

    # def _set_low(self, low):
    #     """lower boundary of action space"""
    #     self.low = low
    #     self._set_action_space()
    #
    # def _set_high(self, high):
    #     """upper boundary of action space"""
    #     self.high = high
    #     self._set_action_space()

    def _set_action_space(self):
        """Set action space for continuous actions in 1D action space"""
        # Continuous action space in a 1D array
        self._action_space = np.zeros(self._action_dim) + self.low

    def shape(self):
        """Return action space"""
        return self._action_space.shape

    def sample(self, hold_threshold=0):
        """Sample random action"""
        # Continuous action space

        # sample random action from action space but make sure that positive entries are not larger than 1
        action = torch.tensor(np.random.uniform(self.low, self.high, self._action_dim))
        action[action.abs() < hold_threshold] = 0.
        if action[action > 0].sum() > 1:
            action_pos = np.where(action > 0)[0]
            action[action_pos] = self.softmax(action[action_pos])
        return action

    @staticmethod
    def softmax(x):
        """Compute softmax values for each sets of scores in x."""
        return x.exp() / torch.sum(x.exp(), dim=0)


class Environment:
    def __init__(self, stock_data, cash: int, observation_length: int, commission=0, random_splits=False):
        # Initialize action and observation space
        # An action from the action space is defined by:
        # - [-1 <= a < 0]: Sell the relative amount of shares given by the absolute value of a
        # - [a == 0]: Hold the current amount of shares
        # - [0 < a <= 1]: Buy shares of the relative value of the total equity
        #       Two scenarios:
        #       1. all buy orders < cash + all sell orders --> Sell before Buy --> Check for feasibility
        #       2. all buy orders < cash --> No need to wait for sell --> Feasible for sure

        # super().__init__()

        # set action and observation space
        self.action_space = ActionSpace(-1, 1, stock_data.shape[1])
        self.observation_space = ObservationSpace(cash, stock_data.shape[1], observation_length)
        self.observation_space.set(portfolio=torch.zeros(1, stock_data.shape[1]), stock_prices=stock_data[:observation_length])

        # set stock data
        self.dataset = stock_data  # whole training dataset
        self.stock_data = self.dataset  # used as a split of the whole dataset for one episode
        self.random_splits = random_splits

        # internal parameters
        self.t = observation_length + 1
        self.cash = cash
        self.portfolio = torch.zeros(1, stock_data.shape[1])
        self._cash_t_1 = cash
        self._cash_init = cash
        self._portfolio_t_1 = torch.zeros(1, stock_data.shape[1])
        self.observation_length = observation_length
        self.commission = commission

    def step(self, action):
        # Execute one time step within the environment
        # Interact with backtesting environment by forwarding sell, buy and hold actions from action array
        # Return: observation, reward, done, info

        self._cash_t_1 = self.total_equity()
        # self._portfolio_t_1 = portfolio.detach().clone()
        current_prices = self.stock_data[self.t]

        # separate sell and buy orders - each stock is either sell, hold or buy
        isell = (action < 0)
        ibuy = (action > 0)*(current_prices > 0)
        sell_orders = action[isell]  # torch.tensor([a if a < 0 else 0 for a in action])
        buy_orders = action[ibuy]  # torch.tensor([a if a > 0 else 0 for a in action])

        # process buy orders
        buy_amounts = buy_orders*self.cash
        if torch.sum(buy_amounts) < self.cash:
            # Update cash: sum of all buys + percentual commission for each buy
            self.cash -= torch.sum(buy_amounts) + torch.sum(buy_amounts*self.commission)
            # update portfolio after buy orders
            self.portfolio[0, ibuy] += buy_amounts / current_prices[ibuy]  # Unit check: [$ / ($/share) = share]

        # process sell orders
        portfolio_sell = self.portfolio[0, isell] * sell_orders
        sell_amount = -portfolio_sell * current_prices[isell]  # Unit check: [share * ($/share) = $]
        # Update cash: sum of all sells - percentual commission for each sell
        self.cash += torch.sum(sell_amount) - torch.sum(sell_amount*self.commission)
        # update portfolio after sell orders
        self.portfolio[0, isell] += portfolio_sell

        # update environment and observation space
        self.t += 1
        stock_prices = self.stock_data[self.t-self.observation_length:self.t-1]
        self.observation_space.set(cash=self.reward().reshape(1, 1).float(), portfolio=self.portfolio, stock_prices=stock_prices)

        # check if episode is done
        # episode is done if time limit is reached or total equity is 0 (no cash and no shares)
        if self.t == len(self.stock_data)-1 or (self.total_equity().item() <= 0.001*self._cash_init):
            done = True
        else:
            done = False

        return self.observation_space(normalized=True), self.reward(), done, {}

    def reward(self):
        # Calculate reward for the current action
        return self.total_equity() - self._cash_t_1

    def hard_reset(self, cash=None, random_split=False, split_length=None):
        if cash is None:
            # cash = self._cash_init
            self.cash = self._cash_init
        # Reset environment to initial state
        self.t = self.observation_length + 1
        self.portfolio *= 0
        self.observation_space = ObservationSpace(self.cash, self.stock_data.shape[1], self.observation_length)
        # if random_split get random junk of total stock data and set as episode's stock data
        if random_split:
            # start = 0
            # end = len(self.stock_data_train)-1
            if split_length < self.observation_length:
                raise ValueError('Split length has to be larger than observation length')
            if split_length is None:
                start = np.random.randint(0, len(self.dataset)-self.observation_length-1)
                end = len(self.dataset)-1
            else:
                start = np.random.randint(0, len(self.dataset)-split_length)
                end = start + split_length
            self.stock_data = self.dataset[start:end]

    def total_equity(self):
        # Calculate total equity
        # Total equity is defined as the sum of cash and the value of all shares
        return self.cash + torch.sum(self.portfolio * self.stock_data[self.t])

    def set_observation_space(self, cash=None, portfolio=None, stock_prices=None, time=None):
        """Set observation space according to input.
        If input=None: use current observation space variable"""
        self.observation_space.set(cash=cash, portfolio=portfolio, stock_prices=stock_prices)
        self.t = min((self.observation_length + 1, time if time is not None else self.t))
