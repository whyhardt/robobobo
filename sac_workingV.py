import copy
import os
import random
from datetime import datetime
from typing import Optional, List, Iterable

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy import signal
from torch.distributions import Normal

from environment import Environment
from utils.get_filter import moving_average


class ReplayBuffer:
    """Class of the replay buffer for experience replay"""

    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        """push new experience into buffer"""
        experience = (state.detach().clone(), action, reward, next_state.detach().clone(), done)  # .detach().clone()
        self.buffer.append(experience)

    def sample(self, batch_size):
        """get randomly drawn samples from buffer. batch_size determines number of samples"""
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        batch = random.sample(self.buffer, batch_size)

        for experience in batch:
            # Decode drawn experiences into necessary format
            state, action, reward, next_state, done = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)

        return torch.concat(state_batch).float(), \
               torch.concat(action_batch).float(), \
               torch.concat(reward_batch).float(), \
               torch.concat(next_state_batch).float(), \
               torch.concat(done_batch).float()

    def __len__(self):
        return len(self.buffer)


class ValueNetwork(nn.Module):
    """Class of value network that resembles the critic"""

    def __init__(self, state_dim, hidden_dim, num_layers=6, dropout=0.3, init_w=3e-3):
        """Determines network architecture at object initialization"""
        super(ValueNetwork, self).__init__()

        self.net = nn.Sequential()
        lin = nn.Linear(state_dim, hidden_dim)
        lin.weight.data.uniform_(-init_w, init_w)
        lin.bias.data.uniform_(-init_w, init_w)
        self.net.append(lin)
        self.net.append(nn.Dropout(dropout))
        for i in range(num_layers):
            lin = nn.Linear(hidden_dim, hidden_dim)
            lin.weight.data.uniform_(-init_w, init_w)
            lin.bias.data.uniform_(-init_w, init_w)
            self.net.append(lin)
            self.net.append(nn.ReLU())
            self.net.append(nn.Dropout(dropout))
        lin = nn.Linear(hidden_dim, 1)
        lin.weight.data.uniform_(-init_w, init_w)
        lin.bias.data.uniform_(-init_w, init_w)
        self.net.append(lin)

    def forward(self, state):
        """Processes state through the network layers and outputs the critic's state value"""
        return self.net(state)


class SoftQNetwork(nn.Module):
    """Class of Q-networks, which are necessary for training the SAC-agent"""

    def __init__(self, num_inputs, num_actions, hidden_dim, num_layers=6, dropout=0.3, init_w=3e-3):
        """Determines network architecture at object initialization"""
        super(SoftQNetwork, self).__init__()

        self.net = nn.Sequential()
        lin = nn.Linear(num_inputs+num_actions, hidden_dim)
        lin.weight.data.uniform_(-init_w, init_w)
        lin.bias.data.uniform_(-init_w, init_w)
        self.net.append(lin)
        self.net.append(nn.Dropout(dropout))
        for i in range(num_layers):
            lin = nn.Linear(hidden_dim, hidden_dim)
            lin.weight.data.uniform_(-init_w, init_w)
            lin.bias.data.uniform_(-init_w, init_w)
            self.net.append(lin)
            self.net.append(nn.ReLU())
            self.net.append(nn.Dropout(dropout))
        lin = nn.Linear(hidden_dim, 1)
        lin.weight.data.uniform_(-init_w, init_w)
        lin.bias.data.uniform_(-init_w, init_w)
        self.net.append(lin)

    def forward(self, state, action):
        """Processes state and corresponding action through the network layers and outputs the Q-value"""
        return self.net(torch.cat([state, action], dim=-1))


class PolicyNetwork(nn.Module):
    """Class of policy network that resembles the actor"""

    def __init__(self, num_inputs, num_actions, hidden_dim, num_layers=6, dropout=0.3, init_w=3e-3, log_std_min=-20, log_std_max=2):
        """Determines network architecture at object initialization.
        The policy network has two outputs i.e., the mean-value and the deviation-value for the randomized
        action-selection by a normal distribution around the mean-value.
        """
        super(PolicyNetwork, self).__init__()

        self.num_actions = num_actions

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.net = nn.Sequential()
        lin = nn.Linear(num_inputs, hidden_dim)
        lin.weight.data.uniform_(-init_w, init_w)
        lin.bias.data.uniform_(-init_w, init_w)
        self.net.append(lin)
        for i in range(num_layers):
            lin = nn.Linear(hidden_dim, hidden_dim)
            lin.weight.data.uniform_(-init_w, init_w)
            lin.bias.data.uniform_(-init_w, init_w)
            self.net.append(lin)
            self.net.append(nn.ReLU())
            self.net.append(nn.Dropout(dropout))

        # This layer computes the mean-value
        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)

        # This layer computes the deviation-value
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        """Processes the state through the network layers to compute the mean-value and deviation-value"""
        x = self.net(state)
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        # log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std


class extract_lstm_output(nn.Module):
    def forward(self, x):
        return x[0]


class PolicySubNetwork(nn.Module):
    """Used to process bigger data amounts before passing condensed information to actual policy network"""

    def __init__(self, input_dim, hidden_dim, seq_len=1, lstm=False, num_layers=2, dropout=.1):
        """Determines network architecture at object initialization"""
        super(PolicySubNetwork, self).__init__()
        # TODO: change lstm to transformer encoder
        self.lstm = lstm
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len

        if lstm:
            self.net = nn.Sequential(
                nn.LSTM(input_dim, hidden_dim, num_layers=self.num_layers, dropout=self.dropout, batch_first=True),
                extract_lstm_output(),
                nn.ReLU(),
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(),
            )
            if num_layers > 1:
                hidden_layers = nn.ModuleList()
                for i in range(num_layers):
                    hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
                    hidden_layers.append(nn.ReLU())
                    hidden_layers.append(nn.Dropout())
                self.net.extend(hidden_layers)

    def forward(self, x):
        x = self.net(x)
        return x[:, -1, :]  # last entry should serve as encoded buy advice


class PolicyRoboboboNetwork(PolicyNetwork):
    """
    Class of policy network that resembles the actor.
    It is able to process bigger data amounts before passing condensed information to actual policy network
    It processes predictions of stock data and passes the encoded information to the actual policy network.
    The encoded information works as some sort of advice for the policy network."""

    def __init__(self, num_inputs, num_actions, hidden_size, num_layers=3, init_w=3e-3, log_std_min=-20, log_std_max=2, policy_sub_networks: Optional[nn.ModuleList]=nn.ModuleList()):
        # calculate num_inputs for policy network by subtracting the number of sub network inputs and adding the number of sub network outputs
        if policy_sub_networks is not None:
            num_inputs_policy_network = num_inputs - sum([sub_net.input_dim*sub_net.seq_len - sub_net.hidden_dim for sub_net in policy_sub_networks])
        else:
            num_inputs_policy_network = num_inputs
        super().__init__(num_inputs=num_inputs_policy_network, num_actions=num_actions, hidden_dim=hidden_size, num_layers=num_layers, init_w=init_w, log_std_min=log_std_min, log_std_max=log_std_max)
        self.policy_sub_networks = policy_sub_networks

    def forward(self, state):
        """
        1. process each corresponding input through the sub networks
        2. concatenate the remaining state tensors with the outputs of the sub networks
        3. process the concatenated tensor through the policy network

        :param state: list of tensors, where the last n tensors are the inputs for the n sub networks
        :return: mean and log_std of the policy network
        """

        # TODO: Check for correct input
        # if isinstance(state, list):
        #     assert isinstance(all(state), torch.Tensor), "state must be a list of tensors"

        state_sub_networks = []
        for i, sub_network in enumerate(self.policy_sub_networks):
            state_sub_networks.append(sub_network(state[-len(self.policy_sub_networks) + i]))
        state = state[:-len(self.policy_sub_networks)]
        state.extend(state_sub_networks)
        for i in range(len(state)):
            if len(state[i].shape) > 2:
                state[i] = state[i].reshape(state[i].shape[0], -1)
        state = torch.concat(state, dim=1)
        mean, log_std = super(PolicyRoboboboNetwork, self).forward(state)
        return mean, log_std


class SACAgent:
    """Class of the SAC-agent. Initializes the different networks, action-selection-mechanism and update-paradigm"""

    num_steps = 0
    state_values = []
    training = False
    policy_state = torch.Tensor

    def __init__(self, env: Environment, temperature=1., state_dim=None, action_dim=None, hidden_dim=256, num_layers=3,
                 hold_threshold=1e-2, replay_buffer_size=1000000):
        """Initializes the networks, determines the availability of cuda
        and initializes the replay buffer and the optimizer.
        """

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.temperature = temperature

        # self.env = NormalizedActions(env)
        self.env = env
        self.action_dim = action_dim if action_dim else self.env.action_space.shape[0]
        self.state_dim = state_dim if state_dim else self.env.observation_space.dim
        self.hidden_dim = hidden_dim

        # initialize SAC networks
        self.value_net = ValueNetwork(self.state_dim, self.hidden_dim, num_layers=num_layers).to(self.device)
        self.target_value_net = ValueNetwork(self.state_dim, self.hidden_dim, num_layers=num_layers).to(self.device)
        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(param.data)

        self.soft_q_net1 = SoftQNetwork(self.state_dim, self.action_dim, self.hidden_dim, num_layers=num_layers).to(self.device)
        self.soft_q_net2 = SoftQNetwork(self.state_dim, self.action_dim, self.hidden_dim, num_layers=num_layers).to(self.device)

        self.policy_net = PolicyNetwork(self.state_dim, self.action_dim, self.hidden_dim, num_layers=num_layers,
                                        init_w=3e-6, log_std_min=-20, log_std_max=2).to(self.device)

        # Initializes the networks' cost-function, optimizer and learning rates
        self.value_criterion = nn.MSELoss()
        self.soft_q_criterion1 = nn.MSELoss()
        self.soft_q_criterion2 = nn.MSELoss()
        self.policy_criterion = nn.L1Loss()

        self.value_lr = 1e-4
        self.soft_q_lr = 1e-4
        self.policy_lr = 1e-4

        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=self.value_lr)
        self.soft_q_optimizer1 = optim.Adam(self.soft_q_net1.parameters(), lr=self.soft_q_lr)
        self.soft_q_optimizer2 = optim.Adam(self.soft_q_net2.parameters(), lr=self.soft_q_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=self.policy_lr)

        # Initializes the replay buffer within the agent
        self.replay_buffer_size = replay_buffer_size
        self.replay_buffer = ReplayBuffer(self.replay_buffer_size)

        # threshold to enable hold action i.e., all absolute actions smaller than this value are considered as hold
        self.hold_threshold = hold_threshold

        self.num_actions = 0
        self.obs_dims = []  # for self.state_tensor_to_list; will be written at first use of self.state_list_to_tensor

    def evaluate(self, state, epsilon=1e-6):
        """Is used during training to determine the entropy H(X)"""
        if self.policy_state.__name__ == list.__name__:
            state = self.state_tensor_to_list(state)
        mean, log_std = self.policy_net.forward(state)
        log_std = torch.clamp(log_std, self.policy_net.log_std_min, self.policy_net.log_std_max)
        std = log_std.exp()

        normal = Normal(0, 1)
        z = normal.sample(std.shape).to(self.device)
        action = mean + std * z
        action = torch.clamp(action, -1., 1.)
        # softmax over all positive (buy) actions to make sure not to spend more than 100% of the cash
        if action[action > 0].sum() > 1:
            action[action > 0] = self.env.action_space.softmax(action[action > 0])
        log_prob = Normal(mean, std).log_prob(mean + std * z) - torch.log(1 - action.pow(2) + epsilon)
        return action, log_prob.sum(dim=-1).unsqueeze(-1), z, mean, log_std

    def get_action(self, state, hold=True):
        """Is used to compute an action during experience gathering and testing"""
        # state = state.unsqueeze(0).to(self.device)
        if self.policy_state.__name__ == list.__name__:
            state = self.state_tensor_to_list(state)
        mean, log_std = self.policy_net.forward(state)
        log_std = torch.clamp(log_std, self.policy_net.log_std_min, self.policy_net.log_std_max)
        if self.training:
            # deviation-value is only during training > 0 to encourage exploration
            # During testing std = 0 since exploration is not needed anymore
            std = log_std.exp()
        else:
            std = torch.zeros_like(log_std)

        # Generate random value for randomized action-selection
        normal = Normal(0, 1)
        z = normal.sample().to(self.device)

        # Draw action by applying scaled tanh-function
        action = mean + std * z
        action = torch.clamp(action, -1., 1.)

        action = action.detach().cpu()

        if hold:
            # if one action is smaller than the hold threshold, this action is set to 0
            action[torch.abs(action) < self.hold_threshold] = 0

        # softmax over all positive (buy) actions to make sure not to spend more than 100% of the cash
        if action[action > 0].sum() > 1:
            action[action > 0] = self.env.action_space.softmax(action[action > 0])

        return action.reshape(max(action.shape),)

    def update(self, batch_size, gamma=0.99, soft_tau=1e-1):
        """Update-paradigm"""

        # Draw experience from replay buffer
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)

        state = state.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        next_state = next_state.to(self.device)
        done = done.to(self.device)

        # Get all values, which are necessary for the network updates
        # (policy iteration algorithm --> policy evaluation and improvement)
        new_action, log_prob, epsilon, mean, log_std = self.evaluate(state)
        predicted_q_value1 = self.soft_q_net1(state, action)
        predicted_q_value2 = self.soft_q_net2(state, action)
        predicted_value = self.value_net(state)

        # Training Q Function
        # Compute target state value from target value network
        target_value = self.target_value_net(next_state)
        # Compute target Q-value by taking action-dependent reward into account
        target_q_value = reward + (1 - done) * gamma * target_value
        # Compute loss
        q_value_loss1 = self.soft_q_criterion1(predicted_q_value1, target_q_value.detach())
        q_value_loss2 = self.soft_q_criterion2(predicted_q_value2, target_q_value.detach())
        self.soft_q_optimizer1.zero_grad()
        q_value_loss1.backward()
        self.soft_q_optimizer1.step()
        self.soft_q_optimizer2.zero_grad()
        q_value_loss2.backward()
        self.soft_q_optimizer2.step()
        # Get new predicted q value from updated q networks for coming updates
        predicted_new_q_value = torch.min(self.soft_q_net1(state, new_action), self.soft_q_net2(state, new_action))

        # Training Value Function
        # Update State-Value-Network with updated Q-Value and logarithmic probability from policy network
        # log_prob represents entropy H(selected action)
        target_value_func = predicted_new_q_value - log_prob * self.temperature
        value_loss = self.value_criterion(predicted_value, target_value_func.detach())
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        # Training Policy Function
        policy_loss = self.policy_criterion(log_prob, (predicted_new_q_value / self.temperature).exp())
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # if any policy_net parameters are nan stop here
        if torch.isnan(torch.stack([torch.sum(param) for param in self.policy_net.parameters()])).any():
            print("Policy Net parameters are nan")

        # Update target value network by polyak-averaging
        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )

    def save_checkpoint(self, path, reward=None):
        sac_dict = {
            'value_net': self.value_net.state_dict(),
            'target_value_net': self.target_value_net.state_dict(),
            'soft_q_net1': self.soft_q_net1.state_dict(),
            'soft_q_net2': self.soft_q_net2.state_dict(),
            'policy_net': self.policy_net.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'value_optimizer': self.value_optimizer.state_dict(),
            'soft_q_optimizer1': self.soft_q_optimizer1.state_dict(),
            'soft_q_optimizer2': self.soft_q_optimizer2.state_dict(),
            'temperature': self.temperature,
            'action_dim': self.action_dim,
            'state_dim': self.state_dim,
            'hidden_dim': self.hidden_dim,
            'num_actions': self.num_actions,
            # 'reward': reward,
        }

        torch.save(sac_dict, path)

    def load_checkpoint(self, path):
        sac_dict = torch.load(path, map_location=self.device)
        self.value_net.load_state_dict(sac_dict['value_net'])
        self.target_value_net.load_state_dict(sac_dict['target_value_net'])
        self.soft_q_net1.load_state_dict(sac_dict['soft_q_net1'])
        self.soft_q_net2.load_state_dict(sac_dict['soft_q_net2'])
        self.policy_net.load_state_dict(sac_dict['policy_net'])
        self.policy_optimizer.load_state_dict(sac_dict['policy_optimizer'])
        self.value_optimizer.load_state_dict(sac_dict['value_optimizer'])
        self.soft_q_optimizer1.load_state_dict(sac_dict['soft_q_optimizer1'])
        self.soft_q_optimizer2.load_state_dict(sac_dict['soft_q_optimizer2'])
        self.temperature = sac_dict['temperature']
        self.num_actions = sac_dict['num_actions']
        print("Loaded checkpoint from path: {}".format(path))

    def train(self):
        self.training = True
        self.policy_net.train()

    def eval(self):
        self.training = False
        self.policy_net.eval()

    def state_list_to_tensor(self, state):
        state = list(state)
        if len(self.obs_dims) == 0:
            # each entry will be the sequence length of each entry
            self.obs_dims = tuple([tuple(x.shape) for x in state])
        for i, e in enumerate(state):
            # flatten state element
            state[i] = e.flatten().reshape(1, -1)
        # concatenate state elements
        state = torch.cat(state, dim=1)
        return state

    def state_tensor_to_list(self, state):
        # transform tensor from shape (batch_size, obs_dim) to list of size (obs)
        # where each entry has shape (batch_size, self.obs_dims[entry, 0], self.obs_dim[entry, 1])
        state_list = []
        for i, shape in enumerate(self.obs_dims):
            state_list.append(state[:, :np.product(shape)].reshape([state.shape[0]] + list(shape)))
            state = state[:, np.product(shape):]
        return state_list


class RoboBobo(SACAgent):
    def __init__(self, env: Environment,
                 temperature=1.,
                 state_dim=None,
                 action_dim=None,
                 hidden_dim=256,
                 num_layers=3,
                 hold_threshold=1e-3,
                 replay_buffer_size=1e6):
        super(RoboBobo, self).__init__(env, temperature=temperature,
                                       state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim, num_layers=3,
                                       hold_threshold=hold_threshold, replay_buffer_size=replay_buffer_size)

        self.policy_state = list
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        # self.state_dim = 1 + env.observation_space.num_stocks

        # modify state dim according to autoencoder:
        #  state_dim =
        #       current cash (float scalar)
        #  +    historic and current REAL portofolio value (matrix: (observation_length x num_stocks))
        #  +    historic, current and predicted ENCODED stock prices (matrix: (observation_length + seq_len) x encoder.output_dim)
        #  +    validation of prediction (scalar) TODO: turn scalar into vector of validation score for each feature

    def process_stock_prices(self, stock_prices):
        """process stock prices
        :param stock_prices: (tensor) stock prices with shape (observation_length, features)"""
        # differentiate stock prices
        stock_prices = torch.diff(stock_prices, dim=0)
        stock_prices = torch.cat((stock_prices[0, :].reshape(1, -1), stock_prices), dim=0)
        # standardize stock prices
        if self.scaler:
            stock_prices = self.scaler.transform(stock_prices)
        stock_prices = torch.FloatTensor(stock_prices).unsqueeze(0).to(self.device)
        # encode state
        if self.autoencoder:
            stock_prices = self.encode(stock_prices)

        # reshape stock prices
        stock_prices = stock_prices.reshape(-1, stock_prices.shape[1] * stock_prices.shape[2])

        # predict follow-up stock prices
        predicted = self.predict(stock_prices).squeeze(2).transpose(2, 1)
        stock_prices = torch.cat((stock_prices, predicted), dim=1)
        # validate prediction
        # validation = self.validate(self.decode(stock_prices).unsqueeze(2).transpose(3, 1))
        # stock_prices = torch.cat((stock_prices, validation), dim=0)

        return stock_prices.squeeze(0)

    def get_action(self, state, hold=True):
        action = super().get_action(state)

        return action

    def adjust_nets(self, policy_sub_networks: Optional[nn.ModuleList]=None):
        # initialize SAC networks
        self.value_net = ValueNetwork(self.state_dim, self.hidden_dim, num_layers=self.num_layers).to(self.device)
        self.target_value_net = ValueNetwork(self.state_dim, self.hidden_dim, num_layers=self.num_layers).to(self.device)
        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(param.data)

        self.soft_q_net1 = SoftQNetwork(self.state_dim, self.action_dim, self.hidden_dim, num_layers=self.num_layers).to(self.device)
        self.soft_q_net2 = SoftQNetwork(self.state_dim, self.action_dim, self.hidden_dim, num_layers=self.num_layers).to(self.device)

        self.policy_net = PolicyRoboboboNetwork(self.state_dim, self.action_dim, self.hidden_dim, num_layers=self.num_layers,
                                                init_w=3e-3, log_std_min=-20, log_std_max=2,
                                                policy_sub_networks=policy_sub_networks).to(self.device)

        # Initializes the networks' cost-function, optimizer and learning rates
        self.value_criterion = nn.MSELoss()
        self.soft_q_criterion1 = nn.MSELoss()
        self.soft_q_criterion2 = nn.MSELoss()
        self.policy_criterion = nn.L1Loss()

        self.value_lr = 1e-4
        self.soft_q_lr = 1e-4
        self.policy_lr = 1e-4

        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=self.value_lr)
        self.soft_q_optimizer1 = optim.Adam(self.soft_q_net1.parameters(), lr=self.soft_q_lr)
        self.soft_q_optimizer2 = optim.Adam(self.soft_q_net2.parameters(), lr=self.soft_q_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=self.policy_lr)

    def set_state_dim(self, num_states: int):
        """sets the dimension of the state"""
        self.state_dim = num_states

    def set_action_dim(self, num_actions: int):
        """sets the dimension of the action space"""
        self.action_dim = num_actions

    def create_policy_sub_network(self, input_dim, hidden_dim, seq_len=1, lstm=False, num_layers=2, dropout=.1):
        return PolicySubNetwork(input_dim, hidden_dim, seq_len=seq_len, lstm=lstm, num_layers=num_layers,
                                dropout=dropout).to(self.device)


class DataProcessor:
    """Class for processing stock data for the agent.
    Applies Filter, Downsampling, Autoencoder and Predictions to the data given as the stock prices.
    Multiple types of predictions can be applied.
    Short-term predictions take the data as it is and predict the next n steps.
    Mid-term predictions take downsampled and filtered data to cover a longer time period and predict the next m steps.
    For each prediction ptype, the data is processed in the following way:
    1. Short-term or mid-term Filter
    2. Short-term or mid-term Downsampling
    3. Short-term or mid-term Autoencoder
    4. Short-term or mid-term Prediction"""

    def __init__(self, scaler=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = {}
        # example for one processor of ptype "ptype":
        # {"ptype": {"filter": {"a": None, "b": None},
        #          "scaler": None,
        #          "downsampling_rate": None,
        #          "autoencoder": None,
        #          "prediction": None, }, }

    def add_processor(self, ptype: str,
                      predictor: Optional[nn.Module]=None,
                      autoencoder: Optional[nn.Module]=None,
                      mvg_avg: Optional[int]=None,
                      downsampling_rate: Optional[int]=None,
                      scaler=None,
                      differentiate=False):
        # check if filter is valid
        # if filter:
        #     if len(filter) != 2:
        #         raise ValueError("Filter must be a tuple with 2 elements, where the first element is "
        #                          "the numerator (a) and the second element is the denominator (b) "
        #                          "for the Butterworth filter from scikit.signal.filtfilt.")

        # add ptype to keys of processor
        self.processor[ptype] = {"differentiate": differentiate,
                                 "mvg_avg": None,
                                 "downsampling_rate": None,
                                 "autoencoder": None,
                                 "predictor": None,
                                 "scaler": None,}
        if differentiate:
            self.processor[ptype]["differentiate"] = differentiate
        if mvg_avg:
            self.processor[ptype]["mvg_avg"] = mvg_avg
        if downsampling_rate:
            self.processor[ptype]["downsampling_rate"] = downsampling_rate
        if scaler:
            self.processor[ptype]["scaler"] = scaler
        if autoencoder:
            self.processor[ptype]["autoencoder"] = autoencoder
        if predictor:
            self.processor[ptype]["predictor"] = predictor

    def process(self, stock_prices: np.ndarray, ptype: str, flatten=False, mask=None):
        """process stock prices
        :param stock_prices: (tensor) stock prices with shape (observation_length, features)
        :param ptype: (str) type of prediction to apply"""

        # check if ptype is valid
        if ptype not in self.processor.keys():
            raise ValueError("ptype must be one of {}".format(tuple(self.processor.keys())))

        masked_values = None
        if mask is not None and (isinstance(mask, int) or isinstance(mask, float) or isinstance(mask, np.ndarray)):
            # mask stock prices
            stock_prices, masked_values = self._mask(stock_prices, mask)

        # differentiate stock prices
        if self.processor[ptype]["differentiate"]:
            stock_prices = self._differentiate(stock_prices)

        # standardize stock prices
        if self.processor[ptype]["scaler"]:
            stock_prices = self._standardize(stock_prices, ptype)

        # filter stock prices
        if self.processor[ptype]["mvg_avg"] is not None:
            stock_prices = self._filter(stock_prices, ptype)

        # encode stock prices
        if self.processor[ptype]["autoencoder"]:
            print('Tried to use encoder in data processor. Currently not implemented.')
            # stock_prices = self._encode(stock_prices, ptype)

        # concatenate masked values and stock prices
        if masked_values is not None:
            # if not isinstance(stock_prices, torch.Tensor):
            #     stock_prices = torch.from_numpy(stock_prices).to(self.device)
            masked_values = np.zeros((len(masked_values), stock_prices.shape[-1])) + mask
            stock_prices = np.concatenate((masked_values, stock_prices), axis=0)  # TODO: only front padding; Make dependent on indeces masked_values

        # downsample stock prices
        if self.processor[ptype]["downsampling_rate"]:
            stock_prices = self._downsample(stock_prices, ptype)

        # predict stock prices
        if self.processor[ptype]["predictor"]:
            stock_prices = self._predict(stock_prices, ptype)
            # decoded = self.processor[ptype]["autoencoder"].decode(stock_prices)
        # check if necessary to decode stock prices

        # reshape stock prices
        if flatten:
            stock_prices = stock_prices.reshape(-1, stock_prices.shape[-1] * stock_prices.shape[-2])

        return stock_prices

    def _differentiate(self, stock_prices):
        """differentiate stock prices
        :param stock_prices: (tensor) stock prices with shape (observation_length, features)"""
        stock_prices = np.diff(stock_prices, axis=0)
        stock_prices = np.concatenate((stock_prices[0, :].reshape(1, -1), stock_prices), axis=0)
        return stock_prices

    def _standardize(self, stock_prices, ptype):
        return self.processor[ptype]["scaler"].transform(stock_prices)

    def _filter(self, stock_prices, ptype):
        win_len = np.min((len(stock_prices), self.processor[ptype]["mvg_avg"]))
        return moving_average(stock_prices, win_len)

    def _downsample(self, stock_prices, ptype):
        return stock_prices[::-self.processor[ptype]["downsampling_rate"]][::-1]

    def _encode(self, stock_prices, ptype):
        # check if stock_prices is a tensor
        if not isinstance(stock_prices, torch.Tensor):
            stock_prices = torch.tensor(stock_prices, dtype=torch.float32).to(self.device)
        return self.processor[ptype]["autoencoder"].encode(stock_prices)

    def _predict(self, stock_prices, ptype):
        # check if stock_prices is a tensor
        if not isinstance(stock_prices, torch.Tensor):
            stock_prices = torch.tensor(stock_prices, dtype=torch.float32).to(self.device)
        # flatten stock_prices
        # stock_prices = stock_prices.reshape(-1, stock_prices.shape[-1] * stock_prices.shape[-2])
        if self.processor[ptype]["predictor"].latent_dim != stock_prices.shape[-1]:
            # draw latent variable from normal distribution and concatenate it to stock_prices
            latent_variable = torch.randn((stock_prices.shape[0], self.processor[ptype]["predictor"].latent_dim - stock_prices.shape[-1])).to(self.device)
            stock_prices = torch.cat((latent_variable, stock_prices), dim=1).unsqueeze(0).float()
        return self.processor[ptype]["predictor"](stock_prices).squeeze(0)  # .squeeze(2).permute(1, 0)

    def _decode(self, stock_prices, ptype):
        # check if stock_prices is a tensor
        if not isinstance(stock_prices, torch.Tensor):
            stock_prices = torch.tensor(stock_prices, dtype=torch.float32).to(self.device)
        return self.processor[ptype]["autoencoder"].decode(stock_prices)

    def _mask(self, stock_prices, mask):
        # mask stock prices if all values of one time step are 'mask'
        masked_values = np.where(np.all(stock_prices == mask, axis=1))[0]
        stock_prices = np.delete(stock_prices, masked_values, axis=0)
        return stock_prices, masked_values


def train_sac(env: Environment, agent: SACAgent, data_processor: DataProcessor,
              max_episodes: int, batch_size: int, parameter_update_interval: int,
              num_random_actions=None, path_checkpoint=None, checkpoint_interval=100):
    """Batch training method
    This method interacts with the environment and trains the agent in batches.

    The environment must be of the following structure:
    - env.reset() --> returns initial state
    - env.step(action) --> returns next state, reward, done, info
    - env.hard_reset() --> resets all environment parameters to initial state
    - env.soft_reset() --> resets only certain environment parameters to initial state
    - env.action_space.sample() --> returns random action from the environment action space

    The agent is provided as a class within this file.
    """

    episode = 0
    total_equity_final = []
    agent.train()
    cpu_device = torch.device('cpu')

    if not num_random_actions:
        num_random_actions = agent.replay_buffer_size/4

    try:
        while episode < max_episodes:
            env.hard_reset(random_split=True, split_length=int(365*2))
            env.set_observation_space(stock_prices=env.stock_data[:env.observation_length])

            done = False
            while not done:
                # if agent is RoboBobo, filter, downsample, scale, encode and predict follow-up stock prices before passing to agent
                if isinstance(agent, RoboBobo):
                    state = list(env.observation_space(normalized=True))
                    # delete last element of state (last stock prices) as it is not used by RoboBobo
                    # del state[-1]
                    for key in data_processor.processor.keys():
                        # get downsampling rate if available
                        if data_processor.processor[key]["downsampling_rate"]:
                            downsampling_rate = data_processor.processor[key]["downsampling_rate"]
                        else:
                            downsampling_rate = 1
                        # get start index of observed stock prices by taking maximum of 0 and current time step minus observation length
                        start_idx = max(0, env.t - 1 - env.observation_length * downsampling_rate)
                        obs_stock_data = np.zeros((env.observation_length * downsampling_rate, env.stock_data.shape[1]))
                        obs_stock_data[-env.t + 1:, :] = env.stock_data[start_idx:env.t - 1, :]
                        state.append(data_processor.process(obs_stock_data, ptype=key, flatten=False, mask=0).to(cpu_device))
                    state = agent.state_list_to_tensor(state)
                else:
                    state = env.observation_space(dtype=torch.Tensor)

                # Decide whether to draw random action or to use agent
                if len(agent.replay_buffer) < num_random_actions:
                    # Draw random action
                    action = env.action_space.sample(hold_threshold=agent.hold_threshold)
                else:
                    # Draw greedy action
                    action = agent.get_action(state.to(agent.device)).to(cpu_device)

                # Give chosen action to environment to adjust internal parameters and to compute new state
                next_state, reward, done, _ = env.step(action)

                if isinstance(agent, RoboBobo):
                    next_state = list(next_state)
                    # del next_state[-1]
                    for key in data_processor.processor.keys():
                        # get downsampling rate if available
                        if data_processor.processor[key]["downsampling_rate"]:
                            downsampling_rate = data_processor.processor[key]["downsampling_rate"]
                        else:
                            downsampling_rate = 1
                        # get start index of observed stock prices by taking maximum of 0 and current time step minus observation length
                        start_idx = max(0, env.t-1-env.observation_length * downsampling_rate)
                        obs_stock_data = np.zeros((env.observation_length * downsampling_rate, env.stock_data.shape[1]))
                        obs_stock_data[-env.t+1:, :] = env.stock_data[start_idx:env.t-1, :]
                        next_state.append(data_processor.process(obs_stock_data, ptype=key, flatten=False, mask=0).to(cpu_device))
                    next_state = agent.state_list_to_tensor(next_state)

                # Append experience to replay buffer
                agent.replay_buffer.push(state,
                                         action.reshape(1, -1),
                                         reward.reshape(1, 1),
                                         next_state,
                                         torch.FloatTensor([done]).reshape(1, 1))

                # Update parameters each n steps
                if agent.num_actions % parameter_update_interval == 0 \
                        and len(agent.replay_buffer) > num_random_actions \
                        and len(agent.replay_buffer) > batch_size:
                    agent.update(batch_size)

                agent.num_actions += 1

            # Collect total equity of current episode
            print("Episode: {} -- time steps: {} -- total equity: {} -- total equity per time: {}".format(episode + 1, env.t, np.round(env.total_equity().item(), 2), np.round(env.total_equity().item()/len(env.stock_data), 2)))
            total_equity_final.append(env.total_equity().item())
            episode += 1

            # Save model for later use
            if path_checkpoint and episode % checkpoint_interval == 0:
                agent.save_checkpoint(path_checkpoint, reward=total_equity_final)

        return total_equity_final, agent
    except KeyboardInterrupt:
        print("KeyboardInterrupt detected.")
        print("Results:")

        # plot_results(episode_rewards, all_rewards, all_actions)
        avg_reward = np.mean(total_equity_final)
        median_reward = np.median(total_equity_final)
        print("Average episodic reward:\t" + str(avg_reward) + "\nMedian episodic reward:\t" + str(median_reward))

        plt.plot(total_equity_final)
        plt.plot(np.convolve(total_equity_final, np.ones(10) / 10, mode='valid'))
        plt.ylabel('Total final equity [$]')
        plt.xlabel('Episode')
        plt.title('Total final equity after each episode in [$]')
        plt.legend(['Total final equity [$]', 'Avg. total final equity [$]'])
        plt.show()


def test_sac(env: Environment, agent: SACAgent, data_processor: DataProcessor, test=True, plot=True, plot_reference=False):
    """Test trained SAC agent"""
    cpu_device = torch.device('cpu')
    done = False
    total_equity = []
    actions = []
    portfolio = []
    if test:
        agent.eval()
    else:
        agent.train()
    env.hard_reset()
    env.set_observation_space(stock_prices=env.stock_data[:env.observation_length])

    while not done:
        if isinstance(agent, RoboBobo):
            state = list(env.observation_space(normalized=True))
            # delete last element of state (last stock prices) as it is not used by RoboBobo
            # del state[-1]
            for key in data_processor.processor.keys():
                # get downsampling rate if available
                if data_processor.processor[key]["downsampling_rate"]:
                    downsampling_rate = data_processor.processor[key]["downsampling_rate"]
                else:
                    downsampling_rate = 1
                # get start index of observed stock prices by taking maximum of 0 and current time step minus observation length
                start_idx = max(0, env.t-1-env.observation_length * downsampling_rate)
                obs_stock_data = np.zeros((env.observation_length * downsampling_rate, env.stock_data.shape[1]))
                obs_stock_data[-env.t+1:, :] = env.stock_data[start_idx:env.t-1, :]
                state.append(data_processor.process(obs_stock_data, ptype=key, flatten=False, mask=0).to(cpu_device))
            state = agent.state_list_to_tensor(state)
        else:
            state = env.observation_space(dtype=torch.Tensor)

        action = agent.get_action(state.float().to(agent.device))
        state, _, done, _ = env.step(action)
        total_equity.append(env.total_equity().item())

        actions.append(action.detach().cpu().numpy())
        portfolio.append(env.portfolio.detach().cpu().squeeze(0).numpy())

    print("Test scenario -- final total equity: {}".format(env.total_equity().item()))

    if plot:
        plt.plot(total_equity)
        plt.plot(np.convolve(total_equity, np.ones(10) / 10, mode='valid'))
        plt.ylabel('Total equity [$]')
        plt.xlabel('Time step')
        plt.title(f"Total final equity in [$] (Grow: {total_equity[-1]/total_equity[0]:.2f})")
        plt.legend(['Total equity [$]', 'Avg. total final equity [$]'])
        plt.show()

        visualize_actions(np.array(actions), min=-1, max=1, title='actions over time')
        visualize_actions(np.array(portfolio), cmap='sequential', title='portfolio over time')

    if plot_reference:
        # plot the average of all stock prices
        avg = torch.mean(env.stock_data, dim=1)
        plt.plot(avg)
        plt.ylabel('Average stock price [$]')
        plt.xlabel('Time step')
        plt.title(f"Avg stock price in [$] (Grow: {avg[-1]/avg[0]:.2f})")
        plt.show()


def visualize_actions(matrix, min=None, max=None, cmap='binary', title=None):
    # Calculate mean and standard deviation per time step
    mean_values = np.mean(matrix, axis=1)
    std_values = np.std(matrix, axis=1)

    # Create a colormap from blue to white to red
    if cmap == 'binary':
        cmap = mpl.colormaps['coolwarm']
    else:
        cmap = mpl.colormaps['Reds']

    # Set the color range based on the minimum and maximum values in the matrix
    vmin = min if min is not None else np.min(matrix)
    vmax = max if max is not None else np.max(matrix)

    # Plot the matrix using imshow
    plt.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.ylabel('time steps')
    plt.xlabel('features')
    plt.title(title if title is not None else '')
    plt.show()

    # Plot the mean value with a solid line
    plt.plot(mean_values, color='black', label='Mean')

    # Plot the standard deviation band
    plt.fill_between(range(len(std_values)), mean_values + std_values, mean_values - std_values,
                       color='gray', alpha=0.3, label='Standard Deviation')

    plt.xlabel('Time Step')
    plt.legend()
    plt.show()

