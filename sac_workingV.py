import copy
import os
import random
from datetime import datetime
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy import signal
from torch.distributions import Normal

from environment import Environment


class ReplayBuffer:
    """Class of the replay buffer for experience replay"""

    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        """push new experience into buffer"""
        experience = (state.detach().clone(), action, reward, next_state.detach().clone(), done)
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
            state_batch.append(state.detach().clone())
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state.detach().clone())
            done_batch.append(done)

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def __len__(self):
        return len(self.buffer)


class ValueNetwork(nn.Module):
    """Class of value network that resembles the critic"""

    def __init__(self, state_dim, hidden_dim, init_w=3e-3):
        """Determines network architecture at object initialization"""
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, hidden_dim)
        self.linearout = nn.Linear(hidden_dim, 1)

        self.linear1.weight.data.uniform_(-init_w, init_w)
        self.linear1.bias.data.uniform_(-init_w, init_w)
        self.linear2.weight.data.uniform_(-init_w, init_w)
        self.linear2.bias.data.uniform_(-init_w, init_w)
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)
        self.linear4.weight.data.uniform_(-init_w, init_w)
        self.linear4.bias.data.uniform_(-init_w, init_w)
        self.linearout.weight.data.uniform_(-init_w, init_w)
        self.linearout.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        """Processes state through the network layers and outputs the critic's state value"""
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        x = self.linearout(x)

        return x


class SoftQNetwork(nn.Module):
    """Class of Q-networks, which are necessary for training the SAC-agent"""

    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3):
        """Determines network architecture at object initialization"""
        super(SoftQNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, hidden_size)
        self.linearout = nn.Linear(hidden_size, 1)

        self.linear1.weight.data.uniform_(-init_w, init_w)
        self.linear1.bias.data.uniform_(-init_w, init_w)
        self.linear2.weight.data.uniform_(-init_w, init_w)
        self.linear2.bias.data.uniform_(-init_w, init_w)
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)
        self.linear4.weight.data.uniform_(-init_w, init_w)
        self.linear4.bias.data.uniform_(-init_w, init_w)
        self.linearout.weight.data.uniform_(-init_w, init_w)
        self.linearout.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        """Processes state and corresponding action through the network layers and outputs the Q-value"""
        x = torch.cat([state, action], 2)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        x = F.tanh(self.linearout(x))
        return x


class PolicyNetwork(nn.Module):
    """Class of policy network that resembles the actor"""

    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3, log_std_min=-20, log_std_max=2):
        """Determines network architecture at object initialization.
        The policy network has two outputs i.e., the mean-value and the deviation-value for the randomized
        action-selection by a normal distribution around the mean-value.
        """
        super(PolicyNetwork, self).__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, hidden_size)

        # This layer computes the mean-value
        self.mean_linear = nn.Linear(hidden_size, num_actions)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)

        # This layer computes the deviation-value
        self.log_std_linear = nn.Linear(hidden_size, num_actions)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)

        self.linear1.weight.data.uniform_(-init_w, init_w)
        self.linear1.bias.data.uniform_(-init_w, init_w)
        self.linear2.weight.data.uniform_(-init_w, init_w)
        self.linear2.bias.data.uniform_(-init_w, init_w)
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)
        self.linear4.weight.data.uniform_(-init_w, init_w)
        self.linear4.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        """Processes the state through the network layers to compute the mean-value and deviation-value"""
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))

        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        # log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std


class SACAgent:
    """Class of the SAC-agent. Initializes the different networks, action-selection-mechanism and update-paradigm"""

    num_steps = 0
    state_values = []
    training = False

    def __init__(self, env: Environment, temperature=1., hidden_dim=256,
                 hold_threshold=1e-3, replay_buffer_size=1000000):
        """Initializes the networks, determines the availability of cuda
        and initializes the replay buffer and the optimizer.
        """

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.temperature = temperature

        # self.env = NormalizedActions(env)
        self.env = env

        self.action_dim = self.env.action_space.shape[0]
        self.state_dim = self.env.observation_space.dim
        self.hidden_dim = hidden_dim

        # initialize SAC networks
        self.value_net = ValueNetwork(self.state_dim, self.hidden_dim).to(self.device)
        self.target_value_net = ValueNetwork(self.state_dim, self.hidden_dim).to(self.device)
        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(param.data)

        self.soft_q_net1 = SoftQNetwork(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)
        self.soft_q_net2 = SoftQNetwork(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)

        self.policy_net = PolicyNetwork(self.state_dim, self.action_dim, self.hidden_dim,
                                        init_w=3e-6, log_std_min=-20, log_std_max=2).to(self.device)

        # Initializes the networks' cost-function, optimizer and learning rates
        self.value_criterion = nn.MSELoss()
        self.soft_q_criterion1 = nn.MSELoss()
        self.soft_q_criterion2 = nn.MSELoss()
        self.policy_criterion = nn.L1Loss()  # TODO: was L1Loss

        self.value_lr = 3e-3
        self.soft_q_lr = 3e-3
        self.policy_lr = 3e-3

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

    def evaluate(self, state, epsilon=1e-6):
        """Is used during training to determine the entropy H(X)"""
        mean, log_std = self.policy_net.forward(state)
        log_std = torch.clamp(log_std, self.policy_net.log_std_min, self.policy_net.log_std_max)
        std = log_std.exp()

        normal = Normal(0, 1)
        z = normal.sample().to(self.device)
        action = mean + std * z
        action = torch.clamp(action, -1., 1.)
        log_prob = Normal(mean, std).log_prob(mean + std * z) - torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(dim=2).unsqueeze(-1)
        return action, log_prob, z, mean, log_std

    def get_action(self, state):
        """Is used to compute an action during experience gathering and testing"""
        state = state.unsqueeze(0).to(self.device)  #.transpose(1, 2)
        mean, log_std = self.policy_net.forward(state)
        log_std = torch.clamp(log_std, self.policy_net.log_std_min, self.policy_net.log_std_max)
        if self.training:
            # deviation-value is only during training > 0 to encourage exploration
            # During testing the deviation = 0 since exploration is not needed anymore
            std = log_std.exp()
        else:
            std = torch.tensor(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)

        # Generate random value for randomized action-selection
        normal = Normal(0, 1)
        z = normal.sample().to(self.device)

        # Draw action by applying scaled tanh-function
        action = mean + std * z
        action = torch.clamp(action, -1., 1.)

        action = action.detach().cpu()

        # softmax over all positive (buy) actions to make sure not to spend more than 100% of the cash
        if action[0, 0][action[0, 0] > 0].sum() > 1:
            action_pos = np.where(action[0, 0] > 0)[0]
            action[0, 0, action_pos] = self.env.action_space.softmax(action[0, 0, action_pos])

        if len(state.size()) == 3:
            action = action[0, 0]
        else:
            action = action[0, 0, 0]

        return action

    def update(self, batch_size, gamma=0.99, soft_tau=1e-1):
        """Update-paradigm"""

        # Draw experience from replay buffer
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)

        # convert list of tensors to tensor
        state = torch.stack(state).float().to(self.device)
        next_state = torch.stack(next_state).float().to(self.device)
        action = torch.stack(action).float().unsqueeze(1).to(self.device)
        reward = torch.stack(reward).float().to(self.device)
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).unsqueeze(1).to(self.device)

        # Get all values, which are necessary for the network updates
        # (policy iteration algorithm --> policy evaluation and improvement)
        predicted_q_value1 = self.soft_q_net1(state, action)
        predicted_q_value2 = self.soft_q_net2(state, action)
        predicted_value = self.value_net(state)
        new_action, log_prob, epsilon, mean, log_std = self.evaluate(state)

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
        target_value_func = predicted_new_q_value - log_prob*self.temperature
        value_loss = self.value_criterion(predicted_value, target_value_func.detach())
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        # Training Policy Function
        policy_loss = self.policy_criterion(log_prob, (predicted_new_q_value/self.temperature).exp())
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
            #'reward': reward,
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

    def eval(self):
        self.training = False

class RoboBobo(SACAgent):
    def __init__(self, env: Environment, temperature=1., hidden_dim=256, hold_threshold=1e-3, replay_buffer_size=1e6):
        super(RoboBobo, self).__init__(env, temperature=temperature, hidden_dim=hidden_dim,
                                       hold_threshold=hold_threshold, replay_buffer_size=replay_buffer_size)

        # set state dim
        self.state_dim = 1 + env.observation_space.num_stocks

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
        if hold:
            # if one action is smaller than the hold threshold, this action is set to 0
            action[torch.abs(action) < self.hold_threshold] = 0

        return action

    def adjust_nets(self):
        # initialize SAC networks
        self.value_net = ValueNetwork(self.state_dim, self.hidden_dim).to(self.device)
        self.target_value_net = ValueNetwork(self.state_dim, self.hidden_dim).to(self.device)
        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(param.data)

        self.soft_q_net1 = SoftQNetwork(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)
        self.soft_q_net2 = SoftQNetwork(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)

        self.policy_net = PolicyNetwork(self.state_dim, self.action_dim, self.hidden_dim,
                                        init_w=3e-6, log_std_min=-20, log_std_max=2).to(self.device)

        # Initializes the networks' cost-function, optimizer and learning rates
        self.value_criterion = nn.MSELoss()
        self.soft_q_criterion1 = nn.MSELoss()
        self.soft_q_criterion2 = nn.MSELoss()
        self.policy_criterion = nn.L1Loss()

        self.value_lr = 3e-3
        self.soft_q_lr = 3e-3
        self.policy_lr = 3e-3

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


# def mini_batch_train(env: Environment, agent: SACAgent,
#                      max_episodes: int, batch_size: int, parameter_update_interval: int,
#                      num_random_actions=None, path_checkpoint=None, checkpoint_interval=100):
#     """Batch training method
#     This method interacts with the environment and trains the agent in batches.
#
#     The environment must be of the following structure:
#     - env.reset() --> returns initial state
#     - env.step(action) --> returns next state, reward, done, info
#     - env.hard_reset() --> resets all environment parameters to initial state
#     - env.soft_reset() --> resets only certain environment parameters to initial state
#     - env.action_space.sample() --> returns random action from the environment action space
#
#     The agent is provided as a class within this file.
#     """
#
#     episode = 0
#     total_equity_final = []
#     agent.train()
#
#     if not num_random_actions:
#         num_random_actions = agent.replay_buffer_size/4
#
#
#     try:
#         while episode < max_episodes:
#             env.hard_reset(random_split=True, split_length=360)  # TODO: Check for other split lengths
#             env.set_observation_space(stock_prices=env.stock_data[:env.observation_length])
#
#             done = False
#             while not done:
#                 # if agent is RoboBobo, encode, predict and validate follow-up stock prices before passing to agent
#                 if isinstance(agent, RoboBobo):
#                     state = list(env.observation_space(normalized=False))
#                     state[2] = agent.process_stock_prices(state[2])
#                     for i, e in enumerate(state):
#                         # flatten state element
#                         state[i] = e.flatten().reshape(1, -1).to(agent.device)
#                     # concatenate state elements
#                     state = torch.cat(state, dim=1)
#                 else:
#                     state = env.observation_space(dtype=torch.Tensor)
#
#                 # epsilon-greedy action selection (epsilon-decay)
#                 if len(agent.replay_buffer) < num_random_actions:
#                     # Draw random action
#                     action = env.action_space.sample()
#                     action[action.abs() < agent.hold_threshold] = 0.
#                 else:
#                     # Draw greedy action
#                     action = agent.get_action(state.float())
#
#                 # Give chosen action to environment to adjust internal parameters and to compute new state
#                 next_state, reward, done, _ = env.step(action)
#
#                 if isinstance(agent, RoboBobo):
#                     next_state = list(next_state)
#                     next_state[2] = agent.process_stock_prices(next_state[2])
#                     for i, e in enumerate(next_state):
#                         # flatten state element
#                         next_state[i] = e.flatten().reshape(1, -1).to(agent.device)
#                     # concatenate state elements
#                     next_state = torch.cat(next_state, dim=1)
#
#                 # Append experience to replay buffer
#                 agent.replay_buffer.push(state, action, reward, next_state, done)
#
#                 # Update parameters each n steps
#                 if agent.num_actions % parameter_update_interval == 0 \
#                         and len(agent.replay_buffer) > num_random_actions \
#                         and len(agent.replay_buffer) > batch_size:
#                         agent.update(batch_size)
#
#                 agent.num_actions += 1
#
#             # Collect total equity of current episode
#             print("Episode: {} -- time steps: {} -- total equity: {} -- total equity per time: {}".format(episode + 1, env.t, np.round(env.total_equity().item(), 2), np.round(env.total_equity().item()/len(env.stock_data), 2)))
#             total_equity_final.append(env.total_equity().item())
#             episode += 1
#
#             # Save model for later use
#             if path_checkpoint and episode % checkpoint_interval == 0:
#                 agent.save_checkpoint(path_checkpoint, reward=total_equity_final)
#
#         return total_equity_final, agent
#     except KeyboardInterrupt:
#         print("KeyboardInterrupt detected.")
#         print("Results:")
#
#         # plot_results(episode_rewards, all_rewards, all_actions)
#         avg_reward = np.mean(total_equity_final)
#         median_reward = np.median(total_equity_final)
#         print("Average episodic reward:\t" + str(avg_reward) + "\nMedian episodic reward:\t" + str(median_reward))
#
#         plt.plot(total_equity_final)
#         plt.plot(np.convolve(total_equity_final, np.ones(10) / 10, mode='valid'))
#         plt.ylabel('Total final equity [$]')
#         plt.xlabel('Episode')
#         plt.title('Total final equity after each episode in [$]')
#         plt.legend(['Total final equity [$]', 'Avg. total final equity [$]'])
#         plt.show()


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

    def add_processor(self, ptype: str, predictor: Optional[nn.Module]=None, autoencoder: Optional[nn.Module]=None, filter: Optional[tuple]=None, downsampling_rate: Optional[int]=None, scaler=None):
        # check if filter is valid
        if filter:
            if len(filter) != 2:
                raise ValueError("Filter must be a tuple with 2 elements, where the first element is "
                                 "the numerator (a) and the second element is the denominator (b) "
                                 "for the Butterworth filter from scikit.signal.filtfilt.")

        # add ptype to keys of processor
        self.processor[ptype] = {"filter": {"a": None, "b": None},
                                "downsampling_rate": None,
                                "autoencoder": None,
                                "predictor": None, }
        if filter:
            self.processor[ptype]["filter"]["a"] = filter[0]
            self.processor[ptype]["filter"]["b"] = filter[1]
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
        stock_prices = self._differentiate(stock_prices)

        # filter stock prices
        if self.processor[ptype]["filter"]["a"] is not None:
            stock_prices = self._filter(stock_prices, ptype)

        # downsample stock prices
        if self.processor[ptype]["downsampling_rate"]:
            stock_prices = self._downsample(stock_prices, ptype)

        # standardize stock prices
        if self.processor[ptype]["scaler"]:
            stock_prices = self._standardize(stock_prices, ptype)

        # encode stock prices
        if self.processor[ptype]["autoencoder"]:
            stock_prices = self._encode(stock_prices, ptype)

        # concatenate masked values and stock prices
        if masked_values is not None:
            if not isinstance(stock_prices, torch.Tensor):
                stock_prices = torch.from_numpy(stock_prices).to(self.device)
            masked_values = torch.zeros((len(masked_values), stock_prices.shape[-1])) + mask
            stock_prices = torch.concat((masked_values, stock_prices), dim=0)

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
        # flip stock_prices for filtering
        # stock_prices_flipped = np.flip(stock_prices, axis=0)
        # concat stock_prices_flipped with with stock_prices until min_length is reached
        # stock_prices_cat = stock_prices#np.concatenate((np.flip(stock_prices, axis=0), stock_prices), axis=0)
        # toggle_pos = 1
        # while len(stock_prices_cat) < 2000:
        #     if toggle_pos == 1:
        #         stock_prices_cat = np.concatenate((stock_prices_cat, np.flip(stock_prices_cat[:len(stock_prices)], axis=0)), axis=0)
        #         toggle_pos = 0
        #     else:   # toggle_pos == 0
        #         stock_prices_cat = np.concatenate((np.flip(stock_prices_cat[-len(stock_prices):], axis=0), stock_prices_cat), axis=0)
        #         toggle_pos = 1
        # # get index of original sequence
        # if toggle_pos == 1:
        #     # last sequence was appended to the front --> original sequence is around the center
        #     index_original = np.arange(len(stock_prices_cat)//2 - len(stock_prices)//2, len(stock_prices_cat)//2 + len(stock_prices), len(stock_prices))
        # else:
        #     # last sequence was appended to the back --> original sequence is right before the center
        #     index_original = np.arange(len(stock_prices_cat)//2 - len(stock_prices), len(stock_prices_cat)//2 + len(stock_prices), len(stock_prices))
        stock_prices_filtered = np.zeros_like(stock_prices)
        float_stock_prices = np.zeros_like(stock_prices)
        for i in range(stock_prices.shape[1]):
            stock_prices_filtered[:, i] = signal.filtfilt(self.processor[ptype]["filter"]["b"], self.processor[ptype]["filter"]["a"], stock_prices[:, i])
            stock_prices_filtered[:, i] = np.flip(stock_prices_filtered[:, i], axis=0)
            # floating average of stock_prices
            float_stock_prices[:, i] = np.convolve(stock_prices[:, i], np.ones((29,)) / 29, mode='same')
            fig, axs = plt.subplots(2, 1, sharex=True)
            axs[0].plot(stock_prices[-len(stock_prices)*2:-len(stock_prices), i])
            axs[0].plot(stock_prices_filtered[-len(stock_prices)*2:-len(stock_prices), i])
            axs[0].plot(float_stock_prices[-len(stock_prices)*2:-len(stock_prices), i])
            axs[1].plot(np.cumsum(stock_prices[:, i]))
            axs[1].plot(np.cumsum(stock_prices[-len(stock_prices)*2:-len(stock_prices), i]))
            axs[1].plot(np.cumsum(stock_prices_filtered[-len(stock_prices):, i]))
            axs[1].plot(np.cumsum(float_stock_prices[-len(stock_prices)*2:-len(stock_prices), i]))
            plt.legend(['normal', 'restored', 'butterworth', 'floating average'])
            plt.show()
        return stock_prices_filtered

    def _downsample(self, stock_prices, ptype):
        return stock_prices[::self.processor[ptype]["downsampling_rate"], :]

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
        stock_prices = stock_prices.reshape(-1, stock_prices.shape[-1] * stock_prices.shape[-2])
        if self.processor[ptype]["predictor"].latent_dim != stock_prices.shape[-1]:
            # draw latent variable from normal distribution and concatenate it to stock_prices
            latent_variable = torch.randn((1, self.processor[ptype]["predictor"].latent_dim - stock_prices.shape[-1]))
            stock_prices = torch.cat((latent_variable, stock_prices), dim=1)
        return self.processor[ptype]["predictor"](stock_prices).squeeze(2).squeeze(0).permute(1, 0)

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

    if not num_random_actions:
        num_random_actions = agent.replay_buffer_size/4

    try:
        while episode < max_episodes:
            env.hard_reset(random_split=True, split_length=365)  # TODO: Check for other split lengths
            env.set_observation_space(stock_prices=env.stock_data[:env.observation_length])

            done = False
            while not done:

                # if agent is RoboBobo, filter, downsample, scale, encode and predict follow-up stock prices before passing to agent
                if isinstance(agent, RoboBobo):
                    state = list(env.observation_space(normalized=False))
                    # delete last element of state (last stock prices) as it is not used by RoboBobo
                    del state[-1]
                    for key in data_processor.processor.keys():
                        # get downsampling rate if available
                        if data_processor.processor[key]["downsampling_rate"]:
                            downsampling_rate = data_processor.processor[key]["downsampling_rate"]
                        else:
                            downsampling_rate = 1
                        # get start index of observed stock prices by taking maximum of 0 and current time step minus observation length
                        start_idx = max(0, env.t - env.observation_length*downsampling_rate)
                        obs_stock_data = np.zeros((env.observation_length*downsampling_rate, env.stock_data.shape[1]))
                        obs_stock_data[-env.t:, :] = env.stock_data[start_idx:env.t, :]  # TODO: Check if t is correct
                        # win_len=np.min((50, obs_stock_data.shape[0]))
                        # obs_stock_data[:win_len, :] = env.stock_data[:win_len, :]  # TODO: Check if t is correct
                        state.append(data_processor.process(obs_stock_data, ptype=key, flatten=True, mask=0))
                    for i, e in enumerate(state):
                        # flatten state element
                        state[i] = e.flatten().reshape(1, -1).to(agent.device)
                    # concatenate state elements
                    state = torch.cat(state, dim=1)
                else:
                    state = env.observation_space(dtype=torch.Tensor)

                # epsilon-greedy action selection (epsilon-decay)
                if len(agent.replay_buffer) < num_random_actions:
                    # Draw random action
                    action = env.action_space.sample()
                    action[action.abs() < agent.hold_threshold] = 0.
                else:
                    # Draw greedy action
                    action = agent.get_action(state.float())

                # Give chosen action to environment to adjust internal parameters and to compute new state
                next_state, reward, done, _ = env.step(action)

                if isinstance(agent, RoboBobo):
                    next_state = list(next_state)
                    next_state[2] = agent.process_stock_prices(next_state[2])
                    for i, e in enumerate(next_state):
                        # flatten state element
                        next_state[i] = e.flatten().reshape(1, -1).to(agent.device)
                    # concatenate state elements
                    next_state = torch.cat(next_state, dim=1)

                # Append experience to replay buffer
                agent.replay_buffer.push(state, action, reward, next_state, done)

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

def test_sac(env: Environment, agent: SACAgent, plot=True, plot_reference=False):
    """Test trained SAC agent"""
    done = False
    total_equity = []
    agent.eval()

    while not done:
        if isinstance(agent, RoboBobo):
            state = list(env.observation_space(normalized=False))
            state[2] = agent.process_stock_prices(state[2])
            for i, e in enumerate(state):
                # flatten state element
                state[i] = e.flatten().reshape(1, -1).to(agent.device)
            # concatenate state elements
            state = torch.cat(state, dim=1)
        else:
            state = env.observation_space(dtype=torch.Tensor)

        action = agent.get_action(state.float())
        _, _, done, _ = env.step(action)
        total_equity.append(env.total_equity().item())

    print("Test scenario -- final total equity: {}".format(env.total_equity().item()))

    if plot:
        plt.plot(total_equity)
        plt.plot(np.convolve(total_equity, np.ones(10) / 10, mode='valid'))
        plt.ylabel('Total equity [$]')
        plt.xlabel('Time step')
        plt.title(f"Total final equity in [$] (Grow: {total_equity[-1]/total_equity[0]:.2f})")
        plt.legend(['Total equity [$]', 'Avg. total final equity [$]'])
        plt.show()

    if plot_reference:
        # plot the average of all stock prices
        avg = torch.mean(env.stock_data, dim=1)
        plt.plot(avg)
        plt.ylabel('Average stock price [$]')
        plt.xlabel('Time step')
        plt.title(f"Avg stock price in [$] (Grow: {avg[-1]/avg[0]:.2f})")
        plt.show()
