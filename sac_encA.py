import copy
import os
import random
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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
            state_batch.append(copy.deepcopy(state))
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(copy.deepcopy(next_state))
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

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

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
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        """Processes state and corresponding action through the network layers and outputs the Q-value"""
        x = torch.cat([state, action], 2)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        x = self.linearout(x)
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
        self.policy_criterion = nn.L1Loss()

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
        z = normal.sample()
        action = (torch.tanh(mean + std * z.to(self.device)) + 1) / 2
        action = torch.clamp(action, -1., 1.)
        log_prob = Normal(mean, std).log_prob(mean + std * z.to(self.device)) - torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(dim=2).unsqueeze(-1)
        return action, log_prob, z, mean, log_std

    def get_action(self, state, training=True, hold=True):
        """Is used to compute an action during experience gathering and testing"""
        state = state.unsqueeze(0).to(self.device)  #.transpose(1, 2)
        mean, log_std = self.policy_net.forward(state)
        log_std = torch.clamp(log_std, self.policy_net.log_std_min, self.policy_net.log_std_max)
        if training:
            # deviation-value is only during training > 0 to encourage exploration
            # During testing the deviation = 0 since exploration is not needed anymore
            std = log_std.exp()
        else:
            std = torch.tensor(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)

        # Generate random value for randomized action-selection
        normal = Normal(0, 1)
        z = normal.sample().to(self.device)

        # Draw action by applying scaled tanh-function
        action = (torch.tanh(mean + std * z) + 1) / 2  # Only positive values! -> +1 and /2

        action = action.detach().cpu()

        # softmax over all positive (buy) actions to make sure not to spend more than 100% of the cash
        if action[0, 0][action[0, 0] > 0].sum() > 1:
            action_pos = np.where(action[0, 0] > 0)[0]
            action[0, 0, action_pos] = self.env.action_space.softmax(action[0, 0, action_pos])

        if len(state.size()) == 3:
            action = action[0, 0]
        else:
            action = action[0, 0, 0]

        if hold:
            # if one action is smaller than the hold threshold, this action is set to 0
            action[torch.abs(action) < self.hold_threshold] = 0

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

        # Update target value network by polyak-averaging
        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )

        # delete all computation graphs to free up memory
        # del state, next_state, action, reward, done, predicted_q_value1, predicted_q_value2, predicted_value, new_action, log_prob, epsilon, mean, log_std, target_value, target_q_value, q_value_loss1, q_value_loss2, predicted_new_q_value, target_value_func, value_loss, policy_loss

    def online_update_value_net(self):
        # TODO: ONLINE TRAINING VALUE NETWORK
        # Online training for critic to adapt to the actor, since actor is drawn from a normal distribution
        # Update value network after n time steps
        # Draw batch of samples from replay buffer
        # Goal: Maximise current steps focused
        # Target value: 1 (curr_steps_focused/passed session time)
        # Actual value: 0-1 (normalized curr_steps_focused)
        pass

    def online_update_policy_net(self):
        # TODO: ONLINE TRAINING POLICY NETWORK
        # Online training for actor to take critic's state value into account
        # Update policy network in each time step
        # Use critic's state value as actual value
        # What could be the reference/target value?
        #   --> Maybe: Critic Value + 1 --> Always try to achieve something better
        pass

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
            'reward': reward,
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


class RoboBobo(SACAgent):
    def __init__(self, env: Environment, temperature=1.,
                 generator=None, discriminator=None, autoencoder=None, scaler=None,
                 hidden_dim=256, hold_threshold=1e-3, replay_buffer_size=1e6):
        super(RoboBobo, self).__init__(env, temperature=temperature, hidden_dim=hidden_dim,
                                       hold_threshold=hold_threshold, replay_buffer_size=replay_buffer_size)

        # initialize generator, discriminator and autoencoder
        self.generator = generator.to(self.device)
        self.discriminator = discriminator.to(self.device)
        self.autoencoder = autoencoder.to(self.device)

        # initialize scaler
        self.scaler = scaler

        # modify action space according to autoencoder
        self.action_dim = self.autoencoder.output_dim

        # modify state dim according to autoencoder:
        #  state_dim =
        #       current cash (float scalar)
        #  +    historic and current REAL portofolio value (matrix: (observation_length x num_stocks))
        #  +    historic, current and predicted ENCODED stock prices (matrix: (observation_length + seq_len) x encoder.output_dim)
        #  +    validation of prediction (scalar) TODO: turn scalar into vector of validation score for each feature
        self.state_dim = 1 \
                         + self.autoencoder.input_dim \
                         + self.autoencoder.output_dim*(self.env.observation_length + self.generator.seq_len) \
                         # + self.discriminator.n_classes

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

    def process_stock_prices(self, stock_prices):
        """process stock prices
        :param stock_prices: (tensor) stock prices with shape (observation_length, features)"""
        # differentiate stock prices
        stock_prices = torch.diff(stock_prices, dim=0)
        stock_prices = torch.cat((stock_prices[0, :].reshape(1, -1), stock_prices), dim=0)
        # standardize stock prices
        stock_prices = torch.FloatTensor(self.scaler.transform(stock_prices)).unsqueeze(0).to(self.device)
        # encode state
        stock_prices = self.encode(stock_prices)
        # predict follow-up stock prices
        predicted = self.predict(stock_prices.reshape(-1, stock_prices.shape[1]*stock_prices.shape[2])).squeeze(2).transpose(2, 1)
        stock_prices = torch.cat((stock_prices, predicted), dim=1)
        # validate prediction
        # validation = self.validate(self.decode(stock_prices).unsqueeze(2).transpose(3, 1))
        # stock_prices = torch.cat((stock_prices, validation), dim=0)

        return stock_prices.squeeze(0)

    def predict(self, state):
        """predictions on most probable follow-up states regardless of action"""
        return self.generator.generate(state)

    def validate(self, state):
        """validation of the predicted follow-up states"""
        return self.discriminator(state)

    def encode(self, state):
        """encoding of the state"""
        return self.autoencoder.encode(state)

    def decode(self, state):
        """decoding of the state"""
        return self.autoencoder.decode(state)

    def evaluate(self, state, epsilon=1e-6):
        """Is used during training to determine the entropy H(X)"""
        mean_enc, log_std_enc = self.policy_net.forward(state)

        # use autoencoder to decode mean and log_std
        mean = self.decode(mean_enc)
        log_std = self.decode(log_std_enc)
        log_std = torch.clamp(log_std, self.policy_net.log_std_min, self.policy_net.log_std_max)

        std = log_std.exp()

        normal = Normal(0, 1)
        z = normal.sample()
        action = (torch.tanh(mean + std * z.to(self.device)) + 1) / 2
        action = torch.clamp(action, -1., 1.)
        log_prob = Normal(mean, std).log_prob(mean + std * z.to(self.device)) - torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(dim=2).unsqueeze(-1)
        return action, log_prob, z, mean, log_std, mean_enc, log_std_enc

    def get_action(self, state, training=True, hold=True):
        """Is used to compute an action during experience gathering and testing"""
        state = state.unsqueeze(0).to(self.device)  #.transpose(1, 2)
        mean_enc, log_std_enc = self.policy_net.forward(state)

        # use autoencoder to decode mean and log_std
        mean = self.decode(mean_enc)
        log_std = self.decode(log_std_enc)
        log_std = torch.clamp(log_std, self.policy_net.log_std_min, self.policy_net.log_std_max)

        if training:
            # deviation-value is only during training > 0 to encourage exploration
            # During testing the deviation = 0 since exploration is not needed anymore
            std = log_std.exp()
        else:
            std = torch.tensor(0).reshape(1, 1, 1).to(self.device)

        # Generate random value for randomized action-selection
        normal = Normal(0, 1)
        z = normal.sample().to(self.device)

        # Draw action by applying scaled tanh-function
        action = (torch.tanh(mean + std * z) + 1) / 2  # Only positive values! -> +1 and /2

        action = action.detach().cpu()

        # softmax over all positive (buy) actions to make sure not to spend more than 100% of the cash
        if action[0, 0][action[0, 0] > 0].sum() > 1:
            action_pos = np.where(action[0, 0] > 0)[0]
            action[0, 0, action_pos] = self.env.action_space.softmax(action[0, 0, action_pos])

        if len(state.size()) == 3:
            action = action[0, 0]
        else:
            action = action[0, 0, 0]

        if hold:
            # if one action is smaller than the hold threshold, this action is set to 0
            action[torch.abs(action) < self.hold_threshold] = 0

        return action, mean, log_std


def mini_batch_train(env: Environment, agent: SACAgent,
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

    if not num_random_actions:
        num_random_actions = agent.replay_buffer_size/4

    try:
        while episode < max_episodes:
            env.hard_reset(random_split=True, split_length=360)  # TODO: Check for other split lengths
            env.set_observation_space(stock_prices=env.stock_data[:env.observation_length])
            done = False
            while not done:
                # if agent is RoboBobo, encode, predict and validate follow-up stock prices before passing to agent
                if isinstance(agent, RoboBobo):
                    state = list(env.observation_space(normalized=False))
                    state[2] = agent.process_stock_prices(state[2])
                    for i, e in enumerate(state):
                        # flatten state element
                        state[i] = e.flatten().reshape(1, -1)
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
                        next_state[i] = e.flatten().reshape(1, -1)
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
            print("Episode: {} -- time steps: {} (total: {}) -- total equity: {} -- total equity per time: {}".format(episode + 1, env.t, agent.num_actions, np.round(env.total_equity().item(), 2), np.round(env.total_equity().item()/len(env.stock_data), 2)))
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


def test_sac(env: Environment, agent: SACAgent, plot=True):
    """Test trained SAC agent"""
    done = False
    total_equity = []
    while not done:
        if isinstance(agent, RoboBobo):
            state = list(env.observation_space(normalized=False))
            state[2] = agent.process_stock_prices(state[2])
            for i, e in enumerate(state):
                # flatten state element
                state[i] = e.flatten().reshape(1, -1)
            # concatenate state elements
            state = torch.cat(state, dim=1)
        else:
            state = env.observation_space(dtype=torch.Tensor)

        action = agent.get_action(state.float(), training=False)
        _, _, done, _ = env.step(action)
        total_equity.append(env.total_equity().item())

    print("Test scenario -- final total equity: {}".format(env.total_equity().item()))

    if plot:
        plt.plot(total_equity)
        plt.plot(np.convolve(total_equity, np.ones(10) / 10, mode='valid'))
        plt.ylabel('Total equity [$]')
        plt.xlabel('Time step')
        plt.title('Total final equity over time steps in [$]')
        plt.legend(['Total equity [$]', 'Avg. total final equity [$]'])
        plt.show()

