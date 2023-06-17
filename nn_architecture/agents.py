import copy
import random
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

from nn_architecture.rl_networks import ValueNetwork, SoftQNetwork, PolicyNetwork, ValueRoboboboNetwork, \
    SoftQRoboboboNetwork, PolicyRoboboboNetwork, PolicySubNetwork, SoftPolicyNetwork


class ReplayBuffer:
    """Class of the replay buffer for experience replay"""

    def __init__(self, capacity=1e6):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        """push new experience into buffer"""
        experience = (state, action, reward, next_state, done)
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

        return np.concatenate(state_batch, dtype=np.float32), \
               np.concatenate(action_batch, dtype=np.float32), \
               np.concatenate(reward_batch, dtype=np.float32), \
               np.concatenate(next_state_batch, dtype=np.float32), \
               np.concatenate(done_batch, dtype=np.float32)

    def __len__(self):
        return len(self.buffer)


class OrnsteinUhlenbeckActionNoise:
    def __init__(self, action_dim, mu=0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.X = np.ones(self.action_dim) * self.mu

    def reset(self):
        self.X = np.ones(self.action_dim) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.X)
        dx = dx + self.sigma * np.random.randn(len(self.X))
        self.X = self.X + dx
        return self.X


class NormalDistributionActionNoise:
    def __init__(self, action_dim, mu=0, sigma=0.2, clip=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.sigma = sigma
        self.clip = torch.tensor(clip)
        self.dist = Normal(self.mu, self.sigma)

    def sample(self):
        # return torch.clip(np.random.normal(self.mu, self.sigma, self.action_dim), -self.clip, self.clip)
        return torch.clip(self.dist.sample(), -self.clip, self.clip)


class Agent:
    device = torch.device("cpu")
    replay_buffer = ReplayBuffer()
    num_actions = 0

    def __init__(self):
        pass

    def get_action_exploitation(self, state):
        raise NotImplementedError

    def get_action_exploration(self, state):
        raise NotImplementedError

    def update(self, batch_size):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def eval(self):
        raise NotImplementedError

    def save_checkpoint(self, path):
        raise NotImplementedError

    def load_checkpoint(self, path):
        raise NotImplementedError


class SACAgent(Agent):
    """Class of the SAC-agent. Initializes the different networks, action-selection-mechanism and update-paradigm"""

    num_steps = 0
    state_values = []
    training = False
    policy_state = torch.Tensor

    def __init__(self, temperature=1., state_dim=None, action_dim=None, hidden_dim=256, num_layers=3,
                 learning_rate=1e-4, init_w=3e-3, hold_threshold=1e-2, replay_buffer_size=1e6,
                 limit_high=None, limit_low=None):
        """Initializes the networks, determines the availability of cuda
        and initializes the replay buffer and the optimizer.
        """
        super(SACAgent, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.temperature = temperature

        # self.env = NormalizedActions(env)
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.limit_high = torch.tensor(limit_high).to(self.device)
        self.limit_low = torch.tensor(limit_low).to(self.device)

        # initialize SAC networks
        self.value_net = ValueNetwork(self.state_dim, self.hidden_dim,
                                      num_layers=num_layers, init_w=init_w).to(self.device)
        self.target_value_net = ValueNetwork(self.state_dim, self.hidden_dim,
                                             num_layers=num_layers, init_w=init_w).to(self.device)
        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(param.data)

        self.soft_q_net1 = SoftQNetwork(self.state_dim, self.action_dim, self.hidden_dim,
                                        num_layers=num_layers, init_w=init_w).to(self.device)
        self.soft_q_net2 = SoftQNetwork(self.state_dim, self.action_dim, self.hidden_dim,
                                        num_layers=num_layers, init_w=init_w).to(self.device)

        self.policy_net = SoftPolicyNetwork(num_inputs=self.state_dim, num_actions=self.action_dim,
                                            hidden_dim=self.hidden_dim, num_layers=num_layers,
                                            init_w=3e-3, log_std_min=-20, log_std_max=2).to(self.device)

        # Initializes the networks' cost-function, optimizer and learning rates
        self.value_criterion = nn.MSELoss()
        self.soft_q_criterion1 = nn.MSELoss()
        self.soft_q_criterion2 = nn.MSELoss()
        self.policy_criterion = nn.L1Loss()

        self.value_lr = learning_rate
        self.soft_q_lr = learning_rate
        self.policy_lr = learning_rate

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

    def get_action_exploration(self, state, log_prob=False, epsilon=1e-6):
        """Is used to compute an action during experience gathering and testing"""
        if self.policy_state.__name__ == list.__name__ and not isinstance(state, self.policy_state):
            state = self.state_tensor_to_list(state)
        mean, log_std = self.policy_net.forward(state)

        log_std = torch.clamp(log_std, self.policy_net.log_std_min, self.policy_net.log_std_max)
        std = log_std.exp()

        # Generate random value for randomized action-selection
        normal = Normal(0, 1)
        z = normal.sample(std.shape).to(self.device)

        # Draw action by applying scaled tanh-function
        action = mean + std * z
        if self.limit_low is not None:
            action = torch.clamp(action, min=self.limit_low)
        if self.limit_high is not None:
            action = torch.clamp(action, max=self.limit_high)

        # if hold:
        #     # if one action is smaller than the hold threshold, this action is set to 0
        #     action[torch.abs(action) < self.hold_threshold] = 0

        # softmax over all positive (buy) actions to make sure not to spend more than 100% of the cash
        # if action[action > 0].sum() > 1:
        #     action[action > 0] = self.env.action_space.softmax(action[action > 0])

        if log_prob:
            log_prob = Normal(mean, std).log_prob(mean + std * z) - torch.log(1 - action.pow(2) + epsilon)
            return mean, log_prob.mean(dim=-1).unsqueeze(dim=-1)  # TODO: check if mean is correct
        else:
            return mean

    def get_action_exploitation(self, state):
        if self.policy_state.__name__ == list.__name__ and not isinstance(state, self.policy_state):
            state = self.state_tensor_to_list(state)
        action, _ = self.policy_net.forward(state)[0]
        if self.limit_low is not None:
            action = torch.clamp(action, min=self.limit_low)
        if self.limit_high is not None:
            action = torch.clamp(action, max=self.limit_high)
        return action

    def update(self, batch_size, gamma=0.99, soft_tau=1e-1):
        """Update-paradigm"""

        # Draw experience from replay buffer
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)

        state = torch.from_numpy(state).float().to(self.device)
        next_state = torch.from_numpy(next_state).float().to(self.device)
        action = torch.from_numpy(action).float().to(self.device)
        reward = torch.from_numpy(reward).float().to(self.device)
        done = torch.from_numpy(done).float().to(self.device)

        # Get all values, which are necessary for the network updates
        new_action, log_prob = self.get_action_exploration(state, log_prob=True)
        predicted_q_value1 = self.soft_q_net1(state, action)
        predicted_q_value2 = self.soft_q_net2(state, action)
        predicted_value = self.value_net(state)

        # Training Q Function
        # Compute target Q-value by taking action-dependent reward into account
        target_value = self.target_value_net(next_state)
        target_q_value = reward + (1 - done) * gamma * target_value

        # Compute loss
        q_value_loss1 = self.soft_q_criterion1(predicted_q_value1, target_q_value.detach().clone())
        self.soft_q_optimizer1.zero_grad()
        q_value_loss1.backward()
        self.soft_q_optimizer1.step()

        q_value_loss2 = self.soft_q_criterion2(predicted_q_value2, target_q_value.detach().clone())
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

    def save_checkpoint(self, path):
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
            state[i] = torch.tensor(e.flatten().reshape(1, -1))
        # concatenate state elements
        state = torch.cat(state, dim=1)
        return state

    def state_tensor_to_list(self, state):
        # transform tensor from shape (batch_size, obs_dim) to list of size (obs)
        # where each entry has shape (batch_size, self.obs_dims[entry, 0], self.obs_dim[entry, 1])
        state_list = []
        for i, shape in enumerate(self.obs_dims):
            state_list.append(state[:, :np.product(shape)].reshape([state.shape[0]] + list(shape)).float())
            state = state[:, np.product(shape):]
        return state_list


class RoboBoboSAC(SACAgent):
    def __init__(self,
                 temperature=1.,
                 state_dim=None,
                 action_dim=None,
                 hidden_dim=256,
                 num_layers=3,
                 learning_rate=1e-4,
                 init_w=3e-3,
                 hold_threshold=1e-3,
                 replay_buffer_size=1e6):
        super(RoboBoboSAC, self).__init__(temperature=temperature,
                                          state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim,
                                          num_layers=3,
                                          learning_rate=learning_rate, hold_threshold=hold_threshold, init_w=init_w,
                                          replay_buffer_size=replay_buffer_size)

        self.policy_state = list
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.init_w = init_w

        # self.state_dim = 1 + env.observation_space.num_stocks

        # modify state dim according to autoencoder:
        #  state_dim =
        #       current cash (float scalar)
        #  +    historic and current REAL portofolio value (matrix: (observation_length x num_stocks))
        #  +    historic, current and predicted ENCODED stock prices (matrix: (observation_length + seq_len) x encoder.output_dim)
        #  +    validation of prediction (scalar) TODO: turn scalar into vector of validation score for each feature

    def adjust_nets(self, policy_sub_networks: Optional[nn.ModuleList] = None):
        # initialize SAC networks
        self.value_net = ValueRoboboboNetwork(self.state_dim, self.hidden_dim, num_layers=self.num_layers,
                                              init_w=self.init_w,
                                              policy_sub_networks=copy.deepcopy(policy_sub_networks)).to(self.device)
        self.target_value_net = ValueRoboboboNetwork(self.state_dim, self.hidden_dim, num_layers=self.num_layers,
                                                     init_w=self.init_w,
                                                     policy_sub_networks=copy.deepcopy(policy_sub_networks)).to(
            self.device)
        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(param.data)

        self.soft_q_net1 = SoftQRoboboboNetwork(self.state_dim, self.action_dim, self.hidden_dim,
                                                num_layers=self.num_layers, init_w=self.init_w,
                                                policy_sub_networks=copy.deepcopy(policy_sub_networks)).to(self.device)
        self.soft_q_net2 = SoftQRoboboboNetwork(self.state_dim, self.action_dim, self.hidden_dim,
                                                num_layers=self.num_layers, init_w=self.init_w,
                                                policy_sub_networks=copy.deepcopy(policy_sub_networks)).to(self.device)

        self.policy_net = PolicyRoboboboNetwork(self.state_dim, self.action_dim, self.hidden_dim,
                                                num_layers=self.num_layers,
                                                init_w=3e-3, log_std_min=-20, log_std_max=2,
                                                policy_sub_networks=copy.deepcopy(policy_sub_networks)).to(self.device)

        # Initializes the networks' cost-function, optimizer and learning rates
        self.value_criterion = nn.MSELoss()
        self.soft_q_criterion1 = nn.MSELoss()
        self.soft_q_criterion2 = nn.MSELoss()
        self.policy_criterion = nn.L1Loss()

        self.value_lr = self.learning_rate
        self.soft_q_lr = self.learning_rate
        self.policy_lr = self.learning_rate

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


class DDPGAgent(Agent):
    num_steps = 0
    state_values = []
    training = False
    policy_state = torch.Tensor

    def __init__(self,
                 state_dim,
                 action_dim,
                 hidden_dim=256,
                 num_layers=3,
                 learning_rate=1e-4,
                 init_w=3e-3,
                 replay_buffer_size=1000000,
                 limit_high=None,
                 limit_low=None,
                 std_noise=0.2,
                 clip_noise=0.2):
        """Initializes the networks, determines the availability of cuda
        and initializes the replay buffer and the optimizer.
        """
        super(DDPGAgent, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # self.env = NormalizedActions(env)
        # self.env = env
        self.action_dim = action_dim  # if action_dim else self.env.action_space.shape[0]
        self.state_dim = state_dim  # if state_dim else self.env.observation_space.dim
        self.hidden_dim = hidden_dim
        self.noise = NormalDistributionActionNoise(self.action_dim, sigma=std_noise, clip=clip_noise)
        self.limit_high = torch.tensor(limit_high).to(self.device)
        self.limit_low = torch.tensor(limit_low).to(self.device)

        # initialize AC network
        self.critic = SoftQNetwork(self.state_dim, self.action_dim, self.hidden_dim, num_layers=num_layers, init_w=init_w).to(self.device)
        self.target_critic = SoftQNetwork(self.state_dim, self.action_dim, self.hidden_dim, num_layers=num_layers, init_w=init_w).to(self.device)
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        self.actor = PolicyNetwork(self.state_dim, self.action_dim, self.hidden_dim, num_layers=num_layers, init_w=3e-3).to(self.device)
        self.target_actor = PolicyNetwork(self.state_dim, self.action_dim, self.hidden_dim, num_layers=num_layers, init_w=3e-3).to(self.device)
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        # Initializes the networks' cost-function, optimizer and learning rates
        self.critic_loss = nn.MSELoss()
        self.actor_loss = nn.L1Loss()

        self.critic_lr = learning_rate
        self.actor_lr = learning_rate

        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)

        # Initializes the replay buffer within the agent
        self.replay_buffer_size = replay_buffer_size
        self.replay_buffer = ReplayBuffer(self.replay_buffer_size)

        self.num_actions = 0
        self.obs_dims = []  # for self.state_tensor_to_list; will be written at first use of self.state_list_to_tensor

    def get_action_exploitation(self, state):
        """Is used to compute an action during experience gathering and testing"""
        if self.policy_state.__name__ == list.__name__ and not isinstance(state, self.policy_state):
            state = self.state_tensor_to_list(state.float())
        action = self.target_actor.forward(state)
        if self.limit_low is not None:
            action = torch.clamp(action, min=self.limit_low)
        if self.limit_high is not None:
            action = torch.clamp(action, max=self.limit_high)
        return action

    def get_action_exploration(self, state):
        """Is used to compute an action during experience gathering and testing"""
        if self.policy_state.__name__ == list.__name__ and not isinstance(state, self.policy_state):
            state = self.state_tensor_to_list(state)
        action = self.actor.forward(state.float())
        action += self.noise.sample().to(self.device)
        if self.limit_low is not None:
            action = torch.clamp(action, min=self.limit_low)
        if self.limit_high is not None:
            action = torch.clamp(action, max=self.limit_high)
        return action

    def update(self, batch_size, gamma=0.99, soft_tau=1e-1):
        # Draw experience from replay buffer
        s1, a1, r1, s2, done = self.replay_buffer.sample(batch_size)

        s1 = torch.from_numpy(s1).float().to(self.device)
        s2 = torch.from_numpy(s2).float().to(self.device)
        a1 = torch.from_numpy(a1).float().to(self.device)
        r1 = torch.from_numpy(r1).float().to(self.device)
        done = torch.from_numpy(done).float().to(self.device)

        # ---------------------- optimize critic ----------------------

        a2 = self.get_action_exploitation(s1).detach()
        next_val = torch.squeeze(self.target_critic.forward(s2, a2).detach())
        y_expected = r1 + gamma * next_val * (1 - done)
        y_predicted = torch.squeeze(self.critic.forward(s1, a1))
        # compute critic loss
        critic_loss = self.critic_loss(y_predicted, y_expected)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------- optimize actor ----------------------

        pred_a1 = self.get_action_exploration(s1)
        actor_loss = -self.critic.forward(s1, pred_a1).mean()  # TODO: target_critic or critic?? try also mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ---------------------- update target networks ----------------------
        #
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - soft_tau) + param.data * soft_tau)

        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - soft_tau) + param.data * soft_tau)

    def train(self):
        self.training = True
        self.actor.train()
        self.target_actor.train()
        self.critic.train()
        self.target_critic.train()

    def eval(self):
        self.training = False
        self.actor.eval()
        self.target_actor.eval()
        self.critic.eval()
        self.target_critic.eval()

    def save_checkpoint(self, path):
        sac_dict = {
            'critic': self.critic.state_dict(),
            'target_critic': self.target_critic.state_dict(),
            'actor': self.actor.state_dict(),
            'target_actor': self.target_actor.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'action_dim': self.action_dim,
            'state_dim': self.state_dim,
            'hidden_dim': self.hidden_dim,
            'num_actions': self.num_actions,
        }

        torch.save(sac_dict, path)

    def load_checkpoint(self, path):
        sac_dict = torch.load(path, map_location=self.device)
        self.critic.load_state_dict(sac_dict['critic'])
        self.target_critic.load_state_dict(sac_dict['target_critic'])
        self.actor.load_state_dict(sac_dict['actor'])
        self.target_actor.load_state_dict(sac_dict['target_actor'])
        self.actor_optimizer.load_state_dict(sac_dict['actor_optimizer'])
        self.critic_optimizer.load_state_dict(sac_dict['critic_optimizer'])
        self.num_actions = sac_dict['num_actions']
        print("Loaded checkpoint from path: {}".format(path))

    def state_list_to_tensor(self, state):
        state = list(state)
        if len(self.obs_dims) == 0:
            # each entry will be the sequence length of each entry
            self.obs_dims = tuple([tuple(x.shape) for x in state])
        for i, e in enumerate(state):
            # flatten state element
            state[i] = torch.tensor(e.flatten().reshape(1, -1))
        # concatenate state elements
        state = torch.cat(state, dim=1)
        return state

    def state_tensor_to_list(self, state):
        # transform tensor from shape (batch_size, obs_dim) to list of size (obs)
        # where each entry has shape (batch_size, self.obs_dims[entry, 0], self.obs_dim[entry, 1])
        state_list = []
        for i, shape in enumerate(self.obs_dims):
            state_list.append(state[:, :np.product(shape)].reshape([state.shape[0]] + list(shape)).float())
            state = state[:, np.product(shape):]
        return state_list


class RoboBoboDDPG(DDPGAgent):
    def __init__(self,
                 state_dim=None,
                 action_dim=None,
                 hidden_dim=256,
                 num_layers=3,
                 learning_rate=1e-4,
                 init_w=3e-3,
                 hold_threshold=1e-3,
                 replay_buffer_size=1e6):
        super(RoboBoboDDPG, self).__init__(state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim,
                                           num_layers=3, learning_rate=learning_rate,
                                           init_w=init_w, replay_buffer_size=replay_buffer_size)

        self.policy_state = list
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.init_w = init_w
        self.hold_threshold = hold_threshold

        # self.state_dim = 1 + env.observation_space.num_stocks

        # modify state dim according to autoencoder:
        #  state_dim =
        #       current cash (float scalar)
        #  +    historic and current REAL portofolio value (matrix: (observation_length x num_stocks))
        #  +    historic, current and predicted ENCODED stock prices (matrix: (observation_length + seq_len) x encoder.output_dim)
        #  +    validation of prediction (scalar) TODO: turn scalar into vector of validation score for each feature

    def adjust_nets(self, policy_sub_networks: Optional[nn.ModuleList] = None):
        # initialize SAC networks
        self.critic = SoftQRoboboboNetwork(self.state_dim, self.action_dim, self.hidden_dim,
                                           num_layers=self.num_layers, init_w=self.init_w,
                                           policy_sub_networks=copy.deepcopy(policy_sub_networks)).to(self.device)
        self.target_critic = SoftQRoboboboNetwork(self.state_dim, self.action_dim, self.hidden_dim,
                                                  num_layers=self.num_layers, init_w=self.init_w,
                                                  policy_sub_networks=copy.deepcopy(policy_sub_networks)).to(
            self.device)
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        self.actor = PolicyRoboboboNetwork(self.state_dim, self.action_dim, self.hidden_dim, num_layers=self.num_layers,
                                           init_w=3e-3, log_std_min=-20, log_std_max=2,
                                           policy_sub_networks=copy.deepcopy(policy_sub_networks)).to(self.device)

        self.target_actor = PolicyRoboboboNetwork(self.state_dim, self.action_dim, self.hidden_dim,
                                                  num_layers=self.num_layers,
                                                  init_w=3e-3, log_std_min=-20, log_std_max=2,
                                                  policy_sub_networks=copy.deepcopy(policy_sub_networks)).to(
            self.device)

        # Initializes the networks' cost-function, optimizer and learning rates
        self.critic_loss = nn.MSELoss()
        self.actor_loss = nn.L1Loss()

        self.critic_lr = self.learning_rate
        self.actor_lr = self.learning_rate

        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)

    def set_state_dim(self, num_states: int):
        """sets the dimension of the state"""
        self.state_dim = num_states

    def set_action_dim(self, num_actions: int):
        """sets the dimension of the action space"""
        self.action_dim = num_actions

    def create_policy_sub_network(self, input_dim, hidden_dim, seq_len=1, lstm=False, num_layers=2, dropout=.1):
        return PolicySubNetwork(input_dim, hidden_dim, seq_len=seq_len, lstm=lstm, num_layers=num_layers,
                                dropout=dropout).to(self.device)


class TD3Agent(Agent):
    num_steps = 0
    state_values = []
    training = False
    policy_state = torch.Tensor

    def __init__(self,
                 state_dim,
                 action_dim,
                 hidden_dim=256,
                 num_layers=3,
                 learning_rate=1e-4,
                 init_w=3e-3,
                 replay_buffer_size=1e6,
                 limit_high=None,
                 limit_low=None,
                 updates_per_step=1,
                 delay=2,
                 std_noise=0.2,
                 clip_noise=0.2):
        """Initializes the networks, determines the availability of cuda
        and initializes the replay buffer and the optimizer.
        """
        super(TD3Agent, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # self.env = NormalizedActions(env)
        # self.env = env
        self.action_dim = action_dim  # if action_dim else self.env.action_space.shape[0]
        self.state_dim = state_dim  # if state_dim else self.env.observation_space.dim
        self.hidden_dim = hidden_dim
        self.noise = NormalDistributionActionNoise(self.action_dim, sigma=std_noise, clip=clip_noise)
        self.limit_high = torch.tensor(limit_high).to(self.device)
        self.limit_low = torch.tensor(limit_low).to(self.device)
        self.delay = delay
        self.updates_per_step = updates_per_step

        # initialize AC network
        self.critic1 = SoftQNetwork(self.state_dim, self.action_dim, self.hidden_dim, num_layers=num_layers, init_w=init_w).to(self.device)
        self.target_critic1 = SoftQNetwork(self.state_dim, self.action_dim, self.hidden_dim, num_layers=num_layers, init_w=init_w).to(self.device)
        for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            target_param.data.copy_(param.data)

        self.critic2 = SoftQNetwork(self.state_dim, self.action_dim, self.hidden_dim, num_layers=num_layers, init_w=init_w).to(self.device)
        self.target_critic2 = SoftQNetwork(self.state_dim, self.action_dim, self.hidden_dim, num_layers=num_layers, init_w=init_w).to(self.device)
        for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(param.data)

        self.actor = PolicyNetwork(self.state_dim, self.action_dim, self.hidden_dim, num_layers=num_layers, init_w=3e-3).to(self.device)
        self.target_actor = PolicyNetwork(self.state_dim, self.action_dim, self.hidden_dim, num_layers=num_layers, init_w=3e-3).to(self.device)
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        # Initializes the networks' cost-function, optimizer and learning rates
        self.critic_loss = nn.MSELoss()

        self.critic_lr = learning_rate
        self.actor_lr = learning_rate

        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=self.critic_lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=self.critic_lr)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)

        # Initializes the replay buffer within the agent
        self.replay_buffer_size = replay_buffer_size
        self.replay_buffer = ReplayBuffer(self.replay_buffer_size)

        self.num_actions = 0
        self.obs_dims = []  # for self.state_tensor_to_list; will be written at first use of self.state_list_to_tensor

    def get_action_exploitation(self, state, target=True):
        """Is used to compute an action during experience gathering and testing"""
        if self.policy_state.__name__ == list.__name__ and not isinstance(state, self.policy_state):
            state = self.state_tensor_to_list(state.float())
        if target:
            action = self.target_actor.forward(state)
        else:
            action = self.actor.forward(state)
        if self.limit_low is not None:
            action = torch.clamp(action, min=self.limit_low)
        if self.limit_high is not None:
            action = torch.clamp(action, max=self.limit_high)
        return action

    def get_action_exploration(self, state, target=False):
        """Is used to compute an action during experience gathering and testing"""
        if self.policy_state.__name__ == list.__name__ and not isinstance(state, self.policy_state):
            state = self.state_tensor_to_list(state)
        if not target:
            action = self.actor.forward(state.float())
        else:
            action = self.target_actor.forward(state.float())
        action += self.noise.sample().to(self.device)
        if self.limit_low is not None:
            action = torch.clamp(action, min=self.limit_low)
        if self.limit_high is not None:
            action = torch.clamp(action, max=self.limit_high)
        return action

    def update(self, batch_size, gamma=0.99, soft_tau=1e-1):
        for i in range(self.updates_per_step):
            # Draw experience from replay buffer
            s1, a1, r1, s2, done = self.replay_buffer.sample(batch_size)

            s1 = torch.from_numpy(s1).float().to(self.device)
            s2 = torch.from_numpy(s2).float().to(self.device)
            a1 = torch.from_numpy(a1).float().to(self.device)
            r1 = torch.from_numpy(r1).float().to(self.device)
            done = torch.from_numpy(done).float().to(self.device)

            # ---------------------- optimize critic ----------------------

            # compute (target) Q values
            self.target_actor.eval()
            a2 = self.get_action_exploration(s1, target=True).detach()
            next_val = torch.min(torch.squeeze(self.target_critic1.forward(s2, a2).detach()), torch.squeeze(self.target_critic2.forward(s2, a2).detach()))
            y_expected = r1 + gamma * next_val * (1 - done)
            y_predicted1 = torch.squeeze(self.critic1.forward(s1, a1))
            y_predicted2 = torch.squeeze(self.critic2.forward(s1, a1))

            # compute critic loss
            critic1_loss = self.critic_loss(y_predicted1, y_expected)
            self.critic1_optimizer.zero_grad()
            critic1_loss.backward()
            self.critic1_optimizer.step()
            critic2_loss = self.critic_loss(y_predicted2, y_expected)
            self.critic2_optimizer.zero_grad()
            critic2_loss.backward()
            self.critic2_optimizer.step()

            if i % self.delay == 0:
                # ---------------------- optimize actor ----------------------

                pred_a1 = self.get_action_exploration(s1, target=False)
                actor_loss = -self.critic1.forward(s1, pred_a1).mean()
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # ---------------------- update target networks ----------------------

                for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
                    target_param.data.copy_(target_param.data * (1.0 - soft_tau) + param.data * soft_tau)

                for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
                    target_param.data.copy_(target_param.data * (1.0 - soft_tau) + param.data * soft_tau)

                for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
                    target_param.data.copy_(target_param.data * (1.0 - soft_tau) + param.data * soft_tau)

    def train(self):
        self.training = True
        self.actor.train()
        self.target_actor.train()
        self.critic1.train()
        self.target_critic1.train()
        self.critic2.train()
        self.target_critic2.train()

    def eval(self):
        self.training = False
        self.actor.eval()
        self.target_actor.eval()
        self.critic1.eval()
        self.target_critic1.eval()
        self.critic1.eval()
        self.target_critic1.eval()

    def save_checkpoint(self, path):
        sac_dict = {
            'critic1': self.critic1.state_dict(),
            'target_critic1': self.target_critic1.state_dict(),
            'critic2': self.critic2.state_dict(),
            'target_critic2': self.target_critic2.state_dict(),
            'actor': self.actor.state_dict(),
            'target_actor': self.target_actor.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic1_optimizer': self.critic1_optimizer.state_dict(),
            'critic2_optimizer': self.critic2_optimizer.state_dict(),
            'action_dim': self.action_dim,
            'state_dim': self.state_dim,
            'hidden_dim': self.hidden_dim,
            'num_actions': self.num_actions,
        }

        torch.save(sac_dict, path)

    def load_checkpoint(self, path):
        sac_dict = torch.load(path, map_location=self.device)
        self.critic1.load_state_dict(sac_dict['critic1'])
        self.target_critic1.load_state_dict(sac_dict['target_critic1'])
        self.critic2.load_state_dict(sac_dict['critic2'])
        self.target_critic2.load_state_dict(sac_dict['target_critic2'])
        self.actor.load_state_dict(sac_dict['actor'])
        self.target_actor.load_state_dict(sac_dict['target_actor'])
        self.actor_optimizer.load_state_dict(sac_dict['actor_optimizer'])
        self.critic1_optimizer.load_state_dict(sac_dict['critic1_optimizer'])
        self.critic2_optimizer.load_state_dict(sac_dict['critic2_optimizer'])
        self.num_actions = sac_dict['num_actions']
        print("Loaded checkpoint from path: {}".format(path))

    def state_list_to_tensor(self, state):
        state = list(state)
        if len(self.obs_dims) == 0:
            # each entry will be the sequence length of each entry
            self.obs_dims = tuple([tuple(x.shape) for x in state])
        for i, e in enumerate(state):
            # flatten state element
            state[i] = torch.tensor(e.flatten().reshape(1, -1))
        # concatenate state elements
        state = torch.cat(state, dim=1)
        return state

    def state_tensor_to_list(self, state):
        # transform tensor from shape (batch_size, obs_dim) to list of size (obs)
        # where each entry has shape (batch_size, self.obs_dims[entry, 0], self.obs_dim[entry, 1])
        state_list = []
        for i, shape in enumerate(self.obs_dims):
            state_list.append(state[:, :np.product(shape)].reshape([state.shape[0]] + list(shape)).float())
            state = state[:, np.product(shape):]
        return state_list