from typing import Optional

import torch
import torch.nn as nn

class ValueNetwork(nn.Module):
    """Class of value network that resembles the critic"""

    def __init__(self, state_dim, hidden_dim, num_layers=6, dropout=0.3, init_w=3e-3):
        """Determines network architecture at object initialization"""
        super(ValueNetwork, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.net = nn.Sequential()
        lin = nn.Linear(state_dim, hidden_dim)
        if init_w:
            lin.weight.data.uniform_(-init_w, init_w)
            lin.bias.data.uniform_(-init_w, init_w)
        self.net.append(lin)
        self.net.append(nn.Dropout(dropout))
        for i in range(num_layers):
            lin = nn.Linear(hidden_dim, hidden_dim)
            if init_w:
                lin.weight.data.uniform_(-init_w, init_w)
                lin.bias.data.uniform_(-init_w, init_w)
            self.net.append(lin)
            self.net.append(nn.ReLU())
            self.net.append(nn.Dropout(dropout))
        lin = nn.Linear(hidden_dim, 1)
        if init_w:
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

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.net = nn.Sequential()
        lin = nn.Linear(num_inputs+num_actions, hidden_dim)
        if init_w:
            lin.weight.data.uniform_(-init_w, init_w)
            lin.bias.data.uniform_(-init_w, init_w)
        self.net.append(lin)
        self.net.append(nn.Dropout(dropout))
        for i in range(num_layers):
            lin = nn.Linear(hidden_dim, hidden_dim)
            if init_w:
                lin.weight.data.uniform_(-init_w, init_w)
                lin.bias.data.uniform_(-init_w, init_w)
            self.net.append(lin)
            self.net.append(nn.ReLU())
            self.net.append(nn.Dropout(dropout))
        lin = nn.Linear(hidden_dim, 1)
        if init_w:
            lin.weight.data.uniform_(-init_w, init_w)
            lin.bias.data.uniform_(-init_w, init_w)
        self.net.append(lin)

    def forward(self, state, action):
        """Processes state and corresponding action through the network layers and outputs the Q-value"""
        return self.net(torch.cat([state, action], dim=-1))


class PolicyNetwork(nn.Module):
    """Class of policy network that resembles the actor"""

    def __init__(self, num_inputs, num_actions, hidden_dim, num_layers=6, dropout=0.3, init_w=3e-3):
        """Determines network architecture at object initialization.
        The policy network has two outputs i.e., the mean-value and the deviation-value for the randomized
        action-selection by a normal distribution around the mean-value.
        """
        super(PolicyNetwork, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.num_actions = num_actions
        self.num_inputs = num_inputs

        self.net = nn.Sequential()
        lin = nn.Linear(num_inputs, hidden_dim)
        if init_w:
            lin.weight.data.uniform_(-init_w, init_w)
            lin.bias.data.uniform_(-init_w, init_w)
        self.net.append(lin)
        for i in range(num_layers):
            lin = nn.Linear(hidden_dim, hidden_dim)
            if init_w:
                lin.weight.data.uniform_(-init_w, init_w)
                lin.bias.data.uniform_(-init_w, init_w)
            self.net.append(lin)
            self.net.append(nn.ReLU())
            self.net.append(nn.Dropout(dropout))

        # This layer computes the mean-value
        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        if init_w:
            self.mean_linear.weight.data.uniform_(-init_w, init_w)
            self.mean_linear.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        """Processes the state through the network layers to compute the mean-value and deviation-value"""
        x = self.net(state)
        mean = self.mean_linear(x)

        return mean


class SoftPolicyNetwork(PolicyNetwork):
    """Class of policy network that resembles the actor"""

    def __init__(self, num_inputs, num_actions, hidden_dim, num_layers=6, dropout=0.3, init_w=3e-3, log_std_min=-20, log_std_max=2):
        """Determines network architecture at object initialization.
        The policy network has two outputs i.e., the mean-value and the deviation-value for the randomized
        action-selection by a normal distribution around the mean-value.
        """
        super(SoftPolicyNetwork, self).__init__(num_inputs=num_inputs, num_actions=num_actions,
                                                hidden_dim=hidden_dim, num_layers=num_layers,
                                                dropout=dropout, init_w=init_w)

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # This layer computes the deviation-value
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)
        if init_w:
            self.log_std_linear.weight.data.uniform_(-init_w, init_w)
            self.log_std_linear.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        """Processes the state through the network layers to compute the mean-value and deviation-value"""
        x = self.net(state)
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

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
        1. process corresponding inputs (multidimensional ones) through the sub networks
        2. concatenate the remaining states with the outputs of the sub networks (now arrays)
        3. process the new state with the actual network

        :param state: list of tensors, where the last n tensors are the inputs for the n sub networks
        :return: return of actual network
        """

        # TODO: Check for correct input
        # if isinstance(state, list):
        #     assert isinstance(all(state), torch.Tensor), "state must be a list of tensors"

        state_sub_networks = []
        for i, sub_network in enumerate(self.policy_sub_networks):
            state_sub_networks.append(sub_network(state[-len(self.policy_sub_networks) + i].to(self.device)))
        state = state[:-len(self.policy_sub_networks)]
        state.extend(state_sub_networks)
        for i in range(len(state)):
            if len(state[i].shape) > 2:
                state[i] = state[i].reshape(state[i].shape[0], -1)
            state[i] = state[i].to(self.device)
        state = torch.concat(state, dim=-1).float()
        return super(PolicyRoboboboNetwork, self).forward(state)


class SoftQRoboboboNetwork(SoftQNetwork):
    def __init__(self, num_inputs, num_actions, hidden_size, num_layers=3, init_w=3e-3, policy_sub_networks: Optional[nn.ModuleList]=nn.ModuleList()):
        # calculate num_inputs for policy network by subtracting the number of sub network inputs and adding the number of sub network outputs
        if policy_sub_networks is not None:
            num_inputs_policy_network = num_inputs - sum([sub_net.input_dim*sub_net.seq_len - sub_net.hidden_dim for sub_net in policy_sub_networks])
        else:
            num_inputs_policy_network = num_inputs
        super().__init__(num_inputs=num_inputs_policy_network, num_actions=num_actions, hidden_dim=hidden_size, num_layers=num_layers, init_w=init_w)
        self.policy_sub_networks = policy_sub_networks

    def forward(self, state, action):
        state_sub_networks = []
        for i, sub_network in enumerate(self.policy_sub_networks):
            state_sub_networks.append(sub_network(state[-len(self.policy_sub_networks) + i].to(self.device)))
        state = state[:-len(self.policy_sub_networks)]
        state.extend(state_sub_networks)
        for i in range(len(state)):
            if len(state[i].shape) > 2:
                state[i] = state[i].reshape(state[i].shape[0], -1).to(self.device)
        state = torch.concat(state, dim=1)
        return super(SoftQRoboboboNetwork, self).forward(state, action)


class ValueRoboboboNetwork(ValueNetwork):
    def __init__(self, num_inputs, hidden_size, num_layers=3, init_w=3e-3, policy_sub_networks: Optional[nn.ModuleList] = nn.ModuleList()):
        # calculate num_inputs for policy network by subtracting the number of sub network inputs and adding the number of sub network outputs
        if policy_sub_networks is not None:
            num_inputs_policy_network = num_inputs - sum(
                [sub_net.input_dim * sub_net.seq_len - sub_net.hidden_dim for sub_net in policy_sub_networks])
        else:
            num_inputs_policy_network = num_inputs
        super().__init__(state_dim=num_inputs_policy_network, hidden_dim=hidden_size, num_layers=num_layers, init_w=init_w)
        self.policy_sub_networks = policy_sub_networks

    def forward(self, state):
        state_sub_networks = []
        for i, sub_network in enumerate(self.policy_sub_networks):
            state_sub_networks.append(sub_network(state[-len(self.policy_sub_networks) + i].to(self.device)))
        state = state[:-len(self.policy_sub_networks)]
        state.extend(state_sub_networks)
        for i in range(len(state)):
            if len(state[i].shape) > 2:
                state[i] = state[i].reshape(state[i].shape[0], -1).to(self.device)
        state = torch.concat(state, dim=1)
        return super(ValueRoboboboNetwork, self).forward(state)


# class TemperatureNetwork(nn.Module):
#     """Temperature network for SAC which takes in Q-Value """