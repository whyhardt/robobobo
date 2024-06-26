from copy import deepcopy
from typing import Callable, Dict, List, Optional, Tuple, Type, Union
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy

from nn_architecture.ae_networks import PositionalEncoder


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


class BasicFeatureExtractor(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, feature_dim: int = 256):
        super().__init__(observation_space, feature_dim)

        self.linear = nn.Linear(observation_space.shape[-1], feature_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(observations)
    
    
class LSTMFeatureExtractor(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, feature_dim: int = 256):
        super().__init__(observation_space, feature_dim)

        self.linear = nn.Linear(observation_space.shape[-1], feature_dim)
        self.lstm = nn.LSTM(feature_dim, feature_dim, batch_first=True, num_layers=1)
        self.linear_out = nn.Linear(feature_dim, feature_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = self.linear(observations)
        x = self.lstm(x)
        return self.linear_out(x[0][:, -1, :])


class AttnLstmFeatureExtractor(BaseFeaturesExtractor):
    """
    This extractor is based on a transformer encoder with a follow-up LSTM.
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, feature_dim: int = 256):
        super().__init__(observation_space, feature_dim)

        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[-1]
        assert n_input_channels % 2 == 0, "Number of input channels must be even."

        # transformer layer
        pos_encoder = PositionalEncoder(d_model=n_input_channels, dropout=0.1)
        lin_in_layer = nn.Linear(n_input_channels, n_input_channels*2)
        transformer_layer = nn.TransformerEncoderLayer(d_model=n_input_channels*2, nhead=7)
        norm_layer = nn.LayerNorm(n_input_channels*2)
        transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=2, norm=norm_layer)
        lin_out_layer = nn.Linear(n_input_channels*2, n_input_channels*2)
        self.transformer = nn.Sequential(pos_encoder, lin_in_layer, transformer_encoder, lin_out_layer)

        # lstm layer
        self.lstm_layer = nn.LSTM(n_input_channels*2, n_input_channels, batch_first=True, num_layers=3)
        self.lin_out = nn.Linear(n_input_channels*10, feature_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = self.transformer(observations)
        x, _ = self.lstm_layer(x)
        return self.lin_out(x[:, -10:, :].reshape(x.shape[0], -1))


class AttnNetworkOn(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the features extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        feature_dim: int,
        last_layer_dim_pi: int = 256,
        last_layer_dim_vf: int = 256,
    ):
        super().__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Re-ordering will be done by pre-preprocessing or wrapper
        assert feature_dim % 2 == 0, "Number of input channels must be even."

        self.activation = nn.Tanh()

        # transformer block
        pos_encoder = PositionalEncoder(d_model=feature_dim, dropout=0.1)
        # lin_in_layer = nn.Linear(feature_dim, feature_dim)
        transformer_layer = nn.TransformerEncoderLayer(d_model=feature_dim, nhead=8)
        norm_layer = nn.LayerNorm(feature_dim)
        transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=2, norm=norm_layer)
        # lin_out_layer = nn.Linear(feature_dim * 2, feature_dim * 2)

        # policy network
        self.pi_transformer = nn.Sequential(pos_encoder, transformer_encoder)
        # self.pi_lstm_layer = nn.LSTM(feature_dim * 2, feature_dim, batch_first=True, num_layers=3)
        self.pi_lin_out = nn.Linear(feature_dim, last_layer_dim_pi)

        # value network
        self.vf_transformer = nn.Sequential(pos_encoder, transformer_encoder)
        # self.vf_lstm_layer = nn.LSTM(feature_dim * 2, feature_dim, batch_first=True, num_layers=3)
        self.vf_lin_out = nn.Linear(feature_dim, last_layer_dim_vf)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        x = self.pi_transformer(features)
        # x, _ = self.pi_lstm_layer(x)
        return self.activation(self.pi_lin_out(x[:, -1:, :])).squeeze(1)#.reshape(x.shape[0], -1)))

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        x = self.vf_transformer(features)
        # x, _ = self.vf_lstm_layer(x)
        return self.activation(self.vf_lin_out(x[:, -1:, :])).squeeze(1)#.reshape(x.shape[0], -1)))


class AttnActorCriticPolicyOn(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ):
        # Disable orthogonal initialization
        kwargs["ortho_init"] = False
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )


    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = AttnNetworkOn(self.features_dim)


class AttnLstmNetworkOn(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the features extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        feature_dim: int,
        last_layer_dim_pi: int = 256,
        last_layer_dim_vf: int = 256,
    ):
        super().__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Re-ordering will be done by pre-preprocessing or wrapper
        assert feature_dim % 2 == 0, "Number of input channels must be even."

        self.activation = nn.Tanh()

        # transformer block
        pos_encoder = PositionalEncoder(d_model=feature_dim, dropout=0.1)
        # lin_in_layer = nn.Linear(feature_dim, feature_dim * 2)
        transformer_layer = nn.TransformerEncoderLayer(d_model=feature_dim, nhead=8)
        norm_layer = nn.LayerNorm(feature_dim)
        transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=2, norm=norm_layer)
        # lin_out_layer = nn.Linear(feature_dim * 2, feature_dim * 2)

        # policy network
        self.pi_transformer = nn.Sequential(deepcopy(pos_encoder), deepcopy(transformer_encoder))
        self.pi_lstm_layer = nn.LSTM(feature_dim, feature_dim//2, batch_first=True, num_layers=3)
        self.pi_lin_out = nn.Linear(feature_dim//2, last_layer_dim_pi)

        # value network
        self.vf_transformer = nn.Sequential(deepcopy(pos_encoder), deepcopy(transformer_encoder))
        self.vf_lstm_layer = nn.LSTM(feature_dim, feature_dim//2, batch_first=True, num_layers=3)
        self.vf_lin_out = nn.Linear(feature_dim//2, last_layer_dim_vf)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        x = self.pi_transformer(features)
        x, _ = self.pi_lstm_layer(x)
        return self.activation(self.pi_lin_out(x[:, -1:, :])).squeeze(1)#.reshape(x.shape[0], -1)))

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        x = self.vf_transformer(features)
        x, _ = self.vf_lstm_layer(x)
        return self.activation(self.vf_lin_out(x[:, -1:, :])).squeeze(1)#.reshape(x.shape[0], -1)))


class AttnLstmActorCriticPolicyOn(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ):
        # Disable orthogonal initialization
        kwargs["ortho_init"] = False
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )


    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = AttnLstmNetworkOn(self.features_dim)



class RecurrentPolicy(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the features extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        feature_dim: int,
        last_layer_dim_pi: int = 256,
        last_layer_dim_vf: int = 256,
        num_layers: int = 3,
    ):
        super().__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        self.activation = nn.Tanh()

        self.hidden_state_pi = torch.ones((1, 1, feature_dim))
        self.hidden_state_vf = torch.ones((1, 1, feature_dim))
        
        # make multiple linear layers
        # last layer has to be of dimension last_layer_dim_[pi, vf]
        # layer before last layer needs to be of dimension feature_dim
        nn_container = nn.ModuleList()
        for l in range(num_layers):
            nn_container.append(nn.Linear(feature_dim, feature_dim))
        self.output_layer = nn.Linear(feature_dim, last_layer_dim_pi)
        
        # policy network
        self.pi_hidden = deepcopy(nn_container)
        self.pi_out = nn.Linear(feature_dim, last_layer_dim_pi)

        # value network
        self.pi_hidden = deepcopy(nn_container)
        self.pi_out = nn.Linear(feature_dim, last_layer_dim_vf)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        self.hidden_state_pi = self.activation(self.pi_hidden(features*self.hidden_state_pi))
        return self.activation(self.pi_out(self.hidden_state_pi))

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        self.hidden_state_vf = self.activation(self.vf_hidden(features*self.hidden_state_vf))
        return self.activation(self.vf_out(self.hidden_state_vf))
    