import copy

import time
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import torch

import gymnasium as gym
from stable_baselines3.common.base_class import BaseAlgorithm

from environment import Environment
from utils.dataprocessor import DataProcessor
from nn_architecture.agents import Agent


def simple_train(
        env: gym.Env,
        agent: BaseAlgorithm,
        num_actions: int,
        batch_size: int,
        parameter_update_interval: int,
        num_random_actions=None,
        path_checkpoint=None,
        checkpoint_interval=100,
        render=True,
        time_limit=1e9):
    """
    This method interacts with the environment and trains the agent in batches on past experience.
    The agent is provided as an instance of the Agent class.
    """

    agent.num_actions = 0
    episode = 0
    episode_rewards = []
    agent.train()

    if num_random_actions is None:
        num_random_actions = num_actions//2

    while agent.num_actions < num_actions:
        state = env.reset()[0]
        t = 0
        done = False
        truncated = False
        episode_reward = 0
        while not done and not truncated:
            # Decide whether to draw random action or to use agent
            if len(agent.replay_buffer) < num_random_actions:
                # Draw random action
                action = env.action_space.sample()
            else:
                # Draw greedy action
                with torch.no_grad():
                    action = agent.get_action_exploration(torch.from_numpy(state).to(agent.device)).cpu().numpy()

            # log AC
            with torch.no_grad():
                agent.logger['means'].append(agent.get_action_exploitation(torch.from_numpy(state).to(agent.device)).cpu().item())
                agent.logger['log_probs'].append(agent.get_action_exploration(torch.from_numpy(state).unsqueeze(0).to(agent.device), log_prob=True)[-1].cpu().item())
                agent.logger['q_values'].append(np.min([agent.q_net1(torch.from_numpy(state).unsqueeze(0).to(agent.device), torch.from_numpy(action).unsqueeze(0).to(agent.device)).cpu().item(), agent.q_net2(torch.from_numpy(state).unsqueeze(0).to(agent.device), torch.from_numpy(action).unsqueeze(0).to(agent.device)).cpu().item()]))
                agent.logger['alphas'].append(agent.alpha.cpu().item())

            # create figure to plot current state and update it continuously
            if render:
                env.render()

            # Give chosen action to environment to adjust internal parameters and to compute new state
            next_state, reward, done, truncated, _ = env.step(action)
            reward = copy.deepcopy(next_state[0])
            
            # Append experience to replay buffer
            agent.replay_buffer.push(copy.deepcopy(state.reshape(1, -1)),
                                     copy.deepcopy(action.reshape(1, -1)),
                                     copy.deepcopy(np.array([reward]).reshape(1, -1)),
                                     copy.deepcopy(next_state.reshape(1, -1)),
                                     copy.deepcopy(np.array([done]).reshape(1, -1)))

            state = next_state

            # Update parameters each n steps
            if t % parameter_update_interval == 0 and len(agent.replay_buffer) > batch_size*10:
                for i in range(parameter_update_interval):
                    if i % agent.delay == 0:
                        update_actor = True
                    else:
                        update_actor = False
                    agent.update(batch_size, update_actor)

            episode_reward += reward
            agent.num_actions += 1
            t += 1
            if t == time_limit or agent.num_actions == num_actions:
                truncated = True

        # Collect total equity of current episode
        print(f"Episode: {episode + 1} -- time steps: {t} -- reward: {np.round(episode_reward, 2)}")
        # total_equity_final.append(env.total_equity().item())
        episode += 1
        episode_rewards.append(copy.deepcopy(episode_reward))

        # Save model for later use
        if path_checkpoint and episode % checkpoint_interval == 0:
            agent.save_checkpoint(path_checkpoint)

    env.close()
    return np.array(episode_rewards, dtype=np.float32), agent


def test(env: Environment, agent: BaseAlgorithm, deterministic=True, plot=True, plot_reference=False, title=''):
    """Test trained SAC agent"""
    done = False
    truncated = False
    rewards = []
    actions = []
    portfolio = []
    cash = []
    state = env.reset()[0]
    observation_length = env.t + 1

    print(f"\nTest scenario (deterministic={deterministic}) started.")
    while not done and not truncated:
        # print(f"Time step: {len(rewards)}; total equity: {np.round(env.total_equity().item(), 2)}")
        with torch.no_grad():
            action = agent.predict(state, deterministic=deterministic)[0]
        state, _, done, truncated, _ = env.step(action)
        # if len(rewards) > 1 and np.abs(rewards[-1] - env.total_equity().item()) > 1e4:
        #     print("Warning: Total equity changed by more than 1000. Maybe somethings wrong")
        rewards.append(copy.deepcopy(env.total_equity().item()))
        actions.append(copy.deepcopy(action))
        portfolio.append(copy.deepcopy(env.portfolio))
        cash.append(copy.deepcopy(env.cash))
        if env.cash == 0:
            print("Cash is zero.")
            if np.sum(portfolio[-2]) - np.sum(portfolio[-1]) != 0:
                print("Warning: Portfolio changed although cash is zero.")
    print(f"Test scenario terminated. Total reward: {rewards[-1]}\n")
    env.close()

    rewards = np.array(rewards).reshape(len(rewards), -1)
    actions = np.array(actions).reshape(len(actions), -1)
    actions_mean = np.mean(actions, axis=1)
    actions_std = np.std(actions, axis=1)
    portfolio = np.array(portfolio).reshape(len(portfolio), -1)
    portfolio_mean = np.mean(portfolio, axis=1)
    portfolio_std = np.std(portfolio, axis=1)

    if plot:
        fig, axs = plt.subplots(4, 1, sharex=True)

        # plot the average of all stock prices
        avg = np.mean(env.stock_data[observation_length-1:rewards.shape[0]+observation_length], axis=1)
        axs[0].plot(avg/avg[0], '--', label='avg price')
        axs[0].plot(rewards/rewards[0], label='total equity')
        axs[0].set_ylabel('rel. price')
        axs[0].set_ylim([0, np.max(rewards/rewards[0])*1.1])
        axs[0].grid()

        axs[1].plot(actions_mean, label='actions')
        axs[1].fill_between(np.arange(len(actions_mean)), actions_mean-actions_std, actions_mean+actions_std, alpha=0.2)
        axs[1].set_ylabel('actions')
        axs[1].grid()

        axs[2].plot(np.mean(portfolio, axis=1), label='portfolio')
        axs[2].fill_between(np.arange(len(portfolio_mean)), portfolio_mean-portfolio_std, portfolio_mean+portfolio_std, alpha=0.2)
        axs[2].set_ylabel('portfolio')
        axs[2].grid()

        axs[3].plot(cash, label='cash')
        axs[3].set_ylabel('cash')
        axs[3].set_xlabel('time steps (days)')
        axs[3].set_xticks(np.arange(0, len(cash), len(cash)//30))
        # set orientation of x labels
        for tick in axs[3].get_xticklabels():
            tick.set_rotation(90)
        # set x labels to every 5th tick
        axs[3].grid()

        plt.title(title)
        plt.show()

        plot_portfolio(portfolio)
        visualize_actions(actions, min=-1, max=1, title='actions over time')

    return rewards, actions, portfolio, cash


def visualize_actions(matrix, min=None, max=None, cmap='binary', title=None):
    # Calculate mean and standard deviation per time step
    mean_values = np.mean(matrix, axis=1)
    std_values = np.std(matrix, axis=1)

    # Create a colormap from blue to white to red
    if cmap is None:
        pass
    elif cmap == 'binary':
        cmap = mpl.colormaps['coolwarm']
    else:
        cmap = mpl.colormaps['Reds']

    # Set the color range based on the minimum and maximum values in the matrix
    vmin = min if min is not None else np.min(matrix)
    vmax = max if max is not None else np.max(matrix)

    if cmap:
        # Plot the matrix using imshow
        plt.imshow(matrix.T, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.xlabel('time steps')
        plt.ylabel('features')
        plt.title(title if title is not None else '')
        # set width of image to 10 inches
        plt.show()
    else:
        # plot each row of the matrix as a separate line
        for i in range(matrix.shape[1]):
            plt.plot(matrix[:, i])
        plt.ylabel('features')
        plt.xlabel('time steps')
        plt.title(title if title is not None else '')
        plt.show()

    # Plot the mean value with a solid line
    # plt.plot(mean_values, color='black', label='Mean')

    # Plot the standard deviation band
    # plt.fill_between(range(len(std_values)), mean_values + std_values, mean_values - std_values,
    #                    color='gray', alpha=0.3, label='Standard Deviation')

    # plt.xlabel('Time Step')
    # plt.legend()
    # plt.show()


def plot_portfolio(portfolio, title='Portfolio over time', normalize=True):
    """
    Plot the portfolio over time. But only non-zero stocks.
    :param portfolio:
    :param title:
    :param normalize:
    :return:
    """

    # remove all zero columns
    portfolio = portfolio[:, np.where(np.sum(portfolio, axis=0) != 0)[0]]

    if normalize:
        portfolio = np.concatenate((np.zeros((1, portfolio.shape[-1])), portfolio), axis=0) / np.max(portfolio, axis=0)

    # plot the portfolio over time
    plt.plot(portfolio)
    plt.ylabel('Portfolio')
    plt.xlabel('Time step')
    plt.title(title)
    plt.show()