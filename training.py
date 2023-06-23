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


def simple_test(env: gym.Env, agent: BaseAlgorithm, test=True, plot=True, plot_reference=False):
    """Test trained SAC agent"""
    done = False
    truncated = False
    rewards = []
    actions = []
    portfolio = []
    # if test:
    #     agent.eval()
    # else:
    #     agent.train()
    state = env.reset()

    print(f"\nTest scenario (agent.train={not test}) started.")
    while not done and not truncated:
        action = agent.predict(state.reshape(1, -1), deterministic=test)[0]
        state, _, done, _ = env.step(action)
        rewards.append(copy.deepcopy(env.total_equity().item()))
        actions.append(copy.deepcopy(action[0]))
        portfolio.append(copy.deepcopy(env.portfolio[0]))
    print(f"Test scenario terminated. Total reward: {np.round(np.sum(rewards), 2)}\n")
    env.close()

    # print("Test scenario -- final total equity: {}".format(env.total_equity().item()))

    if plot:
        # fig, axs = plt.subplots(4, 1, sharex=True)

        plt.plot(rewards, label='total equity')
        plt.plot(np.convolve(rewards, np.ones(10) / 10, mode='valid'), label='total equity (smoothed)')
        plt.ylabel('total equity')
        plt.xlabel('time steps')
        plt.title("total equity of test")
        plt.legend()
        plt.show()

        plt.plot(actions, label='actions')
        plt.plot(np.convolve(actions, np.ones(10) / 10, mode='valid'), label='actions (smoothed)')
        plt.ylabel('actions')
        plt.xlabel('time steps')
        plt.title("actions of test")
        plt.legend()
        plt.show()

        plt.plot(portfolio, label='portfolio')
        plt.plot(np.convolve(portfolio, np.ones(10) / 10, mode='valid'), label='portfolio (smoothed)')
        plt.ylabel('portfolio')
        plt.xlabel('time steps')
        plt.title("portfolio of test")
        plt.legend()
        plt.show()

        # plt.title(f"Total final equity in [$] (Grow: {total_equity[-1]/total_equity[0]:.2f})")
        # axs[0].plot(np.convolve(total_equity, np.ones(10) / 10, mode='valid'))

        # axs[1].plot(env.stock_data)
        # axs[1].set_ylabel('Stock prices [$]')
        #
        # axs[2].plot(portfolio)
        # axs[2].set_ylabel('Portfolio')
        #
        # axs[3].plot(actions)
        # axs[3].set_ylabel('Actions')
        # axs[3].set_xlabel('Time steps')

        # visualize_actions(np.array(actions), cmap=None, min=-1, max=1, title='actions over time')
        # visualize_actions(np.array(portfolio), cmap=None, title='portfolio over time')

    if plot_reference:
        # plot the average of all stock prices
        avg = np.mean(env.stock_data, axis=1)
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
        plt.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.ylabel('time steps')
        plt.xlabel('features')
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
    plt.plot(mean_values, color='black', label='Mean')

    # Plot the standard deviation band
    plt.fill_between(range(len(std_values)), mean_values + std_values, mean_values - std_values,
                       color='gray', alpha=0.3, label='Standard Deviation')

    plt.xlabel('Time Step')
    plt.legend()
    plt.show()
