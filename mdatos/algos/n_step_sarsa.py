import logging
import math
import os
import time

import gym
import numpy as np

from mdatos.utils.agent import discretize_state, get_tensorboard_writter, state_action_to_tuple
from mdatos.utils.misc import FPS, LOGS_DIR, TRAINED_AGENTS_DIR, VIDEOS_DIR
from mdatos.utils.q_table import (
    create_discrete_q_table,
    create_q_table,
    deterministic_q_table,
    epsilon_greedy_q_table,
    load_q_table,
    save_q_table,
)


class NStepSarsa:
    """
    :param env: (string) environment registered in gym
    :param params: (dict) hyperparameters specific for a given environment
    :param discrete: (bool) whether to discretize the state or not
    """

    def __init__(self, env, params, discrete=False):
        logging.debug("Creating NStepSarsa object")

        self.env = env
        self.discrete = discrete
        if discrete:
            self.buckets = params["buckets"]
            self.lower_bounds = params["lower_bounds"]
            self.upper_bounds = params["upper_bounds"]

        self.n = params["n"]
        self.discount_rate = params["discount_rate"]
        self.episodes = params["episodes"]
        self.epsilon = params["epsilon"]
        self.epsilon_rate = params["epsilon_rate"]
        self.learning_rate = params["learning_rate"]
        # terminal_states is only specified for some environments
        if "terminal_states" in params:
            self.terminal_states = params["terminal_states"]

    def train(self, progress_bar):
        self.writer, log_dir = get_tensorboard_writter(self.env.spec.id, "n-step Sarsa")
        logging.info(f"Training n-step Sarsa agent in {self.env.spec.id}")
        logging.info(f"Logging training results in {os.path.normpath(log_dir)}")
        self.create_q_table()
        for ep in range(self.episodes):
            state = self.reset()
            action = epsilon_greedy_q_table(self.q_table, state, self.epsilon, self.env.action_space)
            self.epsilon *= self.epsilon_rate

            initial_reward = 0.0
            self.experience_buffer = list()
            self.experience_buffer.append((initial_reward, state, action))
            self.episode_reward = []

            T = np.inf
            t = 0
            keep_updating_table = True
            while keep_updating_table:
                if t < T:
                    new_state, reward, done, _ = self.make_step(action)
                    self.episode_reward.append(reward)
                    if done:
                        T = t + 1
                        self.experience_buffer.append((reward, new_state))
                    else:
                        action = epsilon_greedy_q_table(self.q_table, new_state, self.epsilon, self.env.action_space)
                        self.experience_buffer.append((reward, new_state, action))
                tau = t - self.n + 1
                if tau >= 0:
                    self.update_q_table(tau, T)

                t += 1
                keep_updating_table = tau != T - 1
            self.writer.add_scalar("mean_episode_reward", np.mean(self.episode_reward), ep)
            self.writer.add_scalar("total_episode_reward", np.sum(self.episode_reward), ep)
            progress_bar.update(ep)

        save_q_table(self.q_table, self.env.spec.id, "n_step_sarsa")

    def create_q_table(self):
        if self.discrete:
            self.q_table = create_discrete_q_table(self.buckets, self.env.action_space.n)
        else:
            self.q_table = create_q_table(self.env.observation_space.n, self.env.action_space.n, self.terminal_states)

    def update_q_table(self, tau, T):
        """
        :param tau: (int) time whose value is being updated
        :param T: (int) time when the episode finished
        """
        total_return = 0
        for i in range(tau + 1, min((tau + self.n + 1), T + 1)):
            reward = self.experience_buffer[i][0]
            total_return += math.pow(self.discount_rate, i - tau - 1) * reward
        if tau + self.n < T:
            tau_experience = self.experience_buffer[tau + self.n]
            state = tau_experience[1]
            action = tau_experience[2]
            index = state_action_to_tuple(state, action)
            total_return += math.pow(self.discount_rate, self.n) * self.q_table[index]
        tau_experience = self.experience_buffer[tau]
        state = tau_experience[1]
        action = tau_experience[2]
        index = state_action_to_tuple(state, action)
        self.q_table[index] += self.learning_rate * (total_return - self.q_table[index])

    def run_agent(self, episodes, record=False):
        """
        :param episodes: (int)
        :param record: (bool)
        """
        if record:
            # TODO there's an issue going on and currently videos are not recorded correctly
            # https://github.com/openai/gym/issues/1925
            # As a workaround, videos can be recorded by installing gym from source (pip install -e gym)
            logging.info(f"Recording video")
            video_dir = os.path.join(VIDEOS_DIR, self.env.spec.id, "n_step_Sarsa")
            self.env = gym.wrappers.Monitor(self.env, video_dir, video_callable=lambda episode_id: True, force=True)

        logging.info(f"Running n-step Sarsa agent")
        self.q_table = load_q_table(self.env.spec.id, "n_step_sarsa")
        for ep in range(episodes):
            state = self.reset()
            self.episode_reward = []
            done = False
            while not done:
                try:
                    self.env.render()
                    time.sleep(1.0 / FPS)
                except NotImplementedError:
                    pass
                action = deterministic_q_table(self.q_table, state)
                new_state, reward, done, _ = self.make_step(action)
                self.episode_reward.append(reward)
                state = new_state
            logging.info(f"Episode {ep + 1}")
            logging.info(f"Total reward: {np.sum(self.episode_reward)}")
            logging.info(f"Mean reward: {np.mean(self.episode_reward)} \n")

    def reset(self):
        state = self.env.reset()
        if self.discrete:
            state = discretize_state(state, self.buckets, self.lower_bounds, self.upper_bounds)

        return state

    def make_step(self, action):
        new_state, reward, done, _ = self.env.step(action)
        if self.discrete:
            new_state = discretize_state(new_state, self.buckets, self.lower_bounds, self.upper_bounds)

        return new_state, reward, done, _
