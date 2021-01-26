import logging
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


class Sarsa:
    """
    :param env: (string) environment registered in gym
    :param params: (dict) hyperparameters specific for a given environment
    :param discrete: (bool) whether to discretize the state or not
    """

    def __init__(self, env, params, discrete=False):
        logging.debug("Creating Sarsa object")

        self.env = env
        self.discrete = discrete
        if discrete:
            self.buckets = params["buckets"]
            self.lower_bounds = params["lower_bounds"]
            self.upper_bounds = params["upper_bounds"]

        self.discount_rate = params["discount_rate"]
        self.episodes = params["episodes"]
        self.epsilon = params["epsilon"]
        self.epsilon_rate = params["epsilon_rate"]
        self.learning_rate = params["learning_rate"]
        # terminal_states is only specified for some environments
        if "terminal_states" in params:
            self.terminal_states = params["terminal_states"]

    def train(self, progress_bar):
        self.writer, log_dir = get_tensorboard_writter(self.env.spec.id, "Sarsa")
        logging.info(f"Training Sarsa agent in {self.env.spec.id}")
        logging.info(f"Logging training results in {os.path.normpath(log_dir)}")
        self.create_q_table()
        for ep in range(self.episodes):
            state = self.reset()
            action = epsilon_greedy_q_table(self.q_table, state, self.epsilon, self.env.action_space)
            self.epsilon *= self.epsilon_rate
            episode_reward = []
            done = False
            while not done:
                new_state, reward, done, _ = self.make_step(action)
                episode_reward.append(reward)
                next_action = epsilon_greedy_q_table(self.q_table, new_state, self.epsilon, self.env.action_space)
                self.epsilon *= self.epsilon_rate
                self.update_q_table(state, new_state, action, next_action, reward, done)
                state = new_state
                action = next_action
            self.writer.add_scalar("mean_episode_reward", np.mean(episode_reward), ep)
            self.writer.add_scalar("total_episode_reward", np.sum(episode_reward), ep)
            progress_bar.update(ep)
        save_q_table(self.q_table, self.env.spec.id, "sarsa")

    def create_q_table(self):
        if self.discrete:
            self.q_table = create_discrete_q_table(self.buckets, self.env.action_space.n)
        else:
            self.q_table = create_q_table(self.env.observation_space.n, self.env.action_space.n, self.terminal_states)

    def run_agent(self, episodes, record=False):
        """
        :param episodes: (int)
        :param record: (bool)
        """
        if record:
            # TODO there's an issue and apparently videos are not recorded correctly
            # https://github.com/openai/gym/issues/1925
            logging.info(f"Recording video")
            video_dir = os.path.join(VIDEOS_DIR, self.env.spec.id, "Sarsa")
            self.env = gym.wrappers.Monitor(self.env, video_dir, video_callable=lambda episode_id: True, force=True)

        logging.info(f"Running Sarsa agent")
        self.q_table = load_q_table(self.env.spec.id, "sarsa")
        for ep in range(episodes):
            state = self.reset()
            episode_reward = []
            done = False
            while not done:
                try:
                    self.env.render()
                    time.sleep(1.0 / FPS)
                except NotImplementedError:
                    pass
                action = deterministic_q_table(self.q_table, state)
                new_state, reward, done, _ = self.make_step(action)
                episode_reward.append(reward)
                state = new_state
            logging.info(f"Episode {ep + 1}")
            logging.info(f"Total reward: {np.sum(episode_reward)}")
            logging.info(f"Mean reward: {np.mean(episode_reward)} \n")

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

    def update_q_table(
        self,
        previous_state,
        current_state,
        previous_action,
        current_action,
        reward,
        done,
    ):
        """
        Environments can have one or more observation parameters but only one action, thus state parameter types
        can vary.

        :param previous_state: (int/np.ndarray/tuple)
        :param current_state: (int/np.ndarray/tuple)
        :param previous_action: (int)
        :param current_action: (int)
        :param reward: (float)
        :param done: (bool)
        """
        index_previous = state_action_to_tuple(previous_state, previous_action)
        previous_state_action_value = self.q_table[index_previous]
        if done:
            current_state_action_value = 0.0
        else:
            index_current = state_action_to_tuple(current_state, current_action)
            current_state_action_value = self.q_table[index_current]
        td_error = reward + self.discount_rate * current_state_action_value - previous_state_action_value

        previous_state_action_value += self.learning_rate * td_error
        self.q_table[index_previous] = previous_state_action_value
