import logging
import os
import time

import gym
import numpy as np
from tensorboardX import SummaryWriter

from mdatos.algos.utils import (
    LOGS_DIR,
    TRAINED_AGENTS_DIR,
    create_q_table,
    deterministic_q_table,
    epsilon_greedy_q_table,
    load_q_table,
    save_q_table,
)


class Sarsa:
    """
    :param env: (string) environment registered in gym
    :param hyperparams: (dict) hyperparameters specific for a given environment
    """

    def __init__(self, env, params):
        logging.info("Creating Sarsa object")

        self.env = env

        self.discount_rate = params["discount_rate"]
        self.episodes = params["episodes"]
        self.epsilon = params["epsilon"]
        self.epsilon_rate = params["epsilon_rate"]
        self.learning_rate = params["learning_rate"]
        self.terminal_states = params["terminal_states"]

        self.sarsa_log_dir = os.path.join(LOGS_DIR, self.env.spec.id)
        os.makedirs(self.sarsa_log_dir, exist_ok=True)
        self.writer = SummaryWriter(os.path.join(LOGS_DIR, self.env.spec.id), filename_suffix="Sarsa")

    def train(self):
        logging.info(f"Training Sarsa agent in {self.env.spec.id}")
        logging.info(f"Logging training results in {self.sarsa_log_dir}")
        self.q_table = create_q_table(self.env.observation_space.n, self.env.action_space.n, self.terminal_states)
        for ep in range(self.episodes):
            state = self.env.reset()
            action = epsilon_greedy_q_table(self.q_table, state, self.epsilon, self.env.action_space)
            self.epsilon *= self.epsilon_rate
            episode_reward = []
            done = False
            while not done:
                new_state, reward, done, _ = self.env.step(action)
                episode_reward.append(reward)
                next_action = epsilon_greedy_q_table(self.q_table, new_state, self.epsilon, self.env.action_space)
                self.epsilon *= self.epsilon_rate
                self.update_q_table(state, new_state, action, next_action, reward, done)
                state = new_state
                action = next_action
            self.writer.add_scalar("mean_episode_reward", np.mean(episode_reward), ep)
            self.writer.add_scalar("total_episode_reward", np.sum(episode_reward), ep)
        save_q_table(self.q_table, self.env.spec.id, "sarsa")

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
        :param previous_state: (int)
        :param current_state: (int)
        :param previous_action: (int)
        :param current_action: (int)
        :param reward: (float)
        :param done: (bool)
        """
        previous_state_action_value = self.q_table[previous_state][previous_action]
        if done:
            current_state_action_value = 0.0
        else:
            current_state_action_value = self.q_table[current_state][current_action]
        td_error = reward + self.discount_rate * current_state_action_value - previous_state_action_value

        previous_state_action_value += self.learning_rate * td_error
        self.q_table[previous_state][previous_action] = previous_state_action_value

    def run_agent(self, episodes):
        """
        :param episodes: (int)
        """
        logging.info(f"Running Sarsa agent...")
        self.q_table = load_q_table(self.env.spec.id, "sarsa")
        for ep in range(episodes):
            state = self.env.reset()
            episode_reward = []
            done = False
            while not done:
                try:
                    self.env.render()
                except NotImplementedError:
                    pass
                action = deterministic_q_table(self.q_table, state)
                new_state, reward, done, _ = self.env.step(action)
                episode_reward.append(reward)
                state = new_state
            logging.info(f"Episode {ep + 1}")
            logging.info(f"Total reward: {np.sum(episode_reward)}")
            logging.info(f"Mean reward: {np.mean(episode_reward)} \n")
