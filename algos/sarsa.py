import logging
import time

import gym

from algos.utils import create_q_table, deterministic_q_table, epsilon_greedy_q_table, load_q_table


class Sarsa:
    """
    :param env_name: (string) environment registered in gym
    :param hyperparams: (dict) hyperparameters specific for a given environment
    """

    def __init__(self, env_name, params):
        logging.info("Creating Sarsa object...")
        self.env = gym.make(env_name)
        self.episodes = params["episodes"]
        self.terminal_states = params["terminal_states"]
        self.epsilon = params["epsilon"]
        self.learning_rate = params["learning_rate"]
        self.discount_rate = params["discount_rate"]

    def train(self):
        """
        :param episodes: (int)
        """
        logging.info("Training Sarsa agent...")
        self.q_table = create_q_table(self.env.observation_space.n, self.env.action_space.n, self.terminal_states)
        for _ in range(self.episodes):
            state = self.env.reset()
            action = epsilon_greedy_q_table(self.q_table, state, self.epsilon, self.env.action_space)
            done = False
            while not done:
                new_state, reward, done, _ = self.env.step(action)
                print(reward)
                next_action = epsilon_greedy_q_table(self.q_table, new_state, self.epsilon, self.env.action_space)
                self.update_q_table(state, new_state, action, next_action, reward, done)
                state = new_state
                action = next_action

    def update_q_table(
        self,
        previous_state,
        current_state,
        previous_action,
        current_action,
        reward,
        done,
    ):
        previous_state_action_value = self.q_table[previous_state][previous_action]
        if done:
            current_state_action_value = 0.0
        else:
            current_state_action_value = self.q_table[current_state][current_action]
        td_error = reward + self.discount_rate * current_state_action_value - previous_state_action_value

        # Compute new value and update table
        previous_state_action_value += self.learning_rate * td_error
        self.q_table[previous_state][previous_action] = previous_state_action_value

    def run_agent(self, episodes):
        logging.info(f"Running Sarsa agent...")
        self.q_table = load_q_table("path")
        for episode in range(episodes):
            logging.info(f"\n Episode {episode}")
            state = self.env.reset()
            done = False
            while not done:
                self.env.render()
                action = deterministic_q_table(self.q_table, state)
                new_state, _, done, _ = self.env.step(action)
                state = new_state
