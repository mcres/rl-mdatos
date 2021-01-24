import logging
import os
import pickle

import numpy as np
from tqdm.auto import tqdm

FPS = 25
LOGS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "logs")
TRAINED_AGENTS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "trained_agents")


def save_q_table(q_table, env_name, algo_name):
    """
    :param q_table: (numpy.array)
    :param env_name: (str) name of the environment on which the Q-Table was trained on
    :param algo_name: (str) name of the algorithm that trained the Q-Table
    """
    logging.info(f"Saving Q-Table: \n {q_table}")
    assert type(q_table == np.ndarray)
    save_dir = os.path.join(TRAINED_AGENTS_DIR, env_name)
    os.makedirs(save_dir, exist_ok=True)
    save_file = os.path.join(save_dir, algo_name)
    with open(save_file, "wb") as file:
        pickle.dump(q_table, file)


def load_q_table(env_name, algo_name):
    """
    :param env_name: (str) name of the environment on which the Q-Table was trained on
    :param algo_name: (str) name of the algorithm that trained the Q-Table
    :return q_table: (numpy.array)
    """
    load_dir = os.path.join(TRAINED_AGENTS_DIR, env_name)
    load_file = os.path.join(load_dir, algo_name)
    with open(load_file, "rb") as file:
        q_table = pickle.load(file)
    assert type(q_table == np.ndarray)
    logging.info(f"Loading Q-Table: \n {q_table}")

    return q_table


def create_q_table(observation_space, action_space, terminal_states=()):
    """
    :param observation_space: (int)
    :param action_space: (int)
    :param terminal_states: (tuple)
    """
    q_table = np.random.randn(observation_space, action_space)

    # state-action pairs of terminal states must be zero
    if terminal_states:
        for ts in terminal_states:
            q_table[ts] = 0.0

    return q_table


def epsilon_greedy_q_table(q_table, state, epsilon, action_space):
    """
    :param q_table: (numpy.array)
    :param state: (tuple)
    :param epsilon: (float)
    :param action_space: (gym.Space)

    :return action: (int)
    """
    if np.random.uniform(0, 1) < epsilon:
        # random action
        return action_space.sample()
    else:
        # choose an action based on the policy
        state_action_values = q_table[state]
        return np.argmax(state_action_values)


def deterministic_q_table(q_table, state):
    """
    :param q_table: (numpy.array)
    :param state: (int)
    """
    state_action_values = q_table[state]
    return np.argmax(state_action_values)


def create_discrete_q_table(buckets, action_space_length):
    """
    :param buckets: (tuple) number of possible values for each environment observation
    :param action_space_length: (int)
    """
    return np.zeros(buckets + (action_space_length,))


def discretize_state(state, buckets, lower_bounds, upper_bounds):
    """
    Discretize the measurement of a continuous observation space into a discrete number of possible values
    Credits: https://medium.com/@flomay/using-q-learning-to-solve-the-cartpole-balancing-problem-c0a7f47d3f9d

    :param state: (np.array) original observation returned by the gym environment
    :param buckets: (tuple) number of possible values for each environment observation
    :param lower_bounds: (list) lower bounds for each environment observation
    :param upper_bounds: (list) upper bounds for each environment observation

    :return discretized_state: (tuple)
    """
    discretized = list()
    for i in range(len(state)):
        scaling = (state[i] + abs(lower_bounds[i])) / (upper_bounds[i] - lower_bounds[i])
        new_state = int(round((buckets[i] - 1) * scaling))
        # make sure new_state is any int between 0 and (buckets[i] - 1)
        new_state = min(buckets[i] - 1, max(0, new_state))
        discretized.append(new_state)

    discretized = tuple(discretized)

    return discretized


def state_action_to_tuple(state, action):
    """
    Combine state and action into a tuple to uniformely access the Q-Table represented by a numpy array,
    independently of its dimensions

    :param state: (np.ndarray or int)
    :param action: (int)

    :return: (tuple)
    """
    if type(state) == int:
        return (state, action)
    elif type(state) == np.ndarray:
        state_list = state.tolist()
        return tuple(state_list.append(action))
    elif type(state) == tuple:
        state = list(state)
        state.append(action)
        return tuple(state)


class TrainingProgressBarManager:
    """
    Progress bar that shows the state of the training in the console
    Credits: https://colab.research.google.com/github/araffin/rl-tutorial-jnrr19/blob/sb3/4_callbacks_hyperparameter_tuning.ipynb#scrollTo=49RVX7ieRUn7

    :param total_episodes: (int) length of the bar indicated by the total number of episodes during training
    """

    def __init__(self, total_episodes):
        self.pbar = None
        self.total_episodes = total_episodes

    def __enter__(self):
        # TODO: remove unexpected bar created by tqdm constructor
        self.pbar = tqdm(total=self.total_episodes)
        return self

    def update(self, episode_no):
        self.pbar.n = episode_no
        self.pbar.update(0)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.pbar.n = self.total_episodes
        self.pbar.update(0)
        self.pbar.close()
