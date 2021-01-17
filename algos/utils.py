import logging
import numpy as np
import os
import pickle

LOGS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "logs")
TRAINED_AGENTS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "trained_agents")


def save_q_table(q_table, env_name, algo_name):
    """
    :param q_table: (numpy.array)
    :param env_name: (str) name of the environment on which the Q-Table was trained on
    :param algo_name: (str) name of the algorithm that trained the Q-Table
    """
    logging.info(f"Saving Q-Table: \n {q_table}")
    assert(type(q_table == np.ndarray))
    save_dir = os.path.join(TRAINED_AGENTS_DIR, env_name)
    os.makedirs(save_dir, exist_ok=True)
    save_file = os.path.join(save_dir, algo_name)
    with open(save_file, 'wb') as file:
        pickle.dump(q_table, file)

def load_q_table(env_name, algo_name):
    """
    :param env_name: (str) name of the environment on which the Q-Table was trained on
    :param algo_name: (str) name of the algorithm that trained the Q-Table
    :return q_table: (numpy.array)
    """
    load_dir = os.path.join(TRAINED_AGENTS_DIR, env_name)
    load_file = os.path.join(load_dir, algo_name)
    with open(load_file, 'rb') as file:
        q_table = pickle.load(file)
    assert(type(q_table == np.ndarray))
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
    :param state: (int)
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

