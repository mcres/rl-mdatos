import numpy as np
import os

LOGS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "logs")
TRAINED_AGENTS_DIR = os.path.join(os.path.abspath(__file__), "..", "data", "trained_agents")


def load_q_table(path):
    """
    :param path: (str) path to file where the Q-Table is stored
    """
    pass


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

