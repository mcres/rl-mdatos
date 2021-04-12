import os

import numpy as np
from rl_mdatos.utils.misc import LOGS_DIR, VIDEOS_DIR, get_dirs_no
from tensorboardX import SummaryWriter


def discretize_state(state, buckets, lower_bounds, upper_bounds):
    """
    Discretize the measurement of a continuous observation space into a discrete number of possible values
    Credits: https://medium.com/@flomay/using-q-learning-to-solve-the-cartpole-balancing-problem-c0a7f47d3f9d

    :param state: (np.array) original observation returned by the gym environment
    :param buckets: (tuple) number of possible values for each environment observation
    :param lower_bounds: (list) lower bounds for each environment observation
    :param upper_bounds: (list) upper bounds for each environment observation

    :return discretized: (tuple)
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


def get_tensorboard_writter(env_name, algo_name):
    """
    :param env_name: (str)
    :param algo_name: (str)

    :return summary_writter: (tensorboardX.SummaryWriter)
    :return experiment_dir:
    """
    log_dir = os.path.join(LOGS_DIR, env_name, algo_name)
    os.makedirs(log_dir, exist_ok=True)

    # create new dir to save the training
    experiment_no = get_dirs_no(log_dir) + 1
    experiment_dir = os.path.join(log_dir, f"Experiment_{experiment_no}")
    summary_writter = SummaryWriter(experiment_dir)

    return summary_writter, experiment_dir


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
