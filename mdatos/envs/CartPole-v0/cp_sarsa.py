import argparse
import logging
import math

import gym

from mdatos.algos.sarsa import DiscreteSarsa
from mdatos.algos.utils import TrainingProgressBarManager

DISCOUNT_RATE = 1.0
EPISODES_TO_TRAIN = 10000
EPSILON = 1.0
EPSILON_RATE = 0.99999
LEARNING_RATE = 0.1

# parameters for discretizing the state
env = gym.make("CartPole-v0")
BUCKETS = (3, 3, 6, 6)
LOWER_BOUNDS = [env.observation_space.low[0], -0.5, env.observation_space.low[2], -math.radians(50) / 1.0]
UPPER_BOUNDS = [env.observation_space.high[0], 0.5, env.observation_space.high[2], math.radians(50) / 1.0]

EPISODES_TO_RUN = 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", "-t", help="Train the agent and save it", action="store_true")
    parser.add_argument("--run", "-r", help="Run a pretrained agent", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    hyperparameters = {
        "discount_rate": DISCOUNT_RATE,
        "episodes": EPISODES_TO_TRAIN,
        "epsilon": EPSILON,
        "epsilon_rate": EPSILON_RATE,
        "learning_rate": LEARNING_RATE,
        "buckets": BUCKETS,
        "lower_bounds": LOWER_BOUNDS,
        "upper_bounds": UPPER_BOUNDS,
    }
    sarsa = DiscreteSarsa(gym.make("CartPole-v0"), hyperparameters)

    if args.train:
        with TrainingProgressBarManager(EPISODES_TO_TRAIN) as tpb:
            sarsa.train(tpb)
    elif args.run:
        sarsa.run_agent(EPISODES_TO_RUN)
