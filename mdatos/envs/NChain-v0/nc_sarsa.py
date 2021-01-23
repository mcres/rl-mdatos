import argparse
import logging

import gym

from mdatos.algos.sarsa import Sarsa

DISCOUNT_RATE = 0.99
EPISODES_TO_TRAIN = 1000
EPSILON = 1.0
EPSILON_RATE = 0.99999
LEARNING_RATE = 0.1
TERMINAL_STATES = ()

EPISODES_TO_RUN = 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", "-t", help="Train the agent and save it", action="store_true")
    parser.add_argument("--run", "-r", help="Run a pretrained agent", action="store_true")

    logging.basicConfig(level=logging.INFO)

    hyperparameters = {
        "discount_rate": DISCOUNT_RATE,
        "episodes": EPISODES_TO_TRAIN,
        "epsilon": EPSILON,
        "epsilon_rate": EPSILON_RATE,
        "learning_rate": LEARNING_RATE,
        "terminal_states": TERMINAL_STATES,
    }
    sarsa = Sarsa(gym.make("NChain-v0", slip=0), hyperparameters)

    if args.train:
        sarsa.train()
    elif args.run:
        sarsa.run_agent(EPISODES_TO_RUN)
