import sys
import logging

sys.path.append("../..")

from algos.sarsa import Sarsa

EPISODES_TO_TRAIN = 1000
TERMINAL_STATES = ()
EPSILON = 1.0
LEARNING_RATE = 0.1
DISCOUNT_RATE = 0.99
EPISODES_TO_RUN = 10


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    hyperparameters = {
        "episodes": EPISODES_TO_TRAIN,
        "terminal_states": TERMINAL_STATES,
        "epsilon": EPSILON,
        "learning_rate": LEARNING_RATE,
        "discount_rate": DISCOUNT_RATE,
    }
    sarsa = Sarsa("NChain-v0", hyperparameters)

    if sys.argv[1] == "train":
        sarsa.train()
    elif sys.argv[1] == "run":
        sarsa.run_agent(EPISODES_TO_RUN)
