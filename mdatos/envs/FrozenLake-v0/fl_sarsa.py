import gym

from mdatos.algos.sarsa import Sarsa
from mdatos.utils.misc import TrainingProgressBarManager, run_standard_parser

DISCOUNT_RATE = 0.97
EPISODES_TO_TRAIN = 2000
EPSILON = 1.0
EPSILON_RATE = 0.999
LEARNING_RATE = 0.1
TERMINAL_STATES = (5, 7, 11, 12, 15)

EPISODES_TO_RUN = 2


if __name__ == "__main__":
    args = run_standard_parser()

    hyperparameters = {
        "discount_rate": DISCOUNT_RATE,
        "episodes": EPISODES_TO_TRAIN,
        "epsilon": EPSILON,
        "epsilon_rate": EPSILON_RATE,
        "learning_rate": LEARNING_RATE,
        "terminal_states": TERMINAL_STATES,
    }
    sarsa = Sarsa(gym.make("FrozenLake-v0", is_slippery=False), hyperparameters)

    if args.train:
        with TrainingProgressBarManager(EPISODES_TO_TRAIN) as tpb:
            sarsa.train(tpb)
    elif args.run:
        sarsa.run_agent(EPISODES_TO_RUN)
