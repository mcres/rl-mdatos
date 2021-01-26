import gym

from mdatos.algos.sarsa import Sarsa
from mdatos.utils.misc import TrainingProgressBarManager, run_standard_parser

DISCOUNT_RATE = 0.99
EPISODES_TO_TRAIN = 1000
EPSILON = 1.0
EPSILON_RATE = 0.99999
LEARNING_RATE = 0.1
TERMINAL_STATES = ()

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
    sarsa = Sarsa(gym.make("NChain-v0", slip=0), hyperparameters)

    if args.train:
        with TrainingProgressBarManager(EPISODES_TO_TRAIN) as tpb:
            sarsa.train(tpb)
    elif args.run:
        sarsa.run_agent(EPISODES_TO_RUN)
