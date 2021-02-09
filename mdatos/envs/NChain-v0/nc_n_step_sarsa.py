import gym

from mdatos.algos.n_step_sarsa import NStepSarsa
from mdatos.utils.misc import TrainingProgressBarManager, run_standard_parser

DISCOUNT_RATE = 0.97
EPISODES_TO_TRAIN = 1000
EPSILON = 1.0
EPSILON_RATE = 0.99
LEARNING_RATE = 0.05
N_STEPS = 3
TERMINAL_STATES = ()

EPISODES_TO_RUN = 2


if __name__ == "__main__":
    args = run_standard_parser()

    hyperparameters = {
        "n": N_STEPS,
        "discount_rate": DISCOUNT_RATE,
        "episodes": EPISODES_TO_TRAIN,
        "epsilon": EPSILON,
        "epsilon_rate": EPSILON_RATE,
        "learning_rate": LEARNING_RATE,
        "terminal_states": TERMINAL_STATES,
    }
    n_step_sarsa = NStepSarsa(gym.make("NChain-v0", slip=0), hyperparameters)

    if args.train:
        with TrainingProgressBarManager(EPISODES_TO_TRAIN) as tpb:
            n_step_sarsa.train(tpb)
    elif args.run:
        n_step_sarsa.run_agent(EPISODES_TO_RUN)
