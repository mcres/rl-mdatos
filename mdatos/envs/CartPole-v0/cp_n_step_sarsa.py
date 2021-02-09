import math

import gym

from mdatos.algos.n_step_sarsa import NStepSarsa
from mdatos.utils.misc import TrainingProgressBarManager, run_standard_parser

DISCOUNT_RATE = 1.0
EPISODES_TO_TRAIN = 10000
EPSILON = 1.0
EPSILON_RATE = 0.999
LEARNING_RATE = 0.1
N_STEPS = 5

# parameters for discretizing the state
env = gym.make("CartPole-v0")
BUCKETS = (3, 3, 6, 6)
LOWER_BOUNDS = [env.observation_space.low[0], -0.5, env.observation_space.low[2], -math.radians(50) / 1.0]
UPPER_BOUNDS = [env.observation_space.high[0], 0.5, env.observation_space.high[2], math.radians(50) / 1.0]

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
        "buckets": BUCKETS,
        "lower_bounds": LOWER_BOUNDS,
        "upper_bounds": UPPER_BOUNDS,
    }
    n_step_sarsa = NStepSarsa(gym.make("CartPole-v0"), hyperparameters, discrete=True)

    if args.train:
        with TrainingProgressBarManager(EPISODES_TO_TRAIN) as tpb:
            n_step_sarsa.train(tpb)
    elif args.run:
        n_step_sarsa.run_agent(EPISODES_TO_RUN, record=args.record)
