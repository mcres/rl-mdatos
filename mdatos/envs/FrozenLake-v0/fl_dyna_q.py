import gym

from mdatos.algos.dyna_q import DynaQ
from mdatos.utils.misc import TrainingProgressBarManager, run_standard_parser

DISCOUNT_RATE = 0.99
EPISODES_TO_TRAIN = 2000
EPSILON = 0.1
EPSILON_RATE = 0.999
LEARNING_RATE = 0.9
NO_PLANNING_STEPS = 20
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
        "no_planning_steps": NO_PLANNING_STEPS,
    }
    dyna_q = DynaQ(gym.make("FrozenLake-v0", is_slippery=False), hyperparameters)

    if args.train:
        with TrainingProgressBarManager(EPISODES_TO_TRAIN) as tpb:
            dyna_q.train(tpb)
    elif args.run:
        dyna_q.run_agent(EPISODES_TO_RUN)
