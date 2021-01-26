import argparse
import logging
import os

from tqdm.auto import tqdm

VIDEOS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "videos")
FPS = 25
LOGS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "logs")
TRAINED_AGENTS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "trained_agents")


def run_standard_parser():
    """
    Create standard parser for running the different algorithms

    :return args:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", "-t", help="train the agent and save it", action="store_true")
    parser.add_argument(
        "--record", "-R", help="record and save a video (only for CartPole and MountainCar", action="store_true"
    )
    parser.add_argument("--run", "-r", help="run a pretrained agent", action="store_true")
    parser.add_argument("--verbose", "-v", help="logging level: 0 for DEBUG, 1 for INFO", type=int)
    args = parser.parse_args()

    if args.verbose == 0:
        logging.basicConfig(level=logging.DEBUG)
    elif args.verbose == 1:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.INFO)

    return args


def get_dirs_no(path):
    """
    Get the number of directories in a given path. It does not include subdirectories.

    :param path: (str)

    :return: (int)
    """
    dir_entries = list(os.scandir(path))
    dirs_list = [1 for dir_ent in dir_entries if dir_ent.is_dir()]

    return len(dirs_list)


class TrainingProgressBarManager:
    """
    Progress bar that shows the state of the training in the console
    Credits: https://colab.research.google.com/github/araffin/rl-tutorial-jnrr19/blob/sb3/4_callbacks_hyperparameter_tuning.ipynb#scrollTo=49RVX7ieRUn7

    :param total_episodes: (int) length of the bar indicated by the total number of episodes during training
    """

    def __init__(self, total_episodes):
        self.pbar = None
        self.total_episodes = total_episodes

    def __enter__(self):
        # TODO: remove unexpected bar created by tqdm constructor
        self.pbar = tqdm(total=self.total_episodes)
        return self

    def update(self, episode_no):
        self.pbar.n = episode_no
        self.pbar.update(0)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.pbar.n = self.total_episodes
        self.pbar.update(0)
        self.pbar.close()
