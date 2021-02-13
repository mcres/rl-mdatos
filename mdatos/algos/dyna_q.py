import logging
import os
import time

import gym
import numpy as np

from mdatos.utils.agent import discretize_state, get_tensorboard_writter, state_action_to_tuple
from mdatos.utils.misc import FPS, LOGS_DIR, TRAINED_AGENTS_DIR, VIDEOS_DIR
from mdatos.utils.q_table import (
    create_q_table,
    deterministic_q_table,
    epsilon_greedy_q_table,
    load_q_table,
    save_q_table,
)


# TODO: add observation space discretization
class DynaQ:
    """
    :param env: (string) environment registered in gym
    :param params: (dict) hyperparameters specific for a given environment
    """

    def __init__(self, env, params):
        logging.debug("Creating Dyna-Q object")

        self.env = env

        self.discount_rate = params["discount_rate"]
        self.episodes = params["episodes"]
        self.epsilon = params["epsilon"]
        self.epsilon_rate = params["epsilon_rate"]
        self.learning_rate = params["learning_rate"]
        self.no_planning_steps = params["no_planning_steps"]
        # terminal_states is only specified for some environments
        if "terminal_states" in params:
            self.terminal_states = params["terminal_states"]

    def train(self, progress_bar):
        self.writer, log_dir = get_tensorboard_writter(self.env.spec.id, "Dyna-Q")
        logging.info(f"Training Dyna-Q agent in {self.env.spec.id}")
        logging.info(f"Logging training results in {os.path.normpath(log_dir)}")
        self.create_q_table()
        self.create_env_model()
        for ep in range(self.episodes):
            state = self.reset()
            episode_reward = []
            done = False
            while not done:
                action = epsilon_greedy_q_table(self.q_table, state, self.epsilon, self.env.action_space)
                self.epsilon *= self.epsilon_rate
                new_state, reward, done, _ = self.make_step(action)
                episode_reward.append(reward)
                best_next_action = epsilon_greedy_q_table(
                    self.q_table, new_state, epsilon=0.0, action_space=self.env.action_space
                )
                self.update_q_table(state, new_state, action, best_next_action, reward, done)
                self.update_model(state, action, new_state, reward)
                self.plan()
                state = new_state
            self.writer.add_scalar("mean_episode_reward", np.mean(episode_reward), ep)
            self.writer.add_scalar("total_episode_reward", np.sum(episode_reward), ep)
            progress_bar.update(ep)
        save_q_table(self.q_table, self.env.spec.id, "dyna_q")

    def create_q_table(self):
        self.q_table = create_q_table(self.env.observation_space.n, self.env.action_space.n, self.terminal_states)

    def create_env_model(self):
        """
        Dyna-Q algorithm has a table-based sample model that assumes the environment is deterministic.
        The model is a table that maps states and actions into next states and rewards obtained and also keeps track
        of which state-action pairs have been visited before.
        Third dimension value is: 1) States, 2) Rewards, 3) Whether it has been visited or not.
        """
        self.model = np.zeros([self.env.observation_space.n, self.env.action_space.n, 3])

    def update_q_table(
        self,
        previous_state,
        current_state,
        previous_action,
        current_best_action,
        reward,
        done,
    ):
        """
        Environments can have one or more observation parameters but only one action, thus state parameter types
        can vary.

        :param previous_state: (int/np.ndarray/tuple)
        :param current_state: (int/np.ndarray/tuple)
        :param previous_action: (int)
        :param current_best_action: (int)
        :param reward: (float)
        :param done: (bool)
        """
        index_previous = state_action_to_tuple(previous_state, previous_action)
        previous_state_action_value = self.q_table[index_previous]
        if done:
            current_state_action_value = 0.0
        else:
            index_current = state_action_to_tuple(current_state, current_best_action)
            current_state_action_value = self.q_table[index_current]
        td_error = reward + self.discount_rate * current_state_action_value - previous_state_action_value

        previous_state_action_value += self.learning_rate * td_error
        self.q_table[index_previous] = previous_state_action_value

    def update_model(self, state, action, new_state, reward):
        """
        A specific state-action pair will simply store the last-observed next state and next reward.
        """
        self.model[state][action][0] = new_state
        self.model[state][action][1] = reward
        self.model[state][action][2] = True

    def plan(self):
        """
        Update the Q-table by sampling experience using the environment model.
        """
        state_action_pairs_visited = np.nonzero(self.model[..., 2])
        if state_action_pairs_visited[0].size == 0:
            # Only previously visited state-action pairs of the model are selected
            return
        for _ in range(self.no_planning_steps):
            random_index = np.random.randint(0, len(state_action_pairs_visited[0]))
            state = int(state_action_pairs_visited[0][random_index])
            action = int(state_action_pairs_visited[1][random_index])

            new_state_from_model = int(self.model[state][action][0])
            reward_from_model = int(self.model[state][action][1])

            best_next_action = epsilon_greedy_q_table(
                self.q_table, new_state_from_model, epsilon=0.0, action_space=self.env.action_space
            )

            done = False
            if hasattr(self, "terminal_states"):
                done = new_state_from_model in self.terminal_states

            self.update_q_table(state, new_state_from_model, action, best_next_action, reward_from_model, done)

    def run_agent(self, episodes, record=False):
        """
        :param episodes: (int)
        :param record: (bool)
        """
        if record:
            # TODO there's an issue going on and currently videos are not recorded correctly
            # https://github.com/openai/gym/issues/1925
            # As a workaround, videos can be recorded by installing gym from source (pip install -e gym)
            logging.info(f"Recording video")
            video_dir = os.path.join(VIDEOS_DIR, self.env.spec.id, "Dyna_Q")
            self.env = gym.wrappers.Monitor(self.env, video_dir, video_callable=lambda episode_id: True, force=True)

        logging.info(f"Running Dyna-Q agent")
        self.q_table = load_q_table(self.env.spec.id, "dyna_q")
        for ep in range(episodes):
            state = self.reset()
            episode_reward = []
            done = False
            while not done:
                try:
                    self.env.render()
                    time.sleep(1.0 / FPS)
                except NotImplementedError:
                    pass
                action = deterministic_q_table(self.q_table, state)
                new_state, reward, done, _ = self.make_step(action)
                episode_reward.append(reward)
                state = new_state
            logging.info(f"Episode {ep + 1}")
            logging.info(f"Total reward: {np.sum(episode_reward)}")
            logging.info(f"Mean reward: {np.mean(episode_reward)} \n")

    def reset(self):
        state = self.env.reset()

        return state

    def make_step(self, action):
        new_state, reward, done, _ = self.env.step(action)

        return new_state, reward, done, _
