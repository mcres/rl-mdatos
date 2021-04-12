# rl-mdatos

This repository contains my final project for the _Data Mining_ subject &mdash; _Minería de Datos_ in Spanish, that's why `mdatos`, taught in the _Master's Degree In Systems And Control Engineering_ at [UNED](https://www.uned.es/universidad/inicio.html) (_Universidad Nacional de Educación a Distancia_) and [UCM](https://www.ucm.es/) (_Universidad Complutense de Madrid_), from Spain.

It is an implementation of several tabular Reinforcement Learning algorithms, which are then applied to [OpenAI Gym](https://github.com/openai/gym) environments.
The algorithms and environments implemented are the following:


Environment | Sarsa | Q-Learning | n-step Sarsa | Dyna-Q 
--- | --- | --- | --- | ---
[NChain-v0](https://gym.openai.com/envs/NChain-v0/)| :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
[FrozenLake-v0](https://gym.openai.com/envs/FrozenLake-v0/)| :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
[CartPole-v0](https://gym.openai.com/envs/CartPole-v0/)| :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_multiplication_x: |
[MountainCar-v0](https://gym.openai.com/envs/MountainCar-v0/)| :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_multiplication_x:|


The goal of this repo is purely educational:
- For more elaborated and complicated RL algorithms, see [cleanrl](https://github.com/vwxyzjn/cleanrl).
- For an intuitive, easy-to-use library widely used in research, see [stable-baselines3](https://github.com/DLR-RM/stable-baselines3) and [rl-baselines3-zoo](https://github.com/DLR-RM/rl-baselines3-zoo).

A Jupyter Notebook written in Spanish that provides basic explanations of RL concepts making use of this repo can be found [here](https://github.com/mcres/rl-mdatos/blob/master/notebook/notebook.ipynb).

The [bibliography](https://github.com/mcres/rl-mdatos#bibliography) I used is probably the most common entry point if you want to learn Reinforcement Learning.

### How to use this repo

In order to train and evaluate the agents in this repo, follow these steps:

Create and activate a virtual environment:

```
$ cd rl-mdatos
$ virtualenv .venv
$ source .venv/bin/activate
```

Install the required packages:

```
$ (.venv) pip install -r requirements.txt
```

Install this very repo in editable mode:

```
$ (.venv) pip install -e .
```

Go to the desired environment. For each environment, there's a script to train, execute and/or record a specific algorithm:

```
$ (.venv) cd rl_mdatos/envs/desired_env
```

To train a Q-Learning agent in `CartPole-v0`:

```
$ (.venv) python cp_q_learning.py --train
```

To execute the trained agent:

```
$ (.venv) python cp_q_learning.py --run
```

To record the execution (this only works for `CartPole-v0` and `MountainCar-v0`):

```
$ (.venv) python cp_q_learning.py --run --record
```

3 types of files are stored in `rl-mdatos/data`:
- `logs`: data generated during training, which can be visualized with `tensorboard` (`tensorboard --logdir data/...`)
- `trained_agents`: files with final parameters of the trained agents, which are loaded at execution time.
- `videos`: videos of the recorded episodes.

## Output

After successfully training the agents, these should be the results.

### NChain-v0

```
INFO:root:Running Q-Learning agent
INFO:root:Episode 1
INFO:root:Total reward: 9960
INFO:root:Mean reward: 9.96
```

### FrozenLake-v0

<p align="center">
  <img src="https://user-images.githubusercontent.com/14854080/114436720-f8cd5180-9bc5-11eb-8640-966c092bb759.png">
</p>

### CartPole-v0

<p align="center">
  <img src="https://user-images.githubusercontent.com/14854080/114436929-36ca7580-9bc6-11eb-829e-206ca43c889d.gif">
</p>

### MountainCar-v0

<p align="center">
  <img src="https://user-images.githubusercontent.com/14854080/114436940-39c56600-9bc6-11eb-90db-5382cc8de0d6.gif">
</p>


## Bibliography

[1]  Richard S. Sutton and Andrew G. Barto. *Reinforcement learning: An introduction*. MIT press, 2018.

[2]  David Silver. *Lectures on Reinforcement Learning*. URL:https://www.davidsilver.uk/teaching/. 2015.

[3]  Stuart J. Russell and Peter Norvig. *Artificial Intelligence - A Modern Approach, Third International Edition*. Pearson Education London, 2010.