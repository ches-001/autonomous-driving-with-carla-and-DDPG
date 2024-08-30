# Continuous Action Autonomous Driving on Carla Simulatiom Environment with Deep Determinisitic Policy Gradient (DDPG)

## Setup
1. Clone the repository: `git clone https://github.com/ches-001/autonomous-driving-with-carla-and-DDPG`

2. Download the [Carla simulation software](https://tiny.carla.org/carla-0-9-8-linux) in the directory of the cloned repository

3. run train.py to train the RL agent on default configurations (run with the `-h` flag to view the arguments)

## Explaination
The carla simulation software was used in this repository to develop an Markov Decision Process (MDP)-like environment to train a reinforcement learning algorithm. After initialization, for every timestep the environment step function returns an observation dictionary, the reward, the terminal state and other info pertaining to that timestep.

the observation dictionary containing the camera observation, the speed of the vehicle and the route option / intention for nagivating the waypoint.

The reward function is a weighted sum of three functions:

## Reference and Inspiration
...