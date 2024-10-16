# Continuous Action Autonomous Driving on Carla Simulatiom Environment with Deep Determinisitic Policy Gradient (DDPG)

## Setup
1. Clone the repository: `git clone https://github.com/ches-001/autonomous-driving-with-carla-and-DDPG`

2. Download the Carla simulation software, ([linux: 0.9.8](https://tiny.carla.org/carla-0-9-8-linux) or [linux: 0.9.13](https://tiny.carla.org/carla-0-9-13-linux) for linux and [win: 0.9.8](https://tiny.carla.org/carla-0-9-8-windows) or [win: 0.9.13](https://tiny.carla.org/carla-0-9-13-windows) for windows have been tested with this repository) in the directory of the cloned repository

4. Carla v0.9.8 requires python 3.7, however if you are using a GPU with updated CUDA drivers upto CUDA-11, CUDA-12 and above, you will need to download v0.9.13 that supports python 3.8, with this you can simply install the carla client API package via `pip install carla==0.9.13` instead of adding the built egg module in `CARLA_0.9.13/WindowsNoEditor/PythonAPI/carla/dist/` folder (for windows) or `CARLA_0.9.13/PythonAPI/carla/dist/` folder (for linux) to system path via `sys.path.append(...)`. Ofcourse you will need to do this if you are using v0.9.8 since `carla==0.9.8` cannot be installed via pip for some strange reasons. To add the built egg module to system path, comment the entire try and except block in the `environment/__init__.py` file and add `sys.path.append('<path_to_egg_module>')`

3. run train.py to train the RL agent on default configurations (run with the `-h` flag to view the arguments)

## Explaination
The carla simulation software is used in this repository to develop an Markov Decision Process (MDP)-like environment to train a reinforcement learning algorithm. After initializing the environment, for every timestep the environment step function returns an observation dictionary, a reward value, the terminal status of the environment and other information pertaining to that timestep, like velocity vector, waypoint position, vehicle position, etc

the observation dictionary contains the camera observation (as image frames) $I_t$, the speed of the vehicle $v_t$ in km/hr scaled by 100 and the route option / intention $R_t$ for nagivating the waypoint.

In this implementation, a neural network based agent is trained with the Deep Deterministic Policy Gradient (DDPG) algorithm, an off-policy training algorithm suitable for multi-dimensional continuous action spaces, the choice of DDPG was influenced by the complexity of the problem, as it is not only suitable for continuous action spaces, but also has a good sample efficiency compared to on-policy techniques like PPO, A2C, A3C and its variants.

### DDPG Overview:
The DDPG algorithm consists of an actor $\pi_{\theta}$, a critic $Q_{\theta}$, a target actor ${\pi_{\theta}}^{\prime}$ and a target critic ${Q_{\theta}}^{\prime}$. The actor is responsible for estimating a continuous value action $a_t$ given an observation $s_t$ as input, the critic estimates the Q (quality) value of the observation and action pair, indicating whether said action was suitable given the observation. The target actor and the target critic are used to estimate the action $a_{t+1}$ at the future state $s_{t+1}$ and estimate the Q values of the future state action pairs respectively, these future pairs are used to calculate the target for the critic network.

The aim of the actor network is to maximize the values of the critic network, this is the same as minimizing the negative value of the critic. Mathematically, the objective of the actor can be expressed as such:

$$L_{\pi} = max_{\pi \theta} \hspace{3mm} \frac 1 N \sum_{i=1}^ N Q_{\theta}(s_t, \hspace{1mm} a_t \hspace{0.5mm}|\hspace{0.5mm} \theta^{Q})$$

This can futher be rewritten as:

$$L_{\pi} = min_{\pi \theta} \hspace{3mm} \frac {-1} {N} \sum_{i=1}^N Q_{\theta}(s_t, \hspace{1mm} a_t \hspace{0.5mm}|\hspace{0.5mm} \theta^{Q})$$

The aim of the critic is to minimize its quality estimates with the target quality estimates calculated from the target actor and target critic networks. This can be expressed mathematically as:

$$L_{Q} = min_{Q \theta} \hspace{3mm} \frac 1 N \sum_{i=1}^N ({Q_{\theta}}(s_i, \hspace {1mm}a_i \hspace{0.5mm}|\hspace{0.5mm} \theta^{Q}) - y_i)^2$$

Where: 
$$y_i = r_i + \gamma {Q_{\theta}}^{\prime}(s_{i+1},\hspace{1mm} {\pi_{\theta}}^{\prime}(s_{i+1})  \hspace{0.5mm} | \hspace{0.5mm} \theta^{Q^{\prime}})$$

$r_i$ corresponds reward for the observation action pair and $\gamma$ is a discount factor within the range of 0 and 1


During training, for any given episode, we transition the environment state one step forward in time, retrieve the experience $e_t$ which comprises $(s_t, a_t, r_t, s_{t+1}, T_t)$ ($T_t$ is terminal status of environment (either 0 or 1)). This experience is stored in a replay buffer at every step of the given episode, we randomly sample experiences $e_i \sim ExpBuffer$ to train the actor and critic networks.

Note that the target actor and target critic networks are initialized with the exact weights as the actor and critic networks respectively. For each parameter update, the target actor and target critic networks are smoothly updated as shown in the expression below:

$${\pi_{\theta}}^{\prime} := (1 - \tau){\pi_{\theta}}^{\prime} + \tau \pi_{\theta}$$

$${Q_{\theta}}^{\prime} := (1 - \tau){Q_{\theta}}^{\prime} + \tau Q_{\theta}$$

Where: $\tau$ is a smoothening factor ranging from 0 to 1.

### Policy Network Overview
The network architecture comprises of a convnet based image encoder and a dense layer based measurement encoder. The image encoder takes the image observation frame at any timestep of the environment and encodes it in latent space, which is a smaller vector space than the image space. The measurement encoder takes in the speed reading of the vehicle at any given timestep and encodes it in latent space as well, here the latent space is bigger than the measurement space with just one dimension. The two encoded feature maps are concatinated and passed through a switch based layer, controlled by the categorical road options / intentions along the route.

The environment comprises of 7 road options / intentions, each describing the intention of a hypothetical expert driver at any given waypoint along the route. These intentions are as listed below:

1. RoadOption.VOID
2. RoadOption.LEFT
3. RoadOption.RIGHT
4. RoadOption.STRAIGHT
5. RoadOption.LANEFOLLOW
6. RoadOption.CHANGELANELEFT
7. RoadOption.CHANGELANERIGHT

The image observations and speed measurements of the vehicle do not contain relevant information pertaining to the route the vehicle needs to follow to get to the desired destination, as such, simply using them alone cannot ensure that the autonomous agent will keep to a given route and reach the destination, hence we use these route options to create and index a parameterized switch layer for our policy network, inspired by [the condition imitation paper](https://vladlen.info/papers/conditional-imitation.pdf). The schematics of the network is same as in the paper, it is as shown below:

![imitation learning architecture (b)](https://i.pinimg.com/736x/5f/58/bd/5f58bd036a7c0e293ff486e118a3a76e.jpg)

### Reward Function Overview
In this implementation, four reward functions were combined to handle speed of vehicle $r_{spd}$, deviation from center of lane $r_{dev}$, collision $r_{col}$ and lane invasion $r_{inv}$

$$r_{spd} = \max{(1 - \frac {\begin{cases}{\|x_p - w_{p-{t}}\|} & \text{if } {\|w_{p-{t+1}} - w_{p-{t}}\|} = 0 \\ \\
\frac {\|((w_{p-{t+1}} - w_{p-{t}}) \times (w_{p-{t}} - x_p))\|} {\|w_{p-{t+1}} - w_{p-{t}}\|} & \text{if } otherwise
\end{cases}} {cd_{max}}, \hspace{3mm} 0.0)}\cdot \begin{cases}
v / v_{min} & \text{if } v < v_{vmin} \\ \\
1.0 - \frac{(v - v_{target})} {v_{max} - v_{target}} & \text{if } v > v_{target} \\ \\
1.0 & \text{if } \text{otherwise}
\end{cases}$$

<br>
<br>

$$r_{dev} = \max{(1 - (\frac {dev(\vec{w_{d}}, \hspace{2mm} \vec{v})} {\theta_{max}}), \hspace{2mm} 0.0)} \cdot ({wpi_i - wpi_{i-1}})$$

Where: $dev(., .)$ and $D(., .)$ calculates the angle between two vectors and the euclidean distance between two points respectively, $\vec{w_d}$ is the waypoint forward direction vector, $\vec{v}$ is the velocity vector of the vehicle, $\theta_{max}$ is the maximum allowed deviation from waypoint direction, $cd_{max}$ is the maximum allowed deviation from center of lane, $w_p$ is the waypoint position in 3D space, $x_p$ is the position of vehicle in 3D space and $wpi$ is the waypoint index.

$$r_{col} = \begin{cases}
    -5 & \text{if } \text{collision} \\
    0 & \text{if } \text{no collision}
\end{cases}$$

$$r_{inv} = \begin{cases}
    -5 & \text{if } \text{lane invasion} \\
    0 & \text{if } \text{no lane invasion}
\end{cases}$$


## Reference and Inspiration

```
@inproceedings{Codevilla2018,
  title={End-to-end Driving via Conditional Imitation Learning},
  author={Codevilla, Felipe and M{\"u}ller, Matthias and L{\'o}pez,
Antonio and Koltun, Vladlen and Dosovitskiy, Alexey},
  booktitle={International Conference on Robotics and Automation (ICRA)},
  year={2018},
}
```

```
@article{Lillicrap2015,
  title={Continuous control with deep reinforcement learning},
  author={Lillicrap, Timothy P and Hunt, Jonathan J and Pritzel, Alexander and Heess, Nicolas and Erez, Tom and Tassa, Yuval and Silver, David and Wierstra, Daan},
  journal={arXiv preprint arXiv:1509.02971},
  year={2015},
}
```