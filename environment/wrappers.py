import numpy as np
from collections import deque
from gym import Wrapper, ObservationWrapper
from gym.spaces import Box
from typing import Dict, Any
from .env import CarlaEnv

class FrameStackWrapper(ObservationWrapper):
    def __init__(self, env: CarlaEnv, num_stack: int):
        super(FrameStackWrapper, self).__init__(env)
        self.num_stack = num_stack
        self._cam_frames = deque(maxlen=self.num_stack)
        self._measurements_frames = deque(maxlen=self.num_stack)
        self.observation_space = {
            "cam_obs": Box(
                np.repeat(self.env.observation_space["cam_obs"].low, num_stack, axis=-1), 
                np.repeat(self.env.observation_space["cam_obs"].high, num_stack, axis=-1),
                dtype=self.env.observation_space["cam_obs"].dtype
            ),
            "measurements": Box(
                np.repeat(self.env.observation_space["measurements"].low, num_stack, axis=0), 
                np.repeat(self.env.observation_space["measurements"].high, num_stack, axis=0),
                dtype=self.env.observation_space["measurements"].dtype
            ),
            "intention": self.env.observation_space["intention"]
        }

    def observation(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        cam_obs = np.concatenate(self._cam_frames, axis=-1)
        measurements = np.concatenate(self._measurements_frames, axis=-1)
        obs["cam_obs"] = cam_obs
        obs["measurements"] = measurements
        return obs
    
    def reset(self):
        obs, info = self.env.reset()
        [(
            self._cam_frames.append(obs["cam_obs"]), 
            self._measurements_frames.append(obs["measurements"])
        ) for _ in range(0, self.num_stack)]
        obs = self.observation(obs)
        return obs, info
    
    def step(self, *args, **kwargs):
        obs, reward, terminal_state, info = self.env.step(*args, **kwargs)
        self._cam_frames.append(obs["cam_obs"])
        self._measurements_frames.append(obs["measurements"])
        obs = self.observation(obs)
        return obs, reward, terminal_state, info
    

class RepeatActionWrapper(Wrapper):
    def __init__(self, env: CarlaEnv, num_repeats: int):
        super(RepeatActionWrapper, self).__init__(env)
        self.num_repeats = num_repeats

    def step(self, *args, **kwargs):
        total_reward = 0
        all_total_rewards = np.zeros(4)
        for i in range(0, self.num_repeats):
            obs, reward, terminal_state, info = self.env.step(*args, **kwargs)
            total_reward += reward
            all_total_rewards += info["all_rewards"]
        info["all_rewards"] = all_total_rewards
        return obs, total_reward, terminal_state, info