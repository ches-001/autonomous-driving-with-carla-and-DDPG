import torch
from environment.env import CarlaEnv
from typing import Dict, Sequence, Union, Optional


class RingTensorBuffer:
    def __init__(self, maxlen: int, env: CarlaEnv):
        self.maxlen = maxlen
        self.start = 0
        self.length = 0
        cam_obs_dim_t = env.observation_space["cam_obs"].shape
        # cam_obs has shape of (H, W, C), we need it to be (C, H, W)
        cam_obs_dim_t = (cam_obs_dim_t[2], cam_obs_dim_t[0], cam_obs_dim_t[1])
        measurements_dim = env.observation_space["measurements"].shape[0]
        intention_dim = 1
        action_dim = env.action_space.shape[0]
        
        self.__cam_obs_buffer = torch.zeros((self.maxlen, *cam_obs_dim_t), dtype=torch.float32, device="cpu")
        self.__measurements_buffer = torch.zeros((self.maxlen, measurements_dim), dtype=torch.float32, device="cpu")
        self.__intentions_buffer = torch.zeros((self.maxlen, intention_dim), dtype=torch.int64, device="cpu")
        self.__actions_buffer = torch.zeros((self.maxlen, action_dim), dtype=torch.float32, device="cpu")
        self.__rewards_buffer = torch.zeros((self.maxlen, 1), dtype=torch.float32, device="cpu")
        self.__future_cam_obs_buffer = torch.zeros((self.maxlen, *cam_obs_dim_t), dtype=torch.float32, device="cpu")
        self.__future_measurements_buffer = torch.zeros((self.maxlen, measurements_dim), dtype=torch.float32, device="cpu")
        self.__future_intentions_buffer = torch.zeros((self.maxlen, intention_dim), dtype=torch.int64, device="cpu")
        self.__terminal_states_buffer = torch.zeros((self.maxlen, 1), dtype=torch.bool, device="cpu")

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: Union[int, torch.LongTensor]) -> Dict[str, torch.Tensor]:
        if torch.is_tensor(idx):
            _invalid_cond = ((idx < 0) | (idx >= self.length)).any()
        elif isinstance(idx, int):
            _invalid_cond = (idx < 0) | (idx >= self.length)
        else:
            raise ValueError("idx must be of type int or LongTensor")
        if _invalid_cond:
            raise IndexError(f"index(es) {idx} out of range")
        item =  {
            "cam_obs": self.__cam_obs_buffer[idx],
            "measurements": self.__measurements_buffer[idx],
            "intentions": self.__intentions_buffer[idx],
            "actions": self.__actions_buffer[idx],
            "rewards": self.__rewards_buffer[idx], 
            "future_cam_obs": self.__future_cam_obs_buffer[idx],
            "future_measurements": self.__future_measurements_buffer[idx],
            "future_intentions": self.__future_intentions_buffer[idx],
            "terminal_states": self.__terminal_states_buffer[idx],
        }
        return item
    
    def memory_gigabytes(self) -> float:
        numel = (
            (self.__cam_obs_buffer.numel() * self.__cam_obs_buffer.element_size()) +
            (self.__measurements_buffer.numel() * self.__measurements_buffer.element_size()) +
            (self.__intentions_buffer.numel() * self.__intentions_buffer.element_size()) +
            (self.__actions_buffer.numel() * self.__actions_buffer.element_size()) +
            (self.__rewards_buffer.numel()  * self.__rewards_buffer.element_size()) +
            (self.__future_cam_obs_buffer.numel() * self.__future_cam_obs_buffer.element_size()) +
            (self.__future_measurements_buffer.numel()  * self.__future_measurements_buffer.element_size()) +
            (self.__future_intentions_buffer.numel() * self.__future_intentions_buffer.element_size())+
            (self.__terminal_states_buffer.numel() * self.__terminal_states_buffer.element_size())
        )
        mem_size = numel / 1e9
        return mem_size
    
    def append(self, item: Dict[str, torch.Tensor]):
        if self.length < self.maxlen:
            self.length += 1
        elif self.length == self.maxlen:
            self.start = (self.start + 1) % self.maxlen
        else:
            # This should not happen
            raise RuntimeError()
        self.__cam_obs_buffer[(self.start + self.length - 1) % self.maxlen] = item["cam_obs"]
        self.__measurements_buffer[(self.start + self.length - 1) % self.maxlen] = item["measurements"]
        self.__intentions_buffer[(self.start + self.length - 1) % self.maxlen] = item["intention"]
        self.__actions_buffer[(self.start + self.length - 1) % self.maxlen] = item["action"]
        self.__rewards_buffer[(self.start + self.length - 1) % self.maxlen] = item["reward"]
        self.__future_cam_obs_buffer[(self.start + self.length - 1) % self.maxlen] = item["future_cam_obs"]
        self.__future_measurements_buffer[(self.start + self.length - 1) % self.maxlen] = item["future_measurements"]
        self.__future_intentions_buffer[(self.start + self.length - 1) % self.maxlen] = item["future_intention"]
        self.__terminal_states_buffer[(self.start + self.length - 1) % self.maxlen] = item["terminal_state"]

    def extend(self, items: Sequence[Dict[str, torch.Tensor]]):
        for item in items: 
            self.append(item)


class ExperienceReplayMemory:
    def __init__(self, env: CarlaEnv, memory_capacity: int=10_000):
        self.env = env
        self.memory_capacity = memory_capacity
        self._memory = RingTensorBuffer(maxlen=memory_capacity, env=env)

    def __len__(self) -> int:
        return len(self._memory)

    def updateMemory(
            self, 
            cam_obs: torch.FloatTensor,
            measurements: torch.FloatTensor,
            intention: torch.LongTensor,
            action: torch.FloatTensor, 
            reward: float, 
            future_cam_obs: torch.FloatTensor,
            future_measurements: torch.FloatTensor,
            future_intention: torch.LongTensor,
            terminal_state: bool
        ):
        item = {}
        item["cam_obs"] = cam_obs
        item["measurements"] = measurements
        item["intention"] = intention
        item["action"] = action
        item["reward"] = reward
        item["future_cam_obs"] = future_cam_obs
        item["future_measurements"] = future_measurements
        item["future_intention"] = future_intention
        item["terminal_state"] = terminal_state
        self._memory.append(item)

    def sampleRandomBatch(self, n_samples: int, device: str="cpu")->Optional[Dict[str, torch.Tensor]]:
        if len(self._memory) < 1:
            return None
        buffer_size = self.__len__()
        indexes = torch.randint(0, buffer_size, size=(n_samples, ))
        batch = self._memory[indexes]
        for k in batch.keys():
            batch[k] = batch[k].to(device)
        return batch

    def clear(self):
        self._memory = RingTensorBuffer(
            maxlen=self.memory_capacity, 
            env=self.env, 
        )