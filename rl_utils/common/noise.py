import math
import torch
from typing import *
from abc import ABC, abstractmethod

class BaseNoise(ABC):
    def __init__(self):
        super().__init__()

    def reset(self):
        pass

    @abstractmethod
    def __call__(self) -> Union[float, torch.FloatTensor]:
        raise NotImplementedError()


class NoNoise(BaseNoise):
    def __call__(self) -> float:
        return 0.0

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
    

class NormalNoise(BaseNoise):
    def __init__(self, mu: torch.FloatTensor, std: torch.FloatTensor, decay_rate: float=0.0):
        self._mu = mu
        self._std = std
        self.decay_rate = decay_rate
        self.decay_val = 1.0

    def __call__(self) -> torch.FloatTensor:
        self.decay_val *= (1 - self.decay_rate)
        return self.decay_val * (torch.normal(self._mu, self._std))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mu={self._mu}, std={self._std})"


class OrnsteinUhlenbeckNoise(BaseNoise):
    def __init__(
        self,
        mu: torch.FloatTensor,
        std: torch.FloatTensor,
        theta: float = 0.15,
        dt: float = 1e-2,
        initial_noise: Optional[torch.FloatTensor] = None,
    ):
        
        self._theta = theta
        self._mu = mu
        self._std = std
        self._dt = dt
        self._dtype = mu.dtype
        self._device = mu.device
        self.initial_noise = initial_noise
        self.noise_prev = torch.zeros_like(self._mu, dtype=self._dtype, device=self._device)
        self.reset()
        super().__init__()

    def __call__(self) -> torch.FloatTensor:
        noise = (
            self.noise_prev
            + self._theta * (self._mu - self.noise_prev) * self._dt
            + self._std * math.sqrt(self._dt) * torch.randn(size=self._mu.shape, device=self._device)
        )
        self.noise_prev = noise
        return noise.to(dtype=self._dtype)

    def reset(self) -> None:
        self.noise_prev = (
            self.initial_noise 
            if self.initial_noise is not None else
            torch.zeros_like(self._mu, device=self._device)
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mu={self._mu}, std={self._std})"
