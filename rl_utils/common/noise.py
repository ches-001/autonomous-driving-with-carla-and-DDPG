import math
import torch
from typing import *


class OrnsteinUhlenbeckNoise:
    def __init__(
        self,
        mu: torch.FloatTensor,
        sigma: torch.FloatTensor,
        theta: float = 0.15,
        dt: float = 1e-2,
        initial_noise: Optional[torch.FloatTensor] = None,
    ):
        
        self._theta = theta
        self._mu = mu
        self._sigma = sigma
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
            + self._sigma * math.sqrt(self._dt) * torch.randn(size=self._mu.shape, device=self._device)
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
        return f"OrnsteinUhlenbeckNoise(mu={self._mu}, sigma={self._sigma})"
