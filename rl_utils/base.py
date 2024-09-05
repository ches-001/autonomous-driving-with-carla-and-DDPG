import os
import torch
import numpy as np
import pandas as pd
from environment.env import CarlaEnv
from modules.architecture import ActorNetwork, CriticNetwork, ActorCriticNetwork
from typing import Tuple, Iterable, Dict, Optional, Union, List


class BaseTrainer:
    actor: Optional[Union[ActorNetwork]]
    critic: Optional[Union[CriticNetwork]]
    actor_critic: Optional[Union[ActorCriticNetwork]]
    env: CarlaEnv
    device: str

    def savePolicyParams(self, param_filename: str):
        param_dir = os.path.join(
            os.getcwd(), 
            f"saved_model/{self.__class__.__name__.replace('Trainer', 'Agent')}"
        )
        if not os.path.isdir(param_dir): 
            os.makedirs(param_dir, exist_ok=True)
        path = os.path.join(param_dir, param_filename)

        if not hasattr(self, "actor_state_dict_path"):
            self.actor_state_dict_path = path

        if self.actor_critic:
            state_dict = self.actor_critic.state_dict() 
        else:
            state_dict = self.actor.state_dict()
        torch.save(state_dict, self.actor_state_dict_path)

    def saveMetrics(self, data: Union[Dict[str, List], pd.DataFrame], filename: str):
        if isinstance(data, dict):
            data = pd.DataFrame(data)
        param_dir = os.path.join(
            os.getcwd(), 
            f"metrics/{self.__class__.__name__.replace('Trainer', 'Agent')}"
        )
        if not os.path.isdir(param_dir): 
            os.makedirs(param_dir, exist_ok=True)
        path = os.path.join(param_dir, filename)
        data.to_csv(path, index=False)

    def format_obs_dict_fn(self, obs_dict: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        # I am well aware that this is a diabolical one liner, but...allow.
        obs_dict: Dict[str, torch.Tensor] = {
            k:(
                torch.from_numpy(v).unsqueeze(0).to(device=self.device)
                  if isinstance(v, np.ndarray) else torch.tensor(
                    [[v]], dtype=torch.int64, device=self.device
                )
            )  for k, v in obs_dict.items()
        }
        obs_dict["cam_obs"] = obs_dict["cam_obs"].permute(0, 3, 1, 2)
        return obs_dict

    def estimateAction(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError
    
    def updateActor(self, *args, **kwargs):
        raise NotImplementedError
    
    def updateCritic(self, *args, **kwargs):
        raise NotImplementedError
    
    def updateActorAndCritic(self):
        raise NotImplementedError

    def train(self, *args, **kwargs) -> Tuple[Dict[str, Iterable], Dict[str, Iterable], Dict[str, Iterable]]:
        raise NotImplementedError

    def evaluate(self, render_env: bool=False) -> Tuple[float, np.ndarray, int]:
        obs_dict, _ = self.env.reset()
        obs_dict = self.format_obs_dict_fn(obs_dict)
        terminal_state = False
        total_rewards = 0
        all_total_rewards = np.zeros((4, ), dtype=np.float32)
        steps = 0
        
        while not terminal_state:
            steps += 1
            action = self.estimateAction(
                obs_dict["cam_obs"], 
                obs_dict["measurements"], 
                obs_dict["intention"],
                with_noise=False
            )
            u = action.squeeze().cpu().numpy()
            _, reward, terminal_state, _ = self.env.step(u)
            total_rewards += reward
            all_total_rewards += self.env.all_rewards
            if render_env:
                self.env.render()
            if terminal_state:
                break
        if render_env:
            self.env.close_render()
        return total_rewards, all_total_rewards, steps

    @classmethod
    def build_trainer(cls, *args, **kwargs):
        raise NotImplementedError