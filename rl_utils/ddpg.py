import time
import logging
import copy
import gym
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .base import BaseTrainer
from .common.exp_replay import ExperienceReplayMemory
from .common.noise import BaseNoise, NoNoise
from .common.utils import polyak_averaging, bn_running_stats_polyak_averaging
from datetime import timedelta
from typing import *
from modules.architecture import ActorNetwork, CriticNetwork, ActorCriticNetwork
    

logger = logging.getLogger(__name__)


class DDPGTrainer(BaseTrainer):
    def __init__(self, 
                env: gym.Env,
                *, 
                actor: Optional[ActorNetwork]=None, 
                critic: Optional[CriticNetwork]=None,
                actor_critic: Optional[ActorCriticNetwork]=None,
                actor_optim_config: Optional[Dict[str, Any]]=None,
                critic_optim_config: Optional[Dict[str, Any]]=None,
                actor_lr_schedule_config: Optional[Dict[str, Any]]=None,
                critic_lr_schedule_config: Optional[Dict[str, Any]]=None,
                replay_buffer_size: int=10_000,
                action_noise_fn: Optional[BaseNoise]=None,
                clip_grads: bool=True,
                tau: float=1e-3,
                gamma: float=0.99, 
                target_critic_redution: str="mean",
                device: str="cpu",
                buffer_device: str="cpu",
            ):
        
        assert (actor and critic) or actor_critic
        assert not ((actor and critic) and actor_critic)
        if not torch.cuda.is_available():
            device = "cpu"

        self.env = env
        self.actor = actor
        self.critic = critic
        self.actor_critic = actor_critic
        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic = copy.deepcopy(self.critic)
        self.target_actor_critic = copy.deepcopy(self.actor_critic)
        self._agent_buffer = ExperienceReplayMemory(
            env, memory_capacity=replay_buffer_size, buffer_device=buffer_device
        )
        self.clip_grads = clip_grads
        self.tau = tau
        self.gamma = gamma
        self.device = device
        self.target_critic_redution = target_critic_redution
        self.action_noise_fn = action_noise_fn or NoNoise()
            
        if actor_critic:
                self.actor_critic.to(self.device)
                self.target_actor_critic.to(self.device)
                # in a shared feature extractor configuration, only the actor updates the
                # feature extractors (image_encoder, _measurement_encoder and _latent_rep)
                self._actor_params = (
                    list(self.actor_critic._image_encoder.parameters()) + 
                    list(self.actor_critic._measurement_encoder.parameters()) +
                    list(self.actor_critic._latent_rep.parameters()) +
                    [self.actor_critic._actor_weights, ] +
                    [self.actor_critic._actor_bias, ]
                )
                self._critic_params = (
                    list(self.actor_critic._action_encoder.parameters()) +
                    [self.actor_critic._critic_weights, ] +
                    [self.actor_critic._critic_bias, ]
                )
                self._untrackGrad(self.target_actor_critic)
        else:
            self.actor.to(self.device)
            self.critic.to(self.device)
            self.target_actor.to(self.device)
            self.target_critic.to(self.device)
            self._actor_params = self.actor.parameters()
            self._critic_params = self.critic.parameters()
            self._untrackGrad(self.target_actor)
            self._untrackGrad(self.target_critic)

        self._hardTargetUpdate()

        if not actor_optim_config:
            self.actor_optimizer = torch.optim.Adam(self._actor_params, lr=1e-3)
        else:
            actor_optim_config = copy.deepcopy(actor_optim_config)
            optim_name = actor_optim_config.pop("optim_name")
            self.actor_optimizer = getattr(torch.optim, optim_name)(self._actor_params, **actor_optim_config)

        if not critic_optim_config:
            self.critic_optimizer = torch.optim.Adam(self._critic_params, lr=1e-3)
        else:
            critic_optim_config = copy.deepcopy(critic_optim_config)
            optim_name = critic_optim_config.pop("optim_name")
            self.critic_optimizer = getattr(torch.optim, optim_name)(self._critic_params, **critic_optim_config)

        if actor_lr_schedule_config:
            actor_lr_schedule_config = copy.deepcopy(actor_lr_schedule_config)
            actor_scheduler_name = actor_lr_schedule_config.pop("name")
            self.actor_lr_scheduler = getattr(torch.optim.lr_scheduler, actor_scheduler_name)(
                self.actor_optimizer, **actor_lr_schedule_config
            )

        if critic_lr_schedule_config:
            critic_lr_schedule_config = copy.deepcopy(critic_lr_schedule_config)
            critic_scheduler_name = critic_lr_schedule_config.pop("name")
            self.critic_lr_scheduler = getattr(torch.optim.lr_scheduler, critic_scheduler_name)(
                self.critic_optimizer, **critic_lr_schedule_config
            )


    def _untrackGrad(self, module: Optional[nn.Module]):
        for params in module.parameters():
            params.requires_grad = False


    def _hardTargetUpdate(self):
        if not self.actor_critic:
            self.target_actor.load_state_dict(self.actor.state_dict())
            self.target_critic.load_state_dict(self.critic.state_dict())
        else:
            self.target_actor_critic.load_state_dict(self.actor_critic.state_dict())


    def _softTargetUpdate(self):
        if not self.actor_critic:
            polyak_averaging(self.actor, self.target_actor, self.tau)
            polyak_averaging(self.critic, self.target_critic, self.tau)
            bn_running_stats_polyak_averaging(self.actor, self.target_actor, self.tau)
            bn_running_stats_polyak_averaging(self.critic, self.target_critic, self.tau)

        else:
            polyak_averaging(self.actor_critic, self.target_actor_critic, self.tau)
            bn_running_stats_polyak_averaging(self.actor_critic, self.target_actor_critic, self.tau)
    

    def estimateAction(
            self, 
            img: torch.FloatTensor, 
            measurements: torch.FloatTensor, 
            intention: torch.LongTensor,
            with_noise: bool=True,
        ) -> torch.FloatTensor:

        img = img.to(self.device)
        measurements = measurements.to(self.device)
        intention = intention.to(self.device)
        with torch.no_grad():
            if self.actor_critic:
                self.actor_critic.eval()
                action = self.actor_critic.actor_forward(img, measurements, intention)
            else:
                self.actor.eval()
                action = self.actor(img, measurements, intention)
            action = action.detach().cpu()

            if with_noise and self.action_noise_fn:  
                noise = self.action_noise_fn()
            else:
                noise = 0.0
            action = action + noise
            low = torch.from_numpy(self.env.action_space.low)
            high = torch.from_numpy(self.env.action_space.high) 
            action = torch.clip(action, low, high)
        return action
    

    def updateActor(self, agent_experience: Dict[str, torch.Tensor]): 
        _module_input = {
            "cam_obs": agent_experience["cam_obs"], 
            "measurements": agent_experience["measurements"], 
            "intentions": agent_experience["intentions"]
        }
        if self.actor_critic:
            self.actor_critic.train()
            agent_actions = self.actor_critic.actor_forward(**_module_input)
            policy_objective = self.actor_critic.critic_forward(
                actions=agent_actions, use_all_critics=False, **_module_input
            )
            policy_loss = -policy_objective.mean()
        else:
            self.actor.train()
            agent_actions = self.actor(**_module_input)
            policy_objective = self.critic(
                actions=agent_actions, use_all_critics=False, **_module_input
            )
            policy_loss = -policy_objective.mean()

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        if self.clip_grads:
            for param in self._actor_params: 
                param.grad.data.copy_(param.grad.data.clamp(-1.0, 1.0))
        self.actor_optimizer.step()
        

    def updateCritic(self, agent_experience: Dict[str, torch.Tensor]):
        _module_input = {
            "cam_obs": agent_experience["cam_obs"], 
            "measurements": agent_experience["measurements"], 
            "intentions": agent_experience["intentions"]
        }
        _future_module_input = {
            "cam_obs": agent_experience["future_cam_obs"], 
            "measurements": agent_experience["future_measurements"], 
            "intentions": agent_experience["future_intentions"]
        }
        actions = agent_experience["actions"]
        rewards = agent_experience["rewards"]
        terminal_states = agent_experience["terminal_states"].int()
       
        if self.actor_critic:
            self.actor_critic.train()
            q_values = self.actor_critic.critic_forward(actions=actions, reduction="none", **_module_input)
            with torch.no_grad():
                agent_future_actions = self.target_actor_critic.actor_forward(**_future_module_input)
                future_q_values = self.target_actor_critic.critic_forward(
                    actions=agent_future_actions, reduction=self.target_critic_redution, **_future_module_input
                )
        else:
            self.critic.train()
            q_values = self.critic(actions=actions, reduction="none", **_module_input)
            with torch.no_grad():
                agent_future_actions = self.target_actor(**_future_module_input)
                future_q_values = self.target_critic(
                    actions=agent_future_actions, reduction=self.target_critic_redution, **_future_module_input
                )

        value_target = rewards + (self.gamma * future_q_values * (1 - terminal_states))
        value_loss = sum([F.mse_loss(q_est, value_target) for q_est in q_values])
        
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        if self.clip_grads:
            for param in self._critic_params:
                param.grad.data.copy_(param.grad.data.clamp(-1.0, 1.0))
        self.critic_optimizer.step()
        

    def updateActorAndCritic(self, batch_size: int, grad_steps: int=1):
        for i in range(0, grad_steps):
            agent_experience = self._agent_buffer.sampleRandomBatch(batch_size, device=self.device)
            self.updateCritic(agent_experience)
            self.updateActor(agent_experience)
            self._softTargetUpdate()

        if hasattr(self, "actor_lr_scheduler"):
            self.actor_lr_scheduler.step()

        if hasattr(self, "critic_lr_scheduler"):
            self.critic_lr_scheduler.step()


    def train(
        self,
        num_steps: int,
        batch_size: int=32,
        eval_interval: int=10,
        policy_weights_filename: str="DDPG_policy.pth.tar",
        grad_steps: int=1,
        verbose: bool=True,
        train_render: bool=False,
        eval_render: bool=False,
        close_env: bool=False) -> Tuple[Dict[str, Iterable], Dict[str, Iterable]]:
        
        _template = {
            "reward": [], 
            "speed_reward": [], 
            "deviation_reward": [], 
            "collision_reward": [], 
            "invasion_reward": [],
        }
        train_performance = copy.deepcopy(_template)
        eval_performance = copy.deepcopy(_template)
        eval_performance["better_than_prior"] = []
        best_reward = -np.inf
        current_episode = 0
        current_step = 0
        best_eval_episode = 0
        start_time = time.time()

        while current_step < num_steps:
            obs_dict, _ = self.env.reset()
            obs_dict = self.format_obs_dict_fn(obs_dict)
            episode_reward = 0
            all_rewards = np.zeros((4, ), dtype=np.float32)
            terminal_state = False 
            episode_step = 0

            self.action_noise_fn.reset()
            while not terminal_state and current_step < num_steps:
                action = self.estimateAction(
                    obs_dict["cam_obs"], 
                    obs_dict["measurements"], 
                    obs_dict["intention"],
                    with_noise=True,
                )
                u = action.squeeze().numpy()
                next_obs_dict, reward, terminal_state, info = self.env.step(u, train_render)
                episode_reward += reward
                all_rewards += info["all_rewards"]
                next_obs_dict = self.format_obs_dict_fn(next_obs_dict)

                self._agent_buffer.updateMemory(
                    cam_obs=obs_dict["cam_obs"],
                    measurements=obs_dict["measurements"],
                    intention=obs_dict["intention"],
                    action=action,
                    reward=reward,
                    future_cam_obs=next_obs_dict["cam_obs"],
                    future_measurements=next_obs_dict["measurements"],
                    future_intention=next_obs_dict["intention"],
                    terminal_state=terminal_state,
                )
                if len(self._agent_buffer) >= batch_size:
                    self.updateActorAndCritic(batch_size, grad_steps)
                obs_dict = next_obs_dict.copy()

                current_step += 1
                episode_step += 1
                if verbose:
                    steer, throttle = u
                    elapsed_time = str(timedelta(seconds=(time.time() - start_time)))
                    _print_msg = (
                        f"episode: {current_episode}| current_step: {current_step}| episode_step: {episode_step}| "
                        f"elapsed_time: {elapsed_time}| accum reward: {episode_reward :.3f}| "
                        f"steer(w noise): {steer :.3f}| throttle (w noise): {throttle :.3f}"
                    )
                    if terminal_state:
                        _print_msg + f" terminal msg: {self.env.terminal_reason}"
                    print(_print_msg, end="\r")

            if train_render:
                self.env.close_render()

            train_performance["reward"].append(episode_reward)
            train_performance["speed_reward"].append(all_rewards[0])
            train_performance["deviation_reward"].append(all_rewards[1])
            train_performance["collision_reward"].append(all_rewards[2])
            train_performance["invasion_reward"].append(all_rewards[3])

            if current_episode % eval_interval==0:
                eval_reward, all_eval_rewards, _ = self.evaluate(eval_render)
                better_than_prior = False
                if eval_reward > best_reward:
                    best_eval_episode = current_episode
                    best_reward = eval_reward
                    better_than_prior = True
                    self.savePolicyParams(policy_weights_filename)

                if verbose:
                    _print_msg = (
                        f"eval reward {eval_reward: .3f}| best eval reward: {best_reward :.3f}"
                        f" @ episode {best_eval_episode}| "
                    )
                    if better_than_prior:
                        _print_msg += f"best model saved :)"
                    print("\n", _print_msg)

                eval_performance["reward"].append(eval_reward)
                eval_performance["speed_reward"].append(all_eval_rewards[0])
                eval_performance["deviation_reward"].append(all_eval_rewards[1])
                eval_performance["collision_reward"].append(all_eval_rewards[2])
                eval_performance["invasion_reward"].append(all_eval_rewards[3])
                eval_performance["better_than_prior"].append(better_than_prior)

            current_episode += 1
            if verbose: print("\n")

        if close_env: 
            self.env.close()
        return train_performance, eval_performance