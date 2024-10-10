import os
import copy
import yaml
import logging
import warnings
warnings.filterwarnings("ignore")
import random
import torch
import argparse
import numpy as np
import multiprocessing as mp
from environment.env import CarlaEnv
from environment.spawn import spawn_npcs
from environment.wrappers import FrameStackWrapper, RepeatActionWrapper
from modules.architecture import ActorNetwork, CriticNetwork, ActorCriticNetwork
from rl_utils.common import noise
from rl_utils import DDPGTrainer
from typing import *


CONFIG_PATH = os.path.join("config", "config.yaml")
LOG_FORMAT="%(asctime)s %(levelname)s %(filename)s: %(message)s"
LOG_DATE_FORMAT="%Y-%m-%d %H:%M:%S"

logger = logging.getLogger(__name__)


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    f.close()
    return config

def build_simulation_env(uri: str, port: int, config: Dict[str, Any]) -> CarlaEnv:
    env = CarlaEnv.make_env_with_client(uri, port, **config["env_config"])
    wrappers_config = config["wrappers_config"]
    if wrappers_config["frame_stack"]["num_stack"]:
       env = FrameStackWrapper(env, wrappers_config["frame_stack"]["num_stack"])
    if wrappers_config["repeat_actions"]["num_repeats"]:
        env = RepeatActionWrapper(env, wrappers_config["repeat_actions"]["num_repeats"])
    return env

def build_actor_critic(
        env: CarlaEnv, 
        share_fe: bool, 
        config: Dict[str, Any]) -> Union[ActorCriticNetwork, Tuple[ActorNetwork, CriticNetwork]]:
    
    config = copy.deepcopy(config["model_config"])
    config.update(dict(
        in_channels=env.observation_space["cam_obs"].shape[-1],
        num_measurements=env.observation_space["measurements"].shape[0],
        num_intentions=env.observation_space["intention"].n,
        action_dim=env.action_space.shape[0]
    ))
    if share_fe:
        actor_critic = ActorCriticNetwork(**config)
        return actor_critic
    critic = CriticNetwork(**config)
    config.pop("action_enc_output_dim")
    config.pop("num_critics")
    actor = ActorNetwork(**config)
    return actor, critic

def build_trainer(
        env: CarlaEnv, 
        config: Dict[str, Any],
        actor_critic: Union[ActorCriticNetwork, Tuple[ActorNetwork, CriticNetwork]]) -> DDPGTrainer:
    
    action_noise_config = config["action_noise_config"].copy()
    action_noise_name = action_noise_config.pop("name")

    if "mu" in action_noise_config:
        action_noise_config["mu"] = torch.zeros(
            env.action_space.shape[0], 
            dtype=torch.float32).fill_(action_noise_config["mu"])
        
    if "std" in action_noise_config:
        action_noise_config["std"] = torch.zeros(
            env.action_space.shape[0], 
            dtype=torch.float32).fill_(action_noise_config["std"])
        
    action_noise_fn = getattr(noise, action_noise_name)(**action_noise_config)
    if not isinstance(actor_critic, tuple):
        trainer = DDPGTrainer(
            env, 
            actor_critic=actor_critic, 
            actor_optim_config=config["optim_config"]["actor"], 
            critic_optim_config=config["optim_config"]["critic"],
            actor_lr_schedule_config=config["lr_scheduler_config"]["actor"],
            critic_lr_schedule_config=config["lr_scheduler_config"]["critic"],
            action_noise_fn=action_noise_fn,
            **config["trainer_config"]
        )
    else:
        actor, critic = actor_critic
        trainer = DDPGTrainer(
            env, 
            actor=actor, 
            critic=critic,
            actor_optim_config=config["optim_config"]["actor"], 
            critic_optim_config=config["optim_config"]["critic"],
            action_noise_fn=action_noise_fn,
            **config["trainer_config"]
        )
    return trainer

def main(config: Dict[str, Any], args: argparse.Namespace):
    logger.info("building CarlaEnv...")
    env = build_simulation_env(args.uri, args.port, config)
    try:
        logger.info("building actor critic model(s)...")
        actor_critic = build_actor_critic(env, args.share_fe, config)

        if isinstance(actor_critic, tuple):
            actor, critic = actor_critic
            actor_num_params = sum([i.numel() for i in actor.parameters()])
            critic_num_params = sum([i.numel() for i in critic.parameters()])
            logger.info(f"Num actor params: {actor_num_params}")
            logger.info(f"Num critic params: {critic_num_params}")
        else:
            actor_critic_num_params = sum([i.numel() for i in actor_critic.parameters()])
            logger.info(f"Num actor-critic params: {actor_critic_num_params}")

        logger.info("building RL trainer...")
        trainer = build_trainer(env, config, actor_critic)

        logger.info(f"Trainer buffer memory: {trainer._agent_buffer._memory.memory_gigabytes() :.4f} GB")

        logger.info("commencing training...")
        train_performance, eval_performance = trainer.train(
            num_steps=args.num_steps,
            batch_size=args.batch_size,
            eval_interval=args.eval_interval,
            policy_weights_filename=args.policy_filename,
            grad_steps=args.grad_steps,
            verbose=(not args.no_verbose),
            train_render=args.train_render,
            eval_render=args.eval_render,
            close_env=True,
        )
        trainer.saveMetrics(train_performance, "train_metrics.csv")
        trainer.saveMetrics(eval_performance, "eval_metrics.csv")
    except (Exception or KeyboardInterrupt) as e:
        env.close()
        if isinstance(e, KeyboardInterrupt):
            logger.info("CarlaEnv instance terminated")
        else:
            raise e


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RL Agent on Carla Environment")
    parser.add_argument("--uri", type=str, default="localhost", metavar="", help="URI of carla server instance")
    parser.add_argument("--port", type=int, default=2_000, metavar="", help="Port number of carla server instance")
    parser.add_argument("--tm_port", type=int, default=8_000, metavar="", help="Port number for carla traffic manager API instance")
    parser.add_argument("--share_fe", action="store_true", help="Actor and Critic share feature extraction modules")
    parser.add_argument("--batch_size", type=int, default=128, metavar="", help="Sample batch size sampled from replay memory")
    parser.add_argument("--num_steps", type=int, default=1_000_000, metavar="", help="Number of training episodes / cycles")
    parser.add_argument("--eval_interval", type=int, default=50, metavar="", help="Number of training episodes before evaluation")
    parser.add_argument("--grad_steps", type=int, default=1, metavar="", help="Number of gradient update iterations")
    parser.add_argument("--policy_filename", type=str, default="CarlaAgent.pth.tar", metavar="", help="Filename to store policy weights")
    parser.add_argument("--no_verbose", action="store_true", help="Reduce training output verbosity")
    parser.add_argument("--train_render", action="store_true", help="Render Environment during training")
    parser.add_argument("--eval_render", action="store_true", help="Render Environment during evaluation")
    parser.add_argument("--no_npcs", action="store_true", help="Used to not spawn NPCs when training")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)

    config = load_config(CONFIG_PATH)

    SEED = config["seed"]
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    spawn_event = mp.Event() if not args.no_npcs else None
    spawn_npc_config = config["spawn_npc_config"].copy()
    spawn_npc_config.update({
        "host":args.uri, 
        "port":args.port, 
        "tm_port": args.tm_port, 
        "spawn_event": spawn_event, 
        "rl_training": True
    })
    if not args.no_npcs:
        npc_spawn_process = mp.Process(target=spawn_npcs, kwargs=spawn_npc_config, daemon=True)
        npc_spawn_process.start()
        # wait for npc spawn function to create all npc actors before commencing training
        spawn_event.wait()
        logger.info("NPCs have been spawned successfully")
        
    main(config, args)

    if not args.no_npcs:
        npc_spawn_process.terminate()