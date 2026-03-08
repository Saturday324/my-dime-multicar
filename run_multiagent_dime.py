import os
import time
import traceback
from typing import Dict, Tuple

import gymnasium as gym
import hydra
import jax
import numpy as np
import omegaconf
import wandb
from omegaconf import DictConfig
from stable_baselines3.common.logger import configure
from tqdm.auto import tqdm

from diffusion.dime import DIME
from models.utils import is_slurm_job
from common.multiagent_env_factory import create_multiagent_env


class _SpaceOnlyEnv(gym.Env):
    """Minimal env used only to provide single-agent spaces."""

    metadata = {}

    def __init__(self, observation_space: gym.Space, action_space: gym.Space):
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            super().reset(seed=seed)
        return self.observation_space.sample(), {}

    def step(self, action):
        return self.observation_space.sample(), 0.0, True, False, {}


def _extract_single_agent_spaces(ma_env) -> Tuple[gym.Space, gym.Space]:
    if not isinstance(ma_env.observation_space, gym.spaces.Dict):
        raise TypeError("Expected multi-agent observation_space to be gym.spaces.Dict")
    if not isinstance(ma_env.action_space, gym.spaces.Dict):
        raise TypeError("Expected multi-agent action_space to be gym.spaces.Dict")
    keys = sorted(ma_env.observation_space.spaces.keys())
    if len(keys) == 0:
        raise ValueError("No per-agent spaces found in multi-agent env.")
    agent0 = keys[0]
    return ma_env.observation_space.spaces[agent0], ma_env.action_space.spaces[agent0]


def _build_multiagent_env(cfg):
    return create_multiagent_env(
        env_name=str(cfg.env_name),
        raw_env_kwargs=getattr(cfg, "env_kwargs", None),
        default_start_seed=int(cfg.seed),
    )


def _create_alg(cfg: DictConfig):
    ma_env = _build_multiagent_env(cfg)
    single_obs_space, single_action_space = _extract_single_agent_spaces(ma_env)
    policy_name = "MultiInputPolicy" if isinstance(single_obs_space, gym.spaces.Dict) else "MlpPolicy"

    run_start = time.strftime("%Y%m%d-%H%M%S")
    run_id = f"seed={cfg.seed}_start={run_start}"
    tensorboard_log_dir = f"./logs/{cfg.wandb['group']}/{cfg.wandb['job_type']}/{run_id}/"
    model_save_dir = f"./checkpoints/{cfg.wandb['group']}/{cfg.wandb['job_type']}/{run_id}/"
    os.makedirs(tensorboard_log_dir, exist_ok=True)
    os.makedirs(model_save_dir, exist_ok=True)

    model = DIME(
        policy_name,
        env=_SpaceOnlyEnv(single_obs_space, single_action_space),
        model_save_path=model_save_dir,
        save_every_n_steps=0,
        cfg=cfg,
        tensorboard_log=tensorboard_log_dir,
        replay_buffer_class=None,
    )
    # Keep metrics in files, avoid noisy terminal rollout tables.
    model.set_logger(configure(tensorboard_log_dir, ["csv", "tensorboard"]))
    return model, ma_env


def _sample_actions(model: DIME, obs_dict: Dict[str, np.ndarray], learning_starts: int, action_space: gym.Space):
    actions = {}
    for agent_id in sorted(obs_dict.keys()):
        obs = obs_dict[agent_id]
        if model.num_timesteps < learning_starts:
            action = action_space.sample()
        else:
            action, _ = model.predict(obs, deterministic=False)
        actions[agent_id] = np.asarray(action, dtype=np.float32)
    return actions


def _add_multiagent_transitions(
    model: DIME,
    obs_dict: Dict[str, np.ndarray],
    next_obs_dict: Dict[str, np.ndarray],
    action_dict: Dict[str, np.ndarray],
    reward_dict: Dict[str, float],
    terminated_dict: Dict[str, bool],
    truncated_dict: Dict[str, bool],
    info_dict: Dict[str, dict],
) -> int:
    n = 0
    for agent_id, obs in obs_dict.items():
        action = np.asarray(action_dict[agent_id], dtype=np.float32)
        reward = float(reward_dict.get(agent_id, 0.0))
        done = bool(terminated_dict.get(agent_id, False) or truncated_dict.get(agent_id, False))
        next_obs = next_obs_dict.get(agent_id, None)
        if next_obs is None:
            next_obs = np.zeros_like(obs, dtype=np.float32)
        else:
            next_obs = np.asarray(next_obs, dtype=np.float32)
        obs = np.asarray(obs, dtype=np.float32)

        model.replay_buffer.add(
            obs[None, ...],
            next_obs[None, ...],
            action[None, ...],
            np.array([reward], dtype=np.float32),
            np.array([done], dtype=np.float32),
            [info_dict.get(agent_id, {})],
        )
        n += 1
    return n


def _next_episode_seed(env, episode_idx: int) -> int:
    start = int(getattr(env, "start_index", 0))
    num_scenarios = int(getattr(env, "num_scenarios", 1))
    if num_scenarios <= 0:
        num_scenarios = 1
    return start + (episode_idx % num_scenarios)


def _reset_multiagent_env(env, seed: int):
    reset_ret = env.reset(seed=seed)
    if isinstance(reset_ret, tuple) and len(reset_ret) >= 1:
        obs = reset_ret[0]
        info = reset_ret[1] if len(reset_ret) > 1 else {}
        return obs, info
    return reset_ret, {}


def _train_multiagent(model: DIME, env, cfg: DictConfig):
    total_timesteps = int(cfg.tot_time_steps)
    log_freq = max(int(getattr(cfg, "log_freq", 100)), 1)
    save_every_n_steps = int(getattr(cfg, "save_every_n_steps", 5000))
    next_save_step = save_every_n_steps if save_every_n_steps > 0 else None
    learning_starts = int(cfg.alg.learning_starts)
    gradient_steps = int(cfg.alg.utd)
    train_freq = int(model.train_freq.frequency)

    episode_idx = 0
    obs_dict, _ = _reset_multiagent_env(env, seed=_next_episode_seed(env, episode_idx))
    episode_group_reward = 0.0
    episode_len = 0
    episode_agent_steps = 0
    env_steps = 0
    total_agent_steps = 0
    model.num_timesteps = 0

    pbar = tqdm(total=total_timesteps, desc="env_steps", unit="step")
    try:
        while env_steps < total_timesteps:
            if len(obs_dict) == 0:
                episode_idx += 1
                obs_dict, _ = _reset_multiagent_env(env, seed=_next_episode_seed(env, episode_idx))
                episode_group_reward = 0.0
                episode_len = 0
                episode_agent_steps = 0
                continue

            actions = _sample_actions(model, obs_dict, learning_starts, model.action_space)
            next_obs_dict, reward_dict, terminated_dict, truncated_dict, info_dict = env.step(actions)
            done_all = bool(terminated_dict.get("__all__", False) or truncated_dict.get("__all__", False))

            added = _add_multiagent_transitions(
                model,
                obs_dict=obs_dict,
                next_obs_dict=next_obs_dict,
                action_dict=actions,
                reward_dict=reward_dict,
                terminated_dict=terminated_dict,
                truncated_dict=truncated_dict,
                info_dict=info_dict,
            )

            env_steps += 1
            pbar.update(1)
            total_agent_steps += added
            # In multi-agent training, timesteps are counted by env steps.
            model.num_timesteps = env_steps
            episode_len += 1
            episode_agent_steps += added
            episode_group_reward += sum(float(v) for v in reward_dict.values())

            if env_steps >= learning_starts and model.replay_buffer.size() >= model.batch_size:
                if train_freq <= 1 or (env_steps % train_freq == 0):
                    effective_gradient_steps = gradient_steps if gradient_steps >= 0 else added
                    if effective_gradient_steps > 0:
                        model.train(model.batch_size, effective_gradient_steps)

            if next_save_step is not None and env_steps >= next_save_step:
                while next_save_step is not None and env_steps >= next_save_step:
                    model._save_model()
                    next_save_step += save_every_n_steps

            if env_steps % log_freq == 0:
                model.logger.record("time/total_timesteps", env_steps, exclude="tensorboard")
                model.logger.record("rollout/env_steps", env_steps)
                model.logger.record("rollout/agent_steps_total", total_agent_steps)
                model.logger.record("rollout/active_agents", len(obs_dict))
                model.logger.dump(step=env_steps)

            if done_all:
                model.logger.record("rollout/episode_group_reward", episode_group_reward)
                model.logger.record("rollout/episode_length", episode_len)
                model.logger.record("rollout/episode_agent_steps", episode_agent_steps)
                model.logger.record("rollout/episode_alive_agents_end", len(next_obs_dict))
                model.logger.dump(step=env_steps)
                episode_idx += 1
                obs_dict, _ = _reset_multiagent_env(env, seed=_next_episode_seed(env, episode_idx))
                episode_group_reward = 0.0
                episode_len = 0
                episode_agent_steps = 0
            else:
                obs_dict = next_obs_dict
    finally:
        pbar.close()

    if hasattr(model, "_save_model") and getattr(model, "model_save_path", None):
        model._save_model()
        print(f"Final checkpoint saved at step {model.num_timesteps}")


def initialize_and_run(cfg):
    cfg = hydra.utils.instantiate(cfg)
    seed = cfg.seed

    if cfg.wandb["activate"]:
        name = f"seed_{seed}"
        wandb_config = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        wandb.init(
            settings=wandb.Settings(_service_wait=300),
            project=cfg.wandb["project"],
            group=cfg.wandb["group"],
            job_type=cfg.wandb["job_type"],
            name=name,
            config=wandb_config,
            entity=cfg.wandb["entity"],
            sync_tensorboard=True,
        )
        if is_slurm_job():
            print(f"SLURM_JOB_ID: {os.environ.get('SLURM_JOB_ID')}")
            wandb.summary["SLURM_JOB_ID"] = os.environ.get("SLURM_JOB_ID")

    model, env = _create_alg(cfg)
    try:
        _train_multiagent(model, env, cfg)
    finally:
        env.close()
        try:
            from metadrive.engine.engine_utils import close_engine

            close_engine()
        except Exception:
            pass


@hydra.main(version_base=None, config_path="configs", config_name="base")
def main(cfg: DictConfig) -> None:
    try:
        starting_time = time.time()
        if cfg.use_jit:
            initialize_and_run(cfg)
        else:
            with jax.disable_jit():
                initialize_and_run(cfg)
        end_time = time.time()
        print(f"Training took: {(end_time - starting_time) / 3600} hours")
        if cfg.wandb["activate"]:
            wandb.finish()
    except Exception as ex:
        print("-- exception occured. traceback :")
        traceback.print_tb(ex.__traceback__)
        print(ex, flush=True)
        print("--------------------------------\n")
        traceback.print_exception(ex)
        if cfg.wandb["activate"]:
            wandb.finish()


if __name__ == "__main__":
    main()


# python run_multiagent_dime.py env=metadrive_ma_roundabout tot_time_steps=300000 