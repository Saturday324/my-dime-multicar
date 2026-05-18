import os
import time
import traceback
from collections import deque
from pathlib import Path
from typing import Dict, Tuple

import gymnasium as gym
import hydra
import jax
import numpy as np
import omegaconf
import wandb
from omegaconf import DictConfig, OmegaConf
from stable_baselines3.common.logger import configure
from tqdm.auto import tqdm

from common.multiagent_env_factory import create_multiagent_env
from diffusion.dime import save_model_state
from diffusion.meta_dime import MetaDIME
from meta.task_replay_buffer import TaskReplayBuffer
from models.utils import is_slurm_job


class _SpaceOnlyEnv(gym.Env):
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


def _close_env(env) -> None:
    if env is None:
        return
    try:
        env.close()
    finally:
        try:
            from metadrive.engine.engine_utils import close_engine

            close_engine()
        except Exception:
            pass


def _load_task_specs(cfg: DictConfig):
    env_dir = Path(__file__).resolve().parent / "configs" / "env"
    task_specs = []
    for task_cfg in cfg.taskset.tasks:
        env_config_name = str(task_cfg.env_config)
        env_cfg = OmegaConf.load(env_dir / f"{env_config_name}.yaml")
        env_kwargs = OmegaConf.to_container(env_cfg.env_kwargs, resolve=True)
        task_specs.append(
            {
                "id": str(task_cfg.id),
                "env_config": env_config_name,
                "env_name": str(env_cfg.env_name),
                "env_kwargs": dict(env_kwargs),
                "start_seed": int(env_kwargs.get("start_seed", cfg.seed)),
                "num_scenarios": int(env_kwargs.get("num_scenarios", 1)),
            }
        )
    return task_specs


def _build_env_for_task(task_spec, default_seed: int):
    return create_multiagent_env(
        env_name=str(task_spec["env_name"]),
        raw_env_kwargs=dict(task_spec["env_kwargs"]),
        default_start_seed=int(default_seed),
    )


def _reset_multiagent_env(env, seed: int):
    reset_ret = env.reset(seed=seed)
    if isinstance(reset_ret, tuple) and len(reset_ret) >= 1:
        obs = reset_ret[0]
        info = reset_ret[1] if len(reset_ret) > 1 else {}
        return obs, info
    return reset_ret, {}


def _task_episode_seed(task_spec, episode_idx: int) -> int:
    start_seed = int(task_spec["start_seed"])
    num_scenarios = max(int(task_spec["num_scenarios"]), 1)
    return start_seed + (int(episode_idx) % num_scenarios)


def _validate_task_spaces(task_specs, seed: int):
    reference_obs_space = None
    reference_action_space = None
    for task_spec in task_specs:
        env = _build_env_for_task(task_spec, seed)
        try:
            obs_space, action_space = _extract_single_agent_spaces(env)
            if reference_obs_space is None:
                reference_obs_space = obs_space
                reference_action_space = action_space
            else:
                if obs_space.shape != reference_obs_space.shape:
                    raise ValueError(
                        f"Observation shape mismatch for task={task_spec['id']}: "
                        f"{obs_space.shape} != {reference_obs_space.shape}"
                    )
                if action_space.shape != reference_action_space.shape:
                    raise ValueError(
                        f"Action shape mismatch for task={task_spec['id']}: "
                        f"{action_space.shape} != {reference_action_space.shape}"
                    )
        finally:
            _close_env(env)
    return reference_obs_space, reference_action_space


def _sample_actions(
    model: MetaDIME,
    obs_dict: Dict[str, np.ndarray],
    task_latent: np.ndarray,
    learning_starts: int,
    action_space: gym.Space,
):
    actions = {}
    for agent_id in sorted(obs_dict.keys()):
        obs = obs_dict[agent_id]
        if model.num_timesteps < learning_starts:
            action = action_space.sample()
        else:
            action, _ = model.predict_with_latent(obs, task_latent, deterministic=False)
        actions[agent_id] = np.asarray(action, dtype=np.float32)
    return actions


def _add_multiagent_transitions(
    task_buffer: TaskReplayBuffer,
    task_id: str,
    obs_dict: Dict[str, np.ndarray],
    next_obs_dict: Dict[str, np.ndarray],
    action_dict: Dict[str, np.ndarray],
    reward_dict: Dict[str, float],
    terminated_dict: Dict[str, bool],
    truncated_dict: Dict[str, bool],
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
        task_buffer.add(
            task_id=task_id,
            observation=np.asarray(obs, dtype=np.float32),
            action=action,
            next_observation=next_obs,
            reward=reward,
            done=done,
        )
        n += 1
    return n


def _save_meta_model(model: MetaDIME, out_dir: str, step: int):
    save_model_state(model.policy.actor_state, out_dir, "actor_state", step)
    save_model_state(model.policy.qf_state, out_dir, "critic_state", step)
    save_model_state(model.context_encoder_state, out_dir, "encoder_state", step)
    save_model_state(model.task_decoder_state, out_dir, "task_decoder_state", step)


def _create_meta_alg(cfg: DictConfig, task_specs):
    single_obs_space, single_action_space = _validate_task_spaces(task_specs, int(cfg.seed))
    policy_name = "MultiInputPolicy" if isinstance(single_obs_space, gym.spaces.Dict) else "MlpPolicy"

    run_start = time.strftime("%Y%m%d-%H%M%S")
    run_id = f"seed={cfg.seed}_start={run_start}"
    taskset_name = str(cfg.taskset.name)
    tensorboard_log_dir = f"./logs/{taskset_name}/{cfg.wandb['job_type']}/{run_id}/"
    model_save_dir = f"./checkpoints/{taskset_name}/{cfg.wandb['job_type']}/{run_id}/"
    best_model_dir = f"./best_models/{taskset_name}/{cfg.wandb['job_type']}/{run_id}/"
    os.makedirs(tensorboard_log_dir, exist_ok=True)
    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(best_model_dir, exist_ok=True)

    model = MetaDIME(
        policy_name,
        env=_SpaceOnlyEnv(single_obs_space, single_action_space),
        model_save_path=model_save_dir,
        save_every_n_steps=0,
        cfg=cfg,
        tensorboard_log=tensorboard_log_dir,
        replay_buffer_class=None,
    )
    model.set_logger(configure(tensorboard_log_dir, ["csv", "tensorboard"]))
    return model, best_model_dir


def _train_meta_multiagent(model: MetaDIME, task_specs, cfg: DictConfig, best_model_dir: str = ""):
    total_timesteps = int(cfg.tot_time_steps)
    log_freq = max(int(getattr(cfg, "log_freq", 100)), 1)
    save_every_n_steps = int(getattr(cfg, "save_every_n_steps", 10000))
    next_save_step = save_every_n_steps if save_every_n_steps > 0 else None
    learning_starts = int(cfg.alg.learning_starts)
    train_freq = int(model.train_freq.frequency)

    task_buffer = TaskReplayBuffer(
        task_ids=[task_spec["id"] for task_spec in task_specs],
        capacity_per_task=max(int(cfg.alg.buffer_size // max(len(task_specs), 1)), 10000),
    )
    rng = np.random.RandomState(int(cfg.seed))
    task_episode_counts = {task_spec["id"]: 0 for task_spec in task_specs}
    task_recent_success = {task_spec["id"]: deque(maxlen=50) for task_spec in task_specs}

    recent_success_rates = []
    best_mean_success_rate = float("-inf")
    task_lookup = {task_spec["id"]: task_spec for task_spec in task_specs}
    current_task_id = None
    env = None
    env_steps = 0
    total_agent_steps = 0
    model.num_timesteps = 0

    pbar = tqdm(total=total_timesteps, desc="env_steps", unit="step")
    try:
        while env_steps < total_timesteps:
            task_id = str(rng.choice([task_spec["id"] for task_spec in task_specs]))
            task_spec = task_lookup[task_id]

            if current_task_id != task_id:
                _close_env(env)
                env = _build_env_for_task(task_spec, int(cfg.seed))
                current_task_id = task_id

            episode_idx = task_episode_counts[task_id]
            task_episode_counts[task_id] += 1
            obs_dict, _ = _reset_multiagent_env(env, seed=_task_episode_seed(task_spec, episode_idx))
            if len(obs_dict) == 0:
                continue

            if task_buffer.size(task_id) >= int(getattr(cfg.meta, "online_context_size", 64)):
                task_latent = model.infer_latent_from_buffer(
                    task_buffer,
                    task_id=task_id,
                    context_batch_size=int(cfg.meta.online_context_size),
                    sample=bool(getattr(cfg.meta, "sample_posterior_during_rollout", False)),
                    recent=True,
                )
            else:
                task_latent = model.prior_latent(batch_size=1)[0]

            episode_group_reward = 0.0
            episode_len = 0
            episode_agent_steps = 0
            episode_vehicle_ids = set(obs_dict.keys())
            episode_success_count = 0
            episode_crash_count = 0

            while env_steps < total_timesteps:
                actions = _sample_actions(
                    model,
                    obs_dict,
                    task_latent,
                    learning_starts,
                    model.action_space,
                )
                next_obs_dict, reward_dict, terminated_dict, truncated_dict, info_dict = env.step(actions)
                done_all = bool(terminated_dict.get("__all__", False) or truncated_dict.get("__all__", False))
                episode_vehicle_ids.update(obs_dict.keys())
                episode_vehicle_ids.update(next_obs_dict.keys())

                added = _add_multiagent_transitions(
                    task_buffer=task_buffer,
                    task_id=task_id,
                    obs_dict=obs_dict,
                    next_obs_dict=next_obs_dict,
                    action_dict=actions,
                    reward_dict=reward_dict,
                    terminated_dict=terminated_dict,
                    truncated_dict=truncated_dict,
                )

                for agent_id, agent_info in info_dict.items():
                    if agent_id == "__all__":
                        continue
                    agent_done = bool(
                        terminated_dict.get(agent_id, False) or truncated_dict.get(agent_id, False)
                    )
                    if agent_done:
                        if agent_info.get("arrive_dest", False):
                            episode_success_count += 1
                        if agent_info.get("crash", False):
                            episode_crash_count += 1

                env_steps += 1
                model.num_timesteps = env_steps
                total_agent_steps += added
                episode_len += 1
                episode_agent_steps += added
                episode_group_reward += sum(float(v) for v in reward_dict.values())
                pbar.update(1)

                if env_steps >= learning_starts and train_freq > 0 and env_steps % train_freq == 0:
                    train_metrics = model.train_meta(task_buffer)
                    if train_metrics is not None:
                        for metric_name, metric_value in train_metrics.items():
                            model.logger.record(f"train/{metric_name}", float(metric_value))

                if next_save_step is not None and env_steps >= next_save_step:
                    model._save_model()
                    next_save_step += save_every_n_steps

                if env_steps % log_freq == 0:
                    model.logger.record("time/total_timesteps", env_steps, exclude="tensorboard")
                    model.logger.record("rollout/env_steps", env_steps)
                    model.logger.record("rollout/agent_steps_total", total_agent_steps)
                    model.logger.record("rollout/task_buffer_total", task_buffer.size())
                    model.logger.record(
                        "rollout/task_buffer_ready_tasks",
                        float(
                            len(
                                task_buffer.eligible_task_ids(
                                    int(getattr(cfg.meta, "min_task_buffer_size", 256))
                                )
                            )
                        ),
                    )
                    model.logger.dump(step=env_steps)

                if done_all:
                    break
                obs_dict = next_obs_dict

            episode_vehicle_count = max(len(episode_vehicle_ids), 1)
            ep_success_rate = episode_success_count / episode_vehicle_count
            task_recent_success[task_id].append(ep_success_rate)
            recent_success_rates.append(ep_success_rate)
            if len(recent_success_rates) > 20:
                recent_success_rates.pop(0)

            model.logger.record("rollout/task_episode_success_rate", ep_success_rate)
            model.logger.record(f"rollout/{task_id}_success_rate_recent", float(np.mean(task_recent_success[task_id])))
            model.logger.record(f"rollout/{task_id}_episode_reward", float(episode_group_reward))
            model.logger.record("rollout/episode_group_reward", episode_group_reward)
            model.logger.record("rollout/episode_vehicle_count", episode_vehicle_count)
            model.logger.record("rollout/episode_length", episode_len)
            model.logger.record("rollout/episode_agent_steps", episode_agent_steps)
            model.logger.record("rollout/episode_success_count", episode_success_count)
            model.logger.record("rollout/episode_crash_count", episode_crash_count)
            model.logger.record("rollout/current_task_index", float([t["id"] for t in task_specs].index(task_id)))
            model.logger.dump(step=env_steps)

            if len(recent_success_rates) >= 10 and best_model_dir:
                mean_sr = float(np.mean(recent_success_rates))
                if mean_sr > best_mean_success_rate:
                    best_mean_success_rate = mean_sr
                    _save_meta_model(model, best_model_dir, env_steps)
                    tqdm.write(
                        f"[Best Model] mean_success_rate(10 eps)={mean_sr:.2%} "
                        f"-> saved to {best_model_dir} (step {env_steps})"
                    )
    finally:
        pbar.close()
        _close_env(env)

    model._save_model()
    print(f"Final meta checkpoint saved at step {model.num_timesteps}")


def initialize_and_run(cfg):
    cfg = hydra.utils.instantiate(cfg)
    task_specs = _load_task_specs(cfg)
    seed = int(cfg.seed)

    if cfg.wandb["activate"]:
        name = f"seed_{seed}"
        wandb_config = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        wandb.init(
            settings=wandb.Settings(_service_wait=300),
            project=cfg.wandb["project"],
            group=f"{cfg.wandb['group']}_{cfg.taskset.name}",
            job_type=cfg.wandb["job_type"],
            name=name,
            config=wandb_config,
            entity=cfg.wandb["entity"],
            sync_tensorboard=True,
        )
        if is_slurm_job():
            print(f"SLURM_JOB_ID: {os.environ.get('SLURM_JOB_ID')}")
            wandb.summary["SLURM_JOB_ID"] = os.environ.get("SLURM_JOB_ID")

    model, best_model_dir = _create_meta_alg(cfg, task_specs)
    _train_meta_multiagent(model, task_specs, cfg, best_model_dir=best_model_dir)


@hydra.main(version_base=None, config_path="configs", config_name="base_meta")
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
