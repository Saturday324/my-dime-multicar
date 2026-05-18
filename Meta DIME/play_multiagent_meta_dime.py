import argparse
import os
from typing import Dict, Tuple

import flax
import gymnasium as gym
import hydra
import numpy as np
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from common.multiagent_env_factory import create_multiagent_env
from diffusion.meta_dime import MetaDIME
from meta.task_replay_buffer import TaskReplayBuffer


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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Meta DIME on a configured multi-agent taskset."
    )
    parser.add_argument("--checkpoint-dir", type=str, required=True, help="Directory with msgpack checkpoints.")
    parser.add_argument("--actor-step", type=int, required=True, help="Actor checkpoint step number.")
    parser.add_argument("--critic-step", type=int, default=None, help="Critic checkpoint step number.")
    parser.add_argument("--encoder-step", type=int, default=None, help="Encoder checkpoint step number.")
    parser.add_argument("--episodes-per-task", type=int, default=3, help="Episodes to run for each task.")
    parser.add_argument("--task-id", action="append", default=None, help="Only evaluate selected task id. Repeatable.")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional hard cap of rollout steps per episode.")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic policy output.")
    parser.add_argument(
        "--latent-mode",
        type=str,
        default="posterior",
        choices=["posterior", "prior"],
        help="Use inferred posterior latent or the zero prior latent.",
    )
    parser.add_argument(
        "--latent-update-freq",
        type=int,
        default=25,
        help="Refresh posterior latent every N env steps after context is available. Use 0 for episode-start only.",
    )
    parser.add_argument(
        "--context-size",
        type=int,
        default=None,
        help="Override cfg.meta.online_context_size for online posterior inference.",
    )
    parser.add_argument(
        "--sample-posterior",
        action="store_true",
        help="Sample z from q(z|context) instead of using posterior mean during evaluation.",
    )
    parser.add_argument(
        "--render-mode",
        type=str,
        default="rgb_array",
        choices=["human", "rgb_array"],
        help="Environment render mode.",
    )
    parser.add_argument(
        "--video-path",
        type=str,
        default=None,
        help="Optional mp4 path. With multiple task/episode rollouts, task and episode suffixes are added.",
    )
    parser.add_argument("--allow-respawn", action="store_true", help="Allow vehicles to respawn after death.")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Extra Hydra overrides for cfg loading, e.g. --override taskset=metadrive_ma_5env",
    )
    return parser.parse_args()


def load_cfg(extra_overrides=None):
    config_dir = os.path.join(os.path.dirname(__file__), "configs")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="base_meta", overrides=list(extra_overrides or []))
    return hydra.utils.instantiate(cfg)


def _load_task_specs(cfg):
    env_dir = os.path.join(os.path.dirname(__file__), "configs", "env")
    task_specs = []
    for task_cfg in cfg.taskset.tasks:
        env_config_name = str(task_cfg.env_config)
        env_cfg = OmegaConf.load(os.path.join(env_dir, f"{env_config_name}.yaml"))
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


def _build_env_for_task(task_spec, cfg, args):
    env_kwargs = dict(task_spec["env_kwargs"])
    env_kwargs["use_render"] = args.render_mode == "human"
    env_kwargs["allow_respawn"] = bool(args.allow_respawn)
    return create_multiagent_env(
        env_name=str(task_spec["env_name"]),
        raw_env_kwargs=env_kwargs,
        default_start_seed=int(cfg.seed),
    )


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


def _render_frame(env, render_mode: str):
    if render_mode != "rgb_array":
        return None
    return env.render(mode="top_down", window=False, screen_size=(1000, 1000), film_size=(1000, 1000))


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


def _extract_single_agent_spaces(ma_env) -> Tuple[gym.Space, gym.Space]:
    if not isinstance(ma_env.observation_space, gym.spaces.Dict):
        raise TypeError("Expected multi-agent observation_space to be gym.spaces.Dict")
    if not isinstance(ma_env.action_space, gym.spaces.Dict):
        raise TypeError("Expected multi-agent action_space to be gym.spaces.Dict")
    agent_ids = sorted(ma_env.observation_space.spaces.keys())
    if len(agent_ids) == 0:
        raise ValueError("No agent spaces found in multi-agent environment.")
    first_agent = agent_ids[0]
    return ma_env.observation_space.spaces[first_agent], ma_env.action_space.spaces[first_agent]


def _validate_task_spaces(task_specs, cfg, args):
    reference_obs_space = None
    reference_action_space = None
    for task_spec in task_specs:
        env = _build_env_for_task(task_spec, cfg, args)
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


def build_meta_model(cfg, single_obs_space: gym.Space, single_action_space: gym.Space):
    space_env = _SpaceOnlyEnv(observation_space=single_obs_space, action_space=single_action_space)
    policy_name = "MultiInputPolicy" if isinstance(single_obs_space, gym.spaces.Dict) else "MlpPolicy"
    return MetaDIME(
        policy_name,
        env=space_env,
        model_save_path=None,
        save_every_n_steps=1,
        cfg=cfg,
        tensorboard_log=None,
        replay_buffer_class=None,
    )


def _checkpoint_file(checkpoint_dir: str, name: str, step: int) -> str:
    path = os.path.join(checkpoint_dir, f"{name}_{step}.msgpack")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


def load_meta_checkpoint(model: MetaDIME, args):
    critic_step = args.critic_step if args.critic_step is not None else args.actor_step
    encoder_step = args.encoder_step if args.encoder_step is not None else args.actor_step
    actor_path = _checkpoint_file(args.checkpoint_dir, "actor_state", args.actor_step)
    critic_path = _checkpoint_file(args.checkpoint_dir, "critic_state", critic_step)
    encoder_path = _checkpoint_file(args.checkpoint_dir, "encoder_state", encoder_step)

    model.load_model_files(actor_path, critic_path)
    with open(encoder_path, "rb") as f:
        model.context_encoder_state = flax.serialization.from_bytes(
            model.context_encoder_state,
            f.read(),
        )
    task_decoder_path = os.path.join(args.checkpoint_dir, f"task_decoder_state_{encoder_step}.msgpack")
    if os.path.exists(task_decoder_path):
        with open(task_decoder_path, "rb") as f:
            model.task_decoder_state = flax.serialization.from_bytes(
                model.task_decoder_state,
                f.read(),
            )
    return critic_step, encoder_step


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
    added = 0
    for agent_id, obs in obs_dict.items():
        action = np.asarray(action_dict[agent_id], dtype=np.float32)
        next_obs = next_obs_dict.get(agent_id, None)
        if next_obs is None:
            next_obs = np.zeros_like(obs, dtype=np.float32)
        done = bool(terminated_dict.get(agent_id, False) or truncated_dict.get(agent_id, False))
        task_buffer.add(
            task_id=task_id,
            observation=np.asarray(obs, dtype=np.float32),
            action=action,
            next_observation=np.asarray(next_obs, dtype=np.float32),
            reward=float(reward_dict.get(agent_id, 0.0)),
            done=done,
        )
        added += 1
    return added


def _infer_task_latent(model: MetaDIME, task_buffer: TaskReplayBuffer, task_id: str, args, context_size: int):
    if args.latent_mode == "prior" or task_buffer.size(task_id) < context_size:
        return model.prior_latent(batch_size=1)[0]
    return model.infer_latent_from_buffer(
        task_buffer,
        task_id=task_id,
        context_batch_size=context_size,
        sample=bool(args.sample_posterior),
        recent=True,
    )


def _predict_actions_for_all_agents(
    model: MetaDIME,
    obs_dict: Dict[str, np.ndarray],
    task_latent: np.ndarray,
    deterministic: bool,
) -> Dict[str, np.ndarray]:
    actions = {}
    for agent_id in sorted(obs_dict.keys()):
        action, _ = model.predict_with_latent(obs_dict[agent_id], task_latent, deterministic=deterministic)
        actions[agent_id] = np.asarray(action, dtype=np.float32)
    return actions


def _episode_video_path(base_path: str, task_id: str, episode_index: int, total_rollouts: int) -> str:
    root, ext = os.path.splitext(base_path)
    if not ext:
        ext = ".mp4"
    if total_rollouts > 1:
        return f"{root}_{task_id}_ep{episode_index + 1}{ext}"
    return f"{root}{ext}"


def evaluate_task(model: MetaDIME, cfg, task_spec, args, task_buffer, context_size: int, total_rollouts: int):
    env = _build_env_for_task(task_spec, cfg, args)
    task_id = str(task_spec["id"])
    imageio = None
    if args.video_path is not None:
        if args.render_mode != "rgb_array":
            raise ValueError("Saving video requires --render-mode rgb_array.")
        import imageio.v2 as imageio

    task_metrics = []
    try:
        for ep in range(int(args.episodes_per_task)):
            frames = []
            obs_dict, _ = _reset_multiagent_env(env, seed=_task_episode_seed(task_spec, ep))
            task_latent = _infer_task_latent(model, task_buffer, task_id, args, context_size)
            done_all = False
            ep_reward = 0.0
            ep_len = 0
            ep_vehicle_ids = set(obs_dict.keys())
            ep_success = 0
            ep_crash = 0
            latent_updates = 0

            while not done_all:
                actions = _predict_actions_for_all_agents(
                    model,
                    obs_dict,
                    task_latent,
                    deterministic=bool(args.deterministic),
                )
                next_obs_dict, reward_dict, terminated_dict, truncated_dict, info_dict = env.step(actions)
                done_all = bool(terminated_dict.get("__all__", False) or truncated_dict.get("__all__", False))
                ep_vehicle_ids.update(obs_dict.keys())
                ep_vehicle_ids.update(next_obs_dict.keys())

                _add_multiagent_transitions(
                    task_buffer,
                    task_id,
                    obs_dict,
                    next_obs_dict,
                    actions,
                    reward_dict,
                    terminated_dict,
                    truncated_dict,
                )

                ep_reward += sum(float(v) for v in reward_dict.values())
                ep_len += 1

                if (
                    args.latent_mode == "posterior"
                    and int(args.latent_update_freq) > 0
                    and ep_len % int(args.latent_update_freq) == 0
                ):
                    task_latent = _infer_task_latent(model, task_buffer, task_id, args, context_size)
                    latent_updates += 1

                for agent_id, agent_info in info_dict.items():
                    if agent_id == "__all__":
                        continue
                    agent_done = bool(
                        terminated_dict.get(agent_id, False) or truncated_dict.get(agent_id, False)
                    )
                    if agent_done:
                        if agent_info.get("arrive_dest", False):
                            ep_success += 1
                        if agent_info.get("crash", False):
                            ep_crash += 1

                if args.max_steps is not None and ep_len >= int(args.max_steps):
                    done_all = True

                if args.render_mode == "rgb_array":
                    frame = _render_frame(env, args.render_mode)
                    if args.video_path is not None and frame is not None:
                        frames.append(np.asarray(frame))

                obs_dict = next_obs_dict

            vehicle_count = max(len(ep_vehicle_ids), 1)
            success_rate = ep_success / vehicle_count
            crash_rate = ep_crash / vehicle_count
            task_metrics.append(
                {
                    "reward": ep_reward,
                    "length": ep_len,
                    "success_rate": success_rate,
                    "crash_rate": crash_rate,
                }
            )
            print(
                f"Task {task_id} episode {ep + 1}: reward={ep_reward:.3f}, "
                f"length={ep_len}, vehicles={vehicle_count}, success={ep_success}, "
                f"crash={ep_crash}, success_rate={success_rate:.2%}, "
                f"latent_updates={latent_updates}"
            )

            if args.video_path is not None:
                if len(frames) == 0:
                    raise RuntimeError(f"No frames were captured for task={task_id} episode={ep + 1}.")
                out_path = _episode_video_path(args.video_path, task_id, ep, total_rollouts)
                imageio.mimsave(out_path, frames, fps=20)
                print(f"Saved video: {out_path}")
    finally:
        _close_env(env)
    return task_metrics


def main():
    args = parse_args()
    cfg = load_cfg(extra_overrides=args.override)
    task_specs = _load_task_specs(cfg)
    if args.task_id is not None:
        selected = set(str(task_id) for task_id in args.task_id)
        task_specs = [task_spec for task_spec in task_specs if task_spec["id"] in selected]
        missing = selected.difference(task_spec["id"] for task_spec in task_specs)
        if missing:
            raise ValueError(f"Unknown task_id(s): {sorted(missing)}")
    if len(task_specs) == 0:
        raise ValueError("No tasks selected for evaluation.")

    single_obs_space, single_action_space = _validate_task_spaces(task_specs, cfg, args)
    model = build_meta_model(cfg, single_obs_space, single_action_space)
    critic_step, encoder_step = load_meta_checkpoint(model, args)

    context_size = int(args.context_size or getattr(cfg.meta, "online_context_size", 64))
    task_buffer = TaskReplayBuffer(
        task_ids=[task_spec["id"] for task_spec in task_specs],
        capacity_per_task=max(context_size * max(int(args.episodes_per_task), 1) * 100, 10000),
    )

    print("Taskset:", cfg.taskset.name)
    print("Tasks:", ", ".join(task_spec["id"] for task_spec in task_specs))
    print("Loaded actor/critic/encoder steps:", args.actor_step, critic_step, encoder_step)
    print("Latent mode:", args.latent_mode, "context_size:", context_size)
    print("Deterministic:", bool(args.deterministic))

    total_rollouts = len(task_specs) * int(args.episodes_per_task)
    all_metrics = []
    for task_spec in task_specs:
        task_metrics = evaluate_task(model, cfg, task_spec, args, task_buffer, context_size, total_rollouts)
        all_metrics.extend(task_metrics)
        mean_sr = float(np.mean([m["success_rate"] for m in task_metrics]))
        mean_crash = float(np.mean([m["crash_rate"] for m in task_metrics]))
        mean_reward = float(np.mean([m["reward"] for m in task_metrics]))
        print(
            f"Task {task_spec['id']} summary: mean_reward={mean_reward:.3f}, "
            f"mean_success_rate={mean_sr:.2%}, mean_crash_rate={mean_crash:.2%}"
        )

    print(
        "Overall summary: "
        f"mean_reward={float(np.mean([m['reward'] for m in all_metrics])):.3f}, "
        f"mean_success_rate={float(np.mean([m['success_rate'] for m in all_metrics])):.2%}, "
        f"mean_crash_rate={float(np.mean([m['crash_rate'] for m in all_metrics])):.2%}"
    )


if __name__ == "__main__":
    main()
