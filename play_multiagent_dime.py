import argparse
import os
from typing import Dict, Tuple

import gymnasium as gym
import hydra
import numpy as np
from hydra import compose, initialize_config_dir

from common.env_factory import resolve_env_kwargs
from common.multiagent_env_factory import create_multiagent_env
from diffusion.dime import DIME


def _render_frame(env, render_mode: str):
    if render_mode != "rgb_array":
        return None
    return env.render(mode="top_down", window=False, screen_size=(1000, 1000), film_size=(1000, 1000))


class _SpaceOnlyEnv(gym.Env):
    """Minimal env to provide spaces for shared-policy construction."""

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
        description="Play DIME in MetaDrive multi-agent environments using shared actor/critic."
    )
    parser.add_argument(
        "--env-name",
        type=str,
        default=None,
        help="Multi-agent env class shortcut override, e.g. metadrive/MultiAgentIntersectionEnv.",
    )
    parser.add_argument("--checkpoint-dir", type=str, required=True, help="Directory with msgpack checkpoints.")
    parser.add_argument("--actor-step", type=int, required=True, help="Actor checkpoint step number.")
    parser.add_argument("--critic-step", type=int, default=None, help="Critic checkpoint step number.")
    parser.add_argument("--episodes", type=int, default=3, help="Number of rollout episodes.")
    parser.add_argument("--num-agents", type=int, default=None, help="Override fixed number of agents.")
    parser.add_argument("--start-seed", type=int, default=None, help="Override episode seed base.")
    parser.add_argument("--num-scenarios", type=int, default=None, help="Override scenario count.")
    parser.add_argument("--horizon", type=int, default=None, help="Override per-agent horizon.")
    parser.add_argument("--traffic-density", type=float, default=None, help="Override traffic density.")
    parser.add_argument("--num-lasers", type=int, default=None, help="Override lidar rays.")
    parser.add_argument("--num-others", type=int, default=None, help="Override nearby-vehicle lidar channels.")
    parser.add_argument("--lidar-distance", type=float, default=None, help="Override lidar sensing distance.")
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
        help="Optional mp4 path. If episodes>1, saves one file per episode with _epN suffix.",
    )
    parser.add_argument("--max-steps", type=int, default=None, help="Optional hard cap of rollout steps per episode.")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic policy output.")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Extra Hydra overrides for cfg loading, e.g. --override alg.critic.v_min=-300",
    )
    return parser.parse_args()


def load_cfg(extra_overrides=None):
    config_dir = os.path.join(os.path.dirname(__file__), "configs")
    overrides = list(extra_overrides or [])
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="base", overrides=overrides)
    return hydra.utils.instantiate(cfg)


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


def build_multiagent_env(cfg, args):
    env_name = args.env_name if args.env_name is not None else str(cfg.env_name)
    env_kwargs = resolve_env_kwargs(getattr(cfg, "env_kwargs", None))
    env_kwargs = dict(env_kwargs)
    env_kwargs["use_render"] = args.render_mode == "human"
    env_kwargs["allow_respawn"] = False

    if args.num_agents is not None:
        env_kwargs["num_agents"] = int(args.num_agents)
    if args.start_seed is not None:
        env_kwargs["start_seed"] = int(args.start_seed)
    if args.num_scenarios is not None:
        env_kwargs["num_scenarios"] = int(args.num_scenarios)
    if args.horizon is not None:
        env_kwargs["horizon"] = int(args.horizon)
    if args.traffic_density is not None:
        env_kwargs["traffic_density"] = float(args.traffic_density)

    if any(v is not None for v in [args.num_lasers, args.num_others, args.lidar_distance]):
        vehicle_config = dict(env_kwargs.get("vehicle_config", {}))
        lidar_cfg = dict(vehicle_config.get("lidar", {}))
        if args.num_lasers is not None:
            lidar_cfg["num_lasers"] = int(args.num_lasers)
        if args.num_others is not None:
            lidar_cfg["num_others"] = int(args.num_others)
        if args.lidar_distance is not None:
            lidar_cfg["distance"] = float(args.lidar_distance)
        vehicle_config["lidar"] = lidar_cfg
        env_kwargs["vehicle_config"] = vehicle_config

    return create_multiagent_env(env_name, env_kwargs, default_start_seed=int(cfg.seed))


def build_shared_model(cfg, single_obs_space: gym.Space, single_action_space: gym.Space):
    space_env = _SpaceOnlyEnv(observation_space=single_obs_space, action_space=single_action_space)
    policy_name = "MultiInputPolicy" if isinstance(single_obs_space, gym.spaces.Dict) else "MlpPolicy"
    model = DIME(
        policy_name,
        env=space_env,
        model_save_path=None,
        save_every_n_steps=1,
        cfg=cfg,
        tensorboard_log=None,
        replay_buffer_class=None,
    )
    return model


def _predict_actions_for_all_agents(model, obs_dict: Dict[str, np.ndarray], deterministic: bool) -> Dict[str, np.ndarray]:
    actions = {}
    for agent_id in sorted(obs_dict.keys()):
        action, _ = model.predict(obs_dict[agent_id], deterministic=deterministic)
        actions[agent_id] = np.asarray(action)
    return actions


def _episode_video_path(base_path: str, episode_index: int, total_episodes: int) -> str:
    root, ext = os.path.splitext(base_path)
    if not ext:
        ext = ".mp4"
    if total_episodes > 1:
        return f"{root}_ep{episode_index + 1}{ext}"
    return f"{root}{ext}"


def _reset_multiagent_env(env, seed: int):
    reset_ret = env.reset(seed=seed)
    if isinstance(reset_ret, tuple) and len(reset_ret) >= 1:
        obs = reset_ret[0]
        info = reset_ret[1] if len(reset_ret) > 1 else {}
        return obs, info
    return reset_ret, {}


def main():
    args = parse_args()
    cfg = load_cfg(extra_overrides=args.override)

    critic_step = args.critic_step if args.critic_step is not None else args.actor_step
    actor_ckpt = os.path.join(args.checkpoint_dir, f"actor_state_{args.actor_step}.msgpack")
    critic_ckpt = os.path.join(args.checkpoint_dir, f"critic_state_{critic_step}.msgpack")
    if not os.path.exists(actor_ckpt):
        raise FileNotFoundError(f"Actor checkpoint not found: {actor_ckpt}")
    if not os.path.exists(critic_ckpt):
        raise FileNotFoundError(f"Critic checkpoint not found: {critic_ckpt}")

    env = build_multiagent_env(cfg, args)
    try:
        single_obs_space, single_action_space = _extract_single_agent_spaces(env)
        model = build_shared_model(cfg, single_obs_space, single_action_space)
        model.load_model(args.checkpoint_dir, args.actor_step, critic_step)

        print("Environment:", str(args.env_name if args.env_name is not None else cfg.env_name))
        print("Shared policy spaces:", single_obs_space, single_action_space)
        print("Loaded actor step:", args.actor_step)
        print("Loaded critic step:", critic_step)
        print("Deterministic:", bool(args.deterministic))
        print("num_agents:", env.config.get("num_agents", "unknown"), "allow_respawn:", False)

        imageio = None
        if args.video_path is not None:
            if args.render_mode != "rgb_array":
                raise ValueError("Saving video requires --render-mode rgb_array.")
            import imageio.v2 as imageio

        scenario_count = int(env.config.get("num_scenarios", 1))
        if scenario_count <= 0:
            raise ValueError("--num-scenarios must be >= 1")
        if args.episodes > scenario_count:
            print(
                f"episodes ({args.episodes}) > num_scenarios ({scenario_count}), "
                "will cycle seeds within valid scenario range."
            )
        start_seed = int(env.config.get("start_seed", cfg.seed))

        for ep in range(args.episodes):
            frames = []
            episode_seed = start_seed + (ep % scenario_count)
            obs_dict, _ = _reset_multiagent_env(env, seed=episode_seed)
            done_all = False
            ep_reward = 0.0
            ep_len = 0

            while not done_all:
                actions = _predict_actions_for_all_agents(model, obs_dict, deterministic=args.deterministic)
                obs_dict, reward_dict, terminated_dict, truncated_dict, _ = env.step(actions)
                done_all = bool(terminated_dict.get("__all__", False) or truncated_dict.get("__all__", False))
                ep_reward += sum(float(v) for v in reward_dict.values())
                ep_len += 1

                if args.max_steps is not None and ep_len >= args.max_steps:
                    done_all = True

                if args.render_mode == "rgb_array":
                    frame = _render_frame(env, args.render_mode)
                    if args.video_path is not None and frame is not None:
                        frames.append(np.asarray(frame))

            print(
                f"Episode {ep + 1}: group_reward={ep_reward:.3f}, "
                f"length={ep_len}, alive_agents_end={len(obs_dict)}"
            )

            if args.video_path is not None:
                if len(frames) == 0:
                    raise RuntimeError(
                        f"No frames were captured for episode {ep + 1}. Use --render-mode rgb_array when saving video."
                    )
                output_path = _episode_video_path(args.video_path, ep, args.episodes)
                os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
                imageio.mimsave(output_path, frames, fps=30)
                print(f"Saved video to: {output_path}")

    finally:
        env.close()
        try:
            from metadrive.engine.engine_utils import close_engine

            close_engine()
        except Exception:
            pass


if __name__ == "__main__":
    main()
'''
python play_multiagent_dime.py \
  --checkpoint-dir "<你的checkpoint目录>" \
  --actor-step <actor步数> \
  --critic-step <critic步数> \
  --episodes 5 \
  --deterministic \
  --render-mode rgb_array \
  --video-path "/workspace/diffusion-rl/DIME/videos/ma_eval.mp4" \
  --override env=metadrive_ma_roundabout
'''