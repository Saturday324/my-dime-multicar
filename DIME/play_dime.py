import argparse
import os

import hydra
import numpy as np
from hydra import compose, initialize_config_dir

from common.buffers import DMCCompatibleDictReplayBuffer
from common.env_factory import create_env, resolve_env_kwargs
from diffusion.dime import DIME


def _is_metadrive_env_name(env_name: str) -> bool:
    return "metadrive" in env_name.lower()


def _render_frame(env, env_name: str, render_mode: str):
    if render_mode != "rgb_array":
        return None
    if _is_metadrive_env_name(env_name):
        return env.render(mode="top_down", window=False, screen_size=(1000, 1000), film_size=(1000, 1000))
    return env.render()


def parse_args():
    parser = argparse.ArgumentParser(description="Load a trained DIME checkpoint and run episodes.")
    parser.add_argument("--env-name", type=str, default=None, help="Gym/Gymnasium env name override.")
    parser.add_argument("--checkpoint-dir", type=str, required=True, help="Directory with msgpack checkpoints.")
    parser.add_argument("--actor-step", type=int, required=True, help="Actor checkpoint step number.")
    parser.add_argument("--critic-step", type=int, default=None, help="Critic checkpoint step number.")
    parser.add_argument("--episodes", type=int, default=3, help="Number of rollout episodes.")
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
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic policy output.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Extra Hydra overrides, e.g. --override alg.critic.v_min=-1600",
    )
    return parser.parse_args()


def load_cfg(env_name_override=None, extra_overrides=None):
    config_dir = os.path.join(os.path.dirname(__file__), "configs")
    overrides = list(extra_overrides or [])
    if env_name_override is not None:
        overrides.append(f"env_name={env_name_override}")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="base", overrides=overrides)
    return hydra.utils.instantiate(cfg)


def resolve_replay_buffer_class(env_name):
    env_name_split = env_name.split("/")
    if env_name_split[0] == "dm_control":
        task_prefix = env_name_split[1].split("-")[0]
        if task_prefix in ["humanoid", "fish", "walker", "quadruped", "finger"]:
            return DMCCompatibleDictReplayBuffer
    return None


def build_model(cfg, render_mode):
    import gymnasium as gym
    env_kwargs = resolve_env_kwargs(getattr(cfg, "env_kwargs", None))
    env = create_env(cfg.env_name, env_kwargs=env_kwargs, render_mode=render_mode)
    rb_class = resolve_replay_buffer_class(cfg.env_name)
    model = DIME(
        "MultiInputPolicy" if isinstance(env.observation_space, gym.spaces.Dict) else "MlpPolicy",
        env=env,
        model_save_path=None,
        save_every_n_steps=1,
        cfg=cfg,
        tensorboard_log=None,
        replay_buffer_class=rb_class,
    )
    return model, env


def main():
    args = parse_args()
    cfg = load_cfg(env_name_override=args.env_name, extra_overrides=args.override)

    critic_step = args.critic_step if args.critic_step is not None else args.actor_step
    actor_ckpt = os.path.join(args.checkpoint_dir, f"actor_state_{args.actor_step}.msgpack")
    critic_ckpt = os.path.join(args.checkpoint_dir, f"critic_state_{critic_step}.msgpack")
    if not os.path.exists(actor_ckpt):
        raise FileNotFoundError(f"Actor checkpoint not found: {actor_ckpt}")
    if not os.path.exists(critic_ckpt):
        raise FileNotFoundError(f"Critic checkpoint not found: {critic_ckpt}")

    model, env = build_model(cfg, args.render_mode)
    model.load_model(args.checkpoint_dir, args.actor_step, critic_step)

    print("Loaded config env:", cfg.env_name)
    print("Loaded actor step:", args.actor_step)
    print("Loaded critic step:", critic_step)
    print("Deterministic:", bool(args.deterministic))

    imageio = None
    if args.video_path is not None:
        if args.render_mode != "rgb_array":
            raise ValueError("Saving video requires --render-mode rgb_array.")
        import imageio.v2 as imageio

    for ep in range(args.episodes):
        frames = []
        obs, _ = env.reset(seed=cfg.seed + ep)
        done = False
        ep_reward = 0.0
        ep_len = 0
        while not done:
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
            ep_reward += float(reward)
            ep_len += 1

            if args.render_mode == "rgb_array":
                frame = _render_frame(env, cfg.env_name, args.render_mode)
                if args.video_path is not None and frame is not None:
                    frames.append(np.asarray(frame))

        print(f"Episode {ep + 1}: reward={ep_reward:.3f}, length={ep_len}")

        if args.video_path is not None:
            if len(frames) == 0:
                raise RuntimeError(
                    f"No frames were captured for episode {ep + 1}. Use --render-mode rgb_array when saving video."
                )

            root, ext = os.path.splitext(args.video_path)
            if not ext:
                ext = ".mp4"
            if args.episodes > 1:
                episode_video_path = f"{root}_ep{ep + 1}{ext}"
            else:
                episode_video_path = f"{root}{ext}"

            os.makedirs(os.path.dirname(episode_video_path) or ".", exist_ok=True)
            imageio.mimsave(episode_video_path, frames, fps=30)
            print(f"Saved video to: {episode_video_path}")

    env.close()


if __name__ == "__main__":
    main()
