import argparse
import json
import os

import imageio.v2 as imageio
import numpy as np

from common.env_factory import create_env
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
    parser = argparse.ArgumentParser(
        description="Load best_model.zip and run DIME policy rollout."
    )
    parser.add_argument(
        "--best-model-path",
        type=str,
        required=True,
        help="Path to best_model.zip",
    )
    parser.add_argument(
        "--env-name",
        type=str,
        default="Humanoid-v4",
        help="Gym/Gymnasium environment name",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Number of episodes to run",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base seed for environment reset",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic action selection",
    )
    parser.add_argument(
        "--render-mode",
        type=str,
        default="rgb_array",
        choices=["human", "rgb_array"],
        help="Render mode, use rgb_array for mp4 export",
    )
    parser.add_argument(
        "--video-path",
        type=str,
        default=None,
        help="Optional output mp4 path (records first episode only)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Optional hard cap per episode, 0 means no extra cap",
    )
    parser.add_argument(
        "--mujoco-gl",
        type=str,
        default="",
        choices=["", "egl", "osmesa", "glfw"],
        help="Optional MuJoCo GL backend override.",
    )
    parser.add_argument(
        "--env-kwargs-json",
        type=str,
        default="{}",
        help="Optional JSON dict passed to env constructor.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.mujoco_gl:
        os.environ["MUJOCO_GL"] = args.mujoco_gl
        if args.mujoco_gl == "egl":
            os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
        elif args.mujoco_gl == "osmesa":
            os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

    if not os.path.exists(args.best_model_path):
        raise FileNotFoundError(f"best model not found: {args.best_model_path}")

    try:
        env_kwargs = json.loads(args.env_kwargs_json)
    except json.JSONDecodeError as ex:
        raise ValueError(f"Invalid --env-kwargs-json: {args.env_kwargs_json}") from ex
    if not isinstance(env_kwargs, dict):
        raise ValueError("--env-kwargs-json must decode to a JSON object.")

    env = create_env(args.env_name, env_kwargs=env_kwargs, render_mode=args.render_mode)
    model = DIME.load(args.best_model_path, env=env)

    print(f"Loaded model: {args.best_model_path}")
    print(f"Environment: {args.env_name}")
    print(f"Deterministic: {args.deterministic}")

    frames = []
    returns = []

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        done = False
        ep_reward = 0.0
        ep_len = 0

        while not done:
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
            ep_reward += float(reward)
            ep_len += 1

            if args.max_steps > 0 and ep_len >= args.max_steps:
                done = True

            if args.render_mode == "rgb_array":
                frame = _render_frame(env, args.env_name, args.render_mode)
                if args.video_path is not None and ep == 0 and frame is not None:
                    frames.append(np.asarray(frame))

        returns.append(ep_reward)
        print(f"Episode {ep + 1}: reward={ep_reward:.3f}, length={ep_len}")

    env.close()

    if args.video_path is not None:
        if args.render_mode != "rgb_array":
            raise ValueError("Saving video requires --render-mode rgb_array.")
        if len(frames) == 0:
            raise RuntimeError("No frame captured; cannot export mp4.")
        os.makedirs(os.path.dirname(args.video_path) or ".", exist_ok=True)
        imageio.mimsave(args.video_path, frames, fps=30)
        print(f"Saved video: {args.video_path}")

    mean_return = float(np.mean(np.asarray(returns))) if len(returns) > 0 else 0.0
    print(f"Mean return over {len(returns)} episode(s): {mean_return:.3f}")


if __name__ == "__main__":
    main()
"""
xvfb-run -s "-screen 0 1280x720x24" python play_best_dime.py \
  --best-model-path "./best_models/DIME_MountainCarContinuous-v0/lr0.0003/seed=0_start=20260303-021156/best_model.zip" \
  --env-name MountainCarContinuous-v0 \
  --episodes 1 \
  --deterministic \
  --render-mode rgb_array \
  --video-path "./videos/mountaincar_best.mp4"
"""