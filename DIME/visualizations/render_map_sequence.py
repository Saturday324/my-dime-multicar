
"""

cd /workspace/diffusion-rl/DIME

conda run -n dime python visualizations/render_map_sequence.py OYUrB \
  --traffic-density 0.0 \
  --traffic-mode basic \
  --traffic-policy idm \
  --scene-steps 30

conda run -n dime python visualizations/render_map_sequence.py yRTOU

cd /workspace/diffusion-rl/DIME
conda run -n dime python visualizations/render_map_sequence.py BCPYS --num-agents 12

"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import hydra
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from hydra import compose, initialize_config_dir

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
METADRIVE_SRC_DIR = os.path.join(ROOT_DIR, "metadrive")
for path in (ROOT_DIR, METADRIVE_SRC_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from common.env_factory import resolve_env_kwargs
from common.multiagent_env_factory import create_multiagent_env
from metadrive.utils.draw_top_down_map import draw_top_down_map

SUPPORTED_BLOCK_IDS = {"S", "C", "r", "R", "O", "X", "y", "Y", "$", "P", "T", "U", "B"}
UNSUPPORTED_BLOCK_IDS = {
    "f": "InFork",
    "F": "OutFork",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render a composed MetaDrive map and an initial multi-agent scene from a block-sequence string."
    )
    parser.add_argument(
        "map_sequence",
        help="Block-sequence string, e.g. XTXTS, SXTXS or CrXROSTR.",
    )
    parser.add_argument(
        "--env",
        default="metadrive_ma_composed",
        help="Hydra env config name. Defaults to metadrive_ma_composed.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Scenario seed used for reset and file naming.",
    )
    parser.add_argument(
        "--num-scenarios",
        type=int,
        default=1,
        help="Number of scenarios in the env config. Usually keep this as 1 for single-image rendering.",
    )
    parser.add_argument(
        "--num-agents",
        type=int,
        default=20,
        help="Number of controlled agents shown in the scene image.",
    )
    parser.add_argument(
        "--traffic-mode",
        default="basic",
        choices=["basic", "resident", "trigger", "respawn", "hybrid"],
        help="Background traffic mode for the rendered scene.",
    )
    parser.add_argument(
        "--traffic-density",
        type=float,
        default=0.0,
        help="Background traffic density. Default is 0.0 to keep rendering lightweight.",
    )
    parser.add_argument(
        "--traffic-policy",
        default="idm",
        choices=["idm", "expert", "mixed"],
        help="Background traffic policy. Default is idm to avoid GPU-heavy expert traffic during visualization.",
    )
    parser.add_argument(
        "--scene-steps",
        type=int,
        default=0,
        help="Optional random rollout steps before capturing the scene image.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=1024,
        help="Output image width and height in pixels.",
    )
    parser.add_argument(
        "--output-dir",
        default="visualizations/map_sequence_renders",
        help="Root directory for generated files.",
    )
    return parser.parse_args()


def validate_map_sequence(map_sequence: str):
    bad_blocks = [ch for ch in map_sequence if ch in UNSUPPORTED_BLOCK_IDS]
    if bad_blocks:
        names = ", ".join(f"{ch} ({UNSUPPORTED_BLOCK_IDS[ch]})" for ch in sorted(set(bad_blocks)))
        raise ValueError(
            f"Map sequence '{map_sequence}' contains unsupported block(s): {names}. "
            "MetaDrive's Fork blocks are marked buggy in this codebase. "
            "Please replace them with ramp blocks 'r' / 'R' or remove them."
        )

    unknown_blocks = [ch for ch in map_sequence if ch not in SUPPORTED_BLOCK_IDS]
    if unknown_blocks:
        unknown = ", ".join(sorted(set(unknown_blocks)))
        supported = "".join(sorted(SUPPORTED_BLOCK_IDS))
        raise ValueError(
            f"Map sequence '{map_sequence}' contains unknown block id(s): {unknown}. "
            f"Supported ids are: {supported}"
        )


def load_cfg(env_name: str, num_scenarios: int):
    config_dir = os.path.join(ROOT_DIR, "configs")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(
            config_name="base",
            overrides=[
                f"env={env_name}",
                f"env_kwargs.num_scenarios={num_scenarios}",
            ],
        )
    return hydra.utils.instantiate(cfg)


def safe_slug(text: str) -> str:
    text = text.replace("$", "dollar")
    text = re.sub(r"[^A-Za-z0-9_-]+", "_", text)
    return text.strip("_") or "map"


def save_image(path: Path, image: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(path, image)


def render_scene_image(env, image_size: int) -> np.ndarray:
    frame = env.render(
        mode="top_down",
        window=False,
        screen_size=(image_size, image_size),
        film_size=(image_size, image_size),
    )
    arr = np.asarray(frame)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def road_to_dict(road):
    return {
        "start": road.start_node,
        "end": road.end_node,
    }


def main():
    args = parse_args()
    validate_map_sequence(args.map_sequence)
    cfg = load_cfg(args.env, args.num_scenarios)

    env_kwargs = resolve_env_kwargs(getattr(cfg, "env_kwargs", None))
    env_kwargs = dict(env_kwargs)
    env_kwargs["use_render"] = False
    env_kwargs["map"] = args.map_sequence
    env_kwargs["start_seed"] = args.seed
    env_kwargs["num_scenarios"] = args.num_scenarios
    env_kwargs["num_agents"] = args.num_agents
    env_kwargs["traffic_mode"] = args.traffic_mode
    env_kwargs["traffic_density"] = float(args.traffic_density)
    env_kwargs["traffic_policy"] = args.traffic_policy

    env = create_multiagent_env(str(cfg.env_name), env_kwargs, default_start_seed=int(cfg.seed))
    slug = safe_slug(args.map_sequence)
    output_dir = Path(ROOT_DIR) / args.output_dir / f"{slug}_seed{args.seed}"

    try:
        obs, _ = env.reset(seed=args.seed)

        if args.scene_steps > 0:
            for _ in range(args.scene_steps):
                obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
                if terminated.get("__all__", False) or truncated.get("__all__", False):
                    break

        map_image = draw_top_down_map(env.current_map, resolution=(args.image_size, args.image_size))
        map_image = np.asarray(map_image)
        if map_image.dtype != np.uint8:
            map_image = np.clip(map_image, 0, 255).astype(np.uint8)

        scene_image = render_scene_image(env, args.image_size)

        map_path = output_dir / "map.png"
        scene_path = output_dir / "scene.png"
        summary_path = output_dir / "summary.json"

        save_image(map_path, map_image)
        save_image(scene_path, scene_image)

        traffic_manager = getattr(env.engine, "traffic_manager", None)
        traffic_count = len(getattr(traffic_manager, "traffic_vehicles", []) or [])
        spawn_roads = list(getattr(env.current_map, "spawn_roads", []) or [])
        summary = {
            "map_sequence": args.map_sequence,
            "env_name": str(cfg.env_name),
            "seed": args.seed,
            "num_scenarios": args.num_scenarios,
            "num_agents": env.config.get("num_agents"),
            "traffic_mode": env.config.get("traffic_mode"),
            "traffic_density": env.config.get("traffic_density"),
            "traffic_policy": env.config.get("traffic_policy"),
            "scene_steps": args.scene_steps,
            "active_agents": len(getattr(env, "agents", {})),
            "obs_agents": len(obs) if isinstance(obs, dict) else None,
            "traffic_vehicles": traffic_count,
            "spawn_roads_count": len(spawn_roads),
            "spawn_roads": [road_to_dict(road) for road in spawn_roads],
            "destinations": sorted(
                {
                    vehicle.config.get("destination")
                    for vehicle in getattr(env, "agents", {}).values()
                    if vehicle.config.get("destination") is not None
                }
            ),
            "files": {
                "map": str(map_path),
                "scene": str(scene_path),
            },
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print(map_path)
        print(scene_path)
        print(summary_path)
    finally:
        env.close()
        try:
            from metadrive.engine.engine_utils import close_engine

            close_engine()
        except Exception:
            pass


if __name__ == "__main__":
    main()
