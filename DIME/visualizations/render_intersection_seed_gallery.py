import argparse
import json
import os
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render a gallery for the initial multi-agent intersection scenes over several seeds."
    )
    parser.add_argument(
        "--env",
        default="metadrive_ma_intersection",
        help="Hydra env config name, e.g. metadrive_ma_intersection.",
    )
    parser.add_argument(
        "--num-scenarios",
        type=int,
        default=10,
        help="How many consecutive seeds to render starting from start_seed.",
    )
    parser.add_argument(
        "--output-dir",
        default="visualizations/intersection_seed_gallery",
        help="Directory to store rendered images and the HTML gallery.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=700,
        help="Per-scene render size in pixels.",
    )
    return parser.parse_args()


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


def render_gallery(cfg, output_dir: Path, image_size: int):
    env_kwargs = resolve_env_kwargs(getattr(cfg, "env_kwargs", None))
    env_kwargs = dict(env_kwargs)
    env_kwargs["use_render"] = False

    env = create_multiagent_env(str(cfg.env_name), env_kwargs, default_start_seed=int(cfg.seed))
    output_dir.mkdir(parents=True, exist_ok=True)

    start_seed = int(env.config.get("start_seed", cfg.seed))
    scenario_count = int(env.config.get("num_scenarios", 1))
    records = []

    rows = max(1, int(np.ceil(scenario_count / 5)))
    cols = min(scenario_count, 5)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.4, rows * 4.0))
    if isinstance(axes, np.ndarray):
        axes = axes.reshape(-1)
    else:
        axes = np.asarray([axes])

    try:
        for idx in range(scenario_count):
            seed = start_seed + idx
            reset_ret = env.reset(seed=seed)
            obs = reset_ret[0] if isinstance(reset_ret, tuple) else reset_ret
            frame = env.render(
                mode="top_down",
                window=False,
                screen_size=(image_size, image_size),
                film_size=(image_size, image_size),
            )
            arr = np.asarray(frame)
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)

            traffic_manager = getattr(env.engine, "traffic_manager", None)
            traffic_count = len(getattr(traffic_manager, "traffic_vehicles", []) or [])

            img_name = f"seed_{seed}.png"
            plt.imsave(output_dir / img_name, arr)

            record = {
                "seed": seed,
                "active_agents": len(getattr(env, "agents", {})),
                "traffic_vehicles": traffic_count,
                "obs_agents": len(obs) if isinstance(obs, dict) else None,
                "image": img_name,
            }
            records.append(record)

            ax = axes[idx]
            ax.imshow(arr)
            ax.set_title(
                f"seed {seed}\nagents={record['active_agents']} traffic={traffic_count}",
                fontsize=10,
            )
            ax.axis("off")

        for ax in axes[len(records):]:
            ax.axis("off")

        fig.suptitle("MultiAgentIntersectionEnv initial scenes for seeds 0-9", fontsize=16)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(output_dir / "contact_sheet.png", dpi=180)

        summary = {
            "env_config": {
                "env_name": str(cfg.env_name),
                "start_seed": start_seed,
                "num_scenarios": scenario_count,
                "traffic_mode": env.config.get("traffic_mode"),
                "traffic_density": env.config.get("traffic_density"),
                "num_agents": env.config.get("num_agents"),
                "allow_respawn": env.config.get("allow_respawn"),
            },
            "records": records,
        }
        with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        write_html_gallery(output_dir, summary)
    finally:
        plt.close(fig)
        env.close()
        try:
            from metadrive.engine.engine_utils import close_engine

            close_engine()
        except Exception:
            pass


def write_html_gallery(output_dir: Path, summary: dict):
    env_cfg = summary["env_config"]
    cards = []
    for record in summary["records"]:
        cards.append(
            f"""
            <article class="card">
              <div class="card-header">
                <div class="seed">seed {record['seed']}</div>
                <div class="stats">agents {record['active_agents']} | traffic {record['traffic_vehicles']}</div>
              </div>
              <img src="{record['image']}" alt="seed {record['seed']} top-down scene" />
            </article>
            """
        )

    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Intersection Seeds 0-9</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@500;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap" rel="stylesheet">
  <style>
    :root {{
      --bg: #0e1216;
      --panel: rgba(18, 24, 31, 0.78);
      --panel-strong: rgba(26, 34, 43, 0.9);
      --text: #edf2f7;
      --muted: #9eb1c3;
      --accent: #f4b860;
      --line: rgba(255, 255, 255, 0.08);
      --shadow: 0 18px 50px rgba(0, 0, 0, 0.32);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--text);
      font-family: "IBM Plex Sans", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(244, 184, 96, 0.18), transparent 34%),
        radial-gradient(circle at right 20%, rgba(83, 135, 214, 0.18), transparent 28%),
        linear-gradient(160deg, #07090c 0%, #0e1216 50%, #111820 100%);
      min-height: 100vh;
    }}
    .wrap {{
      width: min(1380px, calc(100vw - 48px));
      margin: 0 auto;
      padding: 40px 0 56px;
    }}
    .hero {{
      display: grid;
      grid-template-columns: 1.15fr 0.85fr;
      gap: 24px;
      align-items: start;
      margin-bottom: 28px;
    }}
    .title-panel, .meta-panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 28px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(18px);
    }}
    .title-panel {{
      padding: 28px 30px 26px;
    }}
    .eyebrow {{
      color: var(--accent);
      text-transform: uppercase;
      letter-spacing: 0.16em;
      font-size: 12px;
      margin-bottom: 10px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-family: "Cormorant Garamond", serif;
      font-size: 52px;
      line-height: 0.95;
      font-weight: 700;
    }}
    .subtitle {{
      margin: 0;
      max-width: 58ch;
      color: var(--muted);
      line-height: 1.6;
      font-size: 15px;
    }}
    .meta-panel {{
      padding: 22px 24px;
      display: grid;
      gap: 14px;
    }}
    .meta-row {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      padding-bottom: 10px;
      border-bottom: 1px solid var(--line);
      font-size: 14px;
    }}
    .meta-row:last-child {{
      padding-bottom: 0;
      border-bottom: 0;
    }}
    .meta-label {{
      color: var(--muted);
    }}
    .summary {{
      margin: 0 0 20px;
      color: var(--muted);
      font-size: 14px;
      line-height: 1.6;
    }}
    .sheet {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 28px;
      padding: 22px;
      box-shadow: var(--shadow);
      margin-bottom: 24px;
    }}
    .sheet img {{
      width: 100%;
      display: block;
      border-radius: 18px;
      border: 1px solid var(--line);
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 18px;
    }}
    .card {{
      background: var(--panel-strong);
      border: 1px solid var(--line);
      border-radius: 22px;
      padding: 14px;
      box-shadow: var(--shadow);
      transform: translateY(18px);
      opacity: 0;
      animation: rise 0.6s ease forwards;
    }}
    .card:nth-child(2n) {{ animation-delay: 0.06s; }}
    .card:nth-child(3n) {{ animation-delay: 0.12s; }}
    .card:nth-child(4n) {{ animation-delay: 0.18s; }}
    .card-header {{
      display: flex;
      justify-content: space-between;
      gap: 10px;
      align-items: baseline;
      margin-bottom: 12px;
    }}
    .seed {{
      font-size: 18px;
      font-weight: 600;
      color: var(--accent);
    }}
    .stats {{
      font-size: 12px;
      color: var(--muted);
    }}
    .card img {{
      width: 100%;
      display: block;
      border-radius: 14px;
      border: 1px solid var(--line);
    }}
    @keyframes rise {{
      from {{ transform: translateY(18px); opacity: 0; }}
      to {{ transform: translateY(0); opacity: 1; }}
    }}
    @media (max-width: 960px) {{
      .hero {{ grid-template-columns: 1fr; }}
      h1 {{ font-size: 42px; }}
      .wrap {{ width: min(100vw - 24px, 1380px); padding-top: 22px; }}
    }}
  </style>
</head>
<body>
  <main class="wrap">
    <section class="hero">
      <div class="title-panel">
        <div class="eyebrow">Seed Gallery</div>
        <h1>Intersection Seeds 0-9</h1>
        <p class="subtitle">
          这里展示的是同一个十字路口环境在 <strong>seed 0-9</strong> 下的初始场景。
          对这个环境来说，10 个 scenario 不是 10 种不同地图结构，而是同一拓扑下的 10 个随机实例。
        </p>
      </div>
      <div class="meta-panel">
        <div class="meta-row"><span class="meta-label">Env</span><span>{env_cfg['env_name']}</span></div>
        <div class="meta-row"><span class="meta-label">start_seed</span><span>{env_cfg['start_seed']}</span></div>
        <div class="meta-row"><span class="meta-label">num_scenarios</span><span>{env_cfg['num_scenarios']}</span></div>
        <div class="meta-row"><span class="meta-label">traffic_mode</span><span>{env_cfg['traffic_mode']}</span></div>
        <div class="meta-row"><span class="meta-label">traffic_density</span><span>{env_cfg['traffic_density']}</span></div>
        <div class="meta-row"><span class="meta-label">num_agents</span><span>{env_cfg['num_agents']}</span></div>
        <div class="meta-row"><span class="meta-label">allow_respawn</span><span>{env_cfg['allow_respawn']}</span></div>
      </div>
    </section>

    <p class="summary">
      快速判断：如果 10 张图的道路骨架看起来完全一样，那是正常的。你真正轮换的是 seed 导致的初始车辆布局、
      路径分配和随机状态，而不是换了 10 张不同形状的十字路口地图。
    </p>

    <section class="sheet">
      <img src="contact_sheet.png" alt="contact sheet of the 10 initial scenarios" />
    </section>

    <section class="grid">
      {''.join(cards)}
    </section>
  </main>
</body>
</html>
"""

    with open(output_dir / "index.html", "w", encoding="utf-8") as f:
        f.write(html)


def main():
    args = parse_args()
    cfg = load_cfg(args.env, args.num_scenarios)
    output_dir = Path(os.path.join(ROOT_DIR, args.output_dir))
    render_gallery(cfg, output_dir, args.image_size)
    print(output_dir / "index.html")


if __name__ == "__main__":
    main()
