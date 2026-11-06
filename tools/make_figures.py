"""
tools/make_figures.py

產生 README 用的圖：
1. results/figures/learning_curves.png：三種演算法訓練過程的平均獎勵（每 100 局移動平均）
2. results/figures/demo_<algorithm>.gif：用訓練好的 policy 玩一局的動畫

GIF 用 Pillow 直接把遊戲素材畫出來，不需要開 Pygame 視窗。

執行（在專案根目錄）：
python tools/make_figures.py
"""

from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pacman_rl_env import PacmanEnv  # noqa: E402
from mc_agent import FirstVisitMCAgent  # noqa: E402
from sarsa_agent import SarsaAgent  # noqa: E402
from qlearning_agent import QLearningAgent  # noqa: E402

FIG_DIR = ROOT / "results" / "figures"
MATERIAL = ROOT / "material"
BLOCK = 24

ALGOS = [
    # name, log file, agent class, policy file, slug, color
    ("Monte Carlo", "mc_training_log.csv", FirstVisitMCAgent, "mc_pacman_policy.pkl", "mc", "#E07B39"),
    ("SARSA", "sarsa_training_log.csv", SarsaAgent, "sarsa_pacman_policy.pkl", "sarsa", "#2E8B57"),
    ("Q-learning", "qlearning_training_log.csv", QLearningAgent, "qlearning_pacman_policy.pkl", "qlearning", "#3B6FB6"),
]


def moving_average(values, window=100):
    out, total = [], 0.0
    for i, v in enumerate(values):
        total += v
        if i >= window:
            total -= values[i - window]
        out.append(total / min(i + 1, window))
    return out


def learning_curves() -> None:
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for name, log, *_rest, color in ALGOS:
        path = ROOT / log
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            rewards = [float(r["total_reward"]) for r in csv.DictReader(f)]
        ax.plot(range(1, len(rewards) + 1), moving_average(rewards), label=name, color=color, linewidth=1.4)
    ax.axhline(0, color="#888888", linewidth=0.8)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Average reward (moving average, 100 episodes)")
    ax.set_title("Training reward on the same Pac-Man environment")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "learning_curves.png", dpi=160)
    plt.close(fig)


def first_frame(path: Path) -> Image.Image:
    return Image.open(path).convert("RGBA").crop((0, 0, BLOCK, BLOCK))


def draw(env: PacmanEnv, sprites: dict) -> Image.Image:
    img = Image.new("RGBA", (env.width * BLOCK, env.height * BLOCK), (0, 0, 0, 255))
    for y in range(env.height):
        for x in range(env.width):
            tile = env.game_map[y][x]
            if tile == "1":
                img.alpha_composite(sprites["block"], (x * BLOCK, y * BLOCK))
            elif tile == "4":
                img.alpha_composite(sprites["food"], (x * BLOCK, y * BLOCK))
    px, py = env.pacman
    img.alpha_composite(sprites["pacman"], (px * BLOCK, py * BLOCK))
    for i, (ex, ey) in enumerate(env.enemies):
        img.alpha_composite(sprites["enemies"][i % 4], (ex * BLOCK, ey * BLOCK))
    return img.convert("P", palette=Image.ADAPTIVE, colors=64)


def demo_gif(agent_cls, policy: str, slug: str, seed: int, max_frames: int = 450) -> dict:
    sprites = {
        "block": Image.open(MATERIAL / "block.png").convert("RGBA"),
        "food": Image.open(MATERIAL / "food.png").convert("RGBA"),
        "pacman": first_frame(MATERIAL / "user.png"),
        "enemies": [first_frame(MATERIAL / f"enemy{i}.png") for i in range(1, 5)],
    }
    agent = agent_cls.load(str(ROOT / policy))
    agent.rng = random.Random(seed)
    env = PacmanEnv(map_path=str(MATERIAL / "map.maze"), enemy_count=4, max_steps=1000, seed=seed)
    state = env.reset()
    frames, done, info = [draw(env, sprites)], False, {}
    while not done:
        state, _r, done, info = env.step(agent.choose_action(state, training=False))
        if len(frames) < max_frames:
            frames.append(draw(env, sprites))
    frames += [frames[-1]] * 15  # 結尾停一下
    frames[0].save(FIG_DIR / f"demo_{slug}.gif", save_all=True, append_images=frames[1:], duration=70, loop=0, optimize=True)
    return info


def main(seed: int = 0) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    learning_curves()
    print("Saved: results/figures/learning_curves.png")
    for name, _log, cls, policy, slug, _c in ALGOS:
        if (ROOT / policy).exists():
            info = demo_gif(cls, policy, slug, seed)
            print(f"Saved: results/figures/demo_{slug}.gif | seed={seed} | score={info['score']} "
                  f"steps={info['steps']} remaining_food={info['remaining_food']}")


if __name__ == "__main__":
    main(seed=int(sys.argv[1]) if len(sys.argv) > 1 else 0)
