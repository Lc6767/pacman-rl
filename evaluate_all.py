"""
evaluate_all.py

用途：
用同一套條件評估三種演算法（Monte Carlo、SARSA、Q-learning），結果可重現。

和 evaluate_mc.py / evaluate_sarsa.py 的差別：
原本的評估程式中，agent 遇到 Q 值相同的動作時會隨機選一個，
但這個隨機沒有固定種子，所以每次執行的數字都會有一點差異。
這裡每一局都把「環境」和「agent 選動作」的隨機種子固定成同一個值，
三種演算法面對完全相同的 100 局，結果每次執行都一樣。

結局分類（依 pacman_rl_env.py 實際結束一局的三種條件）：
- 勝利：吃完所有豆子
- 死亡：被敵人抓到
- 逾時：走滿 1000 步

注意：環境中「連續 150 步沒吃到豆子」只會扣獎勵，不會結束這一局。
原本的 evaluate_mc.py / evaluate_sarsa.py 把「死前已 150 步以上沒吃豆」的局另外算成
No-food timeout，而走滿 1000 步的局沒有被計入任何類別；這裡改成依實際結局分類。

執行：
python evaluate_all.py
輸出：
results/evaluation_summary.csv
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

from pacman_rl_env import PacmanEnv
from mc_agent import FirstVisitMCAgent
from sarsa_agent import SarsaAgent
from qlearning_agent import QLearningAgent


BASE_DIR = Path(__file__).resolve().parent
MAP_PATH = BASE_DIR / "material" / "map.maze"
RESULTS_DIR = BASE_DIR / "results"

AGENTS = [
    ("Monte Carlo", FirstVisitMCAgent, "mc_pacman_policy.pkl"),
    ("SARSA", SarsaAgent, "sarsa_pacman_policy.pkl"),
    ("Q-learning", QLearningAgent, "qlearning_pacman_policy.pkl"),
]

MAX_STEPS = 1000


def classify(info: dict) -> str:
    if info["remaining_food"] == 0:
        return "win"
    if info["steps"] >= MAX_STEPS:
        return "timeout"
    return "death"


def evaluate(agent_cls, policy_path: Path, test_episodes: int = 100) -> dict:
    agent = agent_cls.load(str(policy_path))

    scores, rewards, steps_list = [], [], []
    outcomes = {"win": 0, "death": 0, "timeout": 0}

    for i in range(test_episodes):
        env = PacmanEnv(
            map_path=str(MAP_PATH),
            enemy_count=4,
            max_steps=MAX_STEPS,
            seed=i,
        )
        agent.rng = random.Random(i)  # 固定 agent 平手時的隨機選擇

        state = env.reset()
        done = False
        total_reward = 0.0
        info = {}

        while not done:
            action = agent.choose_action(state, training=False)
            state, reward, done, info = env.step(action)
            total_reward += reward

        scores.append(info["score"])
        rewards.append(total_reward)
        steps_list.append(info["steps"])
        outcomes[classify(info)] += 1

    n = test_episodes
    return {
        "avg_score": sum(scores) / n,
        "best_score": max(scores),
        "avg_steps": sum(steps_list) / n,
        "avg_reward": sum(rewards) / n,
        "win_rate": outcomes["win"] / n,
        "death_rate": outcomes["death"] / n,
        "timeout_rate": outcomes["timeout"] / n,
    }


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    rows = []

    for name, agent_cls, policy_name in AGENTS:
        policy_path = BASE_DIR / policy_name
        if not policy_path.exists():
            print(f"[skip] {name}: 找不到 {policy_name}，請先執行對應的 train 程式")
            continue

        result = evaluate(agent_cls, policy_path)
        rows.append({"algorithm": name, **result})

        print(
            f"{name:12s} | avg_score={result['avg_score']:7.1f} | "
            f"best={result['best_score']:5d} | avg_steps={result['avg_steps']:6.1f} | "
            f"win={result['win_rate']:.0%} | death={result['death_rate']:.0%} | "
            f"timeout={result['timeout_rate']:.0%} | avg_reward={result['avg_reward']:9.1f}"
        )

    if rows:
        out = RESULTS_DIR / "evaluation_summary.csv"
        with open(out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved: {out.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
