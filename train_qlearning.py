"""
train_qlearning.py

Q-learning Off-policy TD Control 訓練 PacMan

說明：
小組當時的 Q-learning 原始程式沒有保存下來，這是課後重新實作的版本。
環境與參數和 train_sarsa.py 相同：
1. enemy_count = 4
2. max_steps = 1000
3. episodes = 30000
4. alpha = 0.10, gamma = 0.95
5. epsilon 從 0.30 開始，每局乘上 0.9997，最低 0.02

和 SARSA 訓練迴圈的差別：
SARSA 要先選好下一個動作 A' 才能更新；
Q-learning 更新時直接取下一個狀態的最大 Q 值，下一步的動作在下一輪再選。
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt

from pacman_rl_env import PacmanEnv
from qlearning_agent import QLearningAgent


BASE_DIR = Path(__file__).resolve().parent


def train(episodes: int = 30000) -> None:
    env = PacmanEnv(
        map_path=str(BASE_DIR / "material" / "map.maze"),
        enemy_count=4,
        max_steps=1000,
        seed=7,
    )

    agent = QLearningAgent(
        actions=[0, 1, 2, 3],
        alpha=0.10,
        gamma=0.95,
        epsilon=0.30,
        epsilon_min=0.02,
        epsilon_decay=0.9997,
        seed=7,
    )

    logs = []

    for ep in range(1, episodes + 1):
        state = env.reset()

        total_reward = 0.0
        done = False
        last_info = {}

        while not done:
            action = agent.choose_action(state, training=True)
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            last_info = info

            agent.update(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
            )

            state = next_state

        agent.decay_epsilon()

        logs.append({
            "episode": ep,
            "total_reward": total_reward,
            "score": last_info.get("score", 0),
            "steps": last_info.get("steps", 0),
            "remaining_food": last_info.get("remaining_food", 0),
            "steps_since_last_food": last_info.get("steps_since_last_food", 0),
            "epsilon": agent.epsilon,
        })

        if ep % 100 == 0:
            recent = logs[-100:]
            avg_reward = sum(x["total_reward"] for x in recent) / len(recent)
            avg_score = sum(x["score"] for x in recent) / len(recent)
            avg_steps = sum(x["steps"] for x in recent) / len(recent)

            print(
                f"Episode {ep:5d} | "
                f"avg_reward={avg_reward:8.2f} | "
                f"avg_score={avg_score:7.2f} | "
                f"avg_steps={avg_steps:7.2f} | "
                f"epsilon={agent.epsilon:.3f}",
                flush=True,
            )

    agent.save(str(BASE_DIR / "qlearning_pacman_policy.pkl"))
    save_logs(logs, str(BASE_DIR / "qlearning_training_log.csv"))
    plot_rewards(logs, str(BASE_DIR / "qlearning_training_reward.png"))

    print("\nQ-learning training finished!")
    print("Saved: qlearning_pacman_policy.pkl, qlearning_training_log.csv, qlearning_training_reward.png")


def save_logs(logs, filename: str) -> None:
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=logs[0].keys())
        writer.writeheader()
        writer.writerows(logs)


def moving_average(values, window: int = 100):
    result = []

    for i in range(len(values)):
        start = max(0, i - window + 1)
        result.append(sum(values[start:i + 1]) / (i - start + 1))

    return result


def plot_rewards(logs, filename: str) -> None:
    rewards = [x["total_reward"] for x in logs]
    avg_rewards = moving_average(rewards, window=100)

    plt.figure()
    plt.plot(avg_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Average Reward, window=100")
    plt.title("Q-learning Off-policy TD Control for Pac-Man")
    plt.grid(True)
    plt.savefig(filename, dpi=160, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    train(episodes=30000)
