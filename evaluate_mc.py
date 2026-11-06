"""
evaluate_mc.py

用途：
評估 Monte Carlo 訓練後的 PacMan policy。

執行前請先跑：
python train_mc.py

執行：
python evaluate_mc.py
"""

from pathlib import Path

from pacman_rl_env import PacmanEnv
from mc_agent import FirstVisitMCAgent


BASE_DIR = Path(__file__).resolve().parent
POLICY_PATH = BASE_DIR / "mc_pacman_policy.pkl"
MAP_PATH = BASE_DIR / "material" / "map.maze"


def evaluate(test_episodes: int = 100) -> None:
    if not POLICY_PATH.exists():
        raise FileNotFoundError(
            f"找不到 {POLICY_PATH.name}，請先執行 python train_mc.py"
        )

    agent = FirstVisitMCAgent.load(str(POLICY_PATH))

    scores = []
    rewards = []
    steps_list = []
    deaths = 0
    timeout_no_food = 0

    for i in range(test_episodes):
        env = PacmanEnv(
            map_path=str(MAP_PATH),
            enemy_count=4,
            max_steps=1000,
            seed=i,
        )

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

        if info["steps"] < 1000 and info["remaining_food"] > 0:
            if info.get("steps_since_last_food", 0) >= 150:
                timeout_no_food += 1
            else:
                deaths += 1

    print("Monte Carlo evaluate finished")
    print("Test episodes:", test_episodes)
    print("Average score:", sum(scores) / len(scores))
    print("Best score:", max(scores))
    print("Worst score:", min(scores))
    print("Average reward:", sum(rewards) / len(rewards))
    print("Average steps:", sum(steps_list) / len(steps_list))
    print("Death count:", deaths)
    print("Death rate:", deaths / test_episodes)
    print("No-food timeout count:", timeout_no_food)
    print("No-food timeout rate:", timeout_no_food / test_episodes)


if __name__ == "__main__":
    evaluate(test_episodes=100)