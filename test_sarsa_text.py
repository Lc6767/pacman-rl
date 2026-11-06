"""
test_sarsa_text.py

用途：
載入 SARSA 訓練好的 policy，
用文字版地圖觀察 PacMan 的行為。

執行前請先跑：
python train_sarsa.py

執行：
python test_sarsa_text.py
"""

from pathlib import Path
import time

from pacman_rl_env import PacmanEnv
from sarsa_agent import SarsaAgent


BASE_DIR = Path(__file__).resolve().parent
POLICY_PATH = BASE_DIR / "sarsa_pacman_policy.pkl"
MAP_PATH = BASE_DIR / "material" / "map.maze"


def main() -> None:
    if not POLICY_PATH.exists():
        raise FileNotFoundError(
            f"找不到 {POLICY_PATH.name}，請先執行 python train_sarsa.py"
        )

    env = PacmanEnv(
        map_path=str(MAP_PATH),
        enemy_count=4,
        max_steps=1000,
        seed=99,
    )

    agent = SarsaAgent.load(str(POLICY_PATH))

    state = env.reset()
    done = False
    total_reward = 0.0
    info = {}

    step_count = 0

    while not done:
        # 每 10 步顯示一次，避免輸出太長
        if step_count % 10 == 0:
            print(f"Step: {step_count}")
            print(f"Steps since last food: {info.get('steps_since_last_food', 0)}")
            env.render_text()
            time.sleep(0.03)

        action = agent.choose_action(state, training=False)
        state, reward, done, info = env.step(action)
        total_reward += reward
        step_count += 1

    print("Final state:")
    env.render_text()

    print("SARSA text test finished")
    print("Total reward:", total_reward)
    print("Info:", info)


if __name__ == "__main__":
    main()