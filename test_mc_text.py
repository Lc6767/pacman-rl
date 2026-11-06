"""
test_mc_text.py

用途：
載入 Monte Carlo 訓練好的 policy，
用文字版地圖觀察 PacMan 的行為。

執行前請先跑：
python train_mc.py

執行：
python test_mc_text.py
"""

from pathlib import Path
import time

from pacman_rl_env import PacmanEnv
from mc_agent import FirstVisitMCAgent


BASE_DIR = Path(__file__).resolve().parent
POLICY_PATH = BASE_DIR / "mc_pacman_policy.pkl"
MAP_PATH = BASE_DIR / "material" / "map.maze"


def main() -> None:
    if not POLICY_PATH.exists():
        raise FileNotFoundError(
            f"找不到 {POLICY_PATH.name}，請先執行 python train_mc.py"
        )

    env = PacmanEnv(
        map_path=str(MAP_PATH),
        enemy_count=4,
        max_steps=1000,
        seed=99,
    )

    agent = FirstVisitMCAgent.load(str(POLICY_PATH))

    state = env.reset()
    done = False
    total_reward = 0.0
    info = {}

    step_count = 0

    while not done:
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

    print("Monte Carlo text test finished")
    print("Total reward:", total_reward)
    print("Info:", info)


if __name__ == "__main__":
    main()