"""
play_qlearning_pygame.py

用途：
1. 載入 Q-learning 訓練好的 policy：qlearning_pacman_policy.pkl
2. 用 Pygame 顯示 PacMan 自己玩
3. 可輸出 mp4 影片

執行前請先跑：
python train_qlearning.py

需要安裝：
pip install pygame imageio imageio-ffmpeg numpy
"""

from pathlib import Path
from datetime import datetime
import pygame
import imageio.v2 as imageio
import numpy as np

from pacman_rl_env import PacmanEnv
from qlearning_agent import QLearningAgent


BASE_DIR = Path(__file__).resolve().parent
MATERIAL_DIR = BASE_DIR / "material"

BLOCK_SIZE = 24
FPS = 10

MAP_PATH = MATERIAL_DIR / "map.maze"
POLICY_PATH = BASE_DIR / "qlearning_pacman_policy.pkl"

BLOCK_IMAGE = MATERIAL_DIR / "block.png"
FOOD_IMAGE = MATERIAL_DIR / "food.png"
USER_IMAGE = MATERIAL_DIR / "user.png"

ENEMY_IMAGES = [
    MATERIAL_DIR / "enemy1.png",
    MATERIAL_DIR / "enemy2.png",
    MATERIAL_DIR / "enemy3.png",
    MATERIAL_DIR / "enemy4.png",
]

# 是否輸出影片
RECORD_VIDEO = True


VIDEO_NAME = BASE_DIR / "qlearning_pacman_demo.mp4"


def load_first_frame(path: Path) -> pygame.Surface:
    """
    原本 PacMan / Enemy 圖片是 spritesheet，
    這裡只切第一格 24x24 來顯示。
    """
    image = pygame.image.load(str(path)).convert_alpha()
    return image.subsurface(pygame.Rect(0, 0, 24, 24)).copy()


def draw_game(screen: pygame.Surface, env: PacmanEnv, images: dict) -> None:
    """
    將目前環境狀態畫到 Pygame 視窗。
    """
    screen.fill((0, 0, 0))

    block_image = images["block"]
    food_image = images["food"]
    pacman_image = images["pacman"]
    enemy_images = images["enemies"]

    # 畫牆壁與食物
    for y in range(env.height):
        for x in range(env.width):
            px = x * BLOCK_SIZE
            py = y * BLOCK_SIZE

            tile = env.game_map[y][x]

            if tile == "1":
                screen.blit(block_image, (px, py))
            elif tile == "4":
                screen.blit(food_image, (px, py))

    # 畫 PacMan
    pac_x, pac_y = env.pacman
    screen.blit(pacman_image, (pac_x * BLOCK_SIZE, pac_y * BLOCK_SIZE))

    # 畫敵人
    for i, enemy_pos in enumerate(env.enemies):
        ex, ey = enemy_pos
        enemy_image = enemy_images[i % len(enemy_images)]
        screen.blit(enemy_image, (ex * BLOCK_SIZE, ey * BLOCK_SIZE))

    pygame.display.flip()


def surface_to_frame(surface: pygame.Surface) -> np.ndarray:
    """
    將 Pygame 畫面轉成 imageio 可以寫入 mp4 的格式。
    """
    frame = pygame.surfarray.array3d(surface)
    frame = np.transpose(frame, (1, 0, 2))
    return frame


def main() -> None:
    if not POLICY_PATH.exists():
        raise FileNotFoundError(
            f"找不到 {POLICY_PATH.name}，請先執行 python train_qlearning.py"
        )

    pygame.init()

    env = PacmanEnv(
        map_path=str(MAP_PATH),
        enemy_count=4,
        max_steps=1000,
        seed=None,
    )

    agent = QLearningAgent.load(str(POLICY_PATH))

    window_width = env.width * BLOCK_SIZE
    window_height = env.height * BLOCK_SIZE

    screen = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption("Q-learning PacMan Demo")

    clock = pygame.time.Clock()

    images = {
        "block": pygame.image.load(str(BLOCK_IMAGE)).convert_alpha(),
        "food": pygame.image.load(str(FOOD_IMAGE)).convert_alpha(),
        "pacman": load_first_frame(USER_IMAGE),
        "enemies": [load_first_frame(path) for path in ENEMY_IMAGES],
    }

    state = env.reset()
    done = False
    total_reward = 0.0
    info = {}

    writer = None

    if RECORD_VIDEO:
        writer = imageio.get_writer(str(VIDEO_NAME), fps=FPS)

    try:
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True

            draw_game(screen, env, images)

            if RECORD_VIDEO and writer is not None:
                writer.append_data(surface_to_frame(screen))

            action = agent.choose_action(state, training=False)
            state, reward, done, info = env.step(action)
            total_reward += reward

            clock.tick(FPS)

    finally:
        if writer is not None:
            writer.close()

        pygame.quit()

    print("Q-learning pygame demo finished")
    print("Total reward:", total_reward)
    print("Info:", info)

    if RECORD_VIDEO:
        print(f"Video saved to: {VIDEO_NAME}")


if __name__ == "__main__":
    main()