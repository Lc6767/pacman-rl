"""
pacman_rl_env.py

PacMan reinforcement learning environment.

Action 定義：
0 = up
1 = right
2 = down
3 = left

1. State 加入最近食物距離 nearest_food_distance。
2. State 保留最近敵人距離 nearest_enemy_distance。
3. Reward 加強死亡懲罰。
4. Reward 加入敵人距離懲罰。
5. Reward 加入太久沒吃豆子的懲罰。
6. 太久沒吃豆子會直接結束該局，避免學到無意義繞圈。
7. 重複路線懲罰加強。
8. 吃完全部豆子的獎勵加大。
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Tuple


Position = Tuple[int, int]

# state 內容：
# (
#   pacman_x, pacman_y,
#   up_wall, right_wall, down_wall, left_wall,
#   nearest_food_dx, nearest_food_dy,
#   nearest_food_distance,
#   nearest_enemy_dx, nearest_enemy_dy,
#   nearest_enemy_distance
# )
State = Tuple[int, int, int, int, int, int, int, int, int, int, int, int]


class PacmanEnv:
    ACTIONS: Dict[int, Position] = {
        0: (0, -1),   # up
        1: (1, 0),    # right
        2: (0, 1),    # down
        3: (-1, 0),   # left
    }

    def __init__(
        self,
        map_path: str = "material/map.maze",
        enemy_count: int = 4,
        max_steps: int = 1000,
        seed: int | None = None,
    ) -> None:
        self.map_path = Path(map_path)
        self.enemy_count = enemy_count
        self.max_steps = max_steps
        self.rng = random.Random(seed)

        self.original_map: List[List[str]] = self._load_map(self.map_path)
        self.height = len(self.original_map)
        self.width = len(self.original_map[0])

        self.game_map: List[List[str]] = []
        self.pacman: Position = (1, 1)
        self.enemies: List[Position] = []

        self.steps = 0
        self.score = 0

        # 紀錄最近走過的位置，避免 PacMan 一直來回走
        self.recent_positions: List[Position] = []

        # 紀錄多久沒有吃到豆子
        self.steps_since_last_food = 0

    def _load_map(self, path: Path) -> List[List[str]]:
        """
        讀取 map.maze。

        map.maze 裡面：
        1 = 牆壁
        4 = 食物
        0 = 空地
        """
        if not path.exists():
            raise FileNotFoundError(
                f"Cannot find map file: {path}. "
                "Please run this file from the PacMan project folder."
            )

        rows = [
            list(line.strip())
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        if not rows:
            raise ValueError("Map file is empty.")

        width = len(rows[0])

        if any(len(row) != width for row in rows):
            raise ValueError("Map rows have different widths.")

        return rows

    def reset(self) -> State:
        """
        每一局遊戲開始時呼叫。
        對 RL 來說，一局遊戲就是一個 episode。
        """
        self.game_map = [row[:] for row in self.original_map]
        self.pacman = (1, 1)
        self.steps = 0
        self.score = 0

        self.recent_positions = [self.pacman]
        self.steps_since_last_food = 0

        # 起點如果有食物，先清掉
        px, py = self.pacman

        if self.game_map[py][px] == "4":
            self.game_map[py][px] = "0"

        # 隨機產生敵人位置
        walkable = [
            (x, y)
            for y in range(self.height)
            for x in range(self.width)
            if self._can_walk((x, y)) and (x, y) != self.pacman
        ]

        self.enemies = []

        for _ in range(self.enemy_count):
            self.enemies.append(self.rng.choice(walkable))

        return self.get_state()

    def _can_walk(self, pos: Position) -> bool:
        """
        判斷某格能不能走。
        """
        x, y = pos

        return (
            0 <= x < self.width
            and 0 <= y < self.height
            and self.game_map[y][x] != "1"
        )

    def _move(self, pos: Position, action: int) -> Position:
        """
        根據 action 計算下一格位置。
        """
        dx, dy = self.ACTIONS[action]
        return (pos[0] + dx, pos[1] + dy)

    def _remaining_food_count(self) -> int:
        """
        計算地圖上還剩多少食物。
        """
        return sum(row.count("4") for row in self.game_map)

    def _nearest_food_delta(self) -> Tuple[int, int]:
        """
        找最近食物方向。

        回傳不是實際距離，而是方向：
        -1 = 在左邊或上面
         0 = 同一直線
         1 = 在右邊或下面
        """
        px, py = self.pacman

        foods = [
            (x, y)
            for y in range(self.height)
            for x in range(self.width)
            if self.game_map[y][x] == "4"
        ]

        if not foods:
            return (0, 0)

        fx, fy = min(
            foods,
            key=lambda p: abs(p[0] - px) + abs(p[1] - py)
        )

        return (self._sign(fx - px), self._sign(fy - py))

    def _nearest_food_distance(self) -> int:
        """
        計算 PacMan 到最近食物的曼哈頓距離。
        """
        px, py = self.pacman

        foods = [
            (x, y)
            for y in range(self.height)
            for x in range(self.width)
            if self.game_map[y][x] == "4"
        ]

        if not foods:
            return 0

        return min(
            abs(x - px) + abs(y - py)
            for x, y in foods
        )

    def _nearest_enemy_delta(self) -> Tuple[int, int]:
        """
        找最近敵人方向。
        """
        px, py = self.pacman

        if not self.enemies:
            return (0, 0)

        ex, ey = min(
            self.enemies,
            key=lambda p: abs(p[0] - px) + abs(p[1] - py)
        )

        return (self._sign(ex - px), self._sign(ey - py))

    def _nearest_enemy_distance(self) -> int:
        """
        計算 PacMan 到最近敵人的曼哈頓距離。
        """
        px, py = self.pacman

        if not self.enemies:
            return 99

        return min(
            abs(ex - px) + abs(ey - py)
            for ex, ey in self.enemies
        )

    @staticmethod
    def _sign(value: int) -> int:
        """
        把數值轉成方向。
        """
        if value > 0:
            return 1

        if value < 0:
            return -1

        return 0

    def get_state(self) -> State:
        """
        狀態設計：
        (
            pacman_x, pacman_y,
            up_wall, right_wall, down_wall, left_wall,
            nearest_food_dx_sign, nearest_food_dy_sign,
            nearest_food_distance,
            nearest_enemy_dx_sign, nearest_enemy_dy_sign,
            nearest_enemy_distance
        )

        nearest_food_distance 使用 min(distance, 8)，避免 state 空間太大。
        nearest_enemy_distance 使用 min(distance, 5)，避免 state 空間太大。
        """
        px, py = self.pacman

        wall_info = []

        for action in range(4):
            nx, ny = self._move(self.pacman, action)
            wall_info.append(0 if self._can_walk((nx, ny)) else 1)

        food_dx, food_dy = self._nearest_food_delta()
        food_distance = min(self._nearest_food_distance(), 8)

        enemy_dx, enemy_dy = self._nearest_enemy_delta()
        enemy_distance = min(self._nearest_enemy_distance(), 5)

        return (
            px, py,
            wall_info[0], wall_info[1], wall_info[2], wall_info[3],
            food_dx, food_dy,
            food_distance,
            enemy_dx, enemy_dy,
            enemy_distance,
        )

    def step(self, action: int) -> Tuple[State, float, bool, dict]:
        """
        RL 最核心的函式。

        輸入：
            action = 0, 1, 2, 3

        輸出：
            next_state, reward, done, info
        """
        self.steps += 1
        self.steps_since_last_food += 1

        # 每走一步扣分，避免 PacMan 拖時間。
        reward = -1.5
        done = False

        # 記錄移動前位置，用來處理 PacMan 和敵人互相穿過的情況。
        old_pacman = self.pacman
        old_enemies = self.enemies[:]

        # 移動前先記錄最近食物距離
        old_food_distance = self._nearest_food_distance()

        # 1. PacMan 移動
        next_pos = self._move(self.pacman, action)

        if not self._can_walk(next_pos):
            # 撞牆：位置不變，給懲罰
            reward -= 5.0
            next_pos = self.pacman
        else:
            self.pacman = next_pos

        # 2. PacMan 移動後，如果走到敵人原本位置，算死亡。
        if self.pacman in old_enemies:
            reward -= 300.0
            done = True

        # 3. 重複路徑懲罰加強
        # 最近 6 步走過的格子，扣比較重。
        if self.pacman in self.recent_positions[-6:]:
            reward -= 6.0

        # 最近 12 步走過的格子，再額外扣一點。
        if self.pacman in self.recent_positions[-12:]:
            reward -= 3.0

        self.recent_positions.append(self.pacman)

        if len(self.recent_positions) > 30:
            self.recent_positions.pop(0)

        # 4. 靠近食物給獎勵，遠離食物給懲罰
        new_food_distance = self._nearest_food_distance()

        if new_food_distance < old_food_distance:
            reward += 1.0
        elif new_food_distance > old_food_distance:
            reward -= 1.0

        # 5. 敵人距離懲罰
        # 重點：讓 PacMan 知道靠近敵人很危險。
        enemy_distance = self._nearest_enemy_distance()

        if enemy_distance == 1:
            reward -= 20.0
        elif enemy_distance == 2:
            reward -= 8.0
        elif enemy_distance == 3:
            reward -= 3.0

        # 6. 吃食物
        px, py = self.pacman

        if self.game_map[py][px] == "4":
            self.game_map[py][px] = "0"

            # score 是遊戲分數，維持 +10；
            # reward 是訓練訊號，提高成 +25，鼓勵吃豆。
            reward += 30.0
            self.score += 10

            # 吃到豆子，代表有進展，歸零
            self.steps_since_last_food = 0

        # 7. 太久沒吃豆子，代表可能在同一區域繞圈，給額外懲罰
        if self.steps_since_last_food > 20:
            reward -= 2.0

        if self.steps_since_last_food > 50:
            reward -= 5.0

        if self.steps_since_last_food > 100:
            reward -= 10.0

        if self.steps_since_last_food >= 150:
            reward -= 15.0

        # 8. 如果 PacMan 還沒死，敵人才移動
        if not done:
            new_enemies: List[Position] = []

            for enemy_pos in old_enemies:
                actions = list(self.ACTIONS.keys())
                self.rng.shuffle(actions)

                moved = enemy_pos

                for enemy_action in actions:
                    candidate = self._move(enemy_pos, enemy_action)

                    if self._can_walk(candidate):
                        moved = candidate
                        break

                new_enemies.append(moved)

            self.enemies = new_enemies

            # 9. 敵人移動後，如果走到 PacMan 位置，算死亡。
            if self.pacman in self.enemies:
                reward -= 300.0
                done = True

            # 10. PacMan 和敵人互換位置，也算死亡。
            for old_enemy, new_enemy in zip(old_enemies, self.enemies):
                pacman_moved_to_enemy_old_pos = self.pacman == old_enemy
                enemy_moved_to_pacman_old_pos = new_enemy == old_pacman

                if pacman_moved_to_enemy_old_pos and enemy_moved_to_pacman_old_pos:
                    reward -= 300.0
                    done = True
                    break

        # 11. 吃完全部豆子，大獎勵
        if self._remaining_food_count() == 0:
            reward += 1000.0
            done = True

        # 13. 超過最大步數，結束
        if self.steps >= self.max_steps:
            done = True

        info = {
            "score": self.score,
            "steps": self.steps,
            "remaining_food": self._remaining_food_count(),
            "steps_since_last_food": self.steps_since_last_food,
        }

        return self.get_state(), reward, done, info

    def render_text(self) -> None:
        """
        文字版顯示：
        P = PacMan
        E = Enemy
        # = Wall
        . = Food
        空白 = Empty
        """
        enemy_set = set(self.enemies)

        for y in range(self.height):
            row = ""

            for x in range(self.width):
                pos = (x, y)

                if pos == self.pacman:
                    row += "P"
                elif pos in enemy_set:
                    row += "E"
                elif self.game_map[y][x] == "1":
                    row += "#"
                elif self.game_map[y][x] == "4":
                    row += "."
                else:
                    row += " "

            print(row)

        print()