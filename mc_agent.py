"""
mc_agent.py

First-visit Monte Carlo Control：
每一局遊戲結束後，從最後一步往前算 return G，
並更新 Q(s, a)，再用 epsilon-greedy 讓 PacMan 一邊探索、一邊改善策略。
"""

from __future__ import annotations

import pickle
import random
from collections import defaultdict
from typing import DefaultDict, Hashable, List, Tuple


State = Hashable
Action = int
EpisodeStep = Tuple[State, Action, float]


class FirstVisitMCAgent:
    def __init__(
        self,
        actions: List[Action],
        gamma: float = 0.95,
        epsilon: float = 0.20,
        epsilon_min: float = 0.02,
        epsilon_decay: float = 0.9995,
        seed: int | None = None,
    ) -> None:
        self.actions = actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng = random.Random(seed)

        # Q[state][action] = 目前估計的 Q(s,a)
        self.Q: DefaultDict[State, List[float]] = defaultdict(
            lambda: [0.0 for _ in self.actions]
        )

        # 用加總與次數來算平均 return
        self.returns_sum: DefaultDict[Tuple[State, Action], float] = defaultdict(float)
        self.returns_count: DefaultDict[Tuple[State, Action], int] = defaultdict(int)

    def choose_action(self, state: State, training: bool = True) -> Action:
        """
        epsilon-greedy policy

        訓練時：
        - epsilon 機率隨機探索
        - 1 - epsilon 機率選 Q 值最高的動作

        測試時：
        - 直接選 Q 值最高的動作
        """
        if training and self.rng.random() < self.epsilon:
            return self.rng.choice(self.actions)

        q_values = self.Q[state]
        max_q = max(q_values)

        best_actions = [
            action
            for action, q in enumerate(q_values)
            if q == max_q
        ]

        return self.rng.choice(best_actions)

    def update_from_episode(self, episode: List[EpisodeStep]) -> None:
        """
        First-visit Monte Carlo 更新。

        episode 裡面每一筆是：
        (state, action, reward)

        從後往前算：
        G = reward + gamma * G
        """
        G = 0.0
        visited = set()

        for state, action, reward in reversed(episode):
            G = self.gamma * G + reward
            state_action = (state, action)

            # First-visit：
            # 同一局裡同一組 (state, action) 只更新一次
            if state_action not in visited:
                visited.add(state_action)

                self.returns_sum[state_action] += G
                self.returns_count[state_action] += 1

                self.Q[state][action] = (
                    self.returns_sum[state_action]
                    / self.returns_count[state_action]
                )

        # 每一局結束後，慢慢降低探索率
        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay
        )

    def save(self, filename: str) -> None:
        """
        儲存訓練好的 Q-table。
        """
        data = {
            "actions": self.actions,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
            "Q": dict(self.Q),
            "returns_sum": dict(self.returns_sum),
            "returns_count": dict(self.returns_count),
        }

        with open(filename, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, filename: str) -> "FirstVisitMCAgent":
        """
        載入訓練好的 Q-table。
        """
        with open(filename, "rb") as f:
            data = pickle.load(f)

        agent = cls(
            actions=data["actions"],
            gamma=data["gamma"],
            epsilon=data["epsilon"],
            epsilon_min=data["epsilon_min"],
            epsilon_decay=data["epsilon_decay"],
        )

        agent.Q.update(data["Q"])
        agent.returns_sum.update(data["returns_sum"])
        agent.returns_count.update(data["returns_count"])

        return agent