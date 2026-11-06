"""
qlearning_agent.py

Q-learning (Off-policy TD Control) agent for the PacMan RL environment.

說明：
小組當時的 Q-learning 原始程式沒有保存下來。
這個檔案是課後依照 sarsa_agent.py 的架構重新實作的版本，
環境、參數與 SARSA 相同，只改了更新公式。

Q-learning update:
Q(S, A) <- Q(S, A) + alpha * [R + gamma * max_a Q(S', a) - Q(S, A)]

和 SARSA 的差別：
- SARSA（on-policy）用「實際會選的下一個動作 A'」來更新：
  target = R + gamma * Q(S', A')
  A' 可能是 epsilon-greedy 隨機選到的動作，所以會把探索的風險算進去，策略較保守。
- Q-learning（off-policy）用「下一個狀態中最好的動作」來更新：
  target = R + gamma * max_a Q(S', a)
  不管實際上會不會選到那個動作，學的是「最佳策略」的價值，策略較積極。
"""

from __future__ import annotations

import pickle
import random
from collections import defaultdict
from typing import DefaultDict, Hashable, List


State = Hashable
Action = int


class QLearningAgent:
    def __init__(
        self,
        actions: List[Action],
        alpha: float = 0.10,
        gamma: float = 0.95,
        epsilon: float = 0.30,
        epsilon_min: float = 0.02,
        epsilon_decay: float = 0.999,
        seed: int | None = None,
    ) -> None:
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng = random.Random(seed)

        self.Q: DefaultDict[State, List[float]] = defaultdict(
            lambda: [0.0 for _ in self.actions]
        )

    def choose_action(self, state: State, training: bool = True) -> Action:
        """epsilon-greedy：訓練時有 epsilon 的機率隨機探索，其餘選 Q 值最大的動作。"""
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

    def update(
        self,
        state: State,
        action: Action,
        reward: float,
        next_state: State,
        done: bool,
    ) -> None:
        current_q = self.Q[state][action]

        if done:
            target = reward
        else:
            # 唯一和 SARSA 不同的地方：用下一個狀態的最大 Q 值，而不是實際選到的動作
            target = reward + self.gamma * max(self.Q[next_state])

        self.Q[state][action] = current_q + self.alpha * (target - current_q)

    def decay_epsilon(self) -> None:
        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay,
        )

    def save(self, filename: str) -> None:
        data = {
            "actions": self.actions,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
            "Q": dict(self.Q),
        }

        with open(filename, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, filename: str) -> "QLearningAgent":
        with open(filename, "rb") as f:
            data = pickle.load(f)

        agent = cls(
            actions=data["actions"],
            alpha=data["alpha"],
            gamma=data["gamma"],
            epsilon=data["epsilon"],
            epsilon_min=data["epsilon_min"],
            epsilon_decay=data["epsilon_decay"],
        )

        agent.Q.update(data["Q"])
        return agent
