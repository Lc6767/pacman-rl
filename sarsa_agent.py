"""
sarsa_agent.py

SARSA (On-policy TD Control) agent for the PacMan RL environment.

SARSA update:
Q(S, A) <- Q(S, A) + alpha * [R + gamma * Q(S', A') - Q(S, A)]

Compared with Monte Carlo:
- Monte Carlo waits until one whole episode ends, then updates Q.
- SARSA updates after every single step, so it usually learns faster and reacts more immediately.
"""

from __future__ import annotations

import pickle
import random
from collections import defaultdict
from typing import DefaultDict, Hashable, List


State = Hashable
Action = int


class SarsaAgent:
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
        next_action: Action | None,
        done: bool,
    ) -> None:
        current_q = self.Q[state][action]

        if done or next_action is None:
            target = reward
        else:
            target = reward + self.gamma * self.Q[next_state][next_action]

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
    def load(cls, filename: str) -> "SarsaAgent":
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