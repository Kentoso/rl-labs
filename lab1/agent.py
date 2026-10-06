from collections.abc import Callable
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Protocol

import gymnasium as gym
import numpy as np


class Action(IntEnum):
    PUSH_LEFT = 0
    NO_PUSH = 1
    PUSH_RIGHT = 2


class PolicyMethod(Protocol):
    def update(self) -> float: ...

    def act(self, observation: np.ndarray) -> Action: ...


@dataclass
class Episode:
    states: np.ndarray  # (steps + 1, 2): [position, velocity] per step
    total_reward: float
    reached_goal: bool


@dataclass
class TrainingHistory:
    deltas: list[float] = field(default_factory=list)
    eval_iterations: list[int] = field(default_factory=list)
    eval_returns: list[float] = field(default_factory=list)
    eval_success_rates: list[float] = field(default_factory=list)


class CarAgent:
    def __init__(self, env: gym.Env, method: PolicyMethod, seed: int = 0):
        self.env = env
        self.method = method
        self.seed = seed

    def act(self, observation: np.ndarray) -> Action:
        return self.method.act(observation)

    def train(
        self,
        max_iterations: int = 1000,
        tolerance: float = 1e-4,
        eval_every: int = 25,
        eval_episodes: int = 20,
        on_iteration: Callable[[int], None] | None = None,
    ) -> TrainingHistory:
        history = TrainingHistory()
        for iteration in range(1, max_iterations + 1):
            delta = self.method.update()
            history.deltas.append(delta)
            if on_iteration is not None:
                on_iteration(iteration)

            converged = delta < tolerance
            if iteration % eval_every == 0 or converged:
                mean_return, success_rate = self.evaluate(eval_episodes)
                history.eval_iterations.append(iteration)
                history.eval_returns.append(mean_return)
                history.eval_success_rates.append(success_rate)
                print(
                    f"iter {iteration:4d}  change {delta:<8.4g}  "
                    f"return {mean_return:7.1f}  success {success_rate:.0%}"
                )
            if converged:
                print(f"Converged after {iteration} iterations")
                break
        return history

    def evaluate(self, n_episodes: int) -> tuple[float, float]:
        episodes = [
            self.run_episode(self.env, seed=self.seed + i) for i in range(n_episodes)
        ]
        mean_return = float(np.mean([e.total_reward for e in episodes]))
        success_rate = float(np.mean([e.reached_goal for e in episodes]))
        return mean_return, success_rate

    def run_episode(self, env: gym.Env, seed: int | None = None) -> Episode:
        observation, _ = env.reset(seed=seed)
        states = [observation]
        total_reward = 0.0
        while True:
            observation, reward, terminated, truncated, _ = env.step(
                self.act(observation)
            )
            states.append(observation)
            total_reward += float(reward)
            if terminated or truncated:
                return Episode(np.array(states), total_reward, terminated)
