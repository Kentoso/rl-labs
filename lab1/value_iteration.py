import math

import gymnasium as gym
import numpy as np
from gymnasium.envs.classic_control.mountain_car import MountainCarEnv


class ValueIteration:
    def __init__(
        self,
        env: gym.Env,
        n_positions: int = 200,
        n_velocities: int = 200,
        gamma: float = 0.99,
    ):
        car = env.unwrapped
        assert isinstance(car, MountainCarEnv)
        assert isinstance(car.action_space, gym.spaces.Discrete)
        self.force = car.force
        self.gravity = car.gravity
        self.min_position = car.min_position
        self.max_position = car.max_position
        self.max_speed = car.max_speed
        self.goal_position = car.goal_position
        self.goal_velocity = car.goal_velocity
        self.actions = list(range(car.action_space.n))
        self.gamma = gamma

        self.n_positions = n_positions
        self.n_velocities = n_velocities
        # Evenly spaced grid points, both ends included
        self.positions = np.linspace(
            self.min_position, self.max_position, n_positions
        ).tolist()
        self.velocities = np.linspace(
            -self.max_speed, self.max_speed, n_velocities
        ).tolist()
        self.position_step = self.positions[1] - self.positions[0]
        self.velocity_step = self.velocities[1] - self.velocities[0]

        # A discrete state is a grid cell (i, j): position bin i, velocity bin j
        self.states = [(i, j) for i in range(n_positions) for j in range(n_velocities)]
        self.values = {state: 0.0 for state in self.states}

        # The car is deterministic, so the model is one (next_state, done) per state and action
        self.model = {}
        for state in self.states:
            pos, vel = self.to_continuous(state)
            for action in self.actions:
                next_pos, next_vel = self.step(pos, vel, action)
                next_state = self.to_discrete(next_pos, next_vel)
                self.model[state, action] = (
                    next_state,
                    self.is_goal(next_pos, next_vel),
                )

    def update(self) -> float:
        new_values = {}
        for state in self.states:
            if self.is_goal(*self.to_continuous(state)):
                new_values[state] = 0.0
            else:
                new_values[state] = max(self.q_value(state, a) for a in self.actions)

        delta = max(abs(new_values[s] - self.values[s]) for s in self.states)
        self.values = new_values
        return delta

    def act(self, observation) -> int:
        return self.best_action(self.to_discrete(observation[0], observation[1]))

    def best_action(self, state: tuple[int, int]) -> int:
        return max(self.actions, key=lambda a: self.q_value(state, a))

    def q_value(self, state: tuple[int, int], action: int) -> float:
        next_state, done = self.model[state, action]
        if done:
            return -1.0
        return -1.0 + self.gamma * self.values[next_state]

    def to_continuous(self, state: tuple[int, int]) -> tuple[float, float]:
        i, j = state
        return self.positions[i], self.velocities[j]

    def to_discrete(self, pos: float, vel: float) -> tuple[int, int]:
        i = round((pos - self.min_position) / self.position_step)
        j = round((vel + self.max_speed) / self.velocity_step)
        return i, j

    def step(self, pos: float, vel: float, action: int) -> tuple[float, float]:
        # Same physics as MountainCarEnv.step
        vel += (action - 1) * self.force - math.cos(3 * pos) * self.gravity
        vel = min(max(vel, -self.max_speed), self.max_speed)
        pos = min(max(pos + vel, self.min_position), self.max_position)
        if pos == self.min_position and vel < 0:
            vel = 0.0
        return pos, vel

    def is_goal(self, pos: float, vel: float) -> bool:
        return pos >= self.goal_position and vel >= self.goal_velocity
