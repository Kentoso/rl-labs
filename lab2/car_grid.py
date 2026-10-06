from enum import IntEnum
from typing import NewType

import gymnasium as gym
import numpy as np
from gymnasium.envs.classic_control.mountain_car import MountainCarEnv


class Action(IntEnum):
    PUSH_LEFT = 0
    NO_PUSH = 1
    PUSH_RIGHT = 2


# Built once, so random exploration doesn't create a new list on every step
ACTIONS = list(Action)

# Grid cell (i, j): position bin i, velocity bin j
StateIndices = NewType("StateIndices", tuple[int, int])
# The car's real (position, velocity)
PhysicalState = NewType("PhysicalState", tuple[float, float])


# The same grid as in lab1, without the model: the agent only learns from env.step
class CarGrid:
    def __init__(self, env: gym.Env, n_positions: int = 200, n_velocities: int = 200):
        car = env.unwrapped
        assert isinstance(car, MountainCarEnv)
        assert isinstance(car.action_space, gym.spaces.Discrete)
        assert car.action_space.n == len(Action)
        self.min_position = car.min_position
        self.max_position = car.max_position
        self.max_speed = car.max_speed
        self.goal_position = car.goal_position
        self.goal_velocity = car.goal_velocity

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

        self.states = [
            StateIndices((i, j))
            for i in range(n_positions)
            for j in range(n_velocities)
        ]

    def observe(self, observation) -> StateIndices:
        physical = PhysicalState((float(observation[0]), float(observation[1])))
        return self.to_discrete(physical)

    def to_continuous(self, state: StateIndices) -> PhysicalState:
        i, j = state
        return PhysicalState((self.positions[i], self.velocities[j]))

    def to_discrete(self, physical: PhysicalState) -> StateIndices:
        pos, vel = physical
        i = round((pos - self.min_position) / self.position_step)
        j = round((vel + self.max_speed) / self.velocity_step)
        return StateIndices((i, j))
