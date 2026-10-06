import math
from typing import NewType

import gymnasium as gym
import numpy as np
from agent import Action
from gymnasium.envs.classic_control.mountain_car import MountainCarEnv

# Grid cell (i, j): position bin i, velocity bin j
StateIndices = NewType("StateIndices", tuple[int, int])
# The car's real (position, velocity)
PhysicalState = NewType("PhysicalState", tuple[float, float])


# The grid, the model and the values shared by value and policy iteration
class DiscreteCar:
    title = "Discrete car"
    change_label = "Change"

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
        assert car.action_space.n == len(Action)
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

        self.states = [
            StateIndices((i, j))
            for i in range(n_positions)
            for j in range(n_velocities)
        ]
        self.values = {state: 0.0 for state in self.states}

        # The car is deterministic, so the model is one (next_state, done) per state and action
        self.model: dict[tuple[StateIndices, Action], tuple[StateIndices, bool]] = {}
        for state in self.states:
            physical = self.to_continuous(state)
            for action in Action:
                next_physical = self.step(physical, action)
                self.model[state, action] = (
                    self.to_discrete(next_physical),
                    self.is_goal(next_physical),
                )

    def act(self, observation) -> Action:
        physical = PhysicalState((float(observation[0]), float(observation[1])))
        return self.policy_action(self.to_discrete(physical))

    def policy_action(self, state: StateIndices) -> Action:
        return self.best_action(state)

    def best_action(self, state: StateIndices) -> Action:
        return max(Action, key=lambda a: self.q_value(state, a))

    def q_value(self, state: StateIndices, action: Action) -> float:
        next_state, done = self.model[state, action]
        if done:
            return -1.0
        return -1.0 + self.gamma * self.values[next_state]

    def to_continuous(self, state: StateIndices) -> PhysicalState:
        i, j = state
        return PhysicalState((self.positions[i], self.velocities[j]))

    def to_discrete(self, physical: PhysicalState) -> StateIndices:
        pos, vel = physical
        i = round((pos - self.min_position) / self.position_step)
        j = round((vel + self.max_speed) / self.velocity_step)
        return StateIndices((i, j))

    def step(self, physical: PhysicalState, action: Action) -> PhysicalState:
        # Same physics as MountainCarEnv.step
        pos, vel = physical
        # action - 1 is the push direction: PUSH_LEFT -1, NO_PUSH 0, PUSH_RIGHT +1
        vel += (action - 1) * self.force - math.cos(3 * pos) * self.gravity
        vel = float(np.clip(vel, -self.max_speed, self.max_speed))
        pos = float(np.clip(pos + vel, self.min_position, self.max_position))
        if pos == self.min_position and vel < 0:
            vel = 0.0
        return PhysicalState((pos, vel))

    def is_goal(self, physical: PhysicalState) -> bool:
        pos, vel = physical
        return pos >= self.goal_position and vel >= self.goal_velocity
