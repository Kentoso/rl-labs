import gymnasium as gym
from agent import Action
from discrete_car import DiscreteCar, StateIndices


class PolicyIteration(DiscreteCar):
    title = "Policy iteration"
    change_label = "Actions changed"

    def __init__(
        self,
        env: gym.Env,
        n_positions: int = 200,
        n_velocities: int = 200,
        gamma: float = 0.99,
        theta: float = 1e-4,
    ):
        super().__init__(env, n_positions, n_velocities, gamma)
        self.theta = theta
        # Arbitrary starting policy: never push
        self.policy = {state: Action.NO_PUSH for state in self.states}
        self.evaluation_sweeps: list[int] = []

    # One policy iteration step: evaluate the current policy, then improve it
    def update(self) -> float:
        self.evaluate_policy()
        return self.improve_policy()

    def policy_action(self, state: StateIndices) -> Action:
        return self.policy[state]

    def evaluate_policy(self) -> None:
        sweeps = 0
        while True:
            sweeps += 1
            new_values: dict[StateIndices, float] = {}
            for state in self.states:
                if self.is_goal(self.to_continuous(state)):
                    new_values[state] = 0.0
                else:
                    new_values[state] = self.q_value(state, self.policy[state])

            delta = max(abs(new_values[s] - self.values[s]) for s in self.states)
            self.values = new_values
            if delta <= self.theta:
                break
        self.evaluation_sweeps.append(sweeps)

    def improve_policy(self) -> int:
        changed = 0
        for state in self.states:
            current = self.policy[state]
            best = self.best_action(state)
            # Switch only if strictly better, so ties can't make the policy flip back and forth forever
            if self.q_value(state, best) > self.q_value(state, current):
                self.policy[state] = best
                changed += 1
        return changed
