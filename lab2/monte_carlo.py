import random

from car_grid import Action, CarGrid, StateIndices

# One step of an episode: the state, the action taken in it, and the reward that followed
Step = tuple[StateIndices, Action, float]


# Every-visit Monte Carlo control with epsilon-greedy exploration
class MonteCarloAgent:
    def __init__(
        self,
        grid: CarGrid,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
        seed: int = 0,
    ):
        self.grid = grid
        # Q-table: one value per action for every grid cell, all starting at 0
        self.q_values = {state: [0.0] * len(Action) for state in grid.states}
        self.lr = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.rng = random.Random(seed)
        self.training_error: list[float] = []

    def get_action(self, state: StateIndices) -> Action:
        # With probability epsilon explore, otherwise take the best known action
        if self.rng.random() < self.epsilon:
            return self.rng.choice(list(Action))
        return self.greedy_action(state)

    def greedy_action(self, state: StateIndices) -> Action:
        q = self.q_values[state]
        return max(Action, key=lambda a: q[a])

    def update(self, episode: list[Step]) -> None:
        # Walk backwards, accumulating the discounted return G from each step
        g = 0.0
        for state, action, reward in reversed(episode):
            g = reward + self.discount_factor * g
            # Every visit moves Q towards its return; recent episodes weigh more
            error = g - self.q_values[state][action]
            self.q_values[state][action] += self.lr * error
            self.training_error.append(error)

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

    def act(self, observation) -> Action:
        return self.greedy_action(self.grid.observe(observation))
