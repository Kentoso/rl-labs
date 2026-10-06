from agent import Action
from discrete_car import DiscreteCar, StateIndices


class ValueIteration(DiscreteCar):
    title = "Value iteration"
    change_label = "Bellman residual  max |V' − V|"

    def update(self) -> float:
        new_values: dict[StateIndices, float] = {}
        for state in self.states:
            if self.is_goal(self.to_continuous(state)):
                new_values[state] = 0.0
            else:
                new_values[state] = max(self.q_value(state, a) for a in Action)

        delta = max(abs(new_values[s] - self.values[s]) for s in self.states)
        self.values = new_values
        return delta
