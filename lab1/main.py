from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import pygame
from agent import Action, CarAgent, Episode, TrainingHistory
from matplotlib.colors import ListedColormap
from value_iteration import StateIndices, ValueIteration

RESULTS_PATH = Path(__file__).with_name("training_results.png")

INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
SERIES = "#2a78d6"
# Categorical slots for the three actions
ACTION_COLORS = {
    Action.PUSH_LEFT: "#2a78d6",
    Action.NO_PUSH: "#e1e0d9",
    Action.PUSH_RIGHT: "#eb6834",
}
ACTION_LABELS = {
    Action.PUSH_LEFT: "Push left",
    Action.NO_PUSH: "No push",
    Action.PUSH_RIGHT: "Push right",
}


def style_axis(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")


def plot_line(ax: plt.Axes, x, y, **kwargs) -> None:
    ax.plot(x, y, color=SERIES, linewidth=2, **kwargs)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_results(
    history: TrainingHistory, method: ValueIteration, episode: Episode
) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), layout="constrained")
    fig.suptitle(
        "MountainCar · value iteration", x=0.01, ha="left", fontsize=14, color=INK
    )

    ax = axes[0, 0]
    plot_line(ax, range(1, len(history.deltas) + 1), history.deltas)
    style_axis(ax, "Bellman residual", "Iteration", "max |V' − V|")

    ax = axes[0, 1]
    plot_line(
        ax, history.eval_iterations, history.eval_returns, marker="o", markersize=5
    )
    ax.axhline(-200, color=MUTED, linewidth=1, linestyle=":")
    ax.annotate(
        "time limit",
        (history.eval_iterations[-1], -200),
        xytext=(0, 4),
        textcoords="offset points",
        ha="right",
        color=MUTED,
        fontsize=9,
    )
    style_axis(ax, "Greedy policy · mean return", "Iteration", "Return")

    ax = axes[1, 0]
    success = [rate * 100 for rate in history.eval_success_rates]
    plot_line(ax, history.eval_iterations, success, marker="o", markersize=5)
    ax.set_ylim(-5, 105)
    style_axis(
        ax,
        "Greedy policy · success rate",
        "Iteration",
        "Episodes reaching the flag (%)",
    )

    extent = [
        method.min_position,
        method.max_position,
        -method.max_speed,
        method.max_speed,
    ]
    trajectory = episode.states
    # Image rows are velocity bins j, columns are position bins i
    positions, velocities = range(method.n_positions), range(method.n_velocities)
    policy_grid = [
        [method.best_action(StateIndices((i, j))) for i in positions]
        for j in velocities
    ]

    ax = axes[1, 1]
    ax.imshow(
        policy_grid,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap=ListedColormap([ACTION_COLORS[a] for a in Action]),
        vmin=-0.5,
        vmax=2.5,
        interpolation="nearest",
    )
    handles = [plt.Rectangle((0, 0), 1, 1, color=ACTION_COLORS[a]) for a in Action]
    labels = [ACTION_LABELS[a] for a in Action]
    ax.legend(handles, labels, loc="upper left", fontsize=8, frameon=True)
    ax.plot(trajectory[:, 0], trajectory[:, 1], color=INK, linewidth=1.5)
    ax.plot(*trajectory[0], "o", color=INK, markersize=8)
    ax.axvline(method.goal_position, color=INK, linewidth=1, linestyle=":")
    style_axis(ax, "Greedy policy + sample trajectory", "Position", "Velocity")

    return fig


def print_setup(method: ValueIteration) -> None:
    n_pos, n_vel = method.n_positions, method.n_velocities
    n_goal = sum(method.is_goal(method.to_continuous(s)) for s in method.states)
    n_actions = len(Action)

    print("Car physics")
    print(f"  position range     [{method.min_position}, {method.max_position}]")
    print(f"  velocity range     [{-method.max_speed}, {method.max_speed}]")
    print(
        f"  goal               position >= {method.goal_position}, velocity >= {method.goal_velocity}"
    )
    print(f"  engine force       {method.force}")
    print(f"  gravity            {method.gravity}")
    print(
        "  dynamics           v' = v + (a - 1) * force - cos(3x) * gravity,  x' = x + v'"
    )
    print(
        f"  actions ({n_actions})        "
        + ", ".join(f"{a.value} = {a.name}" for a in Action)
    )
    print("  reward             -1 per step until the goal")
    print("Value iteration")
    print(f"  position bins      {n_pos} (step {method.position_step:.4f})")
    print(f"  velocity bins      {n_vel} (step {method.velocity_step:.5f})")
    print(
        f"  discrete states    {n_pos} x {n_vel} = {n_pos * n_vel} ({n_goal} of them goal states)"
    )
    print(f"  state-action pairs {n_pos * n_vel * n_actions}")
    print("  next state         snapped to the nearest grid cell")
    print(f"  discount (gamma)   {method.gamma}")
    print()


def watch_episode(agent: CarAgent, seed: int) -> None:
    env = gym.make("MountainCar-v0", render_mode="human")
    episode = agent.run_episode(env, seed=seed)
    print(
        f"Demo episode: {len(episode.states) - 1} steps. Close the window to see the plots."
    )
    # Keep the last frame on screen until the window is closed
    while not any(event.type == pygame.QUIT for event in pygame.event.get()):
        pygame.time.wait(50)
    env.close()


def main() -> None:
    env = gym.make("MountainCar-v0")
    method = ValueIteration(env)
    agent = CarAgent(env, method)
    print_setup(method)

    history = agent.train()

    episode = agent.run_episode(env, seed=1234)
    print(
        f"Sample episode: return {episode.total_reward}, reached goal: {episode.reached_goal}"
    )
    env.close()

    watch_episode(agent, seed=1234)

    fig = plot_results(history, method, episode)
    fig.savefig(RESULTS_PATH, dpi=120)
    print(f"Saved plots to {RESULTS_PATH}")
    plt.show()


if __name__ == "__main__":
    main()
