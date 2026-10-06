import argparse
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import pygame
from agent import Action, CarAgent, Episode, TrainingHistory
from discrete_car import DiscreteCar, StateIndices
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from policy_iteration import PolicyIteration
from value_iteration import ValueIteration

RESULTS_PATH = Path(__file__).with_name("training_results.png")
POLICY_MAPS_DIR = Path(__file__).with_name("policy_maps")
SNAPSHOT_EVERY = 10

INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
SERIES = "#2a78d6"
GOAL = "#1f9d55"
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
    history: TrainingHistory, method: DiscreteCar, episode: Episode
) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), layout="constrained")
    fig.suptitle(
        f"MountainCar · {method.title.lower()}",
        x=0.01,
        ha="left",
        fontsize=14,
        color=INK,
    )

    ax = axes[0, 0]
    plot_line(ax, range(1, len(history.deltas) + 1), history.deltas)
    style_axis(ax, method.change_label, "Iteration", method.change_label)

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

    ax = axes[1, 1]
    plot_policy_map(ax, method)
    trajectory = episode.states
    ax.plot(trajectory[:, 0], trajectory[:, 1], color=INK, linewidth=1.5)
    ax.plot(*trajectory[0], "o", color=INK, markersize=8)
    style_axis(ax, "Greedy policy + sample trajectory", "Position", "Velocity")

    return fig


def plot_policy_map(ax: plt.Axes, method: DiscreteCar) -> None:
    extent = (
        method.min_position,
        method.max_position,
        -method.max_speed,
        method.max_speed,
    )
    # Image rows are velocity bins j, columns are position bins i
    positions, velocities = range(method.n_positions), range(method.n_velocities)
    policy_grid = [
        [method.policy_action(StateIndices((i, j))) for i in positions]
        for j in velocities
    ]
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
    # Goal zone: position >= goal_position and velocity >= goal_velocity
    goal = plt.Rectangle(
        (method.goal_position, method.goal_velocity),
        method.max_position - method.goal_position,
        method.max_speed - method.goal_velocity,
        fill=False,
        edgecolor=GOAL,
        linewidth=2.5,
        # The zone touches the top-right corner, so don't cut its edges at the axes border
        clip_on=False,
    )
    ax.add_patch(goal)

    handles = [plt.Rectangle((0, 0), 1, 1, color=ACTION_COLORS[a]) for a in Action]
    labels = [ACTION_LABELS[a] for a in Action]
    handles.append(plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor=GOAL, linewidth=2))
    labels.append("Goal zone")
    ax.legend(handles, labels, loc="upper left", fontsize=8, frameon=True)


def save_policy_map(method: DiscreteCar, iteration: int, folder: Path) -> None:
    # A plain Figure only renders to a file. plt.subplots would start the GUI backend,
    # which breaks the pygame demo window that opens later.
    fig = Figure(figsize=(6, 4.5), layout="constrained")
    ax = fig.subplots()
    plot_policy_map(ax, method)
    when = f"policy after iteration {iteration}" if iteration > 0 else "initial policy"
    style_axis(
        ax,
        f"{method.title} · {when}",
        "Position",
        "Velocity",
    )
    fig.savefig(folder / f"iter_{iteration:04d}.png", dpi=100)


def print_setup(method: DiscreteCar) -> None:
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
    print(method.title)
    print(f"  position bins      {n_pos} (step {method.position_step:.4f})")
    print(f"  velocity bins      {n_vel} (step {method.velocity_step:.5f})")
    print(
        f"  discrete states    {n_pos} x {n_vel} = {n_pos * n_vel} ({n_goal} of them goal states)"
    )
    print(f"  state-action pairs {n_pos * n_vel * n_actions}")
    print("  next state         snapped to the nearest grid cell")
    print(f"  discount (gamma)   {method.gamma}")
    if isinstance(method, PolicyIteration):
        print(f"  evaluation theta   {method.theta}")
        print("  initial policy     NO_PUSH everywhere")
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["value", "policy"], default="value")
    args = parser.parse_args()

    env = gym.make("MountainCar-v0")
    method: DiscreteCar
    if args.method == "policy":
        method = PolicyIteration(env)
        # Each policy iteration step is expensive and big, so evaluate after every one
        eval_every = 1
    else:
        method = ValueIteration(env)
        eval_every = 25
    agent = CarAgent(env, method)
    print_setup(method)

    # One folder per method; old snapshots are removed so a shorter run leaves no leftovers
    snapshots = POLICY_MAPS_DIR / args.method
    snapshots.mkdir(parents=True, exist_ok=True)
    for old in snapshots.glob("iter_*.png"):
        old.unlink()

    def snapshot(iteration: int) -> None:
        if iteration % SNAPSHOT_EVERY == 0:
            save_policy_map(method, iteration, snapshots)

    # Iteration 0 is the policy before any training
    save_policy_map(method, 0, snapshots)
    history = agent.train(eval_every=eval_every, on_iteration=snapshot)
    # Also keep the final policy
    save_policy_map(method, len(history.deltas), snapshots)
    print(f"Saved policy maps to {snapshots}")

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
