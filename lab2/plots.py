from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure

from car_grid import Action, StateIndices
from monte_carlo import MonteCarloAgent

RESULTS_DIR = Path(__file__).with_name("results")

INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
SERIES = "#2a78d6"
GOAL = "#1f9d55"
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


def moving_average(values: list[float], window: int) -> np.ndarray:
    return np.convolve(values, np.ones(window) / window, mode="valid")


def plot_policy_map(ax: plt.Axes, agent: MonteCarloAgent) -> None:
    grid = agent.grid
    extent = (grid.min_position, grid.max_position, -grid.max_speed, grid.max_speed)
    # Image rows are velocity bins j, columns are position bins i
    positions, velocities = range(grid.n_positions), range(grid.n_velocities)
    policy_grid = [
        [agent.greedy_action(StateIndices((i, j))) for i in positions]
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
    goal = plt.Rectangle(
        (grid.goal_position, grid.goal_velocity),
        grid.max_position - grid.goal_position,
        grid.max_speed - grid.goal_velocity,
        fill=False,
        edgecolor=GOAL,
        linewidth=2.5,
        clip_on=False,
    )
    ax.add_patch(goal)
    handles = [plt.Rectangle((0, 0), 1, 1, color=ACTION_COLORS[a]) for a in Action]
    labels = [ACTION_LABELS[a] for a in Action]
    handles.append(plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor=GOAL, linewidth=2))
    labels.append("Goal zone")
    ax.legend(handles, labels, loc="upper left", fontsize=8, frameon=True)


def save_policy_map(
    agent: MonteCarloAgent, label: str, episode: int, folder: Path
) -> None:
    # A plain Figure only renders to a file. plt.subplots would start the GUI backend,
    # which breaks the pygame demo window that opens later.
    fig = Figure(figsize=(6, 4.5), layout="constrained")
    ax = fig.subplots()
    plot_policy_map(ax, agent)
    when = f"after episode {episode}" if episode > 0 else "initial policy"
    style_axis(ax, f"{label} · {when}", "Position", "Velocity")
    fig.savefig(folder / f"episode_{episode:06d}.png", dpi=100)
