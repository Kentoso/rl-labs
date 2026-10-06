import argparse

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pygame

from monte_carlo import MonteCarloAgent
from plots import (
    INK,
    MUTED,
    RESULTS_DIR,
    moving_average,
    plot_line,
    plot_policy_map,
    style_axis,
)
from training import Shaping, TrainingRun, run_greedy, train


def watch_episode(agent: MonteCarloAgent, seed: int) -> None:
    env = gym.make("MountainCar-v0", render_mode="human")
    states, _, _ = run_greedy(agent, env, seed)
    print(f"Demo episode: {len(states) - 1} steps. Close the window to see the plots.")
    # Keep the last frame on screen until the window is closed
    while not any(event.type == pygame.QUIT for event in pygame.event.get()):
        pygame.time.wait(50)
    env.close()


def plot_results(run: TrainingRun, trajectory: np.ndarray) -> plt.Figure:
    window = max(1, len(run.training_returns) // 200)
    fig, axes = plt.subplots(2, 3, figsize=(17, 9), layout="constrained")
    grid = run.agent.grid
    fig.suptitle(
        f"MountainCar · Monte Carlo, {grid.n_positions} x {grid.n_velocities} grid, "
        f"{run.shaping.name}",
        x=0.01,
        ha="left",
        fontsize=14,
        color=INK,
    )

    ax = axes[0, 0]
    returns = run.training_returns
    plot_line(ax, range(window, len(returns) + 1), moving_average(returns, window))
    style_axis(
        ax, f"Training return (moving average, {window} episodes)", "Episode", "Return"
    )

    ax = axes[0, 1]
    plot_line(ax, run.eval_points, run.eval_returns, marker="o", markersize=5)
    ax.axhline(-200, color=MUTED, linewidth=1, linestyle=":")
    style_axis(ax, "Greedy policy · mean return", "Episode", "Return")

    ax = axes[0, 2]
    success = [100 * s for s in run.eval_success]
    plot_line(ax, run.eval_points, success, marker="o", markersize=5)
    ax.set_ylim(-5, 105)
    style_axis(
        ax, "Greedy policy · success rate", "Episode", "Episodes reaching the flag (%)"
    )

    ax = axes[1, 0]
    plot_line(ax, range(1, len(run.epsilons) + 1), run.epsilons)
    ax.set_ylim(0, 1.05)
    style_axis(ax, "Exploration rate ε", "Episode", "ε")

    ax = axes[1, 1]
    errors = run.agent.training_error
    plot_line(ax, range(window, len(errors) + 1), moving_average(errors, window))
    style_axis(ax, "Mean |G − Q| per episode (moving average)", "Episode", "|G − Q|")

    ax = axes[1, 2]
    plot_policy_map(ax, run.agent)
    ax.plot(trajectory[:, 0], trajectory[:, 1], color=INK, linewidth=1.5)
    ax.plot(*trajectory[0], "o", color=INK, markersize=8)
    style_axis(ax, "Greedy policy + sample trajectory", "Position", "Velocity")
    return fig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100_000)
    parser.add_argument("--bins", type=int, default=200)
    parser.add_argument("--velocity-weight", type=float, default=0.5)
    parser.add_argument("--position-weight", type=float, default=0.3)
    args = parser.parse_args()

    shaping = Shaping(args.velocity_weight, args.position_weight)
    run = train(shaping, args.episodes, args.bins)

    sample_env = gym.make("MountainCar-v0")
    trajectory, sample_return, reached = run_greedy(run.agent, sample_env, seed=1234)
    sample_env.close()
    print(f"Sample episode: return {sample_return}, reached goal: {reached}")
    watch_episode(run.agent, seed=1234)

    fig = plot_results(run, trajectory)
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"{shaping.name}.png"
    fig.savefig(path, dpi=120)
    print(f"Saved plots to {path}")
    plt.show()


if __name__ == "__main__":
    main()
