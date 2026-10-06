import argparse
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from tqdm import tqdm

from car_grid import Action, CarGrid, StateIndices
from monte_carlo import MonteCarloAgent, Step

RESULTS_PATH = Path(__file__).with_name("training_results.png")

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


# Play a whole episode with the exploring policy, then learn from it
def train_episode(env: gym.Env, agent: MonteCarloAgent) -> None:
    observation, _ = env.reset()
    episode: list[Step] = []
    while True:
        state = agent.grid.observe(observation)
        action = agent.get_action(state)
        observation, reward, terminated, truncated, _ = env.step(action)
        episode.append((state, action, float(reward)))
        if terminated or truncated:
            break
    agent.update(episode)


# Greedy policy, no exploration and no learning
def evaluate(agent: MonteCarloAgent, n_episodes: int) -> tuple[float, float]:
    env = gym.make("MountainCar-v0")
    returns, successes = [], []
    for seed in range(n_episodes):
        observation, _ = env.reset(seed=seed)
        total_reward = 0.0
        while True:
            observation, reward, terminated, truncated, _ = env.step(
                agent.act(observation)
            )
            total_reward += float(reward)
            if terminated or truncated:
                break
        returns.append(total_reward)
        successes.append(terminated)
    env.close()
    return float(np.mean(returns)), float(np.mean(successes))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100_000)
    parser.add_argument("--bins", type=int, default=40)
    args = parser.parse_args()

    n_episodes = args.episodes
    # Constant step size, so older episodes are gradually forgotten
    learning_rate = 0.1
    # Exploration schedule from the Gymnasium tutorial
    start_epsilon = 1.0
    epsilon_decay = start_epsilon / (n_episodes / 2)
    final_epsilon = 0.1
    eval_every = n_episodes // 20
    eval_episodes = 20

    env = gym.make("MountainCar-v0")
    env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)
    env.reset(seed=0)
    grid = CarGrid(env, args.bins, args.bins)
    agent = MonteCarloAgent(
        grid, learning_rate, start_epsilon, epsilon_decay, final_epsilon
    )
    print(f"Grid {args.bins} x {args.bins}, {n_episodes} episodes")

    eval_points, eval_returns, eval_success = [], [], []
    for episode in tqdm(range(1, n_episodes + 1)):
        train_episode(env, agent)
        agent.decay_epsilon()
        if episode % eval_every == 0:
            mean_return, success_rate = evaluate(agent, eval_episodes)
            eval_points.append(episode)
            eval_returns.append(mean_return)
            eval_success.append(success_rate)
            # tqdm.write prints above the progress bar instead of breaking it
            tqdm.write(
                f"episode {episode:7d}  epsilon {agent.epsilon:.2f}  "
                f"greedy return {mean_return:7.1f}  success {success_rate:.0%}"
            )
    env.close()

    window = max(1, n_episodes // 200)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), layout="constrained")
    fig.suptitle(
        f"MountainCar · Monte Carlo, {args.bins} x {args.bins} grid",
        x=0.01,
        ha="left",
        fontsize=14,
        color=INK,
    )

    ax = axes[0, 0]
    training_returns = list(env.return_queue)
    plot_line(
        ax,
        range(window, len(training_returns) + 1),
        moving_average(training_returns, window),
    )
    style_axis(
        ax, f"Training return (moving average, {window} episodes)", "Episode", "Return"
    )

    ax = axes[0, 1]
    plot_line(ax, eval_points, eval_returns, marker="o", markersize=5)
    ax.axhline(-200, color=MUTED, linewidth=1, linestyle=":")
    style_axis(ax, "Greedy policy · mean return", "Episode", "Return")

    ax = axes[1, 0]
    errors = [abs(e) for e in agent.training_error]
    error_window = window * 200
    plot_line(
        ax,
        range(error_window, len(errors) + 1),
        moving_average(errors, error_window),
    )
    style_axis(ax, "Absolute error |G − Q| (moving average)", "Update", "|G − Q|")

    ax = axes[1, 1]
    plot_policy_map(ax, agent)
    style_axis(ax, "Greedy policy", "Position", "Velocity")

    fig.savefig(RESULTS_PATH, dpi=120)
    print(f"Saved plots to {RESULTS_PATH}")
    plt.show()


if __name__ == "__main__":
    main()
