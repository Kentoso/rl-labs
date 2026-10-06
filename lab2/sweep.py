import argparse
from concurrent.futures import ProcessPoolExecutor

from matplotlib.figure import Figure

from plots import GRID, INK, RESULTS_DIR, plot_policy_map, style_axis
from training import Shaping, TrainingRun, train

# Same total weight 0.8 everywhere, only the split between the two terms changes
SHAPINGS = [
    Shaping(0.8, 0.0),
    Shaping(0.6, 0.2),
    Shaping(0.2, 0.6),
    Shaping(0.0, 0.8),
]
# Fixed colour per shaping, in the same order as SHAPINGS
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]


def train_one(shaping: Shaping, n_episodes: int, bins: int) -> TrainingRun:
    # Progress bars from parallel processes would overwrite each other
    return train(shaping, n_episodes, bins, show_progress=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100_000)
    parser.add_argument("--bins", type=int, default=200)
    args = parser.parse_args()

    # Each shaping trains in its own process
    with ProcessPoolExecutor(len(SHAPINGS)) as pool:
        futures = [
            pool.submit(train_one, shaping, args.episodes, args.bins)
            for shaping in SHAPINGS
        ]
        runs = [future.result() for future in futures]

    fig = Figure(figsize=(16, 9), layout="constrained")
    # Top: one final policy map per shaping. Bottom: one wide success chart.
    layout = fig.add_gridspec(2, len(runs), height_ratios=[1, 0.8])
    for k, run in enumerate(runs):
        ax = fig.add_subplot(layout[0, k])
        plot_policy_map(ax, run.agent)
        style_axis(ax, f"{run.shaping.name} · final policy", "Position", "Velocity")

    ax = fig.add_subplot(layout[1, :])
    for run, color in zip(runs, COLORS):
        success = [100 * s for s in run.eval_success]
        # Runs that never succeed overlap at 0%, so the legend also gives the final rate
        ax.plot(
            run.eval_points,
            success,
            color=color,
            linewidth=2,
            marker="o",
            markersize=5,
            label=f"{run.shaping.name} · final {success[-1]:.0f}%",
        )
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_ylim(-5, 105)
    ax.legend(frameon=False, labelcolor=INK, loc="upper left")
    style_axis(
        ax, "Greedy policy · success rate", "Episode", "Episodes reaching the flag (%)"
    )

    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / "sweep.png"
    fig.savefig(path, dpi=110)
    print(f"Saved comparison to {path}")


if __name__ == "__main__":
    main()
