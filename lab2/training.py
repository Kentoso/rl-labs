from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
from tqdm import tqdm

from car_grid import CarGrid
from monte_carlo import MonteCarloAgent, Step
from plots import save_policy_map

POLICY_MAPS_DIR = Path(__file__).with_name("policy_maps")
SNAPSHOTS = 10


# Extra reward per step for speed and for being close to the goal. Each term is in
# [0, weight] and the two weights sum to less than 1, so every step still costs
# something and reaching the goal sooner stays better.
@dataclass(frozen=True)
class Shaping:
    velocity_weight: float
    position_weight: float

    @property
    def name(self) -> str:
        return f"vel{self.velocity_weight:g}_pos{self.position_weight:g}"


@dataclass
class TrainingRun:
    shaping: Shaping
    agent: MonteCarloAgent
    training_returns: list[float]
    eval_points: list[int]
    eval_returns: list[float]
    eval_success: list[float]
    # Epsilon used in each training episode
    epsilons: list[float]


def shaped_reward(grid: CarGrid, observation, reward: float, shaping: Shaping) -> float:
    position, velocity = float(observation[0]), float(observation[1])
    speed = abs(velocity) / grid.max_speed
    # 0 at the left wall, 1 at the goal position. The last step can overshoot the
    # goal (up to 1.06), so cap it to keep the term within [0, weight].
    progress = (position - grid.min_position) / (grid.goal_position - grid.min_position)
    progress = min(progress, 1.0)
    return reward + shaping.velocity_weight * speed + shaping.position_weight * progress


# Play a whole episode with the exploring policy, then learn from it
def train_episode(env: gym.Env, agent: MonteCarloAgent, shaping: Shaping) -> None:
    observation, _ = env.reset()
    episode: list[Step] = []
    while True:
        state = agent.grid.observe(observation)
        action = agent.get_action(state)
        observation, reward, terminated, truncated, _ = env.step(action)
        # The agent learns from the shaped reward; the env still records the original one
        episode.append(
            (
                state,
                action,
                shaped_reward(agent.grid, observation, float(reward), shaping),
            )
        )
        if terminated or truncated:
            break
    agent.update(episode)


# Greedy policy, no exploration and no learning.
# Returns the visited [position, velocity] rows, the return and whether the goal was reached.
def run_greedy(
    agent: MonteCarloAgent, env: gym.Env, seed: int
) -> tuple[np.ndarray, float, bool]:
    observation, _ = env.reset(seed=seed)
    states = [observation]
    total_reward = 0.0
    while True:
        observation, reward, terminated, truncated, _ = env.step(agent.act(observation))
        states.append(observation)
        total_reward += float(reward)
        if terminated or truncated:
            return np.array(states), total_reward, terminated


def evaluate(agent: MonteCarloAgent, n_episodes: int) -> tuple[float, float]:
    env = gym.make("MountainCar-v0")
    runs = [run_greedy(agent, env, seed) for seed in range(n_episodes)]
    env.close()
    mean_return = float(np.mean([total for _, total, _ in runs]))
    success_rate = float(np.mean([reached for _, _, reached in runs]))
    return mean_return, success_rate


def train(
    shaping: Shaping, n_episodes: int, bins: int, show_progress: bool = True
) -> TrainingRun:
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
    grid = CarGrid(env, bins, bins)
    agent = MonteCarloAgent(
        grid, learning_rate, start_epsilon, epsilon_decay, final_epsilon
    )
    print(f"{shaping.name}: grid {bins} x {bins}, {n_episodes} episodes")

    # One folder per shaping; old snapshots are removed so a shorter run leaves no leftovers
    maps_dir = POLICY_MAPS_DIR / shaping.name
    maps_dir.mkdir(parents=True, exist_ok=True)
    for old in maps_dir.glob("episode_*.png"):
        old.unlink()
    snapshot_every = n_episodes // SNAPSHOTS
    save_policy_map(agent, shaping.name, 0, maps_dir)

    run = TrainingRun(shaping, agent, [], [], [], [], [])
    for episode in tqdm(range(1, n_episodes + 1), disable=not show_progress):
        run.epsilons.append(agent.epsilon)
        train_episode(env, agent, shaping)
        agent.decay_epsilon()
        if episode % snapshot_every == 0:
            save_policy_map(agent, shaping.name, episode, maps_dir)
        if episode % eval_every == 0:
            mean_return, success_rate = evaluate(agent, eval_episodes)
            run.eval_points.append(episode)
            run.eval_returns.append(mean_return)
            run.eval_success.append(success_rate)
            # tqdm.write prints above the progress bar instead of breaking it
            tqdm.write(
                f"{shaping.name}  episode {episode:7d}  epsilon {agent.epsilon:.2f}  "
                f"greedy return {mean_return:7.1f}  success {success_rate:.0%}"
            )
    env.close()
    run.training_returns = [float(r) for r in env.return_queue]
    print(f"Saved policy maps to {maps_dir}")
    return run
