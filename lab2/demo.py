import argparse
import pickle
from pathlib import Path

import gymnasium as gym
import pygame

from car_grid import Action, CarGrid

POLICIES_DIR = Path(__file__).with_name("policies")


def main() -> None:
    parser = argparse.ArgumentParser()
    # File name in policies/, e.g. vel0.8_pos0 or vel0_pos0
    parser.add_argument("--policy", default="vel0.8_pos0")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1234, 0, 1, 2])
    args = parser.parse_args()

    with open(POLICIES_DIR / f"{args.policy}.pkl", "rb") as f:
        data = pickle.load(f)
    policy = data["policy"]

    env = gym.make("MountainCar-v0", render_mode="human")
    grid = CarGrid(env, data["n_positions"], data["n_velocities"])

    for seed in args.seeds:
        observation, _ = env.reset(seed=seed)
        print(f"seed {seed}: start position {observation[0]:.3f}", end="", flush=True)
        total_reward = 0.0
        while True:
            action = Action(policy[grid.observe(observation)])
            observation, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            if terminated or truncated:
                break
        print(f", return {total_reward:.0f}, reached goal: {terminated}")

    print("Close the window to exit.")
    while not any(event.type == pygame.QUIT for event in pygame.event.get()):
        pygame.time.wait(50)
    env.close()


if __name__ == "__main__":
    main()
