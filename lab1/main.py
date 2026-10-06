import gymnasium as gym

# Mountain Car: an underpowered car must rock back and forth to reach the flag on the right hill
env = gym.make("MountainCar-v0", render_mode="human")

observation, info = env.reset()
# observation: [car_position, car_velocity]
# position in [-1.2, 0.6], velocity in [-0.07, 0.07]
print(f"Starting observation: {observation}")

episode_over = False
total_reward = 0

while not episode_over:
    # Actions: 0 = accelerate left, 1 = don't accelerate, 2 = accelerate right
    action = env.action_space.sample()

    observation, reward, terminated, truncated, info = env.step(action)
    # reward: -1 for every step until the goal is reached
    # terminated: True when position >= 0.5 (reached the flag)
    # truncated: True after 200 steps

    total_reward += reward
    episode_over = terminated or truncated

print(f"Episode finished! Total reward: {total_reward}")
env.close()
