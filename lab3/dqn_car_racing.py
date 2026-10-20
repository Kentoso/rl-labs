"""Deep Q-Learning on CarRacing-v3 (Box2D).

Environment: https://gymnasium.farama.org/environments/box2d/car_racing/

Observations are 96x96x3 RGB frames. We convert to grayscale, resize to
84x84 and stack 4 consecutive frames so the network can perceive motion.
The discrete action space has 5 actions: do-nothing, steer-right,
steer-left, gas, brake.
"""

from collections import deque

import numpy as np
import gymnasium as gym
from gymnasium.wrappers import (
    GrayscaleObservation,
    ResizeObservation,
    FrameStackObservation,
    RecordVideo,
)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
torch.backends.cudnn.benchmark = True  # autotune convs for fixed input size

FRAME_STACK = 4
FRAME_SIZE = 84
# Separate from the baseline's checkpoint_CarRacing-v3_best.pth so training never overwrites it
CHECKPOINT = "checkpoint_CarRacing-v3_lab3.pth"


def make_env(render_mode=None):
    """CarRacing with grayscale, resize and frame-stacking wrappers."""
    env = gym.make("CarRacing-v3", continuous=False, render_mode=render_mode)
    env = GrayscaleObservation(env)
    env = ResizeObservation(env, (FRAME_SIZE, FRAME_SIZE))
    env = FrameStackObservation(env, FRAME_STACK)
    return env


def make_vec_env(num_envs):
    """Several CarRacing instances stepped in parallel subprocesses."""
    return gym.vector.AsyncVectorEnv([make_env for _ in range(num_envs)])


class QNetwork(nn.Module):
    """CNN mapping a stack of frames to action values (Nature DQN)."""

    def __init__(self, in_channels=FRAME_STACK, a_size=5):
        super().__init__()
        self.a_size = a_size

        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, a_size)

    def forward(self, x):
        x = x / 255.0
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def act(self, state, epsilon=0.0):
        if np.random.rand() <= epsilon:
            return np.random.randint(self.a_size)
        state = torch.tensor(np.asarray(state), dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.forward(state)
        return torch.argmax(q_values, dim=1).item()

    def act_batch(self, states, epsilon=0.0):
        """Epsilon-greedy actions for a batch of states (one per env)."""
        states = torch.tensor(np.asarray(states), dtype=torch.float32, device=device)
        with torch.no_grad():
            actions = self.forward(states).argmax(dim=1).cpu().numpy()
        explore = np.random.rand(len(actions)) < epsilon
        actions[explore] = np.random.randint(self.a_size, size=int(explore.sum()))
        return actions


class ReplayBuffer:
    """Preallocated ring buffer kept on the training device. Frames stay
    uint8 to cut memory 4x; the network casts to float on device. With
    unified memory (Apple silicon) the buffer is plain RAM, so sampling
    indexes it in place instead of copying a batch per update."""

    def __init__(self, capacity, obs_shape):
        self.capacity = capacity
        self.states = torch.zeros((capacity, *obs_shape), dtype=torch.uint8, device=device)
        self.next_states = torch.zeros((capacity, *obs_shape), dtype=torch.uint8, device=device)
        self.actions = torch.zeros(capacity, dtype=torch.int64, device=device)
        self.rewards = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.dones = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.pos = 0
        self.size = 0

    def push(self, states, actions, rewards, next_states, dones):
        """Store a batch of transitions (one per env) with one copy each."""
        n = len(actions)
        if n == 0:
            return
        idx = torch.arange(self.pos, self.pos + n, device=device) % self.capacity
        self.states[idx] = torch.from_numpy(np.ascontiguousarray(states, dtype=np.uint8)).to(device)
        self.next_states[idx] = torch.from_numpy(np.ascontiguousarray(next_states, dtype=np.uint8)).to(device)
        self.actions[idx] = torch.from_numpy(np.ascontiguousarray(actions, dtype=np.int64)).to(device)
        self.rewards[idx] = torch.from_numpy(np.ascontiguousarray(rewards, dtype=np.float32)).to(device)
        self.dones[idx] = torch.from_numpy(np.ascontiguousarray(dones, dtype=np.float32)).to(device)
        self.pos = (self.pos + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, batch_size):
        idx = torch.randint(0, self.size, (batch_size,), device=device)
        states = self.states[idx]
        next_states = self.next_states[idx]
        actions = self.actions[idx].unsqueeze(1)
        rewards = self.rewards[idx].unsqueeze(1)
        dones = self.dones[idx].unsqueeze(1)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return self.size


def clip_grad_norm(params, max_norm):
    """Same result as nn.utils.clip_grad_norm_, but one norm over all grads
    concatenated: MPS has no foreach kernels, so the builtin launches ~25
    tiny ops per call (2.7 ms vs 1.6 ms here)."""
    grads = [p.grad for p in params if p.grad is not None]
    norm = torch.cat([g.reshape(-1) for g in grads]).norm()
    torch._foreach_mul_(grads, (max_norm / (norm + 1e-6)).clamp(max=1.0))
    return norm


class DQNAgent:
    def __init__(self, action_dim, obs_shape, lr=1e-4, gamma=0.99, epsilon=1.0,
                 epsilon_min=0.05, epsilon_decay=0.995, buffer_size=50000,
                 batch_size=128, target_update=1000, double_dqn=True,
                 reward_scale=0.1):
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self.double_dqn = double_dqn
        self.reward_scale = reward_scale

        self.model = QNetwork(a_size=action_dim).to(device)
        self.target_model = QNetwork(a_size=action_dim).to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        # fused: one kernel for the whole update (0.99 -> 0.50 ms on MPS)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, fused=True)
        self.memory = ReplayBuffer(buffer_size, obs_shape)
        self.scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
        self.learn_steps = 0
        self.max_grad_norm = 10.0

        # Training diagnostics, accumulated on the device (no per-update sync)
        # and read by pop_stats(). |TD| edges are in scaled reward units; the
        # smooth_l1 kink is at 1, so ">= 1" is the share in Huber's linear part.
        self.td_edges = torch.tensor([0.1, 1.0, 10.0], device=device)
        self._reset_stats()

    def _reset_stats(self):
        self.stat_updates = 0
        self.td_at_least = torch.zeros(len(self.td_edges), device=device)
        self.n_clipped = torch.zeros((), device=device)
        self.grad_norm_sum = torch.zeros((), device=device)
        self.grad_norm_max = torch.zeros((), device=device)

    def pop_stats(self):
        """Summary since the last call, as a printable line; resets counters."""
        n = self.stat_updates
        if n == 0:
            return "  Stats: no updates yet"
        td = (self.td_at_least / (n * self.batch_size) * 100).tolist()
        line = (f"  Stats over {n} updates | |TD| >=0.1: {td[0]:.1f}%  >=1: {td[1]:.1f}%"
                f"  >=10: {td[2]:.2f}% | grad norm mean {self.grad_norm_sum.item() / n:.2f}"
                f"  max {self.grad_norm_max.item():.2f}"
                f"  clipped {self.n_clipped.item() / n * 100:.2f}%")
        self._reset_stats()
        return line

    def remember(self, states, actions, rewards, next_states, dones):
        # Scale only what the Q-net learns from; episode scores stay in env units
        rewards = np.asarray(rewards, dtype=np.float32) * self.reward_scale
        self.memory.push(states, actions, rewards, next_states, dones)

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        with torch.autocast(device.type, enabled=device.type == "cuda"):
            q_values = self.model(states).gather(1, actions)
            with torch.no_grad():
                if self.double_dqn:
                    # Double DQN: online net picks the next action, target net scores it
                    next_actions = self.model(next_states).argmax(1, keepdim=True)
                    next_q = self.target_model(next_states).gather(1, next_actions)
                else:
                    next_q = self.target_model(next_states).max(1, keepdim=True)[0]
                target = rewards + self.gamma * next_q * (1 - dones)
            loss = F.smooth_l1_loss(q_values, target)

        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        grad_norm = clip_grad_norm(self.model.parameters(), self.max_grad_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        td = (q_values - target).detach().abs().float()
        self.td_at_least += (td >= self.td_edges).sum(0)
        self.n_clipped += grad_norm > self.max_grad_norm
        self.grad_norm_sum += grad_norm
        self.grad_norm_max = torch.maximum(self.grad_norm_max, grad_norm)
        self.stat_updates += 1

        self.learn_steps += 1
        if self.learn_steps % self.target_update == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def train(num_envs=8, n_episodes=1000, updates_per_step=2, checkpoint=CHECKPOINT,
          stats_every=10):
    """Train with several CarRacing envs stepped in parallel.

    Transitions from every env feed a single shared replay buffer and one
    Q-network. Gymnasium's vector API auto-resets a sub-env on the step
    after it finishes, so we skip storing that bridging transition.

    Env stepping (CPU subprocesses) overlaps with the gradient updates
    (GPU): actions are sent with step_async, the updates run while the
    envs compute, and step_wait collects the results.
    """
    env = make_vec_env(num_envs)
    agent = DQNAgent(env.single_action_space.n, env.single_observation_space.shape)

    scores = []
    scores_deque = deque(maxlen=100)
    best_reward = -np.inf

    states, _ = env.reset()
    prev_done = np.zeros(num_envs, dtype=bool)
    episode_rewards = np.zeros(num_envs)
    completed = 0

    while completed < n_episodes:
        actions = agent.model.act_batch(states, epsilon=agent.epsilon)
        env.step_async(actions)
        for _ in range(updates_per_step):
            agent.replay()
        next_states, rewards, terminations, truncations, _ = env.step_wait()
        dones = np.logical_or(terminations, truncations)

        # autoreset step: next_states[i] is a fresh episode start, don't store it
        keep = ~prev_done
        agent.remember(states[keep], actions[keep], rewards[keep], next_states[keep], dones[keep])

        for i in range(num_envs):
            if prev_done[i]:
                continue
            episode_rewards[i] += rewards[i]
            if dones[i]:
                score = episode_rewards[i]
                scores.append(score)
                scores_deque.append(score)
                episode_rewards[i] = 0
                completed += 1
                agent.decay_epsilon()
                avg = np.mean(scores_deque)
                print(f"Episode {completed}\tScore {score:.1f}\tAvg {avg:.1f}\tEps {agent.epsilon:.3f}")
                if avg > best_reward and len(scores_deque) >= 10:
                    best_reward = avg
                    torch.save(agent.model.state_dict(), checkpoint)
                    print(f"  Saved checkpoint (avg {avg:.1f}) -> {checkpoint}")
                if completed % stats_every == 0:
                    print(agent.pop_stats())

        states = next_states
        prev_done = dones

    env.close()
    return scores


def test(env, policy, num_episodes=5):
    total_reward = 0
    for _ in range(num_episodes):
        state, _ = env.reset()
        done = False
        for _ in range(10000):  # max steps per episode
            action = policy.act(state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
    avg = total_reward / num_episodes
    print(f"Total Reward: {avg:.1f}")
    return avg


def render(checkpoint=CHECKPOINT, num_episodes=3,
           video_folder="videos"):
    """Record the trained agent playing to MP4 files (works headless)."""
    env = make_env(render_mode="rgb_array")
    env = RecordVideo(
        env,
        video_folder=video_folder,
        name_prefix="car_racing",
        episode_trigger=lambda e: True,
    )
    policy = QNetwork(a_size=env.action_space.n).to(device)
    policy.load_state_dict(torch.load(checkpoint, map_location=device))
    policy.eval()
    test(env, policy, num_episodes=num_episodes)
    env.close()
    print(f"Saved videos to {video_folder}/")


if __name__ == "__main__":
    print("device:", device)
    train(num_envs=8)
    render()
