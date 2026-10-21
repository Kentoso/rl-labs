"""Deep Q-Learning on CarRacing-v3 (Box2D), improved training.

Environment: https://gymnasium.farama.org/environments/box2d/car_racing/

Observations are 96x96x3 RGB frames. We convert to grayscale, resize to
84x84 and stack 4 consecutive frames so the network can perceive motion.
The discrete action space has 5 actions: do-nothing, steer-right,
steer-left, gas, brake.

Same environment, network and loss as the baseline (dqn_car_racing.py at
ff03938); see CHANGELOG.md for the evidence behind each change:
  - action repeat (training only): each decision holds its action for k
    frames (by default k = 4); the transition stores the summed reward and
    the target discounts by gamma ** frames. k can step down on a schedule.
    Evals and test_car_racing.py act every frame
  - training episodes are truncated after 100 frames without a new tile;
    truncations (that and the 1000-frame time limit) bootstrap, only real
    terminals zero the target
  - epsilon decays linearly per frame; budgets and evals count frames
  - rewards are scaled by 0.1 for learning (keeps TD errors in Huber's
    quadratic part); scores and logs stay in env units
  - Double DQN target (van Hasselt et al. 2016, arXiv:1509.06461)
  - replay buffer on the device, batched acting, env stepping overlapped
    with the updates, fused Adam, MPS support
  - greedy evals on fixed tracks pick best.pth; every run writes CSV logs
"""

import csv
import json
import os
import platform
import subprocess
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

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
torch.set_float32_matmul_precision("high")  # TF32 matmuls on CUDA

FRAME_STACK = 4
FRAME_SIZE = 84
ACTION_NAMES = ["noop", "right", "left", "gas", "brake"]


def make_env(render_mode=None):
    """CarRacing with grayscale, resize and frame-stacking wrappers."""
    env = gym.make("CarRacing-v3", continuous=False, render_mode=render_mode)
    env = GrayscaleObservation(env)
    env = ResizeObservation(env, (FRAME_SIZE, FRAME_SIZE))
    env = FrameStackObservation(env, FRAME_STACK)
    return env


class EpisodeInfo(gym.Wrapper):
    """Adds diagnostics to info: tiles visited so far, tiles on the track and
    the car's speed. Read-only; observations and rewards are untouched."""

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        car_racing = self.env.unwrapped
        info["tiles"] = car_racing.tile_visited_count
        info["track_tiles"] = len(car_racing.track)
        info["speed"] = float(np.linalg.norm(car_racing.car.hull.linearVelocity))
        return obs, reward, terminated, truncated, info


class NoProgressTruncation(gym.Wrapper):
    """Truncate an episode after `patience` frames without a new track tile.

    A frame that reaches a new tile earns +1000/N - 0.1 > 0; every other frame
    earns -0.1 (or -100 when leaving the playfield), so reward > 0 marks a new
    tile. Training-only: it saves the frames a car spends stuck on the grass.
    """

    def __init__(self, env, patience):
        super().__init__(env)
        self.patience = patience
        self.since_tile = 0

    def reset(self, **kwargs):
        self.since_tile = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.since_tile = 0 if reward > 0 else self.since_tile + 1
        if self.since_tile >= self.patience and not (terminated or truncated):
            truncated = True
            info["no_progress"] = True
        return obs, reward, terminated, truncated, info


class ActionRepeat(gym.Wrapper):
    """Repeat each action for k frames and return the summed reward.

    The observation is still make_env's stack of the last 4 consecutive
    frames, so a policy trained this way can act every frame at test time.
    info["frames"] says how many frames ran (fewer than k if the episode ended).
    """

    def __init__(self, env, k):
        super().__init__(env)
        self.k = k

    def step(self, action):
        total = 0.0
        for frames in range(1, self.k + 1):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total += float(reward)
            if terminated or truncated:
                break
        info["frames"] = frames
        return obs, total, terminated, truncated, info


def action_repeat_at(schedule, env_steps):
    """k for the given frame count from ((from_frame, k), ...)."""
    return max((s for s in schedule if s[0] <= env_steps), key=lambda s: s[0])[1]


def make_train_env(no_progress_steps=None, action_repeat=1):
    env = EpisodeInfo(make_env())
    if no_progress_steps is not None:
        env = NoProgressTruncation(env, no_progress_steps)
    return ActionRepeat(env, action_repeat)


def make_vec_env(num_envs, no_progress_steps=None, action_repeat=1):
    """Several CarRacing instances stepped in parallel subprocesses."""
    return gym.vector.AsyncVectorEnv(
        [lambda: make_train_env(no_progress_steps, action_repeat) for _ in range(num_envs)])


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
        """Epsilon-greedy actions for a batch of states (one per env), plus
        max_a Q(s, a) per state for the logs."""
        # uint8 to the device, cast there: 4x less to copy than float32
        states = torch.from_numpy(np.ascontiguousarray(states)).to(device).float()
        with torch.no_grad():
            max_q, greedy = self.forward(states).max(dim=1)
            max_q, greedy = torch.stack([max_q, greedy.float()]).cpu().numpy()
        actions = greedy.astype(np.int64)
        explore = np.random.rand(len(actions)) < epsilon
        actions[explore] = np.random.randint(self.a_size, size=int(explore.sum()))
        return actions, max_q


class ReplayBuffer:
    """Preallocated ring buffer kept on the training device. Frames stay
    uint8 to cut memory 4x; the network casts to float on device. Sampling
    indexes the buffer in place instead of copying a batch per update."""

    def __init__(self, capacity, obs_shape):
        self.capacity = capacity
        self.states = torch.zeros((capacity, *obs_shape), dtype=torch.uint8, device=device)
        self.next_states = torch.zeros((capacity, *obs_shape), dtype=torch.uint8, device=device)
        self.actions = torch.zeros(capacity, dtype=torch.int64, device=device)
        self.rewards = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.dones = torch.zeros(capacity, dtype=torch.float32, device=device)
        # gamma ** frames of each transition: action repeat can change during a run
        self.discounts = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.pos = 0
        self.size = 0

    def push(self, states, actions, rewards, next_states, dones, discounts):
        """Store a batch of transitions (one per env) with one copy each."""
        n = len(actions)
        if n == 0:
            return
        idx = torch.arange(self.pos, self.pos + n, device=device) % self.capacity

        def put(buf, x, dtype):
            buf[idx] = torch.from_numpy(np.ascontiguousarray(x, dtype=dtype)).to(device)

        put(self.states, states, np.uint8)
        put(self.next_states, next_states, np.uint8)
        put(self.actions, actions, np.int64)
        put(self.rewards, rewards, np.float32)
        put(self.dones, dones, np.float32)
        put(self.discounts, discounts, np.float32)
        self.pos = (self.pos + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, batch_size):
        idx = torch.randint(0, self.size, (batch_size,), device=device)
        return (self.states[idx], self.actions[idx].unsqueeze(1),
                self.rewards[idx].unsqueeze(1), self.next_states[idx],
                self.dones[idx].unsqueeze(1), self.discounts[idx].unsqueeze(1))

    def __len__(self):
        return self.size


@dataclass
class Config:
    """Everything that defines a training run; saved to <run_dir>/config.json."""

    run_name: str = "dqn_ar"
    out_dir: str = "runs"
    seed: int = 0
    # data collection; budgets count frames
    num_envs: int = 10
    total_env_steps: int = 500_000
    # training only; evals and the test act every frame. ((from_frame, k), ...):
    # k frames per decision from that frame on, e.g. ((0, 4), (400_000, 2))
    action_repeat_schedule: tuple[tuple[int, int], ...] = ((0, 4),)
    no_progress_steps: int | None = 100  # None = full 1000-frame training episodes
    # gradient updates per frame (0.125 = 0.5 per transition at action repeat 4),
    # so the compute per frame doesn't change with the action repeat
    updates_per_frame: float = 0.125
    # DQN
    lr: float = 1e-4
    # linear lr decay from lr at lr_decay_start frames to lr_end at the end of
    # the budget (lr_end None = constant lr)
    lr_end: float | None = None
    lr_decay_start: int = 0
    gamma: float = 0.99  # per frame
    batch_size: int = 256
    buffer_size: int = 100_000  # transitions
    target_update: int = 500  # gradient updates
    reward_scale: float = 0.1  # applied to stored rewards only
    double_dqn: bool = True  # online net picks the next action, target net scores it
    # exploration: linear decay over the first epsilon_decay_steps frames
    epsilon_start: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay_steps: int = 50_000
    # logging and evaluation
    log_every_updates: int = 500
    eval_every_steps: int = 25_000
    eval_episodes: int = 30
    eval_envs: int = 10  # eval episodes run in rounds of this many
    eval_seed: int = 10_000  # eval tracks: eval_seed .. eval_seed + eval_episodes - 1


class DQNAgent:
    STAT_KEYS = ["loss", "q_taken", "target", "td_abs"]

    def __init__(self, action_dim, obs_shape, cfg):
        self.cfg = cfg
        self.action_dim = action_dim
        self.epsilon = cfg.epsilon_start

        self.model = QNetwork(a_size=action_dim).to(device)
        self.target_model = QNetwork(a_size=action_dim).to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        # fused: one kernel for the whole update (0.99 -> 0.50 ms on MPS)
        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.lr, fused=True)
        self.memory = ReplayBuffer(cfg.buffer_size, obs_shape)
        self.learn_steps = 0
        # summed on the device (no per-update sync), read by pop_stats()
        self.stats = torch.zeros(len(self.STAT_KEYS), device=device)
        self.stat_updates = 0

    def pop_stats(self):
        """Per-update averages since the last call (scaled reward units);
        resets the counters."""
        n = self.stat_updates
        out = {k: v / n for k, v in zip(self.STAT_KEYS, self.stats.tolist())} if n else {}
        self.stats.zero_()
        self.stat_updates = 0
        return out

    def remember(self, states, actions, rewards, next_states, dones, frames):
        # Scale only what the Q-net learns from; episode scores stay in env units
        rewards = np.asarray(rewards, dtype=np.float32) * self.cfg.reward_scale
        discounts = self.cfg.gamma ** np.asarray(frames, dtype=np.float64)
        self.memory.push(states, actions, rewards, next_states, dones, discounts)

    def replay(self):
        cfg = self.cfg
        if len(self.memory) < cfg.batch_size:
            return

        states, actions, rewards, next_states, dones, discounts = \
            self.memory.sample(cfg.batch_size)

        q_values = self.model(states).gather(1, actions)
        with torch.no_grad():
            if cfg.double_dqn:
                next_actions = self.model(next_states).argmax(1, keepdim=True)
                next_q = self.target_model(next_states).gather(1, next_actions)
            else:
                next_q = self.target_model(next_states).max(1, keepdim=True)[0]
            target = rewards + discounts * next_q * (1 - dones)
        loss = F.smooth_l1_loss(q_values, target)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            self.stats += torch.stack([loss, q_values.mean(), target.mean(),
                                       (q_values - target).abs().mean()])
        self.stat_updates += 1

        self.learn_steps += 1
        if self.learn_steps % cfg.target_update == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    def set_epsilon(self, env_steps):
        cfg = self.cfg
        frac = min(env_steps / cfg.epsilon_decay_steps, 1.0)
        self.epsilon = cfg.epsilon_start + frac * (cfg.epsilon_min - cfg.epsilon_start)

    def set_lr(self, env_steps):
        cfg = self.cfg
        if cfg.lr_end is None:
            return cfg.lr
        span = max(cfg.total_env_steps - cfg.lr_decay_start, 1)
        frac = min(max(env_steps - cfg.lr_decay_start, 0) / span, 1.0)
        lr = cfg.lr + frac * (cfg.lr_end - cfg.lr)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr


class EpisodeTracker:
    """Per-env running statistics of the current episode."""

    def __init__(self, n, gamma, reward_scale):
        self.gamma = gamma
        self.reward_scale = reward_scale
        self.ret = np.zeros(n)
        self.disc_ret = np.zeros(n)
        self.discount = np.ones(n)
        self.length = np.zeros(n, dtype=np.int64)  # frames
        self.decisions = np.zeros(n, dtype=np.int64)
        self.q_start = np.zeros(n)
        self.q_sum = np.zeros(n)
        self.speed_sum = np.zeros(n)
        self.actions = np.zeros((n, len(ACTION_NAMES)), dtype=np.int64)

    def step(self, i, action, max_q, reward, frames, speed):
        """One decision of env i that ran `frames` frames."""
        max_q = max_q / self.reward_scale  # back to env reward units
        if self.decisions[i] == 0:
            self.q_start[i] = max_q
        self.q_sum[i] += max_q
        self.speed_sum[i] += speed
        self.actions[i, action] += frames
        self.ret[i] += reward
        self.disc_ret[i] += self.discount[i] * reward
        self.discount[i] *= self.gamma ** frames
        self.length[i] += frames
        self.decisions[i] += 1

    def finish(self, i, terminated, info):
        """Row describing the finished episode of env i; resets its counters."""
        lap = bool(info.get("_lap_finished", np.zeros(i + 1, bool))[i]
                   and info["lap_finished"][i])
        no_progress = bool(info.get("_no_progress", np.zeros(i + 1, bool))[i])
        if lap:
            end = "lap"
        elif terminated:
            end = "off_playfield"
        else:
            end = "no_progress" if no_progress else "timeout"
        decisions = max(int(self.decisions[i]), 1)
        tiles, track_tiles = int(info["tiles"][i]), int(info["track_tiles"][i])
        row = {
            "score": float(self.ret[i]),
            "length": int(self.length[i]),  # frames
            "end": end,
            "tiles": tiles,
            "track_tiles": track_tiles,
            "tile_frac": tiles / track_tiles,
            "speed_mean": float(self.speed_sum[i] / decisions),
            # Q estimate of the first state vs the discounted return that
            # followed: the gap measures over/underestimation
            "q_start": float(self.q_start[i]),
            "return_disc": float(self.disc_ret[i]),
            "q_mean": float(self.q_sum[i] / decisions),
            # frames per action
            **{f"a_{name}": int(c) for name, c in zip(ACTION_NAMES, self.actions[i])},
        }
        self.ret[i] = self.disc_ret[i] = self.q_start[i] = self.q_sum[i] = 0.0
        self.speed_sum[i] = 0.0
        self.discount[i] = 1.0
        self.length[i] = self.decisions[i] = 0
        self.actions[i] = 0
        return row


def evaluate(model, eval_env, seeds, gamma, reward_scale):
    """Greedy episodes on fixed tracks, acting every frame like
    test_car_racing.py. Seeds run in rounds of eval_env.num_envs."""
    n = eval_env.num_envs
    rows = []
    was_training = model.training
    model.eval()
    for start in range(0, len(seeds), n):
        chunk = seeds[start:start + n]
        states, _ = eval_env.reset(seed=[int(s) for s in chunk] + [0] * (n - len(chunk)))
        tracker = EpisodeTracker(n, gamma, reward_scale)
        active = np.arange(n) < len(chunk)
        while active.any():
            actions, max_q = model.act_batch(states)
            states, rewards, terminations, truncations, info = eval_env.step(actions)
            for i in np.flatnonzero(active):
                tracker.step(i, actions[i], max_q[i], rewards[i], 1, info["speed"][i])
                if terminations[i] or truncations[i]:
                    rows.append({"seed": int(chunk[i]),
                                 **tracker.finish(i, terminations[i], info)})
                    active[i] = False
    if was_training:
        model.train()
    return rows


class RunLog:
    """Run directory: config.json, summary.json, CSV tables flushed on every
    row (a killed session keeps everything so far) and checkpoints/."""

    def __init__(self, cfg):
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.dir = Path(cfg.out_dir) / f"{stamp}_{cfg.run_name}"
        (self.dir / "checkpoints").mkdir(parents=True)
        self._files = {}
        self._writers = {}

    def write(self, table, row):
        if table not in self._writers:
            f = open(self.dir / f"{table}.csv", "w", newline="")
            self._files[table] = f
            self._writers[table] = csv.DictWriter(f, fieldnames=list(row))
            self._writers[table].writeheader()
        self._writers[table].writerow(row)
        self._files[table].flush()

    def save_json(self, name, obj):
        with open(self.dir / name, "w") as f:
            json.dump(obj, f, indent=2)

    def save_model(self, model, name):
        torch.save(model.state_dict(), self.dir / "checkpoints" / name)

    def close(self):
        for f in self._files.values():
            f.close()


def environment_info():
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                                text=True, cwd=Path(__file__).parent).stdout.strip()
    except OSError:
        commit = ""
    return {
        "script": Path(__file__).name,
        "device": str(device),
        "gpu": torch.cuda.get_device_name() if device.type == "cuda" else platform.processor(),
        "cpu_count": os.cpu_count(),
        "torch": torch.__version__,
        "gymnasium": gym.__version__,
        "python": platform.python_version(),
        "git_commit": commit,
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def train(cfg=None):
    """Train with several CarRacing envs stepped in parallel; returns the run dir.

    Transitions from every env feed a single shared replay buffer and one
    Q-network. Gymnasium's vector API auto-resets a sub-env on the step
    after it finishes, so we skip storing that bridging transition.

    Env stepping (CPU subprocesses) overlaps with the gradient updates
    (GPU): actions are sent with step_async, the updates run while the
    envs compute, and step_wait collects the results.

    Artifacts in <out_dir>/<timestamp>_<run_name>/:
      episodes.csv  one row per training episode (score, frames, end reason,
                    tile coverage, speed, Q estimate vs discounted return,
                    frames per action, epsilon)
      updates.csv   every log_every_updates: loss, Q, |TD|, throughput and
                    where the time goes
      evals.csv     one row per greedy eval episode on fixed tracks
      checkpoints/  best.pth (best eval mean), last.pth, step<N>k.pth at
                    every eval

    Training episodes may be cut short (no_progress_steps), so their scores
    and avg100 underestimate full-episode scores; evals always run full episodes.
    """
    cfg = cfg or Config()
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    log = RunLog(cfg)
    log.save_json("config.json", {"config": asdict(cfg), "env": environment_info()})
    print(f"Run dir: {log.dir}  (device {device}, {cfg.num_envs} envs, "
          f"action repeat schedule {cfg.action_repeat_schedule}, "
          f"{cfg.updates_per_frame:g} updates per frame)")

    action_repeat = action_repeat_at(cfg.action_repeat_schedule, 0)
    env = make_vec_env(cfg.num_envs, cfg.no_progress_steps, action_repeat)
    eval_env = make_vec_env(min(cfg.eval_envs, cfg.eval_episodes))
    eval_seeds = np.arange(cfg.eval_seed, cfg.eval_seed + cfg.eval_episodes)
    agent = DQNAgent(env.single_action_space.n, env.single_observation_space.shape, cfg)
    tracker = EpisodeTracker(cfg.num_envs, cfg.gamma, cfg.reward_scale)

    scores_deque = deque(maxlen=100)
    best_eval = {"mean": -np.inf, "env_steps": None}

    states, _ = env.reset(seed=cfg.seed)
    prev_done = np.zeros(cfg.num_envs, dtype=bool)
    completed = 0
    env_steps = 0
    update_credit = 0.0  # updates earned by the frames run so far, not yet done
    lr = cfg.lr
    n_evals = 0
    last_eval_at = 0
    t_start = time.time()

    # throughput accounting since the last updates.csv row
    last_log = {"t": t_start, "env_steps": 0, "learn_steps": 0, "loops": 0}
    timing = {"act": 0.0, "update": 0.0, "env_wait": 0.0}

    def summary(**extra):
        return {"episodes": completed, "env_steps": env_steps,
                "learn_steps": agent.learn_steps,
                "wall_min": round((time.time() - t_start) / 60, 1),
                "best_eval": best_eval, **extra}

    def run_eval():
        nonlocal n_evals, last_eval_at
        last_eval_at = env_steps
        t0 = time.time()
        rows = evaluate(agent.model, eval_env, eval_seeds, cfg.gamma, cfg.reward_scale)
        n_evals += 1
        for r in rows:
            log.write("evals", {"eval": n_evals, "episode": completed, "env_steps": env_steps,
                                "learn_steps": agent.learn_steps, **r})
        scores = np.array([r["score"] for r in rows])
        laps = sum(r["end"] == "lap" for r in rows)
        tiles = np.mean([r["tile_frac"] for r in rows])
        log.save_model(agent.model, f"step{env_steps // 1000:05d}k.pth")
        tag = ""
        if scores.mean() > best_eval["mean"]:
            best_eval.update(mean=float(scores.mean()), std=float(scores.std()),
                             episode=completed, env_steps=env_steps)
            log.save_model(agent.model, "best.pth")
            tag = "  * new best"
        print(f"  Eval @ {env_steps} frames (episode {completed}): "
              f"mean {scores.mean():.1f} std {scores.std():.1f} "
              f"min {scores.min():.1f} max {scores.max():.1f} laps {laps}/{len(rows)} "
              f"tiles {tiles:.1%} ({time.time() - t0:.0f}s){tag}")

    while env_steps < cfg.total_env_steps:
        agent.set_epsilon(env_steps)
        lr = agent.set_lr(env_steps)
        k = action_repeat_at(cfg.action_repeat_schedule, env_steps)
        if k != action_repeat:
            # between step_wait and the next step_async, so no step is in flight
            env.set_attr("k", k)
            print(f"  Action repeat {action_repeat} -> {k} at {env_steps} frames")
            action_repeat = k
        t0 = time.time()
        actions, max_q = agent.model.act_batch(states, epsilon=agent.epsilon)
        t1 = time.time()
        # autoreset step: next_states[i] will be a fresh episode start, so env i's
        # transition isn't stored this iteration
        keep = ~prev_done
        n_updates = int(update_credit)
        update_credit -= n_updates
        env.step_async(actions)
        for _ in range(n_updates):
            agent.replay()
        t2 = time.time()
        next_states, rewards, terminations, truncations, info = env.step_wait()
        t3 = time.time()
        timing["act"] += t1 - t0
        timing["update"] += t2 - t1
        timing["env_wait"] += t3 - t2
        last_log["loops"] += 1
        dones = np.logical_or(terminations, truncations)

        # a truncated episode didn't end in the MDP, so its target still
        # bootstraps from the next state; only real terminals zero it
        frames = info["frames"]
        agent.remember(states[keep], actions[keep], rewards[keep], next_states[keep],
                       terminations[keep], frames[keep])
        env_steps += int(frames[keep].sum())
        update_credit += cfg.updates_per_frame * frames[keep].sum()

        for i in np.flatnonzero(keep):
            tracker.step(i, actions[i], max_q[i], rewards[i], frames[i], info["speed"][i])
            if not dones[i]:
                continue
            row = tracker.finish(i, terminations[i], info)
            completed += 1
            scores_deque.append(row["score"])
            avg = float(np.mean(scores_deque))
            log.write("episodes", {
                "episode": completed, "env": int(i), "env_steps": env_steps,
                "learn_steps": agent.learn_steps, "wall_s": round(time.time() - t_start, 2),
                "epsilon": agent.epsilon, "action_repeat": action_repeat, **row, "avg100": avg,
            })
            print(f"Episode {completed}\tScore {row['score']:.1f}\tAvg {avg:.1f}\t"
                  f"Eps {agent.epsilon:.3f}\tLen {row['length']}\t"
                  f"Tiles {row['tile_frac']:.0%}\t{row['end']}")

        states = next_states
        prev_done = dones

        if agent.learn_steps - last_log["learn_steps"] >= cfg.log_every_updates:
            now = time.time()
            dt = now - last_log["t"]
            loops = last_log["loops"]
            log.write("updates", {
                "learn_steps": agent.learn_steps, "env_steps": env_steps,
                "episodes": completed, "wall_s": round(now - t_start, 2),
                "epsilon": agent.epsilon, "lr": lr, "action_repeat": action_repeat,
                "buffer": len(agent.memory),
                "env_steps_per_s": (env_steps - last_log["env_steps"]) / dt,
                "updates_per_s": (agent.learn_steps - last_log["learn_steps"]) / dt,
                # ms per loop iteration; "act" includes waiting for queued GPU work
                **{f"ms_{k}": v / loops * 1000 for k, v in timing.items()},
                **agent.pop_stats(),
            })
            last_log = {"t": now, "env_steps": env_steps,
                        "learn_steps": agent.learn_steps, "loops": 0}
            timing = dict.fromkeys(timing, 0.0)

        # env_steps grows by up to num_envs * k per iteration, so
        # compare eval "slots" rather than testing env_steps % eval_every_steps
        if env_steps // cfg.eval_every_steps > last_eval_at // cfg.eval_every_steps:
            run_eval()
            log.save_json("summary.json", summary())

    if last_eval_at != env_steps:
        run_eval()  # final policy
    log.save_model(agent.model, "last.pth")
    final = summary(finished=True)
    log.save_json("summary.json", final)
    log.close()
    env.close()
    eval_env.close()
    print(f"Done in {final['wall_min']} min. Best eval mean {best_eval['mean']:.1f} "
          f"at {best_eval['env_steps']} frames -> {log.dir / 'checkpoints' / 'best.pth'}")
    return log.dir


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
            if done:
                break
    avg = total_reward / num_episodes
    print(f"Total Reward: {avg:.1f}")
    return avg


def render(checkpoint, num_episodes=3, video_folder="videos"):
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
    run_dir = train()
    render(run_dir / "checkpoints" / "best.pth", video_folder=str(run_dir / "videos"))
