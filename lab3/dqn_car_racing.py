"""Deep Q-Learning on CarRacing-v3 (Box2D).

Environment: https://gymnasium.farama.org/environments/box2d/car_racing/

Observations are 96x96x3 RGB frames. We convert to grayscale, resize to
84x84 and stack 4 consecutive frames so the network can perceive motion.
The discrete action space has 5 actions: do-nothing, steer-right,
steer-left, gas, brake.
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

    def act_batch(self, states, epsilon=0.0, return_q=False):
        """Epsilon-greedy actions for a batch of states (one per env).
        With return_q, also returns max_a Q(s, a) per state."""
        # uint8 to the device, cast there: 4x less to copy than float32
        states = torch.from_numpy(np.ascontiguousarray(states)).to(device).float()
        with torch.no_grad():
            max_q, greedy = self.forward(states).max(dim=1)
            max_q, greedy = torch.stack([max_q.float(), greedy.float()]).cpu().numpy()
        actions = greedy.astype(np.int64)
        explore = np.random.rand(len(actions)) < epsilon
        actions[explore] = np.random.randint(self.a_size, size=int(explore.sum()))
        return (actions, max_q) if return_q else actions


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




def clip_grad_norm(params, max_norm=None):
    """Global gradient norm; with max_norm, also clips like
    nn.utils.clip_grad_norm_. One norm over all grads concatenated: MPS
    has no foreach kernels, so the builtin launches ~25 tiny ops per call
    (2.7 ms vs 1.6 ms here)."""
    grads = [p.grad for p in params if p.grad is not None]
    norm = torch.cat([g.reshape(-1) for g in grads]).norm()
    if max_norm is not None:
        torch._foreach_mul_(grads, (max_norm / (norm + 1e-6)).clamp(max=1.0))
    return norm


@dataclass
class Config:
    """Everything that defines a training run; saved to <run_dir>/config.json."""

    run_name: str = "dqn"
    out_dir: str = "runs"
    seed: int = 0
    # data collection
    num_envs: int = 8
    n_episodes: int = 1000
    # gradient updates per stored transition; the updates per loop iteration
    # follow from it (0.25 = the baseline's 2 updates per step with 8 envs), so
    # changing num_envs doesn't change how often each transition is replayed
    replay_ratio: float = 0.25
    # DQN
    lr: float = 1e-4
    gamma: float = 0.99
    batch_size: int = 128
    buffer_size: int = 50_000
    target_update: int = 1000
    double_dqn: bool = True
    reward_scale: float = 0.1
    max_grad_norm: float | None = 10.0  # None disables clipping
    # exploration, decayed once per finished episode
    epsilon_start: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.995
    # logging and evaluation
    log_every_updates: int = 500
    grad_norm_every: int = 50  # without clipping, measure the norm every N updates
    eval_every: int = 50  # episodes between greedy evaluations
    eval_episodes: int = 10
    eval_seed: int = 10_000  # eval tracks: eval_seed .. eval_seed + eval_episodes - 1


class DQNAgent:
    STAT_KEYS = ["loss", "q_taken", "target", "td_abs", "td_ge_0.1", "td_ge_1", "td_ge_10",
                 "grad_norm_sum", "grad_norm_max", "grad_norm_n", "clipped", "nonfinite"]

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
        self.scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
        self.learn_steps = 0

        # Training diagnostics, summed on the device (no per-update sync) and
        # read by pop_stats(). |TD| edges are in scaled reward units; the
        # smooth_l1 kink is at 1, so "td_ge_1" is the share in Huber's linear part.
        self.td_edges = torch.tensor([0.1, 1.0, 10.0], device=device)
        self._reset_stats()

    def _reset_stats(self):
        self.stat_updates = 0
        self.stats = torch.zeros(len(self.STAT_KEYS), device=device)

    def pop_stats(self):
        """Per-update averages since the last call (Q, target and TD in scaled
        reward units); resets the counters."""
        n = self.stat_updates
        vals = dict(zip(self.STAT_KEYS, self.stats.tolist()))
        self._reset_stats()
        if n == 0:
            return {}
        n_norm = max(vals["grad_norm_n"], 1)
        out = {k: vals[k] / n for k in ["loss", "q_taken", "target", "td_abs",
                                          "td_ge_0.1", "td_ge_1", "td_ge_10"]}
        out.update(
            updates=n,
            grad_norm_mean=vals["grad_norm_sum"] / n_norm,
            grad_norm_max=vals["grad_norm_max"],
            grad_norm_measured=int(vals["grad_norm_n"]),
            clipped_frac=vals["clipped"] / n_norm,
            nonfinite_grads=int(vals["nonfinite"]),
        )
        return out

    def remember(self, states, actions, rewards, next_states, dones):
        # Scale only what the Q-net learns from; episode scores stay in env units
        rewards = np.asarray(rewards, dtype=np.float32) * self.cfg.reward_scale
        self.memory.push(states, actions, rewards, next_states, dones)

    def replay(self):
        cfg = self.cfg
        if len(self.memory) < cfg.batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(cfg.batch_size)

        with torch.autocast(device.type, enabled=device.type == "cuda"):
            q_values = self.model(states).gather(1, actions)
            with torch.no_grad():
                if cfg.double_dqn:
                    # Double DQN: online net picks the next action, target net scores it
                    next_actions = self.model(next_states).argmax(1, keepdim=True)
                    next_q = self.target_model(next_states).gather(1, next_actions)
                else:
                    next_q = self.target_model(next_states).max(1, keepdim=True)[0]
                target = rewards + cfg.gamma * next_q * (1 - dones)
            loss = F.smooth_l1_loss(q_values, target)

        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        grad_norm = None
        if cfg.max_grad_norm is not None:
            grad_norm = clip_grad_norm(self.model.parameters(), cfg.max_grad_norm)
        elif self.learn_steps % cfg.grad_norm_every == 0:
            grad_norm = clip_grad_norm(self.model.parameters())
        self.scaler.step(self.optimizer)
        self.scaler.update()

        with torch.no_grad():
            td = (q_values.float() - target.float()).abs()
            parts = [loss.float(), q_values.float().mean(), target.float().mean(), td.mean(),
                     *(td >= self.td_edges).float().mean(0)]
            if grad_norm is None:
                parts += [torch.zeros((), device=device)] * 5
            else:
                # with fp16 + GradScaler (CUDA) a step can have inf grads; the
                # scaler skips it, and it must not poison the norm stats
                finite = torch.isfinite(grad_norm)
                gn = torch.where(finite, grad_norm, 0.0)
                clipped = finite & (grad_norm > (cfg.max_grad_norm or float("inf")))
                parts += [gn, gn, finite.float(), clipped.float(), (~finite).float()]
            new = torch.stack(parts)
            max_idx = self.STAT_KEYS.index("grad_norm_max")
            self.stats[max_idx] = torch.maximum(self.stats[max_idx], new[max_idx])
            new[max_idx] = 0
            self.stats += new
        self.stat_updates += 1

        self.learn_steps += 1
        if self.learn_steps % cfg.target_update == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.cfg.epsilon_min, self.epsilon * self.cfg.epsilon_decay)


class EpisodeTracker:
    """Per-env running statistics of the current episode."""

    def __init__(self, n, gamma, reward_scale):
        self.gamma = gamma
        self.reward_scale = reward_scale
        self.n_actions = len(ACTION_NAMES)
        self.ret = np.zeros(n)
        self.disc_ret = np.zeros(n)
        self.discount = np.ones(n)
        self.length = np.zeros(n, dtype=np.int64)
        self.q_start = np.zeros(n)
        self.q_sum = np.zeros(n)
        self.actions = np.zeros((n, self.n_actions), dtype=np.int64)

    def step(self, i, action, max_q, reward):
        q_raw = max_q / self.reward_scale  # back to env reward units
        if self.length[i] == 0:
            self.q_start[i] = q_raw
        self.q_sum[i] += q_raw
        self.actions[i, action] += 1
        self.ret[i] += reward
        self.disc_ret[i] += self.discount[i] * reward
        self.discount[i] *= self.gamma
        self.length[i] += 1

    def finish(self, i, terminated, info):
        """Row describing the finished episode of env i; resets its counters."""
        lap = bool(info.get("_lap_finished", np.zeros(i + 1, bool))[i]
                   and info["lap_finished"][i])
        end = "lap" if lap else ("off_playfield" if terminated else "timeout")
        length = int(self.length[i])
        row = {
            "score": float(self.ret[i]),
            "length": length,
            "end": end,
            # Q estimate of the first state vs the discounted return that
            # followed: the gap measures over/underestimation (env units)
            "q_start": float(self.q_start[i]),
            "return_disc": float(self.disc_ret[i]),
            "q_mean": float(self.q_sum[i] / max(length, 1)),
            **{f"a_{name}": int(c) for name, c in zip(ACTION_NAMES, self.actions[i])},
        }
        self.ret[i] = self.disc_ret[i] = self.q_start[i] = self.q_sum[i] = 0.0
        self.discount[i] = 1.0
        self.length[i] = 0
        self.actions[i] = 0
        return row


def evaluate(model, eval_env, seeds, gamma, reward_scale):
    """Greedy episodes on fixed tracks, one per sub-env, run in parallel.

    Same policy as test_car_racing.py (argmax every frame), but on seeded
    tracks so that evaluations are comparable across training and runs.
    """
    n = len(seeds)
    states, _ = eval_env.reset(seed=[int(s) for s in seeds])
    tracker = EpisodeTracker(n, gamma, reward_scale)
    active = np.ones(n, dtype=bool)
    rows = [None] * n
    was_training = model.training
    model.eval()
    while active.any():
        actions, max_q = model.act_batch(states, return_q=True)
        states, rewards, terminations, truncations, info = eval_env.step(actions)
        for i in np.flatnonzero(active):
            tracker.step(i, actions[i], max_q[i], rewards[i])
            if terminations[i] or truncations[i]:
                rows[i] = {"seed": int(seeds[i]), **tracker.finish(i, terminations[i], info)}
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
      episodes.csv  one row per training episode (score, length, end reason,
                    Q estimate vs discounted return, action counts, epsilon)
      updates.csv   every log_every_updates: loss, Q, |TD| distribution,
                    grad norms, throughput and where the time goes
      evals.csv     one row per greedy eval episode on fixed tracks
      checkpoints/  best.pth (best eval mean), best_train_avg.pth (old
                    criterion), last.pth, ep<N>.pth at every eval
    """
    cfg = cfg or Config()
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    log = RunLog(cfg)
    log.save_json("config.json", {"config": asdict(cfg), "env": environment_info()})
    print(f"Run dir: {log.dir}  (device {device}, {cfg.num_envs} envs, "
          f"{cfg.replay_ratio * cfg.num_envs:g} updates per step)")

    env = make_vec_env(cfg.num_envs)
    eval_env = make_vec_env(cfg.eval_episodes)
    eval_seeds = np.arange(cfg.eval_seed, cfg.eval_seed + cfg.eval_episodes)
    agent = DQNAgent(env.single_action_space.n, env.single_observation_space.shape, cfg)
    tracker = EpisodeTracker(cfg.num_envs, cfg.gamma, cfg.reward_scale)

    scores_deque = deque(maxlen=100)
    best_train_avg = -np.inf
    best_eval = {"mean": -np.inf, "episode": None}
    summary = {}

    states, _ = env.reset(seed=cfg.seed)
    prev_done = np.zeros(cfg.num_envs, dtype=bool)
    completed = 0
    env_steps = 0
    update_credit = 0.0  # fractional updates carried over between iterations
    n_evals = 0
    last_eval_at = None
    t_start = time.time()

    # throughput accounting since the last updates.csv row
    last_log = {"t": t_start, "env_steps": 0, "learn_steps": 0, "loops": 0}
    timing = {"act": 0.0, "update": 0.0, "env_wait": 0.0}

    def run_eval():
        nonlocal n_evals, last_eval_at
        last_eval_at = completed
        t0 = time.time()
        rows = evaluate(agent.model, eval_env, eval_seeds, cfg.gamma, cfg.reward_scale)
        n_evals += 1
        for r in rows:
            log.write("evals", {"eval": n_evals, "episode": completed,
                                "learn_steps": agent.learn_steps, **r})
        scores = np.array([r["score"] for r in rows])
        laps = sum(r["end"] == "lap" for r in rows)
        log.save_model(agent.model, f"ep{completed:05d}.pth")
        tag = ""
        if scores.mean() > best_eval["mean"]:
            best_eval.update(mean=float(scores.mean()), std=float(scores.std()),
                             episode=completed)
            log.save_model(agent.model, "best.pth")
            tag = "  * new best"
        print(f"  Eval @ episode {completed}: mean {scores.mean():.1f} std {scores.std():.1f} "
              f"min {scores.min():.1f} max {scores.max():.1f} laps {laps}/{len(rows)} "
              f"({time.time() - t0:.0f}s){tag}")
        return scores

    while completed < cfg.n_episodes:
        t0 = time.time()
        actions, max_q = agent.model.act_batch(states, epsilon=agent.epsilon, return_q=True)
        t1 = time.time()
        # autoreset step: next_states[i] will be a fresh episode start, so env i's
        # transition isn't stored this iteration and earns no updates
        keep = ~prev_done
        update_credit += cfg.replay_ratio * keep.sum()
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

        agent.remember(states[keep], actions[keep], rewards[keep], next_states[keep], dones[keep])
        env_steps += int(keep.sum())

        for i in np.flatnonzero(keep):
            tracker.step(i, actions[i], max_q[i], rewards[i])
            if not dones[i]:
                continue
            row = tracker.finish(i, terminations[i], info)
            completed += 1
            scores_deque.append(row["score"])
            avg = float(np.mean(scores_deque))
            log.write("episodes", {
                "episode": completed, "env": int(i), "env_steps": env_steps,
                "learn_steps": agent.learn_steps, "wall_s": round(time.time() - t_start, 2),
                "epsilon": agent.epsilon, **row, "avg100": avg,
            })
            agent.decay_epsilon()
            print(f"Episode {completed}\tScore {row['score']:.1f}\tAvg {avg:.1f}\t"
                  f"Eps {agent.epsilon:.3f}\tLen {row['length']}\t{row['end']}")
            if avg > best_train_avg and len(scores_deque) >= 10:
                best_train_avg = avg
                log.save_model(agent.model, "best_train_avg.pth")

        states = next_states
        prev_done = dones

        if agent.learn_steps - last_log["learn_steps"] >= cfg.log_every_updates:
            now = time.time()
            dt = now - last_log["t"]
            loops = last_log["loops"]
            stats = agent.pop_stats()
            log.write("updates", {
                "learn_steps": agent.learn_steps, "env_steps": env_steps,
                "episodes": completed, "wall_s": round(now - t_start, 2),
                "epsilon": agent.epsilon, "buffer": len(agent.memory),
                "env_steps_per_s": (env_steps - last_log["env_steps"]) / dt,
                "updates_per_s": (agent.learn_steps - last_log["learn_steps"]) / dt,
                # ms per loop iteration; "act" includes waiting for queued GPU work
                **{f"ms_{k}": v / loops * 1000 for k, v in timing.items()},
                **stats,
            })
            last_log = {"t": now, "env_steps": env_steps,
                        "learn_steps": agent.learn_steps, "loops": 0}
            timing = dict.fromkeys(timing, 0.0)

        # several envs can finish in one step, so compare eval "slots" rather
        # than testing completed % eval_every
        if completed // cfg.eval_every > (last_eval_at or 0) // cfg.eval_every:
            run_eval()
            summary = {"episodes": completed, "env_steps": env_steps,
                       "learn_steps": agent.learn_steps,
                       "wall_min": round((time.time() - t_start) / 60, 1),
                       "best_eval": best_eval, "best_train_avg100": best_train_avg}
            log.save_json("summary.json", summary)

    if last_eval_at != completed:
        run_eval()  # final policy
    log.save_model(agent.model, "last.pth")
    summary = {"episodes": completed, "env_steps": env_steps, "learn_steps": agent.learn_steps,
               "wall_min": round((time.time() - t_start) / 60, 1),
               "best_eval": best_eval, "best_train_avg100": best_train_avg, "finished": True}
    log.save_json("summary.json", summary)
    log.close()
    env.close()
    eval_env.close()
    print(f"Done in {summary['wall_min']} min. Best eval mean {best_eval['mean']:.1f} "
          f"at episode {best_eval['episode']} -> {log.dir / 'checkpoints' / 'best.pth'}")
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


def render(checkpoint="checkpoint_CarRacing-v3_best.pth", num_episodes=3,
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
    run_dir = train(Config())
    render(run_dir / "checkpoints" / "best.pth", video_folder=str(run_dir / "videos"))
