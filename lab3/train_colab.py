# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: tags,-all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
# ---

# %% [markdown]
# # Lab 3 — DQN on CarRacing-v3 (Colab training)
#
# Trains `dqn_car_racing_improved.py` (the baseline plus only the changes proven in
# `CHANGELOG.md`, with action repeat 4) on a Colab GPU and saves every training artifact
# to a run directory:
#
# | File | Contents |
# |---|---|
# | `config.json` | full `Config` + script, device, GPU, CPU count, library versions, git commit |
# | `episodes.csv` | one row per training episode: score, frames, end reason, tile coverage, speed, Q estimate vs discounted return, frames per action, ε |
# | `updates.csv` | every `log_every_updates`: loss, Q, \|TD\|, throughput, time per loop phase |
# | `evals.csv` | one row per greedy eval episode on fixed tracks (acting every frame, like the test) |
# | `summary.json` | best eval so far, totals (rewritten at every eval) |
# | `checkpoints/` | `best.pth` (best eval mean), `last.pth`, `step<N>k.pth` at each eval |
#
# Runtime → Change runtime type → **GPU (L4)**. CSVs are flushed on every row, so with
# Google Drive mounted a disconnected session keeps everything up to that point.

# %% [markdown]
# ## Setup

# %%
# !nvidia-smi -L

# %%
USE_DRIVE = True  # save runs to Google Drive so they survive the session
REPO_URL = "https://github.com/Kentoso/rl-labs.git"
BRANCH = "main"

if USE_DRIVE:
    from google.colab import drive

    drive.mount("/content/drive")
    OUT_DIR = "/content/drive/MyDrive/rl-labs-lab3/runs"
else:
    OUT_DIR = "/content/runs"

# %%
# !test -d /content/rl-labs || git clone -q -b {BRANCH} {REPO_URL} /content/rl-labs
# !git -C /content/rl-labs pull -q && git -C /content/rl-labs log --oneline -1
# %cd /content/rl-labs/lab3

# %%
# Same gymnasium as local (the env and test script must not change). Colab already has
# torch with CUDA, numpy and OpenCV.
# !pip install -q swig
# !pip install -q "gymnasium[box2d]==1.3.0" moviepy

# %%
import json
import os
import sys

sys.path.insert(0, "/content/rl-labs/lab3")
import dqn_car_racing_improved as dqn

print("device:", dqn.device, "| CPUs:", os.cpu_count())
if dqn.device.type == "cuda":
    import torch

    print(torch.cuda.get_device_name())

# %% [markdown]
# ## Config
#
# Budgets, ε, lr and evals count frames. Each training decision holds its action for k = 4
# frames (`action_repeat_schedule` can step k down, but k = 1 hurt in run 7); evals and the
# test act every frame. Targets are 3-step returns over decisions (12 frames at k = 4) with
# γ 0.995 per frame (run 8). `updates_per_frame` keeps the compute per frame the same at every k. `num_envs` leaves
# 2 CPUs free for the main process. `CONFIGS` holds one config per run, and they run one
# after another.

# %% tags=["parameters"]
base = dict(
    out_dir=OUT_DIR,
    seed=0,
    num_envs=max(os.cpu_count() - 2, 1),
    total_env_steps=500_000,
    # (from_frame, k): k frames per decision from that frame on
    action_repeat_schedule=((0, 4),),
    no_progress_steps=100,  # None = full 1000-frame training episodes
    updates_per_frame=0.125,
    lr=1e-4,
    lr_end=None,  # linear decay to this at the end of the budget; None = constant lr
    lr_decay_start=0,
    gamma=0.995,  # per frame; each transition is discounted by gamma ** its frames
    n_step=3,  # decisions summed per target before bootstrapping
    batch_size=256,
    buffer_size=100_000,  # transitions (~5.6 GB on the GPU)
    target_update=2000,  # gradient updates (= 16k frames at 0.125 updates per frame)
    reward_scale=0.1,
    double_dqn=True,
    epsilon_start=1.0,
    epsilon_min=0.05,
    epsilon_decay_steps=50_000,
    log_every_updates=500,
    eval_every_steps=25_000,
    eval_episodes=30,
    eval_envs=10,
    eval_seed=10_000,
)
CONFIGS = [
    dqn.Config(**base, run_name="g995_n3_500k"),
]
CONFIGS

# %% [markdown]
# ## Train

# %%
run_dirs = []
for cfg in CONFIGS:
    run_dirs.append(dqn.train(cfg))

# %%
for run_dir in run_dirs:
    print(run_dir.name, json.dumps(json.load(open(run_dir / "summary.json")), indent=2))

# %% [markdown]
# ## Test with the unchanged test script (optional, slow)
#
# 50 random tracks, same as the baseline numbers (mean 840.6, std 103.2). Evals during
# training already act every frame like the test does.

# %%
for run_dir in run_dirs:
    print(run_dir.name)
    # !python test_car_racing.py "{run_dir}/checkpoints/best.pth" 50

# %% [markdown]
# ## Download (when not using Drive)

# %%
if not USE_DRIVE:
    import shutil

    from google.colab import files

    for run_dir in run_dirs:
        archive = shutil.make_archive(str(run_dir), "zip", run_dir)
        files.download(archive)
