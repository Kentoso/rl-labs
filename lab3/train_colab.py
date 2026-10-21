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
# Trains `dqn_car_racing.py` from the repo on a Colab GPU (L4 or better) and saves every
# training artifact to a run directory:
#
# | File | Contents |
# |---|---|
# | `config.json` | full `Config` + device, GPU, CPU count, library versions, git commit |
# | `episodes.csv` | one row per training episode: score, length, end reason, Q estimate vs discounted return, action counts, ε |
# | `updates.csv` | every `log_every_updates`: loss, Q, \|TD\| distribution, grad norms, throughput, time per loop phase |
# | `evals.csv` | one row per greedy eval episode on fixed tracks |
# | `summary.json` | best eval so far, totals (rewritten at every eval) |
# | `checkpoints/` | `best.pth` (best eval mean), `best_train_avg.pth`, `last.pth`, `ep<N>.pth` at each eval |
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
import dqn_car_racing as dqn

print("device:", dqn.device, "| CPUs:", os.cpu_count())
if dqn.device.type == "cuda":
    import torch

    print(torch.cuda.get_device_name())

# %% [markdown]
# ## Config
#
# Envs are stepped in CPU subprocesses. `num_envs` leaves 2 CPUs free for the main
# process, which launches the GPU work (run 1 used all 12 and was update-bound).
# `replay_ratio` is gradient updates per stored transition; the updates per loop
# iteration are derived from it, so runs with different `num_envs` stay comparable.
# Watch `ms_env_wait` vs `ms_update` in `updates.csv` to see whether the loop waits
# for the envs (CPU) or the updates (GPU).
#
# The budget is in env steps (frames). `CONFIGS` holds one config per run, and they
# run one after another. Current screen: action repeat 4 vs 1 at 100k frames, both
# with a dueling head and a 200k buffer. With repeat 4, `replay_ratio=0.5` per
# transition gives the same updates per frame as repeat 1 at 0.125, so both runs get
# equal compute.

# %% tags=["parameters"]
base = dict(
    out_dir=OUT_DIR,
    seed=0,
    num_envs=max(os.cpu_count() - 2, 1),
    n_episodes=None,
    total_env_steps=100_000,
    no_progress_steps=100,  # None = full 1000-frame training episodes
    bootstrap_truncation=True,  # False = timeouts stored as terminal (baseline)
    lr=1e-4,
    gamma=0.99,  # per frame; with action repeat k the target uses gamma ** k
    batch_size=256,
    buffer_size=200_000,  # ~11 GB of GPU memory (uint8 state + next_state)
    target_update=500,  # gradient updates
    double_dqn=True,
    dueling=True,
    reward_scale=0.1,
    max_grad_norm=10.0,  # None disables clipping
    epsilon_start=1.0,
    epsilon_min=0.05,
    epsilon_decay_steps=50_000,  # None = per-episode decay by epsilon_decay
    epsilon_decay=0.995,
    log_every_updates=500,
    eval_every_steps=25_000,
    eval_episodes=10,
    eval_seed=10_000,
)
CONFIGS = [
    dqn.Config(**base, run_name="duel_ar4", action_repeat=4, replay_ratio=0.5),
    dqn.Config(**base, run_name="duel_ar1", action_repeat=1, replay_ratio=0.125),
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
