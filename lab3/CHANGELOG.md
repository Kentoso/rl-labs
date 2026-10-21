# Lab 3 — change log

Task: beat the baseline DQN on `CarRacing-v3` (discrete actions). The environment
settings (`make_env`) and `test_car_racing.py` must stay unchanged.

Baseline (`checkpoint_CarRacing-v3_best.pth`, 1000 episodes, reference numbers):

| Episodes | Mean  | Std   | Min   | Max   |
|----------|-------|-------|-------|-------|
| 50       | 840.6 | 103.2 | 470.4 | 912.7 |

Evaluate with: `uv run python test_car_racing.py <checkpoint> 50` (from `lab3/`).

---

## 2026-10-20 — Baseline setup (`ff03938`)

**What:** Copied `dqn_car_racing.py`, `test_car_racing.py`, `render_car_racing.py` and the
baseline checkpoint from `PavloZakala/ReinforcementLearning` (main), unmodified.
Added `torch`, `moviepy` (for `RecordVideo`) and `opencv-python-headless` (for
`ResizeObservation`) to `pyproject.toml`; ignored `lab3/videos/`.

**Why:** Reproducible starting point to measure improvements against.

**Notes:** The `objc ... SDL ... implemented in both` warnings come from pygame and
OpenCV each bundling their own SDL (all OpenCV wheels do, including 4.x and headless).
Harmless because we never open a pygame window.

## 2026-10-20 — MPS device support (`42146f0`)

**What:** Device selection is now `cuda` → `mps` → `cpu` (was `cuda` → `cpu`).

**Why:** Train and test on the Apple GPU. `test_car_racing.py` imports `device` from
`dqn_car_racing.py`, so it uses MPS without being edited. Sanity check: 2 test
episodes scored 851.8 / 886.9 on MPS.

## 2026-10-20 — Replay buffer on the device (`4c2881e`)

**What:** `ReplayBuffer` stores its arrays as torch tensors on `device` (still uint8 for
frames) and samples with `torch.randint` + indexing on the device. `push` now takes a
batch: the training loop stores all 8 envs' transitions in one call (autoreset steps
masked out with `keep = ~prev_done`) instead of 8 separate pushes.

**Why:** On Apple silicon the GPU uses ordinary RAM, so a device-side buffer costs no
extra memory, and sampling no longer copies a 128-transition batch (~5.6 MB) from numpy
each update. Batching the push keeps the per-step copy to one transfer per array.

**Measured** (one training step = push 8 transitions + 2 updates, 50k buffer, MPS):

| Buffer                 | ms/step |
|------------------------|---------|
| numpy, per-env pushes  | 25.3    |
| device, batched push   | 22.3    |

~12% faster per step. For comparison, stepping the 8-env `AsyncVectorEnv` takes
~8.1 ms, so the 2 gradient updates (~20 ms), not the envs, are the main per-step cost.

**Checks:** buffer contents match after ring wraparound; `sample` returns the same
shapes/dtypes as before; 2-episode smoke run of `train()` completes.

## 2026-10-20 — Double DQN target (`4c2881e`)

**What:** `DQNAgent(double_dqn=True)` (new default). The bootstrap action is chosen by
the online net and evaluated by the target net:
`y = r + γ · Q_target(s', argmax_a Q_online(s', a))`. `double_dqn=False` restores the
vanilla target `r + γ · max_a Q_target(s', a)` for ablations.

**Why:** The vanilla `max` over noisy Q-estimates is biased upward (E[max] ≥ max E), and
bootstrapping compounds the bias. Overconfident action values are a plausible cause of
the baseline's occasional bad episodes (min 470, std 103). Still off-policy Q-learning
(greedy target), so experience replay stays valid.
Source: van Hasselt, Guez, Silver, *Deep RL with Double Q-learning*, AAAI 2016,
https://arxiv.org/abs/1509.06461

**Measured:** one extra forward pass, so `replay()` goes from 8.9 to 9.9 ms (~2 ms per
training step with 2 updates). With online ≠ target, the Double DQN target is ≤ the
vanilla target on every sample, as expected. Effect on test score: not trained yet.

## 2026-10-20 — Overlap env stepping with updates; no validation videos (uncommitted)

**What:**
- `train()` sends actions with `env.step_async`, runs the gradient updates, then collects
  results with `env.step_wait`. The 8 env subprocesses (CPU) and the updates (GPU) now
  run at the same time instead of one after the other.
- Removed the validation episode + MP4 recorded every 50 episodes (`evaluate()` and
  `validate_every`). `render()` after training is unchanged.

**Why:** Env stepping (~8 ms) was idle time for the GPU, and the updates were idle time
for the envs. Validation videos stalled training 20 times per run for a single
ε-free episode that wasn't used for anything (checkpoint choice uses the training
average).

**Side effect:** updates in a step now run before that step's transition is stored,
i.e. they see the buffer one step older. Irrelevant with a 50k buffer.

**Measured** (real loop, 8 envs, 1000 steps, MPS):

| Loop                       | ms/step |
|----------------------------|---------|
| sequential (`env.step`)    | 34.3    |
| overlapped (async)         | 28.8    |

16% faster; hides ~5.5 of the ~8 ms env step. Plus no video stalls.

## 2026-10-20 — Reward scale /10 for learning (uncommitted)

**What:** `DQNAgent(reward_scale=0.1)` multiplies rewards before they go into the replay
buffer (tile +3.4 → +0.34, frame −0.1 → −0.01, off-playfield −100 → −10). Episode scores,
logs and checkpoint selection still use raw env rewards.

**Why:** Scaling all rewards by c > 0 scales all Q-values by c, so the greedy policy is
unchanged. It does change two things:
- **Huber kink.** `smooth_l1_loss` (β=1) on rewards/10 equals 1/10 × Huber with β=10 on
  raw rewards (Adam cancels the 1/10). TD errors of 1–10 raw units move from the linear
  part (all push ±1, error size lost) to the quadratic part (push ∝ error); the −100
  crash outlier stays capped. Only helps if many TD errors are in that range — not
  measured yet.
- **Output scale.** Expected Q on a good lap is ~1.3/(1−0.99) ≈ 130 raw vs ~13 scaled;
  the network starts with outputs ≈ 0, so scaled targets are closer to where it starts.

**Rejected:** /1000 — every TD error falls in the quadratic part, so Huber ≈ MSE and the
−100 outlier dominates batches again (≈30× a normal sample's gradient).

**Effect on test score:** not trained yet.

## 2026-10-20 — Fused Adam, faster grad clipping, own checkpoint path (uncommitted)

**What:**
- `optim.Adam(..., fused=True)`: one kernel for the whole parameter update (works on
  CPU, MPS and CUDA).
- `clip_grad_norm()` replaces `nn.utils.clip_grad_norm_`: one norm over all gradients
  concatenated, then one `_foreach_mul_`. Same max-norm 10.
- `CHECKPOINT = "checkpoint_CarRacing-v3_lab3.pth"` is the default for `train()` and
  `render()` (which `__main__` calls after training).

**Why:** Profiling `replay()` (10.1 ms) showed `clip_grad_norm_` at ~3 ms (26%) and the
Adam step at ~1.5 ms. MPS has no foreach kernels and ~0.1 ms overhead per op, so the
builtin clip's ~25 tiny ops dominate; the math is trivial. The checkpoint path keeps
training from overwriting the baseline `checkpoint_CarRacing-v3_best.pth` (which
`test_car_racing.py` and `render_car_racing.py` still use by default).

**Checks:** new clipping returns the same norm and identical gradients as the builtin,
both when it clips and when it doesn't (`allclose`, rtol 1e-5).

**Measured** (MPS):

| | before | after |
|---|---|---|
| `clip_grad_norm` | 2.73 ms | 1.57 ms |
| Adam step | 0.99 ms | 0.50 ms |
| `replay()` | 10.12 ms | 8.14 ms |
| real loop, ms/step | 28.8 | 26.0 |

Since the start of the speed work: 34.3 → 26.0 ms/step (−24%), plus no validation
video stalls.

## 2026-10-20 — TD-error and grad-clipping diagnostics (uncommitted)

**What:** `replay()` accumulates, on the device, the share of |TD errors| ≥ 0.1 / 1 / 10
(scaled reward units) and the gradient norm (mean, max, share of updates clipped at 10).
`train()` prints `agent.pop_stats()` every `stats_every=10` episodes.

**Why:** Two open questions need data from a real run:
- Does reward scaling matter for Huber? Only if a large share of |TD| is ≥ 1 (linear part).
- Is grad-norm clipping doing anything? If it (almost) never triggers, it can be dropped
  (~1.6 ms per update).

**Checks:** stats for a fixed batch match a manual recomputation. Overhead: `replay()`
8.14 → 8.22 ms (counters stay on the device; one sync per report).

**Results — 200-episode run** (`logs/stats_run_200ep.log`, 10.4 min, ε 1.0 → 0.37,
training avg −55 → −32, i.e. the agent isn't driving yet):

| | episodes 1–100 | episodes 100–200 |
|---|---|---|
| \|TD\| ≥ 0.1 scaled (≥ 1 raw) | ~1.1% | 2.7–3.3% |
| \|TD\| ≥ 1 scaled (≥ 10 raw) | 0.0% | 0.0% |
| grad norm mean / max | ≤0.01 / 0.31 | 0.04 / 0.11 |
| updates clipped at 10 | 0% | 0% |

- **Clipping** never triggered; the largest norm (0.31) is ~30× below the limit.
- **Huber**: every TD error is in the quadratic part, and ~97% would be even without
  reward scaling (< 1 raw unit). So far the "/10 moves errors out of the linear part"
  argument doesn't apply; only the output-scale argument does.
- **Caveat:** this covers only the phase before the car drives. Q-values (and TD errors)
  should grow once laps start succeeding; a full run's logs will show whether that
  changes either conclusion.

## 2026-10-20 — Detailed run artifacts + Colab notebook (uncommitted)

**What:**
- `train(cfg)` takes a `Config` dataclass (all hyperparameters, incl. `max_grad_norm=None`
  to disable clipping) and writes `runs/<timestamp>_<run_name>/`:
  `config.json` (config + device/GPU/CPU/versions/git commit), `episodes.csv` (score,
  length, end reason lap/off_playfield/timeout, Q(s₀) vs discounted return, mean Q, action
  counts, ε), `updates.csv` (loss, Q, target, |TD| shares, grad norm mean/max, clipped
  share, non-finite grads, env steps/s, updates/s, ms per loop phase act/update/env wait),
  `evals.csv` (greedy episodes on fixed tracks), `summary.json`, `checkpoints/`.
- Greedy evaluation every `eval_every=50` episodes on 10 fixed seeded tracks (10000–10009),
  run in parallel in a second vector env. `best.pth` = best eval mean; the old criterion
  (best training avg100 under ε-greedy) is kept as `best_train_avg.pth` for comparison.
- Without clipping, the grad norm is still measured every `grad_norm_every=50` updates.
- `act_batch(..., return_q=True)` also returns max-Q (no extra forward pass) and copies
  uint8 frames to the device instead of float32.
- `test()` (used by `render()`) now stops at the end of each episode; it kept stepping
  until 10000 steps before.
- `train_colab.py` (jupytext, paired with `train_colab.ipynb`): clone repo, install
  `gymnasium[box2d]==1.3.0`, set `Config`, train, run `test_car_racing.py` on `best.pth`.
  Runs go to Google Drive by default.

**Why:** Analyse and compare runs from data instead of console logs; pick the submitted
checkpoint by greedy performance (what `test_car_racing.py` measures) rather than by the
ε-greedy training average; train on a Colab GPU.

## 2026-10-20 — Replay ratio instead of fixed updates per step (uncommitted)

**What:** `Config.updates_per_step` → `Config.replay_ratio` (gradient updates per stored
transition, default 0.25). Each loop iteration earns `replay_ratio × transitions stored`
updates; the fractional part carries over (10 envs: 2, 3, 2, 3, …).

**Why:** The first Colab run used 12 envs (= CPU count) with the fixed 2 updates per
step: 2/12 ≈ 0.17 updates per transition vs the baseline's 2/8 = 0.25, i.e. ~167k
instead of ~250k updates over 1000 episodes. Fixing the ratio keeps runs comparable
whatever `num_envs` the machine allows (12 envs → 3 updates per step).

---

## 2026-10-20 — Run 1 on Colab L4: `ddqn_rs0.1` (`a5f2758`, 1008 episodes, 51 min)

Config: defaults above, 12 envs, replay ratio 0.25 → 1.0M env steps, 252k updates.

| Greedy eval (10 fixed tracks) | ep 360 | ep 504 | ep 756 | ep 1008 (best) |
|---|---|---|---|---|
| mean score | 151 | 311 | 502 | **541 ± 108** |

`test_car_racing.py best.pth 50` (unchanged test, run locally on MPS): **mean 577.0,
std 92.8, min 276.5, max 728.5**, vs the baseline's 840.6 / 103.2 / 470.4 / 912.7.

Training avg100 at the end: 503, still rising ~40 per 100 episodes. **Not converged**,
and well below the baseline checkpoint (840).

Findings:
- All but 3 of 1008 training episodes end on the 1000-step timeout; the car never
  finishes a lap and covers ~60% of the track. The limit is speed: gas is a 0.2-throttle
  action, and the agent still picks noop ~15% of the time.
- Q is calibrated: eval `q_start` 40.4 vs actual discounted return 44.3 (raw units).
  No overestimation, so Double DQN is doing its job. Q still grows each block, which
  means values are still propagating.
- |TD| ≥ 1 (scaled) stays ≤ 0.1% and the max grad norm is 0.79 vs the 10.0 clip: **clipping
  never triggered**. Huber stays in its quadratic part, so it's effectively MSE here.
- `nonfinite_grads` ≈ 0.2 per 500 updates: GradScaler skipping overflow steps, which is normal.
- Throughput is **GPU-loop bound**: `ms_update` 28 ms per iteration (3 updates, ~9.5 ms
  each) vs `ms_env_wait` 0.9 ms. 9.5 ms per update is slow for an L4; the likely suspect
  is 12 env worker processes on 12 vCPUs starving the main process that launches kernels.

## 2026-10-20 — Faster experiments: no-progress cut, per-step ε, true terminals (`b7f4e42`)

Run 1 took 51 minutes and still wasn't converged, which is too slow to try ideas. Every
piece below is a `Config` flag, so it can be switched off for comparison.

- **No-progress truncation** (`no_progress_steps=100`): the `NoProgressTruncation`
  wrapper, applied in `make_vec_env` for training envs only, truncates an episode after
  100 frames without a new tile. Detection: a new tile gives +1000/N − 0.1 > 0 and every
  other frame gives −0.1 or −100, so `reward > 0` ⇔ new tile. Measured before choosing
  100: driving on track with gas, a new tile comes at least every 12 frames; a random
  policy dithers on track with gaps up to 114–139 frames. 50 would cut the early random
  episodes too soon after reset. With random actions, episodes now end at 142–527
  frames instead of 1000. `make_env`, evals and `test_car_racing.py` are unchanged and
  run full episodes. New end reason in `episodes.csv`: `no_progress`.
- **Truncation bootstraps** (`bootstrap_truncation=True`): the buffer used to store
  `terminated or truncated` as done, so the 1000-step timeout (99.7% of run 1's episodes)
  set the target to r alone, "no future value", for a state the agent can't tell is near
  the time limit. Now only real terminals (off playfield, lap finished) zero the
  bootstrap. The no-progress cut is a truncation, so it bootstraps too; otherwise it
  would teach the same wrong "the world ends here".
- **ε decays linearly per env step** (`epsilon_decay_steps=150_000`): 1.0 → 0.05 over
  150k steps. Per-episode 0.995 decay kept run 1 mostly random until episode ~600, and
  would tie ε to episode length now that episodes vary.
- **Budget and evals in env steps** (`total_env_steps`, `eval_every_steps=50_000`;
  `n_episodes` is now optional): with variable-length episodes, episode counts no longer
  measure training. Eval checkpoints are named `step<N>k.pth`; `evals.csv` gains `env_steps`.
- **Batch 256, replay ratio 0.125, target_update 500**: the same samples per transition
  (0.125 × 256 = 0.25 × 128) with half the update launches. `target_update` counts
  updates, so it's halved to keep the sync at 4000 transitions. Untested: half as many
  optimizer steps at the same lr may learn slower per sample.
- **Colab: `num_envs = cpu_count − 2`**: tests the hypothesis that run 1 was slow because
  12 env workers on 12 vCPUs starved the process that launches GPU work.

Training scores and `avg100` are now from possibly cut episodes, so they're lower than
full-episode scores. Compare runs by eval scores. Plan: one 300k-step screening run
(notebook default), compared against run 1's eval at the same step count.

## 2026-10-20 — Run 2 on Colab L4: `screen_fast` (`b7f4e42`, 300k env steps, 13.5 min)

Greedy eval mean (10 fixed tracks) at equal env steps vs run 1:

| env steps | 50k | 100k | 150k | 200k | 250k | 300k |
|---|---|---|---|---|---|---|
| run 1 (`a5f2758`) | −90 (60k) | −8 (108k) | −54 (156k) | −60 (204k) | −45 (252k) | −27 |
| run 2 | −79 | −51 | −16 | −17 | 312 | **359 ± 94** |

Run 2 at 300k steps is where run 1 was at ~550–650k steps (eval 343–345), so it needs about
half the samples. 514 of 624 training episodes were cut by `no_progress`; mean episode length
rose from 281 frames (ε ≈ 0.9) to 951 frames once ε hit 0.05, so the cut stops costing
anything once the agent drives.

Throughput: 418 env steps/s vs 377 (+11%) with 10 envs instead of 12.
- `ms_update` 11.8 ms per iteration for 1.25 updates = 9.4 ms per batch-256 update, the
  same as run 1's 9.4 ms per batch-128 update. Doubling the batch cost nothing, so updates
  are bound by launch and Python overhead, not by GPU compute.
- `ms_env_wait` rose from 0.9 to 8.5 ms: one vector step of 10 envs takes ~20 ms on Colab's
  vCPUs, so the loop is now bound by envs and updates about equally. The starvation
  hypothesis is unconfirmed: per-update time didn't drop with 2 free CPUs.
- Clipping still never triggered (max grad norm 0.17).

## 2026-10-21 — Run 3 on Colab L4: `final` (`b7f4e42`, 1.5M env steps, 68.5 min)

Run 2's config with `total_env_steps=1_500_000`.

| env steps | 300k | 600k | 900k | 1.05M | 1.2M (best eval) | 1.25M | 1.5M |
|---|---|---|---|---|---|---|---|
| eval mean (10 tracks) | 359 | 582 | 731 | 766 ± 82 | **789 ± 117** | 606 ± 229 | 690 ± 205 |

`test_car_racing.py <ckpt> 50` (unchanged test, local MPS):

| checkpoint | mean | std | min | max |
|---|---|---|---|---|
| baseline | 840.6 | 103.2 | 470.4 | 912.7 |
| `best.pth` (1.2M, eval 789) | 753.7 | 198.5 | −76.0 | 912.3 |
| `step01050k.pth` (eval 766) | **829.3** | **86.2** | 420.8 | 911.7 |

Findings:
- Training avg100 rose steadily to 751 at the end, and laps start appearing after
  ~900k steps. Eval scores, though, swing 600↔790 between consecutive evals after 1M,
  with per-track minimums down to 260–370. The greedy policy is unstable from one
  checkpoint to the next, and a bad checkpoint fails badly on some tracks (−76 in the test).
- **Choosing `best.pth` by the highest mean over 10 eval tracks picks noise.** The
  checkpoint with the highest eval mean scored 76 lower on the 50-track test than one
  with a lower eval mean but a much smaller std. Ten tracks with std ~100–200 give a
  standard error of 30–60 on the mean, about the size of the differences between checkpoints.
- `step01050k.pth` is within noise of the baseline (−11; SE of the difference ≈ 19), with
  a lower std (86 vs 103). Not better yet.
- Q overestimation is mild: training `q_start` 49 vs return 45.5 at the end. |TD| ≥ 1
  (scaled) still ~0, and max grad norm 1.5, so clipping never triggered.
- Throughput was steady at 410 env steps/s.

## 2026-10-21 — Action repeat, dueling head, bigger buffer, tile/speed diagnostics (`632f877`)

Run 3's evals at the 1000-frame limit put tile coverage at ~87% (score ≈ 1000 × tile_frac
− 100), with only 1 of 10 eval tracks finishing a lap. So the remaining gap is mostly that
the car is **too slow**, plus instability between checkpoints. TD errors are small and Q
matches the returns, so the CNN doesn't look like the bottleneck.

- **Action repeat** (`action_repeat`, training only): the `ActionRepeat` wrapper holds each
  action for k frames and returns the summed reward; the target uses `gamma ** k`. Credit
  assignment spans k× fewer decisions (at γ = 0.99, k = 4: 0.9606 per decision), and the
  act/update work per frame drops k×. Env steps still count frames (`info["frames"]`), so
  budgets, ε and evals compare across k. The observation is still `make_env`'s stack of 4
  consecutive frames, so the policy can act every frame. Evals and `test_car_racing.py`
  do act every frame, so every eval checks that transfer. Known imprecision: a no-progress
  cut in the middle of a repeat still discounts by γ^k.
- **Dueling head** (`dueling`, Wang et al. 2016, arXiv:1511.06581):
  Q = V(s) + A(s,a) − mean A. Every update trains V(s), which helps when most actions are
  nearly equal (straights: noop ≈ gas ≈ light steer). The advantage stream keeps the names
  `fc1`/`fc2`; the value stream is `fc_value1`/`fc_value2`. `QNetwork.load_state_dict`
  rebuilds the head from the checkpoint's keys, so the unchanged test script (which builds
  the default network) loads both baseline and dueling checkpoints. Checked: identical
  outputs after the round trip.
- **Replay buffer 50k → 200k** (Colab only): 50k transitions is ~50 episodes, so rare
  situations (hairpins) are forgotten, which fits run 3's eval swings. 200k × 2 stacks ×
  28 KB ≈ 11 GB on the L4 (24 GB). `Config` default stays 50k for local runs.
- **Diagnostics:** the `EpisodeInfo` wrapper (training and eval envs, not the test) adds
  `tiles`, `track_tiles`, `tile_frac` and `speed_mean` to `episodes.csv` and `evals.csv`.
  Episode `length` and the `a_*` action counts are in frames. Eval prints show mean tile coverage.
- Notebook: `CONFIGS` runs several configs in sequence. Screen = 100k frames, ε decay over
  50k, evals every 25k: `duel_ar4` (repeat 4, replay ratio 0.5 = same updates per frame as
  repeat 1 at 0.125) vs `duel_ar1`.

## 2026-10-21 — Runs 4–5, local MPS: action repeat 4 vs 1 (100k frames each)

Both runs: dueling, 10 envs, buffer 100k (never fills in 100k frames, so it behaves the
same as 200k), ε 1 → 0.05 over 50k frames, about 12k updates each (same updates per frame).

| eval mean (10 tracks) | 25k | 50k | 75k | 100k |
|---|---|---|---|---|
| `duel_ar4` (repeat 4, rr 0.5) | 197 | 391 | 658 | **750 ± 85** (tiles 85%) |
| `duel_ar1` (repeat 1, rr 0.125) | −64 | −85 | −78 | 9 ± 104 (tiles 11%) |
| run 3 (`final`, for reference) | — | −79 | −16 (150k) | −51 |

`test_car_racing.py duel_ar4/best.pth 50`: **mean 780.8, std 92.1, min 562.6, max 902.0**.
That's after 100k frames and 9 minutes on the Mac. Run 3 needed 1.05M frames and 50 min
on the L4 to reach 829 / 86.

- **Action repeat is the big win**: at equal frames and equal updates, ar4 is ~740 points
  ahead at 100k. The policy transfers to acting every frame: evals and the test both act
  every frame, and the test min (563) is higher than any run 3 checkpoint's.
- The eval mean speed for ar4 rose from 37 to 44 as it learned; ar1 at 100k is still
  learning to leave the start.
- Throughput, local: ar4 371 frames/s and env-bound (`ms_env_wait` 13.6 vs `ms_update`
  11.6 per iteration of 40 frames); ar1 250 frames/s. ar1's last eval took 301 s instead
  of 10 s, probably the Mac being busy or sleeping, which also lowers its frames/s.

## 2026-10-21 — `dqn_car_racing_improved.py`: baseline + only the proven changes (`2dcc133`)

A second training script, copied from the upstream baseline (`ff03938`) with only the
changes the runs above support. `dqn_car_racing.py` stays as the experiments sandbox;
`train_colab.py` now trains `dqn_car_racing_improved.py`.

| Kept | Evidence |
|---|---|
| Device replay buffer, batched acting, env stepping overlapped with updates, fused Adam, MPS | benchmarks (34.3 → 26.0 ms/step) |
| No-progress cut (100 frames), truncation bootstraps, linear ε per frame (50k), budget/evals in frames | run 2 bundle (2× sample efficiency) |
| Action repeat 4, γ⁴ per transition | runs 4–5 (750 vs 9 at 100k) |
| Batch 256, replay ratio 0.5/transition, target_update 500, buffer 100k, 10 envs | the `duel_ar4` config |
| CSV artifacts; tiles/speed/Q/loss/\|TD\| logs; timings | infrastructure |
| 30 eval tracks (in rounds of 10) every 25k frames pick `best.pth` | run 3 showed 10 tracks pick noise |
| Baseline `make_env`, `QNetwork`, Huber loss, `test()`/`render()` | unchanged (`test()` gets the `break` fix) |

Dropped: gradient clipping (never fired in 5 runs), AMP/GradScaler (unbenchmarked, likely
syncs every update; fp32 with TF32 instead), TD-distribution and grad-norm stats, Double
DQN, reward scale, dueling (Double DQN and reward scale were restored after validation, see
below). The last three were never A/B-tested; they were dropped to keep
only what was shown to matter, so a validation run was required first.

### Validation 1: without Double DQN, reward scale and dueling — failed

100k frames, local MPS, otherwise identical to `duel_ar4`. Pass criterion agreed beforehand:
within ~150 of `duel_ar4`'s 750 at 100k on the same tracks.

| eval mean, seeds 10000–10009 | 25k | 50k | 75k | 100k |
|---|---|---|---|---|
| `duel_ar4` (dueling + Double DQN + /10) | 197 | 391 | 658 | **750 ± 89** |
| `ar_validate` (none of the three) | −55 | 160 | 533 | **519 ± 242** |

Gap 231 → fail. Over all 30 tracks: 541 ± 230, tile coverage 64%, 2/30 laps. Q-values lag
far behind the returns: eval `q_start` 15 vs discounted return 54 (`duel_ar4`: 23 vs 37).
Suspect: **reward scale**. With action repeat 4, one transition sums ~4 frames of tile
reward (up to ~7–10 raw), and mean |TD| is 0.71 raw, so many samples sit in Huber's linear
part where the gradient is capped at ±1 and values grow slowly. With per-frame rewards this
hadn't mattered. Single seed per config, so this is a lead, not proof.

### Validation 2: reward scale /10 restored — Q lag fixed, still failed

Same run with `reward_scale=0.1` added back to `dqn_car_racing_improved.py` (stored rewards only;
logs in env units).

| eval mean, seeds 10000–10009 | 25k | 50k | 75k | 100k |
|---|---|---|---|---|
| `duel_ar4` | 197 | 391 | 658 | **750 ± 89** |
| `ar_validate` (no /10) | −55 | 160 | 533 | 519 ± 242 |
| `ar_validate_rs0.1` (/10) | −50 | 448 | 495 | **453 ± 270** |

Over all 30 tracks: −24 / 435 / 555 / 520, best 555 at 75k; mins down to −29.
- The scale **did** fix the value lag: eval `q_start` 27 vs return 38 at 100k (no /10: 15
  vs 54; `duel_ar4`: 23 vs 37), mean |TD| 0.071 scaled. Learning is faster early on (435 at
  50k vs 177 without /10).
- It did **not** close the gap at 100k: both variants plateau around 520 on 30 tracks with
  a std of ~230 and some tracks failing badly; `duel_ar4` had a std of 89.
- **How many samples sit in Huber's linear part without /10:** measured afterwards on both
  runs' `last.pth`. About 3000 transitions each from 10 training envs (action repeat 4,
  ε = 0.05, no-progress cut); δ = r + γ⁴·max Q(s′) − Q(s,a) in the units the loss sees,
  with the online net standing in for the target net (not saved):

  | final checkpoint | mean \|δ\| | median | p90 | p99 | \|δ\| ≥ 1 (linear Huber) | \|δ\| ≥ 3 |
  |---|---|---|---|---|---|---|
  | no /10 (raw units) | 2.22 | 1.90 | 4.33 | 8.33 | **76.5%** | 23.5% |
  | /10 (scaled units) | 0.21 | 0.17 | 0.41 | 0.78 | **0.2%** | 0% |

  Raw reward per transition: mean 1.6–2.4, max 10–11. Without /10, three quarters of the
  batch get the same ±1 gradient whether they're off by 1.1 or by 8, so the loss behaves
  mostly like L1 (pulls Q toward the median target), and only the remaining quarter carries
  error magnitude. With /10 it's effectively MSE. This describes the end of training: the
  logged run-average mean |TD| without /10 was 0.71, since errors start near 0 while Q ≈ 0.
  Absolute gradient size doesn't matter here (Adam normalizes it away); the relative
  weighting and the 10× closer targets do.
- Remaining suspects: Double DQN and/or dueling, or seed luck.

### Validation 3: seed check of the reference + Double DQN restored — passed

Two more 100k-frame local runs: `duel_ar4` again with `seed=1` (sandbox, same config), and
`dqn_car_racing_improved.py` with /10 **and** Double DQN (new flag `double_dqn`).

| eval mean ± std, seeds 10000–10009 | 25k | 50k | 75k | 100k | q_start / return at 100k |
|---|---|---|---|---|---|
| `duel_ar4` (seed 0) | 197 | 391 | 658 | **750 ± 89** | 23 / 37 |
| `duel_ar4_seed1` | −3 | 437 | 522 | **652 ± 186** | 25 / 39 |
| new, no /10, no DDQN | −55 | 160 | 533 | 519 ± 242 | 15 / 54 |
| new, /10, no DDQN | −50 | 448 | 495 | 453 ± 270 | 27 / 38 |
| **new, /10 + DDQN** | −3 | 463 | 565 | **635 ± 161** | 23 / 39 |

On all 30 tracks, new + /10 + DDQN: −1 / 496 / 606 / **704**, min 370 at 100k, 3/30 laps,
80% tile coverage.

- **The reference is partly seed luck:** the same `duel_ar4` config scores 750 and 652 with
  two seeds, so seed-to-seed spread at 100k is ~100. 750 was the lucky end.
- **New script + /10 + Double DQN is within seed noise of the reference** (635 vs 652 / 750
  on the same tracks) → pass (criterion: within ~150 of 750). **Dueling isn't needed.**
- **Double DQN mattered here:** without it 453 ± 270 with some tracks failing (min 15);
  with it 635 ± 161, min 370 over 30 tracks. The gap (182) is larger than the observed seed
  spread, though each variant of the new script has only one seed.
- **Decision:** `dqn_car_racing_improved.py` keeps reward scale /10 and Double DQN (both on by
  default) and stays without dueling. `duel_ar4` is one seed, and
  nothing yet shows how much a second seed of the same config would vary.

## 2026-10-21 — Run 6 on Colab L4: `ar4_500k` (`2dcc133`, `dqn_car_racing_improved.py`, 500k frames, 38 min)

| eval mean (30 tracks) | 100k | 150k | 200k | 250k | 300k | 375k | 475k (best) | 500k |
|---|---|---|---|---|---|---|---|---|
| mean | 675 | 706 | 710 | 755 | 680 | 771 | **793 ± 165** | 757 |
| laps / 30 | 1 | 3 | 2 | 7 | 5 | 10 | 11 | 9 |

`test_car_racing.py best.pth 50`: **mean 769.1, std 136.4, min 424.5, max 917.2**, below the
baseline's 840.6 / 103.2 (SE of the difference ≈ 24, so the gap is real) and below run 3's
`step01050k.pth` (829.3).

- **Plateau from ~250k frames:** eval mean stays at 680–790 and tile coverage at ~0.85–0.9,
  with 5–11 of 30 tracks finishing a lap. Every eval has a few tracks failing (min 290–440).
- **Overestimation grows:** eval `q_start` climbs steadily (24 at 100k → 57 at 500k) while
  the actual discounted return stays flat at ~40–47. That's ~30% too high at the end even with
  Double DQN, in the plateau phase. In the validation runs (100k) Q still underestimated.
- **Throughput: dropping AMP sped updates up** — `ms_update` 28 ms per 5 updates = **5.7 ms per
  batch-256 update**, vs 9.4 ms with AMP (runs 1–3). The loop is now env-bound: `ms_env_wait`
  48 ms per iteration (10 envs × 4 frames), 430 frames/s.

## 2026-10-21 — Action-repeat schedule, slower target, lr decay (`a54c131`)

Aimed at run 6's plateau (eval ~750 from 250k frames on, Q drifting 30% above the returns,
a few tracks failing). Three changes in `dqn_car_racing_improved.py`, all configurable, run
together in one 1M-frame Colab run (`ar_sched_1m`), so the run can't attribute the effect to
one of them; it's compared with run 6 at matching frame counts.

- **Action-repeat schedule** (`action_repeat_schedule`, replaces `action_repeat`): a
  curriculum, coarse control while the agent learns the game and faster reactions later.
  Notebook: k = 4 until 300k frames (run 6 plateaued at ~250k), 3 until 500k, 2 until 750k,
  1 until 1M, so the last phase trains exactly as the test acts. k changes in the running envs through
  `env.set_attr("k", k)` between `step_wait` and the next `step_async` (checked: frames per
  step follow 4 → 2 → 1).
  - The buffer now stores **γ^frames per transition** and the target uses it, because
    transitions from different k sit in the buffer together. This also fixes the old
    imprecision of a no-progress cut in the middle of a repeat.
  - Q-values stay comparable across k because γ is applied per frame.
  - `replay_ratio` (per transition) → **`updates_per_frame=0.125`** (the validated 0.5 × 1/4),
    so k = 1 doesn't quadruple the compute per frame. Update credit is now earned by the
    frames actually run.
- **Slower target:** `target_update` 500 → **2000** updates (16k frames between syncs at any k),
  against the overestimation drift.
- **lr decay** (`lr_end`, `lr_decay_start`): held at 1e-4 until 300k frames (the first k
  switch), then linear to
  1e-5 at the end of the budget. Holding it first keeps the fast early learning.
- New log columns: `action_repeat` (episodes, updates) and `lr` (updates).

## 2026-10-21 — Run 7 on Colab L4: `ar_sched_1m` (`a54c131`, 1M frames, 77 min)

Config as in the notebook: k schedule 4 → 3 (300k) → 2 (500k) → 1 (750k), `target_update=2000`,
lr 1e-4 held to 300k then linear to 1e-5, everything else as run 6. Evals: 30 fixed tracks,
acting every frame.

| Frames | k | Eval mean | Run 6 eval | q_start vs return_disc (run 6) |
|--------|---|-----------|------------|--------------------------------|
| 100k   | 4 | 472.6     | 675.3      | 11.9 vs 39.6 (23.6 vs 42.2)    |
| 250k   | 4 | **781.7** (best) | 754.5 | 19.8 vs 45.9 (47.0 vs 40.2) |
| 300–475k | 3 | 702–778 | 680–793    | 22–30 vs 38–46                 |
| 500–750k | 2 | 684–755 | —          | 31–39 vs 42–46                 |
| 775k–1M  | 1 | **483–605** | —      | 41–42 vs 38–42                 |

- **Slower target removed the Q overestimation.** Q now climbs to the discounted return from
  below and settles at it (42 vs 39 at 1M) instead of overshooting (run 6: 57 vs 43). Cost:
  slower early learning (473 vs 675 at 100k).
- **k = 3 and k = 2 didn't lift the plateau** (~700–780, same as run 6 at k = 4).
- **k = 1 collapsed**: eval 725 → 483 within 50k frames, eval speed 43 → 29, training
  episodes hit the no-progress cut again (21 of 56 at 800–850k vs ~2 of 50 before). Action mix
  barely changed (brake 1.3% → 3.5%). Likely causes, not separated: the bootstrap chain per
  decision is 4× longer at k = 1, the 100k-transition buffer covers only 100k frames instead of
  ~400k, and lr was already down to ~3e-5, so it couldn't re-adapt.
- Speed: ~430 frames/s at k = 4, ~380 at k = 1 (more `act` calls per frame); 7.7 ms/update.

`test_car_racing.py <ckpt> 50` (unchanged test, local MPS):

| Checkpoint            | Mean      | Std  | Min   | Max   |
|-----------------------|-----------|------|-------|-------|
| `best.pth` (250k, k=4) | **798.4** | 85.3 | 619.9 | 911.7 |
| `step00475k.pth` (k=3) | 792.3     | 98.2 | 372.0 | 907.5 |
| baseline (reported)   | 840.6     | 103.2 | 470.4 | 912.7 |
| baseline (rerun locally, same checkpoint) | 819.0 | 136.7 | 394.9 | 910.3 |

Best test result so far (run 5: 780.8, run 6: 769.1) and the best worst case (min 619.9).
**Baseline rerun:** `checkpoint_CarRacing-v3_best.pth` tested again gives 819.0, 21.6 below the
reported 840.6. The test doesn't seed its tracks, so the same checkpoint moves by ~20 between
runs. Against the rerun, run 7 is 20.6 below with a standard error of the difference of
√(85.3²/50 + 136.7²/50) ≈ 23: **within noise**, not a clear loss as earlier comparisons
against 840.6 suggested (that line in run 6's entry also overstated it). Run 7 has a much
smaller spread (std 85 vs 137, min 620 vs 395). The 250k-frame best checkpoint again comes from the k = 4 phase; the extra 750k frames, the
k schedule and the lr decay added nothing. The test uses random tracks, so the 29-point gain over
run 6 (different seeds, ~1.5–2 SE) is suggestive, not proven.

**Proposed (not yet decided):** drop the k = 1 phase (harmful) and the schedule (no gain); keep
`target_update=2000` (fixes Q drift, best test so far).

## 2026-10-21 — γ 0.995 + 3-step returns

Run 7 showed the score is capped by speed: at its best eval, 24 of 30 tracks time out at 85%
of the tiles, only 6 finish a lap, and speed correlates 0.67 with score. Idea: let consequences
beyond ~100 frames count (spin-outs, corner setup) and propagate reward faster.

- **`gamma` 0.99 → 0.995 per frame:** horizon ~100 → ~200 frames.
- **`n_step` (new, default 1):** `NStepCollector` keeps each env's last n decisions, stores
  (s_t, a_t, Σ per-frame-discounted rewards of n decisions, s_t+n), and the target discounts by
  γ^(frames of all n decisions); at k = 4 and n = 3 that's 12 frames per bootstrap. When an
  episode ends, all pending transitions are flushed with the final state (bootstrapping on
  truncations, zero on real terminals). n = 1 reproduces the old behaviour exactly. Intermediate
  actions are the behaviour policy's (ε-greedy, older weights); no correction, as in Rainbow.

**Run 8, local MPS, `g995_n3_100k`:** run 7's config over its first 100k frames (k = 4,
`target_update=2000`, lr 1e-4, seed 0, 10 envs) with γ 0.995 and n = 3; 6.6 min. Same 30 eval
tracks as run 7, so the comparison is paired per track:

| Frames | Run 8 | Run 7 | Run 6 (target 500) | Run 8 − run 7 (paired, ± SE) | q_start / return_disc, run 8 vs run 7 |
|--------|-------|-------|--------------------|------------------------------|---------------------------------------|
| 25k    | −20.7 | −61.2 | 4.7   | +40 ± 14  | 9.7/26.6 vs 6.9/12.2  |
| 50k    | 298.6 | 26.6  | 279.1 | +272 ± 44 | 11.5/79.8 vs 6.6/24.7 |
| 75k    | 606.3 | 143.3 | 525.7 | +463 ± 52 | 13.4/95.3 vs 8.4/32.8 |
| 100k   | **729.4** | 472.6 | 675.3 | **+257 ± 57** | 21.5/108.2 vs 11.9/39.6 |

- Much faster learning than run 7 with the same slow target: 729 vs 473 at 100k (7 laps vs 0,
  tiles 83% vs 57%, eval speed 46 vs 32), also ahead of run 6's faster target.
- Q lags the return far more (21.5 vs 108; discounted returns are ~2.7× larger at γ 0.995 and
  the target syncs every 2000 updates), so the values are still catching up at 100k.
- One seed only; seed-to-seed spread at 100k has been ~100, the gap here is 257.
- Throughput 380–400 frames/s locally; the n-step bookkeeping isn't visible in it.

**Next:** notebook config `g995_n3_300k` on Colab: k = 4 throughout (schedule dropped after
run 7), constant lr 1e-4, `target_update=2000`, γ 0.995, n = 3, 300k frames, to see whether
the plateau moves above run 7's ~780 eval; then the official test.

## Investigated, not adopted

### Mixed precision on MPS

Measured ms per training update (batch 128):

| Precision                  | ms/update |
|----------------------------|-----------|
| fp32 (kept)                | 5.67      |
| fp16 autocast + GradScaler | 8.21      |
| bf16 autocast              | 5.31      |

**Decision:** stay on fp32. The network is too small for half precision to pay off:
fp16 is slower and bf16 is only ~6% faster, at the cost of less precise Q-targets.
Re-check if the network grows a lot.

### Storing replay frames as float16/float32

Measured ms to sample one batch (states + next_states, 20k-transition buffer) and
move it to MPS:

| Storage                     | ms/batch |
|-----------------------------|----------|
| numpy uint8 → mps (current) | 2.67     |
| numpy float16 → mps         | 4.15     |
| numpy float32 → mps         | 6.24     |
| torch uint8 kept on mps     | 0.86     |

**Decision:** keep uint8. Pixels are 0–255 integers, so uint8 stores them exactly, and
unified memory still copies the batch into a separate Metal buffer, so bigger dtypes
are just slower. A 50k-transition buffer would also grow from 2.8 GB to 5.6/11.3 GB.

### Merging the online-net forward passes in Double DQN

Double DQN runs the online net twice per update: on `states` (with gradients, for the
loss) and on `next_states` (no gradients, only to pick argmax). Idea: one pass on
`cat([states, next_states])` (batch 256) for better GPU utilisation.

| Update (batch 128, MPS)        | ms   |
|--------------------------------|------|
| vanilla DQN (2 forwards)       | 5.55 |
| Double DQN, separate (current) | 6.46 |
| Double DQN, merged online pass | 8.10 |

**Decision:** keep separate passes. In the merged pass autograd records the whole
256-sample batch, so the backward pass also runs through the `next_states` half (whose
gradient is zero). That doubles backward work, which costs more than the saved launch.
The `no_grad` pass is cheap because it stores no activations and has no backward.

### `torch.compile` on MPS

| QNetwork, batch 128 | fwd+bwd | no_grad fwd |
|---------------------|---------|-------------|
| eager               | 3.39 ms | 1.05 ms     |
| `torch.compile`     | 3.05 ms | 1.14 ms     |

**Decision:** not worth it. ~0.3 ms per update at best, and a compiled module's
`state_dict` gets `_orig_mod.` key prefixes, which `test_car_racing.py` couldn't load
without care.

**Ideas still open:**
- ~~Keep the buffer as a uint8 tensor on `mps` and sample on the device~~ — done, see above.
- Store each frame once instead of full 4-frame stacks twice (~8× less memory, so a much
  bigger buffer).
- Per step, the 2 gradient updates (~20 ms) cost more than stepping the envs (~8 ms).
