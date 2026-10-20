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

---

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
