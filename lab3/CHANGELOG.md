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

## 2026-10-20 — Replay buffer on the device (uncommitted)

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

## 2026-10-20 — Double DQN target (uncommitted)

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

**Ideas still open:**
- ~~Keep the buffer as a uint8 tensor on `mps` and sample on the device~~ — done, see above.
- Store each frame once instead of full 4-frame stacks twice (~8× less memory, so a much
  bigger buffer).
- Per step, the 2 gradient updates (~20 ms) cost more than stepping the envs (~8 ms).
