# v11 Design Notes

## Post-Mortem: Why v10 Failed

v10 test results were **identical** to v8/v9 (539.3%/0.85/53.4%). After code inspection, two critical issues were found:

### Bug 1: `"done"` flag never True

In `environment.py`:
- `done()` returns `True` when `idx >= len(df) - 1`
- `get_state()` sets `"done": True` only when `idx >= len(df)`

After the last `step()`, `idx = len(df) - 1`. The framework calls `get_state()` for the terminal state, but `"done"` is `False` because `len(df) - 1 < len(df)`. **The episode-level terminal penalty was NEVER applied.** This also means `use_sparse_reward` would never work.

**Fix**: Changed `get_state()` condition from `idx >= len(self.df)` to `idx >= len(self.df) - 1`.

### Bug 2: Penalty magnitude too small

Even if applied, v10's quadratic penalty for 50% drawdown:
- excess = 50% - 20% = 30%
- penalty = (0.30²) × 2.0 × 100 = **18 reward units**

An episode has ~300 steps with per-step rewards of ~1-3. Total episode reward = ~300-900. A terminal penalty of 18 is only **2-6%** of total reward. PPO's value function cannot learn temporal credit assignment from such a tiny signal.

**Fix**: Linear penalty (stronger gradient) with 6× larger coefficient.

## v11 Design Changes

### 1. Immediate per-step drawdown penalty (NEW)

Instead of relying solely on terminal credit assignment, v11 penalizes the agent **at every step** when `episode_max_dd` increases:

```python
if episode_max_dd > prev_episode_max_dd:
    increase = episode_max_dd - prev_episode_max_dd
    reward -= increase * penalty * 100.0
```

This gives the agent **instant feedback** — the moment it enters a deeper drawdown, it feels the pain.

### 2. Large linear terminal penalty (FIXED)

At episode end, additional penalty for exceeding threshold:

```python
excess = max(episode_max_dd - threshold, 0)
reward -= excess * penalty * 100.0
```

With `penalty=6.0` and `threshold=0.10`:
- 50% drawdown → terminal penalty = 0.40 × 6.0 × 100 = **240 reward units**
- This is ~25-50% of total episode reward — impossible to ignore

### 3. Stronger downside asymmetry (INCREASED)

v10's additive penalty was too small. v11 **multiplies the entire reward** when `pnl_ret < 0`:

```python
if pnl_ret < 0:
    reward *= coef  # coef = 2.0
```

A negative reward of -5 becomes -10. Losses hurt twice as much.

### 4. Temporal context via observation_window (NEW)

`observation_window: 3` — network sees last 3 days of features instead of just today. This gives the MLP some temporal context to detect drawdown momentum without requiring a recurrent layer.

## Key Parameters

| Parameter | v10 | v11 | Rationale |
|-----------|-----|-----|-----------|
| episode_drawdown_penalty | 2.0 | **6.0** | 3x larger + linear = much stronger signal |
| episode_drawdown_threshold | 0.20 | **0.10** | Lower bar → more drawdowns penalized |
| downside_asymmetry_coef | 1.30 | **2.0** | Multiply (not add) — losses hurt 2x |
| observation_window | 1 | **3** | Temporal context for drawdown detection |

## Expected Outcome

- The per-step penalty should immediately reduce position sizes during volatile periods
- The terminal penalty should shape the overall objective toward lower drawdown
- Downside asymmetry should make the agent cut losses faster
- `observation_window: 3` should help the network detect emerging drawdowns

## Risk

- Per-step penalty could make agent **too conservative** — if every tiny drawdown increase is punished, agent might hold near-zero positions
- `observation_window: 3` increases obs_dim from 59 to ~177 — could slow training or cause overfitting
- If v11 still fails, the problem is likely the MLP architecture itself → next step is GRU/LSTM
