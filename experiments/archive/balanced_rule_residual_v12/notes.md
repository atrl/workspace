# v12 Design Notes

## Why v11 Failed

v11 test: **543.1% return, Sharpe 0.85, MaxDD 54.4%** — drawdown got WORSE, not better.

**Root cause**: The per-step drawdown penalty created perverse incentives.

When the market drops and the agent reduces position, the portfolio might still drop (market drops more than position reduction offsets). The agent gets penalized for this drawdown increase even though it took a risk-reducing action. This teaches the agent that:
1. Reducing position doesn't reliably avoid penalties
2. Better to chase returns aggressively to offset the penalties

Result: agent learned to take MORE risk, leading to higher returns (+4%) but worse drawdown (+1%).

## v12: GRU Architecture + Terminal-Only Penalty

### Key Changes from v11

1. **Removed per-step penalty** — back to terminal-only episode drawdown penalty
2. **Added GRU layer** — `gru_hidden_size: 64` processes the 3-day observation window
3. **Kept observation_window: 3** — GRU receives a sequence of 3 timesteps × 59 features
4. **Reduced downside asymmetry** — from 2.0 to 1.5 (v11's 2.0 may have been too aggressive)

### Why GRU Should Help

The fundamental problem with all v2-v11 profiles is the **MLP architecture**:
- Processes each timestep independently
- No memory of previous states
- Cannot learn "I am in a drawdown, I should reduce risk"

GRU solves this by maintaining a hidden state across the episode:
- Hidden state encodes "drawdown memory"
- Agent can learn to associate high hidden-state activation with "reduce position"
- Enables temporal credit assignment for terminal penalties

### Architecture

```
Input: (batch, 184)  # 3×59 features + 6 extra + 1 asset
  → Reshape to (batch, 3, 59)
  → GRU(59 → 64) → hidden state (batch, 3, 64)
  → Take last timestep: (batch, 64)
  → Concatenate with extra features: (batch, 64 + 7) = (batch, 71)
  → MLP(71 → 128 → 64) → features for Actor/Critic heads
```

## Expected Outcome

- GRU should learn to track drawdown state through hidden state
- Terminal penalty should now be effective because GRU can associate current actions with future terminal penalty
- Lower downside asymmetry (1.5) should reduce excessive risk-chasing
- Test MaxDD should improve toward v2's 50.4%

## Risk

- GRU adds ~12K parameters — could overfit on 14 training episodes
- Training might be slower (~20% slower due to sequential processing)
- If GRU doesn't help, next step is to try LSTM or Transformer
