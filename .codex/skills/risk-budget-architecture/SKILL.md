# Risk-Budget Architecture

## When to use

Use when designing or debugging a SOXL trading strategy that uses **budget-usage position mapping** instead of direct target-position mapping. This architecture is the only profile family that has passed the fixed-split promotion gate (test_2022 MaxDD < 30%, avg_pos < 0.35, Sharpe > 0.90).

## Core Idea

Instead of the actor outputting an absolute position (e.g., 0.60 = 60% invested), the actor outputs a **budget usage ratio** (e.g., 0.50 = use 50% of the current mode-owned risk band). A slow mode-router layer decides the allowed risk band, and the fast actor only decides how much of that band to consume.

```
Traditional:  Actor → target_position ∈ [0, 1]
Budget-Usage: Actor → usage_ratio ∈ [0, 1] → position = band_floor + usage × (band_cap - band_floor)
```

## Architecture Diagram

```mermaid
flowchart TD
    A[Market State] --> B{Mode Router}
    B -->|flat_wait| C[Band 0.00-0.25]
    B -->|defensive_hold| D[Band 0.10-0.45]
    B -->|crash_rebuild| E[Band 0.25-0.65]
    B -->|trend_hold| F[Band 0.45-0.95]
    C & D & E & F --> G[Actor: delta ∈ [-0.10, -0.05, 0, 0.05, 0.10]]
    G --> H[Usage = map_delta_to_ratio]
    H --> I[Position = floor + usage × (cap - floor)]
    I --> J[Position Cap Overlay]
    J --> K[Final Position]
```

## Mode Bands (risk_budget_usage_v1)

| Mode | Floor | Cap | Typical Regime |
|------|-------|-----|----------------|
| flat_wait | 0.00 | 0.25 | High regime_risk, no trend |
| defensive_hold | 0.10 | 0.45 | Medium risk, bearish bias |
| crash_rebuild | 0.25 | 0.65 | Deep drawdown, early rebound |
| trend_hold | 0.45 | 0.95 | Confirmed trend, bull market |

## Key Configuration

```yaml
model:
  position_mapping: budget_usage    # Critical: switches from target_position to budget_usage
  use_hybrid_overlay: true
  delta_levels: [-0.10, -0.05, 0.0, 0.05, 0.10]  # 5 actions → [0%, 25%, 50%, 75%, 100%]
  hard_floor: 0.20

mode_router:
  enabled: true
  flat_wait_floor: 0.00
  flat_wait_cap: 0.25
  defensive_hold_floor: 0.10
  defensive_hold_cap: 0.45
  crash_rebuild_floor: 0.25
  crash_rebuild_cap: 0.65
  trend_hold_floor: 0.45
  trend_hold_cap: 0.95

position_cap:
  start_drawdown: 0.10    # Start reducing cap at 10% drawdown
  full_drawdown: 0.30     # Reach min_cap at 30% drawdown
  min_cap: 0.40
  bull_min_cap: 0.65      # Higher minimum in bull regime
  bear_min_cap: 0.30      # Lower minimum in bear regime
```

## Reward Design

The reward function disables pure log-return (`return_scale: 0.0`) and relies on:

| Component | Coefficient | Purpose |
|-----------|-------------|---------|
| Kelly-alpha | alpha_scale=3.2 | Signal quality from rolling win rate / PF |
| Risk budget penalty | 0.25 | Penalize exceeding mode-owned band |
| Micro-rebalance penalty | 0.50 | Stop tiny position adjustments (<5%) |
| Episode drawdown penalty | 3.5 | Strong penalty for >10% intra-episode DD |
| Opportunity cost penalty | 0.20 | Penalize being too flat during rallies |
| Trend alignment bonus | 0.40 | Reward positions aligned with trend |

**Disabled**: activity_bonus, inactivity_penalty, return_scale (pure log)

## Best Known Result

From `risk_budget_usage_v1` (native PPO actor):

| Window | Return | Sharpe | MaxDD | AvgPos |
|--------|--------|--------|-------|--------|
| test_2020 | 40.9% | 1.61 | 23.7% | 0.27 |
| test_2022 | -17.3% | -1.21 | **18.7%** | **0.13** |
| test_2023_2024 | 82.8% | 1.44 | 16.3% | 0.30 |
| test_proxy | 274.4% | **0.94** | 32.5% | 0.27 |

**Promotion gate**: ✅ test_2022 MaxDD=18.7% (<30%), ✅ avg_pos=0.127 (<0.35), ✅ Sharpe=0.942 (>0.90)

## Common Pitfalls

1. **Replay vs Actor gap**: Fixed-replay composition (using already-executed daily paths) gives Sharpe=1.25, but native PPO training under this reward only achieves Sharpe=0.94. The replay is an *upper bound* — do not expect the actor to match it without supervised warm-start.

2. **Conservative policy**: PPO tends to learn conservative usage ratios. The avg_pos (0.27) is much lower than canonical balanced (0.52). This is good for risk control but may miss violent rebounds.

3. **Missing walk-forward**: The profile passed fixed-split gate but walk-forward validation is missing. Always run walk-forward before promoting.

## Next Experiments to Try

- **Supervised warm-start**: Pre-train actor on replay path decisions, then PPO fine-tune.
- **Dynamic band adjustment**: Let band width shrink in high-volatility regimes instead of fixed bands.
- **Multi-asset budget**: Allocate a single risk budget across multiple ETFs instead of per-asset.

## References

- Card: `runtime/model/cards/risk_budget_usage_v1/card.md`
- Experiment notes: `experiments/candidate/risk_budget_usage_v1/notes.md`
- Profile: `experiments/candidate/risk_budget_usage_v1/profile.yaml`
- Artifact: `artifacts/risk_budget_family/soxl_risk_budget_usage_v1_summary.json`
