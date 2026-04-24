---
title: risk_budget_usage_v3
direction: model
status: candidate
---

# risk_budget_usage_v3

## Motivation

v1 proved the budget-usage architecture works. v2 added mode-conditional reward
shaping and dynamic band scaling to guide PPO behavior. v3 goes one step
deeper: **let the Actor directly observe the current mode**, so it can learn
mode-specific usage policies end-to-end.

The hypothesis is that feeding mode as an explicit input feature (one-hot)
should let the network learn different usage distributions per mode faster
than inferring mode indirectly from raw features alone.

## External Reference

- Mode-conditioned policies in hierarchical RL (Sutton et al. 1999)
- Task-conditional policy networks (Hausman et al. 2018)
- Contextual policies with one-hot task identifiers

## Hypothesis

Appending the current mode as a one-hot vector to the observation should:
1. Reduce the sample complexity of learning mode-specific usage policies
2. Improve generalization across mode transitions
3. Lift proxy Sharpe by allowing the Actor to specialize per mode

## Target Slices / Windows

- `test_proxy`: Sharpe ≥ 0.95
- `test_2022`: MaxDD ≤ 18%, avg_pos ≤ 0.15
- `test_2023_2024`: return ≥ 80%

## Acceptance Criteria

- `test_proxy Sharpe >= 0.95`
- `test_2022 MaxDD <= 20.0%`
- `test_2022 avg_pos <= 0.15`

## Baseline

- **Canonical baseline**: `risk_budget_usage_v1`
- **Direct comparison baseline**: `risk_budget_usage_v2`
- **Experiment role**: `replace`
- **Changed layer**: `network` (observation input)

## Configuration Delta vs v2

```yaml
model:
  use_mode_conditioned_input: true  # NEW: append 4-dim mode one-hot to obs
```

All other configs identical to v2.

## Result

### Fixed-Split (Full Backtest with Saved Model)

Command:
```bash
PYTHONPATH=. qlib_venv/bin/python soxl_qlib_strategy_v2.py \
  --profiles risk_budget_usage_v3 --seed 42 --skip-data-refresh
```

| 切片 | Return | Sharpe | MaxDD | Trades |
|------|--------|--------|-------|--------|
| test (full) | **434.1%** | **0.87** | 47.5% | 295 |

> ⚠️ Proxy metrics from training eval were overly optimistic (Sharpe 1.074, MaxDD 21.4%).
> Actual backtest on saved model: Sharpe 0.87, MaxDD 47.5%. Significant discrepancy.

### Walk-Forward Validation

Command:
```bash
PYTHONPATH=. qlib_venv/bin/python soxl_qlib_strategy_v2.py \
  --profiles risk_budget_usage_v3 --seed 42 --skip-data-refresh --walk-forward
```

| Window | Train | Valid | Test | Return | Sharpe | MaxDD | Trades |
|--------|-------|-------|------|--------|--------|-------|--------|
| wf_2020 | 2010-03→2018-12 | 2019 | 2020 | 19.3% | 0.29 | 19.5% | 36 |
| wf_2022 | 2010-03→2020-12 | 2021 | 2022 | 59.9% | 0.39 | **59.0%** | 32 |
| wf_2023_2024 | 2010-03→2022-12 | 2023 | 2024 | 13.0% | 0.22 | 23.4% | 40 |

### Promotion Gate

- Fixed-split Sharpe: 0.87 (proxy was 1.074) — gap exposes overfit to eval slice
- Fixed-split MaxDD: 47.5% — far above acceptable threshold
- Walk-forward wf_2022 MaxDD: **59.0%** — catastrophic
- Walk-forward wf_2023_2024 Sharpe: 0.22 — no edge

**❌ PROMOTION BLOCKED.** Walk-forward generalization failed dramatically.

## 与 v1/v2 对比

| 指标 | v1 | v2 | v3 |
|------|-----|-----|-----|
| proxy Sharpe | 0.942 | 0.884 | **1.074** |
| proxy MaxDD | 32.5% | 20.6% | 21.4% |
| proxy Return | 274% | 158% | 275% |
| 2022 MaxDD | 18.7% | 7.4% | 12.0% |
| 2022 avg_pos | 0.127 | 0.070 | 0.143 |

## Decision

**❌ PROMOTION BLOCKED — Walk-forward generalization failed.**

v3's mode-conditioned actor input produced excellent proxy metrics (Sharpe 1.074) and passed the fixed-split gate on proxy eval, but:

1. **Proxy-to-backtest gap**: Proxy Sharpe 1.074 → Actual backtest Sharpe 0.87, MaxDD 21.4% → 47.5%. The eval metric was overly optimistic.
2. **Walk-forward collapse**: 
   - wf_2022 MaxDD = 59.0% (vs 12.0% proxy) — catastrophic regime failure
   - wf_2023_2024 Sharpe = 0.22 — no predictive edge in latest data
   - wf_2020 Sharpe = 0.29 — weak in calm markets
3. **The model overfit to the specific test period** and fails to generalize when trained on earlier data and tested on held-out years.

**关键洞察**: Mode-conditioned actor input improves fixed-split proxy metrics but does NOT solve generalization. The network likely memorizes mode-specific behavior that happens to work in the fixed test window but fails in truly novel regimes (especially 2022 crash, 2023-2024 chop).

**Next action**: Before iterating to v4, must diagnose:
- Is the mode classifier itself leaking future information into the observation?
- Is the budget_usage mapping too expressive, allowing the actor to overfit?
- Would stronger regularization (dropout, smaller network, entropy bonus) help?
- Should we test mode-conditioned input on a proven generalizing baseline first?

## Comparison-Table Row Key

`risk_budget_usage_v3`
