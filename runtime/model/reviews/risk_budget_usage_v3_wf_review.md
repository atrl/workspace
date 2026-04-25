# risk_budget_usage_v3 Walk-Forward Review

**Date**: 2026-04-24
**Profile**: `risk_budget_usage_v3`
**Direction**: network / execution overlay
**Changed layer**: observation input (+4-dim mode one-hot)
**Baseline**: `risk_budget_usage_v1`
**Direct comparison**: `risk_budget_usage_v2`

## Commands

```bash
# Fixed-split
PYTHONPATH=. qlib_venv/bin/python soxl_qlib_strategy_v2.py \
  --profiles risk_budget_usage_v3 --seed 42 --skip-data-refresh

# Walk-forward
PYTHONPATH=. qlib_venv/bin/python soxl_qlib_strategy_v2.py \
  --profiles risk_budget_usage_v3 --seed 42 --skip-data-refresh --walk-forward
```

## Results

### Fixed-Split (Saved Model Backtest)

| Metric | Value |
|--------|-------|
| Return | 434.1% |
| Sharpe | 0.87 |
| MaxDD | 47.5% |
| Trades | 295 |

> Proxy eval during training: Sharpe 1.074, MaxDD 21.4%. **Proxy-to-backtest gap confirmed.**

### Walk-Forward

| Window | Train | Valid | Test | Return | Sharpe | MaxDD | Trades |
|--------|-------|-------|------|--------|--------|-------|--------|
| wf_2020 | 2010-03→2018-12 | 2019 | 2020 | 19.3% | 0.29 | 19.5% | 36 |
| wf_2022 | 2010-03→2020-12 | 2021 | 2022 | 59.9% | 0.39 | **59.0%** | 32 |
| wf_2023_2024 | 2010-03→2022-12 | 2023 | 2024 | 13.0% | 0.22 | 23.4% | 40 |

## Diagnosis

### 1. Mode Classifier Leak Check: NEGATIVE

Investigated `src/soxl_quant/policy/modes.py` and feature pipeline (`src/soxl_quant/data.py`).

- Rule-based classifier (`classify_rule_mode`) uses only current-row features.
- All rolling windows are strictly backward-looking; no `center=True` found.
- Mode is computed **before** next-day return is applied in both training env and backtest.

**Verdict**: No forward-looking leak at inference time. The walk-forward collapse is NOT caused by mode info leak.

### 2. Proxy-to-Backtest Gap

| | Proxy Eval | Actual Backtest | Delta |
|---|---|---|---|
| Sharpe | 1.074 | 0.87 | -0.20 |
| MaxDD | 21.4% | 47.5% | +26.1pp |

The proxy metric (training eval reward) does not correlate with actual backtest performance. This suggests the reward function or eval episode sampling is not representative of real execution.

### 3. Walk-Forward Collapse

- **wf_2022 MaxDD 59.0%**: The 2022 crash completely breaks the strategy. Mode-conditioned input does not help the actor navigate an unprecedented regime.
- **wf_2023_2024 Sharpe 0.22**: No edge in chop/trend-mixed markets.
- **wf_2020 Sharpe 0.29**: Weak even in non-crash years.

The model overfit to the specific fixed test period. Mode one-hot gives the actor enough expressiveness to memorize period-specific patterns.

## Comparison with v1/v2

| Metric | v1 | v2 | v3 |
|--------|-----|-----|-----|
| proxy Sharpe | 0.942 | 0.884 | 1.074 |
| actual Sharpe | — | — | **0.87** |
| proxy MaxDD | 32.5% | 20.6% | 21.4% |
| actual MaxDD | — | — | **47.5%** |
| wf_2022 MaxDD | — | — | **59.0%** |
| wf_2023_2024 Sharpe | — | — | **0.22** |

## Decision

**❌ PROMOTION BLOCKED.**

Mode-conditioned actor input improves proxy metrics but does not generalize. The mechanism is not validated for production use.

## Next Actions

1. **Test on proven baseline**: Apply mode-conditioned input to `balanced_rule_residual_v19` (a generalizing baseline) to isolate whether the problem is the mechanism or the `budget_usage` architecture.
2. **Budget usage regularization**: Tighter `delta_levels` (e.g., ±0.03) or fewer levels (3-action instead of 5).
3. **Entropy monitoring**: Check if entropy collapses during training, indicating premature convergence.
4. **Proxy metric fix**: Investigate why proxy eval is so inflated vs backtest.

## Links

- Experiment card: `experiments/candidate/risk_budget_usage_v3/notes.md`
- Memory record: `memory/2026-04-24-risk-budget-v3-wf-failure.md`
- Profile: `experiments/candidate/risk_budget_usage_v3/profile.yaml`
- Landscape review: `docs/experiment-landscape-review.md`
