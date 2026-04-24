# risk_budget_usage_v3 Walk-Forward Failure

**Date**: 2026-04-24
**Profile**: `risk_budget_usage_v3`
**Direction**: network / execution overlay
**Status**: candidate (promotion blocked)

## Walk-Forward Results

| Window | Train | Valid | Test | Return | Sharpe | MaxDD | Trades |
|--------|-------|-------|------|--------|--------|-------|--------|
| wf_2020 | 2010-03→2018-12 | 2019 | 2020 | 19.3% | 0.29 | 19.5% | 36 |
| wf_2022 | 2010-03→2020-12 | 2021 | 2022 | 59.9% | 0.39 | **59.0%** | 32 |
| wf_2023_2024 | 2010-03→2022-12 | 2023 | 2024 | 13.0% | 0.22 | 23.4% | 40 |

Fixed-split backtest with saved model:
- Return: 434.1%, Sharpe: 0.87, MaxDD: 47.5%, Trades: 295

## Key Findings

### 1. Proxy-to-Backtest Gap
Proxy eval metrics during training (Sharpe 1.074, MaxDD 21.4%) were significantly more optimistic than the actual backtest on the saved model (Sharpe 0.87, MaxDD 47.5%). This means the proxy metric is NOT a reliable predictor of real backtest performance.

### 2. Walk-Forward Generalization Collapse
The model fails to generalize when trained on earlier data and tested on held-out years:
- wf_2022: MaxDD 59.0% — the 2022 crash completely breaks the strategy
- wf_2023_2024: Sharpe 0.22 — no edge in the most recent data
- wf_2020: Sharpe 0.29 — weak even in non-crash years

### 3. Mode-Conditioned Input Does Not Fix Generalization
While mode-conditioned actor input improved fixed-split proxy metrics, it did NOT solve the core generalization problem. The network likely memorizes mode-specific behavior that happens to work in the fixed test window but fails in truly novel regimes.

### 4. Risk Budget Usage Architecture May Be Too Expressive
The budget_usage mapping with delta_levels [-0.10, -0.05, 0.0, +0.05, +0.10] may give the actor too much flexibility, allowing it to overfit to specific patterns in the training/test window.

## Diagnosis Needed Before v4

1. **Mode info leak check**: Does the mode classifier use future information? Run mode classification on a pure walk-forward basis.
2. **Budget usage regularization**: Would a tighter delta range (e.g., ±0.03) or fewer levels reduce overfit?
3. **Network size**: GRU-64 + MLP 128→64 may be too large for the amount of data per mode.
4. **Baseline validation**: Test mode-conditioned input on a proven generalizing baseline (e.g., balanced_rule_residual_v19) first.
5. **Entropy collapse**: Check if entropy drops too low during training, indicating premature convergence.

## Comparison with v1/v2

| Metric | v1 | v2 | v3 |
|--------|-----|-----|-----|
| proxy Sharpe | 0.942 | 0.884 | 1.074 |
| actual Sharpe | ~0.9? | ~0.8? | **0.87** |
| proxy MaxDD | 32.5% | 20.6% | 21.4% |
| actual MaxDD | ~35%? | ~25%? | **47.5%** |
| wf_2022 MaxDD | ? | ? | **59.0%** |
| wf_2023_2024 Sharpe | ? | ? | **0.22** |

## Decision

**❌ PROMOTION BLOCKED.**

v3 is retained as a candidate for diagnosis purposes (keep_actor=true), but cannot be promoted to canonical. The mode-conditioned input mechanism is NOT validated for generalization.

## Next Action

Run diagnostic experiments before declaring v4:
1. Check mode classifier for forward-looking bias
2. Test mode-conditioned input on balanced_rule_residual_v19
3. Evaluate if smaller network or tighter usage bands help
