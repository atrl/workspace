# Paper Model OOS Risk Check

Date: 2026-05-24

Goal:

- Identify the model currently used by Futu paper trading.
- Confirm the training data boundary.
- Recompose the current paper-trading model on strict post-train data and
  compare strategy return, buy-and-hold return, excess, drawdown, and recent
  risk.

Current paper model:

```text
broker: futu
composer profile: v19reg_v22_budget_usage_bull_guard_v1
slow actor: balanced_rule_residual_v19_reg
fast actor: balanced_rule_residual_v22
latest dry-run summary: runtime/soxl/paper_trading/2026-05-22/dry_run/futu/summary.json
latest signal date: 2026-05-22
latest raw target: 0.8500000003
latest risk-gate capped target: 0.35
```

Training/data split:

```text
train: 2010-03-11 .. 2018-12-31
valid: 2019-01-01 .. 2019-12-31
test:  2020-01-01 .. 2026-12-31
latest local data date: 2026-05-22
safe backtest end: 2026-05-21
```

The strict risk conclusion uses `2020-01-02 .. 2026-05-21`, excluding both the
training split and the 2019 validation/model-selection split.

Command:

```bash
PYTHONPATH=src:. qlib_venv/bin/python scripts/compose_budget_usage_counterfactual.py \
  --slow runtime/model/card/balanced_rule_residual_v19_reg/balanced_rule_residual_v19_reg_soxl_daily.csv \
  --fast runtime/model/card/balanced_rule_residual_v22/balanced_rule_residual_v22_soxl_daily.csv \
  --output-dir runtime/production/card/paper_model_oos_risk_check_v1 \
  --profile paper_model_oos_risk_check_v1 \
  --slow-label balanced_rule_residual_v19_reg_current \
  --fast-label balanced_rule_residual_v22_current \
  --composed-label v19reg_v22_budget_usage_bull_guard_v1_current \
  --description "Current Futu paper-trading composer OOS risk check from strict post-train test data." \
  --bull-participation-guard --bull-ma-window 50 --bull-momentum-window 20 \
  --bull-high-window 50 --bull-min-periods 20 --bull-min-momentum 0.03 \
  --bull-max-drawdown 0.08
```

Strict OOS result:

| window | strategy | buy-hold | excess | sharpe | max_dd | avg_pos | trades |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `2020-01-02..2026-05-21` | `1542.89%` | `816.51%` | `+726.38pp` | `1.109` | `41.59%` | `0.462` | `199` |
| `2020` | `154.60%` | `59.78%` | `+94.82pp` | `1.840` | `25.27%` | `0.536` | `30` |
| `2022` | `-21.57%` | `-86.59%` | `+65.02pp` | `-0.440` | `31.10%` | `0.248` | `26` |
| `2023-2024` | `209.14%` | `191.77%` | `+17.37pp` | `1.453` | `30.67%` | `0.497` | `70` |
| `2025..2026-05-21` | `98.45%` | `544.71%` | `-446.25pp` | `1.155` | `35.01%` | `0.434` | `39` |
| `2026 YTD..2026-05-21` | `16.70%` | `277.62%` | `-260.93pp` | `1.147` | `20.57%` | `0.256` | `5` |

Component comparison on strict OOS:

| comparison | delta_return | delta_sharpe | delta_max_dd | delta_avg_pos | delta_trades |
| --- | ---: | ---: | ---: | ---: | ---: |
| composed vs `balanced_rule_residual_v22` | `+401.36pp` | `+0.134` | `-11.63pp` | `-0.039` | `-1238` |
| composed vs `balanced_rule_residual_v19_reg` | `-124.91pp` | `-0.042` | `-1.95pp` | `+0.012` | `-1231` |

Risk read:

- The model is not failing the full strict-OOS excess check: absolute return and
  cumulative excess are positive after training and validation.
- The main risk is recent upside-capture/opportunity-cost risk. The model
  materially lagged SOXL buy-and-hold in `2025..2026-05-21`, especially in 2026
  YTD.
- The 2022 stress-defense behavior remains the strongest point: the strategy
  lost far less than SOXL buy-and-hold and kept average position near `0.25`.
- Max drawdown is still high at `41.59%`; this is lower than the current fast
  v22 component but not low enough to call the paper model low-risk.
- The 2026-05-22 paper signal already flipped back to a raw `0.85` target with
  a risk-gate cap at `0.35`, so the immediate production risk is execution and
  reconciliation discipline around re-entry rather than stale model selection.

Decision:

- Keep current Futu paper model as the paper-trading candidate.
- Treat recent underperformance versus buy-and-hold as a monitored risk, not as
  an immediate model removal trigger.
- Before live IBKR promotion, add a recovery/upside-capture review focused on
  2025-2026 and the post-drawdown re-entry path.

Artifacts:

- `paper_model_oos_risk_check_v1.json`
- `paper_model_oos_risk_check_v1.md`
- `paper_model_oos_risk_check_v1_soxl_daily.csv`
- `segment_metrics.json`
