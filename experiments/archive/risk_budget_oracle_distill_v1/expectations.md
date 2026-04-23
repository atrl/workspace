# Expectations: risk_budget_oracle_distill_v1

## Closest Historical Success

`risk_budget_usage_v1` fixed replay: `test_proxy 2003.6% / Sharpe 1.248 /
MaxDD 45.0%`, with `test_2022 MaxDD 25.6%`.

## Closest Historical Failure

`risk_budget_bc_warmstart_v1`: train-split BC reached high accuracy but had no
action-4 full-usage labels, selected iter 0, and only reached `test_proxy
250.7%`.

## Why This Is Not A Duplicate

This is not another reward/cap/PPO branch. It is a no-promotion upper-bound
capacity diagnostic using the full replay daily path as labels.

## Validation Commands

```bash
PYTHONPATH=. qlib_venv/bin/python scripts/build_budget_usage_oracle_actor.py --profile risk_budget_oracle_distill_v1 --skip-data-refresh
PYTHONPATH=. qlib_venv/bin/python scripts/check_harness.py
```
