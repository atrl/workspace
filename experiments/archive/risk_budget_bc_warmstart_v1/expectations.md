# Expectations

## Closest Historical Success

`risk_budget_usage_v1` fixed replay: `test_proxy 2003.6% / Sharpe 1.248 /
MaxDD 45.0%`, with strict `test_2022 MaxDD 25.6%`.

## Closest Historical Failure

Native `risk_budget_usage_v1` PPO: fixed-split risk passed, but the actor
learned a conservative policy and only reached `test_proxy 274.4%`.

## Why This Is Not A Duplicate

This changes the optimization path, not another reward/cap coefficient. The new
mechanism is supervised behavior cloning from train-split fast-expert budget
usage before PPO.

## Validation Commands

```bash
PYTHONPATH=. qlib_venv/bin/python scripts/build_budget_usage_bc_actor.py --profile risk_budget_bc_warmstart_v1 --skip-data-refresh
PYTHONPATH=. qlib_venv/bin/python soxl_qlib_strategy_v2.py --profiles risk_budget_bc_warmstart_v1 --skip-data-refresh --seed 20260411
PYTHONPATH=. qlib_venv/bin/python scripts/check_harness.py
```
