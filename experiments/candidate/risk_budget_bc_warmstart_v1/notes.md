# risk_budget_bc_warmstart_v1

## Motivation

`risk_budget_usage_v1` proved that the runtime `budget_usage` mapping is valid,
but PPO trained from scratch discovered a conservative low-usage policy:
`test_proxy 274.4% / Sharpe 0.942 / MaxDD 32.5%`. The earlier fixed replay of
the same mechanism reached `test_proxy 2003.6% / Sharpe 1.248 / MaxDD 45.0%`,
which means the remaining gap is primarily behavioral learning, not mapping
wiring.

## External Reference

- Behavior Cloning / DAGGER-style policy initialization: use supervised
  imitation to compress the action search space before policy-gradient updates.
- Prior repo evidence: Kelly/reward shaping attempts repeatedly collapsed to
  conservative low-position policies when the policy had to discover dynamic
  participation from sparse PPO feedback alone.

## Hypothesis

If the actor first learns the fast expert's budget-usage distribution on the
training split, then short low-lr PPO fine-tuning should preserve more replay
participation than `risk_budget_usage_v1` while retaining strict `2022` risk
control.

## Target Slices / Windows

- Preserve `test_2022 MaxDD <= 30.0%` and `avg_pos <= 0.35`.
- Improve `test_proxy return` versus native `risk_budget_usage_v1` (`274.4%`).
- Preserve `test_proxy Sharpe >= 0.90`.
- Improve `test_2023_2024 return` versus native `risk_budget_usage_v1` (`82.8%`).

## Acceptance Criteria

- BC pretrain uses train split only; no 2020+ replay daily labels are used for
  supervised training.
- BC checkpoint has non-degenerate label distribution and accuracy evidence.
- PPO output beats native `risk_budget_usage_v1` on proxy return without failing
  the strict fixed-split promotion checks.

## Expected Trade-Off

Some `test_proxy` MaxDD can rise relative to native `risk_budget_usage_v1` if it
buys back meaningful upside, but `test_2022 MaxDD > 30.0%` invalidates the run.

## Failure Mode

If BC actor remains too conservative, the fast expert labels are not expressive
enough under train-split zero-leakage reconstruction. If PPO quickly erases BC
participation, the next iteration should freeze lower GRU layers or stop after
BC rather than add more reward pressure.

## Comparison-Table Row Key

`risk_budget_bc_warmstart_v1`

## Result

BC command:

```bash
PYTHONPATH=. /Users/rziiiii/.openclaw/workspace/qlib_venv/bin/python scripts/build_budget_usage_bc_actor.py \
  --profile risk_budget_bc_warmstart_v1 \
  --seed 20260411
```

PPO command:

```bash
PYTHONPATH=. /Users/rziiiii/.openclaw/workspace/qlib_venv/bin/python soxl_qlib_strategy_v2.py \
  --profiles risk_budget_bc_warmstart_v1 \
  --skip-data-refresh \
  --seed 20260411
```

BC dataset:

- samples: `1904`
- label distribution: `{1: 422, 2: 710, 3: 772}`
- no action-4 full-usage labels were generated from the train-split expert
  reconstruction
- final BC loss: `0.950989`
- final BC accuracy: `0.960938`

BC / selected actor metrics:

- valid: `21.1% / Sharpe 0.959 / MaxDD 25.0% / avg_pos 0.384`
- `test_2020`: `36.0% / Sharpe 1.493 / MaxDD 20.5% / avg_pos 0.296`
- `test_2022`: `-5.0% / Sharpe -0.345 / MaxDD 10.7% / avg_pos 0.108`
- `test_2023_2024`: `49.0% / Sharpe 1.010 / MaxDD 20.1% / avg_pos 0.273`
- `test_proxy`: `250.7% / Sharpe 0.911 / MaxDD 27.3% / avg_pos 0.266`

PPO selection history:

- iter 0: score `37.23`, valid `21.1% / Sharpe 0.959 / MaxDD 25.0%`
- iter 5: score `30.84`, valid `17.5% / Sharpe 0.816 / MaxDD 25.0%`
- iter 10: score `30.84`, valid `17.5% / Sharpe 0.816 / MaxDD 25.0%`

## Decision

Drop as a promotable candidate. Fixed-split risk passed, but proxy return
`250.7%` did not beat native `risk_budget_usage_v1` (`274.4%`) and remained far
below fixed replay (`2003.6%`). PPO also degraded the BC initialization, so
checkpoint selection kept iter 0.

The important diagnostic is that train-split expert reconstruction never
produced full-usage action-4 labels. The replay upside likely came from the
specific held-out daily-path composition and mode path timing, not from a
learnable high-usage distribution available in the training split.

## Next

Do not run more PPO fine-tuning on this BC source. The next useful diagnostic is
either:

- a clearly marked no-promotion upper-bound distillation from the full replay
  daily path to measure representational capacity, or
- move away from single-actor PPO toward an explicit runtime policy composer
  where the mode layer and fast participation layer remain separate auditable
  policies.
