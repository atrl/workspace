---
date: 2026-04-23
title: Risk-budget usage replay candidate
kind: review
profiles:
- balanced_rule_residual_imitation_v4
- balanced_rule_residual_imitation_v4_soxl_daily
- balanced_rule_residual_mode_v1_soxl_daily
- soxl
status: reference
tags:
- balanced_rule_residual_imitation_v4
- balanced_rule_residual_imitation_v4_soxl_daily
- balanced_rule_residual_mode_v1_soxl_daily
- checkpoint-selection
- eval-2022
- reference
- review
- soxl
artifacts:
- runtime/backtest
- runtime/post_train_review/balanced_rule_residual_imitation_v4_soxl_daily.csv
- runtime/post_train_review/balanced_rule_residual_mode_v1_soxl_daily.csv
- runtime/post_train_review/risk_budget_usage_v1.json
- runtime/post_train_review/risk_budget_usage_v1.md
- runtime/post_train_review/risk_budget_usage_v1_soxl_daily.csv
workpad: codex_exec_plans/active/2026-04-10-soxl-agent-loop.md
summary: 'Fixed replay treated imitation_v4 as a budget usage ratio inside mode_v1
  risk bands. It passed all replay acceptance checks: test_2022 MaxDD 25.6%, avg_pos
  0.214, test_proxy 2003.6% / Sharpe 1.248 / MaxDD 45.0%. Keep as candidate fixed-replay
  mechanism; next step must implement runtime/backtest parity before any promotion
  claim.'
---

# Risk-budget usage replay candidate

## Context

The prior `two_speed_mode_budget_v1` composition used `mode_v1` as a slow anchor
and let `imitation_v4` participate above that anchor by mode slack. It improved
proxy quality but still failed strict `test_2022 MaxDD <= 30%`.

`risk_budget_usage_v1` changes the interpretation: `imitation_v4` is no longer a
target position. It is a usage ratio inside mode-owned risk bands.

## Evidence

Command:

```bash
PYTHONPATH=. /Users/rziiiii/.openclaw/workspace/qlib_venv/bin/python scripts/replay_two_speed_mode_budget.py \
  --slow-daily /Users/rziiiii/.openclaw/workspace/runtime/post_train_review/balanced_rule_residual_mode_v1_soxl_daily.csv \
  --fast-daily /Users/rziiiii/.openclaw/workspace/runtime/post_train_review/balanced_rule_residual_imitation_v4_soxl_daily.csv \
  --composition budget_usage \
  --profile-name risk_budget_usage_v1 \
  --output-json runtime/post_train_review/risk_budget_usage_v1.json \
  --output-md runtime/post_train_review/risk_budget_usage_v1.md \
  --output-daily runtime/post_train_review/risk_budget_usage_v1_soxl_daily.csv
```

Metrics:

- `test_2020`: `239.0% / Sharpe 2.285 / MaxDD 21.7% / avg_pos 0.490`.
- `test_2022`: `-17.1% / Sharpe -0.358 / MaxDD 25.6% / avg_pos 0.214`.
- `test_2023_2024`: `127.8% / Sharpe 1.278 / MaxDD 32.1% / avg_pos 0.389`.
- `test_proxy`: `2003.6% / Sharpe 1.248 / MaxDD 45.0% / trades 188`.

## Decision

Keep as a candidate fixed-replay mechanism.

Do not promote yet. This is not a full actor/runtime validation. The next step
must implement the same budget-usage composition inside runtime/backtest
positioning and compare against this daily replay.

## Next

Implement runtime/backtest parity:

- mode layer owns risk bands
- actor/RL output is interpreted as budget usage ratio
- validation must reproduce the replay direction on fixed split and test proxy
  before any training branch is justified

## Runtime Parity Follow-Up

Implemented the profile/runtime wiring for the same mechanism:

- `ModelConfig.position_mapping` supports `default` and `budget_usage`.
- `budget_usage` is threaded through `PositionAction`, `evaluate_split`,
  `backtest_profile`, runtime signal generation, and the daily backtest export
  script.
- `experiments/candidate/risk_budget_usage_v1/profile.yaml` owns the replay mode
  bands and is registered in `configs/soxl/profile_registry.yaml`.
- Hybrid actors derive budget usage from `delta_levels`, so the 5-action
  `imitation_v4` actor maps to full 0..1 usage.

Validation:

```bash
PYTHONPATH=. /Users/rziiiii/.openclaw/workspace/qlib_venv/bin/python scripts/check_harness.py --skip-compile
# [summary] passed=49 failures=0

PYTHONPATH=. /Users/rziiiii/.openclaw/workspace/qlib_venv/bin/python - <<'PY'
from src.soxl_quant.config import load_profile
p = load_profile('risk_budget_usage_v1')
assert p.model.position_mapping == 'budget_usage'
assert p.mode_router.enabled is True
print(p.name, p.model.position_mapping, len(p.model.delta_levels), p.mode_router.enabled)
PY
# risk_budget_usage_v1 budget_usage 5 True
```

Blocker: the current artifact inventory retains
`soxl_balanced_rule_residual_imitation_v4_summary.json` but not
`soxl_balanced_rule_residual_imitation_v4_actor.pt`, so full actor-backed parity
must wait for actor restoration or regeneration.

## Actor-Backed Follow-Up

The missing `imitation_v4` actor was regenerated with seed `20260411`, then
evaluated under `risk_budget_usage_v1`.

Evidence:

- regenerated `imitation_v4`: `test_proxy 586.5% / Sharpe 0.924 / MaxDD 50.9%`;
  `test_2022 MaxDD 30.2%`.
- same actor reinterpreted as budget usage: `test_proxy 77.5% / Sharpe 0.647 /
  MaxDD 18.9% / avg_pos 0.141`.
- native `risk_budget_usage_v1` PPO actor: `test_proxy 274.4% / Sharpe 0.942 /
  MaxDD 32.5% / avg_pos 0.269`; `test_2022 MaxDD 18.7%, avg_pos 0.127`.

Decision: do not promote actor-backed `risk_budget_usage_v1`. The runtime
mapping works, but PPO under portfolio reward learns a conservative usage scale.
Next branch should use supervised budget-usage warm start / behavior cloning from
the replay daily path before PPO fine-tuning.

## BC Warm-Start Follow-Up

`risk_budget_bc_warmstart_v1` built train-split supervised labels from the
retained `balanced_rule_residual_imitation_v4` actor, then fine-tuned with low-lr
PPO.

Evidence:

- BC samples: `1904`; label distribution `{1: 422, 2: 710, 3: 772}`.
- No action-4 full-usage labels appeared in the train-split reconstruction.
- BC accuracy: `0.961`, loss `0.951`.
- PPO selected iter 0; iter 5/10 lowered valid score from `37.23` to `30.84`.
- final selected `test_proxy`: `250.7% / Sharpe 0.911 / MaxDD 27.3% / avg_pos
  0.266`.
- final selected `test_2022`: `MaxDD 10.7% / avg_pos 0.108`.

Decision: drop as promotable. The branch controlled risk but did not beat native
`risk_budget_usage_v1` (`274.4% proxy return`). The key diagnostic is that the
train-split fast expert did not provide the high-usage labels needed to reproduce
the fixed replay upside.
