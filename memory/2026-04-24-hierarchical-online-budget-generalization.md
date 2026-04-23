---
date: '2026-04-24'
title: Hierarchical online budget runtime generalization
kind: review
profiles:
- balanced_rule_residual_imitation_v4
- soxl
status: dropped
tags:
- balanced_rule_residual_imitation_v4
- dropped
- eval-2022
- review
- soxl
artifacts: []
workpad: codex_exec_plans/active/2026-04-10-soxl-agent-loop.md
summary: Implemented a deployable hierarchical online composer with explicit slow
  mode refresh above the retained imitation_v4 actor. Rule-router variants improved
  safety modestly, and learned-router variants lifted proxy return to 102-415 percent
  across the 3x ETF set, but all remained materially below fixed replay and the learned
  variants spent more 2022 drawdown budget. Keep the composer as a runtime diagnostic,
  not a production promotion.
---

# Hierarchical online budget runtime generalization

## Context

The fixed `risk_budget_usage_v1` replay is still much stronger than any single
actor/runtime form. This loop tested whether making the runtime explicitly
hierarchical would recover part of that gap without another training branch.

The fast layer stays fixed as the retained `balanced_rule_residual_imitation_v4`
actor. The new deployable layer is:

- slow mode refresh state machine
- configurable cadence (`1d` or `5d`)
- rule or learned router source
- budget-usage position mapping inside the active mode band

## Evidence

Commands:

```bash
PYTHONPATH=. qlib_venv/bin/python tests/test_hierarchical_composer.py
PYTHONPATH=. qlib_venv/bin/python -m py_compile src/soxl_quant/policy/hierarchical_composer.py scripts/hierarchical_online_budget_generalization.py
PYTHONPATH=. qlib_venv/bin/python scripts/hierarchical_online_budget_generalization.py
```

Key results:

- SOXL:
  - `actor_runtime`: `178.2% / Sharpe 0.969 / MaxDD 22.6%`
  - `hier_rule_slow5`: `123.3% / 0.877 / 22.8%`
  - `hier_learned_daily`: `415.2% / 1.082 / 29.1%`
- TQQQ:
  - `actor_runtime`: `56.4% / 0.521 / 23.4%`
  - `hier_learned_slow5`: `214.4% / 0.990 / 25.9%`
- UPRO:
  - `actor_runtime`: `39.0% / 0.484 / 22.4%`
  - `hier_learned_slow5`: `131.9% / 0.929 / 18.3%`
- SPXL:
  - `actor_runtime`: `24.2% / 0.339 / 23.3%`
  - `hier_rule_slow5`: `68.8% / 0.758 / 13.8%`
  - `hier_learned_slow5`: `134.3% / 0.952 / 17.2%`
- TECL:
  - `actor_runtime`: `51.4% / 0.476 / 25.5%`
  - `hier_learned_slow5`: `201.1% / 0.896 / 31.4%`

`2022` drawdown trade-off:

- rule-router hierarchy was usually safer than the current runtime, but only
  improved return modestly
- learned-router hierarchy recovered more participation, but often widened
  `2022` drawdown relative to the current runtime

## Decision

Keep:

- `src/soxl_quant/policy/hierarchical_composer.py`
- `scripts/hierarchical_online_budget_generalization.py`
- the new diagnostic artifacts and daily exports

Do not promote any current hierarchical runtime. The hierarchy helps, which
means the deployment shape matters, but the retained fast actor still does not
carry enough usage policy to approach the fixed replay edge.

## Next

If this line continues, train or distill the fast actor specifically under the
learned-router regime. Do not spend another loop only on refresh cadence or
router threshold drift.
