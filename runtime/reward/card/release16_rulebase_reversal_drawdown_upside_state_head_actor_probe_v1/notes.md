# release16_rulebase_reversal_drawdown_upside_state_head_actor_probe_v1

Role: constrained reward/profile probe from
`runtime/model_selection/card/retraining_execution_handoff_current_v1/retraining_execution_handoff_current_v1.ipynb`.

Baseline: `state_conditioned_drawdown_rebound_reward_train_v1`.

Changed layer: reward/profile configuration. The profile adds `rulebase_signals`
features and enables existing state-conditioned, drawdown-drift, and
trend-preservation reward shaping. It does not introduce a new actor retention
path.

Smoke validation:

```bash
PYTHONPATH=. /Users/rziiiii/dev/workspace/qlib_venv/bin/python -m src.soxl_quant.cli \
  --profiles release16_rulebase_reversal_drawdown_upside_state_head_actor_probe_v1 \
  --smoke --skip-data-refresh --seed 20260615
```

Observed output:
- Dataset: train `1965`, valid `252`, test `1618`, features `73`.
- Training loop completed `8/8` smoke iterations.
- Best validation: return `14.0%`, Sharpe `0.93`, MaxDD `15.0%`, score `26.24`.
- Test replay: return `291.0%`, Sharpe `1.05`, MaxDD `25.3%`, trades `251`.
- Promotion: `blocked`.

Retention note: the smoke actor output was discarded. This profile may be used
for a full constrained actor probe, but actor retention remains blocked until
fixed-split, `wf_2020`, `wf_2022`, `wf_2023_2024`, YTD SOXLBacktest/current-best
alignment, model replay export, and dashboard runnable-source gates all pass.
