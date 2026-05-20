# Scripts

Primary harness entrypoints:

- `build_backtest_dashboard.py`: rebuild the local static dashboard under `dashboard.html`.
- `check_dashboard_static_export.py`: validate the Vercel-ready dashboard static bundle and versioned data payloads.
- `check_harness.py`: repo-owned validation entrypoint that replaces the missing `pytest` dependency path.
- `checkpoint_review.py`: render a durable Markdown review from a saved summary JSON's selection history and gates.
- `execution_overlay_generalization.py`: evaluate execution overlay variants against the retained balanced actor.
- `generalization_backtest.py`: run the retained cross-ETF generalization backtest harness.
- `feature_analysis.py`: generate feature importance / IC analysis outputs under `runtime/factor/`.
- `cost_sensitivity_backtest.py`: replay retained artifacts under alternative cost assumptions for review.
- `export_profile_daily_backtest.py`: export daily replay rows from saved model artifacts for review diffs.
- `build_runtime_index.py`: regenerate `runtime/index.json` for fast knowledge retrieval.
- `prune_artifacts.py`: generate and optionally apply the retained artifact manifest.
- `rank_counterfactual_variants.py`: sort fixed-actor counterfactual outputs by promotion-gate quality and write a Markdown ranking.

Operational utilities:

- `create_linear_soxl_issue.py`: issue/bootstrap helper for repo workflow.
- `fetch_3x_etf_data.py`: refresh external ETF source CSV inputs.
- `position_cap_counterfactual.py`: fixed-actor counterfactual sweep for cap-routing review.

Validation order before starting a new training loop:

1. `qlib_venv/bin/python scripts/check_harness.py`
2. only then run a new training or replay command
