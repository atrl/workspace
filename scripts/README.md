# Scripts

Primary harness entrypoints:

- `build_backtest_dashboard.py`: rebuild the local static dashboard under `dashboard.html`.
- `check_dashboard_static_export.py`: validate the Vercel-ready dashboard static bundle and versioned data payloads.
- `check_harness.py`: repo-owned validation entrypoint that replaces the missing `pytest` dependency path.
- `checkpoint_review.py`: render a durable Markdown review from a saved summary JSON's selection history and gates.
- `build_runtime_index.py`: regenerate `runtime/index.json` for fast knowledge retrieval.
- `prune_artifacts.py`: generate and optionally apply the retained artifact manifest.
- `rank_counterfactual_variants.py`: sort fixed-actor counterfactual outputs by promotion-gate quality and write a Markdown ranking.

Operational utilities:

- `create_linear_soxl_issue.py`: issue/bootstrap helper for repo workflow.
- `fetch_3x_etf_data.py`: refresh external ETF source CSV inputs.

Validation order before starting a new training loop:

1. `qlib_venv/bin/python scripts/check_harness.py`
2. only then run a new training or replay command
