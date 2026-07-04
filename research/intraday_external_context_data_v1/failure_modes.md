# Failure Modes

- data_status: ok_partial_insufficient_for_full_walk_forward
- blocked_missing_api_key: POLYGON_API_KEY was absent; no external minute context can be trusted.
- blocked_provider_rate_limit: provider throttled the request; rerun with backoff or a paid quota window.
- blocked_provider_history: provider plan did not authorize the requested range or symbol.
- ok_partial_insufficient_for_full_walk_forward: some data was built, but it is not enough to retrain the full 2019-2026 walk-forward external context model.
- no fake breadth columns are created when QQQ/SMH/constituent rows are missing.
