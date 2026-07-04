# Failure Modes

- data_status: ok_partial_insufficient_for_full_walk_forward
- earliest_complete_symbol_regular_session: 2025-01-02 09:45:00
- missing_required_history_years: [2019, 2020, 2021, 2022, 2023, 2024]
- provider_blocker_status_counts: {'blocked_no_rows': 1, 'blocked_provider_history': 4, 'blocked_provider_rate_limit': 8}
- next_data_action: upgrade_polygon_history_plan_or_load_alternate_15min_source
- blocked_missing_api_key: POLYGON_API_KEY was absent; no external minute context can be trusted.
- blocked_provider_rate_limit: provider throttled the request; rerun with backoff or a paid quota window.
- blocked_provider_history: provider plan did not authorize the requested range or symbol.
- ok_partial_insufficient_for_full_walk_forward: some data was built, but it is not enough to retrain the full 2019-2026 walk-forward external context model.
- no fake breadth columns are created when QQQ/SMH/constituent rows are missing.
