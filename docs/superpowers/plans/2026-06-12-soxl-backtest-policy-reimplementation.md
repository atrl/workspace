# SOXL Backtest Policy Reimplementation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reimplement the legacy non-qlib `soxl_backtest` rule policy in the current SOXL harness and validate that local inference has close trade timing and returns.

**Architecture:** Add a deterministic policy module that rebuilds legacy-compatible features from existing local SOXL/VIX data, replays the same stateful buy/sell rules without importing the legacy `soxl_backtest` package, and writes notebook-backed comparison evidence. Legacy `SOXLBacktest` remains only the oracle used by tests and the validation notebook.

**Tech Stack:** Python 3.11, pandas, numpy, pytest, nbconvert, existing `qlib_venv` interpreter at `/Users/rziiiii/dev/workspace/qlib_venv/bin/python`.

---

### Task 1: Feature Parity Tests

**Files:**
- Create: `tests/test_legacy_soxl_policy.py`
- Later create: `src/soxl_quant/policy/legacy_soxl.py`

- [ ] **Step 1: Write failing feature parity tests**

Add tests that import `build_legacy_soxl_feature_frame` from `src.soxl_quant.policy.legacy_soxl`, instantiate legacy `SOXLBacktest` over `2026-01-01` to `2026-06-09`, and compare columns `RSI6`, `MA20`, `HV_20`, `HV_30`, `VOL_COEF`, and `panic_score` on overlapping rows.

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
uv run --python /Users/rziiiii/dev/workspace/qlib_venv/bin/python --with pytest pytest tests/test_legacy_soxl_policy.py::test_legacy_feature_builder_matches_non_qlib_soxl_backtest -q
```

Expected: fail with missing module or missing `build_legacy_soxl_feature_frame`.

- [ ] **Step 3: Implement feature builder**

Create `src/soxl_quant/policy/legacy_soxl.py` with:

- `LegacySoxlConfig`
- `build_legacy_soxl_source_frame`
- `build_legacy_soxl_feature_frame`
- `_calculate_rsi`
- `_calculate_panic_score`

The feature builder must use local `data/raw/symbols/SOXL.csv` and
`data/processed/panels/soxl_soxx_vix_full.csv`, reproduce the actual
`SOXLBacktest.__init__` ordering that filters to `START_DATE` before calculating
indicators, use the 180-day `MA20`, and reproduce the legacy sigmoid panic-score
formula.

- [ ] **Step 4: Run feature tests**

Run:

```bash
uv run --python /Users/rziiiii/dev/workspace/qlib_venv/bin/python --with pytest pytest tests/test_legacy_soxl_policy.py::test_legacy_feature_builder_matches_non_qlib_soxl_backtest -q
```

Expected: pass.

### Task 2: Stateful Rule Replay Tests

**Files:**
- Modify: `tests/test_legacy_soxl_policy.py`
- Modify: `src/soxl_quant/policy/legacy_soxl.py`

- [ ] **Step 1: Write failing replay tests**

Add tests for `replay_legacy_soxl_policy`:

- YTD fresh start from `2026-01-01` to `2026-06-09` has return close to `132.08%`.
- YTD trade dates include `2026-03-30`, `2026-05-27`, and `2026-06-05`.
- June fresh start from `2026-06-01` to `2026-06-09` has return close to `3.14%`.
- June trade dates include `2026-06-05`.

- [ ] **Step 2: Run replay tests to verify failure**

Run:

```bash
uv run --python /Users/rziiiii/dev/workspace/qlib_venv/bin/python --with pytest pytest tests/test_legacy_soxl_policy.py::test_legacy_policy_replay_matches_ytd_trade_timing_and_return tests/test_legacy_soxl_policy.py::test_legacy_policy_replay_matches_june_fresh_start -q
```

Expected: fail with missing `replay_legacy_soxl_policy`.

- [ ] **Step 3: Implement deterministic replay**

Extend `legacy_soxl.py` with:

- `LegacySoxlTrade`
- `LegacySoxlReplayResult`
- `LegacySoxlPolicyState`
- `LegacySoxlRulePolicy`
- `replay_legacy_soxl_policy`
- metric helpers for return, drawdown, average position, end position, and trade count.

The replay must implement the non-qlib buy/sell rules without importing
`soxl_backtest`: initial build, panic gates, daily and cumulative drop triggers,
cooldowns, overvalued buy-step reduction, tranche stack, take-profit, stop-loss,
last-tranche sell, integer share sizing, and daily account/NAV history.

- [ ] **Step 4: Run replay tests**

Run:

```bash
uv run --python /Users/rziiiii/dev/workspace/qlib_venv/bin/python --with pytest pytest tests/test_legacy_soxl_policy.py -q
```

Expected: all tests in the new file pass.

### Task 3: Runtime Export Script

**Files:**
- Create: `scripts/export_legacy_soxl_policy_replay.py`
- Modify: `tests/test_legacy_soxl_policy.py`

- [ ] **Step 1: Write failing CLI/export test**

Add a test that calls the export helper or subprocess with an output directory
under `tmp_path`, then verifies `new_policy_daily.csv`, `new_policy_trades.csv`,
`legacy_comparison_metrics.csv`, `trade_timing_diagnostics.csv`, and
`summary.json` are written.

- [ ] **Step 2: Run export test to verify failure**

Run:

```bash
uv run --python /Users/rziiiii/dev/workspace/qlib_venv/bin/python --with pytest pytest tests/test_legacy_soxl_policy.py::test_export_legacy_soxl_policy_replay_writes_sidecars -q
```

Expected: fail with missing script/helper.

- [ ] **Step 3: Implement export script**

Create `scripts/export_legacy_soxl_policy_replay.py` with a reusable
`write_legacy_soxl_policy_replay_outputs(output_dir: Path) -> dict` function and
a CLI entrypoint. It should generate both YTD and June fresh-start comparisons
against the existing oracle CSV
`runtime/production/card/soxl_backtest_jan_june_windows_v1/legacy_soxl_backtest_window_metrics.csv`.

- [ ] **Step 4: Run export test**

Run:

```bash
uv run --python /Users/rziiiii/dev/workspace/qlib_venv/bin/python --with pytest pytest tests/test_legacy_soxl_policy.py::test_export_legacy_soxl_policy_replay_writes_sidecars -q
```

Expected: pass.

### Task 4: Notebook Evidence

**Files:**
- Create: `runtime/production/card/soxl_backtest_policy_reimplementation_v1/soxl_backtest_policy_reimplementation_v1.ipynb`
- Generated sidecars under: `runtime/production/card/soxl_backtest_policy_reimplementation_v1/`

- [ ] **Step 1: Create notebook**

Add an executable notebook that imports
`write_legacy_soxl_policy_replay_outputs`, writes sidecars into the card
directory, displays comparison metrics and timing diagnostics, and ends with a
concise conclusion cell.

- [ ] **Step 2: Execute notebook**

Run:

```bash
PYTHONPATH=. /Users/rziiiii/dev/workspace/qlib_venv/bin/python -m nbconvert --to notebook --execute runtime/production/card/soxl_backtest_policy_reimplementation_v1/soxl_backtest_policy_reimplementation_v1.ipynb --inplace
```

Expected: notebook executes and writes all sidecars.

- [ ] **Step 3: Verify sidecars**

Run:

```bash
PYTHONPATH=. /Users/rziiiii/dev/workspace/qlib_venv/bin/python scripts/export_legacy_soxl_policy_replay.py --output-dir runtime/production/card/soxl_backtest_policy_reimplementation_v1
```

Expected: command exits 0 and prints the summary path.

### Task 5: Final Verification and Commit

**Files:**
- All implementation, test, notebook, and sidecar files from Tasks 1-4.

- [ ] **Step 1: Run focused tests**

Run:

```bash
uv run --python /Users/rziiiii/dev/workspace/qlib_venv/bin/python --with pytest pytest tests/test_legacy_soxl_backtest_data.py tests/test_legacy_soxl_policy.py -q
```

Expected: pass.

- [ ] **Step 2: Run harness check**

Run:

```bash
PYTHONPATH=. /Users/rziiiii/dev/workspace/qlib_venv/bin/python scripts/check_harness.py
```

Expected: no failures.

- [ ] **Step 3: Inspect git status**

Run:

```bash
git status --short
```

Expected: only files from this implementation branch are modified or untracked.

- [ ] **Step 4: Commit and push**

Run:

```bash
git add src/soxl_quant/policy/legacy_soxl.py tests/test_legacy_soxl_policy.py scripts/export_legacy_soxl_policy_replay.py docs/superpowers/plans/2026-06-12-soxl-backtest-policy-reimplementation.md
git add -f runtime/production/card/soxl_backtest_policy_reimplementation_v1
git commit -m "Reimplement SOXL backtest rule policy"
git push -u origin codex/soxl-backtest-policy-reimpl-v1
```

Expected: branch pushed with implementation and evidence.
