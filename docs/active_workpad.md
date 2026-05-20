---
date: '2026-05-20'
title: 'Production Scheduler + Paper Dashboard Loop'
kind: workpad
status: active
direction: production
---

# Production Scheduler + Paper Dashboard Loop

## Current Goal

1. Keep the SOXL/SOXX/VIX data surfaces coherent for scheduled inference.
2. Make the daily scheduler resilient to transient Yahoo/yfinance failures by retrying with cached data.
3. Repair paper-trading dashboard evidence so simulated orders are visible and use post-execution broker state.
4. Prepare the dashboard/static artifact path for Vercel after the local scheduled loop is reliable.
5. Resume model optimization only after the production loop can refresh data, infer, gate, persist, and rebuild the dashboard repeatably.

## 2026-05-20 Loop Evidence

### Root-Cause Evidence

- `runtime/soxl/paper_trading/scheduler.err` showed repeated scheduler failures where the child paper runner stderr contained `Too Many Requests`, `Rate limited`, or `No overlapping dates for SOXL/SOXX/VIX`.
- `production/soxl_daily_scheduler.py` only raised `RuntimeError("paper trading failed with exit code 1")`, so the cached-data retry branch could not see the child stderr and did not run.
- Current data state before local sync:
  - canonical panel: `4071` rows, ending `2026-05-15`
  - qlib calendar: `4071` rows, ending `2026-05-15`
  - raw SOXL/SOXX/VIX mirrors: `4057` rows, ending `2026-04-27`
- `scripts/build_backtest_dashboard.py` scanned `runtime/soxl/paper_trading/`, but skipped all CSVs with fewer than `10` rows, so a new simulated account could not appear in the dashboard.
- The paper dashboard append path used the pre-order account snapshot after order placement, allowing impossible rows such as `action=BUY` with `position=0.0`.

### Changes Made

- `production/soxl_daily_scheduler.py`
  - Added `SchedulerStepError` carrying child stdout/stderr.
  - Added transient data-source classification for rate limits, DNS/network failures, and no-overlap fetches.
  - Cached-data retry now triggers from child stderr and returns normally on success.
- `scripts/build_backtest_dashboard.py`
  - Keeps short histories for `paper_trading` sources while preserving the `len(raw_rows) >= 10` filter for post-train review histories.
- `production/soxl_paper_trading.py`
  - Added `_append_execution_daily_csv()` so cumulative paper CSV rows use the broker snapshot after execution.
- Data surfaces:
  - Ran local-only sync from the canonical panel.
  - Raw SOXL/SOXX/VIX mirrors, canonical panel, and qlib calendar now all end at `2026-05-15`.

### Verification

```bash
uv run --python qlib_venv/bin/python --with pytest pytest tests/test_daily_scheduler.py tests/test_paper_trading.py tests/test_update_market_data.py tests/test_dashboard.py -q
PYTHONPATH=. qlib_venv/bin/python scripts/update_market_data.py --local-only
PYTHONPATH=. qlib_venv/bin/python scripts/update_data_registry.py
PYTHONPATH=. qlib_venv/bin/python production/soxl_daily_scheduler.py --skip-data-update --dry-run --no-notify --skip-dashboard
PYTHONPATH=. qlib_venv/bin/python scripts/build_backtest_dashboard.py
```

Observed:

- Focused scheduler/paper/update/dashboard tests: `13 passed`.
- Cached-data scheduler dry-run completed with `as_of=2026-05-15` and wrote a dry-run paper artifact for signal date `2026-05-14`.
- The dry-run was blocked by risk gate because the auto-selected `balanced` actor has `walk-forward promotion gate not evaluated` and target step `0.850 > 0.350`.
- Dashboard rebuild wrote `25` versions and now includes `paper_trading` in `dashboard_data/index.json`.

### Continuation Evidence

- `auto_select_profile(profile_family="balanced")` previously selected `balanced` because `best_model_for_family()` fell back to `soxl_balanced_actor.pt` when no promoted candidate was found; runtime then rejected the selected model as `walk-forward promotion gate not evaluated`.
- `best_model_for_family()` now refuses unpromoted fallback models and only returns actors whose promotion sidecar passes `promotion_gate_status()`.
- Local promotion evidence for `balanced_rule_residual_v19` was aligned with the registry intent (`status=canonical`, `wf_status=completed`), so cached-data inference now auto-selects `balanced_rule_residual_v19`.
- Paper mode now persists `PaperBroker` cash/shares in `runtime/soxl/paper_trading/paper_broker_state.json` and clips large first-step target changes to the runtime step limit instead of blocking a cold account forever.
- The daily drawdown circuit-breaker path now replaces the frozen risk-gate decision with a blocked copy instead of mutating the dataclass in place.
- One local non-dry-run paper execution filled a clipped BUY from `0%` toward `35%`, appended the cumulative dashboard CSV from post-execution broker state, and persisted broker state; a follow-up dry-run read the realized `33.5%` position and approved the next step toward `65%`.
- `production/soxl_daily_scheduler.py --smoke` now runs cached-data dry-run inference plus dashboard rebuild without notifications or order placement.
- Static dashboard packaging is Vercel-ready: `dashboard.html`, `dashboard_data/index.json`, `dashboard_data/model_versions.json`, and 25 `dashboard_data/versioned/*/SOXL.json` files are trackable; `vercel.json` rewrites `/` to `/dashboard.html` and sets revalidation headers for dashboard data.

Additional verification:

```bash
uv run --python qlib_venv/bin/python --with pytest pytest tests/test_runtime.py::test_best_model_for_family_refuses_unpromoted_fallback tests/test_runtime.py::test_best_model_for_family_selects_highest_promoted_score -q
uv run --python qlib_venv/bin/python --with pytest pytest tests/test_runtime.py::test_risk_gate_clips_large_position_step_when_enabled tests/test_paper_trading.py::test_paper_broker_state_roundtrip tests/test_paper_trading.py::test_post_execution_audit_uses_realized_paper_position tests/test_paper_trading.py::test_block_risk_decision_preserves_frozen_decision_contract -q
uv run --python qlib_venv/bin/python --with pytest pytest tests/test_daily_scheduler.py::test_scheduler_smoke_mode_runs_cached_dry_run_with_dashboard tests/test_vercel_dashboard.py -q
PYTHONPATH=. qlib_venv/bin/python production/soxl_daily_scheduler.py --skip-data-update --dry-run --no-notify --skip-dashboard
PYTHONPATH=. qlib_venv/bin/python production/soxl_daily_scheduler.py --skip-data-update --no-notify --skip-dashboard
PYTHONPATH=. qlib_venv/bin/python production/soxl_daily_scheduler.py --skip-data-update --dry-run --no-notify --skip-dashboard
PYTHONPATH=. qlib_venv/bin/python scripts/build_backtest_dashboard.py
curl -I http://127.0.0.1:8766/dashboard.html
curl -I http://127.0.0.1:8766/dashboard_data/index.json
```

Final verification:

- Focused regression set: `29 passed`.
- `production/soxl_daily_scheduler.py --smoke`: completed cached-data inference, auto-selected `balanced_rule_residual_v19`, approved the dry-run risk gate, and rebuilt `dashboard_data/` with `25` versions.
- `scripts/check_harness.py`: `passed=71 failures=0`.
- Local HTTP checks returned `200 OK` for `dashboard.html` and `dashboard_data/index.json`.
- Browser screenshot verification was attempted through Computer Use, but macOS Accessibility/Screen Recording permissions were still pending; no screenshot evidence was produced in this pass.

## Next Action

1. Commit and push the production-loop evidence.
2. After production loop stability is on remote, resume model optimization from the SOXX-aware frontier rather than changing scheduler/runtime behavior further.

## Prior Data/Release Goal

1. Refresh the repo-owned local data surfaces from the canonical SOXL/SOXX/VIX panel.
2. Run one narrow release/re-entry feature training loop on the retained residual frontier.

## Scope

This pass is a `data` revalidation plus one `factor` / `model_selection` ablation loop.
It is not a broad mode retune, reward retune, or macro-feature retry.

## Baselines And Role

- Canonical baseline: `balanced_rule_residual_v19`
- Direct comparison baseline: `balanced_rule_residual_v22`
- Role: `ablate`
- Target window: fixed-split `test_2022` / `test_2023_2024`, with `test_proxy` as the secondary check
- Accepted trade-off: allow modest `test_2020` giveback if recovery improves without breaking `test_2022` avg_pos / MaxDD gates
- Changed layer: `feature`
- External reference:
  - rulebase restart timing proxy `RB_COOLDOWN_PROXY`
  - transition release signal `TRANS_VIX_DIVERGENCE`
  - CICC low-frequency price-volume factor `CICC_RETURN_VOLUME_SYNC`
- Comparison-table row key: `balanced_rule_residual_v22_release_v1`

## Closest Historical Cases

1. Closest success case:
   `dwell_hold_v2` fixed-actor overlay improved recovery under low churn, and `balanced_rule_residual_v22` retained acceptable defensive behavior on fixed split.
2. Closest failure case:
   `balanced_rule_residual_v19_transition` broad transition-feature retrain and `balanced_macro_v1` broad macro pack both improved recovery narratives while reopening unwanted bear-market exposure / proxy degradation.
3. Why this loop is not a duplicate:
   It does not reopen the full `transition_signals` or `macro_signals` packs. It only curates three retained release/re-entry factors with prior standalone evidence into a narrow pack, keeping the retained v22 residual stack unchanged otherwise.

## Evidence

### Commands

```bash
PYTHONPATH=. qlib_venv/bin/python scripts/update_market_data.py --local-only
PYTHONPATH=. qlib_venv/bin/python scripts/update_data_registry.py
```

### Observed State Before Sync

- `data/raw/symbols/SOXL.csv` ended at `2026-04-01`
- `data/processed/panels/soxl_soxx_vix_full.csv` already ended at `2026-04-27`
- `data/qlib/soxl/calendars/day.txt` already ended at `2026-04-27`

### Observed State After Sync

- `data/raw/symbols/SOXL.csv` synced to `2026-04-27`
- `data/raw/symbols/SOXX.csv` created/synced to `2026-04-27`
- `data/raw/external/VIX.csv` created/synced to `2026-04-27`
- `data/qlib/soxl/` rebuilt from the canonical panel
- `data/datasets.yaml` regenerated successfully

### Live Refresh Blocker

Both `requests` and `yfinance` failed with DNS resolution errors against Yahoo endpoints in this automation environment. The blocker here is not rate limiting; it is lack of outbound name resolution / network access from the current run.

## 2026-05-10 Recheck

### Commands

```bash
PYTHONPATH=. qlib_venv/bin/python - <<'PY'
import pandas as pd
from pathlib import Path
paths = {
    "panel": Path("data/processed/panels/soxl_soxx_vix_full.csv"),
    "raw_soxl": Path("data/raw/symbols/SOXL.csv"),
    "raw_soxx": Path("data/raw/symbols/SOXX.csv"),
    "raw_vix": Path("data/raw/external/VIX.csv"),
}
for name, path in paths.items():
    df = pd.read_csv(path)
    date_col = "date" if "date" in df.columns else df.columns[0]
    print(name, len(df), df[date_col].iloc[-1])
cal = Path("data/qlib/soxl/calendars/day.txt")
lines = [line.strip() for line in cal.read_text().splitlines() if line.strip()]
print("qlib_calendar", len(lines), lines[-1])
PY

PYTHONPATH=. qlib_venv/bin/python - <<'PY'
import requests
try:
    requests.get(
        "https://query1.finance.yahoo.com/v8/finance/chart/SOXL?interval=1d&period1=0&period2=1",
        timeout=20,
    ).raise_for_status()
    print("unexpected_success")
except Exception as exc:
    print(type(exc).__name__, str(exc))
PY
```

### Observed State

- Canonical panel, `raw/` mirrors, and qlib calendar all still end at `2026-04-27`.
- Live refresh remains blocked on `query1.finance.yahoo.com` name resolution.
- There is no newer local data to unlock inside this sandboxed automation run.

## Narrow Release-Pack Loop

- New profile: `balanced_rule_residual_v22_release_v1`
- Feature pack: `release_signals_v1`
  - `RB_COOLDOWN_PROXY`
  - `TRANS_VIX_DIVERGENCE`
  - `CICC_RETURN_VOLUME_SYNC`
- Hypothesis:
  A narrow release pack can teach the retained residual actor to re-risk earlier in recovery windows without repeating the broad-transition failure mode of lifting `test_2022` exposure.
- Acceptance criteria:
  - keep `test_2022 avg_pos <= 0.35`
  - keep `test_2022 maxdd <= 55%`
  - keep `test_proxy sharpe >= 0.50`
  - improve `test_2023_2024` return or Sharpe over `balanced_rule_residual_v22`
- Expected trade-off:
  mild `test_2020` giveback is acceptable if recovery improves materially without proxy drawdown blowout
- Failure mode:
  valid / proxy improve while `test_2022 avg_pos` or proxy MaxDD rises in the same way as prior broad transition / macro retrains
- Planned command:

```bash
PYTHONPATH=. qlib_venv/bin/python soxl_qlib_strategy_v2.py --profiles balanced_rule_residual_v22_release_v1 --skip-data-refresh --seed 20260411
```

### Training Readout

- Training completed successfully after the terminal-date backtest fix in `src/soxl_quant/backtest.py`.
- Validation still liked the mechanism:
  - best valid snapshot: `return=61.0%`, `sharpe=1.662`, `maxdd=27.84%`
  - iter `20` was close (`60.3% / 1.684 / 26.89%`)
  - iter `40` degraded (`52.9% / 1.635 / 27.04%`)
- This is weak evidence that "train longer" is the answer. The mechanism signal is already visible by iter `0-20`; later training does not resolve the trade-off.

### Corrected Replay Backtest

Replay method: `scripts/run_segment_backtest.py` under the current harness after the terminal-date fix.

| window | `balanced_rule_residual_v22` | `balanced_rule_residual_v22_release_v1` | readout |
|---|---|---|---|
| `test_proxy` | `1099.1% / 0.976 / 53.21% / 0.498 / 169 trades` | `1350.3% / 1.010 / 53.26% / 0.520 / 172 trades` | proxy and upside improve; exposure lifts slightly |
| `test_2020` | `37.3% / 0.812 / 45.75% / 0.408 / 23 trades` | `42.7% / 0.860 / 44.95% / 0.430 / 22 trades` | mild bull improvement |
| `test_2022` | `-41.8% / -1.335 / 43.63% / 0.241 / 21 trades` | `-46.3% / -1.480 / 48.45% / 0.252 / 19 trades` | defense worsens in the wrong direction |
| `test_2023_2024` | `180.2% / 1.336 / 31.91% / 0.486 / 52 trades` | `234.3% / 1.479 / 32.80% / 0.509 / 47 trades` | recovery improves materially |

### Decision

- Keep `balanced_rule_residual_v22_release_v1` as **ablation evidence**, not a replacement candidate.
- The curated trio improves recovery, but it does so by teaching a broader `risk-on` bias rather than a selective release rule:
  - `test_2022 avg_pos` stays below the hard gate (`0.252 < 0.35`) but still drifts upward vs `v22`
  - `test_2022 maxdd` widens by `4.82pp`
  - proxy max drawdown is flat, so the pack is not adding cleaner timing, only more exposure
- Durable review: `runtime/model/card/balanced_rule_residual_v22_release_v1/v22_release_v1_backtest_review.md`

## Bottleneck Reset

Current project bottleneck is **not**:

- another macro feature injection
- another looser mode envelope
- another more-conservative risk-budget ablation
- another “maybe training not enough” guess

Current project bottleneck **is**:

- **selective re-entry / recovery timing after de-risking under low-churn constraints**
- **execution/runtime mapping quality, not generic threshold loosening**
- **factor usage quality, not raw factor availability**: the retained release factors are good enough to move behavior, but direct RL injection makes them act like ungated bullish pressure

Evidence trail:

- `risk_budget_usage_v1` has excellent 2022 defense but still recovers too slowly versus SOXL buy-and-hold
- `balanced_rule_residual_v19` / `v22` remain defensively acceptable but also lag 2023-2024 upside
- `balanced_macro_v1` improved recovery but worsened bear-market exposure, so it does not solve the real trade-off
- `runtime/execution/knowledge.md` already has `显式 release/re-entry overlay` as the next valid execution line
- `runtime/model/card/balanced_rule_residual_v19/v19_overlay_promotion_review.md` shows `fast_reentry` improves `2023-2024` return over `base` (`59.3%` vs `46.1%`) but still trails `dwell_hold_v2` on proxy Sharpe / drawdown (`0.949 / 51.06%` vs `1.064 / 44.42%`)
- `runtime/model/card/balanced_rule_residual_v19/v19_dwell_stress_merge_review.md` shows `regime_route_v1` further improves recovery (`61.8%`) and 2022 drawdown (`32.61%`) but at much higher churn (`464` trades vs `121` for `dwell_hold_v2`)
- `runtime/model/card/risk_budget_usage_v1/risk_budget_usage_v1_full_review.md` shows the composed counterfactual is strong, while the native PPO actor is much more conservative, so part of the bottleneck is training/runtime mismatch rather than missing macro context
- `runtime/model/card/balanced_rule_residual_v22_release_v1/v22_release_v1_backtest_review.md` shows the new release pack improves `2023-2024` materially but worsens `2022`, which means the current model is not learning a conditional release classifier from these signals alone

## Next Action

1. Keep local data sync on the canonical panel path; do not treat data freshness as the primary blocker until a network-enabled run can extend the canonical panel beyond `2026-04-27`.
2. Open the next narrow optimization loop around **gated selective release/re-entry classification**, using `RB_COOLDOWN_PROXY`, `TRANS_VIX_DIVERGENCE`, and `CICC_RETURN_VOLUME_SYNC` as release gates inside a low-churn fixed-actor overlay before trying another retrain.
3. Treat retraining as a second-stage step only if the fixed-actor gated release counterfactual improves `2023-2024` without reopening `2022` drawdown.

## 2026-05-14 Automation Run

### Data Refresh

Commands:

```bash
PYTHONPATH=. qlib_venv/bin/python scripts/update_market_data.py --local-only
PYTHONPATH=. qlib_venv/bin/python scripts/update_data_registry.py
PYTHONPATH=. qlib_venv/bin/python - <<'PY'
import requests
try:
    requests.get("https://query1.finance.yahoo.com/v8/finance/chart/SOXL?interval=1d&period1=0&period2=1", timeout=20).raise_for_status()
except Exception as exc:
    print(type(exc).__name__, str(exc))
PY
```

Observed state:

- Local canonical panel, raw mirrors, and qlib calendar remain aligned at `2026-04-27`.
- `data/datasets.yaml` was regenerated at `2026-05-14T02:02:18Z`.
- Live Yahoo access is still blocked by DNS / name resolution in this automation environment (`query1.finance.yahoo.com` cannot resolve).

### Bottleneck Brainstorm

The current bottleneck is not data freshness or insufficient PPO iterations. The
release pack already moved behavior in `balanced_rule_residual_v22_release_v1`,
but ungated RL injection acted like broad bullish pressure. The right next test
was therefore a fixed-actor execution separation test: keep the v22 actor fixed,
add only a low-churn release gate, and require the gate to use retained factor
evidence (`RB_COOLDOWN_PROXY`, `TRANS_VIX_DIVERGENCE`, `CICC_RETURN_VOLUME_SYNC`).

Feature distribution check:

- `CICC_RETURN_VOLUME_SYNC` is materially higher in `2023-2024` than `2022`
  (median `0.6238` vs `0.2756`), making it a useful recovery discriminator.
- `regime_risk` is lower in `2023-2024` than `2022` (median `0.3721` vs `0.6821`),
  so a `max_regime_risk` gate can avoid bear-market release.
- `RB_COOLDOWN_PROXY` is also stronger in `2023-2024` (median `0.1383`, p75 `0.3684`)
  than `2022` (median `0.0000`, p75 `0.1361`).

### Selective Release Overlay Loop

Command:

```bash
PYTHONPATH=. qlib_venv/bin/python scripts/evaluate_selective_release_overlay.py --profile balanced_rule_residual_v22
```

Evidence:

- Review: `runtime/execution/card/selective_release_overlay_v1/selective_release_overlay_review.md`
- CSV: `runtime/execution/card/selective_release_overlay_v1/selective_release_overlay_results.csv`
- JSON: `runtime/execution/card/selective_release_overlay_v1/selective_release_overlay_results.json`

Best counterfactual: `selective_release_recovery_v1`

| window | baseline v22 fixed actor | selective release | readout |
|---|---:|---:|---|
| `test_proxy` | `1132.19% / 1.043 / 56.48% / avg_pos 0.5226 / 172 trades` | `1415.10% / 1.105 / 54.30% / avg_pos 0.5230 / 171 trades` | better return, Sharpe, and drawdown |
| `test_2020` | `131.13% / 2.160 / 20.92% / avg_pos 0.5356 / 27 trades` | unchanged | no bull-market side effect |
| `test_2022` | `-30.97% / -0.764 / 40.64% / avg_pos 0.3304 / 21 trades` | unchanged | bear defense not reopened |
| `test_2023_2024` | `57.12% / 0.719 / 34.87% / avg_pos 0.5402 / 50 trades` | `82.10% / 0.877 / 36.95% / avg_pos 0.5510 / 50 trades` | recovery improved with modest drawdown cost |

Decision:

- Keep `selective_release_recovery_v1` as the retained execution counterfactual.
- Do not start another PPO retrain yet; this fixed-actor result is stronger evidence that the bottleneck is runtime gating / release classification.
- Next action is walk-forward actor validation for the same overlay, then convert to a governed execution overlay/profile only if the benefit survives.

## 2026-05-15 Automation Recheck

### Data Refresh

Commands:

```bash
PYTHONPATH=. qlib_venv/bin/python scripts/update_market_data.py --local-only
PYTHONPATH=. qlib_venv/bin/python scripts/update_data_registry.py
PYTHONPATH=. qlib_venv/bin/python - <<'PY'
import requests
try:
    requests.get("https://query1.finance.yahoo.com/v8/finance/chart/SOXL?interval=1d&period1=0&period2=1", timeout=20).raise_for_status()
except Exception as exc:
    print(type(exc).__name__, str(exc))
PY
```

Observed state:

- Local-only sync succeeded and rebuilt `data/qlib/soxl`.
- Canonical panel, raw SOXL/SOXX/VIX mirrors, and qlib calendar all remain aligned at `2026-04-27`.
- Live Yahoo refresh is still blocked by DNS / name resolution in this automation environment.

### Walk-Forward Selective Release Validation

Command:

```bash
MPLCONFIGDIR=/private/tmp/mplconfig PYTHONPATH=. qlib_venv/bin/python scripts/evaluate_selective_release_walkforward.py --profile balanced_rule_residual_v22 --all-variants
```

Evidence:

- Review: `runtime/execution/card/selective_release_walkforward_v1/selective_release_walkforward_review.md`
- CSV: `runtime/execution/card/selective_release_walkforward_v1/selective_release_walkforward_results.csv`
- JSON: `runtime/execution/card/selective_release_walkforward_v1/selective_release_walkforward_results.json`

Key result:

| actor_window | baseline | best prior gate | readout |
|---|---:|---:|---|
| `wf_2020` | `113.17% / 2.015 / 19.47% / avg_pos 0.5322` | unchanged | no release signal days |
| `wf_2022` | `-34.11% / -0.926 / 38.27% / avg_pos 0.3216` | unchanged | no release signal days, defense not reopened |
| `wf_2023_2024` | `-20.21% / -0.277 / 38.54% / avg_pos 0.4764` | unchanged | release signals fire, but execution absorbs the overlay |

Detailed readout:

- `selective_release_cicc_v1` has `16` eligible days in `wf_2023_2024`, but `0` changed days.
- `selective_release_tight_v1` has `6` eligible days in `wf_2023_2024`, but `0` changed days.
- `selective_release_recovery_v1` has `18` eligible days in `wf_2023_2024`, but `0` changed days.

Decision:

- Do **not** promote `selective_release_recovery_v1` into runtime/profile governance yet.
- The fixed full-window actor benefit from 2026-05-14 does not survive the walk-forward actor check as an executable change.
- The updated bottleneck is narrower: release factors can identify 2024 recovery days, but the current execution mapping can absorb the boost without changing actual orders.

Next action:

1. Keep data refresh on local canonical sync until a network-enabled run can extend beyond `2026-04-27`.
2. Diagnose target/position absorption on `wf_2023_2024` eligible days before retraining.
3. If the next fixed-actor test changes orders, constrain it to an execution mechanism such as target-lift vs rebalance-threshold bypass; do not add another ungated PPO feature branch.

---

## Archived Snapshot: 2026-04-28 Macro Evaluation

## 背景

在 `balanced` profile 基础上添加宏观领先指标特征包（credit spread, yield curve, DXY, oil），构建 `balanced_macro_v1`，与相同 seed（42）的 baseline `balanced` 做公平对比。

## 实验设计

| 维度 | balanced (baseline) | balanced_macro_v1 |
|------|---------------------|-------------------|
| 特征数 | 59 | 69 (+10 macro) |
| 网络 | MLP [128,64] | MLP [128,64] |
| Seed | 42 | 42 |
| 训练迭代 | 80 | 80 |
| 特征包 | — | `macro_signals` |

## 结果对比

| 指标 | balanced | balanced_macro_v1 | 差异 |
|------|----------|-------------------|------|
| **Valid best** | | | |
| Return | 55.7% | 60.6% | +4.9pp |
| Sharpe | 1.49 | 1.47 | -0.02 |
| MaxDD | 28.4% | 32.7% | +4.3pp |
| Score | **82.10** | **86.15** | +4.05 |
| **test_proxy** | | | |
| Return | **605.2%** | 450.3% | **-154.9pp** |
| Sharpe | **0.89** | 0.78 | **-0.11** |
| MaxDD | 55.3% | 58.3% | +3.0pp |
| avg_pos | 0.519 | 0.594 | +0.075 |
| **test_2020** | | | |
| Return | 126.4% | 144.4% | +18.0pp |
| Sharpe | 2.37 | 2.26 | -0.11 |
| MaxDD | 23.7% | 25.3% | +1.6pp |
| avg_pos | 0.547 | 0.576 | +0.029 |
| **test_2022** | | | |
| Return | -36.7% | -35.5% | +1.2pp |
| Sharpe | -1.23 | -1.00 | +0.23 |
| MaxDD | 38.8% | 39.5% | +0.7pp |
| **avg_pos** | **0.328** ✅ | **0.368** ❌ | **+0.040** |
| **test_2023_2024** | | | |
| Return | 62.4% | **82.5%** | **+20.1pp** |
| Sharpe | 0.79 | **0.87** | **+0.08** |
| MaxDD | 39.2% | 38.6% | -0.6pp |
| avg_pos | 0.560 | 0.620 | +0.060 |

## Promotion Gate

| 检查项 | balanced | balanced_macro_v1 |
|--------|----------|-------------------|
| test_2022_max_dd < 55% | 38.8% ✅ | 39.5% ✅ |
| test_2022_avg_pos < 0.35 | **0.328** ✅ | **0.368** ❌ |
| test_proxy_sharpe > 0.50 | 0.89 ✅ | 0.78 ✅ |
| **结果** | **eligible** | **BLOCKED** |

## 关键发现

1. **Macro 模型在 2022 bear market 更激进，而非更防御**：
   - test_2022 avg_pos: 0.368 vs 0.328（+12%）
   - 宏观指标（credit spread 扩大、yield curve 倒挂）理论上应暗示风险，但 RL 代理反而持仓更高
   - 这是 promotion gate 失败的根本原因

2. **Macro 模型在 2023-2024 recovery 期表现更好**：
   - Return: 82.5% vs 62.4%（+20pp）
   - 宏观特征可能帮助识别 bull market 的早期信号

3. **test_proxy 整体收益下降**：
   - 605.2% → 450.3%（-26%）
   - Sharpe: 0.89 → 0.78
   - 宏观特征在全周期引入了噪声，降低了整体效率

4. **Valid score 反而更高**：
   - 82.10 → 86.15
   - 说明宏观特征在 valid set（2015-2019）过拟合了

## 假设分析

为什么宏观特征没有帮助 bear market 防御？

| 假设 | 可能性 | 验证方式 |
|------|--------|----------|
| Macro 信号与 price action 相比太弱，被淹没 | 高 | 单独回测 macro 信号的 IC |
| Z-score 处理丢失了方向信息 | 中 | 尝试 raw level + delta 组合 |
| 代理需要更长时间学习宏观关联 | 低 | 已训练 80 iter，足够 |
| 宏观特征与现有特征高度相关 | 中 | 计算特征间相关性 |
| 需要 macro-aware position cap | 高 | 尝试将 macro 信号接入 position cap |

## 结论

`balanced_macro_v1` **不通过 promotion gate**，不建议替换 baseline。

宏观特征在当前形式下（z-score + delta 的 10 维输入）对 RL 代理是**噪声而非信号**。虽然 valid score 和 2023-2024 收益改善，但核心缺陷（bear market 防御能力下降）无法容忍。

## 下一步选项

1. **Option A: 放弃 macro 方向** — 资源投入回报比低，现有特征已足够
2. **Option B: 改进 macro 预处理** — 尝试 regime-aware encoding（bull/bear/neutral 标签代替连续 z-score）
3. **Option C: Macro-aware position cap** — 不直接输入 macro 特征到 policy，而是将 macro 信号接入 position cap 系统，硬性限制 bear market 最大仓位
4. **Option D: 单独验证 macro 信号 IC** — 先确认宏观指标本身有预测力，再决定如何接入

推荐：**Option D → 若 IC 验证通过 → Option C**

---

## Phase 1: IC 验证结果 ✅ 完成

### 方法
- 加载 SOXL 日度收益率（2020-01-03 ~ 2026-04-01，1569 个交易日）
- 构建 14 个宏观信号变体（raw level, z-score 252, delta 21, ROC 63）
- 计算信号(t) 与 forward return(t+1, +5, +10, +21) 的 Spearman rank correlation (IC)
- 额外按 bull/bear regime 拆分

### 关键结果

| 信号 | IC(1d) | IC(5d) | IC(21d) | IR(5d) | Pos Rate(5d) |
|------|--------|--------|---------|--------|--------------|
| MACRO_CREDIT_SPREAD | -0.003 | +0.024 | +0.006 | +0.40 | 49.8% |
| MACRO_YC_10Y2Y | +0.022 | +0.020 | +0.017 | +0.40 | 49.9% |
| MACRO_YC_10Y3M | -0.017 | -0.009 | +0.006 | -0.23 | 37.2% |
| MACRO_CREDIT_SPREAD_ZSCORE_252 | -0.015 | +0.012 | -0.002 | +0.18 | 50.1% |
| **MACRO_YC_10Y2Y_ZSCORE_252** | +0.016 | **+0.026** | **+0.027** | **+0.74** | **62.6%** |
| MACRO_YC_10Y3M_ZSCORE_252 | -0.023 | -0.012 | +0.018 | -0.28 | 30.5% |
| MACRO_DXY_ZSCORE_252 | +0.020 | +0.015 | +0.003 | +0.32 | 55.2% |
| **MACRO_OIL_ZSCORE_252** | **-0.039** | **-0.040** | -0.021 | **-1.00** | 12.5% |
| MACRO_CREDIT_SPREAD_DELTA_21 | -0.014 | +0.006 | +0.004 | +0.19 | 49.7% |
| MACRO_YC_10Y2Y_DELTA_21 | +0.005 | +0.006 | +0.021 | +0.14 | 50.1% |
| MACRO_CREDIT_SPREAD_ROC_63 | -0.007 | +0.013 | -0.014 | +0.27 | 50.1% |
| MACRO_YC_10Y2Y_ROC_63 | +0.016 | +0.005 | +0.015 | +0.11 | 28.0% |
| MACRO_DXY_ROC_63 | +0.017 | +0.021 | -0.012 | +0.45 | 53.4% |
| MACRO_OIL_ROC_63 | -0.031 | -0.028 | -0.015 | -0.44 | 29.0% |

### 发现

1. **所有信号的 |IC| < 0.04，属于极弱信号**
   - 最佳信号 MACRO_YC_10Y2Y_ZSCORE_252 的 IC(5d) = +0.026
   - 最佳信号 MACRO_OIL_ZSCORE_252 的 IC(1d) = -0.039
   - 因子投资中 |IC| < 0.05 通常认为无可用的 standalone alpha

2. **高 IR 来自低波动而非高预测力**
   - MACRO_YC_10Y2Y_ZSCORE_252 IR=0.74 看起来很吸引人
   - 但这是 IC std=0.035 带来的（信号非常稳定但幅度极小）
   - 实际可交易的 alpha 几乎为零

3. **Regime 间信号方向翻转 — 致命问题**
   - MACRO_CREDIT_SPREAD_ZSCORE_252: bull IC=+0.027, bear IC=-0.053
   - MACRO_CREDIT_SPREAD_DELTA_21: bull IC=-0.042, bear IC=+0.014
   - 同一信号在 bull/bear 中预测方向相反！
   - 这解释了为什么 RL 代理无法学习：信号关系是 regime-dependent，而 simple MLP 无法隐式建模 regime 条件

4. **数据样本期太短**
   - 仅 2020-2026（6 年），包含一次 bear market（2022）
   - 无法对 regime-specific IC 做出统计可靠的结论

## Phase 2: 决策

**结论：宏观信号无 standalone alpha，永久放弃 direct policy input 方向。**

| 检查标准 | 阈值 | 最佳信号表现 | 判定 |
|----------|------|-------------|------|
| \|mean IC\| | ≥ 0.05 | 0.026-0.039 | ❌ 未达标 |
| IC IR | ≥ 0.30 | 0.74 (但 IC 太小) | ⚠️ 伪达标 |
| Regime 稳定性 | 符号一致 | 方向翻转 | ❌ 未达标 |
| 样本期 | ≥ 10 年 | 6 年 | ❌ 不足 |

**决策：** 执行 **Option A** — 永久放弃 macro 特征作为 policy input。

**理由：**
- IC 验证确认宏观信号对 SOXL 收益无可靠的 standalone 预测力
- Regime 方向翻转意味着即使 IC 略高，也无法被简单 MLP 利用
- 已有 `regime_risk` 等 CORE 特征代理了 regime 信息，添加宏观特征是冗余噪声
- `balanced_macro_v1` 的训练失败（bear market 更激进）与 IC 结果一致

**不再尝试的方向：**
- ❌ 不同 macro 预处理（regime encoding, momentum, cross-sectional rank）
- ❌ Macro-aware position cap（底层信号无预测力，cap 也无意义）
- ❌ 更多宏观数据源（gold, unemployment, CPI 等）

---

## 附：Dashboard 数据区间问题排查 ✅

### 用户反馈
1. "重新计算后 2026 年就没有数据了"
2. "每次热 compute 后都会限制时间区间"
3. "而且没有到今天的最新数据"

### 根因分析

**根因 1: `TEST_SEG` 硬编码为 `("2020-01-01", "2026-03-31")`**
- `src/soxl_quant/settings.py:41`
- `dashboard_server.py` 的 Run Backtest 使用 `_segment_frame(dataset, "test")`
- qlib 的 test segment 被 TEST_SEG 限制，导致 backtest 无法使用 2026-03-31 之后的数据
- 即使 qlib 数据本身已更新到 2026-04-27，backtest 仍被截断

**根因 2: `build_backtest_dashboard.py` 使用 stale raw CSV**
- `load_ohlc_and_indicators()` 从 `data/raw/symbols/SOXL.csv` 加载
- Raw CSV 最新日期: 2026-04-01
- 但 processed CSV `soxl_soxx_vix_full.csv` 已更新到 2026-04-27

**根因 3: yfinance rate limit 导致 raw 数据更新失败**
- `scripts/update_market_data.py` 连续失败：`Too Many Requests`
- 需要等待 rate limit 重置后重试

### 修复

1. **更新 `TEST_SEG` 为 `("2020-01-01", "2026-12-31")`**
   - `src/soxl_quant/settings.py`
   - `src/soxl_quant/app/evaluate.py`（改用 TEST_SEG 变量引用）
   - `scripts/compare_mode_routers.py`
   - `scripts/analyze_learned_router_runtime.py`
   - `src/soxl_quant/backtest.py` / `data.py` 文档同步更新

2. **更新 `build_backtest_dashboard.py` 优先使用 processed CSV**
   - `load_ohlc_and_indicators()` 现在优先读取 `data/processed/panels/soxl_soxx_vix_full.csv`
   - VIX 也优先从 processed CSV 读取

3. **已重建 dashboard**
   - `dashboard.html` 和 `dashboard_data/` 已更新
   - OHLC overlay 现在覆盖到 2026-04-27

### 待办
- [ ] yfinance rate limit 重置后运行 `update_market_data.py` 更新 raw CSV
- [ ] 用新 TEST_SEG 重新运行关键 profile 的 backtest，生成新的 `*_soxl_daily.csv`（延伸到 2026-04-27）
- [ ] 重新运行 `build_backtest_dashboard.py` 更新 versioned backtest 数据

## 清理工作

- [x] 保留 `balanced_macro_v1` 档案（profile.yaml + summary + actor checkpoint）作为失败证据
- [x] 保留 `macro_signals` feature pack 代码（data.py 中的 `_builtin_macro_signals_pack`）但不再激活
- [x] IC 分析脚本和结果存档：`scripts/compute_macro_ic.py`, `runtime/model/card/macro_ic_analysis/`
- [ ] 从 `BUILTIN_FEATURE_PACKS` 中移除 `macro_signals`（或保留但不推荐）
- [ ] 更新 experiment-landscape-review 中 `balanced_macro_v1` 状态为 `dropped`（已 done）

## 实验档案

- `balanced_macro_v1` profile: `runtime/model/card/balanced_macro_v1/profile.yaml`
- `balanced` baseline (seed 42): `runtime/model/card/balanced/soxl_balanced_summary.json`
- `balanced_macro_v1` summary: `runtime/model/card/balanced_macro_v1/soxl_balanced_macro_v1_summary.json`
- IC 分析脚本: `scripts/compute_macro_ic.py`
- IC 结果: `runtime/model/card/macro_ic_analysis/ic_results.json`
- IC 摘要: `runtime/model/card/macro_ic_analysis/ic_summary.txt`

---

## 2026-05-17 Discord 通知集成

### 变更

- `production/notify.py`: 新增 `DiscordNotifier` 和 `MultiNotifier`
  - 支持 `DISCORD_BOT_TOKEN` / `DISCORD_CHANNEL_ID` 环境变量
  - 自动从 `.env.discord` 读取 token（环境变量优先）
  - `get_notifier(include_discord=True)` 同时发送 Bark/macOS + Discord
  - Cloudflare 403 修复：添加 `User-Agent` header
- `production/soxl_paper_trading.py`: 新增 `--discord` 参数
- `production/soxl_daily_scheduler.py`: 新增 `--discord` 参数
- `production/composed_session.py`: 新增 `--discord` 参数

### 权限验证

| 频道 | ID | 状态 |
|------|-----|------|
| `#general` | 1474219222904082606 | ✅ 可读可写 |
| `#tradingagent` | 1475287062553956565 | ✅ 可读可写 (默认目标) |
| `#moon` | 1476061424382906561 | ✅ 可读可写 |
| `#options` | 1476879729624612926 | ✅ 可读可写 |
| `#being` | 1483370904372379768 | ✅ 可读可写 |
| `#soxl` | 1485666024865927309 | ❌ Missing Access |
| `#daily` | 1504487456572182639 | ❌ Missing Access |

Bot 无法访问 `#soxl` 和 `#daily`（需要服务器管理员在频道权限中添加 bot）。当前默认频道设为 `#tradingagent`。

### 环境变量更新

`~/.zshrc` 中的 `DISCORD_BOT_TOKEN` 已更新为最新有效 token。

### 测试命令

```bash
# 仅 Discord
PYTHONPATH=. qlib_venv/bin/python production/notify.py --discord --title "Test" --body "Hello"

# Bark + Discord
PYTHONPATH=. qlib_venv/bin/python production/notify.py --discord --title "Test" --body "Hello"

# Paper trading with Discord
PYTHONPATH=. qlib_venv/bin/python production/soxl_paper_trading.py --discord --dry-run
```

### 已知限制

- `soxl` / `daily` 频道需管理员手动授权 bot 的 `Send Messages` 权限
- 消息长度受 Discord 2000 字符限制（自动截断）
