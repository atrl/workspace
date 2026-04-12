# SOXL Agent Loop

Status: active
Owner: Codex
Last updated: 2026-04-12 19:08 CST

## Goal

Build a `qlib`-native SOXL strategy that beats the old 200-iteration baseline on return without losing control of drawdown quality, and does so under a repeatable non-leaking evaluation harness.

## Current baselines

- Old qlib baseline:
  - `437% / Sharpe 0.85 / MaxDD 40.2%`
- Recent active version:
  - `556.6% / Sharpe 0.80 / MaxDD 71.6% / 366 trades`
- Latest smoke after dynamic action-band refactor:
  - `354.1% / Sharpe 0.77 / MaxDD 56.0% / 826 trades`
- Latest full balanced result on current branch:
  - `510.9% / Sharpe 0.79 / MaxDD 68.9% / 814 trades`
- Balanced walk-forward (2026-04-11 heartbeat run):
  - Full test: `368.7% / Sharpe 0.80 / MaxDD 51.6% / 894 trades`
  - test_2020: `92.9% / Sharpe 2.30 / MaxDD 17.1%`
  - test_2022: `-26.5% / Sharpe -0.84 / MaxDD 34.6%`
  - test_2023_2024: `46.0% / Sharpe 0.73 / MaxDD 30.5%`
  - WF_2020: `167.7% / Sharpe 0.57 / MaxDD 69.4%`
  - WF_2022: `53.3% / Sharpe 0.38 / MaxDD 65.4%`
  - WF_2023_2024: `1.1% / Sharpe 0.10 / MaxDD 38.8%`
- Current critical weakness:
  - `test_2022 = -39.8% / Sharpe -0.65 / MaxDD 53.3%`
- Walk-forward smoke:
  - `wf_2020 = 416.9% / Sharpe 0.73 / MaxDD 81.6%`

## Active hypotheses

1. Long-trend anchor plus cumulative-drop state is more useful for SOXL than adding more reward penalties.
2. Dynamic action bands can outperform a fixed 0-1 uniform action grid because SOXL benefits from a core-and-satellite position shape.
3. Training should oversample stress-like windows from the train segment by state, not by future calendar year.
4. Checkpoint selection should likely stop earlier than the configured max iterations because validation appears to plateau around `20-80` iterations.
5. Productionizing the strategy requires a clean separation between actor training, signal replay, and broker execution rather than more ad-hoc scripts.

## Validation harness

- Main run:
  - `python3 soxl_qlib_strategy_v2.py --profiles balanced`
- Walk-forward:
  - `python3 soxl_qlib_strategy_v2.py --profiles balanced --walk-forward`
- Smoke:
  - `python3 soxl_qlib_strategy_v2.py --profiles smoke --smoke --walk-forward`
- Static validation:
  - `python3 -m py_compile soxl_qlib_strategy_v2.py src/soxl_quant/*.py`
- Paper replay:
  - `python3 production/soxl_paper_runner.py --profile balanced --start 2024-01-01 --end 2024-12-31`

## Non-leak rules

- Fixed split:
  - `train=2010-03-11..2018-12-31`
  - `valid=2019-01-01..2019-12-31`
  - `test=2020-01-01..2026-03-31`
- No special sampling by future years such as `2020` or `2022`.
- Stress oversampling only uses features visible inside the training segment.
- Any claim of improvement must include walk-forward evidence.

## Decision log

- 2026-04-10 23:24:
  - Added long-trend anchor, cumulative-drop, and volatility-coefficient features.
- 2026-04-10 23:24:
  - Replaced fixed action mapping with dynamic action bands.
- 2026-04-10 23:38:
  - Removed year-based oversampling because it could leak test-period structure.
- 2026-04-10 23:50:
  - Added walk-forward execution path and slice reporting to the CLI.
- 2026-04-10 23:57:
  - Switched repository process from ad-hoc prompting to explicit agent-loop documentation.
- 2026-04-11 13:25:
  - Added actor artifact persistence plus a paper-trading replay layer under `src/soxl_quant/runtime.py`.
- 2026-04-11 13:25:
  - Moved process guidance into `AGENTS.md` instead of maintaining a parallel `agent.md` contract.
- 2026-04-11 18:37:
  - Reworked reward toward trade quality by tolerating controlled pullbacks, penalizing stale inactivity earlier, and penalizing trapped losing holds.
- 2026-04-11 18:37:
  - Added summary sidecar persistence under `artifacts/models/soxl_<profile>_summary.json` so validation and slice metrics survive the current qlib/Python 3.14 shutdown failure.
- 2026-04-11 19:26:
  - Added deterministic seeding for `random`, `numpy`, and `torch` in the training path plus a `--seed` CLI argument and persisted seed metadata in saved artifacts.
- 2026-04-11 19:26:
  - Proved smoke reproducibility with two back-to-back seeded runs producing identical summary JSON and identical actor checkpoint hashes.
- 2026-04-11 19:43:
  - Ran full seeded balanced training and confirmed deeper training depth did not fix `2022`; the saved balanced actor still carries high exposure through drawdowns.
- 2026-04-12 19:08:
  - Wired `position_cap` config all the way through training selection, fast slice evaluation, qlib backtest, saved actor bundles, and paper replay.
- 2026-04-12 19:08:
  - Ran seeded smoke A/B on delayed hard-cap activation and found both `12%` and `15%` starts worse than the `10%` baseline, so the default start stays at `10%`.

## Current run

- Running:
  - none
- Latest validated outputs:
  - smoke artifact replay base:
    - `2024 paper = 164 commands / avg util 68.0% / end value 755,689`
  - smoke artifact replay after reward rework:
    - `2024 paper = 172 commands / avg util 57.9% / end value 848,820`
  - direct artifact slice check after first reward-rework smoke:
    - `test_2022 = -45.1% / Sharpe -0.75 / MaxDD 59.1%`
- Seeded reproducible smoke baseline:
  - `valid = 88.5% / Sharpe 1.55 / MaxDD 44.5%`
  - `test_2022 = -63.7% / Sharpe -0.70 / MaxDD 74.4%`
- Full seeded balanced result:
  - `test_2022 = -62.1% / Sharpe -0.73 / MaxDD 71.3% / avg_pos 0.776`
  - `2022` exposure behavior:
    - avg position while underwater worse than `10%` = `0.779`
    - position above `0.70` on `78.4%` of 2022 decision days
- Balanced 20-iter capped artifact:
  - `test_2022 = -25.2% / Sharpe -0.88 / MaxDD 32.2% / avg_pos 0.273`
  - `2024` paper replay:
    - `171` commands
    - `38.2%` avg utilization
    - ending value `886,043`
- Walk-forward with cap:
  - `wf_2020 = 161.7% / Sharpe 0.56 / MaxDD 69.5%`
  - `wf_2022 = 41.2% / Sharpe 0.34 / MaxDD 59.5%`
  - `wf_2023_2024` reached validation and completed the `2024` backtest loop, but the process failed before printing the final result line
  - latest overwritten summary for that final pass shows `test_2023_2024 = -11.3% / Sharpe -0.22 / MaxDD 30.4% / avg_pos 0.382`
- Hard-cap floor sweep:
  - floor `0.3` smoke:
    - `test_2022 = -47.8% / avg_pos 0.450`
    - `test_2023_2024 = 43.5% / avg_pos 0.700`
  - floor `0.4` smoke:
    - `test_2022 = -50.2% / avg_pos 0.508`
    - `test_2023_2024 = 48.8% / avg_pos 0.731`
  - floor `0.5` smoke:
    - `test_2022 = -52.6% / avg_pos 0.567`
    - `test_2023_2024 = 54.0% / avg_pos 0.762`
  - decision:
    - keep the hard cap and set the stressed-state floor to `0.4` as the current compromise parameter
- Hard-cap activation sweep with live-config plumbing fixed:
  - start `10%` smoke baseline:
    - `valid = 75.2% / Sharpe 1.62 / MaxDD 32.9% / avg_pos 0.784`
    - `test_2022 = -50.2% / Sharpe -1.01 / MaxDD 55.8% / avg_pos 0.508`
    - `test_2023_2024 = 48.8% / Sharpe 0.67 / MaxDD 49.0% / avg_pos 0.731`
  - start `12%` smoke:
    - `valid = 74.8% / Sharpe 1.61 / MaxDD 33.4% / avg_pos 0.789`
    - `test_2022 = -50.1% / Sharpe -0.99 / MaxDD 55.7% / avg_pos 0.511`
    - `test_2023_2024 = 47.5% / Sharpe 0.67 / MaxDD 49.1% / avg_pos 0.746`
  - start `15%` smoke:
    - `valid = 71.1% / Sharpe 1.54 / MaxDD 34.5% / avg_pos 0.797`
    - `test_2022 = -49.8% / Sharpe -0.94 / MaxDD 55.8% / avg_pos 0.517`
    - `test_2023_2024 = 40.1% / Sharpe 0.62 / MaxDD 50.2% / avg_pos 0.768`
  - decision:
    - drop delayed engagement as the next tuning direction
    - keep `start_drawdown = 0.10` with `full_drawdown = 0.30` and `min_cap = 0.40`
- Latest failure:
  - qlib training/backtest lifecycle still does not shut down cleanly under the current Python 3.14 environment
  - the final walk-forward `2024` window remains partially blocked by this non-clean exit pattern
  - balanced retrains still cannot be relied on to persist artifacts after full trainer completion

## Next actions

1. Keep floor `0.4` fixed and tune the cap curve shape rather than delaying activation:
  - delayed starts at `12%` and `15%` did not improve the smoke trade-off
  - next sweep should test whether the ramp to the `0.40` floor should be less linear or should reach full restriction later than `30%` without moving the initial trigger off `10%`
2. Keep measuring whether late-period weakness is specifically `2024`-regime related:
  - use walk-forward and replay together rather than only the fixed `2022` slice
3. Eliminate or isolate the Python 3.14 qlib shutdown failure so walk-forward windows can complete and print final metrics cleanly.
4. Add a broker adapter interface for Futu/IB paper execution without changing the actor or replay code.

## Run completed 2026-04-11 17:48

- Command: `python3 soxl_qlib_strategy_v2.py --profiles balanced --walk-forward`
- balanced: 368.7% / Sharpe 0.81 / MaxDD 51.6% / 894 trades
- test_2022: **-26.5%** (improved from -39.8%)
- wf_2023_2024: 1.1% / Sharpe 0.10 / MaxDD 38.8% ⚠️ weak
- Trade-off: 2022 better but overall return down from ~510%
- Next hypothesis: reward rework helped drawdown control but may have over-penalized upside in trending periods. Need to investigate why walk-forward 2023-2024 collapsed.

## Harness / production boundary implementation 2026-04-11 21:xx

- Hypothesis:
  - the project is blocked less by another reward tweak than by weak execution boundaries
  - if replay, risk gating, and broker execution share one contract, we can productionize safely without forking the logic again
- Implementation:
  - refactored `src/soxl_quant/runtime.py` around an explicit `BrokerAdapter`
  - added structured runtime records:
    - `SignalIntent`
    - `RiskGateDecision`
    - `BrokerExecutionResult`
    - `ReconciliationRecord`
  - kept `PaperBroker` as the contract-test implementation and upgraded replay output to persist:
    - `risk_gates.jsonl`
    - `executions.jsonl`
    - `reconcile.jsonl`
  - added Interactive Brokers semi-auto scaffolding:
    - `production/ib_client.py`
    - `production/ib_broker.py`
    - `production/ib_manual_session.py`
    - `production/IB_SEMI_AUTO.md`
  - added high-value comments/docstrings across:
    - `src/soxl_quant/rl.py`
    - `src/soxl_quant/cli.py`
    - `src/soxl_quant/data.py`
    - `src/soxl_quant/backtest.py`
    - `production/soxl_paper_runner.py`
- Validation:
  - `python3 -m py_compile src/soxl_quant/runtime.py src/soxl_quant/rl.py src/soxl_quant/cli.py src/soxl_quant/data.py src/soxl_quant/backtest.py production/soxl_paper_runner.py production/ib_client.py production/ib_broker.py production/ib_manual_session.py tests/test_runtime.py`
    - passed
  - direct runtime contract checks:
    - `test_position_ledger_tracks_entry_and_age()`
    - `test_paper_broker_creates_lot_sized_orders()`
    - `test_risk_gate_blocks_large_one_shot_position_jump()`
    - `test_reconcile_positions_flags_meaningful_gap()`
    - result: `runtime tests: OK`
- Keep / drop decision:
  - keep the broker-adapter runtime split
  - keep IB as the first-class semi-auto target
  - do not claim live-ready yet; actual IB cancellation/fill reconciliation remains incomplete
- Next action:
  - wire the checkpoint gate to `2022` / walk-forward thresholds before using the new semi-auto path for any real account

## Promotion-gate wiring 2026-04-12 20:xx

- Hypothesis:
  - semi-auto execution is still unsafe if model promotion remains implicit
  - the saved actor bundle and summary need an explicit pass/fail verdict based on fixed-split and walk-forward thresholds
- Implementation:
  - added `PromotionGateConfig` to `src/soxl_quant/settings.py`
  - added `promotion_gate` blocks to:
    - `configs/soxl/balanced.yaml`
    - `configs/soxl/smoke.yaml`
    - `configs/soxl/aggressive.yaml`
  - updated `src/soxl_quant/cli.py` to compute and persist:
    - fixed-split promotion checks from `test_2022` drawdown / average exposure and `test_proxy` Sharpe
    - walk-forward promotion checks from `wf_2023_2024` return / Sharpe
    - final `promotion.eligible` verdict in the summary sidecar
  - updated `src/soxl_quant/runtime.py` so `plan_manual_session()` blocks execution when:
    - the summary sidecar is missing
    - walk-forward promotion was never evaluated
    - the model failed the promotion gate
  - added runtime coverage for the non-promoted-model block path

## Walk-forward worker isolation 2026-04-12 21:xx

- Hypothesis:
  - the remaining Python `3.14` / qlib blocker is mostly a process-lifecycle problem, not a metric-computation problem
  - if each walk-forward window runs in an isolated worker and writes its result before teardown, the parent harness can still complete even when qlib cleanup is noisy
- Implementation:
  - updated `src/soxl_quant/cli.py` to add:
    - compact cross-process result serialization via `compact_result()`
    - hidden worker arguments for a single train/valid/test slice
    - `run_worker_profile()` which writes the worker JSON and exits via `os._exit(0)`
    - `run_walk_forward_isolated()` which executes each walk-forward window in a subprocess and reloads the compact JSON
  - main CLI now uses isolated walk-forward workers instead of in-process window execution
- Validation:
  - `python3 -m py_compile src/soxl_quant/cli.py soxl_qlib_strategy_v2.py`
    - passed
  - compact payload check:
    - `compact_result: OK`
- Keep / drop decision:
  - keep the worker-isolation harness
  - next validation should rerun a full `--walk-forward` pass and confirm the last window now always leaves a readable result, even if worker shutdown is noisy
- Validation:
  - `python3 -m py_compile src/soxl_quant/settings.py src/soxl_quant/cli.py src/soxl_quant/runtime.py tests/test_runtime.py tests/test_position_cap.py tests/test_reward.py`
    - passed
  - `python3 tests/test_position_cap.py`
    - passed with `test_position_cap: OK`
  - direct promotion-gate smoke check in Python:
    - balanced gate with `test_2022 max_dd=34.6`, `avg_pos=0.299`, `test_proxy sharpe=0.81`, `wf_2023_2024 return=1.1`, `sharpe=0.10`
    - returned `eligible=True`
  - direct runtime contract checks:
    - `runtime tests: OK`
- Keep / drop decision:
  - keep the promotion gate as a hard requirement for any semi-auto execution path
  - do not allow runtime to infer “probably okay” from raw metrics when summary promotion is missing
- Next action:
  - rerun a full balanced `--walk-forward` cycle with the latest code so the persisted `soxl_balanced_summary.json` gets a real promotion verdict from actual outputs

## Walk-forward artifact isolation 2026-04-12 23:xx

- Hypothesis:
  - the isolated worker harness is still not promotion-safe if each worker rewrites `artifacts/models/soxl_<profile>_actor.pt` and `..._summary.json`
  - canonical artifact names must stay reserved for the fixed-split training pass, while walk-forward windows write namespaced worker bundles
- Implementation:
  - added artifact path helpers in `src/soxl_quant/settings.py`:
    - `artifact_stem()`
    - `model_artifact_path()`
    - `summary_artifact_path()`
  - updated `src/soxl_quant/cli.py` so `run_profile()` accepts an optional `artifact_tag`
  - wired walk-forward window names through both in-process and isolated worker paths, so workers now persist bundles like:
    - `soxl_balanced_wf_2023_2024_actor.pt`
    - `soxl_balanced_wf_2023_2024_summary.json`
  - kept canonical names unchanged for the primary fixed-split artifact:
    - `soxl_balanced_actor.pt`
    - `soxl_balanced_summary.json`
  - added focused coverage in `tests/test_artifact_paths.py`
- Validation:
  - `python3 -m py_compile src/soxl_quant/settings.py src/soxl_quant/cli.py tests/test_artifact_paths.py`
    - passed
  - `python3 tests/test_artifact_paths.py`
    - passed with `test_artifact_paths: OK`
- Keep / drop decision:
  - keep worker artifact namespacing
  - do not let walk-forward workers write the canonical promotion artifact paths again
- Next action:
  - rerun `python3 soxl_qlib_strategy_v2.py --profiles balanced --walk-forward --skip-data-refresh --seed 20260411` in the proper quant environment and confirm:
    - the parent summary stays on `soxl_balanced_summary.json`
    - worker windows leave separate `wf_*` artifacts
    - `promotion.walk_forward_evaluated` persists deterministically after the final window

## Walk-forward validation refresh 2026-04-12 23:xx

- Hypothesis:
  - the isolated worker harness plus worker artifact namespacing is now sufficient to let the full seeded balanced walk-forward pass finish cleanly enough for deterministic promotion writeback
- Environment:
  - created local `qlib_venv` with `python3.11`
  - pinned `tianshou==0.4.10` because `pyqlib 0.9.7` still imports `tianshou.policy`
  - lifted `protobuf` to `5.29.5` so `mlflow` and `opentelemetry` imported cleanly during `qlib.init()`
- Command:
  - `qlib_venv/bin/python soxl_qlib_strategy_v2.py --profiles balanced --walk-forward --skip-data-refresh --seed 20260411`
- Result:
  - the full seeded balanced walk-forward completed end-to-end and printed the final `[done]` summary
  - canonical fixed-split artifact stayed on:
    - `artifacts/models/soxl_balanced_actor.pt`
    - `artifacts/models/soxl_balanced_summary.json`
  - isolated workers persisted their own artifacts and compact payloads:
    - `artifacts/models/soxl_balanced_wf_2020_summary.json`
    - `artifacts/models/soxl_balanced_wf_2022_summary.json`
    - `artifacts/models/soxl_balanced_wf_2023_2024_summary.json`
    - `runtime/soxl/wf_workers/balanced-wf_2020-seed20260411.json`
    - `runtime/soxl/wf_workers/balanced-wf_2022-seed20260411.json`
    - `runtime/soxl/wf_workers/balanced-wf_2023_2024-seed20260411.json`
  - parent fixed-split summary now has deterministic promotion writeback:
    - `promotion.walk_forward_evaluated = true`
    - `promotion.eligible = true`
- Key metrics:
  - fixed split:
    - `test_2020 = 92.9% / Sharpe 2.30 / MaxDD 17.1% / avg_pos 0.461`
    - `test_2022 = -26.5% / Sharpe -0.84 / MaxDD 34.6% / avg_pos 0.299`
    - `test_2023_2024 = 46.0% / Sharpe 0.73 / MaxDD 30.5% / avg_pos 0.458`
    - `test_full = 368.7% / Sharpe 0.80 / MaxDD 51.6% / trades 894`
  - walk-forward:
    - `wf_2020 = 167.7% / Sharpe 0.57 / MaxDD 69.4% / trades 113`
    - `wf_2022 = 53.3% / Sharpe 0.38 / MaxDD 65.4% / trades 127`
    - `wf_2023_2024 = 1.1% / Sharpe 0.10 / MaxDD 38.8% / trades 123`
  - promotion verdict:
    - fixed split passed:
      - `test_2022 max_dd = 34.57 <= 35`
      - `test_2022 avg_pos = 0.299 <= 0.35`
      - `test_proxy sharpe = 0.815 >= 0.75`
    - walk-forward passed:
      - `wf_2023_2024 return = 1.11 >= 0`
      - `wf_2023_2024 sharpe = 0.10 >= 0`
- Failure mode:
  - worker and parent shutdown still emit `DataQueue.__del__` and `resource_tracker` warnings
  - despite the noise, artifacts and promotion summary persisted correctly under the isolated worker harness
- Keep / drop decision:
  - keep the isolated worker harness and namespaced worker artifacts
  - keep the balanced seeded promotion result as the new validated evidence baseline
  - drop the assumption that Python `3.14` is the only reproducer; the same qlib cleanup warnings still appear under the local `python3.11` environment, but they no longer block promotion persistence
- Next action:
  - move from harness validation back to strategy improvement, using this refreshed promoted balanced bundle as the current documented baseline
