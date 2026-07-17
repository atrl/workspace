# SOXL Risk-Adjusted Model Frontier Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compare tail-first, regime-mixture, and compact sequence models under one qlib-native risk-adjusted allocation and promotion contract.

**Architecture:** Build shared leakage-safe labels, soft-budget mapping, and risk gates first; implement each model route as a separate module and research card; then create a frontier card that compares only frozen recorder-backed results. Keep all large daily predictions, signals, positions, and portfolio reports in qlib recorders.

**Tech Stack:** Python 3.11, qlib 0.9.7, pandas, numpy, scikit-learn, LightGBM, CatBoost, PyTorch, pytest, Jupyter nbconvert

---

### Task 1: Shared Labels, Positions, And Promotion Gates

**Files:**
- Create: `src/model/risk_adjusted_common.py`
- Modify: `src/model/__init__.py`
- Create: `tests/test_risk_adjusted_model_common.py`

- [ ] Write failing tests for train-only crash thresholds, horizon purging,
  future-perturbation leakage, continuous position floor/cap behavior,
  fail-closed data quality, and the full risk-adjusted promotion gate.
- [ ] Run `PYTHONPATH=. /Users/rziiiii/dev/workspace/qlib_venv/bin/python -m pytest tests/test_risk_adjusted_model_common.py -q` and confirm failure is caused by the missing shared API.
- [ ] Implement immutable segment contracts, label construction, soft-budget
  mapping, risk metrics, walk-forward aggregation, and promotion evaluation.
- [ ] Rerun the focused test and require exit code 0.
- [ ] Commit `feat: add shared risk adjusted model contracts`.

### Task 2: Tail-First Model Route

**Files:**
- Create: `src/model/tail_budget.py`
- Modify: `src/model/__init__.py`
- Create: `tests/test_soxl_tail_budget.py`

- [ ] Write failing tests that fit expected-return, downside, and crash models
  through a qlib Dataset; select blends on validation only; and produce bounded
  soft positions without non-finite values.
- [ ] Run the focused test and confirm the model route is absent.
- [ ] Implement ridge/LGB expected-return candidates, CatBoost quantile/crash
  candidates, qlib DEnsemble downside candidate, validation-only blending, and
  deterministic failure rows for unavailable candidates.
- [ ] Rerun shared and tail-route tests and require exit code 0.
- [ ] Commit `feat: add tail first SOXL budget models`.

### Task 3: Regime Mixture-Of-Experts Route

**Files:**
- Create: `src/model/regime_moe.py`
- Modify: `src/model/__init__.py`
- Create: `tests/test_soxl_regime_moe.py`

- [ ] Write failing tests for train-only regime thresholds, minimum regime
  occupancy, soft gate probabilities, specialist predictions, and position
  bounds.
- [ ] Run the focused test and confirm failure is caused by the missing route.
- [ ] Implement the trend ridge, reversal CatBoost, stress downside expert,
  multinomial soft gate, and explicit `blocked_sparse_regime` result.
- [ ] Rerun shared and mixture-route tests and require exit code 0.
- [ ] Commit `feat: add SOXL regime mixture models`.

### Task 4: Compact Multivariate Sequence Route

**Files:**
- Create: `src/model/panel_sequence.py`
- Modify: `src/model/__init__.py`
- Create: `tests/test_soxl_panel_sequence.py`

- [ ] Write failing tests for leakage-safe 20/60-day windows, deterministic
  seeds, compact TCN/GRU/Transformer candidates, early stopping, prediction
  alignment, and the PIT constituent-panel gate.
- [ ] Run the focused test and confirm failure is caused by the missing route.
- [ ] Implement compact PyTorch qlib-compatible estimators and the diagnostic
  SOXL/SOXX/VIX multivariate mode; block only the cross-sectional mode when PIT
  holdings are absent.
- [ ] Rerun shared and sequence-route tests and require exit code 0.
- [ ] Commit `feat: add compact SOXL sequence models`.

### Task 5: Qlib Workflow And Four Slim Research Cards

**Files:**
- Create: `src/workflow/soxl_risk_adjusted_frontier.py`
- Modify: `src/workflow/__init__.py`
- Create: `tests/test_soxl_risk_adjusted_frontier_workflow.py`
- Modify: `tests/test_qlib_native_notebook_cards.py`
- Create: `research/soxl_tail_budget_v1/`
- Create: `research/soxl_regime_moe_v1/`
- Create: `research/soxl_panel_sequence_v1/`
- Create: `research/soxl_risk_adjusted_model_frontier_v1/`

- [ ] Write failing workflow and card-contract tests for qlib initialization,
  DatasetH use, recorder evidence, relative paths, artifact policy, frozen-test
  comparison, blocked-route retention, and Chinese conclusion cells.
- [ ] Run the focused workflow/card tests and confirm expected missing-surface
  failures.
- [ ] Implement canonical qlib data loading, segment construction, route
  orchestration, recorder logging, compact sidecars, notebook generation, and
  frontier comparison without test-time reselection.
- [ ] Execute all four notebooks with nbconvert in dependency order.
- [ ] Rerun route, workflow, and card-contract tests and require exit code 0.
- [ ] Commit `feat: build SOXL risk adjusted model frontier cards`.

### Task 6: Final Review, Verification, And Delivery

**Files:**
- Refresh only the four new card roots and their referenced recorder runs.

- [ ] Request spec-compliance review and fix all missing or extra behavior.
- [ ] Request code-quality review and fix every Critical or Important issue.
- [ ] Run all focused route/workflow/notebook-card tests.
- [ ] Run the repo qlib workflow/model-selection contract tests.
- [ ] Run the applicable broad suite and separately report verified pre-existing
  failures rather than claiming a false all-green result.
- [ ] Execute `git diff --check`, inspect `git status --short`, inspect the
  staged diff, and verify every card references its final recorder.
- [ ] Commit final fixes, push `feature/soxl-risk-adjusted-model-frontier-v1`,
  and report each route's metrics and promotion decision.
