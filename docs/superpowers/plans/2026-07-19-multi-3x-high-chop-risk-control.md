# Multi-3x High-Chop Risk Control V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and execute a qlib-native SOXL/SOXX/cash risk-control research card that learns transferable 20-day path states from an expanded long-equity 3x ETF panel and converts calibrated breakdown risk into transparent, constrained portfolio weights.

**Architecture:** Extend the existing pair-data registry, create a focused feature/label module, add an asymmetric two-asset portfolio controller, and compose them in one recorder-first research workflow. Model selection is nested and validation-only; the notebook reports stitched out-of-sample SOXL results against direct static and legacy benchmarks.

**Tech Stack:** Python 3.11, pandas, NumPy, scikit-learn, LightGBM, Microsoft Qlib `D.features`/`DatasetH`/`DataHandlerLP`/Recorder, pytest, nbconvert.

---

## File Map

- Modify `src/data/multi_3x_etf_update.py`: expanded reusable ETF/underlying registry.
- Create `src/factor/multi_3x_high_chop.py`: lagged normalized features, cross-asset breadth, continuous chop intensity, and mutually exclusive path labels.
- Create `src/backtest/multi_asset_risk_control.py`: SOXL/SOXX/cash continuous mapping, hard gates, asymmetric pacing, drawdown budget, costs, and metrics.
- Create `src/workflow/multi_3x_high_chop_risk_control.py`: qlib provider build/readback, DatasetH evidence, nested model selection, SOXL outer folds, recorder, card and notebook outputs.
- Modify `src/factor/__init__.py`, `src/backtest/__init__.py`, and `src/workflow/__init__.py`: public exports only.
- Create `tests/test_multi_3x_high_chop_factors.py`.
- Create `tests/test_multi_asset_risk_control.py`.
- Create `tests/test_multi_3x_high_chop_risk_control_workflow.py`.
- Modify `tests/test_multi_3x_etf_data_update.py` and `tests/test_qlib_native_notebook_cards.py`.
- Create `research/multi_3x_high_chop_risk_control_v1/` through the executed notebook workflow.

### Task 1: Expanded Pair Registry

**Files:**
- Modify: `src/data/multi_3x_etf_update.py`
- Modify: `tests/test_multi_3x_etf_data_update.py`

- [ ] **Step 1: Write the failing registry contract test**

```python
def test_multi_3x_registry_contains_diversified_long_equity_pairs():
    from src.data.multi_3x_etf_update import MULTI_3X_ETF_PAIRS

    pairs = {(item.etf, item.underlying) for item in MULTI_3X_ETF_PAIRS}
    assert len(pairs) == 15
    assert {("TNA", "IWM"), ("FAS", "XLF"), ("LABU", "XBI"), ("ERX", "XLE")} <= pairs
    assert not any(etf.startswith("S") and etf in {"SQQQ", "SPXU"} for etf, _ in pairs)
```

- [ ] **Step 2: Run the test and verify the old four-pair registry fails**

Run: `PYTHONPATH=. /Users/rziiiii/dev/workspace/qlib_venv/bin/python -m pytest tests/test_multi_3x_etf_data_update.py::test_multi_3x_registry_contains_diversified_long_equity_pairs -q`

Expected: FAIL because the registry has four entries.

- [ ] **Step 3: Expand the fixed registry**

```python
MULTI_3X_ETF_PAIRS = (
    Multi3xEtfPairSpec("TQQQ", "QQQ"), Multi3xEtfPairSpec("UPRO", "SPY"),
    Multi3xEtfPairSpec("SPXL", "SPY"), Multi3xEtfPairSpec("TECL", "XLK"),
    Multi3xEtfPairSpec("UDOW", "DIA"), Multi3xEtfPairSpec("TNA", "IWM"),
    Multi3xEtfPairSpec("MIDU", "MDY"), Multi3xEtfPairSpec("FAS", "XLF"),
    Multi3xEtfPairSpec("LABU", "XBI"), Multi3xEtfPairSpec("CURE", "XLV"),
    Multi3xEtfPairSpec("NAIL", "ITB"), Multi3xEtfPairSpec("RETL", "XRT"),
    Multi3xEtfPairSpec("ERX", "XLE"), Multi3xEtfPairSpec("DRN", "XLRE"),
    Multi3xEtfPairSpec("WANT", "XLY"),
)
```

- [ ] **Step 4: Run the complete data-update unit file**

Run: `PYTHONPATH=. /Users/rziiiii/dev/workspace/qlib_venv/bin/python -m pytest tests/test_multi_3x_etf_data_update.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit the registry**

```bash
git add src/data/multi_3x_etf_update.py tests/test_multi_3x_etf_data_update.py
git commit -m "feat: expand long equity 3x ETF registry"
```

### Task 2: Transferable Features And Path Labels

**Files:**
- Create: `src/factor/multi_3x_high_chop.py`
- Modify: `src/factor/__init__.py`
- Create: `tests/test_multi_3x_high_chop_factors.py`

- [ ] **Step 1: Write failing tests for path labels, past-only normalization, and PIT isolation**

```python
def test_path_labels_are_mutually_exclusive_and_tail_is_purged():
    out = build_multi_3x_high_chop_frame(_synthetic_panel(), label_horizon=20)
    labels = out["path_label"].dropna().astype(int)
    assert set(labels.unique()) <= {0, 1, 2, 3}
    assert out.groupby("instrument").tail(21)["path_label"].isna().all()

def test_future_mutation_does_not_change_prior_features():
    original = build_multi_3x_high_chop_frame(_synthetic_panel())
    changed_panel = _synthetic_panel()
    changed_panel.loc[changed_panel.datetime > "2024-06-30", "close"] *= 2
    changed = build_multi_3x_high_chop_frame(changed_panel)
    assert original.loc[original.datetime == "2024-06-28", MODEL_FEATURES].equals(
        changed.loc[changed.datetime == "2024-06-28", MODEL_FEATURES]
    )

def test_constituent_columns_are_diagnostic_only():
    assert not any(name.startswith("soxx_const_") for name in TRANSFERABLE_FEATURES)
```

- [ ] **Step 2: Verify the new module is missing**

Run: `PYTHONPATH=. /Users/rziiiii/dev/workspace/qlib_venv/bin/python -m pytest tests/test_multi_3x_high_chop_factors.py -q`

Expected: collection FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement the feature contract**

Implement these public names:

```python
PATH_LABELS = {"benign_chop": 0, "breakout": 1, "breakdown": 2, "whipsaw": 3}
TRANSFERABLE_FEATURES = (
    "ret_5_pctile252", "ret_20_pctile252", "ret_60_pctile252",
    "range_pos_252", "dist_high_252", "drawdown_60_pctile252",
    "trend_efficiency_20_pctile252", "sign_change_density_20_pctile252",
    "rv_20_pctile252", "downside_semivariance_20_pctile252",
    "tracking_error_20_pctile252", "tracking_error_expansion_pctile252",
    "realized_vix_gap", "range_vix_gap", "abs_vix_change_5_pctile252",
    "cross_asset_dispersion", "cross_asset_positive_breadth",
    "cross_asset_downside_breadth", "cohort_leadership_turnover",
    "high_chop_intensity",
)
```

Implement `build_multi_3x_high_chop_frame(panel, label_horizon=20)` and
`purge_segment_tail(frame, end, horizon=20)` with annualized `rv_20`,
`sigma_20 = rv_20 * sqrt(20/252)`, label priority
whipsaw → breakdown → breakout → benign, and shift all model features by one
session within asset. Cross-asset breadth is computed after per-asset lagging.

- [ ] **Step 4: Run focused factor tests**

Run: `PYTHONPATH=. /Users/rziiiii/dev/workspace/qlib_venv/bin/python -m pytest tests/test_multi_3x_high_chop_factors.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit the factor layer**

```bash
git add src/factor/multi_3x_high_chop.py src/factor/__init__.py tests/test_multi_3x_high_chop_factors.py
git commit -m "feat: add transferable high-chop path factors"
```

### Task 3: SOXL/SOXX/Cash Risk Controller

**Files:**
- Create: `src/backtest/multi_asset_risk_control.py`
- Modify: `src/backtest/__init__.py`
- Create: `tests/test_multi_asset_risk_control.py`

- [ ] **Step 1: Write failing controller tests**

```python
def test_probability_confirmation_is_required_for_hard_gate():
    result = apply_multi_asset_risk_control(_controller_frame(p_breakdown=0.40, price_break=True, breadth_break=True))
    assert result.iloc[-1].hard_gate_active is False

def test_three_confirmations_zero_soxl_and_cap_soxx():
    result = apply_multi_asset_risk_control(_controller_frame(p_breakdown=0.80, price_break=True, breadth_break=True))
    assert result.iloc[-1].target_soxl == 0.0
    assert result.iloc[-1].target_soxx <= 0.10

def test_weights_and_asymmetric_pacing():
    result = apply_multi_asset_risk_control(_controller_sequence())
    assert (result[["target_soxl", "target_soxx", "target_cash"]] >= 0).all().all()
    assert np.allclose(result[["target_soxl", "target_soxx", "target_cash"]].sum(axis=1), 1.0)
    assert result.target_soxl.diff().max() <= 0.20 + 1e-12
```

- [ ] **Step 2: Verify module absence fails**

Run: `PYTHONPATH=. /Users/rziiiii/dev/workspace/qlib_venv/bin/python -m pytest tests/test_multi_asset_risk_control.py -q`

Expected: collection FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement controller and evaluation APIs**

Create the frozen `MultiAssetRiskControlSpec` with the exact defaults from the
approved design: target volatility 35%, SOXX replacement 50%, breakdown and
recovery thresholds 0.65/0.40, hard caps 15%/25%/10%, drawdown trigger and
recovery -8%/-4%, risky cap 25%, cooldown five days, rebalance threshold 10%,
risk-up step 20%, and minimum state duration three days. Expose
`apply_multi_asset_risk_control(frame, spec)` and
`evaluate_multi_asset_risk_control(frame, weights, cost_per_turnover)`.

The state loop must update drawdown from prior open-to-open strategy returns,
allow immediate risk-down changes, limit combined risk-up changes, calculate
SOXL and SOXX turnover separately, and make cash the residual.

- [ ] **Step 4: Run controller tests**

Run: `PYTHONPATH=. /Users/rziiiii/dev/workspace/qlib_venv/bin/python -m pytest tests/test_multi_asset_risk_control.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit controller**

```bash
git add src/backtest/multi_asset_risk_control.py src/backtest/__init__.py tests/test_multi_asset_risk_control.py
git commit -m "feat: add asymmetric SOXL SOXX risk controller"
```

### Task 4: Qlib Workflow And Model Selection

**Files:**
- Create: `src/workflow/multi_3x_high_chop_risk_control.py`
- Modify: `src/workflow/__init__.py`
- Create: `tests/test_multi_3x_high_chop_risk_control_workflow.py`

- [ ] **Step 1: Write failing workflow contract tests**

```python
def test_underlying_groups_keep_spy_products_together():
    assert UNDERLYING_GROUPS["UPRO"] == UNDERLYING_GROUPS["SPXL"] == "SPY"

def test_candidate_selection_uses_validation_only():
    selected = select_risk_control_candidate(_candidate_metrics())
    assert selected["selection_uses_outer_test"] is False
    assert selected["candidate"] == "multinomial_logistic"

def test_dataset_contract_uses_qlib_dataset(tmp_path):
    dataset = build_path_dataset(
        _tiny_provider(tmp_path),
        train=("2022-01-03", "2022-12-30"),
        valid=("2023-01-03", "2023-12-29"),
        test=("2024-01-02", "2024-12-31"),
    )
    assert dataset.qlib_dataset_api == ["DatasetH", "DataHandlerLP", "QlibDataLoader"]
```

- [ ] **Step 2: Verify workflow module absence fails**

Run: `PYTHONPATH=. /Users/rziiiii/dev/workspace/qlib_venv/bin/python -m pytest tests/test_multi_3x_high_chop_risk_control_workflow.py -q`

Expected: collection FAIL.

- [ ] **Step 3: Implement workflow boundaries**

Define `EXPERIMENT = "multi_3x_high_chop_risk_control_v1"` and the complete
underlying groups: SOXL/SOXX, TQQQ/QQQ, TECL/XLK, UPRO+SPXL/SPY, UDOW/DIA,
TNA/IWM, MIDU/MDY, FAS/XLF, LABU/XBI, CURE/XLV, NAIL/ITB, RETL/XRT, ERX/XLE,
DRN/XLRE, and WANT/XLY. Create `Multi3xHighChopRiskControlSpec` with outer
years 2023-2026 and 3-basis-point base turnover cost. Expose
`run_multi_3x_high_chop_risk_control_card`, `build_path_dataset`, and
`select_risk_control_candidate`.

Build the normalized qlib provider, read it back with `D.features`, instantiate
`DatasetH`, compare rule/logistic/LightGBM multiclass candidates, calibrate on
validation only, and stitch outer-year SOXL weights. Require per-fold class
minimums and log blocked folds instead of selecting an oracle.

- [ ] **Step 4: Implement benchmark and promotion summaries**

The workflow writes compact tables for model selection, calibration, annual
walk-forward, benchmark comparison, risk analysis, regime metrics, window
metrics, and failure modes. Promotion evaluates every fixed gate from the
design and returns `not_promoted_risk_control_gates_failed` if any gate fails.

- [ ] **Step 5: Run workflow tests**

Run: `PYTHONPATH=. /Users/rziiiii/dev/workspace/qlib_venv/bin/python -m pytest tests/test_multi_3x_high_chop_risk_control_workflow.py -q`

Expected: all tests pass.

- [ ] **Step 6: Commit workflow**

```bash
git add src/workflow/multi_3x_high_chop_risk_control.py src/workflow/__init__.py tests/test_multi_3x_high_chop_risk_control_workflow.py
git commit -m "feat: add qlib high-chop risk-control workflow"
```

### Task 5: Live Data, Executed Notebook, And Card Contract

**Files:**
- Create/update: `data/processed/panels/*_with_underlying_vix.csv`
- Create/update: `data/qlib/<provider>/`
- Create: `research/multi_3x_high_chop_risk_control_v1/*`
- Modify: `tests/test_qlib_native_notebook_cards.py`

- [ ] **Step 1: Refresh the expanded pair data through the settled 2026-07-02 endpoint**

Run a Python entry call with `pairs=MULTI_3X_ETF_PAIRS`, `start="2008-01-01"`,
`end="2026-07-02"`, `replace_overlap=True`. Do not use a pandas-only research
path; verify every generated provider manifest and qlib readback audit.

- [ ] **Step 2: Add the failing notebook-card contract**

Assert the experiment name, qlib Dataset APIs, 16 research symbols including
SOXL, next-open timing, four path classes, validation-only selection, portfolio
weight invariants, promotion gates, slim-card policy, expected compact
sidecars, and recorder-only large artifacts.

- [ ] **Step 3: Generate and execute the notebook**

Run: `PYTHONPATH=. /Users/rziiiii/dev/workspace/qlib_venv/bin/python -m nbconvert --execute research/multi_3x_high_chop_risk_control_v1/multi_3x_high_chop_risk_control_v1.ipynb --to notebook --inplace`

Expected: exit 0 and a final Chinese conclusion based on the generated summary.

- [ ] **Step 4: Run card and focused integration tests**

Run: `PYTHONPATH=. /Users/rziiiii/dev/workspace/qlib_venv/bin/python -m pytest tests/test_multi_3x_etf_data_update.py tests/test_multi_3x_high_chop_factors.py tests/test_multi_asset_risk_control.py tests/test_multi_3x_high_chop_risk_control_workflow.py tests/test_qlib_native_notebook_cards.py -q`

Expected in the worktree: new-card tests pass; pre-existing absolute-path card
tests may retain the documented worktree-only failures. Re-run the whole set in
the main checkout after integration for the authoritative path-portability
result.

- [ ] **Step 5: Inspect generated artifacts and commit only durable files**

Keep the executed notebook, compact sidecars, provider manifests, and required
canonical snapshots. Exclude transient SDK logs, caches, and root-level large
prediction/position files. Commit the verified card and data lineage.

### Task 6: Final Verification And Delivery

**Files:**
- Verify all changed files and generated artifacts.

- [ ] **Step 1: Run focused verification**

```bash
PYTHONPATH=. /Users/rziiiii/dev/workspace/qlib_venv/bin/python -m pytest \
  tests/test_multi_3x_etf_data_update.py \
  tests/test_multi_3x_high_chop_factors.py \
  tests/test_multi_asset_risk_control.py \
  tests/test_multi_3x_high_chop_risk_control_workflow.py -q
```

- [ ] **Step 2: Run the repository-mandated qlib tests**

```bash
PYTHONPATH=. /Users/rziiiii/dev/workspace/qlib_venv/bin/python -m pytest \
  tests/test_qlib_native_workflow.py \
  tests/test_qlib_model_selection_validation.py \
  tests/test_qlib_native_notebook_cards.py -q
```

- [ ] **Step 3: Run the full suite**

Run: `PYTHONPATH=. /Users/rziiiii/dev/workspace/qlib_venv/bin/python -m pytest -q`

Expected: no new failures. Any worktree-only absolute-path failures must be
rechecked after integrating the branch into main.

- [ ] **Step 4: Verify source control scope**

Run `git status --short`, `git diff --check`, inspect the staged diff, confirm
no recorder-only root CSVs or caches are staged, commit the remaining verified
changes, and push `feature/multi-3x-high-chop-risk-control-v1`.

- [ ] **Step 5: Report evidence**

Report selected model/controller, annual and stitched return, volatility,
maximum drawdown, CVaR, upside capture, exposure, turnover, benchmark deltas,
stress-window behavior, promotion decision, tests, commit, and pushed branch.
