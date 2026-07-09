# Hardware Cycle CogAlpha Daily Mining Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Guarantee daily qlib-aligned hardware-cycle data semantics and rerun CognitiveAlpha factor mining on the expanded hardware/cloud GPU feature set.

**Architecture:** Raw market proxies and AIMultiple cloud GPU points are converted into a daily qlib provider. Monthly cloud GPU values remain carried daily state, while update flags, carried flags, and staleness fields make the low-frequency provenance explicit. CognitiveAlpha mines lagged hardware candidates and writes a slim qlib research card with recorder-first evidence.

**Tech Stack:** Python, pandas, qlib Data API/Workflow Recorder, pytest, nbconvert.

---

### Task 1: Daily Cloud State Semantics

**Files:**
- Modify: `tests/test_hardware_cycle_data.py`
- Modify: `src/data/hardware_cycle.py`

- [ ] **Step 1: Write failing data test**

Add a test that builds a short daily calendar with one cloud GPU value on a trading day and another on a weekend. Assert the first trading day on or after the source date is marked `source_observed=1`, carried days are marked `is_carried=1`, and `staleness_days` increments between updates.

- [ ] **Step 2: Verify red**

Run:

```bash
PYTHONPATH=. qlib_venv/bin/python -m pytest tests/test_hardware_cycle_data.py::test_build_hardware_cycle_daily_frame_marks_cloud_state_updates_and_staleness -q
```

Expected: FAIL because the new state columns do not exist.

- [ ] **Step 3: Implement daily state columns**

Update `_pivot_cloud_gpu_index` so each cloud field writes:

```text
<field>
<field>_available
<field>_source_observed
<field>_is_carried
<field>_staleness_days
```

Map non-trading source dates to the first qlib calendar date on or after the source timestamp.

- [ ] **Step 4: Verify green**

Run the same focused data test, then all hardware data tests.

### Task 2: Lagged Hardware Factor Surface

**Files:**
- Modify: `tests/test_hardware_cycle_factors.py`
- Modify: `src/factor/hardware_cycle.py`

- [ ] **Step 1: Write failing factor test**

Assert `build_hardware_cycle_factor_frame` exposes lagged update, carried, and staleness features from the qlib provider.

- [ ] **Step 2: Verify red**

Run:

```bash
PYTHONPATH=. qlib_venv/bin/python -m pytest tests/test_hardware_cycle_factors.py::test_hardware_cycle_factor_frame_lags_cloud_state_features -q
```

Expected: FAIL because those factor columns are missing.

- [ ] **Step 3: Implement factor columns**

For each cloud GPU base field, add lagged state features:

```text
hardware_<field>_source_observed_lag1
hardware_<field>_is_carried_lag1
hardware_<field>_staleness_days_lag1
```

- [ ] **Step 4: Verify green**

Run all hardware factor tests.

### Task 3: CognitiveAlpha Hardware Candidate Library

**Files:**
- Create: `src/factor/hardware_cycle_cogalpha.py`
- Modify: `src/factor/__init__.py`
- Create: `tests/test_hardware_cycle_cogalpha_mining.py`

- [ ] **Step 1: Write failing candidate test**

Assert `build_hardware_cycle_cogalpha_candidates` returns no-lookahead candidates, separates proxy-promotable and cloud-diagnostic roles, and materializes non-constant series from a sample frame.

- [ ] **Step 2: Verify red**

Run:

```bash
PYTHONPATH=. qlib_venv/bin/python -m pytest tests/test_hardware_cycle_cogalpha_mining.py::test_hardware_cycle_cogalpha_candidates_are_lagged_and_materialize -q
```

Expected: FAIL because the module does not exist.

- [ ] **Step 3: Implement candidate library**

Build deterministic candidates for accelerator/cloud scarcity, capex confirmation, memory bottleneck spread, server supply-chain confirmation, CPU compute breadth, and storage cycle confirmation. Every candidate uses already lagged hardware features and declares zero lookahead.

- [ ] **Step 4: Verify green**

Run the new candidate tests.

### Task 4: qlib Research Card

**Files:**
- Create: `src/workflow/hardware_cycle_cogalpha_mining.py`
- Modify: `src/workflow/__init__.py`
- Modify: `tests/test_hardware_cycle_cogalpha_mining.py`
- Create via workflow: `research/hardware_cycle_cogalpha_mining_v1/*`

- [ ] **Step 1: Write failing workflow export/summary test**

Assert the workflow export exists and its daily semantics payload records carried cloud state, update flags, staleness, qlib provider URI, and recorder-first policy.

- [ ] **Step 2: Verify red**

Run:

```bash
PYTHONPATH=. qlib_venv/bin/python -m pytest tests/test_hardware_cycle_cogalpha_mining.py -q
```

Expected: FAIL because the workflow module/export is missing.

- [ ] **Step 3: Implement qlib card workflow**

Build the card from qlib SOXL data plus qlib hardware features, run CognitiveAlpha factor mining, materialize selected factors, feed them into the existing open-to-open position model, log sidecars and recorder artifacts, and write notebook/conclusion/summary.

- [ ] **Step 4: Execute notebook and focused tests**

Run:

```bash
PYTHONPATH=. qlib_venv/bin/python -m nbconvert --execute research/hardware_cycle_cogalpha_mining_v1/hardware_cycle_cogalpha_mining_v1.ipynb --to notebook --inplace
PYTHONPATH=. qlib_venv/bin/python -m pytest tests/test_hardware_cycle_data.py tests/test_hardware_cycle_factors.py tests/test_hardware_cycle_cogalpha_mining.py -q
PYTHONPATH=. qlib_venv/bin/python -m pytest tests/test_qlib_native_workflow.py tests/test_qlib_model_selection_validation.py tests/test_qlib_native_notebook_cards.py -q
```

### Task 5: Commit And Push

**Files:**
- All modified source, tests, data manifests, and slim card artifacts.

- [ ] **Step 1: Inspect status and diff**

Run:

```bash
git status --short
git diff --stat
git diff --cached --stat
```

- [ ] **Step 2: Commit**

Run:

```bash
git add <verified files>
git commit -m "feat: mine hardware cycle cognitive alpha factors"
```

- [ ] **Step 3: Push**

Run:

```bash
git push
```
