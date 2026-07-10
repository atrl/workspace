# Hardware Cycle GPU Factor Selection Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent unselected hardware factors from entering the archived factor library and keep factor mining strictly inside the training window before validation-based position selection.

**Architecture:** Preserve the existing qlib provider, factor candidates, recorder, and position model. Tighten the reusable factor-library archive selector to respect selection state, then make the hardware-cycle card pass only its train segment into factor mining and record that segment in card evidence.

**Tech Stack:** Python 3.11, pandas, pytest, qlib Workflow/Recorder, Jupyter nbconvert

---

### Task 1: Correct Factor-Library Archive Selection

**Files:**
- Modify: `src/factor/cognitive_alpha_mining.py`
- Test: `tests/test_cognitive_alpha_factor_mining.py`

- [x] **Step 1: Write a failing regression test**

Add a synthetic evolution result whose metrics contain one `selected_fallback` candidate and one higher-scoring `not_selected` candidate. Assert that `_select_factor_library_archive` archives the selected fallback candidate and excludes the unselected candidate.

- [x] **Step 2: Run the focused test and confirm the expected failure**

Run:

```bash
PYTHONPATH=. qlib_venv/bin/python -m pytest tests/test_cognitive_alpha_factor_mining.py -q
```

Expected: the new assertion fails because the current archive selector prefers the higher-scoring `not_selected` row.

- [x] **Step 3: Implement the minimal selection fix**

Change `_select_factor_library_archive` so its eligible rows are limited to `redundancy_status in {selected, selected_fallback}`. Prefer qualified or elite `selected` rows; otherwise use the already selected fallback rows. Never promote `not_selected` or `drop_high_correlation` rows into the library.

- [x] **Step 4: Run the focused test and confirm it passes**

Run the same pytest command and require exit code 0.

### Task 2: Isolate Factor Mining To Train

**Files:**
- Modify: `src/workflow/hardware_cycle_cogalpha_mining.py`
- Test: `tests/test_hardware_cycle_cogalpha_mining.py`

- [x] **Step 1: Write a failing workflow test**

Patch `run_cognitive_alpha_factor_mining`, capture the maximum `datetime` passed to it, and assert it equals `spec.train[1]`, not `spec.validation[1]`. Assert the summary records the factor-mining segment as the train segment.

- [x] **Step 2: Run the focused test and confirm the expected failure**

Run:

```bash
PYTHONPATH=. qlib_venv/bin/python -m pytest tests/test_hardware_cycle_cogalpha_mining.py -q
```

Expected: the captured frame extends through validation or the summary lacks the training-segment evidence.

- [x] **Step 3: Implement train-only mining and evidence**

Slice the mining frame at `spec.train[1]`. Add `factor_mining_segment` to the summary and workflow output so the notebook/card can prove factor discovery did not inspect validation labels.

- [x] **Step 4: Run the focused test and confirm it passes**

Run the same pytest command and require exit code 0.

### Task 3: Refresh Qlib Card Evidence

**Files:**
- Modify generated small sidecars under `research/hardware_cycle_cogalpha_mining_v1/`
- Modify: `research/hardware_cycle_cogalpha_mining_v1/hardware_cycle_cogalpha_mining_v1.ipynb`

- [x] **Step 1: Execute the notebook in place**

```bash
PYTHONPATH=. qlib_venv/bin/python -m nbconvert --execute research/hardware_cycle_cogalpha_mining_v1/hardware_cycle_cogalpha_mining_v1.ipynb --to notebook --inplace
```

- [x] **Step 2: Inspect refreshed evidence**

Confirm `summary.json`, `factor_library.json`, model-selection summaries, `conclusion.md`, and recorder metadata agree on selected factors, train-only mining, metrics, and promotion decision. Keep positions and walk-forward positions recorder-only.

### Task 4: Verify, Commit, And Push

**Files:**
- Verify all modified source, tests, plan, notebook, and small sidecars

- [x] **Step 1: Run focused and repository verification**

```bash
PYTHONPATH=. qlib_venv/bin/python -m pytest tests/test_hardware_cycle_data.py tests/test_hardware_cycle_factors.py tests/test_hardware_cycle_cogalpha_mining.py tests/test_hardware_cycle_probability_budget_open2open.py tests/test_cognitive_alpha_factor_mining.py -q
PYTHONPATH=. qlib_venv/bin/python -m pytest tests/test_qlib_native_workflow.py tests/test_qlib_model_selection_validation.py tests/test_qlib_native_notebook_cards.py -q
PYTHONPATH=. qlib_venv/bin/python -m pytest -q
```

- [x] **Step 2: Inspect scope and artifact policy**

Run `git status --short`, inspect `git diff --check`, review the source/test diff, and verify no recorder-only root artifacts or unrelated generated files are staged.

- [x] **Step 3: Commit and push**

```bash
git add <exact verified files>
git commit -m "fix: isolate hardware factor selection"
git push origin main
```
