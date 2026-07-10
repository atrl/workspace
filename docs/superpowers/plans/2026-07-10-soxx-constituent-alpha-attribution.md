# SOXX Constituent Alpha Attribution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a qlib research card that separates single-name signal quality, SOXX aggregation, SOXL leverage mapping, and execution censoring.

**Architecture:** Add reusable attribution analysis under `src/analysis`, orchestrate qlib providers and recorder evidence in a dedicated workflow, and expose the result through an executable research notebook with slim-card sidecars. Keep production position rules unchanged and fail promotion closed on missing PIT holdings.

**Tech Stack:** Python 3.11, pandas, numpy, scikit-learn, qlib Data API/Workflow/Recorder, pytest, Jupyter nbconvert

---

### Task 1: Attribution Analysis Contracts

**Files:**
- Create: `src/analysis/soxx_constituent_attribution.py`
- Modify: `src/analysis/__init__.py`
- Create: `tests/test_soxx_constituent_alpha_attribution.py`

- [x] Write failing tests for lagged constituent signals, train-only orientation, leveraged path mapping, and execution censoring.
- [x] Run the focused tests and confirm they fail because the analysis API is missing.
- [x] Implement the minimal reusable analysis functions.
- [x] Run focused tests and require exit code 0.

### Task 2: Qlib Research Workflow And Card

**Files:**
- Create: `src/workflow/soxx_constituent_alpha_attribution.py`
- Modify: `src/workflow/__init__.py`
- Modify: `tests/test_qlib_native_notebook_cards.py`
- Create: `research/soxx_constituent_alpha_attribution_v1/soxx_constituent_alpha_attribution_v1.ipynb`

- [x] Write failing workflow/card contract tests.
- [x] Implement qlib provider reads, recorder loading/logging, compact sidecars, summary, workflow, conclusion, and notebook generation.
- [x] Execute the notebook in place.
- [x] Verify the card remains `not_promoted_missing_pit_holdings` and contains no recorder-only root artifacts.

### Task 3: Verification And Delivery

**Files:**
- Refresh the new research card and recorder artifacts.

- [x] Run focused attribution and qlib card tests.
- [x] Run the repository qlib contract tests and full pytest suite.
- [x] Inspect artifact policy, staged diff, and `git status --short`.
- [x] Commit with `feat: add SOXX constituent alpha attribution` and push `main`.
