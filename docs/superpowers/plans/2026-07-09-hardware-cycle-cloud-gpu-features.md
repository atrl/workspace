# Hardware Cycle Cloud GPU Features Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a qlib-backed hardware-cycle/cloud-GPU external feature set and evaluate it in a new SOXL probability-budget research card.

**Architecture:** Add a data builder under `src/data/`, a qlib factor reader under `src/factor/`, and a new research workflow under `src/workflow/`. The workflow reuses the existing probability-budget model contract but adds a hardware feature group instead of modifying the validated baseline card.

**Tech Stack:** Python, pandas, requests, BeautifulSoup, qlib `D.features`, repo `build_qlib_store_from_frame`, pytest, nbconvert.

---

### Task 1: Data Source Builder

**Files:**
- Create: `src/data/hardware_cycle.py`
- Modify: `src/data/__init__.py`
- Test: `tests/test_hardware_cycle_data.py`

- [ ] Write parser tests for AIMultiple embedded chart-line payloads.
- [ ] Implement Yahoo proxy fetch, AIMultiple chart extraction, daily index assembly, raw snapshot writing, and qlib provider build.
- [ ] Verify provider readback through `D.features`.

### Task 2: Factor Builder

**Files:**
- Create: `src/factor/hardware_cycle.py`
- Modify: `src/factor/__init__.py`
- Test: `tests/test_hardware_cycle_factors.py`

- [ ] Write lag/no-leakage tests from a synthetic qlib provider.
- [ ] Implement lagged return, z-score, monthly cloud-GPU change, and availability features.
- [ ] Verify all model-facing columns are shifted by at least one day.

### Task 3: Research Workflow

**Files:**
- Create: `src/workflow/hardware_cycle_probability_budget_open2open.py`
- Modify: `src/workflow/__init__.py`
- Test: `tests/test_hardware_cycle_probability_budget_open2open.py`

- [ ] Write tests for the hardware feature group and summary provenance.
- [ ] Build a new research card `research/hardware_cycle_probability_budget_open2open_v1`.
- [ ] Store large positions and walk-forward positions only in the qlib recorder.

### Task 4: Execute And Verify

**Files:**
- Create/update: `research/hardware_cycle_probability_budget_open2open_v1/*`

- [ ] Run focused pytest for hardware data, factors, and workflow.
- [ ] Execute the new notebook with `nbconvert --execute`.
- [ ] Run the qlib native card tests.
- [ ] Inspect `git status --short` and staged diff.
- [ ] Commit and push.
