# Multi-3x Price-Volume Transfer Implementation Plan

**Goal:** Build and execute a qlib-native, factor-only experiment that tests
whether price-volume pressure transfers across long 3x ETFs and adds zero-shot
SOXL breakdown information beyond price-only high-chop factors.

**Architecture:** Add a reusable point-in-time factor builder under `src/factor`
and a diagnostics/card runner under `src/workflow`. The workflow reads the paired
providers through qlib, builds and audits a derived qlib store, uses a qlib
Dataset, logs evidence with the qlib recorder, and writes a slim research card.

**Tech stack:** Python, pandas, numpy, scipy/scikit-learn metrics, qlib
Data/Dataset/Workflow, pytest, nbformat/nbconvert.

---

### Task 1: Lock factor contracts with failing tests

**Files:**
- Create: `tests/test_multi_3x_price_volume_factors.py`

Test required columns, feature bounds, one-session delay, future-mutation
invariance, expected direction under synthetic failed-breakout volume, and safe
handling of zero ranges/constant volume. Run the test and confirm RED because the
module does not exist.

### Task 2: Implement price-volume factors

**Files:**
- Create: `src/factor/multi_3x_price_volume.py`
- Modify: `src/factor/__init__.py`

Implement the eight factor families, trailing percentile normalization, point-in-
time lagging, price-only baseline scores, and binary breakdown label preparation.
Run factor tests until GREEN, then run the existing high-chop factor tests.

### Task 3: Lock transfer diagnostics with failing tests

**Files:**
- Create: `tests/test_multi_3x_price_volume_transfer_workflow.py`

Test group-aware diagnostics, quintile lift, factor retention gates, redundancy
filtering, zero-shot SOXL isolation, fail-closed insufficient samples, and qlib
Dataset construction. Run the test and confirm RED.

### Task 4: Implement qlib-native workflow

**Files:**
- Create: `src/workflow/multi_3x_price_volume_transfer.py`
- Modify: `src/workflow/__init__.py`

Implement source loading through `D.features`, derived-store creation/readback,
development group diagnostics, validation-only factor retention, frozen composite
outer evaluation, qlib recorder logging, slim-card serialization, and explicit
decision labels. Run focused tests until GREEN.

### Task 5: Create and execute research notebook

**Files:**
- Create: `research/multi_3x_price_volume_transfer_v1/`
- Modify: `tests/test_qlib_native_notebook_cards.py`

Create an executable diagnostic notebook with hypothesis, qlib source/Dataset/
Recorder path, bounded result tables, and final Chinese conclusion. Execute it
top-to-bottom and keep row-level artifacts recorder-only.

### Task 6: Validate evidence and repository health

Run independent reconciliations of card metrics against recorder artifacts,
focused factor/workflow/card tests, the mandated qlib-native trio, and full pytest.
Document pre-existing failures separately from regressions. Inspect `git status`
and the staged diff.

### Task 7: Commit and push

Commit the executed card, code, tests, and design/plan. Push the feature branch.
If push fails because of network or permissions, report the exact blocker.
