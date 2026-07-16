# SOXX Hidden Dispersion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and execute a qlib-native diagnostic card that tests whether SOXX constituent dispersion with a muted VIX response improves risk or directional evidence beyond the existing weighted-residual baseline.

**Architecture:** Add a focused analysis module that turns qlib-loaded constituent OHLCV plus SOXX/SOXL/VIX into lagged hidden-dispersion features and train-oriented diagnostics. Add a workflow module that reads both qlib providers, records the daily panel through `R.start`, writes a slim research card, and generates an executable notebook that reruns the same workflow.

**Tech Stack:** Python 3.11, pandas, NumPy, scikit-learn metrics, Microsoft qlib Data API and Workflow Recorder, pytest, nbconvert.

---

### Task 1: Hidden-dispersion feature panel

**Files:**
- Create: `src/analysis/soxx_hidden_dispersion.py`
- Modify: `src/analysis/__init__.py`
- Create: `tests/test_soxx_hidden_dispersion.py`

- [ ] **Step 1: Write the failing feature-panel tests**

Add deterministic synthetic constituent and index frames and assert that:

```python
panel = build_hidden_dispersion_panel(
    constituent_frame,
    index_frame,
    train=("2024-01-02", "2024-03-29"),
    lag_days=1,
    horizon=5,
)
assert set(HIDDEN_DISPERSION_FEATURES).issubset(panel.daily.columns)
assert panel.daily["residual_dispersion_5"].notna().any()
assert panel.daily["soxx_5d"].equals(
    panel.daily["soxx_open"].shift(-6) / panel.daily["soxx_open"].shift(-1) - 1.0
)
```

Add a perturbation test that changes only the final-day constituent OHLCV and
asserts every feature dated before that day remains identical. Add a constant-
return test asserting finite cancellation, entropy, and correlation-breakdown
values rather than division errors.

- [ ] **Step 2: Run tests and verify the RED state**

Run:

```bash
PYTHONPATH=. /Users/rziiiii/dev/workspace/qlib_venv/bin/python -m pytest tests/test_soxx_hidden_dispersion.py -q
```

Expected: collection fails with `ModuleNotFoundError: No module named 'src.analysis.soxx_hidden_dispersion'`.

- [ ] **Step 3: Implement the minimal feature panel**

Create:

```python
HIDDEN_DISPERSION_FEATURES = (
    "residual_dispersion_5",
    "range_energy_5",
    "cancellation_ratio_5",
    "correlation_breakdown_20",
    "breadth_entropy_5",
    "leadership_turnover_5",
    "abnormal_volume_breadth_5",
    "dispersion_vix_gap",
    "weighted_residual_x_dispersion_vix_gap",
)

@dataclass(frozen=True)
class HiddenDispersionPanel:
    constituents: pd.DataFrame
    daily: pd.DataFrame
    betas: dict[str, float]
    train: tuple[str, str]
    lag_days: int
    horizon: int
```

Implement `build_hidden_dispersion_panel(...)` with required-column validation,
train-only constituent beta estimates, equal-weight daily aggregation, rolling
windows, expanding z-scores, one-day lagging, and future open-to-open labels.
Use neutral finite values for zero-denominator cancellation and entropy cases;
do not backfill missing history.

Export the dataclass, feature constant, builder, and evaluator from
`src.analysis.__init__`.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the Task 1 command. Expected: all tests in the new file pass.

- [ ] **Step 5: Commit the feature panel**

```bash
git add src/analysis/soxx_hidden_dispersion.py src/analysis/__init__.py tests/test_soxx_hidden_dispersion.py
git commit -m "feat: add SOXX hidden dispersion features"
```

### Task 2: Train-oriented evaluation and candidate comparison

**Files:**
- Modify: `src/analysis/soxx_hidden_dispersion.py`
- Modify: `tests/test_soxx_hidden_dispersion.py`

- [ ] **Step 1: Write failing evaluation tests**

Assert `evaluate_hidden_dispersion(panel, segments=...)` returns exactly:

```python
{"feature_metrics", "nonoverlap", "candidate_comparison", "stability"}
```

Assert all metric rows contain `train_direction`, all non-overlap rows cover
offsets `0..4` and set `overlap_purged=True`, and candidate comparison contains
`weighted_residual_baseline`, `hidden_dispersion_composite`, and
`baseline_plus_interaction` for `soxx_5d`, `soxl_5d`, and `soxx_abs_5d`.

- [ ] **Step 2: Run the new evaluation test and verify RED**

Run:

```bash
PYTHONPATH=. /Users/rziiiii/dev/workspace/qlib_venv/bin/python -m pytest tests/test_soxx_hidden_dispersion.py -q
```

Expected: failure because `evaluate_hidden_dispersion` does not yet return the required tables.

- [ ] **Step 3: Implement train-only orientation and composites**

For each feature and target, set direction from train Spearman correlation only.
Create train-standardized candidate signals with this fixed feature map:

```python
CANDIDATE_FEATURES = {
    "weighted_residual_baseline": ("weighted_residual",),
    "hidden_dispersion_composite": (
        "residual_dispersion_5",
        "cancellation_ratio_5",
        "correlation_breakdown_20",
        "leadership_turnover_5",
        "abnormal_volume_breadth_5",
        "dispersion_vix_gap",
    ),
    "baseline_plus_interaction": (
        "weighted_residual",
        "dispersion_vix_gap",
        "weighted_residual_x_dispersion_vix_gap",
    ),
}
```

Average only available standardized components. Evaluate Spearman rank
correlation, directional AUC for signed targets, quartile spread, observations,
five non-overlapping offsets, and train/validation/test sign stability.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the Task 2 command. Expected: all tests pass.

- [ ] **Step 5: Commit evaluation logic**

```bash
git add src/analysis/soxx_hidden_dispersion.py tests/test_soxx_hidden_dispersion.py
git commit -m "feat: evaluate hidden dispersion candidates"
```

### Task 3: Qlib-native workflow and slim research card

**Files:**
- Create: `src/workflow/soxx_hidden_dispersion.py`
- Modify: `src/workflow/__init__.py`
- Create: `tests/test_soxx_hidden_dispersion_workflow.py`
- Modify: `tests/test_qlib_native_notebook_cards.py`

- [ ] **Step 1: Write failing workflow-contract tests**

Test pure helpers with a temporary card directory and a minimal summary. Assert
the generated notebook contains `import qlib`, `qlib.init`, `D.features`,
`R.start`, the workflow runner, and a final Chinese conclusion. Assert summary
policy and sidecars are:

```python
assert summary["artifact_policy"]["policy"] == "slim_card_recorder_first"
assert summary["artifact_policy"]["recorder_only_artifacts"] == [
    "daily_hidden_dispersion_panel.csv"
]
assert summary["promotion_decision"] == "not_promoted_missing_pit_holdings"
```

Add a notebook-card contract test for the final research directory and recorder
artifact path.

- [ ] **Step 2: Run workflow tests and verify RED**

Run:

```bash
PYTHONPATH=. /Users/rziiiii/dev/workspace/qlib_venv/bin/python -m pytest tests/test_soxx_hidden_dispersion_workflow.py -q
```

Expected: collection fails because the workflow module is absent.

- [ ] **Step 3: Implement qlib workflow and card generation**

Add `SoxxHiddenDispersionSpec` with existing train/validation/test windows and
provider URIs. Read fields through qlib:

```python
index_fields = ["$open", "$vix"]
constituent_fields = ["$open", "$high", "$low", "$adj_close", "$volume", "$weight_pct"]
```

Build/evaluate the panel, start `R.start(experiment_name=...)`, log parameters,
write the daily panel to a temporary directory, log it as a recorder artifact,
and write compact metric CSVs, `summary.json`, `workflow.yaml`, `conclusion.md`,
and the executable notebook. Fail closed on missing manifests, providers,
fields, or recorder writes.

- [ ] **Step 4: Run workflow and existing attribution tests**

Run:

```bash
PYTHONPATH=. /Users/rziiiii/dev/workspace/qlib_venv/bin/python -m pytest tests/test_soxx_hidden_dispersion.py tests/test_soxx_hidden_dispersion_workflow.py tests/test_soxx_constituent_alpha_attribution.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit workflow logic**

```bash
git add src/workflow/soxx_hidden_dispersion.py src/workflow/__init__.py tests/test_soxx_hidden_dispersion_workflow.py tests/test_qlib_native_notebook_cards.py
git commit -m "feat: add qlib hidden dispersion research workflow"
```

### Task 4: Execute and validate the research card

**Files:**
- Create: `research/soxx_hidden_dispersion_v1/soxx_hidden_dispersion_v1.ipynb`
- Create: `research/soxx_hidden_dispersion_v1/workflow.yaml`
- Create: `research/soxx_hidden_dispersion_v1/summary.json`
- Create: `research/soxx_hidden_dispersion_v1/conclusion.md`
- Create: `research/soxx_hidden_dispersion_v1/feature_metrics.csv`
- Create: `research/soxx_hidden_dispersion_v1/nonoverlap_feature_metrics.csv`
- Create: `research/soxx_hidden_dispersion_v1/model_comparison.csv`
- Create: `research/soxx_hidden_dispersion_v1/risk_analysis.csv`
- Recorder-only: `research/soxx_hidden_dispersion_v1/qlib_recorder/**/artifacts/soxx_hidden_dispersion/daily_hidden_dispersion_panel.csv`

- [ ] **Step 1: Execute the workflow once**

Run:

```bash
PYTHONPATH=. /Users/rziiiii/dev/workspace/qlib_venv/bin/python -c "from src.workflow.soxx_hidden_dispersion import run_soxx_hidden_dispersion_card; run_soxx_hidden_dispersion_card()"
```

Expected: exit `0` and a complete slim card with recorder id.

- [ ] **Step 2: Execute the generated notebook**

Run:

```bash
PYTHONPATH=. /Users/rziiiii/dev/workspace/qlib_venv/bin/python -m nbconvert --execute research/soxx_hidden_dispersion_v1/soxx_hidden_dispersion_v1.ipynb --to notebook --inplace --ExecutePreprocessor.timeout=1200
```

Expected: exit `0`; notebook, summary, conclusion, and recorder id are synchronized.

- [ ] **Step 3: Inspect evidence and promotion decision**

Check `summary.json`, metric CSVs, conclusion, recorder artifact presence, and
confirm that the result does not promote despite any positive diagnostic metric.

- [ ] **Step 4: Run focused and repository verification**

Run:

```bash
PYTHONPATH=. /Users/rziiiii/dev/workspace/qlib_venv/bin/python -m pytest tests/test_soxx_hidden_dispersion.py tests/test_soxx_hidden_dispersion_workflow.py tests/test_soxx_constituent_alpha_attribution.py tests/test_qlib_native_workflow.py tests/test_qlib_model_selection_validation.py tests/test_qlib_native_notebook_cards.py -q
PYTHONPATH=. /Users/rziiiii/dev/workspace/qlib_venv/bin/python -m pytest -q
git diff --check
```

Expected: both pytest commands and `git diff --check` exit `0`.

- [ ] **Step 5: Commit and push final evidence**

Inspect `git status --short` and the staged diff, then run:

```bash
git add research/soxx_hidden_dispersion_v1 docs/superpowers/plans/2026-07-16-soxx-hidden-dispersion.md
git commit -m "research: validate SOXX hidden dispersion features"
git push -u origin feature/soxx-hidden-dispersion-v1
```
