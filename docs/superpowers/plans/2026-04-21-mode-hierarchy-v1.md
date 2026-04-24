# Mode Hierarchy V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a rule-based high-level mode router that constrains the existing residual actor with persistent trading modes across training, evaluation, backtest, and runtime paths.

**Architecture:** Introduce a new `policy/modes.py` module that owns mode classification, dwell logic, position bands, and execution-threshold multipliers. Keep the actor unchanged for phase 1: the high-level router is rule-based, while the existing actor continues to output residual actions that are clipped inside the current mode’s band. Wire the mode state through simulator, evaluation, backtest, and runtime so behavior is consistent across the harness. Per user preference, validation should rely on model backtests and harness outputs rather than unit tests.

**Tech Stack:** Python 3.11, dataclasses, existing qlib / tianshou policy stack, repo test scripts under `tests/`

---

### Task 1: Add Mode Configuration Surface

**Files:**
- Modify: `src/soxl_quant/config.py`

- [ ] **Step 1: Add the mode router dataclass**

Add a frozen dataclass to `src/soxl_quant/config.py`:

```python
@dataclass(frozen=True)
class ModeRouterConfig:
    enabled: bool = False
    trend_hold_min_dwell_days: int = 12
    defensive_hold_min_dwell_days: int = 6
    crash_rebuild_min_dwell_days: int = 5
    flat_wait_min_dwell_days: int = 4
```

Then add it to `StrategyConfig`:

```python
mode_router: ModeRouterConfig = field(default_factory=ModeRouterConfig)
```

- [ ] **Step 2: Verify profile loading still works**

Run:

```bash
PYTHONPATH=. qlib_venv/bin/python -c "from src.soxl_quant.config import load_profile; p = load_profile('balanced_rule_residual_v19'); print(p.mode_router.enabled)"
```

Expected: prints `False`

- [ ] **Step 3: Commit**

```bash
git add src/soxl_quant/config.py
git commit -m "feat: add mode router config scaffold"
```

### Task 2: Add Rule-Based Mode Router Module

**Files:**
- Create: `src/soxl_quant/policy/modes.py`

- [ ] **Step 1: Write minimal implementation**

Create `src/soxl_quant/policy/modes.py` with:

```python
@dataclass(frozen=True)
class ModeState:
    name: str = "flat_wait"
    days_in_mode: int = 0


def classify_rule_mode(obs: np.ndarray) -> str:
    ...


def next_mode(obs: np.ndarray, state: ModeState, cfg: ModeRouterConfig) -> ModeState:
    ...
```

Rules:
- `flat_wait` when risk is high and anchor/trend are weak
- `trend_hold` when anchor/trend are strong and risk is low
- `crash_rebuild` when drawdown is deep but risk is improving
- otherwise `defensive_hold`

Enforce dwell with per-mode minimum days from config.

- [ ] **Step 2: Verify the router classifies a few synthetic contexts**

Run:

```bash
PYTHONPATH=. qlib_venv/bin/python - <<'PY'
import numpy as np
from src.soxl_quant.data import FEATURE_COLUMNS, STATIC_INDEX
from src.soxl_quant.policy.modes import classify_rule_mode
obs = np.zeros(len(FEATURE_COLUMNS), dtype=np.float32)
obs[STATIC_INDEX["regime_risk"]] = 0.80
obs[STATIC_INDEX["realized_vol_10"]] = 0.75
obs[STATIC_INDEX["anchor_gap_180"]] = -0.05
print(classify_rule_mode(obs))
PY
```

Expected: prints `flat_wait`

- [ ] **Step 3: Commit**

```bash
git add src/soxl_quant/policy/modes.py
git commit -m "feat: add rule-based mode router"
```

### Task 3: Add Mode Bands And Execution Multipliers

**Files:**
- Modify: `src/soxl_quant/policy/positioning.py`
- Modify: `src/soxl_quant/policy/execution.py`

- [ ] **Step 1: Write minimal implementation**

In `src/soxl_quant/policy/modes.py`, add:

```python
def mode_band(mode: str) -> tuple[float, float]:
    ...


def mode_threshold_multiplier(mode: str, direction: str) -> float:
    ...
```

In `src/soxl_quant/policy/positioning.py`, thread `mode_band` through `hybrid_overlay_position`, `action_to_position`, and `target_position_for_profile`.

In `src/soxl_quant/policy/execution.py`, thread `mode` into `rebalance_threshold` and multiply the threshold using `mode_threshold_multiplier`.

- [ ] **Step 2: Verify the new clipping path with a synthetic command**

Run:

```bash
PYTHONPATH=. qlib_venv/bin/python - <<'PY'
import numpy as np
from src.soxl_quant.data import FEATURE_COLUMNS
from src.soxl_quant.policy.positioning import hybrid_overlay_position
obs = np.zeros(len(FEATURE_COLUMNS), dtype=np.float32)
pos = hybrid_overlay_position(2, obs, delta_levels=np.asarray([-0.5, 0.0, 0.5], dtype=np.float32), mode_band=(0.15, 0.25))
print(round(pos, 3))
PY
```

Expected: prints a value inside `0.15..0.25`

- [ ] **Step 3: Commit**

```bash
git add src/soxl_quant/policy/modes.py src/soxl_quant/policy/positioning.py src/soxl_quant/policy/execution.py
git commit -m "feat: add mode-aware bands and thresholds"
```

### Task 4: Wire Mode State Through Simulator, Evaluation, Backtest, Runtime

**Files:**
- Modify: `src/soxl_quant/policy/environment.py`
- Modify: `src/soxl_quant/policy/evaluation.py`
- Modify: `src/soxl_quant/backtest.py`
- Modify: `src/soxl_quant/runtime.py`

- [ ] **Step 1: Write minimal implementation**

Add a small integration helper in `modes.py`:

```python
def route_mode(obs: np.ndarray, state: ModeState | None, cfg: ModeRouterConfig) -> ModeState:
    ...
```

Then in simulator / evaluation / backtest / runtime:
- maintain `mode_state`
- update it once per step or decision day
- pass `mode_state.name` into `target_position_for_profile()` and `rebalance_threshold()`

- [ ] **Step 2: Run a focused model backtest comparison**

Run:

```bash
PYTHONPATH=. qlib_venv/bin/python soxl_qlib_strategy_v2.py --profiles balanced_rule_residual_mode_v1 --skip-data-refresh --seed 12345
```

Expected:
- training completes
- summary artifact exists
- slice output includes `test_2022` and `test_2023_2024`

- [ ] **Step 3: Commit**

```bash
git add src/soxl_quant/policy/environment.py src/soxl_quant/policy/evaluation.py src/soxl_quant/backtest.py src/soxl_quant/runtime.py src/soxl_quant/policy/modes.py
git commit -m "feat: wire mode state through harness paths"
```

### Task 5: Add A First Candidate Profile And Validation Evidence

**Files:**
- Create: `experiments/candidate/balanced_rule_residual_mode_v1/profile.yaml`
- Create: `experiments/candidate/balanced_rule_residual_mode_v1/notes.md`
- Create: `experiments/candidate/balanced_rule_residual_mode_v1/expectations.md`
- Modify: `configs/soxl/profile_registry.yaml`
- Modify: `codex_exec_plans/completed/2026-04/2026-04-10-soxl-agent-loop.md`

- [ ] **Step 1: Write minimal implementation**

Create the candidate package as a clone of `balanced_rule_residual_v19` with:
- `mode_router.enabled: true`
- no reward changes
- no feature-pack changes

Add notes with:
- closest success = `dwell_hold_v2`
- closest failure = `balanced_rule_residual_v24`
- new mechanism = persistent mode abstraction

- [ ] **Step 2: Verify profile loading and registry coverage**

Run: `PYTHONPATH=. qlib_venv/bin/python -c "import tests.test_profile_registry as t; t.test_registry_covers_all_profiles(); t.test_registry_has_single_canonical_per_family(); print('test_profile_registry: OK')"`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add experiments/candidate/balanced_rule_residual_mode_v1 configs/soxl/profile_registry.yaml codex_exec_plans/completed/2026-04/2026-04-10-soxl-agent-loop.md
git commit -m "feat: add first mode hierarchy candidate profile"
```

### Task 6: Verify Harness Compatibility

**Files:**
- Modify: `memory/2026-04-21-*.md` as needed

- [ ] **Step 1: Run focused backtest validation**

Run:

```bash
PYTHONPATH=. qlib_venv/bin/python soxl_qlib_strategy_v2.py --profiles balanced_rule_residual_mode_v1 --skip-data-refresh --seed 12345
```

Expected:
- training completes
- summary artifact exists
- no crash in simulator / evaluation / backtest path

- [ ] **Step 2: Run harness checks**

Run:

```bash
PYTHONPATH=. qlib_venv/bin/python scripts/check_harness.py
```

Expected: `failures=0`

- [ ] **Step 3: Record evidence**

Update:
- `codex_exec_plans/completed/2026-04/2026-04-10-soxl-agent-loop.md`
- a new memory note summarizing the scaffold result

- [ ] **Step 4: Commit**

```bash
git add codex_exec_plans/completed/2026-04/2026-04-10-soxl-agent-loop.md memory/
git commit -m "chore: record mode hierarchy scaffold validation"
```
