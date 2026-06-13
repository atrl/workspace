# SOXL Dashboard V2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local TradingView-like SOXL dashboard v2 that renders SOXL K-line candles, model BUY/SELL and position evidence, model-difference panels, and feature inspection from replayable model artifacts.

**Architecture:** Add a tested Python data-contract layer that writes `dashboard_data/v2/` JSON from `dashboard_data/model_replay_freshness.json`, existing replay CSVs, and existing OHLC/feature loaders. Add a new local Vite + TypeScript frontend under `web/dashboard-v2/` that reads those JSON files, renders Lightweight Charts candlesticks and event markers, and exposes dense comparison/inspection panels without touching the legacy `dashboard.html` path.

**Tech Stack:** Python 3.11 with pandas/pytest for data building and contract checks; Vite + TypeScript + Vitest + TradingView Lightweight Charts for the local frontend; Browser/Playwright screenshot checks for final visual verification.

---

## File Structure

- Create `src/soxl_quant/app/dashboard_v2.py`
  - Pure data-contract helpers for manifest loading, market row normalization, model row normalization, event derivation, comparison math, and JSON writing.
- Create `scripts/build_dashboard_v2_data.py`
  - Thin CLI wrapper around `src.soxl_quant.app.dashboard_v2.build_dashboard_v2_data()`.
- Create `scripts/check_dashboard_v2_data.py`
  - Static checker that validates `dashboard_data/v2/manifest.json`, market rows, model rows, and default comparison rows.
- Create `tests/test_dashboard_v2_data.py`
  - TDD coverage for the data builder behaviors listed in the design.
- Create `tests/test_dashboard_v2_checker.py`
  - TDD coverage for checker failure messages and success path.
- Create `web/dashboard-v2/package.json`
  - Local frontend package isolated from the Python harness.
- Create `web/dashboard-v2/index.html`
  - Vite entrypoint.
- Create `web/dashboard-v2/tsconfig.json`
  - Strict TypeScript config.
- Create `web/dashboard-v2/vite.config.ts`
  - Vite dev/build config and Vitest jsdom setup.
- Create `web/dashboard-v2/src/types.ts`
  - Shared dashboard JSON types.
- Create `web/dashboard-v2/src/domain.ts`
  - Pure frontend data-selection, overlap, marker, delta, and inspector helpers.
- Create `web/dashboard-v2/src/data.ts`
  - Fetches `dashboard_data/v2/` JSON from the local dev server or a configured data base URL.
- Create `web/dashboard-v2/src/chart.ts`
  - Lightweight Charts setup and rendering adapters.
- Create `web/dashboard-v2/src/main.ts`
  - DOM orchestration for toolbar, chart workspace, comparison panels, inspector, and source panel.
- Create `web/dashboard-v2/src/styles.css`
  - Dense research-terminal layout and responsive constraints.
- Create `web/dashboard-v2/src/domain.test.ts`
  - Vitest coverage for model selection limits, overlap clipping, baseline deltas, markers, and inspector rows.
- Create `web/dashboard-v2/src/testFixtures.ts`
  - Small deterministic frontend fixture.

## Task 1: Data Builder Contract Tests

**Files:**
- Create: `tests/test_dashboard_v2_data.py`

- [ ] **Step 1: Write failing tests for data-contract behavior**

```python
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.soxl_quant.app.dashboard_v2 import (
    build_dashboard_v2_data,
    derive_model_events,
    normalize_model_rows,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def test_build_dashboard_v2_data_writes_manifest_market_model_and_comparison(tmp_path: Path):
    freshness = tmp_path / "dashboard_data" / "model_replay_freshness.json"
    freshness.parent.mkdir()
    model_csv = tmp_path / "runtime" / "model" / "card" / "smoke" / "smoke_soxl_daily.csv"
    model_csv.parent.mkdir(parents=True)
    _write_csv(
        model_csv,
        [
            {"datetime": "2026-01-02", "strategy_nav": 100.0, "buyhold_nav": 100.0, "position": 0.0, "action": "HOLD", "price": 10.0},
            {"datetime": "2026-01-05", "strategy_nav": 105.0, "buyhold_nav": 110.0, "position": 0.7, "action": "BUY", "price": 11.0},
            {"datetime": "2026-01-06", "strategy_nav": 103.0, "buyhold_nav": 108.0, "position": 0.2, "action": "SELL", "price": 10.8},
        ],
    )
    freshness.write_text(
        json.dumps(
            {
                "generated_at": "2026-06-13T00:00:00+00:00",
                "latest_data_date": "2026-01-06",
                "safe_backtest_end": "2026-01-06",
                "profiles": [
                    {
                        "profile": "smoke",
                        "status": "canonical",
                        "actor": "runtime/model/card/smoke/soxl_smoke_actor.pt",
                        "profile_yaml": "runtime/harness/card/smoke/profile.yaml",
                        "output": "runtime/model/card/smoke/smoke_soxl_daily.csv",
                        "result": "current",
                    }
                ],
            }
        )
    )
    market_rows = {
        "2026-01-02": {"open": 9.5, "high": 10.2, "low": 9.4, "close": 10.0, "volume": 1000, "rsi6": 50.0, "ma180": None, "hv_20": None, "vol_coef": 1.0, "panic_score": 0.0, "market_state": False},
        "2026-01-05": {"open": 10.2, "high": 11.5, "low": 10.1, "close": 11.0, "volume": 1200, "rsi6": 60.0, "ma180": None, "hv_20": None, "vol_coef": 1.1, "panic_score": -5.0, "market_state": False},
        "2026-01-06": {"open": 11.0, "high": 11.1, "low": 10.5, "close": 10.8, "volume": 1300, "rsi6": 58.0, "ma180": None, "hv_20": None, "vol_coef": 1.1, "panic_score": -3.0, "market_state": False},
    }
    feature_rows = {"2026-01-05": {"vix": 18.0, "rsi": 60.0, "spread": 0.4}}

    output_dir = tmp_path / "dashboard_data" / "v2"
    build_dashboard_v2_data(
        root=tmp_path,
        output_dir=output_dir,
        freshness_path=freshness,
        market_rows=market_rows,
        feature_rows=feature_rows,
        selected_profiles=["smoke"],
        baseline_profile="smoke",
    )

    manifest = json.loads((output_dir / "manifest.json").read_text())
    market = json.loads((output_dir / "market" / "SOXL.json").read_text())
    model = json.loads((output_dir / "models" / "smoke.json").read_text())
    comparison = json.loads((output_dir / "comparison" / "default.json").read_text())

    assert manifest["safe_backtest_end"] == "2026-01-06"
    assert manifest["default_baseline_profile"] == "smoke"
    assert market["rows"][1]["vix"] == 18.0
    assert model["rows"][1]["action"] == "BUY"
    assert model["events"][0]["kind"] == "BUY"
    assert comparison["common_start"] == "2026-01-02"
    assert comparison["rows"][1]["action_mismatch_count"] == 0


def test_normalize_model_rows_rejects_invalid_numeric_fields(tmp_path: Path):
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("datetime,strategy_nav,buyhold_nav,position,action,price\n2026-01-02,nope,100,0,HOLD,10\n")

    with pytest.raises(ValueError, match="strategy_nav"):
        normalize_model_rows(csv_path, safe_backtest_end="2026-01-02")


def test_derive_model_events_uses_actions_and_position_changes():
    rows = [
        {"date": "2026-01-02", "action": "HOLD", "position": 0.0, "price": 10.0, "strategy_nav": 100.0},
        {"date": "2026-01-03", "action": "HOLD", "position": 0.4, "price": 10.5, "strategy_nav": 101.0},
        {"date": "2026-01-04", "action": "SELL", "position": 0.1, "price": 10.2, "strategy_nav": 100.5},
    ]

    events = derive_model_events(rows, material_position_delta=0.25)

    assert [event["kind"] for event in events] == ["POSITION_INCREASE", "SELL", "POSITION_DECREASE"]
    assert events[0]["position_delta"] == 0.4
```

- [ ] **Step 2: Run tests to verify they fail because the module does not exist**

Run:

```bash
uv run --python qlib_venv/bin/python --with pytest pytest tests/test_dashboard_v2_data.py -q
```

Expected: FAIL during import with `ModuleNotFoundError: No module named 'src.soxl_quant.app.dashboard_v2'`.

## Task 2: Data Builder Implementation

**Files:**
- Create: `src/soxl_quant/app/dashboard_v2.py`
- Create: `scripts/build_dashboard_v2_data.py`
- Modify only if needed: `tests/test_dashboard_v2_data.py`

- [ ] **Step 1: Implement the pure data-contract module**

Create `src/soxl_quant/app/dashboard_v2.py` with these public functions:

```python
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.build_backtest_dashboard import FEATURE_COLS, load_features, load_ohlc_and_indicators


REQUIRED_MODEL_FIELDS = ("datetime", "strategy_nav", "buyhold_nav", "position", "action", "price")
ACTION_VALUES = {"BUY", "SELL", "HOLD"}


@dataclass(frozen=True)
class DashboardV2Result:
    output_dir: Path
    manifest_path: Path
    market_path: Path
    model_paths: dict[str, Path]
    comparison_path: Path


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _relative_to(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _as_float(value: object, field: str, row_index: int) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid numeric field {field!r} at row {row_index}") from exc
    if pd.isna(number):
        raise ValueError(f"invalid numeric field {field!r} at row {row_index}")
    return number
```

Then implement:

```python
def load_replay_freshness(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"missing replay freshness file: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload.get("profiles"), list):
        raise ValueError("model_replay_freshness.json must contain profiles list")
    return payload


def normalize_market_rows(market_rows: dict[str, dict[str, Any]], feature_rows: dict[str, dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    if not market_rows:
        raise ValueError("missing market OHLCV data")
    feature_rows = feature_rows or {}
    rows: list[dict[str, Any]] = []
    for date in sorted(market_rows):
        source = dict(market_rows[date])
        row = {"date": date}
        for field in ("open", "high", "low", "close", "volume", "rsi6", "ma180", "hv_20", "vol_coef", "panic_score", "market_state"):
            row[field] = source.get(field)
        for field in FEATURE_COLS:
            if field in feature_rows.get(date, {}):
                row[field] = feature_rows[date][field]
        rows.append(row)
    if len({row["date"] for row in rows}) != len(rows):
        raise ValueError("duplicate market dates")
    return rows


def normalize_model_rows(path: Path, *, safe_backtest_end: str | None) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"missing replay CSV: {path}")
    frame = pd.read_csv(path)
    missing = [field for field in REQUIRED_MODEL_FIELDS if field not in frame.columns]
    if missing:
        raise ValueError(f"missing replay CSV columns: {', '.join(missing)}")
    frame["datetime"] = pd.to_datetime(frame["datetime"], errors="raise")
    frame = frame.sort_values("datetime")
    if frame["datetime"].duplicated().any():
        raise ValueError(f"duplicate model dates in {path}")
    if safe_backtest_end is not None:
        frame = frame[frame["datetime"] <= pd.Timestamp(safe_backtest_end)]
    rows: list[dict[str, Any]] = []
    first_strategy_nav: float | None = None
    for idx, source in frame.reset_index(drop=True).iterrows():
        strategy_nav = _as_float(source["strategy_nav"], "strategy_nav", idx)
        buyhold_nav = _as_float(source["buyhold_nav"], "buyhold_nav", idx)
        position = _as_float(source["position"], "position", idx)
        price = _as_float(source["price"], "price", idx)
        action = str(source.get("action") or "HOLD").upper()
        if action not in ACTION_VALUES:
            action = "HOLD"
        if first_strategy_nav is None:
            first_strategy_nav = strategy_nav
        excess = strategy_nav - buyhold_nav
        excess_pct = (strategy_nav / first_strategy_nav - 1.0) if first_strategy_nav else 0.0
        rows.append(
            {
                "date": source["datetime"].strftime("%Y-%m-%d"),
                "strategy_nav": strategy_nav,
                "buyhold_nav": buyhold_nav,
                "position": position,
                "action": action,
                "price": price,
                "excess": excess,
                "excess_pct": excess_pct,
            }
        )
    if not rows:
        raise ValueError(f"no replay rows remain after safe_backtest_end filter: {path}")
    return rows
```

Then implement event, metrics, comparison, and orchestration functions:

```python
def derive_model_events(rows: list[dict[str, Any]], *, material_position_delta: float = 0.10) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    previous_position = 0.0
    for row in rows:
        position = float(row["position"])
        delta = position - previous_position
        action = str(row.get("action") or "HOLD").upper()
        if action in {"BUY", "SELL"}:
            events.append({"date": row["date"], "kind": action, "price": row.get("price"), "position": position, "position_delta": delta})
        if abs(delta) >= material_position_delta:
            events.append(
                {
                    "date": row["date"],
                    "kind": "POSITION_INCREASE" if delta > 0 else "POSITION_DECREASE",
                    "price": row.get("price"),
                    "position": position,
                    "position_delta": delta,
                }
            )
        previous_position = position
    return events


def compute_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    start_nav = float(rows[0]["strategy_nav"])
    end_nav = float(rows[-1]["strategy_nav"])
    peak = start_nav
    max_drawdown = 0.0
    position_sum = 0.0
    for row in rows:
        nav = float(row["strategy_nav"])
        peak = max(peak, nav)
        if peak:
            max_drawdown = min(max_drawdown, nav / peak - 1.0)
        position_sum += float(row["position"])
    return {
        "return": end_nav / start_nav - 1.0 if start_nav else 0.0,
        "max_drawdown": max_drawdown,
        "avg_position": position_sum / len(rows),
        "row_count": float(len(rows)),
    }


def build_comparison_payload(model_payloads: dict[str, dict[str, Any]], *, baseline_profile: str, compared_profiles: list[str]) -> dict[str, Any]:
    if baseline_profile not in model_payloads:
        raise ValueError(f"baseline profile not available: {baseline_profile}")
    profiles = [profile for profile in compared_profiles if profile in model_payloads]
    date_sets = [{row["date"] for row in model_payloads[profile]["rows"]} for profile in profiles]
    common_dates = sorted(set.intersection(*date_sets)) if date_sets else []
    if not common_dates:
        raise ValueError("no overlapping date range between selected model rows")
    by_profile = {
        profile: {row["date"]: row for row in model_payloads[profile]["rows"]}
        for profile in profiles
    }
    baseline_rows = by_profile[baseline_profile]
    rows: list[dict[str, Any]] = []
    for date in common_dates:
        baseline_row = baseline_rows[date]
        action_mismatch_count = 0
        profile_values: dict[str, dict[str, Any]] = {}
        for profile in profiles:
            row = by_profile[profile][date]
            action_mismatch = row["action"] != baseline_row["action"]
            action_mismatch_count += int(action_mismatch)
            profile_values[profile] = {
                "strategy_nav": row["strategy_nav"],
                "position": row["position"],
                "action": row["action"],
                "nav_delta_vs_baseline": row["strategy_nav"] - baseline_row["strategy_nav"],
                "position_delta_vs_baseline": row["position"] - baseline_row["position"],
                "action_mismatch": action_mismatch,
            }
        rows.append({"date": date, "profiles": profile_values, "action_mismatch_count": action_mismatch_count})
    rankings = sorted(
        (
            {"profile": profile, **compute_metrics([by_profile[profile][date] for date in common_dates])}
            for profile in profiles
        ),
        key=lambda item: item["return"],
        reverse=True,
    )
    return {
        "baseline_profile": baseline_profile,
        "compared_profiles": profiles,
        "common_start": common_dates[0],
        "common_end": common_dates[-1],
        "rows": rows,
        "rankings": rankings,
    }


def build_dashboard_v2_data(
    *,
    root: Path,
    output_dir: Path,
    freshness_path: Path | None = None,
    market_rows: dict[str, dict[str, Any]] | None = None,
    feature_rows: dict[str, dict[str, Any]] | None = None,
    selected_profiles: list[str] | None = None,
    baseline_profile: str | None = None,
) -> DashboardV2Result:
    root = root.resolve()
    freshness_path = freshness_path or root / "dashboard_data" / "model_replay_freshness.json"
    freshness = load_replay_freshness(freshness_path)
    safe_backtest_end = freshness.get("safe_backtest_end")
    latest_market_date = freshness.get("latest_data_date")
    market_payload_rows = normalize_market_rows(market_rows or load_ohlc_and_indicators("SOXL"), feature_rows if feature_rows is not None else load_features())
    ready_profiles = [profile for profile in freshness["profiles"] if profile.get("result") in {"current", "refreshed"}]
    by_name = {str(profile["profile"]): profile for profile in ready_profiles}
    defaults = selected_profiles or [name for name in ("smoke", "balanced_rule_residual_v22") if name in by_name] or list(by_name)[:2]
    if len(defaults) == 1:
        defaults = defaults + [defaults[0]]
    defaults = defaults[:4]
    baseline = baseline_profile or ("balanced_rule_residual_v22" if "balanced_rule_residual_v22" in defaults else defaults[0])
    if baseline not in defaults:
        defaults = [baseline] + defaults
    model_payloads: dict[str, dict[str, Any]] = {}
    model_paths: dict[str, Path] = {}
    for profile_name in defaults:
        source = by_name[profile_name]
        replay_path = root / str(source["output"])
        rows = normalize_model_rows(replay_path, safe_backtest_end=str(safe_backtest_end) if safe_backtest_end else None)
        payload = {
            "profile": profile_name,
            "status": source.get("status"),
            "actor": source.get("actor"),
            "profile_yaml": source.get("profile_yaml"),
            "source_path": _relative_to(root, replay_path),
            "replay_start": rows[0]["date"],
            "replay_end": rows[-1]["date"],
            "row_count": len(rows),
            "rows": rows,
            "events": derive_model_events(rows),
            "metrics": {"full": compute_metrics(rows)},
        }
        model_payloads[profile_name] = payload
        model_path = output_dir / "models" / f"{profile_name}.json"
        _write_json(model_path, payload)
        model_paths[profile_name] = model_path
    comparison = build_comparison_payload(model_payloads, baseline_profile=baseline, compared_profiles=defaults)
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "latest_market_date": latest_market_date,
        "safe_backtest_end": safe_backtest_end,
        "available_profiles": sorted(by_name),
        "default_baseline_profile": baseline,
        "default_comparison_profiles": defaults,
        "source_paths": {
            "model_replay_freshness": _relative_to(root, freshness_path),
            "market": "data/processed/panels/soxl_soxx_vix_full.csv",
        },
    }
    market_payload = {"symbol": "SOXL", "rows": market_payload_rows}
    manifest_path = output_dir / "manifest.json"
    market_path = output_dir / "market" / "SOXL.json"
    comparison_path = output_dir / "comparison" / "default.json"
    _write_json(manifest_path, manifest)
    _write_json(market_path, market_payload)
    _write_json(comparison_path, comparison)
    return DashboardV2Result(output_dir=output_dir, manifest_path=manifest_path, market_path=market_path, model_paths=model_paths, comparison_path=comparison_path)
```

- [ ] **Step 2: Add the CLI wrapper**

Create `scripts/build_dashboard_v2_data.py`:

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.soxl_quant.app.dashboard_v2 import build_dashboard_v2_data


def main() -> int:
    parser = argparse.ArgumentParser(description="Build dashboard v2 JSON data contract.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "dashboard_data" / "v2")
    parser.add_argument("--freshness-path", type=Path, default=ROOT / "dashboard_data" / "model_replay_freshness.json")
    parser.add_argument("--profile", action="append", dest="profiles", help="Profile to include; may be repeated.")
    parser.add_argument("--baseline", help="Baseline profile for comparison.")
    args = parser.parse_args()

    result = build_dashboard_v2_data(
        root=ROOT,
        output_dir=args.output_dir,
        freshness_path=args.freshness_path,
        selected_profiles=args.profiles,
        baseline_profile=args.baseline,
    )
    print(f"dashboard_v2_data={result.output_dir}")
    print(f"manifest={result.manifest_path}")
    print(f"comparison={result.comparison_path}")
    print(f"models={len(result.model_paths)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: Run tests to verify Task 1 is green**

Run:

```bash
uv run --python qlib_venv/bin/python --with pytest pytest tests/test_dashboard_v2_data.py -q
```

Expected: `3 passed`.

- [ ] **Step 4: Run the data builder on current repo data**

Run:

```bash
PYTHONPATH=. qlib_venv/bin/python scripts/build_dashboard_v2_data.py --profile smoke --profile balanced_rule_residual_v22 --baseline balanced_rule_residual_v22
```

Expected: command exits `0` and prints `dashboard_v2_data=/Users/rziiiii/dev/workspace/dashboard_data/v2`, `comparison=.../dashboard_data/v2/comparison/default.json`, and `models=2`.

- [ ] **Step 5: Commit Task 2**

Run:

```bash
git add src/soxl_quant/app/dashboard_v2.py scripts/build_dashboard_v2_data.py tests/test_dashboard_v2_data.py dashboard_data/v2
git commit -m "Add dashboard v2 data contract"
```

## Task 3: Data Contract Checker

**Files:**
- Create: `tests/test_dashboard_v2_checker.py`
- Create: `scripts/check_dashboard_v2_data.py`

- [ ] **Step 1: Write failing checker tests**

```python
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.check_dashboard_v2_data import check_dashboard_v2_data


def test_check_dashboard_v2_data_passes_valid_contract(tmp_path: Path):
    root = tmp_path / "dashboard_data" / "v2"
    (root / "market").mkdir(parents=True)
    (root / "models").mkdir()
    (root / "comparison").mkdir()
    (root / "manifest.json").write_text(json.dumps({"safe_backtest_end": "2026-01-06", "default_baseline_profile": "smoke", "default_comparison_profiles": ["smoke"], "available_profiles": ["smoke"]}))
    (root / "market" / "SOXL.json").write_text(json.dumps({"symbol": "SOXL", "rows": [{"date": "2026-01-06", "open": 10, "high": 11, "low": 9, "close": 10.5, "volume": 1000}]}))
    (root / "models" / "smoke.json").write_text(json.dumps({"profile": "smoke", "rows": [{"date": "2026-01-06", "strategy_nav": 100, "buyhold_nav": 100, "position": 0.5, "action": "BUY", "price": 10.5}], "events": [{"date": "2026-01-06", "kind": "BUY"}]}))
    (root / "comparison" / "default.json").write_text(json.dumps({"baseline_profile": "smoke", "compared_profiles": ["smoke"], "common_start": "2026-01-06", "common_end": "2026-01-06", "rows": [{"date": "2026-01-06", "profiles": {"smoke": {"action": "BUY"}}}]}))

    result = check_dashboard_v2_data(root)

    assert result == {"profiles": 1, "market_rows": 1, "comparison_rows": 1}


def test_check_dashboard_v2_data_fails_missing_model_position(tmp_path: Path):
    root = tmp_path / "dashboard_data" / "v2"
    (root / "market").mkdir(parents=True)
    (root / "models").mkdir()
    (root / "comparison").mkdir()
    (root / "manifest.json").write_text(json.dumps({"default_comparison_profiles": ["smoke"], "available_profiles": ["smoke"]}))
    (root / "market" / "SOXL.json").write_text(json.dumps({"rows": [{"date": "2026-01-06", "open": 10, "high": 11, "low": 9, "close": 10.5, "volume": 1000}]}))
    (root / "models" / "smoke.json").write_text(json.dumps({"profile": "smoke", "rows": [{"date": "2026-01-06", "strategy_nav": 100, "buyhold_nav": 100, "action": "BUY", "price": 10.5}], "events": []}))
    (root / "comparison" / "default.json").write_text(json.dumps({"rows": []}))

    with pytest.raises(ValueError, match="position"):
        check_dashboard_v2_data(root)
```

- [ ] **Step 2: Run tests to verify they fail because the checker does not exist**

Run:

```bash
uv run --python qlib_venv/bin/python --with pytest pytest tests/test_dashboard_v2_checker.py -q
```

Expected: FAIL during import with `ModuleNotFoundError` or `ImportError`.

- [ ] **Step 3: Implement checker**

Create `scripts/check_dashboard_v2_data.py`:

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REQUIRED_MARKET_FIELDS = {"date", "open", "high", "low", "close", "volume"}
REQUIRED_MODEL_FIELDS = {"date", "strategy_nav", "buyhold_nav", "position", "action", "price"}


def _load(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"missing dashboard v2 file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _require_fields(row: dict[str, Any], required: set[str], label: str) -> None:
    missing = sorted(field for field in required if field not in row)
    if missing:
        raise ValueError(f"{label} missing fields: {', '.join(missing)}")


def _require_unique_dates(rows: list[dict[str, Any]], label: str) -> None:
    dates = [row.get("date") for row in rows]
    if len(set(dates)) != len(dates):
        raise ValueError(f"{label} contains duplicate dates")


def check_dashboard_v2_data(root: Path) -> dict[str, int]:
    manifest = _load(root / "manifest.json")
    market = _load(root / "market" / "SOXL.json")
    comparison = _load(root / "comparison" / "default.json")
    profiles = manifest.get("default_comparison_profiles")
    if not isinstance(profiles, list) or not profiles:
        raise ValueError("manifest default_comparison_profiles must be a non-empty list")
    market_rows = market.get("rows")
    if not isinstance(market_rows, list) or not market_rows:
        raise ValueError("market SOXL rows must be a non-empty list")
    for row in market_rows:
        _require_fields(row, REQUIRED_MARKET_FIELDS, "market row")
    _require_unique_dates(market_rows, "market rows")
    for profile in profiles:
        model = _load(root / "models" / f"{profile}.json")
        rows = model.get("rows")
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"model {profile} rows must be a non-empty list")
        for row in rows:
            _require_fields(row, REQUIRED_MODEL_FIELDS, f"model {profile} row")
        _require_unique_dates(rows, f"model {profile} rows")
        if not isinstance(model.get("events"), list):
            raise ValueError(f"model {profile} events must be a list")
    comparison_rows = comparison.get("rows")
    if not isinstance(comparison_rows, list):
        raise ValueError("comparison rows must be a list")
    return {"profiles": len(profiles), "market_rows": len(market_rows), "comparison_rows": len(comparison_rows)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Check dashboard v2 data contract.")
    parser.add_argument("root", nargs="?", type=Path, default=Path("dashboard_data/v2"))
    args = parser.parse_args()
    result = check_dashboard_v2_data(args.root)
    print(f"dashboard_v2_check=ok profiles={result['profiles']} market_rows={result['market_rows']} comparison_rows={result['comparison_rows']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run checker tests and live checker**

Run:

```bash
uv run --python qlib_venv/bin/python --with pytest pytest tests/test_dashboard_v2_checker.py tests/test_dashboard_v2_data.py -q
PYTHONPATH=. qlib_venv/bin/python scripts/check_dashboard_v2_data.py dashboard_data/v2
```

Expected: pytest reports `5 passed`; checker prints `dashboard_v2_check=ok`.

- [ ] **Step 5: Commit Task 3**

Run:

```bash
git add scripts/check_dashboard_v2_data.py tests/test_dashboard_v2_checker.py
git commit -m "Add dashboard v2 data checker"
```

## Task 4: Frontend Project Skeleton And Domain Tests

**Files:**
- Create: `web/dashboard-v2/package.json`
- Create: `web/dashboard-v2/tsconfig.json`
- Create: `web/dashboard-v2/vite.config.ts`
- Create: `web/dashboard-v2/index.html`
- Create: `web/dashboard-v2/src/types.ts`
- Create: `web/dashboard-v2/src/domain.ts`
- Create: `web/dashboard-v2/src/testFixtures.ts`
- Create: `web/dashboard-v2/src/domain.test.ts`

- [ ] **Step 1: Create frontend package files**

Use this `package.json`:

```json
{
  "scripts": {
    "dev": "vite --host 127.0.0.1",
    "build": "tsc --noEmit && vite build",
    "test": "vitest run"
  },
  "dependencies": {
    "lightweight-charts": "^5.2.0"
  },
  "devDependencies": {
    "typescript": "^5.9.0",
    "vite": "^7.2.0",
    "vitest": "^4.0.0",
    "jsdom": "^27.0.0"
  }
}
```

Use strict TypeScript and jsdom Vitest config:

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "useDefineForClassFields": true,
    "module": "ESNext",
    "lib": ["ES2022", "DOM", "DOM.Iterable"],
    "skipLibCheck": true,
    "moduleResolution": "Bundler",
    "allowImportingTsExtensions": true,
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "strict": true
  },
  "include": ["src"]
}
```

```typescript
import { defineConfig } from 'vite';

export default defineConfig({
  root: '.',
  server: {
    host: '127.0.0.1',
    port: 5173,
  },
  test: {
    environment: 'jsdom',
  },
});
```

- [ ] **Step 2: Write failing frontend domain tests**

```typescript
import { describe, expect, test } from 'vitest';
import { buildActionMarkers, clampComparisonProfiles, computeCommonWindow, computeInspectorRows, computeVisibleComparison } from './domain';
import { comparisonFixture, manifestFixture, marketFixture, modelFixture } from './testFixtures';

describe('dashboard v2 domain helpers', () => {
  test('clamps comparison profiles to two through four unique models', () => {
    expect(clampComparisonProfiles(['a'], ['a', 'b', 'c'])).toEqual(['a', 'b']);
    expect(clampComparisonProfiles(['a', 'a', 'b', 'c', 'd', 'e'], ['a', 'b', 'c', 'd', 'e'])).toEqual(['a', 'b', 'c', 'd']);
  });

  test('computes common overlap window across selected models and market rows', () => {
    expect(computeCommonWindow(marketFixture.rows, [modelFixture.rows])).toEqual({ start: '2026-01-02', end: '2026-01-06' });
  });

  test('builds action markers from event rows instead of price movement', () => {
    expect(buildActionMarkers(modelFixture.events).map((marker) => marker.text)).toEqual(['BUY', 'SELL']);
  });

  test('computes visible comparison deltas against baseline', () => {
    const rows = computeVisibleComparison(comparisonFixture, '2026-01-02', '2026-01-06');
    expect(rows[1].profiles.smoke.nav_delta_vs_baseline).toBe(5);
  });

  test('computes inspector rows for hovered date', () => {
    const rows = computeInspectorRows({
      date: '2026-01-05',
      manifest: manifestFixture,
      market: marketFixture,
      models: { smoke: modelFixture },
      comparison: comparisonFixture,
    });
    expect(rows.some((row) => row.label === 'SOXL close' && row.value === '11.00')).toBe(true);
    expect(rows.some((row) => row.label === 'smoke action' && row.value === 'BUY')).toBe(true);
  });
});
```

- [ ] **Step 3: Run tests to verify they fail because domain helpers do not exist**

Run:

```bash
cd web/dashboard-v2
npm install
npm test
```

Expected: FAIL with unresolved imports from `./domain`.

- [ ] **Step 4: Implement types, fixtures, and domain helpers**

Implement `types.ts` with exact JSON contracts from the Python builder:

```typescript
export type Action = 'BUY' | 'SELL' | 'HOLD';
export type EventKind = 'BUY' | 'SELL' | 'POSITION_INCREASE' | 'POSITION_DECREASE';

export interface Manifest {
  generated_at: string;
  latest_market_date: string;
  safe_backtest_end: string;
  available_profiles: string[];
  default_baseline_profile: string;
  default_comparison_profiles: string[];
  source_paths: Record<string, string>;
}

export interface MarketRow {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  [key: string]: string | number | boolean | null;
}

export interface ModelRow {
  date: string;
  strategy_nav: number;
  buyhold_nav: number;
  position: number;
  action: Action;
  price: number;
  excess: number;
  excess_pct: number;
}

export interface ModelEvent {
  date: string;
  kind: EventKind;
  price: number;
  position: number;
  position_delta: number;
}

export interface ModelPayload {
  profile: string;
  status: string;
  source_path: string;
  replay_start: string;
  replay_end: string;
  row_count: number;
  rows: ModelRow[];
  events: ModelEvent[];
}

export interface MarketPayload {
  symbol: 'SOXL';
  rows: MarketRow[];
}

export interface ComparisonProfileValue {
  strategy_nav: number;
  position: number;
  action: Action;
  nav_delta_vs_baseline: number;
  position_delta_vs_baseline: number;
  action_mismatch: boolean;
}

export interface ComparisonRow {
  date: string;
  profiles: Record<string, ComparisonProfileValue>;
  action_mismatch_count: number;
}

export interface ComparisonPayload {
  baseline_profile: string;
  compared_profiles: string[];
  common_start: string;
  common_end: string;
  rows: ComparisonRow[];
}

export interface InspectorRow {
  label: string;
  value: string;
  tone?: 'buy' | 'sell' | 'warning' | 'muted';
}
```

Implement `domain.ts` as pure functions:

```typescript
import type { ComparisonPayload, ComparisonRow, InspectorRow, Manifest, MarketPayload, MarketRow, ModelEvent, ModelPayload, ModelRow } from './types';

export function clampComparisonProfiles(selected: string[], available: string[]): string[] {
  const unique = selected.filter((profile, index) => selected.indexOf(profile) === index && available.includes(profile));
  const filled = [...unique];
  for (const profile of available) {
    if (filled.length >= 2) break;
    if (!filled.includes(profile)) filled.push(profile);
  }
  return filled.slice(0, 4);
}

export function computeCommonWindow(marketRows: MarketRow[], modelRowsList: ModelRow[][]): { start: string; end: string } | null {
  const sets = [new Set(marketRows.map((row) => row.date)), ...modelRowsList.map((rows) => new Set(rows.map((row) => row.date)))];
  const common = [...sets[0]].filter((date) => sets.every((set) => set.has(date))).sort();
  if (common.length === 0) return null;
  return { start: common[0], end: common[common.length - 1] };
}

export function buildActionMarkers(events: ModelEvent[]) {
  return events
    .filter((event) => event.kind === 'BUY' || event.kind === 'SELL')
    .map((event) => ({
      time: event.date,
      position: event.kind === 'BUY' ? 'belowBar' as const : 'aboveBar' as const,
      color: event.kind === 'BUY' ? '#0f766e' : '#b91c1c',
      shape: event.kind === 'BUY' ? 'arrowUp' as const : 'arrowDown' as const,
      text: event.kind,
    }));
}

export function computeVisibleComparison(comparison: ComparisonPayload, start: string, end: string): ComparisonRow[] {
  return comparison.rows.filter((row) => row.date >= start && row.date <= end);
}

function formatNumber(value: number | null | undefined, digits = 2): string {
  return typeof value === 'number' && Number.isFinite(value) ? value.toFixed(digits) : 'n/a';
}

export function computeInspectorRows(input: {
  date: string;
  manifest: Manifest;
  market: MarketPayload;
  models: Record<string, ModelPayload>;
  comparison: ComparisonPayload;
}): InspectorRow[] {
  const marketRow = input.market.rows.find((row) => row.date === input.date);
  const comparisonRow = input.comparison.rows.find((row) => row.date === input.date);
  const rows: InspectorRow[] = [
    { label: 'date', value: input.date },
    { label: 'safe end', value: input.manifest.safe_backtest_end, tone: 'muted' },
  ];
  if (marketRow) {
    rows.push({ label: 'SOXL close', value: formatNumber(Number(marketRow.close)) });
    rows.push({ label: 'volume', value: formatNumber(Number(marketRow.volume), 0), tone: 'muted' });
    for (const key of ['vix', 'rsi', 'rsi6', 'volatility', 'spread', 'vix_zscore', 'panic_score']) {
      if (marketRow[key] !== undefined && marketRow[key] !== null) {
        rows.push({ label: key, value: String(marketRow[key]) });
      }
    }
  }
  for (const [profile, model] of Object.entries(input.models)) {
    const modelRow = model.rows.find((row) => row.date === input.date);
    if (modelRow) {
      rows.push({ label: `${profile} action`, value: modelRow.action, tone: modelRow.action === 'BUY' ? 'buy' : modelRow.action === 'SELL' ? 'sell' : 'muted' });
      rows.push({ label: `${profile} position`, value: formatNumber(modelRow.position, 3) });
    }
  }
  if (comparisonRow) {
    rows.push({ label: 'action mismatches', value: String(comparisonRow.action_mismatch_count), tone: comparisonRow.action_mismatch_count > 0 ? 'warning' : 'muted' });
  }
  return rows;
}
```

- [ ] **Step 5: Run frontend domain tests**

Run:

```bash
cd web/dashboard-v2
npm test
```

Expected: `5 passed`.

- [ ] **Step 6: Commit Task 4**

Run:

```bash
git add web/dashboard-v2
git commit -m "Add dashboard v2 frontend domain layer"
```

## Task 5: Frontend Data Loading And UI Rendering

**Files:**
- Create: `web/dashboard-v2/src/data.ts`
- Create: `web/dashboard-v2/src/chart.ts`
- Create: `web/dashboard-v2/src/main.ts`
- Create: `web/dashboard-v2/src/styles.css`
- Modify: `web/dashboard-v2/index.html`

- [ ] **Step 1: Implement data loading**

```typescript
import type { ComparisonPayload, Manifest, MarketPayload, ModelPayload } from './types';

export interface DashboardData {
  manifest: Manifest;
  market: MarketPayload;
  models: Record<string, ModelPayload>;
  comparison: ComparisonPayload;
}

async function fetchJson<T>(baseUrl: string, path: string): Promise<T> {
  const response = await fetch(`${baseUrl.replace(/\/$/, '')}/${path}`);
  if (!response.ok) {
    throw new Error(`failed to load ${path}: HTTP ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export async function loadDashboardData(baseUrl = '/dashboard_data/v2'): Promise<DashboardData> {
  const manifest = await fetchJson<Manifest>(baseUrl, 'manifest.json');
  const market = await fetchJson<MarketPayload>(baseUrl, 'market/SOXL.json');
  const comparison = await fetchJson<ComparisonPayload>(baseUrl, 'comparison/default.json');
  const models: Record<string, ModelPayload> = {};
  for (const profile of manifest.default_comparison_profiles) {
    models[profile] = await fetchJson<ModelPayload>(baseUrl, `models/${profile}.json`);
  }
  return { manifest, market, models, comparison };
}
```

- [ ] **Step 2: Implement Lightweight Charts adapter**

```typescript
import { createChart, type IChartApi, type ISeriesApi, type UTCTimestamp } from 'lightweight-charts';
import { buildActionMarkers } from './domain';
import type { MarketPayload, ModelPayload } from './types';

function toTimestamp(date: string): UTCTimestamp {
  return Math.floor(new Date(`${date}T00:00:00Z`).getTime() / 1000) as UTCTimestamp;
}

export interface ChartHandles {
  chart: IChartApi;
  candleSeries: ISeriesApi<'Candlestick'>;
}

export function renderMarketChart(container: HTMLElement, market: MarketPayload, model: ModelPayload): ChartHandles {
  container.replaceChildren();
  const chart = createChart(container, {
    height: container.clientHeight || 460,
    layout: { background: { color: '#111827' }, textColor: '#d1d5db' },
    grid: { vertLines: { color: '#273244' }, horzLines: { color: '#273244' } },
    rightPriceScale: { borderColor: '#374151' },
    timeScale: { borderColor: '#374151' },
    crosshair: { mode: 1 },
  });
  const candleSeries = chart.addCandlestickSeries({
    upColor: '#0f766e',
    downColor: '#b91c1c',
    borderVisible: false,
    wickUpColor: '#34d399',
    wickDownColor: '#f87171',
  });
  candleSeries.setData(
    market.rows.map((row) => ({
      time: toTimestamp(row.date),
      open: Number(row.open),
      high: Number(row.high),
      low: Number(row.low),
      close: Number(row.close),
    }))
  );
  candleSeries.setMarkers(
    buildActionMarkers(model.events).map((marker) => ({
      ...marker,
      time: toTimestamp(String(marker.time)),
    }))
  );
  chart.timeScale().fitContent();
  return { chart, candleSeries };
}
```

- [ ] **Step 3: Implement dense local UI**

`main.ts` should create:

- Top toolbar: baseline label, selected model chips, preset/custom date inputs.
- Main chart area: full-width K-line chart, not a card nested in another card.
- Right inspector: rows from `computeInspectorRows()`.
- Lower panels: model NAV rows, position row, action mismatch list, data source warning panel.

Use stable IDs in DOM:

```html
<div id="app">
  <header id="toolbar"></header>
  <main id="workspace">
    <section id="chart-panel"><div id="market-chart"></div></section>
    <aside id="inspector"></aside>
  </main>
  <section id="comparison-panel"></section>
  <section id="source-panel"></section>
</div>
```

The chart must render `manifest.default_comparison_profiles[0]` initially and update inspector content on crosshair move by nearest visible date.

- [ ] **Step 4: Implement CSS**

Use a dark neutral terminal layout with distinct green/red/cyan/amber accents, fixed toolbar height, `minmax(0, 1fr)` chart sizing, and a mobile breakpoint that stacks inspector below the chart.

- [ ] **Step 5: Run frontend build**

Run:

```bash
cd web/dashboard-v2
npm run build
```

Expected: TypeScript and Vite build exit `0`.

- [ ] **Step 6: Commit Task 5**

Run:

```bash
git add web/dashboard-v2
git commit -m "Add dashboard v2 local UI"
```

## Task 6: Browser Visual Verification

**Files:**
- Create generated evidence only under `runtime/production/card/dashboard_v2_visual_v1/`

- [ ] **Step 1: Build fresh data and frontend**

Run:

```bash
PYTHONPATH=. qlib_venv/bin/python scripts/build_dashboard_v2_data.py --profile smoke --profile balanced_rule_residual_v22 --baseline balanced_rule_residual_v22
PYTHONPATH=. qlib_venv/bin/python scripts/check_dashboard_v2_data.py dashboard_data/v2
cd web/dashboard-v2
npm run build
```

Expected: all three commands exit `0`; checker prints `dashboard_v2_check=ok`.

- [ ] **Step 2: Run local dev server**

Run:

```bash
cd web/dashboard-v2
npm run dev -- --port 5173
```

Expected: Vite prints a local URL at `http://127.0.0.1:5173/`.

- [ ] **Step 3: Open and inspect in browser**

Use the Browser plugin or Playwright to open:

```text
http://127.0.0.1:5173/
```

Verify:

- K-line candles are visible.
- BUY/SELL markers are visible on the price timeline.
- Inspector shows SOXL close, feature values, selected model action, and position.
- Lower comparison panel shows NAV/position/action-mismatch content for at least two models.
- Source panel shows `safe_backtest_end` and source paths.

- [ ] **Step 4: Capture screenshots and pixel evidence**

Save:

- `runtime/production/card/dashboard_v2_visual_v1/dashboard_v2_desktop.png`
- `runtime/production/card/dashboard_v2_visual_v1/dashboard_v2_mobile.png`
- `runtime/production/card/dashboard_v2_visual_v1/visual_check.json`

`visual_check.json` should contain:

```json
{
  "desktop_nonwhite_ratio_min": 0.05,
  "mobile_nonwhite_ratio_min": 0.05,
  "required_text": ["SOXL", "BUY", "SELL", "Position", "safe_backtest_end"]
}
```

- [ ] **Step 5: Commit visual evidence**

Run:

```bash
git add runtime/production/card/dashboard_v2_visual_v1
git commit -m "Add dashboard v2 visual evidence"
```

## Task 7: Final Regression And Push

**Files:**
- No new source files unless a verification failure exposes a required fix.

- [ ] **Step 1: Run focused regression set**

Run:

```bash
uv run --python qlib_venv/bin/python --with pytest pytest tests/test_dashboard_v2_data.py tests/test_dashboard_v2_checker.py tests/test_dashboard.py tests/test_dashboard_server.py tests/test_dashboard_model_replays.py -q
cd web/dashboard-v2 && npm test && npm run build
PYTHONPATH=. qlib_venv/bin/python scripts/check_dashboard_v2_data.py dashboard_data/v2
```

Expected:

- Pytest exits `0`.
- `npm test` exits `0`.
- `npm run build` exits `0`.
- Checker prints `dashboard_v2_check=ok`.

- [ ] **Step 2: Confirm scoped git status**

Run:

```bash
git status --short
```

Expected: only dashboard v2 source, tests, generated `dashboard_data/v2`, and visual evidence are staged or modified by this work. Pre-existing unrelated `data/`, `runtime/index.json`, model report CSV/JSON, and `.codex/config.toml` remain unstaged unless the user explicitly asks to include them.

- [ ] **Step 3: Push all dashboard v2 commits**

Run:

```bash
git push
```

Expected: push exits `0`.

## Self-Review Notes

- Spec coverage: Tasks 1-3 cover replay-source manifest, market/model/comparison JSON, events, position, actions, and data checker. Tasks 4-5 cover local Vite/TypeScript frontend, two-to-four model selection rules, overlap clipping, markers, inspector, and source panel. Task 6 covers browser visual proof. Task 7 covers regression and push.
- Scope: The plan keeps legacy `dashboard.html` and `scripts/build_backtest_dashboard.py` intact. It does not optimize for Vercel or remote static deployment.
- Type consistency: Python emits `date`, `strategy_nav`, `buyhold_nav`, `position`, `action`, `price`, `excess`, `excess_pct`, `events`, `manifest`, `market`, `models`, and `comparison`; TypeScript types and helpers use the same names.
- Placeholder scan: The plan contains no unresolved marker text and no unspecified code step.
