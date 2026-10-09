"""Legacy diagnostic recheck of the Stage1 risk gate. No promote/overturn branch.

The production card blocks on ``worst_rolling_5d_pct = -12.11%`` over the window
``2023-01-04..2026-08-07``. This module reports each legacy recorder's own risk
metrics and its comparability to that reference. It never declares the
production block a mutation of provenance:

- comparability requires identical source identity, cost and the *complete*
  reference calendar; "covers the window" is not the same as "same window";
- a shorter or differently-identified run is reported ``not_comparable``
  (``needs_data``), never as a pass that overturns the production failure;
- each legacy recorder's identity is resolved from the fixed on-disk store
  layout, its artifact SHA-256 is recorded, and a read-only Qlib ``Recorder``
  read is attempted and its outcome recorded honestly.

The module is read-only and never writes or mutates a recorder.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

EXPERIMENT = "stage1_gate_recheck_20261010"
SCHEMA_VERSION = "stage1_gate_recheck_v2"
QLIB_DOCS = [
    "https://qlib.readthedocs.io/en/latest/component/data.html",
    "https://qlib.readthedocs.io/en/latest/component/recorder.html",
]
SOURCE_CARD = "intraday_on_policy_staged_validation_v1"
ARTIFACT = f"{SOURCE_CARD}/strategy_returns.csv"
PRODUCTION_SUMMARY = "production/stage1_futu_sim_v1/summary.json"
FLOORS = {
    "max_drawdown_pct": -30.0,
    "worst_day_pct": -12.0,
    "cvar5_pct": -8.0,
    "worst_rolling_5d_pct": -12.0,
}


def _require(condition, message):
    if not condition:
        raise ValueError("blocked: " + message)


@dataclass(frozen=True)
class RecheckSpec:
    experiment: str = EXPERIMENT
    root: Path = Path(".")


def resolve_root(root: Path | str | None) -> Path:
    if root is not None:
        return Path(root).resolve()
    current = Path.cwd().resolve()
    while current != current.parent:
        if (current / "AGENTS.md").exists() and (current / "SPEC.md").exists():
            return current
        current = current.parent
    raise ValueError("could not resolve repository root")


def production_reference(root: Path) -> dict:
    path = root / PRODUCTION_SUMMARY
    _require(path.is_file(), "production summary is missing")
    summary = json.loads(path.read_text(encoding="utf-8"))
    window = (summary.get("backtest_comparison") or {}).get("window") or {}
    stage1 = (summary.get("backtest_comparison") or {}).get("stage1_10bps") or {}
    return {
        "source": str(path.relative_to(root)),
        "window_start": window.get("start"),
        "window_end": window.get("end"),
        "expected_rows": window.get("stage1_rows"),
        "cost_bps": 10.0,
        "worst_rolling_5d_pct": stage1.get("worst_rolling_5d_pct"),
        "risk_gate_pass": summary.get("risk_gate_pass"),
        "source_recorder_id": summary.get("source_recorder_id"),
        "recorder_id": summary.get("recorder_id"),
        "source_experiment": summary.get("source_experiment"),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_returns(frame: pd.DataFrame, *, source: str) -> pd.Series:
    _require("signal_date" in frame.columns, f"{source}: signal_date column is missing")
    _require("strategy_return" in frame.columns, f"{source}: strategy_return column is missing")
    dates = pd.to_datetime(frame["signal_date"])
    _require(not dates.isna().any(), f"{source}: signal_date has missing values")
    _require(not dates.duplicated().any(), f"{source}: duplicate signal_date rows")
    _require(dates.is_monotonic_increasing, f"{source}: signal_date is not ordered")
    returns = pd.to_numeric(frame["strategy_return"], errors="coerce")
    _require(np.isfinite(returns.to_numpy(float)).all(), f"{source}: non-finite strategy_return")
    _require((returns > -1.0).all(), f"{source}: strategy_return <= -100%")
    return returns.reset_index(drop=True)


def risk_metrics(returns: pd.Series) -> dict:
    """Risk metrics that include the initial NAV of 1.0 in the drawdown peak.

    With fewer than five sessions the rolling 5-day worst is undefined and is
    reported as ``None`` rather than NaN, so a short sample cannot slip a finite
    value past the gate.
    """
    r = pd.to_numeric(returns, errors="coerce").to_numpy(float)
    _require(r.size > 0, "empty return series")
    _require(np.isfinite(r).all(), "non-finite return series")
    _require((r > -1.0).all(), "return series at or below -100%")
    growth = np.concatenate([[1.0], np.cumprod(1.0 + r)])
    peak = np.maximum.accumulate(growth)
    tail = np.sort(r)[:max(1, int(np.ceil(0.05 * r.size)))]
    rolling5 = pd.Series(r).rolling(5).apply(lambda x: float(np.prod(1.0 + x) - 1.0), raw=True)
    worst5 = None if r.size < 5 else 100.0 * float(rolling5.min())
    return {
        "sessions": int(r.size),
        "return_pct": 100.0 * float(growth[-1] - 1.0),
        "max_drawdown_pct": 100.0 * float((growth / peak - 1.0).min()),
        "worst_day_pct": 100.0 * float(r.min()),
        "cvar5_pct": 100.0 * float(tail.mean()),
        "worst_rolling_5d_pct": worst5,
    }


def gate_checks(metrics: dict) -> dict:
    """Gate checks require a finite value; an undefined metric never passes."""
    checks = {}
    for key, floor in FLOORS.items():
        value = metrics.get(key)
        finite = value is not None and np.isfinite(value)
        checks[key] = bool(finite and value >= floor)
    return {"checks": checks, "pass": all(checks.values()), "floors": FLOORS}


def _git_tracked(root: Path, relative: str) -> bool:
    try:
        out = subprocess.run(["git", "ls-files", "--", relative], cwd=root,
                             capture_output=True, text=True, check=False)
        return bool(out.stdout.strip())
    except OSError:
        return False


def resolve_legacy_recorders(root: Path) -> list[dict]:
    """Enumerate committed recorders by the fixed on-disk store layout.

    Path shape: ``<card>/qlib_recorder/<experiment_id>/<recorder_id>/artifacts/
    <experiment_name>/strategy_returns.csv``. Identity comes from the path, and
    each artifact's SHA-256 is recorded.
    """
    base = root / "research" / SOURCE_CARD / "qlib_recorder"
    if not base.is_dir():
        return []
    out = []
    for path in sorted(base.glob(f"*/*/artifacts/{SOURCE_CARD}/strategy_returns.csv")):
        experiment_id = path.parents[3].name
        recorder_id = path.parents[2].name
        relative = str(path.relative_to(root))
        out.append({
            "experiment_id": experiment_id,
            "recorder_id": recorder_id,
            "artifact": relative,
            "artifact_sha256": _sha256(path) if path.is_file() else None,
            "git_tracked": _git_tracked(root, relative),
        })
    return out


def qlib_read_attempt(root: Path, *, recorder_id: str, experiment_name: str,
                      artifact: str) -> dict:
    """Read the named CSV artifact through the real read-only Qlib Recorder API.

    Qlib exposes ``list_artifacts`` (not ``list_objects``) for listing and
    ``download_artifact(path, dst_path)`` for non-pickled files such as CSV.
    ``R.get_recorder`` already resolves the experiment with ``create=False`` and
    the recorder with ``create=False, start=False``, so no run is written. Any
    real exception is captured verbatim rather than inferred.
    """
    import os
    import tempfile

    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    try:
        import qlib
        from qlib.workflow import R
    except Exception as error:  # noqa: BLE001
        return {"status": "reader_error", "reason": f"qlib import failed: {error}"}
    uri = (root / "research" / SOURCE_CARD / "qlib_recorder").resolve()
    try:
        qlib.init(provider_uri=str((root / "data/qlib/soxl").resolve()), region="us")
        with R.uri_context(str(uri)):
            # create=False on both the experiment and the recorder; no run is created.
            recorder = R.get_recorder(recorder_id=recorder_id, experiment_name=experiment_name)
            listed = recorder.list_artifacts()
            # A TemporaryDirectory context cleans the download up on exit.
            with tempfile.TemporaryDirectory() as dst:
                local = recorder.download_artifact(artifact, dst)
                downloaded = Path(local)
                downloaded_sha256 = _sha256(downloaded) if downloaded.is_file() else None
                frame = pd.read_csv(downloaded)
            return {
                "status": "readable", "recorder_id": recorder.id,
                "artifact_uri": str(recorder.artifact_uri),
                "artifact_listing": listed,
                "downloaded_rows": int(len(frame)),
                "downloaded_sha256": downloaded_sha256,
                "frame": frame,
                "api": ["qlib.workflow.R.get_recorder", "Recorder.list_artifacts",
                        "Recorder.download_artifact"],
            }
    except Exception as error:  # noqa: BLE001 - a real read error, recorded verbatim
        return {"status": "reader_error", "reason": f"{type(error).__name__}: {error}"}


def _same_window(reference: dict, start: str, end: str) -> bool:
    """True only when the run's window exactly equals the reference window."""
    return reference["window_start"] == start and reference["window_end"] == end


def run_stage1_gate_recheck(*, root: Path | str | None = None, card_dir: Path | str | None = None) -> dict[str, Any]:
    spec_root = resolve_root(root)
    spec = RecheckSpec(root=spec_root)
    target_card = Path(card_dir) if card_dir is not None else (spec.root / "research" / spec.experiment)
    target_card.mkdir(parents=True, exist_ok=True)

    reference = production_reference(spec.root)
    rows = []
    for entry in resolve_legacy_recorders(spec.root):
        path = spec.root / entry["artifact"]
        read = qlib_read_attempt(spec.root, recorder_id=entry["recorder_id"],
                                 experiment_name=SOURCE_CARD, artifact=ARTIFACT)
        # Qlib is the only primary source. There is no pandas-only fallback: if the
        # recorder is not readable through the real API, the row is needs_data and
        # no metrics or gate result are produced.
        if read.get("status") != "readable":
            rows.append({**entry, "status": "needs_data",
                         "reason": f"qlib read not readable: {read.get('reason')}",
                         "qlib_read": {k: v for k, v in read.items() if k != "frame"},
                         "comparability": "not_comparable"})
            continue
        # The downloaded bytes must match the on-disk artifact identity.
        if read.get("downloaded_sha256") != entry.get("artifact_sha256"):
            rows.append({**entry, "status": "needs_data",
                         "reason": "downloaded artifact SHA-256 differs from the stored artifact",
                         "qlib_read": {k: v for k, v in read.items() if k != "frame"},
                         "comparability": "not_comparable"})
            continue
        try:
            returns = _validate_returns(read["frame"], source=entry["artifact"])
        except (ValueError, KeyError) as error:
            rows.append({**entry, "status": "needs_data", "reason": str(error),
                         "qlib_read": {k: v for k, v in read.items() if k != "frame"},
                         "comparability": "not_comparable"})
            continue
        frame = read["frame"]
        dates = pd.to_datetime(frame["signal_date"])
        metrics = risk_metrics(returns)
        gates = gate_checks(metrics)
        start, end = str(dates.min().date()), str(dates.max().date())
        identity_match = entry["recorder_id"] == reference["source_recorder_id"]
        window_match = _same_window(reference, start, end)
        # Cost interpretation and complete-calendar alignment cannot be proven by
        # this diagnostic, so they are reported as unknown and comparability is
        # never claimed here.
        cost_status = "unknown"
        calendar_status = "unknown"
        rows.append({
            **entry, "status": "ok", "start": start, "end": end, **metrics,
            "gate_pass": gates["pass"],
            **{f"gate_{key}": value for key, value in gates["checks"].items()},
            "identity_match": identity_match,
            "window_match": window_match,
            "cost_status": cost_status,
            "calendar_status": calendar_status,
            "comparability": "not_comparable",
            "evidence_source": "qlib_download",
            "qlib_read": {k: v for k, v in read.items() if k != "frame"},
        })
    table = pd.DataFrame(rows)

    summary = {
        "experiment": spec.experiment,
        "schema_version": SCHEMA_VERSION,
        "card_status": "legacy_diagnostic",
        "decision": "legacy_diagnostic_no_recheck_verdict",
        "promotion_eligible": False,
        "objective_achieved": False,
        "source_card": SOURCE_CARD,
        "production_reference": reference,
        "floors": FLOORS,
        # Keep the original row list so mixed rows (reader_error, short/None,
        # normal) never get NaN-coerced by a DataFrame round-trip.
        "recorders": rows,
        "comparable_count": 0,
        "production_gate_override": False,
        "recheck_verdict": "none",
        "qlib_primary_source": True,
        "pandas_only_fallback": False,
        "retraction": (
            "This card makes no claim that the production risk failure is a provenance artifact. "
            "Legacy committed recorders are reported as diagnostics only; their Qlib readability and "
            "window/identity comparability are recorded, and only an exact same-window, same-identity, "
            "same-cost run could confirm or overturn the production result."),
        "artifact_policy": {"policy": "slim_card_recorder_first", "workflow_config": "workflow.yaml",
                            "recorder_only_artifacts": []},
        "limitations": [
            "Legacy diagnostic only; there is no pass/overturn branch and production_gate_override is false.",
            "Qlib is the only primary source; an unreadable recorder is needs_data with no metrics or gate result.",
            "Cost and complete-calendar alignment are unknown here, so comparability is never claimed.",
            "Passing a risk gate here would not promote the strategy; eligible forward evidence is still required.",
        ],
    }
    write_card_files(summary, card_dir=target_card)
    return summary


def write_card_files(summary: dict, *, card_dir: Path) -> None:
    card_dir.mkdir(parents=True, exist_ok=True)
    (card_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str, ensure_ascii=False, allow_nan=False),
        encoding="utf-8")
    # CSV is a separate tabular view; the JSON keeps the raw rows above.
    pd.DataFrame(summary["recorders"]).to_csv(card_dir / "gate_recheck.csv", index=False)
    workflow = {
        "schema_version": SCHEMA_VERSION, "experiment": summary["experiment"],
        "source_card": summary["source_card"], "floors": summary["floors"],
        "artifact_policy": summary["artifact_policy"], "qlib_docs_reference": QLIB_DOCS,
    }
    (card_dir / "workflow.yaml").write_text(yaml.safe_dump(workflow, sort_keys=False), encoding="utf-8")
    (card_dir / "conclusion.md").write_text(_conclusion(summary), encoding="utf-8")


def _conclusion(summary: dict) -> str:
    ref = summary["production_reference"]
    lines = [
        "# Stage1 风险门 legacy 诊断（无 recheck 结论，中文结论）\n",
        f"- 生产参考：窗口 {ref['window_start']}..{ref['window_end']}（{ref['expected_rows']} 行），"
        f"成本 {ref['cost_bps']}bp，source recorder `{ref['source_recorder_id']}`，"
        f"`worst_rolling_5d_pct={ref['worst_rolling_5d_pct']}`。\n",
        f"- 同窗同身份可比数：{summary['comparable_count']}；recheck 结论：`{summary['recheck_verdict']}`。\n",
        "- 各 legacy recorder：\n",
    ]
    for row in summary["recorders"]:
        if row.get("status") != "ok":
            lines.append(f"  - {row['recorder_id'][:12]}: needs_data（{row.get('reason')}）\n")
            continue
        worst5 = row['worst_rolling_5d_pct']
        worst5_txt = "undefined" if worst5 is None else f"{worst5:.2f}%"
        lines.append(
            f"  - {row['recorder_id'][:12]}（{row['start']}..{row['end']}，window_match={row['window_match']}，"
            f"identity_match={row['identity_match']}，cost={row['cost_status']}，calendar={row['calendar_status']}，"
            f"可比={row['comparability']}，qlib={row['qlib_read']['status']}）：收益 {row['return_pct']:.1f}%，"
            f"MaxDD {row['max_drawdown_pct']:.2f}%，worst5d {worst5_txt}\n")
    lines.append(f"- 撤回声明：{summary['retraction']}\n")
    return "".join(lines)
