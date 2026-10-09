"""Reproducible re-run of the Stage1 production risk gate from committed evidence.

The production card is blocked by a single risk-gate miss
(``worst_rolling_5d_pct = -12.11%``) that comes from a recorder id which is not
present in the repository. This workflow recomputes the four risk gates from the
recorders that ARE committed, so the block either reproduces on verifiable
evidence or is shown to be a provenance artifact.

It reads only committed strategy-return artifacts and writes a compact
sidecar. It does not resolve, start, or modify any recorder.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

EXPERIMENT = "stage1_gate_recheck_20261010"
SCHEMA_VERSION = "stage1_gate_recheck_v1"
QLIB_DOCS = [
    "https://qlib.readthedocs.io/en/latest/component/data.html",
    "https://qlib.readthedocs.io/en/latest/component/recorder.html",
]
SOURCE_CARD = "intraday_on_policy_staged_validation_v1"
STRATEGY_FILE = "strategy_returns.csv"
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


def committed_strategy_returns(root: Path) -> list[tuple[str, Path]]:
    base = root / "research" / SOURCE_CARD / "qlib_recorder"
    matches = sorted(base.glob(f"*/**/{STRATEGY_FILE}"))
    _require(matches, "no committed strategy-return artifacts were found")
    found = []
    for path in matches:
        recorder_id = path.parents[2].name if len(path.parents) > 2 else path.parent.name
        found.append((recorder_id, path))
    return found


def risk_metrics(returns: pd.Series) -> dict:
    r = returns.to_numpy(float)
    growth = np.cumprod(1.0 + r)
    tail = np.sort(r)[:max(1, int(np.ceil(0.05 * len(r))))]
    rolling5 = pd.Series(r).rolling(5).apply(lambda x: float(np.prod(1.0 + x) - 1.0), raw=True)
    return {
        "sessions": int(len(r)),
        "return_pct": 100.0 * float(growth[-1] - 1.0),
        "max_drawdown_pct": 100.0 * float((growth / np.maximum.accumulate(growth) - 1.0).min()),
        "worst_day_pct": 100.0 * float(r.min()),
        "cvar5_pct": 100.0 * float(tail.mean()),
        "worst_rolling_5d_pct": 100.0 * float(rolling5.min()),
    }


def gate_checks(metrics: dict) -> dict:
    checks = {key: bool(metrics[key] >= floor) for key, floor in FLOORS.items()}
    return {"checks": checks, "pass": all(checks.values()), "floors": FLOORS}


def run_stage1_gate_recheck(*, root: Path | str | None = None, card_dir: Path | str | None = None) -> dict[str, Any]:
    spec_root = resolve_root(root)
    spec = RecheckSpec(root=spec_root)
    target_card = Path(card_dir) if card_dir is not None else (spec.root / "research" / spec.experiment)
    target_card.mkdir(parents=True, exist_ok=True)

    rows = []
    for recorder_id, path in committed_strategy_returns(spec.root):
        frame = pd.read_csv(path, parse_dates=["signal_date"]).sort_values("signal_date")
        metrics = risk_metrics(frame["strategy_return"])
        gates = gate_checks(metrics)
        rows.append({
            "recorder_id": recorder_id,
            "artifact": str(path.relative_to(spec.root)),
            "start": str(frame["signal_date"].min().date()),
            "end": str(frame["signal_date"].max().date()),
            **metrics,
            "gate_pass": gates["pass"],
            **{f"gate_{key}": value for key, value in gates["checks"].items()},
        })
    table = pd.DataFrame(rows)
    all_pass = bool(table["gate_pass"].all())
    any_fail = bool((~table["gate_pass"]).any())
    summary = {
        "experiment": spec.experiment,
        "schema_version": SCHEMA_VERSION,
        "card_status": "diagnostic",
        "decision": ("committed_evidence_passes_all_gates" if all_pass and not any_fail
                     else "committed_evidence_has_gate_failure"),
        "promotion_eligible": False,
        "source_card": SOURCE_CARD,
        "floors": FLOORS,
        "recorders": table.to_dict(orient="records"),
        "all_committed_pass": all_pass,
        "any_committed_fail": any_fail,
        "artifact_policy": {"policy": "slim_card_recorder_first", "workflow_config": "workflow.yaml",
                            "recorder_only_artifacts": []},
        "limitations": [
            "Recomputes the gate only from recorders already committed to the repository.",
            "A pass here does not by itself promote the strategy; forward evidence is still required.",
            "The production card's un-committed recorder cannot be audited by this recheck.",
        ],
    }
    write_card_files(summary, card_dir=target_card)
    return summary


def write_card_files(summary: dict, *, card_dir: Path) -> None:
    card_dir.mkdir(parents=True, exist_ok=True)
    (card_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    pd.DataFrame(summary["recorders"]).to_csv(card_dir / "gate_recheck.csv", index=False)
    workflow = {
        "schema_version": SCHEMA_VERSION, "experiment": summary["experiment"],
        "source_card": summary["source_card"], "floors": summary["floors"],
        "artifact_policy": summary["artifact_policy"], "qlib_docs_reference": QLIB_DOCS,
    }
    (card_dir / "workflow.yaml").write_text(yaml.safe_dump(workflow, sort_keys=False), encoding="utf-8")
    (card_dir / "conclusion.md").write_text(_conclusion(summary), encoding="utf-8")


def _conclusion(summary: dict) -> str:
    lines = [
        "# Stage1 风险门可复现复核（中文结论）\n",
        f"- 目的：用**已提交**的 recorder 重算四个风险门，验证生产卡的 `worst_rolling_5d_pct=-12.11%` 是否可复现。\n",
        f"- 来源卡：{summary['source_card']}；门槛：{summary['floors']}。\n",
        "- 结果：\n",
    ]
    for row in summary["recorders"]:
        lines.append(
            f"  - recorder {row['recorder_id'][:12]}（{row['start']}..{row['end']}）："
            f"收益 {row['return_pct']:.1f}%，MaxDD {row['max_drawdown_pct']:.2f}%，"
            f"worstDay {row['worst_day_pct']:.2f}%，CVaR5 {row['cvar5_pct']:.2f}%，"
            f"worst5d {row['worst_rolling_5d_pct']:.2f}%，gate={'PASS' if row['gate_pass'] else 'FAIL'}\n")
    lines.append(f"- 已提交证据整体：{'全部 PASS' if summary['all_committed_pass'] else '存在 FAIL'}。\n")
    lines.append(f"- 决定 `{summary['decision']}`。\n")
    lines.append(
        "- 限制：只复核已提交 recorder；生产卡未提交的 recorder 无法在此验证；"
        "通过本身不等于晋升，仍需连续前向证据。\n")
    return "".join(lines)
