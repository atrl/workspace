"""Stage1 lineage and walk-forward integrity card.

Reads the Stage1 research lineage, checks holdout hygiene and recorder
provenance, and states plainly whether the strategy has ever passed a fresh
walk-forward. This is the durable record that the reused diagnostic window is
not a clean holdout.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.analysis.stage1_history_audit import (
    LINEAGE,
    PRODUCTION_CARD,
    audit_lineage,
    lineage_findings,
    walk_forward_verdict,
)

EXPERIMENT = "stage1_history_audit_20261010"
SCHEMA_VERSION = "stage1_history_audit_v1"
QLIB_DOCS = [
    "https://qlib.readthedocs.io/en/latest/component/data.html",
    "https://qlib.readthedocs.io/en/latest/component/recorder.html",
]
REQUIRED_FORWARD_SESSIONS = 60
STAGE1_RUNTIME = "runtime/soxl/stage1_futu_sim"


def _require(condition, message):
    if not condition:
        raise ValueError("blocked: " + message)


@dataclass(frozen=True)
class HistorySpec:
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


def managed_forward_sessions(root: Path) -> int:
    """Count distinct execution sessions the paper cycle actually traded."""
    state = root / STAGE1_RUNTIME / "execution_state_v2.json"
    if not state.is_file():
        return 0
    payload = json.loads(state.read_text(encoding="utf-8"))
    sessions = {
        value.get("execution_session_date")
        for value in payload.get("completed_slots", {}).values()
        if isinstance(value, dict) and value.get("status") in ("filled_confirmed", "hold_completed")
    }
    return len({s for s in sessions if s})


def run_stage1_history_audit(*, root: Path | str | None = None, card_dir: Path | str | None = None) -> dict[str, Any]:
    spec_root = resolve_root(root)
    spec = HistorySpec(root=spec_root)
    target_card = Path(card_dir) if card_dir is not None else (spec.root / "research" / spec.experiment)
    target_card.mkdir(parents=True, exist_ok=True)

    table = audit_lineage(spec.root, lineage=LINEAGE, production=PRODUCTION_CARD)
    findings = lineage_findings(table)
    forward = managed_forward_sessions(spec.root)
    verdict = walk_forward_verdict(findings, forward_sessions=forward,
                                   required_forward_sessions=REQUIRED_FORWARD_SESSIONS)

    summary = {
        "experiment": spec.experiment,
        "schema_version": SCHEMA_VERSION,
        "card_status": "diagnostic",
        "decision": ("fresh_walk_forward_passed" if verdict["passed_fresh_walk_forward"]
                     else "no_fresh_walk_forward_evidence"),
        "promotion_eligible": False,
        "objective_achieved": False,
        "lineage": table.to_dict(orient="records"),
        "findings": findings,
        "walk_forward_verdict": verdict,
        "forward_sessions": forward,
        "required_forward_sessions": REQUIRED_FORWARD_SESSIONS,
        "artifact_policy": {"policy": "slim_card_recorder_first", "workflow_config": "workflow.yaml",
                            "recorder_only_artifacts": []},
        "limitations": [
            "Reads card summaries and the runtime execution state only; it does not replay positions.",
            "A reused diagnostic window that the project has already inspected is not fresh evidence.",
        ],
    }
    write_card_files(summary, card_dir=target_card)
    return summary


def write_card_files(summary: dict, *, card_dir: Path) -> None:
    card_dir.mkdir(parents=True, exist_ok=True)
    (card_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    pd.DataFrame(summary["lineage"]).to_csv(card_dir / "lineage.csv", index=False)
    workflow = {
        "schema_version": SCHEMA_VERSION, "experiment": summary["experiment"],
        "required_forward_sessions": summary["required_forward_sessions"],
        "artifact_policy": summary["artifact_policy"], "qlib_docs_reference": QLIB_DOCS,
    }
    (card_dir / "workflow.yaml").write_text(yaml.safe_dump(workflow, sort_keys=False), encoding="utf-8")
    (card_dir / "conclusion.md").write_text(_conclusion(summary), encoding="utf-8")


def _conclusion(summary: dict) -> str:
    f = summary["findings"]
    v = summary["walk_forward_verdict"]
    lines = [
        "# Stage1 溯源与 walk-forward 完整性审查（中文结论）\n",
        f"- 审查卡数：{f['cards']}；其中研究晋级卡 {f['promoted_cards']} 张。\n",
        f"- recorder 声明 {f['recorders_declared']} 个，存在 {f['recorders_present']} 个，"
        f"**缺失 {f['recorders_missing']} 个**：{f['missing_recorder_cards'] or '无'}。\n",
        f"- 被标记为非新鲜 outer 的卡：{f['outer_not_fresh_cards'] or '无'}。\n",
        f"- 是否有卡用 outer 选参：{f['any_selection_used_outer']}。\n",
        f"- 受管理前向交易日：{summary['forward_sessions']}（门槛 {summary['required_forward_sessions']}）。\n",
        f"- **是否通过新鲜 walk-forward：{v['passed_fresh_walk_forward']}**。\n",
    ]
    if v["failed_reasons"]:
        lines.append("- 未通过原因：\n")
        for reason in v["failed_reasons"]:
            lines.append(f"  - {reason}\n")
    if f["missing_recorder_cards"]:
        lines.append(
            "- 溯源缺陷：上述卡的 recorder 不在仓库中，其结论无法独立复现；"
            "生产来源链中的卡片尤其受影响。\n")
    lines.append(f"- 决定 `{summary['decision']}`。\n")
    lines.append(
        "- 结论：2023-2026 为被项目反复查看的 reused diagnostic，不能当作样本外证据；"
        "在取得 ≥60 个受管理前向交易日并补齐 recorder 之前，不得宣称通过 walk-forward。\n")
    return "".join(lines)
