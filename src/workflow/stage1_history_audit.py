"""Stage1 lineage and forward-evidence eligibility card, fail-closed.

Audits the legacy research lineage and the active release separately, reports
the active release's development risk-gate failures, and states the eligible
paired forward-session count under the existing ``forward_plan.json`` contract.
Raw observed slot dates are reported but never counted as eligible forward.
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
    active_release,
    active_release_development_risk,
    audit_lineage,
    forward_evidence_report,
    lineage_findings,
    load_forward_plan,
    walk_forward_verdict,
)

EXPERIMENT = "stage1_history_audit_20261010"
SCHEMA_VERSION = "stage1_history_audit_v2"
QLIB_DOCS = [
    "https://qlib.readthedocs.io/en/latest/component/data.html",
    "https://qlib.readthedocs.io/en/latest/component/recorder.html",
]
STAGE1_RUNTIME = "runtime/soxl/stage1_futu_sim"
REGIME_BOOKS = "runtime/soxl/stage1_regime_paper/stage1_regime_paper_20261001/books"


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


def regime_paired_book_sessions(root: Path) -> dict[str, int]:
    """Complete-session history length of each independent paired paper book.

    These are reported as an independent line and are never summed with the
    active SIM's observed dates.
    """
    base = root / REGIME_BOOKS
    if not base.is_dir():
        return {}
    out = {}
    for path in sorted(base.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        history = payload.get("history")
        if isinstance(history, list):
            out[path.stem] = len(history)
    return out


def load_contamination(root: Path) -> dict:
    path = root / STAGE1_RUNTIME / "paper_cycle.json"
    if not path.is_file():
        return {"windows": [], "status": "needs_data"}
    payload = json.loads(path.read_text(encoding="utf-8"))
    repair = payload.get("execution_repair") or {}
    contaminated = repair.get("prior_recovery_slots_contaminated") or []
    effective = repair.get("effective_session_date")
    return {
        "source": str(path.relative_to(root)),
        "effective_session_date": effective,
        "contaminated_slots": contaminated,
        "windows": ["2026-08-03..2026-08-07"] if contaminated else [],
        "status": "declared" if contaminated else "none",
    }


def run_stage1_history_audit(*, root: Path | str | None = None, card_dir: Path | str | None = None) -> dict[str, Any]:
    spec_root = resolve_root(root)
    spec = HistorySpec(root=spec_root)
    target_card = Path(card_dir) if card_dir is not None else (spec.root / "research" / spec.experiment)
    target_card.mkdir(parents=True, exist_ok=True)

    table = audit_lineage(spec.root, lineage=LINEAGE, production=PRODUCTION_CARD)
    findings = lineage_findings(table)
    forward_plan = load_forward_plan(spec.root)
    contamination = load_contamination(spec.root)
    paired_books = regime_paired_book_sessions(spec.root)
    forward = forward_evidence_report(root=spec.root, forward_plan=forward_plan,
                                      contamination=contamination)
    verdict = walk_forward_verdict(findings, forward_evidence=forward)
    release = active_release(spec.root)
    development_risk = active_release_development_risk(spec.root)

    summary = {
        "experiment": spec.experiment,
        "schema_version": SCHEMA_VERSION,
        "card_status": "diagnostic",
        "decision": ("fresh_walk_forward_passed" if verdict["passed_fresh_walk_forward"]
                     else "no_fresh_walk_forward_evidence"),
        "promotion_eligible": False,
        "objective_achieved": False,
        "legacy_lineage": table.to_dict(orient="records"),
        "lineage_findings": findings,
        "active_release": release,
        "active_release_development_risk": development_risk,
        "forward_plan": forward_plan,
        "forward_plan_status": {"declared_plan": True, "verified": False,
                                "note": "read from the release manifest text; the seal is not verified here"},
        "forward_evidence": forward,
        "walk_forward_verdict": verdict,
        "independent_paper_books": paired_books,
        "contamination": contamination,
        "evidence_lines": {
            "active_sim_raw_observed_dates": forward["raw_observed_count"],
            "active_sim_settled_dates": forward["settled_count"],
            "active_sim_eligible_forward": forward["eligible_sessions"],
            "qualification": forward["qualification"],
            "legacy_lineage_recorders_present": findings["recorders_present"],
            "independent_paper_books": paired_books,
            "note": "三条线分别报告，不相加、不混用",
        },
        "artifact_policy": {"policy": "slim_card_recorder_first", "workflow_config": "workflow.yaml",
                            "recorder_only_artifacts": []},
        "limitations": [
            "Diagnostic only: there is no success branch; qualification is always needs_data here.",
            "Reads card summaries, the runtime execution state, paper-cycle repair metadata and the sealed release manifest; no replay.",
            "Raw completed_slots and settled dates are reported, never counted as eligible paired forward sessions.",
            "The active release's allocation/action identity is resolved and hash-verified via load_runtime_release, not by card path.",
            "Formal promotion requires the existing sealed validation contract, not this diagnostic.",
        ],
    }
    write_card_files(summary, card_dir=target_card)
    return summary


def write_card_files(summary: dict, *, card_dir: Path) -> None:
    card_dir.mkdir(parents=True, exist_ok=True)
    (card_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    pd.DataFrame(summary["legacy_lineage"]).to_csv(card_dir / "lineage.csv", index=False)
    workflow = {
        "schema_version": SCHEMA_VERSION, "experiment": summary["experiment"],
        "forward_plan_gate": {
            "minimum_complete_paired_sessions": summary["forward_plan"].get("minimum_complete_paired_sessions"),
            "minimum_sessions_per_regime": summary["forward_plan"].get("minimum_sessions_per_regime"),
            "minimum_regime_transitions": summary["forward_plan"].get("minimum_regime_transitions"),
        },
        "artifact_policy": summary["artifact_policy"], "qlib_docs_reference": QLIB_DOCS,
    }
    (card_dir / "workflow.yaml").write_text(yaml.safe_dump(workflow, sort_keys=False), encoding="utf-8")
    (card_dir / "conclusion.md").write_text(_conclusion(summary), encoding="utf-8")


def _risk_line(risk: dict) -> str:
    """Build the development-risk line only from verified release data."""
    if risk.get("status") != "ok":
        return (f"- 现役 release 开发门：**无法核验**（{risk.get('reason')}）；"
                "不得据此声明策略优劣。\n")
    failed = risk.get("failed_development_gates") or []
    if failed:
        return (f"- 现役 release 开发门：失败项 {failed}；"
                f"gate_pass={risk['development_gate_pass']}（来自已验签 load_runtime_release 的 action_summary）。\n")
    return (f"- 现役 release 开发门：全部通过；"
            f"gate_pass={risk['development_gate_pass']}（来自已验签 load_runtime_release 的 action_summary）。\n")


def _conclusion(summary: dict) -> str:
    f = summary["lineage_findings"]
    v = summary["walk_forward_verdict"]
    fwd = summary["forward_evidence"]
    risk = summary["active_release_development_risk"]
    rel = summary["active_release"]
    lines = [
        "# Stage1 溯源与有效前向证据审查（fail-closed，中文结论）\n",
        f"- 现役 release：`{rel['release_id']}`（allocation `{rel['allocation_recorder_id']}`，"
        f"action `{rel['action_recorder_id']}`）；研究状态 `{rel['research_promotion_status']}`。\n",
        _risk_line(risk),
        f"- 旧 lineage：{f['cards']} 张卡中 ok {f['ok_cards']} 张；"
        f"缺卡 {len(f['missing_card_cards'])}、缺字段 {len(f['missing_field_cards'])}、"
        f"空 recorder {len(f['empty_recorder_cards'])}；recorder 缺失 {f['recorders_missing']} 个。\n",
        f"- 是否有卡用 outer 选参（未知按用了处理）：{f['any_selection_used_outer']}。\n",
        f"- 有效前向证据：raw observed dates {fwd['raw_observed_count']} 个，"
        f"settled dates {fwd['settled_count']} 个；eligible paired sessions = None"
        f"（qualification=`{fwd['qualification']}`，门槛 {fwd['required_sessions']}）。\n",
        f"- 独立 paper books（单列 raw/settled，不相加）：{summary['independent_paper_books'] or '无'}。\n",
        "- 本审查只做诊断，无成功分支；正式晋升只接受既有完整封存验证合同。\n",
        f"- 污染期：{summary['contamination'].get('windows') or '无'}。\n",
        f"- **是否通过新鲜 walk-forward：{v['passed_fresh_walk_forward']}**。\n",
    ]
    if v["failed_reasons"]:
        lines.append("- 未通过原因：\n")
        for reason in v["failed_reasons"]:
            lines.append(f"  - {reason}\n")
    lines.append(f"- 决定 `{summary['decision']}`。\n")
    risk = summary.get("active_release_development_risk", {})
    if risk.get("status") == "ok" and risk.get("failed_development_gates"):
        lines.append(
            f"- 结论：之前引用的旧 summary recorder 不是现役模型；现役 release 的 action 开发期"
            f"在 {risk['failed_development_gates']} 上劣于其自身基线；`completed_slots` 只是原始观测日，"
            "缺完整/时序/配对/身份校验，不得当作合格前向。撤回“只是 provenance 缺陷”的说法。\n")
    elif risk.get("status") == "ok":
        lines.append(
            "- 结论：之前引用的旧 summary recorder 不是现役模型；现役 release 的开发门通过；"
            "`completed_slots` 只是原始观测日，不得当作合格前向。撤回“只是 provenance 缺陷”的说法。\n")
    else:
        lines.append(
            "- 结论：旧 summary recorder 不是现役模型；现役 release 开发门**无法核验**，不声明策略优劣；"
            "`completed_slots` 只是原始观测日，缺完整/时序/配对/身份校验，不得当作合格前向。\n")
    return "".join(lines)
