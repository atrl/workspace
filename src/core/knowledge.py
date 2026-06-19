from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class KnowledgeSource:
    source_id: str
    source_type: str
    title: str
    url: str
    summary: str
    license_note: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def build_hypothesis_card(
    source: KnowledgeSource,
    *,
    target_subsystem: str = "research",
    hypothesis: str = "待转化为可测试假设。",
    test_plan: str = "先进入 research notebook，再决定是否晋升到稳定框架。",
) -> dict[str, Any]:
    return {
        "schema_version": "knowledge_card.v1",
        "source": source.to_dict(),
        "target_subsystem": target_subsystem,
        "hypothesis": hypothesis,
        "test_plan": test_plan,
        "promotion_status": "candidate",
        "skill_boundary": "method_only",
        "skill_usage": "skill 只用于启发研究方法和发散思路，不承载项目业务状态。",
    }


def _default_target(source: KnowledgeSource) -> str:
    if source.source_type in {"github", "arxiv", "paper"}:
        return "src/model"
    if source.source_type in {"twitter", "x"}:
        return "src/analysis"
    return "research"


_CANDIDATE_PATTERNS: tuple[tuple[str, str, str, str], ...] = (
    ("multi-factor", "factor_diagnostics", "factor_pipeline_reference", "src/factor"),
    ("factor correlation", "factor_diagnostics", "factor_pipeline_reference", "src/factor"),
    ("factor", "factor_diagnostics", "factor_pipeline_reference", "src/factor"),
    ("patchtst", "patch_tst_mse", "model_replacement", "src/model"),
    ("patch tst", "patch_tst_mse", "model_replacement", "src/model"),
    ("dlinear", "dlinear_mse", "model_replacement", "src/model"),
    ("autoformer", "autoformer_mse", "model_replacement", "src/model"),
    ("fedformer", "fedformer_mse", "model_replacement", "src/model"),
    ("sfm", "sfm_mse", "model_replacement", "src/model"),
    ("rsr", "rsr_mse", "model_replacement", "src/model"),
    ("thgnn", "thgnn_mse", "model_replacement", "src/model"),
    ("sthan", "sthan_mse", "model_replacement", "src/model"),
    ("sthgcn", "sthgcn_mse", "model_replacement", "src/model"),
    ("qlib", "qlib_recorder_analysis", "analysis_extension", "src/analysis"),
    ("regime", "regime_window_review", "method_signal", "src/analysis"),
    ("quantbench", "quantbench_model_matrix", "model_matrix_reference", "src/model"),
)

_EVIDENCE_REQUIREMENTS: tuple[str, ...] = (
    "license_review",
    "research_notebook",
    "focused_pytest",
    "manifest_or_export",
    "chinese_conclusion",
)

_TARGET_INTERFACES = {
    "src/model": "src/model.registry.ModelSpec + src.model.training.fit_predict_model",
    "src/analysis": "src.analysis.qlib_recorder + src.analysis.build_regime_window_analysis_report",
    "src/factor": "src.factor.prepare_factor_feature_sets",
    "src/objective": "src.objective.evaluate_prediction_objectives",
    "research": "research/<experiment>/<experiment>.ipynb",
}


def _source_text(source: KnowledgeSource) -> str:
    return " ".join([source.source_type, source.title, source.summary, source.url]).lower()


def _candidate_for_source(source: KnowledgeSource) -> tuple[str, str, str]:
    text = _source_text(source)
    for pattern, name, candidate_type, target in _CANDIDATE_PATTERNS:
        if pattern in text:
            return name, candidate_type, target
    target = _default_target(source)
    candidate_type = "method_signal" if source.source_type in {"twitter", "x"} else "research_hypothesis"
    slug = source.source_id.replace(":", "_").replace("/", "_").replace(" ", "_").lower()
    return slug, candidate_type, target


def _priority_score(source: KnowledgeSource, *, target_subsystem: str, candidate_type: str) -> float:
    source_score = {
        "paper": 4.0,
        "arxiv": 4.0,
        "github": 3.0,
        "twitter": 1.0,
        "x": 1.0,
    }.get(source.source_type, 2.0)
    target_score = {
        "src/model": 3.0,
        "src/analysis": 2.0,
        "src/factor": 2.0,
        "src/objective": 2.0,
    }.get(target_subsystem, 1.0)
    type_score = {
        "model_replacement": 3.0,
        "analysis_extension": 2.0,
        "factor_pipeline_reference": 2.0,
        "model_matrix_reference": 2.0,
        "method_signal": 1.0,
    }.get(candidate_type, 1.0)
    return source_score + target_score + type_score


def _dedupe_sources(sources: Iterable[KnowledgeSource]) -> tuple[list[KnowledgeSource], int]:
    seen: set[str] = set()
    result: list[KnowledgeSource] = []
    deduped = 0
    for source in sources:
        if source.source_id in seen:
            deduped += 1
            continue
        seen.add(source.source_id)
        result.append(source)
    return result, deduped


def _license_gate(source: KnowledgeSource) -> str:
    return (
        "license gate: 只能保存 URL、自有摘要和结构化假设；"
        f"进入实现前必须复核来源许可。来源备注：{source.license_note}"
    )


def _allowed_next_surface(candidate_name: str) -> str:
    return f"research/{candidate_name}_validation_v1/{candidate_name}_validation_v1.ipynb"


def build_knowledge_backlog(sources: Iterable[KnowledgeSource]) -> list[dict[str, Any]]:
    """Convert external source cards into a prioritized, testable research backlog."""

    unique_sources, _ = _dedupe_sources(sources)
    rows: list[dict[str, Any]] = []
    for source in unique_sources:
        candidate_name, candidate_type, target_subsystem = _candidate_for_source(source)
        priority_score = _priority_score(
            source,
            target_subsystem=target_subsystem,
            candidate_type=candidate_type,
        )
        target_interface = _TARGET_INTERFACES.get(target_subsystem, _TARGET_INTERFACES["research"])
        rows.append(
            {
                "schema_version": "knowledge_backlog_item.v1",
                "priority_rank": 0,
                "priority_score": round(priority_score, 6),
                "source_id": source.source_id,
                "source_type": source.source_type,
                "title": source.title,
                "url": source.url,
                "candidate_name": candidate_name,
                "candidate_type": candidate_type,
                "target_subsystem": target_subsystem,
                "target_interface": target_interface,
                "hypothesis": (
                    f"评估 {source.title} 是否能转化为 SOXL 的 {target_subsystem} "
                    "稳定能力或 research notebook 假设。"
                ),
                "test_plan": (
                    "先在 research notebook 做最小可复现实验；"
                    "通过聚焦测试、manifest 和中文结论后，再考虑进入 src 或 production。"
                ),
                "acceptance_criteria": (
                    "research notebook 有中文结论；有聚焦 pytest；"
                    "dashboard/export 只读 manifest；未通过 license gate 前不得 vendoring 代码。"
                ),
                "risk_notes": (
                    "外部知识可能存在许可、实现漂移、过拟合、数据泄露或社交媒体噪声；"
                    "只能作为 hypothesis，不直接变成生产逻辑。"
                ),
                "license_gate": _license_gate(source),
                "promotion_gate_status": "blocked_pending_research",
                "allowed_next_surface": _allowed_next_surface(candidate_name),
                "blocked_surfaces_before_gate": ["src", "production", "skill"],
                "evidence_requirements": list(_EVIDENCE_REQUIREMENTS),
                "required_evidence_count": len(_EVIDENCE_REQUIREMENTS),
                "gate_notes": "未满足晋升门禁前，只能保留为 research backlog；不得写入 src、production 或 skill。",
                "promotion_status": "backlog",
                "source_used_as": "hypothesis_only",
                "skill_boundary": "method_only",
                "summary": source.summary,
            }
        )
    rows.sort(key=lambda item: (-float(item["priority_score"]), str(item["source_id"])))
    for index, item in enumerate(rows, start=1):
        item["priority_rank"] = index
    return rows


def _backlog_summary(items: list[dict[str, Any]], *, deduped: int) -> dict[str, Any]:
    return {
        "schema_version": "knowledge_backlog.summary.v1",
        "source_count": len(items),
        "deduped": deduped,
        "target_subsystems": sorted({str(item["target_subsystem"]) for item in items}),
        "candidate_types": sorted({str(item["candidate_type"]) for item in items}),
        "top_candidates": [str(item["candidate_name"]) for item in items[:5]],
    }


def write_knowledge_backlog(output_dir: str | Path, sources: Iterable[KnowledgeSource]) -> dict[str, int]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    unique_sources, deduped = _dedupe_sources(sources)
    items = build_knowledge_backlog(unique_sources)
    summary = _backlog_summary(items, deduped=deduped)
    payload = {
        "schema_version": "knowledge_backlog.v1",
        "summary": summary,
        "items": items,
    }
    (output / "knowledge_backlog.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    fieldnames = [
        "source_id",
        "candidate_name",
        "candidate_type",
        "target_subsystem",
        "target_interface",
        "priority_rank",
        "priority_score",
        "promotion_status",
        "promotion_gate_status",
        "required_evidence_count",
        "source_used_as",
        "skill_boundary",
        "url",
    ]
    with (output / "knowledge_backlog.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in items:
            writer.writerow({field: item.get(field, "") for field in fieldnames})
    review_lines = [
        "# 外部知识 Backlog Review",
        "",
        "## 中文结论",
        "",
        f"- 本次写入 {len(items)} 条 backlog，去重 {deduped} 条。",
        f"- 目标接口覆盖：{', '.join(summary['target_subsystems']) if items else '无'}。",
        "- 晋升门禁：每条候选必须先补 license review、research notebook、聚焦 pytest、manifest/export 和中文结论。",
        "- 所有来源只作为 hypothesis_only；未通过 license gate 前不进入实现。",
        "- 可运行功能必须进入 `src/`、`research/*.ipynb` 或 `production/*.ipynb`，不进入 skill。",
        "- 下一步按 priority_rank 逐项建立 research notebook、聚焦测试和 manifest 证据。",
    ]
    (output / "knowledge_backlog_review.md").write_text("\n".join(review_lines) + "\n", encoding="utf-8")
    return {"written": len(items), "deduped": deduped}


def write_knowledge_queue(output_dir: str | Path, sources: Iterable[KnowledgeSource]) -> dict[str, int]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cards: list[dict[str, Any]] = []
    seen: set[str] = set()
    deduped = 0
    for source in sources:
        if source.source_id in seen:
            deduped += 1
            continue
        seen.add(source.source_id)
        _, _, target_subsystem = _candidate_for_source(source)
        cards.append(
            build_hypothesis_card(
                source,
                target_subsystem=target_subsystem,
                hypothesis=f"评估 {source.title} 是否能转化为 SOXL 的因子、模型、objective 或 regime analysis 假设。",
                test_plan="在 research notebook 中做可复现实验；通过后再进入 src 稳定接口或 production notebook。",
            )
        )
    payload = {"schema_version": "knowledge_queue.v1", "cards": cards}
    (output / "knowledge_queue.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    review_lines = [
        "# 外部知识抓取队列 Review",
        "",
        "## 中文结论",
        "",
        f"- 本次写入 {len(cards)} 条结构化研究卡，去重 {deduped} 条。",
        "- 外部来源只能先作为假设候选，不直接进入生产逻辑。",
        "- 不要把抓取结果写入 skill；skill 只保留方法论和发散提示。",
        "- 后续应在 research notebook 中验证，再决定是否进入 `src/` 或 `production/`。",
    ]
    (output / "knowledge_ingest_review.md").write_text("\n".join(review_lines) + "\n", encoding="utf-8")
    return {"written": len(cards), "deduped": deduped}
