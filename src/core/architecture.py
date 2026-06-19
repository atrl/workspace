from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .contracts import stable_hash
from .notebooks import scan_notebook_cards


ARCHITECTURE_EXTENSION_DECISION_SCHEMA_VERSION = "core.architecture_extension_decision.v1"
ARCHITECTURE_EXTENSION_REGISTRY_SCHEMA_VERSION = "core.architecture_extension_registry.v1"
PLATFORM_REQUIREMENT_AUDIT_SCHEMA_VERSION = "core.platform_requirement_audit.v1"
PLATFORM_REQUIREMENT_AUDIT_SUMMARY_SCHEMA_VERSION = "core.platform_requirement_audit.summary.v1"
PLATFORM_REQUIREMENT_AUDIT_WRITE_RESULT_SCHEMA_VERSION = "core.platform_requirement_audit_write_result.v1"


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in sorted(value.items())}
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value


def _clean_items(values: Iterable[str]) -> list[str]:
    return [str(value).strip() for value in values if str(value).strip()]


def _require_text(value: str, field_name: str) -> str:
    cleaned = str(value).strip()
    if not cleaned:
        raise ValueError(f"{field_name} is required")
    return cleaned


def architecture_extension_content_hash(payload: Mapping[str, Any]) -> str:
    content = {str(key): value for key, value in dict(payload).items() if key != "decision_hash"}
    return stable_hash(_json_ready(content))


def build_architecture_extension_decision(
    *,
    decision_id: str,
    target_surface: str,
    change_type: str,
    reason: str,
    impact_scope: Iterable[str],
    evidence_notebook: str,
    verification_commands: Iterable[str],
    alternatives_considered: Iterable[str] = (),
    constraints: Iterable[str] = (),
    status: str = "accepted",
    created_at: str = "",
) -> dict[str, Any]:
    impact = _clean_items(impact_scope)
    commands = _clean_items(verification_commands)
    if not impact:
        raise ValueError("impact_scope is required")
    if not commands:
        raise ValueError("verification_commands is required")

    payload: dict[str, Any] = {
        "schema_version": ARCHITECTURE_EXTENSION_DECISION_SCHEMA_VERSION,
        "decision_id": _require_text(decision_id, "decision_id"),
        "target_surface": _require_text(target_surface, "target_surface"),
        "change_type": _require_text(change_type, "change_type"),
        "reason": _require_text(reason, "reason"),
        "impact_scope": impact,
        "evidence_notebook": _require_text(evidence_notebook, "evidence_notebook"),
        "verification_commands": commands,
        "alternatives_considered": _clean_items(alternatives_considered),
        "constraints": _clean_items(constraints),
        "status": _require_text(status, "status"),
        "created_at": str(created_at).strip(),
        "extension_requires_reason": True,
        "blocked_without_evidence": True,
        "skill_boundary": "method_only",
        "review_gate": "src extension requires reason, evidence notebook, focused verification, and Chinese review.",
    }
    payload["decision_hash"] = architecture_extension_content_hash(payload)
    return _json_ready(payload)


def _load_registry(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"schema_version": ARCHITECTURE_EXTENSION_REGISTRY_SCHEMA_VERSION, "decisions": {}}
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.setdefault("schema_version", ARCHITECTURE_EXTENSION_REGISTRY_SCHEMA_VERSION)
    payload.setdefault("decisions", {})
    return payload


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _rel(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _requirement(
    *,
    requirement_id: str,
    category: str,
    requirement: str,
    status: str,
    evidence_paths: Iterable[str],
    metrics: Mapping[str, Any],
    notes: str,
) -> dict[str, Any]:
    return _json_ready(
        {
            "schema_version": "core.platform_requirement_audit_item.v1",
            "requirement_id": requirement_id,
            "category": category,
            "requirement": requirement,
            "status": status,
            "evidence_paths": _clean_items(evidence_paths),
            "metrics": dict(metrics),
            "notes": notes,
        }
    )


def _status_from_bool(condition: bool) -> str:
    return "pass" if condition else "fail"


def _contains_paths(payload: Mapping[str, Any], required: Iterable[str]) -> tuple[bool, list[str]]:
    paths = payload.get("paths", {})
    paths = paths if isinstance(paths, dict) else {}
    required_values = list(required)
    missing = [path for path in required_values if path not in paths]
    return not missing, missing


def build_platform_requirement_audit(root: str | Path) -> dict[str, Any]:
    """Map the platform objective to current repo evidence without claiming completion."""

    repo_root = Path(root).resolve()
    required_src_surfaces = [
        "src/core",
        "src/data",
        "src/factor",
        "src/model",
        "src/objective",
        "src/backtest",
        "src/analysis",
        "src/dashboard",
        "src/schedule",
        "src/api",
        "research",
        "production",
    ]
    missing_surfaces = [surface for surface in required_src_surfaces if not (repo_root / surface).is_dir()]
    requirements: list[dict[str, Any]] = [
        _requirement(
            requirement_id="stable_src_surfaces",
            category="architecture",
            requirement="全平台稳定 surface 必须沉到 src/，research/production 作为 notebook 证据面。",
            status=_status_from_bool(not missing_surfaces),
            evidence_paths=required_src_surfaces,
            metrics={
                "required_surface_count": len(required_src_surfaces),
                "missing_surface_count": len(missing_surfaces),
                "missing_surfaces": missing_surfaces,
            },
            notes="目录存在性只证明 surface 可用；具体行为由后续需求项和测试证明。",
        ),
        _requirement(
            requirement_id="reward_retired_objective_surface",
            category="objective",
            requirement="不再新增 RL/reward 训练面，reward 概念由 src/objective 替代。",
            status=_status_from_bool((repo_root / "src" / "objective").is_dir() and not (repo_root / "src" / "reward").exists()),
            evidence_paths=["src/objective"],
            metrics={
                "objective_surface_exists": (repo_root / "src" / "objective").is_dir(),
                "reward_surface_exists": (repo_root / "src" / "reward").exists(),
            },
            notes="只检查稳定 surface；旧未跟踪缓存不作为平台契约证据。",
        ),
    ]

    notebook_scan = scan_notebook_cards(repo_root)
    notebook_summary = dict(notebook_scan["summary"])
    requirements.append(
        _requirement(
            requirement_id="research_production_notebook_contract",
            category="notebook",
            requirement="research/production 每个卡片必须有单一主 notebook，且最后有中文结论。",
            status=_status_from_bool(
                notebook_summary.get("card_count", 0) > 0
                and notebook_summary.get("ready_count") == notebook_summary.get("card_count")
                and notebook_summary.get("missing_chinese_conclusion_count") == 0
                and notebook_summary.get("multiple_notebook_count") == 0
            ),
            evidence_paths=["research", "production"],
            metrics={
                "card_count": notebook_summary.get("card_count", 0),
                "ready_count": notebook_summary.get("ready_count", 0),
                "missing_chinese_conclusion_count": notebook_summary.get("missing_chinese_conclusion_count", 0),
                "multiple_notebook_count": notebook_summary.get("multiple_notebook_count", 0),
                "production_published_count": notebook_summary.get("production_published_count", 0),
                "production_template_count": notebook_summary.get("production_template_count", 0),
            },
            notes="notebook 是中文 review 和复现实验的主证据；CSV/JSON 只作为 sidecar。",
        )
    )

    from src.model.registry import QUANTBENCH_MODEL_NAMES, REQUESTED_QUANTBENCH_MODEL_NAMES

    requested = set(REQUESTED_QUANTBENCH_MODEL_NAMES)
    registered = set(QUANTBENCH_MODEL_NAMES)
    requirements.append(
        _requirement(
            requirement_id="quantbench_requested_model_registry",
            category="model",
            requirement="src/model 必须覆盖用户指定的 QuantBench 风格模型注册表。",
            status=_status_from_bool(requested == registered and len(requested) == 32),
            evidence_paths=["src/model/registry.py", "tests/test_model_adapter_coverage.py"],
            metrics={
                "requested_model_count": len(requested),
                "registered_model_count": len(registered),
                "missing_registered_models": sorted(requested - registered),
                "extra_registered_models": sorted(registered - requested),
            },
            notes="注册表覆盖 32 个用户点名模型；production ledger 另行证明训练/回测记录。",
        )
    )

    production_card = repo_root / "production" / "paper_sim_replay_v1"
    model_ledger_path = production_card / "matrix_card" / "model_run_ledger_summary.json"
    model_ledger = _load_json(model_ledger_path)
    model_counts_ok = all(
        int(model_ledger.get(key, 0) or 0) == int(model_ledger.get("expected_model_count", 0) or 0)
        for key in [
            "ledger_model_count",
            "trained_model_count",
            "predicted_model_count",
            "backtested_model_count",
            "objective_evaluated_model_count",
            "promotion_evaluated_model_count",
            "analysis_recorded_model_count",
            "manifested_model_count",
        ]
    )
    requirements.append(
        _requirement(
            requirement_id="quantbench_training_backtest_ledger",
            category="model",
            requirement="所有模型必须完成训练、预测、回测、objective、promotion 和 analysis 记录。",
            status=_status_from_bool(bool(model_ledger.get("all_models_completed")) and model_counts_ok),
            evidence_paths=[_rel(repo_root, model_ledger_path), "production/paper_sim_replay_v1/matrix_card/model_run_ledger.csv"],
            metrics={
                "expected_model_count": int(model_ledger.get("expected_model_count", 0) or 0),
                "trained_model_count": int(model_ledger.get("trained_model_count", 0) or 0),
                "predicted_model_count": int(model_ledger.get("predicted_model_count", 0) or 0),
                "backtested_model_count": int(model_ledger.get("backtested_model_count", 0) or 0),
                "objective_evaluated_model_count": int(model_ledger.get("objective_evaluated_model_count", 0) or 0),
                "promotion_evaluated_model_count": int(model_ledger.get("promotion_evaluated_model_count", 0) or 0),
                "analysis_recorded_model_count": int(model_ledger.get("analysis_recorded_model_count", 0) or 0),
                "all_models_completed": bool(model_ledger.get("all_models_completed")),
                "missing_models": list(model_ledger.get("missing_models", [])),
            },
            notes="production 证据面记录 32 个 base model x feature set variants 共 96 条模型运行。",
        )
    )

    analysis_manifest_path = production_card / "matrix_card" / "analysis_manifest.json"
    analysis_manifest = _load_json(analysis_manifest_path)
    required_analysis_paths = [
        "regime_metrics",
        "window_metrics",
        "window_regime_metrics",
        "model_comparison",
        "model_effect_deltas",
        "model_regime_window_diagnostics",
        "model_difference_explanations",
        "failure_modes",
        "objective_losses",
        "qlib_risk_metrics",
    ]
    analysis_ok, missing_analysis_paths = _contains_paths(analysis_manifest, required_analysis_paths)
    analysis_paths = analysis_manifest.get("paths", {})
    analysis_paths = analysis_paths if isinstance(analysis_paths, dict) else {}
    requirements.append(
        _requirement(
            requirement_id="analysis_regime_window_model_difference",
            category="analysis",
            requirement="src/analysis 必须输出 qlib/regime/window/model difference/failure/objective loss 统一分析产物。",
            status=_status_from_bool(analysis_ok),
            evidence_paths=[_rel(repo_root, analysis_manifest_path)],
            metrics={
                "analysis_paths": sorted(str(key) for key in analysis_paths),
                "missing_analysis_paths": missing_analysis_paths,
            },
            notes="dashboard、schedule 和 production 都应消费 manifest/export，而不是散落实验文件。",
        )
    )

    production_manifest_path = production_card / "production_manifest.json"
    dashboard_export_path = production_card / "dashboard_export.json"
    production_manifest = _load_json(production_manifest_path)
    dashboard_export = _load_json(dashboard_export_path)
    production_manifest_paths = production_manifest.get("paths", {})
    production_manifest_paths = production_manifest_paths if isinstance(production_manifest_paths, dict) else {}
    dashboard_artifacts = dashboard_export.get("artifacts", {})
    dashboard_artifacts = dashboard_artifacts if isinstance(dashboard_artifacts, dict) else {}
    dashboard_required = ["dashboard_export", "analysis_manifest", "summary"]
    dashboard_manifest_ok = all(key in production_manifest_paths for key in dashboard_required)
    dashboard_artifacts_ok = all(key in dashboard_artifacts for key in ["model_comparison", "regime_metrics", "window_metrics"])
    requirements.append(
        _requirement(
            requirement_id="dashboard_manifest_export_contract",
            category="dashboard",
            requirement="src/dashboard 只读取统一 manifest/export，并能展示模型、regime、window 和回测产物。",
            status=_status_from_bool(dashboard_manifest_ok and dashboard_artifacts_ok),
            evidence_paths=[_rel(repo_root, production_manifest_path), _rel(repo_root, dashboard_export_path)],
            metrics={
                "dashboard_schema_version": dashboard_export.get("schema_version", ""),
                "production_manifest_paths": sorted(str(key) for key in production_manifest_paths),
                "dashboard_artifacts": sorted(str(key) for key in dashboard_artifacts),
            },
            notes="检查 dashboard export 与 production manifest 的公开读取面。",
        )
    )

    data_registry_path = production_card / "data" / "data_registry.json"
    data_update_manifest_rel = str(production_manifest_paths.get("data_update_manifest", ""))
    data_update_manifest_path = production_card / data_update_manifest_rel if data_update_manifest_rel else Path()
    data_registry = _load_json(data_registry_path)
    data_update_manifest = _load_json(data_update_manifest_path)
    calendar_rel = str(data_update_manifest.get("dataset", {}).get("calendar_path", "")) if data_update_manifest else ""
    requirements.append(
        _requirement(
            requirement_id="data_canonical_snapshot_and_calendar",
            category="data",
            requirement="src/data 必须写出 canonical snapshot、data registry，并对齐 qlib calendar。",
            status=_status_from_bool(
                data_registry.get("schema_version") == "data.registry.v1"
                and data_update_manifest.get("schema_version") == "data.update_manifest.v1"
                and bool(calendar_rel)
                and (repo_root / calendar_rel).exists()
            ),
            evidence_paths=[_rel(repo_root, data_registry_path), _rel(repo_root, data_update_manifest_path), calendar_rel],
            metrics={
                "data_registry_schema": data_registry.get("schema_version", ""),
                "data_update_schema": data_update_manifest.get("schema_version", ""),
                "calendar_path": calendar_rel,
                "data_update_status": data_update_manifest.get("status", ""),
                "row_count": data_update_manifest.get("dataset", {}).get("metadata", {}).get("row_count", 0)
                if data_update_manifest
                else 0,
            },
            notes="只引用 production paper replay 发布的数据 registry/snapshot 证据。",
        )
    )

    production_summary_path = production_card / "production_summary.json"
    schedule_health_path = production_card / "schedule_health.json"
    paper_execution_path = production_card / "paper_execution_result.json"
    production_summary = _load_json(production_summary_path)
    schedule_health = _load_json(schedule_health_path)
    paper_execution = _load_json(paper_execution_path)
    requirements.append(
        _requirement(
            requirement_id="production_paper_api_guardrail",
            category="api",
            requirement="src/api v1 只允许 paper/sim，不做真实 broker 下单。",
            status=_status_from_bool(
                production_summary.get("mode") == "paper"
                and production_summary.get("live_broker_enabled") is False
                and paper_execution.get("status") == "paper_executed"
            ),
            evidence_paths=[_rel(repo_root, production_summary_path), _rel(repo_root, paper_execution_path)],
            metrics={
                "mode": production_summary.get("mode", ""),
                "live_broker_enabled": production_summary.get("live_broker_enabled", None),
                "paper_execution_status": paper_execution.get("status", ""),
                "paper_execution_schema_version": paper_execution.get("schema_version", ""),
            },
            notes="该项只证明 v1 paper/sim guardrail；真实 broker 下单仍在 out of scope。",
        )
    )
    requirements.append(
        _requirement(
            requirement_id="schedule_data_dashboard_api_chain",
            category="schedule",
            requirement="src/schedule 必须编排数据更新、dashboard export、paper execution、通知、锁和重试健康面。",
            status=_status_from_bool(
                production_summary.get("schedule_status") == "success"
                and schedule_health.get("schema_version") == "dashboard.schedule_health.v1"
                and int(schedule_health.get("failed_job_count", 1)) == 0
            ),
            evidence_paths=[_rel(repo_root, production_summary_path), _rel(repo_root, schedule_health_path)],
            metrics={
                "schedule_status": production_summary.get("schedule_status", ""),
                "schedule_health_schema_version": schedule_health.get("schema_version", ""),
                "job_count": schedule_health.get("job_count", 0),
                "failed_job_count": schedule_health.get("failed_job_count", 0),
                "total_retry_error_count": schedule_health.get("total_retry_error_count", 0),
            },
            notes="readiness gate 的 review 状态单独在 production_readiness_review_gate 中跟踪。",
        )
    )
    requirements.append(
        _requirement(
            requirement_id="production_readiness_review_gate",
            category="production",
            requirement="production health 必须显式记录 readiness gate，review/block 不能被误报为 ready。",
            status="review"
            if production_summary.get("schedule_readiness_overview_status") == "review"
            else _status_from_bool(production_summary.get("schedule_readiness_overview_status") == "ready"),
            evidence_paths=[_rel(repo_root, production_summary_path), _rel(repo_root, production_manifest_path)],
            metrics={
                "schedule_readiness_overview_status": production_summary.get("schedule_readiness_overview_status", ""),
                "schedule_readiness_overview_gates": production_summary.get("schedule_readiness_overview_gates", ""),
                "objective_loss_status": production_summary.get("objective_loss_status", ""),
                "objective_loss_top_component": production_summary.get("objective_loss_top_component", ""),
            },
            notes="production readiness 由 dashboard.readiness_overview.v1 统一判断；review/block 不得被误报为 ready。",
        )
    )

    knowledge_summary_path = repo_root / "research" / "knowledge_ingest" / "knowledge_ingest_summary.json"
    knowledge_summary = _load_json(knowledge_summary_path)
    knowledge_review_status = (
        "review"
        if int(knowledge_summary.get("promotion_audit_blocked_count", 0) or 0) > 0
        else _status_from_bool(knowledge_summary.get("promotion_evidence_source") == "research_card_scan")
    )
    requirements.append(
        _requirement(
            requirement_id="external_knowledge_promotion_backlog",
            category="knowledge",
            requirement="外部知识必须先进入 research/knowledge_ingest，并经过结构化假设、证据扫描和 promotion audit。",
            status=knowledge_review_status,
            evidence_paths=[_rel(repo_root, knowledge_summary_path), "research/knowledge_ingest/knowledge_promotion_audit.json"],
            metrics={
                "source_count": knowledge_summary.get("source_count", 0),
                "promotion_evidence_source": knowledge_summary.get("promotion_evidence_source", ""),
                "promotion_audit_ready_count": knowledge_summary.get("promotion_audit_ready_count", 0),
                "promotion_audit_blocked_count": knowledge_summary.get("promotion_audit_blocked_count", 0),
            },
            notes="blocked 候选保留在 research backlog，不能直接进入 src/production/skill。",
        )
    )

    skill_summary_path = repo_root / "research" / "skill_boundary_audit_v1" / "analysis_summary.json"
    skill_summary = _load_json(skill_summary_path)
    requirements.append(
        _requirement(
            requirement_id="skill_boundary_method_only",
            category="skill",
            requirement="skill 只保留方法论/发散思考，不承载项目状态、路径、结果或功能入口。",
            status=_status_from_bool(
                skill_summary.get("core_contract") == "core.skill_boundary_audit.v1"
                and skill_summary.get("method_only_status") == "pass"
                and int(skill_summary.get("project_state_blocked_issue_count", 0) or 0) >= 1
            ),
            evidence_paths=[_rel(repo_root, skill_summary_path), "src/core/skill_boundary.py"],
            metrics={
                "core_contract": skill_summary.get("core_contract", ""),
                "method_only_status": skill_summary.get("method_only_status", ""),
                "method_only_issue_count": skill_summary.get("method_only_issue_count", 0),
                "project_state_status": skill_summary.get("project_state_status", ""),
                "project_state_blocked_issue_count": skill_summary.get("project_state_blocked_issue_count", 0),
            },
            notes="该项证明已有 skill 边界审计能力和 research 证据，不表示每个本地 skill 都已迁移。",
        )
    )

    status_counts = {
        status: int(sum(item["status"] == status for item in requirements))
        for status in sorted({str(item["status"]) for item in requirements})
    }
    failed_count = int(status_counts.get("fail", 0))
    review_count = int(status_counts.get("review", 0))
    summary_status = "fail" if failed_count else "review" if review_count else "pass"
    summary = {
        "schema_version": PLATFORM_REQUIREMENT_AUDIT_SUMMARY_SCHEMA_VERSION,
        "status": summary_status,
        "requirement_count": int(len(requirements)),
        "passed_count": int(status_counts.get("pass", 0)),
        "review_count": review_count,
        "failed_count": failed_count,
        "status_counts": status_counts,
        "review_requirement_ids": [item["requirement_id"] for item in requirements if item["status"] == "review"],
        "failed_requirement_ids": [item["requirement_id"] for item in requirements if item["status"] == "fail"],
        "completion_claim": (
            "current_platform_requirements_satisfied" if failed_count == 0 and review_count == 0 else "not_claimed"
        ),
        "audit_note": "该 audit 用当前 repo 证据映射长期目标；有 fail/review 时不能声明目标完成。",
    }
    return _json_ready(
        {
            "schema_version": PLATFORM_REQUIREMENT_AUDIT_SCHEMA_VERSION,
            "summary": summary,
            "requirements": requirements,
        }
    )


def write_platform_requirement_audit(output_dir: str | Path, audit: Mapping[str, Any]) -> dict[str, str]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    payload = _json_ready(dict(audit))
    audit_path = output / "platform_requirement_audit.json"
    csv_path = output / "platform_requirement_audit.csv"
    review_path = output / "platform_requirement_audit_review.md"
    audit_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    fieldnames = [
        "requirement_id",
        "status",
        "category",
        "evidence_paths",
        "notes",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for item in payload.get("requirements", []):
            writer.writerow(
                {
                    "requirement_id": item.get("requirement_id", ""),
                    "status": item.get("status", ""),
                    "category": item.get("category", ""),
                    "evidence_paths": ";".join(str(path) for path in item.get("evidence_paths", [])),
                    "notes": item.get("notes", ""),
                }
            )

    summary = dict(payload.get("summary", {}))
    review_lines = [
        "# 平台需求覆盖 Audit",
        "",
        "## 中文结论",
        "",
        f"- 做了什么：把长期 SOXL 全平台目标拆成 {summary.get('requirement_count', 0)} 条可审计需求，并读取当前 repo 的 manifest、notebook、model ledger、production summary 和 knowledge ingest 证据。",
        f"- 核心指标：status={summary.get('status', '')}；pass={summary.get('passed_count', 0)}；review={summary.get('review_count', 0)}；fail={summary.get('failed_count', 0)}。",
        f"- regime 差异：本 audit 不重新计算行情 regime；它检查 `analysis_manifest` 是否已有 regime/window/model-difference/failure/objective-loss 证据面。",
        "- 是否晋升：否。本 audit 不是交易策略或模型晋升，只是完成度证据面；status=review 时不能声明长期目标完成。",
        "- 风险：该 audit 证明的是当前文件证据是否存在且口径一致，不能替代逐项人工 review 或未来数据更新后的重新验证。",
        f"- 下一步：优先处理 review 项：{', '.join(summary.get('review_requirement_ids', [])) or '无'}。",
    ]
    review_path.write_text("\n".join(review_lines) + "\n", encoding="utf-8")
    return {
        "schema_version": PLATFORM_REQUIREMENT_AUDIT_WRITE_RESULT_SCHEMA_VERSION,
        "audit_path": str(audit_path),
        "csv_path": str(csv_path),
        "review_path": str(review_path),
    }


def _registry_record(*, decision: Mapping[str, Any], decision_path: Path) -> dict[str, Any]:
    return {
        "decision_id": str(decision.get("decision_id", "")),
        "target_surface": str(decision.get("target_surface", "")),
        "change_type": str(decision.get("change_type", "")),
        "status": str(decision.get("status", "")),
        "decision_hash": str(decision.get("decision_hash", "")),
        "decision_path": str(decision_path),
        "evidence_notebook": str(decision.get("evidence_notebook", "")),
        "created_at": str(decision.get("created_at", "")),
    }


def _validate_decision_payload(decision: Mapping[str, Any]) -> None:
    _require_text(str(decision.get("decision_id", "")), "decision_id")
    _require_text(str(decision.get("target_surface", "")), "target_surface")
    _require_text(str(decision.get("change_type", "")), "change_type")
    _require_text(str(decision.get("reason", "")), "reason")
    _require_text(str(decision.get("evidence_notebook", "")), "evidence_notebook")
    if not _clean_items(decision.get("impact_scope", [])):
        raise ValueError("impact_scope is required")
    if not _clean_items(decision.get("verification_commands", [])):
        raise ValueError("verification_commands is required")


def _write_review(path: Path, *, decision: Mapping[str, Any], registry_path: Path) -> None:
    lines = [
        "# 架构扩展决策 Review",
        "",
        "## 中文结论",
        "",
        f"- 决策 ID：{decision.get('decision_id', '')}。",
        f"- 目标 surface：{decision.get('target_surface', '')}。",
        f"- 扩展原因：{decision.get('reason', '')}",
        f"- 影响范围：{', '.join(str(item) for item in decision.get('impact_scope', []))}。",
        f"- 证据 notebook：{decision.get('evidence_notebook', '')}。",
        f"- 验证命令：{'; '.join(str(item) for item in decision.get('verification_commands', []))}。",
        "- 是否晋升：否。本记录只说明稳定 src surface 的扩展理由，不代表策略或模型晋升。",
        "- 风险：没有 evidence notebook、聚焦验证和中文结论时，不应继续扩展稳定 src 契约。",
        f"- 下一步：在 registry `{registry_path}` 中追踪后续架构扩展记录。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_architecture_extension_decision(
    *,
    output_dir: str | Path,
    decision: Mapping[str, Any],
    registry_path: str | Path | None = None,
) -> dict[str, str]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    payload = _json_ready(dict(decision))
    _validate_decision_payload(payload)
    payload["decision_hash"] = architecture_extension_content_hash(payload)

    decision_id = _require_text(str(payload.get("decision_id", "")), "decision_id")
    decision_path = output / f"{decision_id}.json"
    decision_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    registry_file = Path(registry_path) if registry_path is not None else output / "architecture_extension_registry.json"
    registry = _load_registry(registry_file)
    decisions = registry.setdefault("decisions", {})
    decisions[decision_id] = _registry_record(decision=payload, decision_path=decision_path)
    registry["registry_hash"] = stable_hash(
        {
            "schema_version": registry["schema_version"],
            "decisions": decisions,
        }
    )
    registry_file.parent.mkdir(parents=True, exist_ok=True)
    registry_file.write_text(json.dumps(_json_ready(registry), ensure_ascii=False, indent=2), encoding="utf-8")

    review_path = output / "architecture_extension_review.md"
    _write_review(review_path, decision=payload, registry_path=registry_file)

    return {
        "schema_version": "core.architecture_extension_write_result.v1",
        "decision_path": str(decision_path),
        "registry_path": str(registry_file),
        "review_path": str(review_path),
        "decision_hash": str(payload["decision_hash"]),
        "registry_hash": str(registry["registry_hash"]),
    }
