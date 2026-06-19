from __future__ import annotations

import json

import pytest

from src.core import (
    architecture_extension_content_hash,
    build_architecture_extension_decision,
    build_platform_requirement_audit,
    write_architecture_extension_decision,
    write_platform_requirement_audit,
)


def test_build_architecture_extension_decision_requires_reason_evidence_and_verification():
    with pytest.raises(ValueError, match="reason"):
        build_architecture_extension_decision(
            decision_id="src-analysis-regime-v1",
            target_surface="src/analysis",
            change_type="new_surface",
            reason="",
            impact_scope=("src/analysis", "research/analysis_regime_window_diagnostics_v1"),
            evidence_notebook="research/analysis_regime_window_diagnostics_v1/analysis_regime_window_diagnostics_v1.ipynb",
            verification_commands=("pytest tests/test_analysis_qlib_extensions.py -q",),
        )

    with pytest.raises(ValueError, match="evidence_notebook"):
        build_architecture_extension_decision(
            decision_id="src-analysis-regime-v1",
            target_surface="src/analysis",
            change_type="new_surface",
            reason="需要稳定 regime/window 分析接口供 dashboard 和 promotion 复用。",
            impact_scope=("src/analysis",),
            evidence_notebook="",
            verification_commands=("pytest tests/test_analysis_qlib_extensions.py -q",),
        )

    with pytest.raises(ValueError, match="verification_commands"):
        build_architecture_extension_decision(
            decision_id="src-analysis-regime-v1",
            target_surface="src/analysis",
            change_type="new_surface",
            reason="需要稳定 regime/window 分析接口供 dashboard 和 promotion 复用。",
            impact_scope=("src/analysis",),
            evidence_notebook="research/analysis_regime_window_diagnostics_v1/analysis_regime_window_diagnostics_v1.ipynb",
            verification_commands=(),
        )


def test_build_architecture_extension_decision_has_recomputable_content_hash():
    payload = build_architecture_extension_decision(
        decision_id="src-analysis-regime-v1",
        target_surface="src/analysis",
        change_type="new_surface",
        reason="需要稳定 regime/window 分析接口供 dashboard 和 promotion 复用。",
        impact_scope=("src/analysis", "src/dashboard", "research/analysis_regime_window_diagnostics_v1"),
        evidence_notebook="research/analysis_regime_window_diagnostics_v1/analysis_regime_window_diagnostics_v1.ipynb",
        verification_commands=(
            "PYTHONPATH=. /Users/rziiiii/dev/workspace/qlib_venv/bin/python -m pytest tests/test_analysis_qlib_extensions.py -q",
        ),
        alternatives_considered=("保留在单次 notebook 中", "写入 skill 作为流程提示"),
        constraints=("skill 不能承载业务状态", "dashboard 只能读取 manifest/export"),
        created_at="2026-06-19T03:30:00Z",
    )

    tampered = dict(payload)
    tampered["reason"] = "changed after review"

    assert payload["schema_version"] == "core.architecture_extension_decision.v1"
    assert payload["decision_hash"] == architecture_extension_content_hash(payload)
    assert architecture_extension_content_hash(tampered) != payload["decision_hash"]
    assert payload["extension_requires_reason"] is True
    assert payload["skill_boundary"] == "method_only"
    assert payload["blocked_without_evidence"] is True


def test_write_architecture_extension_decision_writes_registry_and_chinese_review(tmp_path):
    decision = build_architecture_extension_decision(
        decision_id="src-core-extension-record-v1",
        target_surface="src/core",
        change_type="governance_contract",
        reason="需要把 src 稳定后继续扩展的原因和验证证据固化为可审阅契约。",
        impact_scope=("src/core", "research/architecture_extension_contract_v1"),
        evidence_notebook="research/architecture_extension_contract_v1/architecture_extension_contract_v1.ipynb",
        verification_commands=("PYTHONPATH=. /Users/rziiiii/dev/workspace/qlib_venv/bin/python -m pytest tests/test_core_architecture_decisions.py -q",),
        created_at="2026-06-19T03:30:00Z",
    )

    result = write_architecture_extension_decision(
        output_dir=tmp_path / "research" / "architecture_extension_contract_v1",
        decision=decision,
        registry_path=tmp_path / "research" / "architecture_extension_contract_v1" / "architecture_extension_registry.json",
    )

    decision_payload = json.loads((tmp_path / result["decision_path"]).read_text(encoding="utf-8"))
    registry = json.loads((tmp_path / result["registry_path"]).read_text(encoding="utf-8"))
    review = (tmp_path / result["review_path"]).read_text(encoding="utf-8")

    assert decision_payload["decision_hash"] == result["decision_hash"]
    assert registry["schema_version"] == "core.architecture_extension_registry.v1"
    assert registry["decisions"]["src-core-extension-record-v1"]["decision_hash"] == result["decision_hash"]
    assert len(registry["registry_hash"]) == 64
    assert "中文结论" in review
    assert "扩展原因" in review
    assert "src-core-extension-record-v1" in review


def test_write_architecture_extension_decision_rejects_unvalidated_payload(tmp_path):
    decision = {
        "schema_version": "core.architecture_extension_decision.v1",
        "decision_id": "bad-extension",
        "target_surface": "src/core",
        "change_type": "governance_contract",
        "reason": "",
        "impact_scope": ["src/core"],
        "evidence_notebook": "research/example/example.ipynb",
        "verification_commands": ["pytest tests/test_core_architecture_decisions.py -q"],
    }

    with pytest.raises(ValueError, match="reason"):
        write_architecture_extension_decision(output_dir=tmp_path, decision=decision)


def test_build_platform_requirement_audit_maps_goal_to_current_repo_evidence():
    audit = build_platform_requirement_audit(".")
    requirements = {item["requirement_id"]: item for item in audit["requirements"]}

    assert audit["schema_version"] == "core.platform_requirement_audit.v1"
    assert audit["summary"]["status"] == "review"
    assert audit["summary"]["requirement_count"] >= 10
    assert audit["summary"]["failed_count"] == 0
    assert audit["summary"]["review_count"] >= 1

    assert requirements["stable_src_surfaces"]["status"] == "pass"
    assert requirements["reward_retired_objective_surface"]["status"] == "pass"
    assert requirements["research_production_notebook_contract"]["status"] == "pass"
    assert requirements["quantbench_requested_model_registry"]["status"] == "pass"
    assert requirements["quantbench_training_backtest_ledger"]["status"] == "pass"
    assert requirements["production_paper_api_guardrail"]["status"] == "pass"
    assert requirements["external_knowledge_promotion_backlog"]["status"] == "pass"
    assert requirements["production_readiness_review_gate"]["status"] == "review"

    model_registry = requirements["quantbench_requested_model_registry"]
    model_ledger = requirements["quantbench_training_backtest_ledger"]
    analysis = requirements["analysis_regime_window_model_difference"]
    knowledge = requirements["external_knowledge_promotion_backlog"]

    assert model_registry["metrics"]["requested_model_count"] == 32
    assert model_registry["metrics"]["registered_model_count"] == 32
    assert model_ledger["metrics"]["expected_model_count"] == 96
    assert model_ledger["metrics"]["trained_model_count"] == 96
    assert model_ledger["metrics"]["backtested_model_count"] == 96
    assert model_ledger["metrics"]["all_models_completed"] is True
    assert knowledge["metrics"]["promotion_audit_ready_count"] == 10
    assert knowledge["metrics"]["promotion_audit_blocked_count"] == 0
    assert {"regime_metrics", "window_metrics", "model_comparison", "objective_losses"}.issubset(
        set(analysis["metrics"]["analysis_paths"])
    )
    assert "production/paper_sim_replay_v1/matrix_card/model_run_ledger_summary.json" in model_ledger["evidence_paths"]


def test_write_platform_requirement_audit_writes_json_csv_and_chinese_review(tmp_path):
    audit = build_platform_requirement_audit(".")

    result = write_platform_requirement_audit(tmp_path, audit)

    payload = json.loads((tmp_path / "platform_requirement_audit.json").read_text(encoding="utf-8"))
    csv_text = (tmp_path / "platform_requirement_audit.csv").read_text(encoding="utf-8")
    review = (tmp_path / "platform_requirement_audit_review.md").read_text(encoding="utf-8")

    assert result["schema_version"] == "core.platform_requirement_audit_write_result.v1"
    assert result["audit_path"].endswith("platform_requirement_audit.json")
    assert payload["summary"]["status"] == audit["summary"]["status"]
    assert "requirement_id,status,category" in csv_text
    assert "中文结论" in review
    assert "是否晋升：否" in review


def test_platform_requirement_audit_notebook_records_current_review_status():
    card = "research/platform_requirement_audit_v1"
    notebook_path = f"{card}/platform_requirement_audit_v1.ipynb"
    summary = json.loads(open(f"{card}/platform_requirement_audit_summary.json", encoding="utf-8").read())
    audit = json.loads(open(f"{card}/platform_requirement_audit.json", encoding="utf-8").read())
    notebook = json.loads(open(notebook_path, encoding="utf-8").read())
    conclusion = "".join(notebook["cells"][-1]["source"])

    assert summary["schema_version"] == "research.platform_requirement_audit_v1"
    assert summary["core_schema_version"] == "core.platform_requirement_audit.v1"
    assert summary["status"] == "review"
    assert summary["failed_count"] == 0
    assert summary["review_count"] == 1
    assert summary["completion_claim"] == "not_claimed"
    assert summary["review_requirement_ids"] == [
        "production_readiness_review_gate",
    ]
    assert audit["summary"]["status"] == summary["status"]
    assert "中文结论" in conclusion
    assert "是否晋升" in conclusion
