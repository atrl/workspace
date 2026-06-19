from __future__ import annotations

import json

import pytest

from src.core import (
    architecture_extension_content_hash,
    build_architecture_extension_decision,
    write_architecture_extension_decision,
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
