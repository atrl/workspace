from __future__ import annotations

from src.core import build_skill_boundary_audit


def test_skill_boundary_audit_allows_method_only_research_guidance():
    skill_text = """
    # Regime Review Method

    Use this skill to brainstorm candidate market regimes, ask whether factor
    evidence should be split by volatility and drawdown, and remind the model to
    turn any useful idea into a research notebook before implementation.

    Keep concrete data paths, model results, fixed parameters, and project state
    outside the skill.
    """

    audit = build_skill_boundary_audit("regime-method", skill_text, source_path=".codex/skills/regime/SKILL.md")

    assert audit["schema_version"] == "core.skill_boundary_audit.v1"
    assert audit["status"] == "pass"
    assert audit["method_only"] is True
    assert audit["issue_count"] == 0
    assert audit["blocked_issue_count"] == 0
    assert audit["migration_required"] is False
    assert audit["allowed_skill_role"] == "method_prompt"


def test_skill_boundary_audit_blocks_project_state_paths_commands_and_results():
    skill_text = """
    # SOXL Qlib Experiment Workflow

    Put each experiment under runtime/model_selection/card/rolling_selector_v1/.
    Import framework APIs from soxl_backtest.framework and run:

    PYTHONPATH=. qlib_venv/bin/python -m pytest tests/test_soxlbacktest_framework.py -q

    Current best model: ridge_mse sharpe=1.42 max_drawdown=12.5%.
    """

    audit = build_skill_boundary_audit(
        "soxl-qlib-experiment",
        skill_text,
        source_path=".codex/skills/soxl-qlib-experiment/SKILL.md",
    )

    assert audit["status"] == "block"
    assert audit["method_only"] is False
    assert audit["migration_required"] is True
    assert audit["blocked_issue_count"] >= 3
    assert audit["recommended_destination"] == "research_notebook_or_src_core_manifest"
    assert audit["blocked_surfaces_before_migration"] == ["skill"]
    issue_types = {issue["issue_type"] for issue in audit["issues"]}
    assert {
        "concrete_project_path",
        "executable_command",
        "business_logic_entrypoint",
        "project_state_result",
    }.issubset(issue_types)
    assert all("recommendation" in issue and issue["recommendation"] for issue in audit["issues"])
