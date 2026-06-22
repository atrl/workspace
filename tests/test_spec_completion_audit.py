from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CARD = ROOT / "research" / "spec_completion_audit_v1"


def test_spec_completion_audit_artifact_proves_all_required_contracts():
    summary_path = CARD / "summary.json"
    matrix_path = CARD / "spec_completion_matrix.csv"
    notebook_path = CARD / "spec_completion_audit_v1.ipynb"
    conclusion_path = CARD / "conclusion.md"

    assert notebook_path.exists(), "completion audit must be an executable notebook artifact"
    assert summary_path.exists(), "completion audit must write machine-readable summary.json"
    assert matrix_path.exists(), "completion audit must write requirement-level matrix"
    assert conclusion_path.exists(), "completion audit must write a Chinese conclusion"

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = list(csv.DictReader(matrix_path.open(encoding="utf-8")))
    statuses = {row["requirement_id"]: row for row in rows}

    expected_requirements = {
        "objective.spec_all_content",
        "objective.qlib_framework_verified",
        "objective.soxlbacktest_migrated_reference",
        "objective.model_logic_complete",
        "objective.alpha158_plus_legacy_features",
        "objective.spec_models_beat_soxlbacktest",
        "spec.qlib_native_api",
        "spec.qlib_provider_dataset_recorder_report_traceability",
        "spec.fail_closed_no_pandas_fallback",
        "spec.analysis_outputs",
        "spec.dashboard_schedule_api_contracts",
        "spec.notebooks_executed_chinese_conclusions",
        "spec.no_skill_or_scattered_csv_primary_workflow",
    }

    assert summary["status"] == "complete"
    assert summary["open_gap_count"] == 0
    assert set(summary["requirement_ids"]) == expected_requirements
    assert set(statuses) == expected_requirements
    assert {row["status"] for row in rows} == {"proven"}
    assert all(row["evidence"] for row in rows)
    assert all(row["verification"] for row in rows)
    assert "完成" in conclusion_path.read_text(encoding="utf-8")
