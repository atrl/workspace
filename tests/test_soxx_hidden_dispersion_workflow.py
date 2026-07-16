from __future__ import annotations

import json

import pytest


def _summary(tmp_path):
    return {
        "experiment": "soxx_hidden_dispersion_v1",
        "card_type": "diagnostic_soxx_hidden_dispersion",
        "status": "complete",
        "promotion_decision": "not_promoted_missing_pit_holdings_and_no_incremental_evidence",
        "as_of": "2026-07-02",
        "provider_uri": (tmp_path / "data" / "qlib" / "soxl").as_posix(),
        "recorder_uri": (tmp_path / "research" / "card" / "qlib_recorder").as_posix(),
        "recorder_id": "recorder-test",
        "segments": {
            "train": ["2019-01-02", "2024-12-31"],
            "validation": ["2025-01-02", "2025-12-31"],
            "test": ["2026-01-02", "2026-07-02"],
        },
        "constituent_data": {
            "provider_uri": (tmp_path / "data" / "qlib" / "soxx_constituents").as_posix(),
            "instrument_count": 30,
            "point_in_time_status": "blocked_current_holdings_backfill",
        },
        "candidate_diagnostic": {
            "baseline_test_soxx_rank_correlation": 0.10,
            "combined_test_soxx_rank_correlation": 0.15,
            "incremental_test_soxx_rank_correlation": 0.05,
            "combined_test_soxx_abs_rank_correlation": 0.20,
            "nonoverlap_positive_offset_ratio": 0.8,
            "nonoverlap_finite_offset_count": 5,
            "stable_candidate_count": 1,
        },
        "qlib_analyzer": {
            "api": "qlib.contrib.evaluate.risk_analysis",
            "sidecar": "qlib_risk_analysis.csv",
        },
        "artifact_policy": {
            "policy": "slim_card_recorder_first",
            "workflow_config": "workflow.yaml",
            "recorder_only_artifacts": ["daily_hidden_dispersion_panel.csv"],
        },
        "sidecars": {
            "feature_metrics": "feature_metrics.csv",
            "nonoverlap_feature_metrics": "nonoverlap_feature_metrics.csv",
            "model_comparison": "model_comparison.csv",
            "qlib_risk_analysis": "qlib_risk_analysis.csv",
            "risk_analysis": "risk_analysis.csv",
        },
    }


def test_hidden_dispersion_notebook_is_executable_qlib_workflow(tmp_path):
    from src.workflow.soxx_hidden_dispersion import _write_notebook

    summary = _summary(tmp_path)
    _write_notebook(tmp_path, summary)

    notebook_path = tmp_path / "soxx_hidden_dispersion_v1.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    notebook_text = notebook_path.read_text(encoding="utf-8")
    assert notebook["nbformat"] == 4
    assert "import qlib" in notebook_text
    assert "qlib.init" in notebook_text
    assert "D.features" in notebook_text
    assert "R.start" in notebook_text
    assert "run_soxx_hidden_dispersion_card" in notebook_text
    assert "结论与反思" in notebook_text


def test_hidden_dispersion_workflow_fails_closed_without_qlib_manifests(tmp_path):
    from src.workflow.soxx_hidden_dispersion import SoxxHiddenDispersionSpec, _run_card

    spec = SoxxHiddenDispersionSpec(root=tmp_path)
    with pytest.raises(FileNotFoundError, match="qlib store manifest missing"):
        _run_card(spec)


def test_hidden_dispersion_summary_keeps_recorder_only_panel_and_pit_gate(tmp_path):
    from src.workflow.soxx_hidden_dispersion import _risk_analysis

    summary = _summary(tmp_path)
    risk = _risk_analysis(summary)

    assert summary["artifact_policy"]["policy"] == "slim_card_recorder_first"
    assert summary["artifact_policy"]["recorder_only_artifacts"] == [
        "daily_hidden_dispersion_panel.csv"
    ]
    assert summary["promotion_decision"] == "not_promoted_missing_pit_holdings_and_no_incremental_evidence"
    pit = risk.loc[risk["check"].eq("point_in_time_holdings")].iloc[0]
    assert pit["status"] == "fail"


def test_hidden_dispersion_offset_gate_requires_all_five_finite_offsets(tmp_path):
    from src.workflow.soxx_hidden_dispersion import _risk_analysis

    summary = _summary(tmp_path)
    summary["candidate_diagnostic"]["nonoverlap_positive_offset_ratio"] = 1.0
    summary["candidate_diagnostic"]["nonoverlap_finite_offset_count"] = 1

    risk = _risk_analysis(summary)

    offset = risk.loc[risk["check"].eq("nonoverlap_offset_stability")].iloc[0]
    assert offset["status"] == "fail"
