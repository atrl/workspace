from __future__ import annotations

import json
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]


def _actual_summaries() -> dict[str, dict]:
    cards = {
        "tail_budget": "soxl_tail_budget_v1",
        "regime_moe": "soxl_regime_moe_v1",
        "panel_sequence": "soxl_panel_sequence_v1",
    }
    return {
        route: json.loads((ROOT / "research" / name / "summary.json").read_text(encoding="utf-8"))
        for route, name in cards.items()
    }


def test_frontier_compares_frozen_route_results_without_test_reselection():
    from src.workflow.soxl_risk_adjusted_frontier import compare_frozen_route_summaries

    result = compare_frozen_route_summaries(_actual_summaries())

    assert result["decision"] == "not_promoted_no_risk_adjusted_candidate"
    assert result["selection_source"] == "recorder_run_manifests"
    assert result["test_reselection_performed"] is False
    assert set(result["routes"]) == set(_actual_summaries())
    assert result["routes"]["panel_sequence"]["route_status"] == "ok"
    assert result["evidence_source"] == "qlib_recorder_objects"


def test_frontier_fails_closed_on_window_or_recorder_mismatch():
    from src.workflow.soxl_risk_adjusted_frontier import compare_frozen_route_summaries

    summaries = _actual_summaries()
    summaries["regime_moe"]["data_range"] = ["2024-02-01", "2026-07-02"]
    with pytest.raises(ValueError, match="summary and recorder manifest differ"):
        compare_frozen_route_summaries(summaries)

    summaries = _actual_summaries()
    summaries["regime_moe"]["recorder_id"] = "stale-recorder-id"
    with pytest.raises((ValueError, RuntimeError), match="recorder"):
        compare_frozen_route_summaries(summaries)


def test_frontier_rejects_tampered_summary_against_recorder_truth():
    from src.workflow.soxl_risk_adjusted_frontier import compare_frozen_route_summaries

    summaries = _actual_summaries()
    summaries["tail_budget"]["metrics"]["test"]["candidate"]["total_return"] += 10.0
    with pytest.raises(ValueError, match="summary and recorder metrics differ"):
        compare_frozen_route_summaries(summaries)


@pytest.mark.parametrize(
    ("field", "mutate"),
    [
        ("data_lineage", lambda summary: summary["data_lineage"].update(manifest_hash="tampered")),
        ("segments", lambda summary: summary["segments"]["test"].__setitem__(0, "2025-01-02")),
        ("selected_candidate", lambda summary: summary.__setitem__("selected_candidate", "tampered")),
        ("promotion", lambda summary: summary["promotion"].__setitem__("promoted", True)),
    ],
)
def test_frontier_rejects_tampered_summary_manifest_mirrors(field, mutate):
    from src.workflow.soxl_risk_adjusted_frontier import compare_frozen_route_summaries

    summaries = _actual_summaries()
    mutate(summaries["tail_budget"])
    with pytest.raises(ValueError, match="summary and recorder manifest differ"):
        compare_frozen_route_summaries(summaries)


def test_card_payload_uses_relative_paths_and_slim_recorder_policy(tmp_path: Path):
    from src.workflow.soxl_risk_adjusted_frontier import build_card_summary

    payload = build_card_summary(
        experiment="soxl_tail_budget_v1",
        route="tail_budget",
        provider_uri=Path("data/qlib/soxl"),
        recorder_uri=Path("git-common-parent:.qlib-recorders/soxl-risk-adjusted-frontier-v1"),
        recorder_id="rid-tail",
        recorder_local_dir=tmp_path / "mlruns" / "run",
        data_range=("2024-01-02", "2026-07-02"),
        segments={
            "train": ("2024-01-02", "2025-03-31"),
            "valid": ("2025-04-01", "2025-09-30"),
            "test": ("2025-10-01", "2026-07-02"),
        },
        route_payload={
            "route_status": "ok",
            "selected_candidate": "ridge_expected",
            "selection_frozen_before_test": True,
            "test_labels_used_for_selection": False,
        },
        sidecars={"model_selection": "model_selection.csv"},
    )

    policy = payload["artifact_policy"]
    assert payload["provider_uri"] == "data/qlib/soxl"
    assert payload["recorder_uri"].startswith("git-common-parent:")
    assert "recorder_local_dir" not in payload
    assert policy["policy"] == "slim_card_recorder_first"
    assert policy["workflow_config"] == "workflow.yaml"
    assert set(policy["recorder_only_artifacts"].values()) == {
        "artifacts/pred.pkl",
        "artifacts/label.pkl",
        "artifacts/model_selection/all_model_predictions.csv",
        "artifacts/portfolio_analysis/report_normal_1day.pkl",
        "artifacts/portfolio_analysis/positions_normal_1day.pkl",
        "artifacts/portfolio_analysis/port_analysis_1day.pkl",
        "artifacts/evidence/legacy_report.pkl",
        "artifacts/evidence/run_manifest.pkl",
    }
    assert all(not Path(path).is_absolute() for path in payload["sidecars"].values())
    json.dumps(payload, allow_nan=False)


def test_generated_risk_adjusted_cards_are_executable_slim_and_chinese():
    root = Path(__file__).resolve().parents[1]
    names = (
        "soxl_tail_budget_v1",
        "soxl_regime_moe_v1",
        "soxl_panel_sequence_v1",
        "soxl_risk_adjusted_model_frontier_v1",
    )
    forbidden = {
        "predictions.csv",
        "signals.csv",
        "all_model_predictions.csv",
        "portfolio_report.csv",
        "qlib_portfolio_report.csv",
        "qlib_positions.csv",
    }
    for name in names:
        card = root / "research" / name
        summary = json.loads((card / "summary.json").read_text(encoding="utf-8"))
        notebook = (card / f"{name}.ipynb").read_text(encoding="utf-8")
        conclusion = (card / "conclusion.md").read_text(encoding="utf-8")

        assert summary["as_of"] == "2026-07-02"
        assert summary["artifact_policy"]["policy"] == "slim_card_recorder_first"
        assert summary["selection_frozen_before_test"] is True
        assert summary["test_labels_used_for_selection"] is False
        assert {path.name for path in card.iterdir()}.isdisjoint(forbidden)
        assert "qlib.init" in notebook
        assert "DatasetH" in notebook
        assert "DataHandlerLP" in notebook
        assert "R.start" in notebook
        assert "做了什么" in conclusion
        assert "核心指标" in conclusion
        assert "晋升决定" in conclusion
        assert "风险" in conclusion
        assert "下一步" in conclusion
        notebook_node = nbformat.read(card / f"{name}.ipynb", as_version=4)
        code_source = "\n".join(
            cell.source for cell in notebook_node.cells if cell.cell_type == "code"
        )
        assert "sync_notebook_conclusion" in code_source
        last = notebook_node.cells[-1]
        assert last.cell_type == "markdown"
        assert summary["recorder_id"] in last.source
        for heading in ("做了什么", "核心指标", "Regime / window 差异", "晋升决定", "风险", "下一步"):
            assert heading in last.source
        serialized_outputs = json.dumps(
            [cell.get("outputs", []) for cell in notebook_node.cells], ensure_ascii=False
        )
        for forbidden_output in ("Traceback", "MlflowException", "background exception"):
            assert forbidden_output not in serialized_outputs


def test_route_cards_use_native_qlib_record_outputs_and_frontier_has_own_policy():
    root = Path(__file__).resolve().parents[1]
    for name in ("soxl_tail_budget_v1", "soxl_regime_moe_v1", "soxl_panel_sequence_v1"):
        summary = json.loads((root / "research" / name / "summary.json").read_text(encoding="utf-8"))
        assert summary["artifact_generation_api"] == ["SignalRecord.generate", "PortAnaRecord.generate"]
        assert summary["metrics_source"] == "qlib_portfolio_analysis/report_normal_1day.pkl"
        assert set(summary["artifact_policy"]["recorder_only_artifacts"].values()).issubset(
            set(summary["recorder_artifacts"])
        )
        from src.workflow.soxl_risk_adjusted_frontier import load_route_recorder_evidence

        evidence = load_route_recorder_evidence(summary)
        manifest = evidence["run_manifest"]
        report = evidence["report"]
        positions = evidence["positions"]
        risk = evidence["risk"]
        legacy_report = evidence["legacy_report"]
        assert {"account", "return", "turnover", "cost", "value", "cash", "bench"}.issubset(
            report.columns
        )
        assert isinstance(positions, dict) and positions
        assert isinstance(risk, pd.DataFrame) and not risk.empty
        assert {"account", "return", "turnover", "cost", "value", "cash", "bench"}.issubset(
            legacy_report.columns
        )
        assert summary["metrics"]["test"]["candidate"]["total_turnover"] == pytest.approx(
            float(report["turnover"].sum()), abs=1e-12
        )
        assert summary["metrics"]["test"]["legacy"]["total_turnover"] == pytest.approx(
            float(legacy_report["turnover"].sum()), abs=1e-12
        )
        assert summary["execution_config"] == {
            "cost_rate": 0.0004,
            "deal_price": "open",
            "timing": "signal_after_close_t_to_next_open",
        }
        assert manifest["schema_version"] == 1
        assert manifest["manifest_id"] == summary["run_manifest_id"]
        assert manifest["recorder_id"] == summary["recorder_id"]
        assert manifest["provider_manifest_hash"] == summary["data_lineage"]["manifest_hash"]
        assert manifest["data_lineage"] == summary["data_lineage"]
        assert manifest["effective_segments"] == summary["segments"]
        assert manifest["data_range"] == summary["data_range"]
        assert manifest["label_contract"] == summary["label_contract"]
        assert manifest["execution_config"] == summary["execution_config"]
        assert manifest["selected_candidate"] == summary["selected_candidate"]
        assert len(manifest["frozen_policy_id"]) == 64
        assert manifest["metrics"] == summary["metrics"]
        assert manifest["promotion"] == summary["promotion"]

    card = root / "research" / "soxl_risk_adjusted_model_frontier_v1"
    summary = json.loads((card / "summary.json").read_text(encoding="utf-8"))
    assert "route_manifests" not in summary["frontier"]
    assert summary["canonical_dataset_audit"]["DatasetH_prepare_rows"] > 0
    assert summary["canonical_dataset_audit"]["training_performed"] is False
    assert summary["artifact_policy"]["recorder_only_artifacts"] == {
        "canonical_dataset_audit": "artifacts/frontier/canonical_dataset_audit.pkl",
        "route_manifests": "artifacts/frontier/route_manifests.pkl",
    }
    assert set(summary["artifact_policy"]["recorder_only_artifacts"].values()).issubset(
        set(summary["recorder_artifacts"])
    )
    workflow = (card / "workflow.yaml").read_text(encoding="utf-8")
    assert "market: all" in workflow
    assert "region: us" in workflow
    assert "segments:" in workflow
    comparison = pd.read_csv(card / "backtest_benchmark_comparison.csv")
    assert {"candidate_return", "legacy_return", "return_gap"}.issubset(comparison.columns)


def test_internal_missing_dates_fail_data_and_label_evidence():
    from src.workflow.soxl_risk_adjusted_frontier import build_calendar_alignment_evidence

    index = pd.bdate_range("2026-01-02", periods=12)
    features = pd.DataFrame({"x": 1.0}, index=index)
    labels = pd.DataFrame({"future_return_5d": 0.1}, index=index)
    features.loc[index[4], "x"] = np.nan
    labels.loc[index[7], "future_return_5d"] = np.nan
    evidence = build_calendar_alignment_evidence(index, features, labels)
    assert evidence["calendar_rows_complete"] is True
    assert evidence["data_quality_valid"] is False
    assert evidence["labels_complete"] is False
    assert evidence["internal_feature_missing_dates"] == [index[4].date().isoformat()]
    assert evidence["internal_label_missing_dates"] == [index[7].date().isoformat()]


def test_notebook_conclusion_generator_tracks_latest_summary(tmp_path: Path):
    from src.workflow.soxl_risk_adjusted_frontier import sync_notebook_conclusion

    notebook = nbformat.v4.new_notebook(
        cells=[nbformat.v4.new_code_cell("x = 1"), nbformat.v4.new_markdown_cell("stale")]
    )
    path = tmp_path / "card.ipynb"
    nbformat.write(notebook, path)
    summary = _actual_summaries()["tail_budget"]
    sync_notebook_conclusion(path, summary)
    synced = nbformat.read(path, as_version=4)
    assert synced.cells[-1].cell_type == "markdown"
    assert summary["recorder_id"] in synced.cells[-1].source
    assert "核心指标" in synced.cells[-1].source
