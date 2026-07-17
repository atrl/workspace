from __future__ import annotations

import json
from pathlib import Path

import nbformat
import pandas as pd
import pytest


def _summary(
    route: str,
    *,
    recorder_id: str = "rid",
    status: str = "ok",
    selected_candidate: str = "candidate",
    start: str = "2024-01-02",
    end: str = "2026-07-02",
    promoted: bool = False,
    recorder_local_dir: str = "/tmp/missing-recorder",
) -> dict:
    from src.workflow.soxl_risk_adjusted_frontier import REQUIRED_PROMOTION_CHECKS

    return {
        "experiment": route,
        "route": route,
        "route_status": status,
        "selected_candidate": selected_candidate,
        "selection_frozen_before_test": True,
        "test_labels_used_for_selection": False,
        "data_range": [start, end],
        "segments": {
            "train": ["2024-01-02", "2025-03-31"],
            "valid": ["2025-04-01", "2025-09-30"],
            "test": ["2025-10-01", end],
        },
        "label_contract": {"horizon": 5, "purge_rows": 6},
        "benchmark": {"name": "legacy_soxl_backtest", "matched_window": True},
        "recorder_id": recorder_id,
        "recorder_uri": "/tmp/mlruns",
        "recorder_local_dir": recorder_local_dir,
        "recorder_artifacts": [
            "artifacts/pred.pkl",
            "artifacts/label.pkl",
            "artifacts/model_selection/all_model_predictions.csv",
            "artifacts/portfolio_analysis/report_normal_1day.pkl",
            "artifacts/portfolio_analysis/positions_normal_1day.pkl",
            "artifacts/portfolio_analysis/port_analysis_1day.pkl",
        ],
        "promotion": {
            "promoted": promoted,
            "decision": "promoted_risk_adjusted_candidate" if promoted else "not_promoted_failed_risk_gates",
            "checks": {key: promoted for key in REQUIRED_PROMOTION_CHECKS},
            "failed_checks": [] if promoted else list(REQUIRED_PROMOTION_CHECKS),
        },
    }


def _recorder_dir(tmp_path: Path, route: str) -> Path:
    from src.workflow.soxl_risk_adjusted_frontier import RECORDER_ONLY_ARTIFACTS

    root = tmp_path / route
    for relative in RECORDER_ONLY_ARTIFACTS.values():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"artifact")
    return root


def test_frontier_compares_frozen_route_results_without_test_reselection(tmp_path: Path):
    from src.workflow.soxl_risk_adjusted_frontier import compare_frozen_route_summaries

    summaries = {
        name: _summary(
            name,
            recorder_id=f"rid-{name}",
            recorder_local_dir=_recorder_dir(tmp_path, name).as_posix(),
        )
        for name in ("tail_budget", "regime_moe", "panel_sequence")
    }
    summaries["panel_sequence"]["route_status"] = "blocked_missing_required_source"
    summaries["panel_sequence"]["selected_candidate"] = "blocked_missing_required_source"

    result = compare_frozen_route_summaries(summaries)

    assert result["decision"] == "not_promoted_no_risk_adjusted_candidate"
    assert result["selection_source"] == "frozen_route_summaries"
    assert result["test_reselection_performed"] is False
    assert set(result["routes"]) == set(summaries)
    assert result["routes"]["panel_sequence"]["route_status"].startswith("blocked_")


def test_frontier_fails_closed_on_window_or_recorder_mismatch(tmp_path: Path):
    from src.workflow.soxl_risk_adjusted_frontier import compare_frozen_route_summaries

    summaries = {
        name: _summary(
            name,
            recorder_id=f"rid-{name}",
            recorder_local_dir=_recorder_dir(tmp_path, name).as_posix(),
        )
        for name in ("tail_budget", "regime_moe", "panel_sequence")
    }
    summaries["regime_moe"]["data_range"] = ["2024-02-01", "2026-07-02"]
    with pytest.raises(ValueError, match="matched data windows"):
        compare_frozen_route_summaries(summaries)

    summaries["regime_moe"] = _summary("regime_moe", recorder_id="")
    with pytest.raises(ValueError, match="recorder evidence"):
        compare_frozen_route_summaries(summaries)


def test_frontier_revalidates_literal_promotion_checks_and_artifact_access(tmp_path: Path):
    from src.workflow.soxl_risk_adjusted_frontier import (
        REQUIRED_PROMOTION_CHECKS,
        compare_frozen_route_summaries,
    )

    summaries = {
        name: _summary(
            name,
            recorder_id=f"rid-{name}",
            promoted=True,
            recorder_local_dir=_recorder_dir(tmp_path, name).as_posix(),
        )
        for name in ("tail_budget", "regime_moe", "panel_sequence")
    }
    summaries["tail_budget"]["promotion"]["checks"].pop(next(iter(REQUIRED_PROMOTION_CHECKS)))
    with pytest.raises(ValueError, match="promotion check evidence"):
        compare_frozen_route_summaries(summaries)

    summaries["tail_budget"] = _summary(
        "tail_budget",
        recorder_id="rid-tail",
        promoted=True,
        recorder_local_dir=_recorder_dir(tmp_path, "tail-fixed").as_posix(),
    )
    missing = Path(summaries["regime_moe"]["recorder_local_dir"]) / "artifacts/pred.pkl"
    missing.unlink()
    with pytest.raises(ValueError, match="recorder artifact is not accessible"):
        compare_frozen_route_summaries(summaries)


def test_card_payload_uses_relative_paths_and_slim_recorder_policy(tmp_path: Path):
    from src.workflow.soxl_risk_adjusted_frontier import build_card_summary

    payload = build_card_summary(
        experiment="soxl_tail_budget_v1",
        route="tail_budget",
        provider_uri=Path("data/qlib/soxl"),
        recorder_uri=Path(".qlib_recorder/soxl_risk_adjusted_frontier"),
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
    assert payload["recorder_uri"] == ".qlib_recorder/soxl_risk_adjusted_frontier"
    assert policy["policy"] == "slim_card_recorder_first"
    assert policy["workflow_config"] == "workflow.yaml"
    assert set(policy["recorder_only_artifacts"].values()) == {
        "artifacts/pred.pkl",
        "artifacts/label.pkl",
        "artifacts/model_selection/all_model_predictions.csv",
        "artifacts/portfolio_analysis/report_normal_1day.pkl",
        "artifacts/portfolio_analysis/positions_normal_1day.pkl",
        "artifacts/portfolio_analysis/port_analysis_1day.pkl",
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
        last = notebook_node.cells[-1]
        assert last.cell_type == "markdown"
        assert summary["recorder_id"] in last.source
        for heading in ("做了什么", "核心指标", "Regime / window 差异", "晋升决定", "风险", "下一步"):
            assert heading in last.source


def test_route_cards_use_native_qlib_record_outputs_and_frontier_has_own_policy():
    root = Path(__file__).resolve().parents[1]
    for name in ("soxl_tail_budget_v1", "soxl_regime_moe_v1", "soxl_panel_sequence_v1"):
        summary = json.loads((root / "research" / name / "summary.json").read_text(encoding="utf-8"))
        assert summary["artifact_generation_api"] == ["SignalRecord.generate", "PortAnaRecord.generate"]
        assert summary["metrics_source"] == "qlib_portfolio_analysis/report_normal_1day.pkl"
        assert set(summary["artifact_policy"]["recorder_only_artifacts"].values()).issubset(
            set(summary["recorder_artifacts"])
        )
        recorder = Path(summary["recorder_local_dir"]) / "artifacts" / "portfolio_analysis"
        report = pd.read_pickle(recorder / "report_normal_1day.pkl")
        positions = pd.read_pickle(recorder / "positions_normal_1day.pkl")
        risk = pd.read_pickle(recorder / "port_analysis_1day.pkl")
        assert {"account", "return", "turnover", "cost", "value", "cash", "bench"}.issubset(
            report.columns
        )
        assert isinstance(positions, dict) and positions
        assert isinstance(risk, pd.DataFrame) and not risk.empty

    card = root / "research" / "soxl_risk_adjusted_model_frontier_v1"
    summary = json.loads((card / "summary.json").read_text(encoding="utf-8"))
    assert summary["canonical_dataset_audit"]["DatasetH_prepare_rows"] > 0
    assert summary["canonical_dataset_audit"]["training_performed"] is False
    assert summary["artifact_policy"]["recorder_only_artifacts"] == {
        "canonical_dataset_audit": "artifacts/frontier/canonical_dataset_audit.pkl",
        "route_summaries": "artifacts/frontier/route_summaries.pkl",
    }
    assert set(summary["artifact_policy"]["recorder_only_artifacts"].values()).issubset(
        set(summary["recorder_artifacts"])
    )
    workflow = (card / "workflow.yaml").read_text(encoding="utf-8")
    assert "market: all" in workflow
    assert "region: us" in workflow
    assert "segments:" in workflow
