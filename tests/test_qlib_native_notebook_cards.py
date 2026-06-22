from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _summary(card: Path) -> dict:
    return json.loads((card / "summary.json").read_text(encoding="utf-8"))


def test_research_card_has_executed_qlib_recorder_and_analysis_sidecars():
    card = ROOT / "research" / "qlib_native_reference_v1"
    summary = _summary(card)
    analysis = json.loads((card / "analysis_summary.json").read_text(encoding="utf-8"))
    notebook_text = (card / "qlib_native_reference_v1.ipynb").read_text(encoding="utf-8")

    assert summary["experiment"] == "qlib_native_reference_v1"
    assert "qlib.workflow.R.start" in summary["qlib_native_api"]
    assert "SignalRecord" in summary["qlib_native_api"]
    assert "SigAnaRecord" in summary["qlib_native_api"]
    assert analysis["source"] == "qlib_recorder"
    assert analysis["recorder_id"] == summary["recorder_id"]
    assert set(summary["sidecars"]) >= {
        "predictions",
        "signals",
        "portfolio_report",
        "analysis_summary",
        "regime_metrics",
        "window_metrics",
        "model_comparison",
        "model_selection",
        "all_model_predictions",
        "selected_model",
        "qlib_portfolio_report",
        "risk_analysis",
        "backtest_window_metrics",
        "backtest_walk_forward_metrics",
        "backtest_benchmark_comparison",
        "explainability",
        "failure_modes",
    }
    for rel_path in summary["sidecars"].values():
        assert (card / rel_path).exists(), rel_path

    recorder_dir = Path(summary["recorder_local_dir"])
    assert (recorder_dir / "artifacts" / "pred.pkl").exists()
    assert (recorder_dir / "artifacts" / "label.pkl").exists()
    assert (recorder_dir / "artifacts" / "sig_analysis" / "ic.pkl").exists()
    assert (recorder_dir / "artifacts" / "portfolio_analysis" / "report_normal_1day.pkl").exists()
    assert (recorder_dir / "artifacts" / "portfolio_analysis" / "port_analysis_1day.pkl").exists()
    assert not (recorder_dir / "artifacts" / "code_status.txt").exists()
    assert summary["recorder_id"] in notebook_text
    assert summary["model_selection"]["source"] == "qlib_dataset"
    assert summary["model_selection"]["candidate_count"] >= 6
    assert "PortAnaRecord" in notebook_text
    assert "model_selection.csv" in notebook_text
    assert "backtest_benchmark_comparison.csv" in notebook_text
    assert "risk_analysis.csv" in notebook_text


def test_signal_weight_search_card_beats_legacy_soxlbacktest_with_qlib_portana():
    card = ROOT / "research" / "qlib_signal_weight_search_v1"
    summary = _summary(card)
    notebook_text = (card / "qlib_signal_weight_search_v1.ipynb").read_text(encoding="utf-8")
    conclusion = (card / "conclusion.md").read_text(encoding="utf-8")

    benchmark = summary["legacy_soxlbacktest_benchmark"]
    assert summary["experiment"] == "qlib_signal_weight_search_v1"
    assert summary["feature_set"]["source"] == "Alpha158.get_feature_config+legacy_soxl_features"
    assert summary["feature_set"]["feature_count"] >= 166
    assert summary["target_position_strategy"]["class"] == "SignalWeightStrategy"
    assert "WeightStrategyBase" in summary["qlib_native_api"]
    assert "PortAnaRecord" in summary["qlib_native_api"]
    assert benchmark["benchmark_name"] == "legacy_soxl_backtest"
    assert benchmark["exceeds_legacy_rulebase"] is True
    assert benchmark["strategy_return_pct"] > benchmark["legacy_rulebase_return_pct"]
    assert benchmark["excess_return_pct"] > 0
    assert "不等于无未来函数的生产策略" in conclusion
    assert summary["recorder_id"] in conclusion
    assert "SignalWeightStrategy" in notebook_text
    assert "WeightStrategyBase" in notebook_text

    for rel_path in summary["sidecars"].values():
        assert (card / rel_path).exists(), rel_path


def test_validated_signal_weight_card_beats_legacy_after_validation_selection():
    card = ROOT / "research" / "qlib_validated_signal_weight_v1"
    summary = _summary(card)
    notebook_text = (card / "qlib_validated_signal_weight_v1.ipynb").read_text(encoding="utf-8")
    conclusion = (card / "conclusion.md").read_text(encoding="utf-8")
    benchmark = summary["legacy_soxlbacktest_benchmark"]
    model_selection = summary["model_selection"]

    assert summary["experiment"] == "qlib_validated_signal_weight_v1"
    assert summary["segments"] == {
        "train": ["2012-01-03", "2017-12-29"],
        "valid": ["2018-01-02", "2018-12-31"],
        "test": ["2019-01-02", "2026-06-09"],
    }
    assert model_selection["selection_segment"] == "valid"
    assert model_selection["report_segment"] == "test"
    assert model_selection["selection_metric"] == "objective_score"
    assert model_selection["selection_column"] == "selection_objective_score"
    assert model_selection["selected_model"] == "linear_ic"
    assert summary["target_position_strategy"]["threshold_source"] == "train_prediction_median"
    assert "WeightStrategyBase" in summary["qlib_native_api"]
    assert "PortAnaRecord" in summary["qlib_native_api"]
    assert benchmark["benchmark_name"] == "legacy_soxl_backtest"
    assert benchmark["exceeds_legacy_rulebase"] is True
    assert benchmark["strategy_return_pct"] > benchmark["legacy_rulebase_return_pct"]
    assert benchmark["excess_return_pct"] > 0
    assert "test=2019-01-02 到 2026-06-09 只用于报告" in conclusion
    assert summary["recorder_id"] in conclusion
    assert summary["recorder_id"] in notebook_text

    for rel_path in summary["sidecars"].values():
        assert (card / rel_path).exists(), rel_path


def test_production_card_is_paper_sim_and_references_qlib_recorder():
    card = ROOT / "production" / "qlib_paper_sim_v1"
    summary = _summary(card)
    production_summary = json.loads((card / "production_summary.json").read_text(encoding="utf-8"))
    notebook_text = (card / "qlib_paper_sim_v1.ipynb").read_text(encoding="utf-8")

    assert summary["experiment"] == "qlib_paper_sim_v1"
    assert production_summary["execution_mode"] == "paper_sim"
    assert production_summary["real_broker_orders"] is False
    assert production_summary["recorder_id"] == summary["recorder_id"]
    assert (card / production_summary["order_intents"]).exists()
    assert (card / production_summary["positions"]).exists()
    assert "qlib.data.D.features" in production_summary["qlib_native_api"]
    assert "SignalRecord" in production_summary["qlib_native_api"]
    assert "PortAnaRecord" in production_summary["qlib_native_api"]
    assert summary["model_selection"]["source"] == "qlib_dataset"
    assert summary["model_selection"]["candidate_count"] >= 6
    assert {"model_selection", "backtest_benchmark_comparison"}.issubset(summary["sidecars"])
    assert summary["recorder_id"] in notebook_text
    assert "model_selection.csv" in notebook_text
    assert "backtest_benchmark_comparison.csv" in notebook_text
    assert "risk_analysis.csv" in notebook_text
