from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORDER_ONLY_ROOT_FILES = {
    "all_model_predictions.csv",
    "predictions.csv",
    "signals.csv",
    "portfolio_report.csv",
    "qlib_portfolio_report.csv",
    "qlib_positions.csv",
}
RECORDER_ONLY_SIDECAR_KEYS = {
    "all_model_predictions",
    "predictions",
    "signals",
    "portfolio_report",
    "qlib_portfolio_report",
    "qlib_positions",
}
EXPECTED_RESEARCH_SIDECARS = {
    "analysis_summary",
    "backtest_benchmark_comparison",
    "backtest_walk_forward_metrics",
    "backtest_window_metrics",
    "explainability",
    "failure_modes",
    "model_comparison",
    "model_selection",
    "regime_metrics",
    "risk_analysis",
    "selected_model",
    "window_metrics",
}


def _summary(card: Path) -> dict:
    return json.loads((card / "summary.json").read_text(encoding="utf-8"))


def _tracked_artifact_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _assert_slim_artifact_policy(card: Path, summary: dict) -> None:
    sidecars = set(summary["sidecars"])
    policy = summary["artifact_policy"]
    recorder_only = policy["recorder_only_artifacts"]
    recorder_artifacts = set(summary["recorder_artifacts"])

    assert (card / "workflow.yaml").exists()
    assert summary["workflow_config"] == "workflow.yaml"
    assert policy["policy"] == "slim_card_recorder_first"
    assert policy["primary_artifact"] == f"{summary['experiment']}.ipynb"
    assert policy["machine_entrypoint"] == "summary.json"
    assert sidecars.isdisjoint(RECORDER_ONLY_SIDECAR_KEYS)
    assert RECORDER_ONLY_ROOT_FILES.isdisjoint({path.name for path in card.iterdir() if path.is_file()})
    assert set(policy["kept_sidecars"]) == set(summary["sidecars"].values())
    assert recorder_only["predictions"] == "artifacts/pred.pkl"
    assert recorder_only["labels"] == "artifacts/label.pkl"
    assert recorder_only["all_model_predictions"] == "artifacts/model_selection/all_model_predictions.csv"
    assert recorder_only["portfolio_report"] == "artifacts/portfolio_analysis/report_normal_1day.pkl"
    assert recorder_only["positions"] == "artifacts/portfolio_analysis/positions_normal_1day.pkl"
    assert recorder_only["risk_analysis"] == "artifacts/portfolio_analysis/port_analysis_1day.pkl"
    assert set(recorder_only.values()).issubset(recorder_artifacts)


def test_qlib_card_artifacts_are_portable_to_current_main_checkout():
    stale_tokens = [
        ".worktrees/qlib-native-spec-v1",
        "/Users/rziiiii/dev/workspace/.worktrees/qlib-native-spec-v1",
    ]
    tracked_cards = [
        ROOT / "research" / "qlib_native_reference_v1",
        ROOT / "research" / "qlib_signal_weight_search_v1",
        ROOT / "research" / "qlib_validated_signal_weight_v1",
        ROOT / "production" / "qlib_paper_sim_v1",
    ]

    for card in tracked_cards:
        summary = _summary(card)
        provider_uri = Path(summary["provider_uri"])
        recorder_uri = Path(summary["recorder_uri"])
        recorder_local_dir = Path(summary["recorder_local_dir"])

        assert provider_uri == (ROOT / "data" / "qlib" / "soxl").resolve()
        assert provider_uri.exists()
        assert recorder_uri == (card / "qlib_recorder").resolve()
        assert recorder_uri.exists()
        assert recorder_local_dir == (recorder_uri / summary["experiment_id"] / summary["recorder_id"]).resolve()
        assert recorder_local_dir.exists()

        for artifact in card.rglob("*"):
            if artifact.is_file() and artifact.suffix in {".json", ".md", ".yaml", ".ipynb", ".csv"}:
                text = _tracked_artifact_text(artifact)
                for token in stale_tokens:
                    assert token not in text, f"{artifact} contains stale worktree path {token}"


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
    assert set(summary["sidecars"]) == EXPECTED_RESEARCH_SIDECARS
    _assert_slim_artifact_policy(card, summary)
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


def test_signal_weight_search_card_records_legacy_comparison_with_qlib_portana():
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
    assert isinstance(benchmark["exceeds_legacy_rulebase"], bool)
    assert isinstance(benchmark["strategy_return_pct"], float)
    assert isinstance(benchmark["legacy_rulebase_return_pct"], float)
    assert isinstance(benchmark["excess_return_pct"], float)
    assert "不等于无未来函数的生产策略" in conclusion
    assert "不直接晋升生产" in conclusion
    assert summary["recorder_id"] in conclusion
    assert "SignalWeightStrategy" in notebook_text
    assert "WeightStrategyBase" in notebook_text
    assert set(summary["sidecars"]) == EXPECTED_RESEARCH_SIDECARS
    _assert_slim_artifact_policy(card, summary)

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
    assert set(summary["sidecars"]) == EXPECTED_RESEARCH_SIDECARS
    _assert_slim_artifact_policy(card, summary)

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
    assert summary["model_selection"]["mode"] == "fixed_model"
    assert summary["model_selection"]["candidate_count"] == 1
    assert summary["model_selection"]["fixed_model_name"] == "linear_ic"
    assert summary["model_selection"]["promoted_from"] == "research/qlib_validated_signal_weight_v1"
    assert summary["model_selection"]["selection_metric"] == "objective_score"
    assert summary["segments"] == {
        "train": ["2012-01-03", "2017-12-29"],
        "valid": ["2018-01-02", "2018-12-31"],
        "test": ["2019-01-02", "2026-06-09"],
    }
    assert summary["legacy_soxlbacktest_benchmark"]["exceeds_legacy_rulebase"] is True
    assert set(summary["sidecars"]) == EXPECTED_RESEARCH_SIDECARS
    _assert_slim_artifact_policy(card, summary)
    assert production_summary["fixed_model_name"] == "linear_ic"
    assert production_summary["promoted_from"] == "research/qlib_validated_signal_weight_v1"
    assert {"model_selection", "backtest_benchmark_comparison"}.issubset(summary["sidecars"])
    assert summary["recorder_id"] in notebook_text
    assert "model_selection.csv" in notebook_text
    assert "backtest_benchmark_comparison.csv" in notebook_text
    assert "risk_analysis.csv" in notebook_text


def test_agentic_alpha_rgevd_search_card_records_external_knowledge_and_goal_gate():
    card = ROOT / "research" / "agentic_alpha_rgevd_search_v1"
    summary = _summary(card)
    notebook_text = (card / "agentic_alpha_rgevd_search_v1.ipynb").read_text(encoding="utf-8")
    conclusion = (card / "conclusion.md").read_text(encoding="utf-8")

    assert summary["experiment"] == "agentic_alpha_rgevd_search_v1"
    assert summary["card_type"] == "diagnostic_search"
    assert summary["promotion_decision"] == "not_promoted"
    assert summary["goal_gate"]["max_drawdown_floor"] == -20.0
    assert summary["search_report"]["goal_achieved"] is False
    assert summary["search_report"]["best_candidate"] is None
    assert summary["baseline"]["name"] == "qlib_paper_sim_v1"
    assert summary["baseline"]["max_drawdown_pct"] < -20.0
    assert summary["external_source_counts"] == {"arxiv": 2, "github": 1, "paper": 1, "twitter": 1}
    assert summary["qlib_native_api"] == ["qlib.init", "qlib.data.D.features", "qlib.workflow.R.start"]
    assert summary["qlib_docs"] == {
        "data": "https://qlib.readthedocs.io/en/latest/component/data.html",
        "workflow": "https://qlib.readthedocs.io/en/latest/component/workflow.html",
    }
    assert summary["artifact_policy"]["policy"] == "slim_card_recorder_first"
    assert summary["artifact_policy"]["workflow_config"] == "workflow.yaml"
    assert summary["artifact_policy"]["reproducible_from_recorder"] is True
    assert set(summary["artifact_policy"]["recorder_only_artifacts"]) == {
        "all_model_predictions",
        "labels",
        "portfolio_report",
        "positions",
        "predictions",
        "risk_analysis",
        "signals",
    }
    assert set(summary["sidecars"]) == {
        "candidate_metrics",
        "external_sources",
        "hypotheses",
        "search_summary",
    }
    for rel_path in summary["sidecars"].values():
        assert (card / rel_path).exists(), rel_path

    candidate_rows = (card / summary["sidecars"]["candidate_metrics"]).read_text(encoding="utf-8")
    assert "vol_target_trend_grid" in candidate_rows
    assert "drawdown_gate" in candidate_rows
    assert "return_gate" in candidate_rows
    assert "github" in (card / summary["sidecars"]["external_sources"]).read_text(encoding="utf-8")
    assert "arxiv" in (card / summary["sidecars"]["external_sources"]).read_text(encoding="utf-8")
    assert summary["recorder_id"] in notebook_text
    assert "qlib.init" in notebook_text
    assert "D.features" in notebook_text
    assert "R.start" in notebook_text
    assert "未找到同时满足" in conclusion


def test_agentic_alpha_model_replay_search_card_closes_model_and_cross_asset_branches():
    card = ROOT / "research" / "agentic_alpha_model_replay_search_v1"
    summary = _summary(card)
    notebook_text = (card / "agentic_alpha_model_replay_search_v1.ipynb").read_text(encoding="utf-8")
    conclusion = (card / "conclusion.md").read_text(encoding="utf-8")

    assert summary["experiment"] == "agentic_alpha_model_replay_search_v1"
    assert summary["card_type"] == "diagnostic_search"
    assert summary["promotion_decision"] == "not_promoted"
    assert summary["search_report"]["goal_achieved"] is False
    assert summary["search_report"]["best_candidate"] is None
    assert summary["search_report"]["passes_both_count"] == 0
    assert summary["baseline"]["name"] == "qlib_paper_sim_v1"
    assert summary["goal_gate"] == {
        "max_drawdown_floor": -20.0,
        "require_return_above_baseline": True,
    }
    assert set(summary["branch_results"]) == {
        "production_signal_overlay",
        "qlib_daily_lgbm_replay",
        "cross_asset_lgbm_replay",
    }
    assert summary["branch_results"]["production_signal_overlay"]["coverage_end"] == "2026-06-09"
    assert summary["branch_results"]["cross_asset_lgbm_replay"]["coverage_end"] == "2026-04-10"
    assert summary["branch_results"]["qlib_daily_lgbm_replay"]["best_drawdown_gate_return_pct"] < summary["baseline"][
        "return_pct"
    ]
    assert summary["qlib_docs"] == {
        "data": "https://qlib.readthedocs.io/en/latest/component/data.html",
        "model": "https://qlib.readthedocs.io/en/latest/component/model.html",
        "workflow": "https://qlib.readthedocs.io/en/latest/component/workflow.html",
    }
    assert summary["artifact_policy"]["policy"] == "slim_card_recorder_first"
    assert set(summary["sidecars"]) == {
        "candidate_metrics",
        "external_sources",
        "failure_modes",
        "search_summary",
    }
    for rel_path in summary["sidecars"].values():
        assert (card / rel_path).exists(), rel_path

    candidates = (card / summary["sidecars"]["candidate_metrics"]).read_text(encoding="utf-8")
    assert "production_signal_overlay" in candidates
    assert "qlib_daily_lgbm_replay" in candidates
    assert "cross_asset_lgbm_replay" in candidates
    assert "drawdown_gate" in candidates
    assert "return_gate" in candidates
    assert summary["recorder_id"] in notebook_text
    assert "clear_mem_cache=True" in notebook_text
    assert "LGBMClassifier" in notebook_text
    assert "未找到同时满足" in conclusion


def test_agentic_alpha_intraday_t1_search_card_records_qlib_store_and_goal_gate():
    card = ROOT / "research" / "agentic_alpha_intraday_t1_search_v1"
    summary = _summary(card)
    notebook_text = (card / "agentic_alpha_intraday_t1_search_v1.ipynb").read_text(encoding="utf-8")
    conclusion = (card / "conclusion.md").read_text(encoding="utf-8")
    provider_uri = (ROOT / "data" / "qlib" / "soxl_intraday_t1").resolve()

    assert summary["experiment"] == "agentic_alpha_intraday_t1_search_v1"
    assert summary["card_type"] == "diagnostic_search"
    assert summary["promotion_decision"] == "not_promoted"
    assert Path(summary["provider_uri"]) == provider_uri
    assert summary["intraday_t1"]["source"] == "data/raw/external/intraday_features_soxl_1m.csv"
    assert summary["intraday_t1"]["provider_manifest"] == "data/qlib/soxl_intraday_t1/qlib_store_manifest.json"
    assert summary["intraday_t1"]["lag_policy"] == "T+1_daily_safe"
    assert summary["intraday_t1"]["coverage_start"] == "2024-01-03"
    assert summary["intraday_t1"]["coverage_end"] == "2026-06-05"
    assert summary["intraday_t1"]["feature_count"] >= 5
    assert summary["intraday_t1"]["audit_status"] == "ok"
    assert summary["search_report"]["goal_achieved"] is False
    assert summary["search_report"]["best_candidate"] is None
    assert summary["search_report"]["passes_both_count"] == 0
    assert summary["baseline"]["name"] == "qlib_paper_sim_v1"
    assert summary["goal_gate"] == {
        "max_drawdown_floor": -20.0,
        "require_return_above_baseline": True,
    }
    assert set(summary["branch_results"]) == {
        "intraday_t1_lgbm_replay",
        "intraday_t1_overlay",
    }
    assert summary["branch_results"]["intraday_t1_lgbm_replay"]["coverage_end"] == "2026-06-05"
    assert summary["branch_results"]["intraday_t1_overlay"]["coverage_end"] == "2026-06-05"
    assert summary["qlib_docs"] == {
        "data": "https://qlib.readthedocs.io/en/latest/component/data.html",
        "model": "https://qlib.readthedocs.io/en/latest/component/model.html",
        "workflow": "https://qlib.readthedocs.io/en/latest/component/workflow.html",
    }
    assert summary["artifact_policy"]["policy"] == "slim_card_recorder_first"
    assert set(summary["sidecars"]) == {
        "candidate_metrics",
        "external_sources",
        "failure_modes",
        "intraday_t1_feature_audit",
        "search_summary",
    }
    for rel_path in summary["sidecars"].values():
        assert (card / rel_path).exists(), rel_path

    candidates = (card / summary["sidecars"]["candidate_metrics"]).read_text(encoding="utf-8")
    assert "intraday_t1_lgbm_replay" in candidates
    assert "intraday_t1_overlay" in candidates
    assert "drawdown_gate" in candidates
    assert "return_gate" in candidates
    assert summary["recorder_id"] in notebook_text
    assert "build_qlib_store_from_frame" in notebook_text
    assert "INTRADAY_SOXL_OPEN_RANGE_RETURN_30M" in notebook_text
    assert "D.features" in notebook_text
    assert "R.start" in notebook_text
    assert "LGBMClassifier" in notebook_text
    assert "未找到同时满足" in conclusion
