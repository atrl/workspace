from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


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
    assert summary["intraday_t1"]["coverage_start"] == "2019-01-02"
    assert summary["intraday_t1"]["coverage_end"] == "2026-07-02"
    assert summary["intraday_t1"]["feature_count"] >= 10
    assert "$intraday_soxl_amplitude" in summary["intraday_t1"]["features"]
    assert "$intraday_soxl_close_vwap_deviation" in summary["intraday_t1"]["features"]
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
        "intraday_t1_xgb_replay",
        "intraday_t1_overlay",
    }
    assert summary["branch_results"]["intraday_t1_lgbm_replay"]["coverage_end"] == "2026-07-02"
    assert summary["branch_results"]["intraday_t1_xgb_replay"]["coverage_end"] == "2026-07-02"
    assert summary["branch_results"]["intraday_t1_overlay"]["coverage_end"] == "2026-07-02"
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
        "feature_importance",
        "intraday_t1_feature_audit",
        "search_summary",
    }
    for rel_path in summary["sidecars"].values():
        assert (card / rel_path).exists(), rel_path

    candidates = (card / summary["sidecars"]["candidate_metrics"]).read_text(encoding="utf-8")
    assert "intraday_t1_lgbm_replay" in candidates
    assert "intraday_t1_xgb_replay" in candidates
    assert "intraday_t1_overlay" in candidates
    assert "drawdown_gate" in candidates
    assert "return_gate" in candidates
    assert summary["recorder_id"] in notebook_text
    assert "build_qlib_store_from_frame" in notebook_text
    assert "INTRADAY_SOXL_OPEN_RANGE_RETURN_30M" in notebook_text
    assert "D.features" in notebook_text
    assert "R.start" in notebook_text
    assert "LGBMClassifier" in notebook_text
    assert "XGBClassifier" in notebook_text
    assert "INTRADAY_SOXL_CLOSE_VWAP_DEVIATION" in notebook_text
    assert "未找到同时满足" in conclusion


def test_agentic_alpha_intraday_5m15m_lgb_xgb_card_records_minute_qlib_research():
    card = ROOT / "research" / "agentic_alpha_intraday_5m15m_lgb_xgb_v1"
    summary = _summary(card)
    notebook_text = (card / "agentic_alpha_intraday_5m15m_lgb_xgb_v1.ipynb").read_text(encoding="utf-8")
    conclusion = (card / "conclusion.md").read_text(encoding="utf-8")

    assert summary["experiment"] == "agentic_alpha_intraday_5m15m_lgb_xgb_v1"
    assert summary["card_type"] == "diagnostic_search"
    assert summary["promotion_decision"] == "not_promoted_intraday_stability_not_proven"
    assert summary["as_of"] == "2026-07-02"
    assert summary["model_selection"]["source"] == "qlib_dataset"
    assert summary["model_selection"]["candidate_count"] == 6
    assert set(summary["model_selection"]["models"]) == {"lgb", "xgb", "lgb_xgb_consensus"}
    assert summary["selected_model"]["freq"] in {"5min", "15min"}
    assert summary["selected_model"]["model"] in {"lgb", "xgb", "lgb_xgb_consensus"}
    assert summary["purged_split"]["purge_bars"] == 1
    assert summary["purged_split"]["label_policy"] == "same_session_next_bar_open_to_close"
    assert {
        "qlib.init",
        "qlib.data.D.features",
        "DatasetH",
        "DataHandlerLP",
        "QlibDataLoader",
        "qlib.workflow.R.start",
        "LGBMClassifier",
        "XGBClassifier",
    }.issubset(set(summary["qlib_native_api"]))
    assert summary["artifact_policy"]["policy"] == "slim_card_recorder_first"
    assert summary["artifact_policy"]["recorder_only_artifacts"] == {
        "predictions": "artifacts/pred.pkl",
        "labels": "artifacts/label.pkl",
        "all_model_predictions": "artifacts/model_selection/all_model_predictions.csv",
    }
    assert set(summary["artifact_policy"]["recorder_only_artifacts"].values()).issubset(
        set(summary["recorder_artifacts"])
    )
    assert RECORDER_ONLY_ROOT_FILES.isdisjoint({path.name for path in card.iterdir() if path.is_file()})

    expected_sidecars = {
        "analysis_summary",
        "backtest_walk_forward_metrics",
        "cost_sensitivity",
        "failure_modes",
        "feature_importance",
        "model_selection",
        "monthly_stability",
        "selected_model",
    }
    assert set(summary["sidecars"]) == expected_sidecars
    for rel_path in summary["sidecars"].values():
        assert (card / rel_path).exists(), rel_path

    for freq, expected_start in {"5min": "2018-12-31 09:35:00", "15min": "2018-12-31 09:45:00"}.items():
        provider_uri = (ROOT / "data" / "qlib" / f"soxl_intraday_{freq}").resolve()
        manifest = json.loads((provider_uri / "qlib_store_manifest.json").read_text(encoding="utf-8"))
        assert Path(summary["provider_uris"][freq]) == provider_uri
        assert summary["provider_audit"][freq]["provider_uri"] == provider_uri.as_posix()
        assert summary["provider_audit"][freq]["calendar_start"] == expected_start
        assert summary["provider_audit"][freq]["calendar_end"] == "2026-07-02 16:00:00"
        assert summary["provider_audit"][freq]["purged_split"]["same_session_last_bar_rows_dropped"] == 1885
        assert manifest["freq"] == freq
        assert manifest["status"] == "ok"
        assert set(manifest["instruments"]) == {"SOXL", "SOXX"}
        assert manifest["calendar_end"] == "2026-07-02 16:00:00"

    recorder_dir = Path(summary["recorder_local_dir"])
    assert (recorder_dir / "artifacts" / "pred.pkl").exists()
    assert (recorder_dir / "artifacts" / "label.pkl").exists()
    assert (recorder_dir / "artifacts" / "model_selection" / "all_model_predictions.csv").exists()

    model_selection = pd.read_csv(card / summary["sidecars"]["model_selection"])
    assert set(model_selection["freq"]) == {"5min", "15min"}
    assert set(model_selection["model"]) == {"lgb", "xgb", "lgb_xgb_consensus"}
    cost_sensitivity = pd.read_csv(card / summary["sidecars"]["cost_sensitivity"])
    assert {0.0, 5.0, 10.0, 20.0}.issubset(set(cost_sensitivity["cost_bps"]))
    monthly = pd.read_csv(card / summary["sidecars"]["monthly_stability"])
    assert {"2026-01", "2026-06"}.issubset(set(monthly["month"]))
    walk_forward = pd.read_csv(card / summary["sidecars"]["backtest_walk_forward_metrics"])
    assert set(walk_forward["fold"]) == {"wf_2024", "wf_2025", "wf_2026"}
    assert set(walk_forward["freq"]) == {"5min", "15min"}

    assert summary["recorder_id"] in notebook_text
    assert "run_intraday_bar_model_card" in notebook_text
    assert "DatasetH" in notebook_text
    assert "R.start" in notebook_text
    assert "不晋升" in conclusion


def test_agentic_alpha_intraday_15m_regime_gated_card_records_cost_stable_selection():
    card = ROOT / "research" / "agentic_alpha_intraday_15m_regime_gated_v1"
    summary = _summary(card)
    notebook_text = (card / "agentic_alpha_intraday_15m_regime_gated_v1.ipynb").read_text(encoding="utf-8")
    conclusion = (card / "conclusion.md").read_text(encoding="utf-8")

    assert summary["experiment"] == "agentic_alpha_intraday_15m_regime_gated_v1"
    assert summary["card_type"] == "intraday_regime_gated_model_search"
    assert summary["promotion_decision"] in {
        "not_promoted_cost_stress_not_proven",
        "not_promoted_walk_forward_not_proven",
        "not_promoted_intraday_replay_missing",
    }
    assert summary["as_of"] == "2026-07-02"
    assert set(summary["provider_uris"]) == {"15min"}
    assert summary["selected_model"]["freq"] == "15min"
    assert summary["model_selection"]["candidate_count"] == 3
    assert set(summary["model_selection"]["models"]) == {"lgb", "xgb", "lgb_xgb_consensus"}
    assert summary["model_selection"]["selection_segment"] == "valid"
    assert summary["model_selection"]["selection_uses_test_segment"] is False
    assert summary["feature_set"]["name"] == "regime_v1"
    assert summary["feature_set"]["lookback_only"] is True
    assert summary["trade_gate"]["name"] == "regime_v2_dislocation"
    assert summary["trade_gate"]["column"] == "regime_gate_v2_dislocation"
    assert summary["trade_gate"]["dislocation_threshold"] == 0.0012
    assert summary["trade_gate"]["eligible_on_selected_test"] > 0
    assert summary["execution_policy"]["frequency"] == "15min"
    assert summary["execution_policy"]["label_policy"] == "same_session_next_bar_open_to_close"
    assert summary["execution_policy"]["entry_timing"] == "bar_t_plus_1_open"
    assert summary["execution_policy"]["exit_timing"] == "bar_t_plus_1_close"
    assert summary["execution_policy"]["holds_overnight"] is False
    assert summary["execution_policy"]["open_to_open"] is False
    assert summary["return_col"] == "target_return"
    assert summary["model_target_col"] == "target_positive"
    assert summary["execution_timing"] == "bar_t_close_signal_next_15m_open_to_close"
    assert summary["cost_bps"] == 10.0
    assert summary["cost_stress_bps"] == [0.0, 10.0, 20.0]
    assert summary["artifact_policy"]["policy"] == "slim_card_recorder_first"
    assert summary["artifact_policy"]["recorder_only_artifacts"] == {
        "predictions": "artifacts/pred.pkl",
        "labels": "artifacts/label.pkl",
        "all_model_predictions": "artifacts/model_selection/all_model_predictions.csv",
    }
    assert set(summary["artifact_policy"]["recorder_only_artifacts"].values()).issubset(
        set(summary["recorder_artifacts"])
    )
    assert RECORDER_ONLY_ROOT_FILES.isdisjoint({path.name for path in card.iterdir() if path.is_file()})
    assert {
        "qlib.init",
        "qlib.data.D.features",
        "DatasetH",
        "DataHandlerLP",
        "QlibDataLoader",
        "qlib.workflow.R.start",
        "LGBMClassifier",
        "XGBClassifier",
    }.issubset(set(summary["qlib_native_api"]))

    expected_sidecars = {
        "analysis_summary",
        "backtest_walk_forward_metrics",
        "cost_sensitivity",
        "failure_modes",
        "feature_importance",
        "model_selection",
        "monthly_stability",
        "regime_gate_metrics",
        "selected_model",
    }
    assert set(summary["sidecars"]) == expected_sidecars
    for rel_path in summary["sidecars"].values():
        assert (card / rel_path).exists(), rel_path

    model_selection = pd.read_csv(card / summary["sidecars"]["model_selection"])
    assert set(model_selection["freq"]) == {"15min"}
    assert set(model_selection["model"]) == {"lgb", "xgb", "lgb_xgb_consensus"}
    assert model_selection["selection_eligible"].dtype == bool
    assert "historical_walk_forward_positive_ratio" in model_selection.columns
    assert model_selection["test_net_return_pct"].notna().all()
    assert model_selection["selection_score"].notna().all()

    cost_sensitivity = pd.read_csv(card / summary["sidecars"]["cost_sensitivity"])
    assert set(cost_sensitivity["cost_bps"]) == {0.0, 10.0, 20.0}
    assert set(cost_sensitivity["model"]) == {"lgb", "xgb", "lgb_xgb_consensus"}
    walk_forward = pd.read_csv(card / summary["sidecars"]["backtest_walk_forward_metrics"])
    assert set(walk_forward["fold"]) == {"wf_2024", "wf_2025", "wf_2026"}
    assert set(walk_forward["model"]) == {"lgb", "xgb", "lgb_xgb_consensus"}
    assert set(walk_forward["freq"]) == {"15min"}
    selected_walk_forward = walk_forward.loc[walk_forward["model"] == summary["selected_model"]["model"]]
    assert (selected_walk_forward["test_net_return_pct"] > 0.0).all()
    assert selected_walk_forward["test_max_drawdown_pct"].min() > -10.0
    regime_gate = pd.read_csv(card / summary["sidecars"]["regime_gate_metrics"])
    assert set(regime_gate["segment"]) == {"train", "valid", "test"}
    assert (regime_gate["eligible_rows"] > 0).all()

    recorder_dir = Path(summary["recorder_local_dir"])
    assert (recorder_dir / "artifacts" / "pred.pkl").exists()
    assert (recorder_dir / "artifacts" / "label.pkl").exists()
    assert (recorder_dir / "artifacts" / "model_selection" / "all_model_predictions.csv").exists()
    assert summary["recorder_id"] in notebook_text
    assert "run_intraday_regime_gated_model_card" in notebook_text
    assert "DatasetH" in notebook_text
    assert "R.start" in notebook_text
    assert "regime_v2_dislocation" in conclusion
    assert "10bp" in conclusion
    assert "20bp" in conclusion
    assert "日内执行" in conclusion


def test_cognitive_alpha_intraday_execution_overlay_card_records_daily_plus_intraday_overlay():
    card = ROOT / "research" / "cognitive_alpha_intraday_execution_overlay_v1"
    summary = _summary(card)
    notebook_text = (card / "cognitive_alpha_intraday_execution_overlay_v1.ipynb").read_text(encoding="utf-8")
    conclusion = (card / "conclusion.md").read_text(encoding="utf-8")

    assert summary["experiment"] == "cognitive_alpha_intraday_execution_overlay_v1"
    assert summary["card_type"] == "execution_overlay_research"
    assert summary["as_of"] == "2026-07-02"
    assert summary["source_daily_model"]["card"] == "research/cognitive_alpha_goal_codex_exec_v1"
    assert summary["source_daily_model"]["execution_timing"] == "close_t_signal_next_open_to_following_open"
    assert summary["source_intraday_model"]["card"] == "research/agentic_alpha_intraday_15m_regime_gated_v1"
    assert summary["execution_overlay"]["frequency"] == "15min"
    assert summary["execution_overlay"]["decision_model_timing"] == "daily_close_t_signal"
    assert summary["execution_overlay"]["overlay_signal_timing"] == "15m_bar_close_to_next_bar_open"
    assert summary["execution_overlay"]["variants"] == [
        "baseline_next_open",
        "overlay_a_signal_gate",
        "overlay_b_twap4",
        "overlay_c_risk_exit",
    ]
    assert summary["model_selection"]["selection_segment"] == "validation"
    assert summary["model_selection"]["report_segment"] == "test"
    assert summary["model_selection"]["selection_uses_test_segment"] is False
    assert {10.0, 20.0, 50.0}.issubset(set(summary["cost_stress_bps"]))
    assert {
        "qlib.init",
        "qlib.data.D.features",
        "qlib.workflow.R.start",
        "qlib.workflow.Recorder.log_artifact",
    }.issubset(set(summary["qlib_native_api"]))
    assert summary["artifact_policy"]["policy"] == "slim_card_recorder_first"
    assert summary["artifact_policy"]["recorder_only_artifacts"] == {
        "overlay_trades": "artifacts/execution_overlay/overlay_trades.csv",
        "daily_positions": "artifacts/execution_overlay/daily_selected_positions.csv",
        "intraday_predictions": "artifacts/execution_overlay/intraday_predictions_15m.csv",
    }
    assert set(summary["artifact_policy"]["recorder_only_artifacts"].values()).issubset(
        set(summary["recorder_artifacts"])
    )
    assert RECORDER_ONLY_ROOT_FILES.isdisjoint({path.name for path in card.iterdir() if path.is_file()})

    expected_sidecars = {
        "analysis_summary",
        "backtest_walk_forward_metrics",
        "cost_sensitivity",
        "failure_modes",
        "model_comparison",
        "selected_model",
    }
    assert set(summary["sidecars"]) == expected_sidecars
    for rel_path in summary["sidecars"].values():
        assert (card / rel_path).exists(), rel_path

    model_comparison = pd.read_csv(card / summary["sidecars"]["model_comparison"])
    assert set(model_comparison["variant"]) == set(summary["execution_overlay"]["variants"])
    assert {"validation", "test"}.issubset(set(model_comparison["segment"]))
    cost_sensitivity = pd.read_csv(card / summary["sidecars"]["cost_sensitivity"])
    assert {10.0, 20.0, 50.0}.issubset(set(cost_sensitivity["cost_bps"]))
    walk_forward = pd.read_csv(card / summary["sidecars"]["backtest_walk_forward_metrics"])
    assert set(walk_forward["fold"]) == {"wf_2025_h2", "wf_2026_ytd"}

    recorder_dir = Path(summary["recorder_local_dir"])
    assert (recorder_dir / "artifacts" / "execution_overlay" / "overlay_trades.csv").exists()
    assert (recorder_dir / "artifacts" / "execution_overlay" / "daily_selected_positions.csv").exists()
    assert (recorder_dir / "artifacts" / "execution_overlay" / "intraday_predictions_15m.csv").exists()
    assert summary["recorder_id"] in notebook_text
    assert "run_cognitive_alpha_intraday_execution_overlay_card" in notebook_text
    assert "qlib.init" in notebook_text
    assert "R.start" in notebook_text
    assert "日级模型 + 日内执行 overlay" in conclusion


def test_cognitive_alpha_intraday_external_context_overlay_card_records_upgraded_intraday_source():
    card = ROOT / "research" / "cognitive_alpha_intraday_external_context_overlay_v1"
    summary = _summary(card)
    notebook_text = (card / "cognitive_alpha_intraday_external_context_overlay_v1.ipynb").read_text(encoding="utf-8")
    conclusion = (card / "conclusion.md").read_text(encoding="utf-8")

    assert summary["experiment"] == "cognitive_alpha_intraday_external_context_overlay_v1"
    assert summary["card_type"] == "execution_overlay_research"
    assert summary["as_of"] == "2026-07-02"
    assert summary["source_daily_model"]["card"] == "research/cognitive_alpha_goal_codex_exec_v1"
    assert summary["source_intraday_model"]["card"] == "research/agentic_alpha_intraday_external_context_label_v1"
    assert summary["source_intraday_model"]["label_policy_name"] == "triple_barrier_1h_2h_execution_improvement"
    assert summary["source_intraday_model"]["external_context"]["data_status"] == "ok_partial_insufficient_for_full_walk_forward"
    assert summary["source_intraday_model"]["signal_prob_col"] == "prob_lgb"
    assert summary["source_intraday_model"]["gate_policy"] == "all_predicted_15m_rows"
    assert summary["execution_overlay"]["variants"] == [
        "baseline_next_open",
        "overlay_a_signal_gate",
        "overlay_b_twap4",
        "overlay_c_risk_exit",
    ]
    assert summary["execution_overlay"]["intraday_label_policy"] == "triple_barrier_1h_2h_execution_improvement"
    assert summary["execution_overlay"]["uses_external_intraday_context"] is True
    assert summary["model_selection"]["selection_segment"] == "validation"
    assert summary["model_selection"]["report_segment"] == "test"
    assert summary["model_selection"]["selection_uses_test_segment"] is False
    assert summary["model_selection"]["threshold_search"]["enabled"] is True
    assert summary["model_selection"]["threshold_search"]["selection_uses_test_segment"] is False
    assert summary["model_selection"]["threshold_search"]["target_candidate_count"] == 0
    assert summary["selected_model"]["variant"] == "baseline_next_open"
    assert summary["selected_model"]["validation_passes_target"] is False
    assert summary["walk_forward"]["required_policy_satisfied"] is False
    assert "not_promoted" in summary["promotion_decision"]
    assert summary["artifact_policy"]["policy"] == "slim_card_recorder_first"
    assert RECORDER_ONLY_ROOT_FILES.isdisjoint({path.name for path in card.iterdir() if path.is_file()})

    expected_sidecars = {
        "analysis_summary",
        "backtest_walk_forward_metrics",
        "cost_sensitivity",
        "failure_modes",
        "model_comparison",
        "selected_model",
        "threshold_selection",
    }
    assert set(summary["sidecars"]) == expected_sidecars
    for rel_path in summary["sidecars"].values():
        assert (card / rel_path).exists(), rel_path

    model_comparison = pd.read_csv(card / summary["sidecars"]["model_comparison"])
    assert set(model_comparison["variant"]) == set(summary["execution_overlay"]["variants"])
    assert {"validation", "test"}.issubset(set(model_comparison["segment"]))
    threshold_selection = pd.read_csv(card / summary["sidecars"]["threshold_selection"])
    assert threshold_selection["selection_uses_test_segment"].eq(False).all()
    assert "overlay_a_signal_gate" in set(threshold_selection["variant"])
    assert not threshold_selection["passes_target"].any()
    assert {10.0, 20.0, 50.0}.issubset(set(pd.read_csv(card / summary["sidecars"]["cost_sensitivity"])["cost_bps"]))

    recorder_dir = Path(summary["recorder_local_dir"])
    assert (recorder_dir / "artifacts" / "execution_overlay" / "overlay_trades.csv").exists()
    assert (recorder_dir / "artifacts" / "execution_overlay" / "daily_selected_positions.csv").exists()
    assert (recorder_dir / "artifacts" / "execution_overlay" / "intraday_predictions_15m.csv").exists()
    assert summary["recorder_id"] in notebook_text
    assert "run_cognitive_alpha_intraday_external_context_overlay_card" in notebook_text
    assert "external_context" in notebook_text
    assert "R.start" in notebook_text
    assert "外部 15m 上下文执行 overlay" in conclusion


def test_cognitive_alpha_intraday_regime_overlay_diagnostic_card_records_causal_policy_search():
    card = ROOT / "research" / "cognitive_alpha_intraday_regime_overlay_diagnostic_v1"
    summary = _summary(card)
    notebook_text = (card / "cognitive_alpha_intraday_regime_overlay_diagnostic_v1.ipynb").read_text(encoding="utf-8")
    conclusion = (card / "conclusion.md").read_text(encoding="utf-8")

    assert summary["experiment"] == "cognitive_alpha_intraday_regime_overlay_diagnostic_v1"
    assert summary["card_type"] == "intraday_regime_overlay_diagnostic"
    assert summary["as_of"] == "2026-07-02"
    assert summary["source_daily_model"]["card"] == "research/cognitive_alpha_goal_codex_exec_v1"
    assert summary["source_intraday_model"]["card"] == "research/agentic_alpha_intraday_external_context_label_v1"
    assert summary["source_intraday_model"]["signal_prob_col"] == "prob_lgb"
    assert summary["execution_overlay"]["variants"] == [
        "baseline_next_open",
        "overlay_a_signal_gate",
        "overlay_b_twap4",
        "overlay_c_risk_exit",
    ]
    assert summary["regime_policy"]["causal_scope"] == "daily_close_known_features_only"
    assert summary["regime_policy"]["excludes_first_hour_features_from_twap_selection"] is True
    assert summary["regime_policy"]["selection_uses_test_segment"] is False
    assert summary["regime_policy"]["condition_threshold_source"] == "validation_quantile"
    assert summary["regime_policy"]["validation_target_candidate_count"] >= 1
    assert summary["regime_policy"]["validation_and_test_target_candidate_count"] == 0
    assert summary["model_selection"]["selection_uses_test_segment"] is False
    assert summary["selected_model"]["validation_passes_target"] is True
    assert summary["selected_model"]["test_passes_target"] is False
    assert "not_promoted" in summary["promotion_decision"]
    assert summary["artifact_policy"]["policy"] == "slim_card_recorder_first"
    assert summary["artifact_policy"]["recorder_only_artifacts"] == {
        "overlay_trades": "artifacts/regime_overlay/overlay_trades.csv",
        "policy_selection": "artifacts/regime_overlay/policy_selection.csv",
        "regime_features": "artifacts/regime_overlay/regime_features.csv",
    }
    assert set(summary["artifact_policy"]["recorder_only_artifacts"].values()).issubset(
        set(summary["recorder_artifacts"])
    )
    assert RECORDER_ONLY_ROOT_FILES.isdisjoint({path.name for path in card.iterdir() if path.is_file()})

    expected_sidecars = {
        "analysis_summary",
        "failure_modes",
        "model_comparison",
        "policy_selection",
        "selected_model",
    }
    assert set(summary["sidecars"]) == expected_sidecars
    for rel_path in summary["sidecars"].values():
        assert (card / rel_path).exists(), rel_path

    policy_selection = pd.read_csv(card / summary["sidecars"]["policy_selection"])
    assert policy_selection["selection_uses_test_segment"].eq(False).all()
    assert set(policy_selection["condition_threshold_source"]) == {"validation_quantile"}
    assert policy_selection["passes_validation_target"].any()
    assert not policy_selection["passes_validation_and_test_target"].any()

    recorder_dir = Path(summary["recorder_local_dir"])
    assert (recorder_dir / "artifacts" / "regime_overlay" / "overlay_trades.csv").exists()
    assert (recorder_dir / "artifacts" / "regime_overlay" / "regime_features.csv").exists()
    assert (recorder_dir / "artifacts" / "regime_overlay" / "policy_selection.csv").exists()
    assert summary["recorder_id"] in notebook_text
    assert "run_cognitive_alpha_intraday_regime_overlay_diagnostic_card" in notebook_text
    assert "validation-only quantiles" in notebook_text
    assert "causal regime overlay" in conclusion


def test_cognitive_alpha_intraday_delayed_observable_overlay_card_records_robust_delayed_entry():
    card = ROOT / "research" / "cognitive_alpha_intraday_delayed_observable_overlay_v1"
    summary = _summary(card)
    notebook_text = (card / "cognitive_alpha_intraday_delayed_observable_overlay_v1.ipynb").read_text(encoding="utf-8")
    conclusion = (card / "conclusion.md").read_text(encoding="utf-8")

    assert summary["experiment"] == "cognitive_alpha_intraday_delayed_observable_overlay_v1"
    assert summary["card_type"] == "intraday_delayed_observable_overlay"
    assert summary["as_of"] == "2026-07-02"
    assert summary["source_daily_model"]["card"] == "research/cognitive_alpha_goal_codex_exec_v1"
    assert summary["source_intraday_model"]["card"] == "research/agentic_alpha_intraday_external_context_label_v1"
    assert summary["source_intraday_model"]["signal_prob_col"] == "prob_lgb"
    assert summary["execution_overlay"]["overlay_signal_timing"] == "first_30_or_60m_observed_then_15m_bar_close_to_next_bar_open"
    assert summary["execution_overlay"]["intraday_overlays_hold_overnight"] is False
    assert summary["model_selection"]["selection_segment"] == "validation"
    assert summary["model_selection"]["report_segment"] == "test"
    assert summary["model_selection"]["selection_uses_test_segment"] is False
    assert summary["model_selection"]["required_cost_bps"] == [20.0, 50.0]
    assert summary["model_selection"]["max_skip_rate"] == 0.15
    assert summary["model_selection"]["validation_and_test_target_row_count"] > 0
    assert summary["walk_forward"]["policy"] == "annual_train_to_T_minus_2_valid_T_minus_1_test_T"
    assert summary["walk_forward"]["selection_uses_test_segment"] is False
    assert summary["walk_forward"]["required_policy_satisfied"] is False
    assert summary["walk_forward"]["required_test_years"] == [2023, 2024, 2025, 2026]
    assert summary["walk_forward"]["feasible_test_years"] == []
    assert summary["walk_forward"]["missing_required_test_years"] == [2023, 2024, 2025, 2026]
    assert summary["walk_forward"]["data_status"] == "ok_partial_insufficient_for_full_walk_forward"
    assert "insufficient_external_intraday_history" in summary["walk_forward"]["blocking_reasons"]
    assert summary["selected_model"]["validation_passes_target"] is True
    assert summary["selected_model"]["passes_test_target"] is True
    assert summary["selected_model"]["observation_minutes"] in {30, 60}
    assert "not_promoted" in summary["promotion_decision"]
    assert summary["artifact_policy"]["policy"] == "slim_card_recorder_first"
    assert summary["artifact_policy"]["recorder_only_artifacts"] == {
        "all_policy_selection": "artifacts/delayed_observable_overlay/all_policy_selection.csv",
        "selected_trades": "artifacts/delayed_observable_overlay/selected_trades.csv",
    }
    assert set(summary["artifact_policy"]["recorder_only_artifacts"].values()).issubset(
        set(summary["recorder_artifacts"])
    )
    assert RECORDER_ONLY_ROOT_FILES.isdisjoint({path.name for path in card.iterdir() if path.is_file()})

    expected_sidecars = {
        "analysis_summary",
        "failure_modes",
        "model_comparison",
        "policy_selection",
        "selected_model",
        "selected_policy_cost_sensitivity",
        "walk_forward_feasibility",
    }
    assert set(summary["sidecars"]) == expected_sidecars
    for rel_path in summary["sidecars"].values():
        assert (card / rel_path).exists(), rel_path

    walk_forward = pd.read_csv(card / summary["sidecars"]["walk_forward_feasibility"])
    assert set(walk_forward["test_year"]) == {2023, 2024, 2025, 2026}
    assert walk_forward["selection_uses_test_segment"].eq(False).all()
    assert walk_forward["feasible"].eq(False).all()
    assert "missing_train_coverage" in set(walk_forward["blocking_reason"])

    selected_costs = pd.read_csv(card / summary["sidecars"]["selected_policy_cost_sensitivity"])
    required = selected_costs.loc[selected_costs["cost_bps"].isin([20.0, 50.0])]
    assert set(required["cost_bps"]) == {20.0, 50.0}
    assert required["passes_validation_target"].all()
    assert required["passes_test_target"].all()
    assert required["test_max_drawdown_pct"].max() > -18.0

    recorder_dir = Path(summary["recorder_local_dir"])
    assert (recorder_dir / "artifacts" / "delayed_observable_overlay" / "selected_trades.csv").exists()
    assert (recorder_dir / "artifacts" / "delayed_observable_overlay" / "all_policy_selection.csv").exists()
    assert summary["recorder_id"] in notebook_text
    assert "run_cognitive_alpha_intraday_delayed_observable_overlay_card" in notebook_text
    assert "first 30/60 minutes" in notebook_text
    assert "delayed observable intraday overlay" in conclusion


def test_agentic_alpha_intraday_execution_label_card_records_upgraded_labels_and_context_features():
    card = ROOT / "research" / "agentic_alpha_intraday_execution_label_v1"
    summary = _summary(card)
    notebook_text = (card / "agentic_alpha_intraday_execution_label_v1.ipynb").read_text(encoding="utf-8")
    conclusion = (card / "conclusion.md").read_text(encoding="utf-8")

    assert summary["experiment"] == "agentic_alpha_intraday_execution_label_v1"
    assert summary["card_type"] == "intraday_execution_label_model_search"
    assert summary["as_of"] == "2026-07-02"
    assert summary["provider_uris"] == {
        "15min": (ROOT / "data" / "qlib" / "soxl_intraday_15min").resolve().as_posix()
    }
    assert summary["label_policy"]["name"] == "triple_barrier_1h_2h_execution_improvement"
    assert summary["label_policy"]["entry_timing"] == "bar_t_close_signal_next_15m_open"
    assert summary["label_policy"]["horizon_bars"] == 8
    assert summary["label_policy"]["uses_same_bar_future"] is False
    assert summary["feature_set"]["name"] == "execution_context_v1"
    assert summary["feature_set"]["lookback_only"] is True
    assert "soxl_open_gap_from_prior_close" in summary["feature_set"]["new_features"]
    assert "soxl_realized_vol_expansion" in summary["feature_set"]["new_features"]
    assert "soxl_volume_price_absorption" in summary["feature_set"]["new_features"]
    assert summary["external_intraday_context"]["available_symbols"] == ["SOXL", "SOXX"]
    assert {"QQQ", "SMH", "NVDA", "AVGO", "AMD", "TSM", "ASML"}.issubset(
        set(summary["external_intraday_context"]["missing_requested_symbols"])
    )
    assert summary["model_selection"]["selection_segment"] == "valid"
    assert summary["model_selection"]["report_segment"] == "test"
    assert summary["model_selection"]["selection_uses_test_segment"] is False
    assert set(summary["model_selection"]["models"]) == {"lgb", "xgb", "lgb_xgb_consensus"}
    assert set(summary["walk_forward"]["test_years"]) == {2023, 2024, 2025, 2026}
    assert {10.0, 20.0, 50.0}.issubset(set(summary["cost_stress_bps"]))
    assert {
        "qlib.init",
        "qlib.data.D.features",
        "DatasetH",
        "DataHandlerLP",
        "QlibDataLoader",
        "qlib.workflow.R.start",
        "LGBMClassifier",
        "XGBClassifier",
    }.issubset(set(summary["qlib_native_api"]))
    assert summary["artifact_policy"]["policy"] == "slim_card_recorder_first"
    assert summary["artifact_policy"]["recorder_only_artifacts"] == {
        "predictions": "artifacts/pred.pkl",
        "labels": "artifacts/label.pkl",
        "all_model_predictions": "artifacts/model_selection/all_model_predictions.csv",
    }
    assert set(summary["artifact_policy"]["recorder_only_artifacts"].values()).issubset(
        set(summary["recorder_artifacts"])
    )
    assert RECORDER_ONLY_ROOT_FILES.isdisjoint({path.name for path in card.iterdir() if path.is_file()})

    expected_sidecars = {
        "analysis_summary",
        "backtest_walk_forward_metrics",
        "cost_sensitivity",
        "failure_modes",
        "feature_importance",
        "label_diagnostics",
        "model_selection",
        "selected_model",
    }
    assert set(summary["sidecars"]) == expected_sidecars
    for rel_path in summary["sidecars"].values():
        assert (card / rel_path).exists(), rel_path

    model_selection = pd.read_csv(card / summary["sidecars"]["model_selection"])
    assert set(model_selection["model"]) == {"lgb", "xgb", "lgb_xgb_consensus"}
    assert model_selection["selection_score"].notna().all()
    walk_forward = pd.read_csv(card / summary["sidecars"]["backtest_walk_forward_metrics"])
    assert set(walk_forward["test_year"]) == {2023, 2024, 2025, 2026}
    label_diagnostics = pd.read_csv(card / summary["sidecars"]["label_diagnostics"])
    assert {"profit", "stop", "timeout"}.issubset(set(label_diagnostics["triple_barrier_outcome"]))

    recorder_dir = Path(summary["recorder_local_dir"])
    assert (recorder_dir / "artifacts" / "pred.pkl").exists()
    assert (recorder_dir / "artifacts" / "label.pkl").exists()
    assert (recorder_dir / "artifacts" / "model_selection" / "all_model_predictions.csv").exists()
    assert summary["recorder_id"] in notebook_text
    assert "run_intraday_execution_label_model_card" in notebook_text
    assert "triple_barrier" in notebook_text
    assert "R.start" in notebook_text
    assert "triple-barrier" in conclusion
    assert "不晋升" in conclusion


def test_intraday_external_context_data_card_records_requested_symbols_and_fail_closed_coverage():
    card = ROOT / "research" / "intraday_external_context_data_v1"
    summary = _summary(card)
    notebook_text = (card / "intraday_external_context_data_v1.ipynb").read_text(encoding="utf-8")
    conclusion = (card / "conclusion.md").read_text(encoding="utf-8")

    requested_symbols = {"QQQ", "SMH", "SOXX", "NVDA", "AVGO", "AMD", "TSM", "ASML"}
    assert summary["experiment"] == "intraday_external_context_data_v1"
    assert summary["card_type"] == "intraday_external_context_data"
    assert summary["source"] == "polygon_aggs_custom_bars"
    assert summary["requested_range"] == ["2025-01-02", "2026-07-02"]
    assert set(summary["requested_symbols"]) == requested_symbols
    assert summary["frequency"] == "15min"
    assert summary["data_status"] in {
        "ok",
        "ok_partial_insufficient_for_full_walk_forward",
        "blocked_missing_api_key",
        "blocked_provider_rate_limit",
        "blocked_provider_timeout",
        "blocked_provider_history",
        "blocked_no_rows",
    }
    assert set(summary["available_symbols"]).issubset(requested_symbols)
    assert set(summary["missing_symbols"]).issubset(requested_symbols)
    assert set(summary["available_symbols"]).union(summary["missing_symbols"]) == requested_symbols
    assert {"QQQ", "SMH", "NVDA", "AVGO", "AMD", "TSM", "ASML"}.issubset(
        set(summary["objective_context_gap_closed_or_audited"])
    )
    assert {
        "qlib.init",
        "qlib.data.D.features",
        "qlib.workflow.R.start",
        "qlib.workflow.Recorder.log_artifact",
        "build_qlib_store_from_frame",
    }.issubset(set(summary["qlib_native_api"]))
    assert summary["qlib_docs"] == {
        "data": "https://qlib.readthedocs.io/en/latest/component/data.html",
        "highfreq": "https://qlib.readthedocs.io/en/latest/component/highfreq.html",
        "workflow": "https://qlib.readthedocs.io/en/latest/component/workflow.html",
    }
    assert summary["artifact_policy"]["policy"] == "slim_card_recorder_first"
    assert summary["artifact_policy"]["workflow_config"] == "workflow.yaml"
    assert summary["artifact_policy"]["recorder_only_artifacts"] == {
        "source_audit": "artifacts/intraday_external_context/source_audit.csv",
        "qlib_feature_sample": "artifacts/intraday_external_context/qlib_feature_sample.csv",
    }
    assert set(summary["artifact_policy"]["recorder_only_artifacts"].values()).issubset(
        set(summary["recorder_artifacts"])
    )
    assert RECORDER_ONLY_ROOT_FILES.isdisjoint({path.name for path in card.iterdir() if path.is_file()})

    expected_sidecars = {
        "analysis_summary",
        "coverage_audit",
        "failure_modes",
        "qlib_store_manifest",
        "source_audit",
    }
    assert set(summary["sidecars"]) == expected_sidecars
    for rel_path in summary["sidecars"].values():
        assert (card / rel_path).exists(), rel_path

    source_audit = pd.read_csv(card / summary["sidecars"]["source_audit"])
    assert set(source_audit["symbol"]) == requested_symbols
    assert {"status", "http_status", "row_count", "message"}.issubset(source_audit.columns)
    coverage_audit = pd.read_csv(card / summary["sidecars"]["coverage_audit"])
    assert set(coverage_audit["symbol"]).issubset(requested_symbols)
    assert {"coverage_start", "coverage_end", "regular_session_rows"}.issubset(coverage_audit.columns)

    if summary["provider_uri"]:
        provider_uri = Path(summary["provider_uri"])
        manifest = json.loads((provider_uri / "qlib_store_manifest.json").read_text(encoding="utf-8"))
        assert manifest["freq"] == "15min"
        assert manifest["source"] == "polygon_aggs_custom_bars"
        assert set(manifest["instruments"]) == set(summary["available_symbols"])
        assert "$vwap" in manifest["fields"]
        assert Path(summary["sidecars"]["qlib_store_manifest"]).name == "qlib_store_manifest.json"

    recorder_dir = Path(summary["recorder_local_dir"])
    assert recorder_dir.exists()
    assert summary["recorder_id"] in notebook_text
    assert "run_intraday_external_context_data_card" in notebook_text
    assert "build_qlib_store_from_frame" in notebook_text
    assert "R.start" in notebook_text
    assert "外部 15m 上下文" in conclusion


def test_agentic_alpha_intraday_external_context_label_card_records_external_features_and_partial_gate():
    card = ROOT / "research" / "agentic_alpha_intraday_external_context_label_v1"
    summary = _summary(card)
    notebook_text = (card / "agentic_alpha_intraday_external_context_label_v1.ipynb").read_text(encoding="utf-8")
    conclusion = (card / "conclusion.md").read_text(encoding="utf-8")

    external_symbols = {"QQQ", "SMH", "SOXX", "NVDA", "AVGO", "AMD", "TSM", "ASML"}
    assert summary["experiment"] == "agentic_alpha_intraday_external_context_label_v1"
    assert summary["card_type"] == "intraday_execution_label_model_search"
    assert summary["promotion_decision"] == "not_promoted_external_history_insufficient_for_required_walk_forward"
    assert Path(summary["provider_uris"]["15min"]) == (ROOT / "data" / "qlib" / "soxl_intraday_15min").resolve()
    assert Path(summary["provider_uris"]["external_15min"]) == (
        ROOT / "data" / "qlib" / "soxl_external_intraday_15min"
    ).resolve()
    assert summary["external_intraday_context"]["provider_card"] == "research/intraday_external_context_data_v1"
    assert set(summary["external_intraday_context"]["available_symbols"]) == external_symbols
    assert summary["external_intraday_context"]["missing_requested_symbols"] == []
    assert summary["external_intraday_context"]["required_walk_forward_satisfied"] is False
    assert summary["external_intraday_context"]["data_status"] == "ok_partial_insufficient_for_full_walk_forward"
    assert summary["feature_set"]["name"] == "execution_context_external_v2"
    assert summary["feature_set"]["lookback_only"] is True
    assert "qqq_smh_soxx_lead_lag_3" in summary["feature_set"]["external_features"]
    assert "semis_breadth_mom3_pos" in summary["feature_set"]["external_features"]
    assert "semis_equal_weight_mom3" in summary["feature_set"]["external_features"]
    assert summary["model_selection"]["selection_segment"] == "valid"
    assert summary["model_selection"]["report_segment"] == "test"
    assert summary["model_selection"]["selection_uses_test_segment"] is False
    assert set(summary["model_selection"]["models"]) == {"lgb", "xgb", "lgb_xgb_consensus"}
    assert summary["walk_forward"]["required_policy_satisfied"] is False
    assert summary["artifact_policy"]["policy"] == "slim_card_recorder_first"
    assert summary["artifact_policy"]["recorder_only_artifacts"] == {
        "predictions": "artifacts/pred.pkl",
        "labels": "artifacts/label.pkl",
        "all_model_predictions": "artifacts/model_selection/all_model_predictions.csv",
    }
    assert RECORDER_ONLY_ROOT_FILES.isdisjoint({path.name for path in card.iterdir() if path.is_file()})

    expected_sidecars = {
        "analysis_summary",
        "backtest_walk_forward_metrics",
        "cost_sensitivity",
        "failure_modes",
        "feature_importance",
        "label_diagnostics",
        "model_selection",
        "selected_model",
    }
    assert set(summary["sidecars"]) == expected_sidecars
    for rel_path in summary["sidecars"].values():
        assert (card / rel_path).exists(), rel_path

    model_selection = pd.read_csv(card / summary["sidecars"]["model_selection"])
    assert set(model_selection["model"]) == {"lgb", "xgb", "lgb_xgb_consensus"}
    feature_importance = pd.read_csv(card / summary["sidecars"]["feature_importance"])
    assert feature_importance["feature"].str.contains("semis_|qqq_|smh_|external").any()

    recorder_dir = Path(summary["recorder_local_dir"])
    assert (recorder_dir / "artifacts" / "pred.pkl").exists()
    assert (recorder_dir / "artifacts" / "label.pkl").exists()
    assert (recorder_dir / "artifacts" / "model_selection" / "all_model_predictions.csv").exists()
    assert summary["recorder_id"] in notebook_text
    assert "run_intraday_execution_label_model_card" in notebook_text
    assert "external_provider_uri" in notebook_text
    assert "R.start" in notebook_text
    assert "外部 15m 上下文" in conclusion
    assert "不晋升" in conclusion


def test_agentic_alpha_high_chop_execution_validation_card_records_minute_execution_gap():
    card = ROOT / "research" / "agentic_alpha_high_chop_execution_validation_v1"
    summary = _summary(card)
    notebook_text = (card / "agentic_alpha_high_chop_execution_validation_v1.ipynb").read_text(encoding="utf-8")
    conclusion = (card / "conclusion.md").read_text(encoding="utf-8")

    assert summary["experiment"] == "agentic_alpha_high_chop_execution_validation_v1"
    assert summary["card_type"] == "execution_validation"
    assert summary["promotion_decision"] == "execution_not_proven"
    assert summary["as_of"] == "2026-07-02"
    assert summary["source_strategy"]["card"] == "research/agentic_alpha_high_chop_lgb_xgb_v1"
    assert summary["validation_summary"]["latest"]["date"] == "2026-07-02"
    assert summary["validation_summary"]["latest"]["execution_status"] == "near_cash"
    assert summary["validation_summary"]["latest"]["target_position"] < 0.01
    assert summary["validation_summary"]["latest"]["rth_low_from_open"] < -0.20
    assert summary["validation_summary"]["risk_checks"]["near_cash_days_recent"] == 5
    assert "futu.OpenQuoteContext.request_history_kline(K_1M, extended_time=True, session=ALL)" in summary[
        "external_data_api"
    ]
    assert summary["qlib_native_api"] == ["qlib.init", "qlib.data.D.features", "qlib.workflow.R.start"]
    assert summary["artifact_policy"]["policy"] == "slim_card_recorder_first"
    assert set(summary["sidecars"]) == {
        "execution_validation",
        "execution_validation_summary",
        "failure_modes",
    }
    for rel_path in summary["sidecars"].values():
        assert (card / rel_path).exists(), rel_path
    assert RECORDER_ONLY_ROOT_FILES.isdisjoint({path.name for path in card.iterdir() if path.is_file()})

    validation_csv = (card / summary["sidecars"]["execution_validation"]).read_text(encoding="utf-8")
    assert "afterhours_mae_from_rth_close" in validation_csv
    assert "2026-07-02" in validation_csv
    assert summary["recorder_id"] in notebook_text
    assert "fetch_futu_intraday_bars" in notebook_text
    assert "D.features" in notebook_text
    assert "R.start" in notebook_text
    assert "execution_not_proven" in conclusion


def test_agentic_alpha_budget_overlay_search_card_closes_cppi_tipp_branch():
    card = ROOT / "research" / "agentic_alpha_budget_overlay_search_v1"
    summary = _summary(card)
    notebook_text = (card / "agentic_alpha_budget_overlay_search_v1.ipynb").read_text(encoding="utf-8")
    conclusion = (card / "conclusion.md").read_text(encoding="utf-8")

    assert summary["experiment"] == "agentic_alpha_budget_overlay_search_v1"
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
        "production_signal_budget_overlay",
        "tipp_budget_overlay",
    }
    hard_gate = summary["branch_results"]["production_signal_budget_overlay"]
    tipp = summary["branch_results"]["tipp_budget_overlay"]
    assert hard_gate["best_return_pct"] > summary["baseline"]["return_pct"]
    assert hard_gate["best_return_max_drawdown_pct"] < -20.0
    assert hard_gate["best_drawdown_gate_return_pct"] < summary["baseline"]["return_pct"]
    assert hard_gate["best_drawdown_gate_max_drawdown_pct"] >= -20.0
    assert tipp["best_drawdown_gate_return_pct"] < summary["baseline"]["return_pct"]
    assert tipp["best_drawdown_gate_max_drawdown_pct"] >= -20.0
    assert summary["external_source_counts"] == {"arxiv": 1, "github": 1, "paper": 2, "twitter": 1}
    assert summary["qlib_docs"] == {
        "data": "https://qlib.readthedocs.io/en/latest/component/data.html",
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
    assert "production_signal_budget_overlay" in candidates
    assert "tipp_budget_overlay" in candidates
    assert "drawdown_gate" in candidates
    assert "return_gate" in candidates
    assert summary["recorder_id"] in notebook_text
    assert "CPPI" in notebook_text
    assert "TIPP" in notebook_text
    assert "pred.pkl" in notebook_text
    assert "D.features" in notebook_text
    assert "R.start" in notebook_text
    assert "未找到同时满足" in conclusion


def test_agentic_alpha_ml_pair_search_card_closes_crash_model_and_pair_hedge_branches():
    card = ROOT / "research" / "agentic_alpha_ml_pair_search_v1"
    summary = _summary(card)
    notebook_text = (card / "agentic_alpha_ml_pair_search_v1.ipynb").read_text(encoding="utf-8")
    conclusion = (card / "conclusion.md").read_text(encoding="utf-8")

    assert summary["experiment"] == "agentic_alpha_ml_pair_search_v1"
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
        "soxl_soxx_pair_hedge",
        "walk_forward_crash_return_ensemble",
    }
    crash_model = summary["branch_results"]["walk_forward_crash_return_ensemble"]
    pair_hedge = summary["branch_results"]["soxl_soxx_pair_hedge"]
    assert crash_model["best_return_pct"] > summary["baseline"]["return_pct"]
    assert crash_model["best_return_max_drawdown_pct"] < -20.0
    assert crash_model["best_drawdown_gate_return_pct"] < summary["baseline"]["return_pct"]
    assert crash_model["best_drawdown_gate_max_drawdown_pct"] >= -20.0
    assert pair_hedge["best_return_pct"] > summary["baseline"]["return_pct"]
    assert pair_hedge["best_return_max_drawdown_pct"] < -20.0
    assert pair_hedge["best_drawdown_gate_return_pct"] < summary["baseline"]["return_pct"]
    assert pair_hedge["best_drawdown_gate_max_drawdown_pct"] >= -20.0
    assert summary["external_source_counts"] == {"arxiv": 1, "github": 1, "paper": 3, "twitter": 1}
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
    assert "walk_forward_crash_return_ensemble" in candidates
    assert "soxl_soxx_pair_hedge" in candidates
    assert "drawdown_gate" in candidates
    assert "return_gate" in candidates
    assert summary["recorder_id"] in notebook_text
    assert "LGBMRegressor" in notebook_text
    assert "LGBMClassifier" in notebook_text
    assert "SOXX" in notebook_text
    assert "D.features" in notebook_text
    assert "R.start" in notebook_text
    assert "未找到同时满足" in conclusion


def test_agentic_alpha_tail_risk_external_search_card_records_external_store_and_goal_gate():
    card = ROOT / "research" / "agentic_alpha_tail_risk_external_search_v1"
    summary = _summary(card)
    notebook_text = (card / "agentic_alpha_tail_risk_external_search_v1.ipynb").read_text(encoding="utf-8")
    conclusion = (card / "conclusion.md").read_text(encoding="utf-8")

    assert summary["experiment"] == "agentic_alpha_tail_risk_external_search_v1"
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
    assert summary["external_data"]["source_fields"] == ["VIX9D", "VIX3M", "VVIX", "SKEW"]
    assert summary["external_data"]["source_coverage_end"] == "2026-04-17"
    assert summary["external_data"]["candidate_coverage_end"] == "2026-04-16"
    assert summary["external_data"]["provider_manifest"] == "data/qlib/soxl_tail_risk_external_v1/qlib_store_manifest.json"
    assert (ROOT / summary["external_data"]["provider_manifest"]).exists()
    assert set(summary["branch_results"]) == {
        "tail_risk_term_structure_ensemble",
        "vol_target_tail_risk_throttle",
    }
    for result in summary["branch_results"].values():
        assert result["best_return_pct"] < summary["baseline"]["return_pct"]
        assert result["best_return_max_drawdown_pct"] < -20.0
        assert result["best_drawdown_gate_return_pct"] < summary["baseline"]["return_pct"]
        assert result["best_drawdown_gate_max_drawdown_pct"] >= -20.0
    assert summary["external_source_counts"] == {"arxiv": 1, "github": 1, "paper": 4, "twitter": 1}
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
        "qlib_store_manifest",
        "search_summary",
    }
    for rel_path in summary["sidecars"].values():
        assert (card / rel_path).exists(), rel_path

    candidates = (card / summary["sidecars"]["candidate_metrics"]).read_text(encoding="utf-8")
    assert "tail_risk_term_structure_ensemble" in candidates
    assert "vol_target_tail_risk_throttle" in candidates
    assert "drawdown_gate" in candidates
    assert "return_gate" in candidates
    assert summary["recorder_id"] in notebook_text
    assert "build_qlib_store_from_frame" in notebook_text
    assert "VIX9D" in notebook_text
    assert "VIX3M" in notebook_text
    assert "VVIX" in notebook_text
    assert "SKEW" in notebook_text
    assert "D.features" in notebook_text
    assert "R.start" in notebook_text
    assert "未找到同时满足" in conclusion


def test_agentic_alpha_cross_asset_rotation_search_card_records_rotation_store_and_goal_gate():
    card = ROOT / "research" / "agentic_alpha_cross_asset_rotation_search_v1"
    summary = _summary(card)
    notebook_text = (card / "agentic_alpha_cross_asset_rotation_search_v1.ipynb").read_text(encoding="utf-8")
    conclusion = (card / "conclusion.md").read_text(encoding="utf-8")

    assert summary["experiment"] == "agentic_alpha_cross_asset_rotation_search_v1"
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
    assert summary["cross_asset_data"]["symbols"] == ["SOXL", "TQQQ", "TECL", "SPXL", "UPRO"]
    assert summary["cross_asset_data"]["source_coverage_end"] == "2026-04-15"
    assert summary["cross_asset_data"]["candidate_coverage_end"] == "2026-04-14"
    assert summary["cross_asset_data"]["provider_manifest"] == "data/qlib/soxl_cross_asset_rotation_v1/qlib_store_manifest.json"
    assert (ROOT / summary["cross_asset_data"]["provider_manifest"]).exists()
    assert set(summary["branch_results"]) == {
        "cross_asset_breadth_soxl",
        "relative_momentum_rotation",
    }
    rotation = summary["branch_results"]["relative_momentum_rotation"]
    breadth = summary["branch_results"]["cross_asset_breadth_soxl"]
    assert rotation["best_return_pct"] < summary["baseline"]["return_pct"]
    assert rotation["best_return_max_drawdown_pct"] < -20.0
    assert rotation["best_drawdown_gate_return_pct"] < summary["baseline"]["return_pct"]
    assert rotation["best_drawdown_gate_max_drawdown_pct"] >= -20.0
    assert breadth["best_return_pct"] < summary["baseline"]["return_pct"]
    assert breadth["best_return_max_drawdown_pct"] < -20.0
    assert breadth["best_drawdown_gate_available"] is False
    assert summary["external_source_counts"] == {"arxiv": 1, "github": 2, "paper": 3, "twitter": 1}
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
        "qlib_store_manifest",
        "search_summary",
    }
    for rel_path in summary["sidecars"].values():
        assert (card / rel_path).exists(), rel_path

    candidates = (card / summary["sidecars"]["candidate_metrics"]).read_text(encoding="utf-8")
    assert "relative_momentum_rotation" in candidates
    assert "cross_asset_breadth_soxl" in candidates
    assert "drawdown_gate" in candidates
    assert "return_gate" in candidates
    assert summary["recorder_id"] in notebook_text
    assert "build_qlib_store_from_frame" in notebook_text
    assert "TQQQ" in notebook_text
    assert "SPXL" in notebook_text
    assert "D.features" in notebook_text
    assert "R.start" in notebook_text
    assert "未找到同时满足" in conclusion


def test_agentic_alpha_factor_library_reversal_search_card_closes_factor_library_branches():
    card = ROOT / "research" / "agentic_alpha_factor_library_reversal_search_v1"
    summary = _summary(card)
    notebook_text = (card / "agentic_alpha_factor_library_reversal_search_v1.ipynb").read_text(encoding="utf-8")
    conclusion = (card / "conclusion.md").read_text(encoding="utf-8")

    assert summary["experiment"] == "agentic_alpha_factor_library_reversal_search_v1"
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
    assert summary["data_range"] == ["2019-01-02", "2026-06-09"]
    assert summary["feature_warmup_range"] == ["2010-03-11", "2026-06-10"]
    assert set(summary["branch_results"]) == {
        "factor_library_reversal_reentry",
        "factor_library_trend_sell_gate",
        "factor_library_vol_managed_score",
    }
    assert summary["branch_results"]["factor_library_reversal_reentry"]["coverage_end"] == "2026-06-09"
    assert summary["branch_results"]["factor_library_trend_sell_gate"]["best_drawdown_gate_return_pct"] < summary[
        "baseline"
    ]["return_pct"]
    assert summary["branch_results"]["factor_library_vol_managed_score"]["best_return_pct"] > summary["baseline"][
        "return_pct"
    ]
    assert summary["branch_results"]["factor_library_vol_managed_score"]["best_return_max_drawdown_pct"] < -20.0
    assert summary["factor_library"]["source_db"] == "data/soxl_factor_library.db"
    assert "RB_RSI6_OVERSOLD" in summary["factor_library"]["selected_features"]
    assert summary["factor_library"]["candidate_coverage_end"] == "2026-06-09"
    assert summary["external_source_counts"] == {"arxiv": 1, "github": 1, "paper": 2, "twitter": 1}
    assert summary["qlib_native_api"] == ["qlib.init", "qlib.data.D.features", "qlib.workflow.R.start"]
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
    assert "factor_library_reversal_reentry" in candidates
    assert "factor_library_trend_sell_gate" in candidates
    assert "factor_library_vol_managed_score" in candidates
    assert "drawdown_gate" in candidates
    assert "return_gate" in candidates
    assert summary["recorder_id"] in notebook_text
    assert "D.features" in notebook_text
    assert "R.start" in notebook_text
    assert "data/soxl_factor_library.db" in notebook_text
    assert "未找到同时满足" in conclusion


def test_agentic_alpha_long_short_crisis_alpha_search_card_closes_trend_following_branches():
    card = ROOT / "research" / "agentic_alpha_long_short_crisis_alpha_search_v1"
    summary = _summary(card)
    notebook_text = (card / "agentic_alpha_long_short_crisis_alpha_search_v1.ipynb").read_text(encoding="utf-8")
    conclusion = (card / "conclusion.md").read_text(encoding="utf-8")

    assert summary["experiment"] == "agentic_alpha_long_short_crisis_alpha_search_v1"
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
    assert summary["data_range"] == ["2019-01-02", "2026-06-09"]
    assert set(summary["branch_results"]) == {
        "crisis_alpha_overlay",
        "long_short_time_series_momentum",
    }
    for result in summary["branch_results"].values():
        assert result["coverage_end"] == "2026-06-09"
        assert result["passes_both_count"] == 0
        assert result["best_drawdown_gate_available"] is False
        assert result["best_return_pct"] < summary["baseline"]["return_pct"]
        assert result["best_return_max_drawdown_pct"] < -20.0
    assert summary["long_short_crisis_alpha"]["selected_features"] == [
        "SOXL_TREND_60",
        "SOXL_TREND_120",
        "SOXL_TREND_200",
        "SOXX_TREND_60",
        "SOXL_SOXX_RATIO_MOM_20",
        "SOXL_VIX",
        "SOXL_HV20",
    ]
    assert summary["external_source_counts"] == {"github": 1, "paper": 3, "twitter": 1}
    assert summary["qlib_native_api"] == ["qlib.init", "qlib.data.D.features", "qlib.workflow.R.start"]
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
    assert "long_short_time_series_momentum" in candidates
    assert "crisis_alpha_overlay" in candidates
    assert "drawdown_gate" in candidates
    assert summary["recorder_id"] in notebook_text
    assert "D.features" in notebook_text
    assert "R.start" in notebook_text
    assert "Time Series Momentum" in notebook_text
    assert "未找到同时满足" in conclusion


def test_agentic_alpha_canary_sprint_search_card_closes_short_hold_branches():
    card = ROOT / "research" / "agentic_alpha_canary_sprint_search_v1"
    summary = _summary(card)
    notebook_text = (card / "agentic_alpha_canary_sprint_search_v1.ipynb").read_text(encoding="utf-8")
    conclusion = (card / "conclusion.md").read_text(encoding="utf-8")

    assert summary["experiment"] == "agentic_alpha_canary_sprint_search_v1"
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
    assert summary["data_range"] == ["2019-01-02", "2026-06-09"]
    assert set(summary["branch_results"]) == {
        "canary_sprint_short_hold",
        "dip_sprint_short_hold",
        "guarded_sprint_budget",
    }
    for result in summary["branch_results"].values():
        assert result["coverage_end"] == "2026-06-09"
        assert result["passes_both_count"] == 0
        assert result["best_drawdown_gate_available"] is True
        assert result["best_drawdown_gate_return_pct"] < summary["baseline"]["return_pct"]
        assert result["best_drawdown_gate_max_drawdown_pct"] >= -20.0
    assert summary["canary_sprint"]["selected_features"] == [
        "SOXX_TREND_20",
        "SOXX_TREND_60",
        "SOXL_DIP_2",
        "SOXL_VIX",
        "SOXL_AUTOCORR_10",
        "SOXL_HV20",
    ]
    assert summary["external_source_counts"] == {"arxiv": 1, "github": 1, "paper": 3, "twitter": 1}
    assert summary["qlib_native_api"] == ["qlib.init", "qlib.data.D.features", "qlib.workflow.R.start"]
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
    assert "canary_sprint_short_hold" in candidates
    assert "dip_sprint_short_hold" in candidates
    assert "guarded_sprint_budget" in candidates
    assert "drawdown_gate" in candidates
    assert "return_gate" in candidates
    assert summary["recorder_id"] in notebook_text
    assert "D.features" in notebook_text
    assert "R.start" in notebook_text
    assert "Compounding Effects in Leveraged ETFs" in notebook_text
    assert "未找到同时满足" in conclusion


def test_agentic_alpha_meta_label_probability_gate_card_records_sanity_gate():
    card = ROOT / "research" / "agentic_alpha_meta_label_probability_gate_search_v1"
    summary = _summary(card)
    notebook_text = (card / "agentic_alpha_meta_label_probability_gate_search_v1.ipynb").read_text(encoding="utf-8")
    conclusion = (card / "conclusion.md").read_text(encoding="utf-8")

    assert summary["experiment"] == "agentic_alpha_meta_label_probability_gate_search_v1"
    assert summary["card_type"] == "diagnostic_search"
    assert summary["promotion_decision"] == "not_promoted"
    assert summary["search_report"]["goal_achieved"] is False
    assert summary["search_report"]["passes_both_count"] == 0
    assert summary["search_report"]["raw_passes_both_count"] >= 1
    assert summary["baseline"]["name"] == "qlib_paper_sim_v1"
    assert summary["goal_gate"] == {
        "max_drawdown_floor": -20.0,
        "require_return_above_baseline": True,
        "max_exposure_sanity": 10.0,
        "min_trade_count_sanity": 20,
    }
    assert set(summary["branch_results"]) == {"meta_label_probability_gate"}
    branch = summary["branch_results"]["meta_label_probability_gate"]
    assert branch["raw_passes_both_count"] >= 1
    assert branch["passes_both_count"] == 0
    assert branch["best_drawdown_gate_max_exposure"] > summary["goal_gate"]["max_exposure_sanity"]
    assert summary["meta_label_probability_gate"]["model_families"] == ["LogisticRegression", "HistGradientBoosting"]
    assert summary["external_source_counts"] == {"arxiv": 1, "github": 2, "paper": 1}
    assert summary["qlib_native_api"] == ["qlib.init", "qlib.data.D.features", "qlib.workflow.R.start"]
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
    assert "meta_label_probability_gate" in candidates
    assert "exposure_sanity_gate" in candidates
    assert summary["recorder_id"] in notebook_text
    assert "D.features" in notebook_text
    assert "R.start" in notebook_text
    assert "meta-labeling" in notebook_text
    assert "极端暴露" in conclusion


def test_agentic_alpha_session_effect_search_card_closes_overnight_intraday_branches():
    card = ROOT / "research" / "agentic_alpha_session_effect_search_v1"
    summary = _summary(card)
    notebook_text = (card / "agentic_alpha_session_effect_search_v1.ipynb").read_text(encoding="utf-8")
    conclusion = (card / "conclusion.md").read_text(encoding="utf-8")

    assert summary["experiment"] == "agentic_alpha_session_effect_search_v1"
    assert summary["card_type"] == "diagnostic_search"
    assert summary["promotion_decision"] == "not_promoted"
    assert summary["search_report"]["goal_achieved"] is False
    assert summary["search_report"]["passes_both_count"] == 0
    assert summary["baseline"]["name"] == "qlib_paper_sim_v1"
    assert summary["goal_gate"] == {
        "max_drawdown_floor": -20.0,
        "require_return_above_baseline": True,
    }
    assert summary["data_range"] == ["2019-01-02", "2026-06-09"]
    assert set(summary["branch_results"]) == {
        "intraday_gap_reversal",
        "overnight_momentum",
        "tug_of_war_close_to_close",
    }
    for result in summary["branch_results"].values():
        assert result["coverage_end"] == "2026-06-09"
        assert result["passes_both_count"] == 0
    assert summary["session_effect"]["selected_features"] == [
        "SOXL_OVERNIGHT_RETURN",
        "SOXL_INTRADAY_RETURN",
        "SOXL_TUG_OF_WAR",
        "SOXX_TREND",
        "SOXL_VIX",
        "SOXL_RANGE",
    ]
    assert summary["external_source_counts"] == {"arxiv": 1, "github": 2, "paper": 2, "twitter": 1}
    assert summary["qlib_native_api"] == ["qlib.init", "qlib.data.D.features", "qlib.workflow.R.start"]
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
    assert "overnight_momentum" in candidates
    assert "intraday_gap_reversal" in candidates
    assert "tug_of_war_close_to_close" in candidates
    assert summary["recorder_id"] in notebook_text
    assert "D.features" in notebook_text
    assert "R.start" in notebook_text
    assert "overnight" in notebook_text
    assert "未找到同时满足" in conclusion


def test_agentic_alpha_macro_credit_search_card_closes_macro_credit_branches():
    card = ROOT / "research" / "agentic_alpha_macro_credit_search_v1"
    summary = _summary(card)
    notebook_text = (card / "agentic_alpha_macro_credit_search_v1.ipynb").read_text(encoding="utf-8")
    conclusion = (card / "conclusion.md").read_text(encoding="utf-8")

    assert summary["experiment"] == "agentic_alpha_macro_credit_search_v1"
    assert summary["card_type"] == "diagnostic_search"
    assert summary["promotion_decision"] == "not_promoted"
    assert summary["search_report"]["goal_achieved"] is False
    assert summary["search_report"]["passes_both_count"] == 0
    assert summary["baseline"]["name"] == "qlib_paper_sim_v1"
    assert summary["goal_gate"] == {
        "max_drawdown_floor": -20.0,
        "require_return_above_baseline": True,
    }
    assert summary["data_range"] == ["2019-01-02", "2026-06-09"]
    assert set(summary["branch_results"]) == {
        "macro_composite_throttle",
        "macro_credit_rate_gate",
        "trend_drawdown_stop",
    }
    for result in summary["branch_results"].values():
        assert result["coverage_end"] == "2026-06-08"
        assert result["passes_both_count"] == 0
    assert summary["macro_credit"]["selected_features"] == [
        "SOXL_CLOSE",
        "SOXX_TREND",
        "YIELD_CURVE",
        "CREDIT_SPREAD",
        "BAA_YIELD",
        "DXY",
        "OIL",
    ]
    assert summary["macro_credit_data"]["provider_manifest"] == (
        "data/qlib/soxl_macro_credit_v1/qlib_store_manifest.json"
    )
    assert (ROOT / summary["macro_credit_data"]["provider_manifest"]).exists()
    assert summary["external_source_counts"] == {"arxiv": 1, "github": 2, "paper": 3, "twitter": 1}
    assert summary["qlib_native_api"] == [
        "qlib.init",
        "qlib.data.D.features",
        "qlib.workflow.R.start",
        "build_qlib_store_from_frame",
    ]
    assert summary["artifact_policy"]["policy"] == "slim_card_recorder_first"
    assert set(summary["sidecars"]) == {
        "candidate_metrics",
        "external_sources",
        "failure_modes",
        "qlib_store_manifest",
        "search_summary",
    }
    for rel_path in summary["sidecars"].values():
        assert (card / rel_path).exists(), rel_path

    candidates = (card / summary["sidecars"]["candidate_metrics"]).read_text(encoding="utf-8")
    assert "macro_credit_rate_gate" in candidates
    assert "macro_composite_throttle" in candidates
    assert "trend_drawdown_stop" in candidates
    assert summary["recorder_id"] in notebook_text
    assert "D.features" in notebook_text
    assert "R.start" in notebook_text
    assert "credit spread" in notebook_text
    assert "未找到同时满足" in conclusion


def test_agentic_alpha_ensemble_search_card_records_passing_candidate():
    card = ROOT / "research" / "agentic_alpha_ensemble_search_v1"
    summary = _summary(card)
    notebook_text = (card / "agentic_alpha_ensemble_search_v1.ipynb").read_text(encoding="utf-8")
    conclusion = (card / "conclusion.md").read_text(encoding="utf-8")

    assert summary["experiment"] == "agentic_alpha_ensemble_search_v1"
    assert summary["card_type"] == "validated_search"
    assert summary["promotion_decision"] == "promote_to_production_review"
    assert summary["search_report"]["goal_achieved"] is True
    assert summary["search_report"]["best_candidate"] == "agentic_multi_branch_ensemble_v1"
    assert summary["search_report"]["passes_both_count"] >= 1
    assert summary["baseline"]["name"] == "qlib_paper_sim_v1"
    assert summary["goal_gate"] == {
        "max_drawdown_floor": -20.0,
        "require_return_above_baseline": True,
    }
    assert summary["data_range"] == ["2019-01-02", "2026-06-09"]
    selected = summary["selected_candidate"]
    assert selected["candidate"] == "agentic_multi_branch_ensemble_v1"
    assert selected["return_pct"] > summary["baseline"]["return_pct"]
    assert selected["max_drawdown_pct"] >= summary["goal_gate"]["max_drawdown_floor"]
    assert selected["component_count"] == 6
    assert set(summary["component_branches"]) == {
        "canary_sprint_short_hold",
        "macro_credit_rate_gate",
        "signed_overnight_momentum",
        "tsmom_stop",
    }
    assert summary["external_source_counts"] == {"arxiv": 1, "github": 2, "paper": 5, "twitter": 1}
    assert summary["qlib_native_api"] == ["qlib.init", "qlib.data.D.features", "qlib.workflow.R.start"]
    assert summary["artifact_policy"]["policy"] == "slim_card_recorder_first"
    assert set(summary["sidecars"]) == {
        "candidate_metrics",
        "component_summary",
        "external_sources",
        "failure_modes",
        "search_summary",
        "selected_model",
    }
    for rel_path in summary["sidecars"].values():
        assert (card / rel_path).exists(), rel_path

    candidates = (card / summary["sidecars"]["candidate_metrics"]).read_text(encoding="utf-8")
    selected_model = json.loads((card / summary["sidecars"]["selected_model"]).read_text(encoding="utf-8"))
    assert "agentic_multi_branch_ensemble_v1" in candidates
    assert "beats_baseline_and_drawdown_gate" in candidates
    assert selected_model["candidate"] == "agentic_multi_branch_ensemble_v1"
    assert len(selected_model["weights"]) == 6
    assert summary["recorder_id"] in notebook_text
    assert "D.features" in notebook_text
    assert "R.start" in notebook_text
    assert "evaluate_agentic_ensemble_candidates" in notebook_text
    assert "同时满足收益与回撤门禁" in conclusion


def test_agentic_alpha_position_sizing_search_card_compares_managed_position_budget():
    card = ROOT / "research" / "agentic_alpha_position_sizing_search_v1"
    summary = _summary(card)
    notebook_text = (card / "agentic_alpha_position_sizing_search_v1.ipynb").read_text(encoding="utf-8")
    conclusion = (card / "conclusion.md").read_text(encoding="utf-8")

    assert summary["experiment"] == "agentic_alpha_position_sizing_search_v1"
    assert summary["card_type"] == "position_sizing_search"
    assert summary["promotion_decision"] == "iterate_not_promoted_until_raw_position_reference_exceeded"
    assert summary["goal_gate"]["max_drawdown_floor"] == -20.0
    assert summary["goal_gate"]["require_return_above_soxlbacktest"] is True
    assert summary["goal_gate"]["require_return_above_raw_position_reference"] is True
    assert summary["data_range"] == ["2019-01-02", "2026-06-09"]
    assert summary["report_window"] == ["2026-01-02", "2026-06-09"]
    selected = summary["selected_candidate"]
    raw = summary["raw_position_reference"]
    legacy = summary["legacy_soxlbacktest_benchmark"]
    assert selected["candidate"] == "cap1_h0.2_chgnone_hold1"
    assert selected["max_position"] <= 1.0
    assert selected["days_above_100"] == 0
    assert selected["return_pct"] > legacy["rulebase_return_pct"]
    assert selected["max_drawdown_pct"] >= summary["goal_gate"]["max_drawdown_floor"]
    assert selected["return_pct"] < raw["return_pct"]
    assert selected["next_day_reversal_count"] < raw["next_day_reversal_count"]
    assert summary["model_position_gate"]["decision"] == "not_primary"
    assert summary["qlib_native_api"] == ["qlib.init", "qlib.data.D.features", "qlib.workflow.R.start"]
    assert summary["artifact_policy"]["policy"] == "slim_card_recorder_first"
    assert set(summary["sidecars"]) == {
        "backtest_benchmark_comparison",
        "candidate_metrics",
        "external_sources",
        "failure_modes",
        "position_sizing_summary",
        "selected_model",
    }
    for rel_path in summary["sidecars"].values():
        assert (card / rel_path).exists(), rel_path

    assert summary["recorder_id"] in notebook_text
    assert "D.features" in notebook_text
    assert "R.start" in notebook_text
    assert "evaluate_position_sizing_policies" in notebook_text
    assert "未超过 raw target-position reference" in conclusion


def test_agentic_alpha_constrained_weight_search_card_exceeds_raw_position_reference():
    card = ROOT / "research" / "agentic_alpha_constrained_weight_search_v1"
    summary = _summary(card)
    notebook_text = (card / "agentic_alpha_constrained_weight_search_v1.ipynb").read_text(encoding="utf-8")
    conclusion = (card / "conclusion.md").read_text(encoding="utf-8")

    assert summary["experiment"] == "agentic_alpha_constrained_weight_search_v1"
    assert summary["card_type"] == "validated_constrained_weight_search"
    assert summary["promotion_decision"] == "promote_to_production_review"
    assert summary["goal_gate"]["max_drawdown_floor"] == -20.0
    assert summary["goal_gate"]["require_return_above_raw_position_reference"] is True
    assert summary["data_range"] == ["2019-01-02", "2026-06-09"]
    assert summary["report_window"] == ["2026-01-02", "2026-06-09"]
    selected = summary["selected_candidate"]
    raw = summary["raw_position_reference"]
    assert selected["candidate"] == "canary_tsmom_budget_h025"
    assert selected["return_pct"] > raw["return_pct"]
    assert selected["max_drawdown_pct"] >= summary["goal_gate"]["max_drawdown_floor"]
    assert selected["max_position"] <= 1.0
    assert selected["min_position"] >= 0.0
    assert selected["effective_component_count"] >= 2
    assert summary["legacy_soxlbacktest_benchmark"]["rulebase_return_pct"] < selected["return_pct"]
    assert summary["qlib_native_api"] == ["qlib.init", "qlib.data.D.features", "qlib.workflow.R.start"]
    assert summary["artifact_policy"]["policy"] == "slim_card_recorder_first"
    assert set(summary["sidecars"]) == {
        "backtest_benchmark_comparison",
        "candidate_metrics",
        "constrained_weight_summary",
        "failure_modes",
        "selected_model",
    }
    for rel_path in summary["sidecars"].values():
        assert (card / rel_path).exists(), rel_path

    assert summary["recorder_id"] in notebook_text
    assert "D.features" in notebook_text
    assert "R.start" in notebook_text
    assert "evaluate_weighted_position_candidates" in notebook_text
    assert "超过 raw target-position reference" in conclusion


def test_agentic_alpha_ml_position_model_card_reports_validation_selected_underperformance():
    card = ROOT / "research" / "agentic_alpha_ml_position_model_v1"
    summary = _summary(card)
    notebook_text = (card / "agentic_alpha_ml_position_model_v1.ipynb").read_text(encoding="utf-8")
    conclusion = (card / "conclusion.md").read_text(encoding="utf-8")

    assert summary["experiment"] == "agentic_alpha_ml_position_model_v1"
    assert summary["card_type"] == "ml_position_model_search"
    assert summary["promotion_decision"] == "not_promoted_validation_selected_ml_underperformed"
    assert summary["segments"] == {
        "train": ["2019-01-02", "2024-12-31"],
        "validation": ["2025-01-02", "2025-12-31"],
        "test": ["2026-01-02", "2026-06-09"],
    }
    assert summary["goal_gate"]["selection_segment"] == "validation"
    assert summary["goal_gate"]["report_segment"] == "test"
    assert summary["goal_gate"]["max_position"] == 1.0
    assert summary["goal_gate"]["min_validation_trade_count"] == 10
    selected = summary["selected_candidate"]
    constrained = summary["constrained_weight_reference"]
    legacy = summary["legacy_soxlbacktest_benchmark"]
    oracle = summary["oracle_test_best_candidate"]
    assert selected["selected_by"] == "validation_selection_objective_score"
    assert selected["test"]["max_position"] <= 1.0
    assert selected["test"]["days_above_100"] == 0
    assert selected["test"]["return_pct"] < legacy["rulebase_return_pct"]
    assert selected["test"]["return_pct"] < constrained["return_pct"]
    assert oracle["return_pct"] < legacy["rulebase_return_pct"]
    assert summary["ml_position_report"]["candidate_count"] >= 100
    assert set(summary["ml_position_report"]["model_families"]) == {
        "HistGradientBoostingClassifier",
        "HistGradientBoostingRegressor",
        "LGBMClassifier",
        "LGBMRegressor",
        "LogisticRegression",
        "Ridge",
    }
    assert summary["qlib_native_api"] == ["qlib.init", "qlib.data.D.features", "qlib.workflow.R.start"]
    assert summary["artifact_policy"]["policy"] == "slim_card_recorder_first"
    assert set(summary["sidecars"]) == {
        "backtest_benchmark_comparison",
        "candidate_metrics",
        "failure_modes",
        "ml_position_summary",
        "selected_model",
    }
    for rel_path in summary["sidecars"].values():
        assert (card / rel_path).exists(), rel_path

    candidates = (card / summary["sidecars"]["candidate_metrics"]).read_text(encoding="utf-8")
    assert "hist_gradient_boosting_regressor" in candidates
    assert "lgbm_classifier" in candidates
    assert summary["recorder_id"] in notebook_text
    assert "D.features" in notebook_text
    assert "R.start" in notebook_text
    assert "evaluate_ml_position_candidates" in notebook_text
    assert "不晋升" in conclusion


def test_agentic_alpha_ml_throttle_card_beats_soxl_but_not_constrained_weight():
    card = ROOT / "research" / "agentic_alpha_ml_throttle_search_v1"
    summary = _summary(card)
    notebook_text = (card / "agentic_alpha_ml_throttle_search_v1.ipynb").read_text(encoding="utf-8")
    conclusion = (card / "conclusion.md").read_text(encoding="utf-8")

    assert summary["experiment"] == "agentic_alpha_ml_throttle_search_v1"
    assert summary["card_type"] == "ml_throttle_position_search"
    assert summary["promotion_decision"] == "not_promoted_ml_throttle_under_constrained_weight"
    assert summary["segments"] == {
        "train": ["2019-01-02", "2024-12-31"],
        "validation": ["2025-01-02", "2025-12-31"],
        "test": ["2026-01-02", "2026-06-09"],
    }
    assert summary["goal_gate"]["model_target_col"] == "label_return_5d"
    assert summary["goal_gate"]["return_col"] == "forward_return"
    assert summary["goal_gate"]["max_position"] == 1.0
    assert summary["goal_gate"]["max_validation_reversal_ratio"] == 0.4

    selected = summary["selected_candidate"]
    previous_ml = summary["previous_ml_position_reference"]
    constrained = summary["constrained_weight_reference"]
    legacy = summary["legacy_soxlbacktest_benchmark"]
    assert selected["selected_by"] == "validation_selection_objective_score"
    assert selected["test"]["position_mode"] == "base_throttle"
    assert selected["test"]["model_target_col"] == "label_return_5d"
    assert selected["test"]["max_position"] <= 1.0
    assert selected["test"]["days_above_100"] == 0
    assert selected["test"]["next_day_reversal_ratio"] <= 0.01
    assert selected["test"]["return_pct"] > previous_ml["return_pct"]
    assert selected["test"]["return_pct"] > legacy["rulebase_return_pct"]
    assert selected["test"]["return_pct"] < constrained["return_pct"]

    report = summary["ml_throttle_report"]
    assert report["candidate_count"] >= 1000
    assert report["comparison"]["beats_previous_ml"] is True
    assert report["comparison"]["beats_soxlbacktest"] is True
    assert report["comparison"]["beats_constrained_weight_reference"] is False
    assert {"ExtraTreesClassifier", "RandomForestRegressor", "LGBMClassifier"}.issubset(
        set(report["model_families"])
    )
    assert summary["qlib_native_api"] == ["qlib.init", "qlib.data.D.features", "qlib.workflow.R.start"]
    assert summary["artifact_policy"]["policy"] == "slim_card_recorder_first"
    assert set(summary["sidecars"]) == {
        "backtest_benchmark_comparison",
        "candidate_metrics",
        "failure_modes",
        "ml_throttle_summary",
        "selected_model",
    }
    for rel_path in summary["sidecars"].values():
        assert (card / rel_path).exists(), rel_path

    candidates = (card / summary["sidecars"]["candidate_metrics"]).read_text(encoding="utf-8")
    assert "base_throttle" in candidates
    assert "extra_trees_classifier" in candidates
    assert summary["recorder_id"] in notebook_text
    assert "D.features" in notebook_text
    assert "R.start" in notebook_text
    assert "label_return_5d" in notebook_text
    assert "evaluate_ml_position_candidates" in notebook_text
    assert "不替代当前约束权重模型" in conclusion


def test_agentic_alpha_ml_factor_enhanced_throttle_card_records_external_factor_groups():
    card = ROOT / "research" / "agentic_alpha_ml_factor_enhanced_throttle_v1"
    summary = _summary(card)
    notebook_text = (card / "agentic_alpha_ml_factor_enhanced_throttle_v1.ipynb").read_text(
        encoding="utf-8"
    )
    conclusion = (card / "conclusion.md").read_text(encoding="utf-8")

    assert summary["experiment"] == "agentic_alpha_ml_factor_enhanced_throttle_v1"
    assert summary["card_type"] == "ml_factor_enhanced_throttle_search"
    assert summary["segments"] == {
        "train": ["2019-01-02", "2024-12-31"],
        "validation": ["2025-01-02", "2025-12-31"],
        "test": ["2026-01-02", "2026-06-09"],
    }
    assert summary["goal_gate"]["model_target_col"] == "label_return_5d"
    assert summary["goal_gate"]["return_col"] == "forward_return"
    assert summary["goal_gate"]["max_position"] == 1.0

    factor_sources = summary["factor_sources"]
    assert set(factor_sources) == {
        "base_daily",
        "cross_asset_rotation",
        "intraday_t1",
        "macro_credit",
        "tail_risk_external",
    }
    assert factor_sources["base_daily"]["provider_manifest"] == "data/qlib/soxl/qlib_store_manifest.json"
    assert factor_sources["macro_credit"]["provider_manifest"] == (
        "data/qlib/soxl_macro_credit_v1/qlib_store_manifest.json"
    )
    assert factor_sources["macro_credit"]["coverage_end"] == "2026-06-09"
    assert factor_sources["intraday_t1"]["lag_policy"] == "T+1_daily_safe"
    assert factor_sources["intraday_t1"]["coverage_end"] == "2026-06-05"
    assert factor_sources["tail_risk_external"]["coverage_end"] == "2026-04-16"
    assert factor_sources["cross_asset_rotation"]["coverage_end"] == "2026-04-15"
    assert factor_sources["tail_risk_external"]["freshness_status"] == "stale_after_2026-04-16"
    assert factor_sources["cross_asset_rotation"]["freshness_status"] == "stale_after_2026-04-15"

    feature_groups = summary["feature_groups"]
    assert feature_groups["base_throttle_features"] == 30
    assert feature_groups["enhanced_total"] > feature_groups["base_throttle_features"]
    assert feature_groups["macro_credit"] >= 8
    assert feature_groups["tail_risk_external"] >= 5
    assert feature_groups["intraday_t1"] >= 4
    assert feature_groups["cross_asset_rotation"] >= 4

    selected = summary["selected_candidate"]
    assert selected["selected_by"] == "validation_selection_objective_score"
    assert selected["test"]["position_mode"] == "base_throttle"
    assert selected["test"]["max_position"] <= 1.0
    assert selected["test"]["days_above_100"] == 0
    assert selected["test"]["next_day_reversal_ratio"] <= summary["goal_gate"]["max_validation_reversal_ratio"]

    report = summary["ml_factor_enhanced_report"]
    assert report["candidate_count"] >= 1000
    assert report["feature_count_delta_vs_ml_throttle"] > 0
    assert "beats_ml_throttle" in report["comparison"]
    assert {"ExtraTreesClassifier", "RandomForestRegressor", "LGBMClassifier"}.issubset(
        set(report["model_families"])
    )
    assert summary["qlib_native_api"] == ["qlib.init", "qlib.data.D.features", "qlib.workflow.R.start"]
    assert summary["artifact_policy"]["policy"] == "slim_card_recorder_first"
    assert set(summary["sidecars"]) == {
        "backtest_benchmark_comparison",
        "candidate_metrics",
        "failure_modes",
        "ml_factor_enhanced_summary",
        "selected_model",
    }
    for rel_path in summary["sidecars"].values():
        assert (card / rel_path).exists(), rel_path

    candidates = (card / summary["sidecars"]["candidate_metrics"]).read_text(encoding="utf-8")
    assert "base_throttle" in candidates
    assert "extra_trees_classifier" in candidates
    assert summary["recorder_id"] in notebook_text
    assert "TAIL_PROVIDER_URI" in notebook_text
    assert "MACRO_PROVIDER_URI" in notebook_text
    assert "INTRADAY_PROVIDER_URI" in notebook_text
    assert "CROSS_ASSET_PROVIDER_URI" in notebook_text
    assert "D.features" in notebook_text
    assert "R.start" in notebook_text
    assert "evaluate_ml_position_candidates" in notebook_text
    assert "不构成买卖建议" in conclusion


def test_agentic_alpha_high_chop_lgb_xgb_card_uses_new_factors_and_latest_signal():
    card = ROOT / "research" / "agentic_alpha_high_chop_lgb_xgb_v1"
    summary = _summary(card)
    notebook_text = (card / "agentic_alpha_high_chop_lgb_xgb_v1.ipynb").read_text(encoding="utf-8")
    conclusion = (card / "conclusion.md").read_text(encoding="utf-8")

    assert summary["experiment"] == "agentic_alpha_high_chop_lgb_xgb_v1"
    assert summary["card_type"] == "high_chop_lgb_xgb_model_search"
    assert summary["promotion_decision"] == "research_pass_execution_not_proven"
    assert summary["as_of"] == "2026-07-02"
    assert summary["latest_signal_date"] == "2026-07-02"
    assert summary["segments"] == {
        "train": ["2019-01-02", "2024-12-31"],
        "validation": ["2025-01-02", "2025-12-31"],
        "test": ["2026-01-02", "2026-06-25"],
        "latest_score": ["2026-01-02", "2026-07-02"],
    }
    assert Path(summary["recorder_uri"]) == (card / "qlib_recorder").resolve()
    assert Path(summary["recorder_local_dir"]).exists()

    selected = summary["selected_candidate"]
    assert selected["selected_by"] == "validation_selection_objective_score_near_tie_implementation_tiebreaker"
    assert selected["model_spec"]["model"] == "xgb_classifier"
    assert selected["test"]["position_mode"] == "base_throttle"
    assert selected["test"]["return_pct"] > summary["ytd_buyhold_benchmark"]["return_pct"]
    assert selected["test"]["max_drawdown_pct"] >= summary["goal_gate"]["max_drawdown_floor"]
    assert selected["test"]["high_chop_trade_count"] >= summary["goal_gate"]["min_high_chop_trades"]
    assert selected["test"]["high_chop_avg_abs_position"] >= summary["goal_gate"]["min_high_chop_avg_abs_position"]
    assert selected["test"]["post_high_chop_trade_count"] >= summary["goal_gate"]["min_post_chop_trades"]
    assert selected["test"]["passes_goal_gate"] is True
    assert selected["test"]["return_pct"] < summary["base_constrained_reference"]["return_pct"]

    report = summary["high_chop_lgb_xgb_report"]
    assert report["candidate_count"] >= 1800
    assert report["goal_passer_count"] > 0
    assert report["selection_policy"]["score_tolerance"] == summary["goal_gate"]["selection_score_tolerance"]
    assert report["selection_policy"]["tiebreakers"] == summary["goal_gate"]["selection_tiebreakers"]
    assert report["selection_policy"]["standalone_whole_position_policy"] == (
        "diagnostic_only_until_validation_and_test_stability_are_proven"
    )
    standalone = report["standalone_whole_position_diagnostic"]
    assert standalone["selection_disabled"] is True
    assert standalone["best_validation_candidate"]["position_mode"] in {"probability", "binary"}
    assert standalone["same_candidate_test"]["return_pct"] < 0.0
    assert set(report["model_families"]) == {
        "lgbm_classifier",
        "lgbm_regressor",
        "xgb_classifier",
        "xgb_regressor",
    }
    assert {
        "high_chop_pressure",
        "chop_breakdown_risk",
        "chop_reentry_quality",
        "soxx_lead_10",
        "levered_decay_20",
        "panic_reversal_5",
        "volume_climax_z20",
    }.issubset(set(report["feature_families"]))
    assert report["comparison"]["beats_ytd_buyhold"] is True
    assert report["comparison"]["passes_goal_gate"] is True
    assert report["comparison"]["beats_base_constrained_ref"] is False

    latest = summary["latest_signal"]
    assert latest["label_incomplete"] is True
    assert latest["trained_through"] == "2025-12-31"
    assert 0.0 <= latest["target_position"] <= 1.0
    assert 0.0 <= latest["probability"] <= 1.0
    assert latest["recent_25d_trade_count"] >= 1

    assert summary["qlib_native_api"] == ["qlib.init", "qlib.data.D.features", "qlib.workflow.R.start"]
    assert summary["artifact_policy"]["policy"] == "slim_card_recorder_first"
    assert set(summary["sidecars"]) == {
        "backtest_benchmark_comparison",
        "candidate_metrics",
        "execution_readiness",
        "failure_modes",
        "high_chop_lgb_xgb_summary",
        "latest_signal",
        "regime_metrics",
        "selected_model",
        "window_metrics",
    }
    for rel_path in summary["sidecars"].values():
        assert (card / rel_path).exists(), rel_path
    assert RECORDER_ONLY_ROOT_FILES.isdisjoint({path.name for path in card.iterdir() if path.is_file()})

    candidates = (card / summary["sidecars"]["candidate_metrics"]).read_text(encoding="utf-8")
    assert "lgbm_classifier" in candidates
    assert "xgb_classifier" in candidates
    assert "chop_breakdown_risk" in candidates
    assert "selection_disabled" in candidates
    assert summary["recorder_id"] in notebook_text
    assert "R.set_uri" in notebook_text
    assert "D.features" in notebook_text
    assert "R.start" in notebook_text
    assert "evaluate_high_chop_ml_candidates" in notebook_text
    assert "build_high_chop_ml_frame" in notebook_text
    assert "最新 2026-07-02" in conclusion


def test_agentic_alpha_high_chop_open2open_lgb_xgb_card_records_execution_aligned_reject():
    card = ROOT / "research" / "agentic_alpha_high_chop_open2open_lgb_xgb_v1"
    summary = _summary(card)
    notebook_text = (card / "agentic_alpha_high_chop_open2open_lgb_xgb_v1.ipynb").read_text(encoding="utf-8")
    conclusion = (card / "conclusion.md").read_text(encoding="utf-8")

    assert summary["experiment"] == "agentic_alpha_high_chop_open2open_lgb_xgb_v1"
    assert summary["card_type"] == "high_chop_open2open_lgb_xgb_model_search"
    assert summary["promotion_decision"] == "research_reject_open2open_goal_gate"
    assert summary["as_of"] == "2026-07-02"
    assert summary["latest_signal_date"] == "2026-07-02"
    assert summary["execution_timing"] == "close_t_signal_next_open_to_following_open"
    assert summary["return_col"] == "open_to_open_return"
    assert summary["model_target_col"] == "label_open_to_open_5d"
    assert summary["segments"] == {
        "train": ["2019-01-02", "2024-12-31"],
        "validation": ["2025-01-02", "2025-12-31"],
        "test": ["2026-01-02", "2026-06-24"],
        "latest_score": ["2026-01-02", "2026-07-02"],
    }
    assert Path(summary["recorder_uri"]) == (card / "qlib_recorder").resolve()
    assert Path(summary["recorder_local_dir"]).exists()

    selected = summary["selected_candidate"]
    assert selected["selected_by"] == "validation_selection_objective_score_near_tie_implementation_tiebreaker"
    assert selected["model_spec"]["model"] == "xgb_classifier"
    assert selected["test"]["position_mode"] == "base_throttle"
    assert selected["test"]["return_col"] == "open_to_open_return"
    assert selected["test"]["model_target_col"] == "label_open_to_open_5d"
    assert selected["test"]["passes_goal_gate"] is False
    assert selected["test"]["return_pct"] < summary["base_constrained_reference"]["return_pct"]
    assert selected["test"]["return_pct"] < summary["ytd_buyhold_benchmark"]["return_pct"]
    assert selected["test"]["max_drawdown_pct"] < summary["goal_gate"]["max_drawdown_floor"]

    report = summary["high_chop_open2open_lgb_xgb_report"]
    assert report["return_col"] == "open_to_open_return"
    assert report["model_target_col"] == "label_open_to_open_5d"
    assert report["execution_timing"] == "close_t_signal_next_open_to_following_open"
    assert report["candidate_count"] >= 1800
    assert report["goal_passer_count"] == 0
    assert report["oracle_test_source"] == "best_test_diagnostic_no_goal_passers"
    assert set(report["model_families"]) == {
        "lgbm_classifier",
        "lgbm_regressor",
        "xgb_classifier",
        "xgb_regressor",
    }
    assert report["comparison"]["beats_ytd_buyhold"] is False
    assert report["comparison"]["beats_base_constrained_ref"] is False
    assert report["comparison"]["passes_goal_gate"] is False
    assert summary["oracle_test_best_candidate"]["return_col"] == "open_to_open_return"
    assert summary["oracle_test_best_candidate"]["passes_goal_gate"] is False

    latest = summary["latest_signal"]
    assert latest["return_col"] == "open_to_open_return"
    assert latest["model_target_col"] == "label_open_to_open_5d"
    assert latest["execution_timing"] == "close_t_signal_next_open_to_following_open"
    assert latest["label_incomplete"] is True
    assert latest["trained_through"] == "2025-12-31"
    assert 0.0 <= latest["target_position"] <= 1.0

    assert summary["qlib_native_api"] == ["qlib.init", "qlib.data.D.features", "qlib.workflow.R.start"]
    assert summary["artifact_policy"]["policy"] == "slim_card_recorder_first"
    assert set(summary["sidecars"]) == {
        "backtest_benchmark_comparison",
        "candidate_metrics",
        "execution_readiness",
        "failure_modes",
        "high_chop_open2open_lgb_xgb_summary",
        "latest_signal",
        "regime_metrics",
        "selected_model",
        "window_metrics",
    }
    for rel_path in summary["sidecars"].values():
        assert (card / rel_path).exists(), rel_path
    assert RECORDER_ONLY_ROOT_FILES.isdisjoint({path.name for path in card.iterdir() if path.is_file()})

    candidates = (card / summary["sidecars"]["candidate_metrics"]).read_text(encoding="utf-8")
    assert "open_to_open_return" in candidates
    assert "label_open_to_open_5d" in candidates
    assert "lgbm_classifier" in candidates
    assert "xgb_classifier" in candidates
    assert summary["recorder_id"] in notebook_text
    assert "R.set_uri" in notebook_text
    assert "D.features" in notebook_text
    assert "R.start" in notebook_text
    assert "evaluate_high_chop_ml_candidates" in notebook_text
    assert "build_high_chop_ml_frame" in notebook_text
    assert "open-to-open" in conclusion
    assert "不构成买卖建议" in (card / "execution_readiness.md").read_text(encoding="utf-8")


def test_agentic_alpha_cognitive_alpha_soxl_card_records_codex_alpha158_framework():
    card = ROOT / "research" / "agentic_alpha_cognitive_alpha_soxl_v1"
    summary = _summary(card)
    notebook_text = (card / "agentic_alpha_cognitive_alpha_soxl_v1.ipynb").read_text(encoding="utf-8")
    conclusion = (card / "conclusion.md").read_text(encoding="utf-8")

    assert summary["experiment"] == "agentic_alpha_cognitive_alpha_soxl_v1"
    assert summary["card_type"] == "cognitive_alpha_soxl_throttle_search"
    assert summary["paper_reference"]["arxiv_id"] == "2511.18850"
    assert summary["segments"] == {
        "train": ["2019-01-02", "2024-12-31"],
        "validation": ["2025-01-02", "2025-12-31"],
        "test": ["2026-01-02", "2026-06-23"],
    }

    factor_mining = summary["cognitive_alpha_mining"]
    assert factor_mining["workflow"] == "cognitive_alpha_mining"
    assert factor_mining["paper_reference"] == "arxiv:2511.18850"
    assert factor_mining["alpha158_candidate_count"] >= 50
    assert factor_mining["seed_candidate_count"] >= 8
    assert factor_mining["selected_count"] > 0
    assert {"IC", "ICIR", "RankIC", "RankICIR", "MutualInformation"}.issubset(
        set(factor_mining["metrics"])
    )
    assert "market_structure_cycle" in factor_mining["agent_levels"]
    assert "geometric_fusion" in factor_mining["agent_levels"]

    feature_groups = summary["feature_groups"]
    assert feature_groups["alpha158_selected"] > 0
    assert feature_groups["cognitive_seed_selected"] >= 0
    assert feature_groups["enhanced_total"] > feature_groups["base_throttle_features"]

    selected = summary["selected_candidate"]
    assert selected["selected_by"] == "validation_selection_objective_score"
    assert selected["test"]["position_mode"] == "base_throttle"
    assert selected["test"]["max_position"] <= 1.0
    assert selected["test"]["days_above_100"] == 0
    assert selected["test"]["next_day_reversal_ratio"] <= summary["goal_gate"]["max_validation_reversal_ratio"]

    report = summary["cognitive_alpha_throttle_report"]
    assert report["candidate_count"] >= 1000
    assert "beats_factor_enhanced_throttle" in report["comparison"]
    assert "beats_soxlbacktest" in report["comparison"]
    assert "beats_constrained_weight_reference" in report["comparison"]
    assert summary["qlib_native_api"] == [
        "qlib.init",
        "qlib.data.D.features",
        "DatasetH",
        "DataHandlerLP",
        "Alpha158.get_feature_config",
        "qlib.workflow.R.start",
    ]
    assert summary["artifact_policy"]["policy"] == "slim_card_recorder_first"
    assert set(summary["sidecars"]) == {
        "backtest_benchmark_comparison",
        "candidate_metrics",
        "cognitive_alpha_correlation",
        "cognitive_alpha_metrics",
        "cognitive_alpha_summary",
        "failure_modes",
        "selected_model",
    }
    for rel_path in summary["sidecars"].values():
        assert (card / rel_path).exists(), rel_path

    alpha_metrics = (card / summary["sidecars"]["cognitive_alpha_metrics"]).read_text(encoding="utf-8")
    assert "mutual_information" in alpha_metrics
    assert "quality_status" in alpha_metrics
    assert "selection_tier" in alpha_metrics
    assert summary["recorder_id"] in notebook_text
    assert "build_alpha158_legacy_dataset" in notebook_text
    assert "build_cognitive_alpha_seed_candidates" in notebook_text
    assert "evaluate_cognitive_alpha_candidates" in notebook_text
    assert "cognitive_alpha_codex_prompt_seed" in notebook_text
    assert "evaluate_ml_position_candidates" in notebook_text
    assert "不构成买卖建议" in conclusion


def test_cognitive_alpha_evolution_smoke_card_records_resumable_goal_runner():
    card = ROOT / "research" / "cognitive_alpha_evolution_smoke_v1"
    summary = _summary(card)
    notebook_text = (card / "cognitive_alpha_evolution_smoke_v1.ipynb").read_text(encoding="utf-8")
    conclusion = (card / "conclusion.md").read_text(encoding="utf-8")

    assert summary["experiment"] == "cognitive_alpha_evolution_smoke_v1"
    assert summary["card_type"] == "diagnostic_cognitive_alpha_goal_runner"
    assert summary["promotion_decision"] == "not_promoted_smoke_runner_only"
    assert summary["paper_reference"]["arxiv_id"] == "2511.18850"
    assert summary["segments"]["test"][1] == summary["as_of"]
    assert summary["evolution"]["workflow"] == "cognitive_alpha_evolution"
    assert summary["evolution"]["profile"] == "soxl_smoke"
    assert summary["evolution"]["completed_generations"] == 2
    assert summary["evolution"]["resume_start_generation"] == 0
    assert summary["evolution"]["selected_count"] > 0
    assert summary["evolution"]["repair_attempt_count"] >= 1
    assert summary["evolution"]["initial_candidate_count"] == 9
    assert summary["evolution"]["candidate_provider"]["provider_id"] == "deterministic_code_agent_pool"
    assert summary["evolution"]["candidate_provider"]["profile_agent_count"] == 3
    assert summary["evolution"]["candidate_provider"]["guidance_style_count"] == 5
    assert summary["evolution"]["evolution_ops"] == ["mutation", "crossover", "crossover_mutation"]
    assert summary["qlib_native_api"] == [
        "qlib.init",
        "qlib.data.D.features",
        "DatasetH",
        "DataHandlerLP",
        "Alpha158.get_feature_config",
        "qlib.workflow.R.start",
    ]
    assert summary["qlib_docs"] == {
        "data": "https://qlib.readthedocs.io/en/latest/component/data.html",
        "workflow": "https://qlib.readthedocs.io/en/latest/component/workflow.html",
    }
    assert summary["artifact_policy"]["policy"] == "slim_card_recorder_first"
    assert summary["artifact_policy"]["workflow_config"] == "workflow.yaml"
    assert set(summary["sidecars"]) == {
        "evolution_summary",
        "generation_metrics",
        "selected_features",
        "failure_modes",
    }
    assert set(summary["artifact_policy"]["recorder_only_artifacts"]) == {
        "candidate_pool",
        "fitness",
        "lineage",
        "quality",
        "rejected",
        "state",
    }
    for rel_path in summary["sidecars"].values():
        assert (card / rel_path).exists(), rel_path
    recorder_dir = Path(summary["recorder_local_dir"])
    assert recorder_dir.exists()
    assert (recorder_dir / "artifacts" / "cognitive_alpha_evolution" / "state.json").exists()
    assert (recorder_dir / "artifacts" / "cognitive_alpha_evolution" / "candidate_pool" / "generation_000.jsonl").exists()
    assert (recorder_dir / "artifacts" / "cognitive_alpha_evolution" / "candidate_pool" / "generation_001.jsonl").exists()
    assert summary["recorder_id"] in notebook_text
    assert "run_cognitive_alpha_evolution" in notebook_text
    assert "build_cognitive_alpha_code_candidate_provider" in notebook_text
    assert "build_alpha158_legacy_dataset" in notebook_text
    assert "D.features" in notebook_text
    assert "R.start" in notebook_text
    assert "不构成买卖建议" in conclusion


def test_cognitive_alpha_evolution_full_card_records_full_goal_runner_profile():
    card = ROOT / "research" / "cognitive_alpha_evolution_full_v1"
    summary = _summary(card)
    notebook_text = (card / "cognitive_alpha_evolution_full_v1.ipynb").read_text(encoding="utf-8")
    conclusion = (card / "conclusion.md").read_text(encoding="utf-8")

    assert summary["experiment"] == "cognitive_alpha_evolution_full_v1"
    assert summary["card_type"] == "diagnostic_cognitive_alpha_full_goal_runner"
    assert summary["promotion_decision"] == "not_promoted_full_evolution_only"
    assert summary["paper_reference"]["arxiv_id"] == "2511.18850"
    assert summary["evolution"]["workflow"] == "cognitive_alpha_evolution"
    assert summary["evolution"]["profile"] == "soxl_full"
    assert summary["evolution"]["completed_generations"] == 24
    assert summary["evolution"]["spec"]["initial_pool_size"] == 80
    assert summary["evolution"]["spec"]["parent_pool_size"] == 32
    assert summary["evolution"]["spec"]["children_per_generation"] == 96
    assert summary["evolution"]["selected_count"] > 0
    assert summary["evolution"]["max_candidate_name_length"] <= 96
    assert summary["evolution"]["initial_candidate_count"] == 80
    assert summary["evolution"]["candidate_provider"]["provider_id"] == "deterministic_code_agent_pool"
    assert summary["evolution"]["candidate_provider"]["profile_agent_count"] == 21
    assert summary["evolution"]["candidate_provider"]["guidance_style_count"] == 5
    assert summary["artifact_policy"]["policy"] == "slim_card_recorder_first"
    assert set(summary["sidecars"]) == {
        "evolution_summary",
        "failure_modes",
        "selected_features",
    }
    assert set(summary["artifact_policy"]["recorder_only_artifacts"]) == {
        "candidate_pool",
        "fitness",
        "lineage",
        "quality",
        "rejected",
        "state",
    }
    assert not (card / "candidate_pool").exists()
    assert not (card / "generation_metrics.csv").exists()
    recorder_dir = Path(summary["recorder_local_dir"])
    assert (recorder_dir / "artifacts" / "cognitive_alpha_evolution" / "state.json").exists()
    assert (
        recorder_dir
        / "artifacts"
        / "cognitive_alpha_evolution"
        / "candidate_pool"
        / "generation_023.jsonl"
    ).exists()
    assert "build_cognitive_alpha_evolution_profile" in notebook_text
    assert "build_cognitive_alpha_code_candidate_provider" in notebook_text
    assert "soxl_full" in notebook_text
    assert "run_cognitive_alpha_evolution" in notebook_text
    assert "D.features" in notebook_text
    assert "R.start" in notebook_text
    assert "不构成买卖建议" in conclusion


def test_cognitive_alpha_external_provider_smoke_card_records_codex_provider_readiness():
    card = ROOT / "research" / "cognitive_alpha_external_provider_smoke_v1"
    summary = _summary(card)
    notebook_text = (card / "cognitive_alpha_external_provider_smoke_v1.ipynb").read_text(encoding="utf-8")
    conclusion = (card / "conclusion.md").read_text(encoding="utf-8")
    failure_modes = (card / "failure_modes.md").read_text(encoding="utf-8")

    assert summary["experiment"] == "cognitive_alpha_external_provider_smoke_v1"
    assert summary["card_type"] == "diagnostic_cognitive_alpha_codex_provider_smoke"
    assert summary["promotion_decision"] == "not_promoted_codex_provider_readiness_only"
    assert summary["paper_reference"]["arxiv_id"] == "2511.18850"
    assert "external_provider_readiness" not in summary
    assert summary["codex_provider_readiness"]["status"] in {"ok", "warning"}
    assert summary["codex_provider_readiness"]["auth_mode"] == "chatgpt"
    assert summary["codex_provider_readiness"]["openai_api_key_required"] is False
    assert summary["codex_provider_readiness"]["provider_builder"] == "build_cognitive_alpha_codex_exec_candidate_provider"
    assert summary["codex_provider_readiness"]["execution_mode"] == "codex_exec_chatgpt_auth_no_openai_api_key"
    assert summary["codex_provider_readiness"]["provider_command"] == [
        "codex",
        "exec",
        "--output-schema",
        "cognitive_alpha_candidate_schema.json",
    ]
    assert summary["evolution"]["workflow"] == "cognitive_alpha_evolution"
    assert summary["evolution"]["profile"] == "soxl_smoke"
    assert summary["evolution"]["completed_generations"] == 2
    assert summary["evolution"]["initial_candidate_count"] == 4
    assert summary["evolution"]["repair_attempt_count"] >= 1
    assert summary["evolution"]["candidate_provider"]["provider_id"] == "fixture_external_command_provider"
    assert summary["evolution"]["candidate_provider"]["provider_kind"] == "external_llm_code"
    assert summary["evolution"]["candidate_provider"]["llm_model"] == "fixture-command"
    assert summary["evolution"]["candidate_provider"]["openai_api_key_required"] is False
    assert summary["evolution"]["candidate_provider"]["codex_exec_contract_compatible"] is True
    assert summary["evolution"]["candidate_provider"]["prompt_count"] == 4
    assert summary["evolution"]["candidate_provider"]["response_count"] == 4
    assert summary["evolution"]["candidate_provider"]["external_llm_configured"] is True
    assert summary["artifact_policy"]["policy"] == "slim_card_recorder_first"
    assert set(summary["sidecars"]) == {
        "evolution_summary",
        "failure_modes",
        "selected_features",
    }
    assert set(summary["artifact_policy"]["recorder_only_artifacts"]) == {
        "candidate_pool",
        "fitness",
        "lineage",
        "quality",
        "rejected",
        "state",
    }
    for rel_path in summary["sidecars"].values():
        assert (card / rel_path).exists(), rel_path
    for forbidden in RECORDER_ONLY_ROOT_FILES:
        assert not (card / forbidden).exists(), forbidden
    recorder_dir = Path(summary["recorder_local_dir"])
    assert (recorder_dir / "artifacts" / "cognitive_alpha_evolution" / "state.json").exists()
    assert (
        recorder_dir
        / "artifacts"
        / "cognitive_alpha_evolution"
        / "candidate_pool"
        / "generation_001.jsonl"
    ).exists()
    assert summary["recorder_id"] in notebook_text
    assert "codex_cli_readiness" in notebook_text
    assert "build_cognitive_alpha_codex_exec_candidate_provider" in notebook_text
    assert "build_cognitive_alpha_command_llm_candidate_provider" in notebook_text
    assert "src.workflow.cognitive_alpha_fixture_provider" in notebook_text
    assert "D.features" in notebook_text
    assert "R.start" in notebook_text
    assert "fixture provider" in notebook_text
    assert "Codex CLI" in conclusion
    assert "不构成买卖建议" in conclusion
    for text in (json.dumps(summary, ensure_ascii=False), conclusion, failure_modes):
        assert "OPENAI_API_KEY_NOT_CONFIGURED" not in text
        assert "OpenAI API key" not in text


def test_cognitive_alpha_goal_codex_exec_card_records_real_codex_exec_goal_chain():
    card = ROOT / "research" / "cognitive_alpha_goal_codex_exec_v1"
    summary = _summary(card)
    notebook_text = (card / "cognitive_alpha_goal_codex_exec_v1.ipynb").read_text(encoding="utf-8")
    conclusion = (card / "conclusion.md").read_text(encoding="utf-8")
    workflow = (card / "workflow.yaml").read_text(encoding="utf-8")
    failure_modes = (card / "failure_modes.md").read_text(encoding="utf-8")

    assert summary["experiment"] == "cognitive_alpha_goal_codex_exec_v1"
    assert summary["card_type"] == "diagnostic_cognitive_alpha_codex_exec_goal_chain"
    assert summary["paper_reference"]["arxiv_id"] == "2511.18850"
    assert summary["promotion_decision"] in {
        "candidate_for_validated_research",
        "not_promoted_under_base_constrained_ref",
        "not_promoted_failed_position_risk_gates",
    }
    assert summary["codex_provider"]["provider_builder"] == "build_cognitive_alpha_codex_exec_candidate_provider"
    assert summary["codex_provider"]["provider_id"] == "codex_exec_candidate_provider"
    assert summary["codex_provider"]["provider_kind"] == "codex_exec_code"
    assert summary["codex_provider"]["execution_mode"] == "codex_exec_chatgpt_auth_no_openai_api_key"
    assert summary["codex_provider"]["sandbox"] == "read-only"
    assert summary["codex_provider"]["output_schema"] == "cognitive_alpha_candidate_schema.json"
    assert summary["codex_provider"]["openai_api_key_required"] is False
    assert summary["codex_provider"]["prompt_count"] > 0
    assert summary["codex_provider"]["response_count"] == summary["codex_provider"]["prompt_count"]
    assert summary["max_attempts"] >= 2
    assert summary["attempt_index"] >= 1
    assert summary["goal_loop"]["status"] == summary["status"]
    assert summary["goal_loop"]["attempt_index"] == summary["attempt_index"]
    assert summary["goal_loop"]["max_attempts"] == summary["max_attempts"]
    assert summary["goal_loop"]["terminal_reason"] == summary["terminal_reason"]
    assert summary["next_action"] == summary["goal_loop"]["next_action"]
    assert len(summary["attempts"]) == summary["attempt_index"]
    if summary["promotion_decision"].startswith("not_promoted"):
        assert summary["status"] == "needs_iteration"
        assert summary["terminal_reason"] == "max_attempts_exhausted_needs_iteration"
        assert summary["attempt_index"] == summary["max_attempts"]
        assert summary["remaining_attempts"] == 0
        assert all(attempt["status"] == "needs_iteration" for attempt in summary["attempts"])
        assert summary["next_action"]["action"] == "continue_codex_exec_factor_generation"
        assert summary["next_action"]["next_attempt"] == summary["attempt_index"] + 1
        assert summary["next_action"]["requires_budget_refresh"] is True
        assert summary["next_action"]["required_max_attempts"] == summary["attempt_index"] + 1
        assert summary["evolution"]["candidate_provider"]["feedback_context_included"] is True
    if summary["promotion_decision"] == "candidate_for_validated_research":
        assert summary["status"] == "complete"
        assert summary["terminal_reason"] == "promotion_or_terminal_decision"
        assert summary["remaining_attempts"] == 0
        assert summary["next_action"]["action"] == "stop_terminal_decision"
        assert summary["position_model"]["comparison"]["beats_base_constrained_ref"] is True
        assert summary["position_model"]["comparison"]["beats_constrained_weight_reference"] is True
        assert summary["position_model"]["comparison"]["beats_soxl_backtest"] is True
        assert summary["position_model"]["comparison"]["beats_soxl_buyhold"] is True
        assert summary["position_model"]["comparison"]["selected_vs_base_constrained_ref_return_gap_pct"] > 0.0
        assert summary["position_model"]["risk_gates"]["selected_passes_all"] is True
        assert summary["position_model"]["walk_forward"]["all_folds_pass_risk_gates"] is True
        assert summary["position_model"]["selected_candidate"]["test"]["position_mode"] == "base_plus_probability"
        assert summary["position_model"]["latest_signal"]["target_position"] <= 0.001
    assert summary["codex_provider_readiness"]["status"] in {"ok", "warning"}
    assert summary["codex_provider_readiness"]["auth_mode"] == "chatgpt"
    assert summary["codex_provider_readiness"]["stored_api_key"] == "false"
    assert summary["codex_provider_readiness"]["stored_chatgpt_tokens"] == "true"

    assert summary["evolution"]["workflow"] == "cognitive_alpha_evolution"
    assert summary["evolution"]["candidate_provider"]["provider_id"] == "codex_exec_candidate_provider"
    assert summary["evolution"]["candidate_provider"]["provider_kind"] == "codex_exec_code"
    assert summary["evolution"]["candidate_provider"]["codex_direct_compatible"] is True
    assert summary["evolution"]["candidate_provider"]["openai_api_key_required"] is False
    assert summary["evolution"]["candidate_provider"]["prompt_count"] > 0
    assert summary["evolution"]["candidate_provider"]["generated_candidate_count"] == summary["evolution"]["initial_candidate_count"]
    assert (
        summary["evolution"]["candidate_provider"]["generated_candidate_count"]
        >= summary["evolution"]["candidate_provider"]["requested_initial_pool_size"]
    )
    assert summary["evolution"]["candidate_provider"]["invalid_candidate_count"] >= 0
    assert summary["evolution"]["completed_generations"] >= 2
    assert summary["evolution"]["selected_count"] > 0
    assert summary["evolution"]["repair_attempt_count"] >= 1
    assert summary["evolution"]["evolution_ops"] == ["mutation", "crossover", "crossover_mutation"]
    assert {"ic", "icir", "rank_ic", "rank_icir", "mutual_information"} == set(summary["evolution"]["metric_names"])
    assert len(summary["evolution"]["agent_levels_covered"]) >= 2
    assert summary["position_model"]["workflow"] == "cognitive_alpha_position_model"
    assert summary["position_model"]["walk_forward"]["fold_count"] >= 2
    assert "beats_constrained_weight_reference" in summary["position_model"]["comparison"]
    assert summary["position_model"]["latest_signal"]["as_of"] == summary["as_of"]
    assert 0.0 <= summary["position_model"]["latest_signal"]["target_position"] <= 1.0
    assert summary["direct_factor_backtest"]["workflow"] == "cognitive_alpha_weighted_factor_search"
    assert summary["direct_factor_backtest"]["promotion_decision"] in {
        "diagnostic_direct_factor_candidate",
        "not_promoted_no_selection_eligible",
        "not_promoted_under_constrained_weight_reference",
    }
    assert summary["research_gate"]["final_promotion_source"] in {"position_model", "production_paper_sim"}
    assert summary["research_gate"]["direct_factor_backtest_role"] == "prior_only_not_final_promotion"
    assert summary["production_paper_sim"]["status"] in {"blocked_not_promoted", "candidate_for_paper_sim"}
    assert summary["production_paper_sim"]["required_gate"] == "validated_research_position_model_or_production_paper_sim"
    assert summary["intraday_15m_factors"]["source"] == "qlib.data.D.features"
    assert summary["intraday_15m_factors"]["freq"] == "15min"
    assert summary["intraday_15m_factors"]["lag_days"] == 1
    assert summary["intraday_15m_factors"]["role"] == "model_features_not_gate"
    assert summary["intraday_15m_factors"]["feature_count"] >= 10
    assert "intraday15m_soxl_minus_3x_soxx_ret_lag1" in summary["intraday_15m_factors"]["feature_cols"]
    assert summary["constituent_intraday_combo_factors"]["role"] == "model_features_not_gate"
    assert summary["constituent_intraday_combo_factors"]["feature_count"] >= 5
    assert "combo_semi_breadth_thrust_15m_lag1" in summary["constituent_intraday_combo_factors"]["feature_cols"]
    feature_cols = set(summary["position_model"]["feature_cols"])
    assert set(summary["intraday_15m_factors"]["feature_cols"]).issubset(feature_cols)
    assert set(summary["constituent_intraday_combo_factors"]["feature_cols"]).issubset(feature_cols)

    assert summary["qlib_native_api"] == [
        "qlib.init",
        "qlib.data.D.features",
        "DatasetH",
        "DataHandlerLP",
        "Alpha158.get_feature_config",
        "qlib.workflow.R.start",
    ]
    assert summary["artifact_policy"]["policy"] == "slim_card_recorder_first"
    assert summary["artifact_policy"]["workflow_config"] == "workflow.yaml"
    assert set(summary["sidecars"]) == {
        "backtest_walk_forward_metrics",
        "benchmark_comparison",
        "direct_factor_benchmark_comparison",
        "direct_factor_candidate_metrics",
        "direct_factor_summary",
        "evolution_summary",
        "failure_modes",
        "generation_metrics",
        "goal_manifest",
        "latest_signal",
        "model_selection",
        "position_summary",
        "selected_features",
        "state",
    }
    assert set(summary["artifact_policy"]["recorder_only_artifacts"]) == {
        "candidate_pool",
        "direct_weighted_positions",
        "fitness",
        "lineage",
        "positions",
        "quality",
        "rejected",
        "state",
    }
    for rel_path in summary["sidecars"].values():
        assert (card / rel_path).exists(), rel_path
    for forbidden in RECORDER_ONLY_ROOT_FILES:
        assert not (card / forbidden).exists(), forbidden

    goal_manifest = json.loads((card / summary["sidecars"]["goal_manifest"]).read_text(encoding="utf-8"))
    goal_state = json.loads((card / summary["sidecars"]["state"]).read_text(encoding="utf-8"))
    assert goal_manifest["status"] == summary["status"]
    assert goal_manifest["terminal_reason"] == summary["terminal_reason"]
    assert goal_manifest["attempt_index"] == summary["attempt_index"]
    assert goal_manifest["max_attempts"] == summary["max_attempts"]
    assert goal_manifest["remaining_attempts"] == summary["remaining_attempts"]
    assert goal_manifest["next_action"] == summary["next_action"]
    assert goal_manifest["completed_stages"] == goal_manifest["stage_order"]
    assert goal_state["status"] == summary["status"]
    assert goal_state["terminal_reason"] == summary["terminal_reason"]
    assert goal_state["current_attempt"] == summary["attempt_index"]
    assert goal_state["remaining_attempts"] == summary["remaining_attempts"]
    assert goal_state["next_action"] == summary["next_action"]
    assert goal_state["completed_stages"] == goal_manifest["stage_order"]
    assert goal_manifest["candidate_generation"]["candidate_provider"]["provider_id"] == "codex_exec_candidate_provider"
    assert goal_manifest["validation_gates"]["require_candidate_provider_no_openai_api_key"] is True
    assert goal_manifest["validation_gates"]["required_candidate_provider_ids"] == ["codex_exec_candidate_provider"]
    assert goal_manifest["validation_gates"]["required_candidate_provider_kinds"] == ["codex_exec_code"]
    assert goal_manifest["validation_gates"]["required_metric_names"] == [
        "ic",
        "icir",
        "rank_ic",
        "rank_icir",
        "mutual_information",
    ]
    assert goal_manifest["model_and_position"]["workflow"] == "cognitive_alpha_position_model"
    assert goal_manifest["direct_factor_backtest"]["workflow"] == "cognitive_alpha_weighted_factor_search"
    assert goal_manifest["research_card"]["promotion_decision"] == summary["position_model"]["promotion_decision"]

    recorder_dir = Path(summary["recorder_local_dir"])
    assert (recorder_dir / "artifacts" / "cognitive_alpha_position" / "positions.csv").exists()
    assert (recorder_dir / "artifacts" / "cognitive_alpha_direct_weighted" / "direct_weighted_positions.csv").exists()
    assert (recorder_dir / "artifacts" / "cognitive_alpha_evolution" / "state.json").exists()
    assert (
        recorder_dir
        / "artifacts"
        / "cognitive_alpha_evolution"
        / "candidate_pool"
        / "generation_001.jsonl"
    ).exists()
    assert summary["recorder_id"] in notebook_text
    assert (card / "attempts" / "attempt_001" / "summary.json").exists()
    if summary["attempt_index"] > 1:
        first_attempt = json.loads(
            (card / "attempts" / "attempt_001" / "summary.json").read_text(encoding="utf-8")
        )
        assert first_attempt["status"] == "needs_iteration"
        assert first_attempt["goal_manifest"]["next_action"]["action"] == "continue_codex_exec_factor_generation"
    final_attempt_dir = card / "attempts" / f"attempt_{summary['attempt_index']:03d}"
    final_attempt = json.loads((final_attempt_dir / "summary.json").read_text(encoding="utf-8"))
    final_attempt_manifest = json.loads((final_attempt_dir / "goal_manifest.json").read_text(encoding="utf-8"))
    final_attempt_state = json.loads((final_attempt_dir / "state.json").read_text(encoding="utf-8"))
    final_attempt_conclusion = (final_attempt_dir / "conclusion.md").read_text(encoding="utf-8")
    assert final_attempt["status"] == summary["status"]
    assert final_attempt["terminal_reason"] == summary["terminal_reason"]
    assert final_attempt["attempts"] == summary["attempts"]
    assert final_attempt["next_action"] == summary["next_action"]
    assert final_attempt_manifest["status"] == goal_manifest["status"]
    assert final_attempt_manifest["next_action"] == goal_manifest["next_action"]
    assert final_attempt_state["status"] == goal_state["status"]
    assert final_attempt_state["next_action"] == goal_state["next_action"]
    assert f"goal_loop: status={summary['status']}" in final_attempt_conclusion
    assert "build_cognitive_alpha_codex_exec_candidate_provider" in notebook_text
    assert "codex_exec_candidate_provider" in notebook_text
    assert "--output-schema" in notebook_text
    assert "--ask-for-approval" in notebook_text
    assert "read-only" in notebook_text
    assert "D.features" in notebook_text
    assert "build_alpha158_legacy_dataset" in notebook_text
    assert "run_cognitive_alpha_position_model" in notebook_text
    assert "run_cognitive_alpha_weighted_factor_search" in notebook_text
    assert "CognitiveAlphaGoalRunner" in notebook_text
    assert "R.start" in notebook_text
    assert "provider_builder: build_cognitive_alpha_codex_exec_candidate_provider" in workflow
    assert "intraday_15m_factors:" in workflow
    assert "constituent_intraday_combo_factors:" in workflow
    assert "intraday_15m_factors:" in conclusion
    assert "constituent_intraday_combo_factors:" in conclusion
    assert "不构成买卖建议" in conclusion
    for text in (json.dumps(summary, ensure_ascii=False), conclusion, failure_modes, notebook_text):
        assert "fixture_external_command_provider" not in text
        assert "fixture provider" not in text


def test_cognitive_alpha_goal_codex_exec_resume_refreshes_exhausted_iteration_budget(tmp_path):
    from src.workflow.cognitive_alpha_goal_codex_exec import (
        CognitiveAlphaGoalCodexExecCardSpec,
        _effective_max_attempts,
        _resume_start_attempt,
    )

    card = tmp_path / "research" / "cognitive_alpha_goal_codex_exec_v1"
    card.mkdir(parents=True)
    (card / "summary.json").write_text(
        json.dumps(
            {
                "status": "needs_iteration",
                "terminal_reason": "max_attempts_exhausted_needs_iteration",
                "attempt_index": 2,
                "max_attempts": 2,
                "remaining_attempts": 0,
                "next_action": {
                    "action": "continue_codex_exec_factor_generation",
                    "next_attempt": 3,
                    "requires_budget_refresh": True,
                    "required_max_attempts": 3,
                },
            }
        ),
        encoding="utf-8",
    )

    spec = CognitiveAlphaGoalCodexExecCardSpec(root=tmp_path, resume=True, max_attempts=2)
    start_attempt = _resume_start_attempt(card)

    assert start_attempt == 3
    assert _effective_max_attempts(card, spec=spec, start_attempt=start_attempt) == 3


def test_risk_hardened_open2open_v2_card_records_strict_lgb_xgb_ablation_and_execution_stress():
    card = ROOT / "research" / "risk_hardened_open2open_v2"
    summary = _summary(card)
    notebook_text = (card / "risk_hardened_open2open_v2.ipynb").read_text(encoding="utf-8")
    conclusion = (card / "conclusion.md").read_text(encoding="utf-8")
    workflow = (card / "workflow.yaml").read_text(encoding="utf-8")
    model_selection = pd.read_csv(card / "model_selection.csv")

    assert summary["experiment"] == "risk_hardened_open2open_v2"
    assert summary["card_type"] == "risk_hardened_open2open_research_card"
    assert summary["benchmark_basis"] == "open_to_open"
    assert summary["return_col"] == "open_to_open_return"
    assert summary["model_target_col"] == "label_open_to_open_5d"
    assert set(summary["model_selection"]["model_families"]) >= {
        "lgbm_classifier",
        "lgbm_regressor",
        "xgb_classifier",
        "xgb_regressor",
        "lgb_xgb_consensus",
    }
    assert summary["model_selection"]["selected_model"] == "lgb_xgb_consensus"
    assert summary["implementation_notes"]["lgb_xgb_consensus_selection"] is True
    assert not model_selection.loc[
        model_selection["model"].eq("lgb_xgb_consensus"), "selection_disabled"
    ].map(bool).any()
    assert not model_selection.loc[
        model_selection["model"].isin({"lgbm_classifier", "lgbm_regressor", "xgb_classifier", "xgb_regressor"}),
        "selection_eligible",
    ].map(bool).any()
    selected_test = model_selection.loc[
        model_selection["selected"].map(bool) & model_selection["segment"].eq("test")
    ].iloc[0]
    previous_floor = summary["risk_hardened_goal"]["previous_model_return_floor"]
    assert previous_floor["source_card"] == "cognitive_alpha_goal_codex_exec_v1"
    assert previous_floor["return_col"] == "open_to_open_return"
    assert previous_floor["pass"] is True
    assert selected_test["return_pct"] >= previous_floor["floor_return_pct"]
    if selected_test["return_pct"] < selected_test["baseline_return_pct"]:
        assert bool(selected_test["baseline_risk_improvement_pass"]) is True
        assert selected_test["max_drawdown_vs_baseline_gap_pct"] > 0.0
        assert selected_test["max_daily_loss_vs_baseline_gap_pct"] > 0.0
        assert selected_test["cvar_5pct_vs_baseline_gap_pct"] > 0.0
        assert selected_test["worst_5_daily_loss_vs_baseline_gap_pct"] > 0.0
        assert summary["risk_hardened_goal"]["baseline_risk_fallback_pass"] is True
    assert summary["model_selection"]["selection_metric"] == "risk_adjusted_excess_return_with_tail_penalties"
    assert summary["position_model"]["risk_gates"]["validation_max_daily_loss_floor"] == -12.0
    assert summary["position_model"]["risk_gates"]["test_max_daily_loss_floor"] == -12.0
    assert summary["position_model"]["risk_gates"]["validation_cvar_5pct_floor"] == -8.0
    assert summary["position_model"]["selected_candidate"]["test"]["cvar_5pct_pct"] is not None
    assert summary["position_model"]["selected_candidate"]["test"]["worst_5_daily_loss_pct"] is not None
    assert summary["position_model"]["selected_candidate"]["test"]["underwater_day_ratio"] is not None
    assert {row["feature_group"] for row in summary["factor_ablation"]["groups"]} == {
        "base",
        "base_cogalpha",
        "base_soxx",
        "base_cogalpha_soxx",
    }
    assert {row["training_window"] for row in summary["training_window_comparison"]["windows"]} >= {
        "expanding_2019",
        "rolling_3y",
    }
    assert summary["constituent_factors"]["point_in_time_status"]["status"] in {
        "single_snapshot_survivorship_biased",
        "multi_snapshot_partial_history",
    }
    assert summary["execution_stress"]["status"] in {"ok", "needs_data"}
    assert {"model_selection", "factor_ablation", "execution_stress", "explainability"}.issubset(
        summary["sidecars"]
    )
    assert summary["artifact_policy"]["policy"] == "slim_card_recorder_first"
    for rel_path in summary["sidecars"].values():
        assert (card / rel_path).exists(), rel_path
    for forbidden in RECORDER_ONLY_ROOT_FILES:
        assert not (card / forbidden).exists(), forbidden
    assert "run_risk_hardened_open2open_card" in notebook_text
    assert "D.features" in notebook_text
    assert "R.start" in notebook_text
    assert "LGBMClassifier" in notebook_text
    assert "XGBClassifier" in notebook_text
    assert "risk_adjusted_excess_return_with_tail_penalties" in workflow
    assert "不构成买卖建议" in conclusion


def test_cognitive_alpha_position_model_card_records_lightgbm_benchmark_and_signal():
    card = ROOT / "research" / "cognitive_alpha_position_model_v1"
    summary = _summary(card)
    notebook_text = (card / "cognitive_alpha_position_model_v1.ipynb").read_text(encoding="utf-8")
    conclusion = (card / "conclusion.md").read_text(encoding="utf-8")
    workflow = (card / "workflow.yaml").read_text(encoding="utf-8")

    assert summary["experiment"] == "cognitive_alpha_position_model_v1"
    assert summary["card_type"] == "diagnostic_cognitive_alpha_position_model"
    assert summary["promotion_decision"] == summary["position_model"]["promotion_decision"]
    assert summary["promotion_decision"] != "candidate_for_validated_research_direct_factor"
    assert summary["paper_reference"]["arxiv_id"] == "2511.18850"
    assert summary["codex_runtime"]["status"] in {"ok", "warning"}
    assert summary["codex_runtime"]["auth_mode"] == "chatgpt"
    assert summary["codex_runtime"]["openai_api_key_required"] is False
    assert summary["segments"]["test"][1] == summary["as_of"]
    assert summary["evolution"]["workflow"] == "cognitive_alpha_evolution"
    assert summary["evolution"]["profile"] == "soxl_full"
    assert summary["evolution"]["completed_generations"] == 24
    assert summary["evolution"]["initial_candidate_count"] == 80
    assert summary["evolution"]["candidate_provider"]["provider_id"] == "deterministic_code_agent_pool"
    assert summary["evolution"]["candidate_provider"]["profile_agent_count"] == 21
    assert summary["evolution"]["candidate_provider"]["guidance_style_count"] == 5
    assert summary["evolution"]["candidate_provider"]["codex_direct_compatible"] is True
    assert summary["evolution"]["candidate_provider"]["openai_api_key_required"] is False
    assert summary["factor_mining"]["workflow"] == "cognitive_alpha_factor_mining"
    assert summary["factor_mining"]["source_layer"] == "src.factor"
    assert summary["factor_mining"]["position_model_included"] is False
    assert summary["factor_mining"]["position_signal_included"] is False
    assert summary["factor_mining"]["factor_library"]["archive_policy"]["candidate_scope"] == "all_generations"
    assert summary["position_model"]["feature_count"] > summary["factor_mining"]["factor_library"]["selected_count"]
    assert set(summary["factor_mining"]["factor_library"]["selected_feature_names"]).issubset(
        set(summary["position_model"]["feature_cols"])
    )
    assert any(column.startswith("alpha158_") for column in summary["position_model"]["feature_cols"])
    assert {"vix", "realized_vol_20"}.issubset(summary["position_model"]["feature_cols"])
    direct_component_payload = summary["direct_factor_backtest"]["selected_candidate"]["component_cols"]
    if isinstance(direct_component_payload, str):
        direct_component_payload = json.loads(direct_component_payload)
    direct_component_cols = set(direct_component_payload)
    assert direct_component_cols.issubset(set(summary["position_model"]["feature_cols"]))
    assert summary["direct_factor_backtest"]["workflow"] == "cognitive_alpha_weighted_factor_search"
    assert summary["direct_factor_backtest"]["promotion_decision"] == "not_promoted_no_selection_eligible"
    assert summary["direct_factor_backtest"]["comparison"]["selected_selection_eligible"] is False
    assert summary["direct_factor_backtest"]["comparison"]["beats_constrained_weight_reference"] is True
    assert (
        summary["direct_factor_backtest"]["comparison"][
            "selected_vs_constrained_weight_reference_return_gap_pct"
        ]
        > 0.0
    )
    assert summary["direct_factor_backtest"]["selected_candidate"]["max_position"] <= 1.0
    assert summary["direct_factor_backtest"]["selected_candidate"]["days_above_100"] == 0
    assert summary["direct_factor_backtest"]["selected_candidate"]["selection_eligible"] is False
    assert summary["direct_factor_backtest"]["selected_candidate"]["passes_effective_component_gate"] is False
    assert summary["direct_factor_backtest"]["selected_candidate"]["effective_component_count"] < summary[
        "direct_factor_backtest"
    ]["spec"]["min_effective_component_count"]
    assert summary["position_model"]["workflow"] == "cognitive_alpha_position_model"
    assert set(summary["position_model"]["model_families"]) >= {
        "hist_gradient_boosting_classifier",
        "lgbm_classifier",
        "logistic_regression",
        "ridge",
    }
    assert summary["position_model"]["risk_gates"]["validation_max_drawdown_floor"] is not None
    assert summary["position_model"]["risk_gates"]["test_max_drawdown_floor"] is not None
    assert summary["position_model"]["risk_gates"]["test_max_daily_loss_floor"] is not None
    assert summary["position_model"]["risk_gates"]["max_test_total_turnover"] is not None
    assert summary["position_model"]["risk_gates"]["max_test_extreme_loss_day_count"] is not None
    assert summary["position_model"]["risk_gates"]["min_test_avg_abs_position"] > 0.0
    assert summary["position_model"]["risk_gates"]["max_test_zero_position_day_ratio"] < 1.0
    assert summary["position_model"]["risk_gates"]["extreme_loss_threshold"] < 0.0
    assert summary["position_model"]["selected_candidate"]["test"]["max_position"] <= 1.0
    assert summary["position_model"]["selected_candidate"]["test"]["days_above_100"] == 0
    assert "max_daily_loss_pct" in summary["position_model"]["selected_candidate"]["test"]
    assert "extreme_loss_day_count" in summary["position_model"]["selected_candidate"]["test"]
    assert "zero_position_day_ratio" in summary["position_model"]["selected_candidate"]["test"]
    assert summary["position_model"]["walk_forward"]["fold_count"] >= 2
    assert "positive_test_fold_ratio" in summary["position_model"]["walk_forward"]
    assert "mean_test_positive_month_ratio" in summary["position_model"]["walk_forward"]
    assert "mean_test_zero_position_day_ratio" in summary["position_model"]["walk_forward"]
    assert "max_test_extreme_loss_day_count" in summary["position_model"]["walk_forward"]
    assert "all_folds_pass_risk_gates" in summary["position_model"]["walk_forward"]
    assert "beats_constrained_weight_reference" in summary["position_model"]["comparison"]
    assert "beats_soxl_backtest" in summary["position_model"]["comparison"]
    assert summary["latest_signal"]["as_of"] == summary["as_of"]
    assert summary["latest_signal"]["action"] in {"BUY", "SELL", "HOLD"}
    assert 0.0 <= summary["latest_signal"]["target_position"] <= 1.0
    assert summary["qlib_native_api"] == [
        "qlib.init",
        "qlib.data.D.features",
        "DatasetH",
        "DataHandlerLP",
        "Alpha158.get_feature_config",
        "qlib.workflow.R.start",
    ]
    assert summary["artifact_policy"]["policy"] == "slim_card_recorder_first"
    assert set(summary["sidecars"]) == {
        "benchmark_comparison",
        "direct_factor_benchmark_comparison",
        "direct_factor_candidate_metrics",
        "direct_factor_summary",
        "failure_modes",
        "goal_manifest",
        "latest_signal",
        "model_selection",
        "position_summary",
        "backtest_walk_forward_metrics",
    }
    assert set(summary["artifact_policy"]["recorder_only_artifacts"]) == {
        "candidate_pool",
        "fitness",
        "lineage",
        "positions",
        "direct_weighted_positions",
        "quality",
        "rejected",
        "state",
    }
    for rel_path in summary["sidecars"].values():
        assert (card / rel_path).exists(), rel_path
    walk_forward = json.loads((card / "position_summary.json").read_text(encoding="utf-8"))["walk_forward"]
    assert walk_forward["fold_count"] == summary["position_model"]["walk_forward"]["fold_count"]
    walk_forward_metrics = (card / summary["sidecars"]["backtest_walk_forward_metrics"]).read_text(encoding="utf-8")
    assert "test_return_pct" in walk_forward_metrics
    assert "risk_gate_pass" in walk_forward_metrics
    assert "test_positive_month_ratio" in walk_forward_metrics
    assert "test_zero_position_day_ratio" in walk_forward_metrics
    assert "test_extreme_loss_day_count" in walk_forward_metrics
    assert summary["production_paper_sim"]["status"] == "blocked_not_promoted"
    assert summary["production_paper_sim"]["required_gate"] == "validated_research_position_model_and_walk_forward"
    goal_manifest = json.loads((card / summary["sidecars"]["goal_manifest"]).read_text(encoding="utf-8"))
    assert goal_manifest["status"] == "needs_iteration"
    assert goal_manifest["terminal_reason"] == "max_attempts_exhausted_needs_iteration"
    assert goal_manifest["next_action"]["action"] == "continue_codex_exec_factor_generation"
    assert goal_manifest["stage_order"] == [
        "data_gate",
        "candidate_generation",
        "quality_checker",
        "fitness_evaluation",
        "evolution_loop",
        "model_and_position",
        "direct_factor_backtest",
        "research_card",
        "signal_push",
    ]
    assert goal_manifest["completed_stages"] == goal_manifest["stage_order"]
    assert goal_manifest["data_gate"]["max_datetime"] == summary["as_of"]
    assert goal_manifest["candidate_generation"]["candidate_provider"]["provider_id"] == "deterministic_code_agent_pool"
    assert goal_manifest["candidate_generation"]["candidate_provider"]["profile_agent_count"] == 21
    assert goal_manifest["validation_gates"]["min_walk_forward_fold_count"] == 3
    assert goal_manifest["candidate_generation"]["initial_candidate_count"] == 80
    assert goal_manifest["validation_gates"]["min_initial_candidate_count"] == 80
    assert goal_manifest["validation_gates"]["min_completed_generations"] == 24
    assert goal_manifest["validation_gates"]["required_candidate_provider_ids"] == ["deterministic_code_agent_pool"]
    assert goal_manifest["validation_gates"]["required_candidate_provider_kinds"] == ["repo_owned_code"]
    assert goal_manifest["validation_gates"]["require_candidate_provider_no_openai_api_key"] is True
    assert goal_manifest["validation_gates"]["require_direct_factor_beats_constrained_weight_reference"] is False
    assert goal_manifest["validation_gates"]["required_direct_factor_benchmark_names"] == [
        "constrained_weight_reference",
        "soxl_backtest",
        "soxl_buyhold",
        "previous_factor_enhanced",
    ]
    assert goal_manifest["candidate_generation"]["candidate_provider"]["codex_direct_compatible"] is True
    assert goal_manifest["candidate_generation"]["candidate_provider"]["openai_api_key_required"] is False
    assert goal_manifest["validation_gates"]["required_metric_names"] == [
        "ic",
        "icir",
        "rank_ic",
        "rank_icir",
        "mutual_information",
    ]
    assert goal_manifest["fitness_evaluation"]["metric_names"] == [
        "ic",
        "icir",
        "rank_ic",
        "rank_icir",
        "mutual_information",
    ]
    assert goal_manifest["evolution_loop"]["completed_generations"] == 24
    assert goal_manifest["model_and_position"]["latest_signal"]["as_of"] == summary["as_of"]
    assert goal_manifest["direct_factor_backtest"]["workflow"] == "cognitive_alpha_weighted_factor_search"
    assert goal_manifest["direct_factor_backtest"]["promotion_decision"] == "not_promoted_no_selection_eligible"
    assert goal_manifest["direct_factor_backtest"]["comparison"]["selected_selection_eligible"] is False
    assert goal_manifest["direct_factor_backtest"]["comparison"]["beats_constrained_weight_reference"] is True
    assert goal_manifest["direct_factor_backtest"]["selected_candidate"]["selection_eligible"] is False
    assert goal_manifest["direct_factor_backtest"]["selected_candidate"]["effective_component_count"] < goal_manifest[
        "direct_factor_backtest"
    ]["spec"]["min_effective_component_count"]
    assert goal_manifest["research_card"]["promotion_decision"] == summary["position_model"]["promotion_decision"]
    assert not (card / "positions.csv").exists()
    recorder_dir = Path(summary["recorder_local_dir"])
    assert (recorder_dir / "artifacts" / "cognitive_alpha_position" / "positions.csv").exists()
    assert (recorder_dir / "artifacts" / "cognitive_alpha_direct_weighted" / "direct_weighted_positions.csv").exists()
    assert (recorder_dir / "artifacts" / "cognitive_alpha_evolution" / "state.json").exists()
    assert "run_cognitive_alpha_factor_mining" in notebook_text
    assert "run_cognitive_alpha_evolution(" not in notebook_text
    assert "build_cognitive_alpha_code_candidate_provider" in notebook_text
    assert "run_cognitive_alpha_position_model" in notebook_text
    assert "run_cognitive_alpha_weighted_factor_search" in notebook_text
    assert "CognitiveAlphaGoalRunner" in notebook_text
    assert "factor_mining:" in workflow
    assert "workflow: cognitive_alpha_factor_mining" in workflow
    assert "position_signal_included: false" in workflow
    assert "direct_factor_backtest:" in workflow
    assert "LGBMClassifier" in notebook_text
    assert "logistic_regression" in notebook_text
    assert "hist_gradient_boosting_classifier" in notebook_text
    assert "ridge" in notebook_text
    assert "D.features" in notebook_text
    assert "R.start" in notebook_text
    assert "不构成买卖建议" in conclusion
