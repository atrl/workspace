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
    assert "soxl_full" in notebook_text
    assert "run_cognitive_alpha_evolution" in notebook_text
    assert "D.features" in notebook_text
    assert "R.start" in notebook_text
    assert "不构成买卖建议" in conclusion


def test_cognitive_alpha_position_model_card_records_lightgbm_benchmark_and_signal():
    card = ROOT / "research" / "cognitive_alpha_position_model_v1"
    summary = _summary(card)
    notebook_text = (card / "cognitive_alpha_position_model_v1.ipynb").read_text(encoding="utf-8")
    conclusion = (card / "conclusion.md").read_text(encoding="utf-8")

    assert summary["experiment"] == "cognitive_alpha_position_model_v1"
    assert summary["card_type"] == "diagnostic_cognitive_alpha_position_model"
    assert summary["paper_reference"]["arxiv_id"] == "2511.18850"
    assert summary["segments"]["test"][1] == summary["as_of"]
    assert summary["evolution"]["workflow"] == "cognitive_alpha_evolution"
    assert summary["position_model"]["workflow"] == "cognitive_alpha_position_model"
    assert summary["position_model"]["model_families"] == ["lgbm_classifier"]
    assert summary["position_model"]["selected_candidate"]["test"]["max_position"] <= 1.0
    assert summary["position_model"]["selected_candidate"]["test"]["days_above_100"] == 0
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
        "failure_modes",
        "latest_signal",
        "model_selection",
        "position_summary",
    }
    assert set(summary["artifact_policy"]["recorder_only_artifacts"]) == {
        "candidate_pool",
        "fitness",
        "lineage",
        "positions",
        "quality",
        "rejected",
        "state",
    }
    for rel_path in summary["sidecars"].values():
        assert (card / rel_path).exists(), rel_path
    assert not (card / "positions.csv").exists()
    recorder_dir = Path(summary["recorder_local_dir"])
    assert (recorder_dir / "artifacts" / "cognitive_alpha_position" / "positions.csv").exists()
    assert (recorder_dir / "artifacts" / "cognitive_alpha_evolution" / "state.json").exists()
    assert "run_cognitive_alpha_evolution" in notebook_text
    assert "run_cognitive_alpha_position_model" in notebook_text
    assert "LGBMClassifier" in notebook_text
    assert "D.features" in notebook_text
    assert "R.start" in notebook_text
    assert "不构成买卖建议" in conclusion
