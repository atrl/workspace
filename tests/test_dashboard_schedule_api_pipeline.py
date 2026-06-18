from __future__ import annotations

import json

import pandas as pd

from src.api import PaperBrokerAdapter
from src.dashboard import load_model_matrix_dashboard, write_model_matrix_dashboard_export
from src.schedule import ScheduleJob, run_job_once


def test_dashboard_loads_quantbench_matrix_sidecars_without_raw_model_internals():
    payload = load_model_matrix_dashboard("research/quantbench_model_matrix_smoke_v1")

    assert payload["summary"]["model_count"] == 32
    assert payload["top_models"][0]["rank"] == 1
    assert "model_comparison" in payload["artifacts"]
    assert "qlib_risk_metrics" in payload["artifacts"]
    assert "model_effect_deltas" in payload["artifacts"]
    assert "predictions" not in payload["artifacts"]
    assert "daily_backtest" not in payload["artifacts"]


def test_dashboard_loads_model_run_ledger_as_public_coverage_artifact():
    payload = load_model_matrix_dashboard("research/quantbench_all_model_run_ledger_v1")

    assert payload["model_run_coverage"]["schema_version"] == "model.run_ledger_summary.v1"
    assert payload["model_run_coverage"]["expected_model_count"] == 32
    assert payload["model_run_coverage"]["trained_model_count"] == 32
    assert payload["model_run_coverage"]["backtested_model_count"] == 32
    assert payload["model_run_coverage"]["all_models_completed"] is True
    assert payload["model_run_coverage"]["missing_models"] == []
    assert len(payload["model_run_ledger"]) == 32
    assert payload["model_run_ledger"][0]["schema_version"] == "model.run_ledger.v1"
    assert {"model", "trained", "backtested", "analysis_recorded", "manifested"}.issubset(
        payload["model_run_ledger"][0]
    )
    assert "model_run_ledger" in payload["artifacts"]
    assert "model_run_ledger_summary" in payload["artifacts"]
    assert "predictions" not in payload["artifacts"]
    assert "daily_backtest" not in payload["artifacts"]


def test_dashboard_loads_model_review_briefing_and_segment_matrix_from_manifest():
    payload = load_model_matrix_dashboard("research/analysis_segment_effect_matrix_v1")

    assert payload["manifest_source"]["kind"] == "analysis_manifest"
    assert payload["analysis_overview"]["schema_version"] == "analysis.overview.v1"
    assert payload["analysis_overview"]["top_weakness"]["decision_hint"] in {
        "block_or_rework",
        "reduce_or_guardrail",
        "",
    }
    assert payload["analysis_overview"]["largest_relative_return_spread_segments"]
    assert payload["segment_effect_matrix"][0]["schema_version"] == "analysis.segment_effect_matrix.v1"
    assert payload["model_review_briefing"][0]["schema_version"] == "analysis.model_review_briefing.v1"
    assert payload["model_review_briefing"][0]["review_signal"] in {"weakness", "strength", "risk"}
    assert "segment_effect_matrix" in payload["artifacts"]
    assert "model_review_briefing" in payload["artifacts"]
    assert "predictions" not in payload["artifacts"]


def test_dashboard_loads_feature_set_matrix_with_aggregates_and_diagnostics():
    payload = load_model_matrix_dashboard("research/factor_feature_set_matrix_v1", top_n=5)

    assert payload["schema_version"] == "dashboard.model_matrix.v1"
    assert payload["summary"]["status"] == "feature_set_matrix_completed"
    assert payload["top_models"][0]["feature_set"]
    assert payload["top_models"][0]["base_model"]
    assert payload["feature_set_summary"]
    assert {"feature_set", "variant_count", "best_model", "avg_stability_score"}.issubset(
        payload["feature_set_summary"][0]
    )
    assert payload["qlib_risk_metrics"]
    assert payload["model_effect_deltas"]
    assert "diagnosis" in payload["model_effect_deltas"][0]
    assert "feature_set_specs" in payload["artifacts"]
    assert "factor_ic" in payload["artifacts"]
    assert "predictions" not in payload["artifacts"]
    assert "daily_backtest" not in payload["artifacts"]


def test_dashboard_loads_matrix_sidecars_from_analysis_manifest_without_fixed_filenames(tmp_path):
    matrix_card = tmp_path / "matrix_card"
    matrix_card.mkdir()

    (matrix_card / "summary_custom.json").write_text(
        json.dumps(
            {
                "schema_version": "analysis.report.v1",
                "status": "manifest_first_completed",
                "model_count": 1,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [{"rank": 1, "model": "ridge_mse", "return_pct": 2.5, "relative_return_pct": 1.0, "stability_score": 3.0}]
    ).to_csv(matrix_card / "comparison_custom.csv", index=False)
    pd.DataFrame([{"model": "ridge_mse", "regime": "bull", "return_pct": 2.5}]).to_csv(
        matrix_card / "regime_custom.csv", index=False
    )
    pd.DataFrame([{"model": "ridge_mse", "window": "test", "return_pct": 2.5}]).to_csv(
        matrix_card / "window_custom.csv", index=False
    )
    pd.DataFrame([{"model": "ridge_mse", "return_kind": "strategy_return", "annualized_return": 0.25}]).to_csv(
        matrix_card / "risk_custom.csv", index=False
    )
    pd.DataFrame(
        [
            {
                "schema_version": "objective.evaluation.v1",
                "model": "ridge_mse",
                "mse": 0.0001,
                "classification_accuracy": 1.0,
                "ic": 0.8,
                "rank_ic": 0.7,
            }
        ]
    ).to_csv(matrix_card / "objective_scores_custom.csv", index=False)
    pd.DataFrame(
        [
            {
                "schema_version": "objective.promotion_score.v1",
                "rank": 1,
                "model": "ridge_mse",
                "promotion_score": 12.5,
                "promotion_decision": "promote_candidate",
                "promotion_blockers": "",
            }
        ]
    ).to_csv(matrix_card / "promotion_scores_custom.csv", index=False)
    pd.DataFrame(
        [{"model": "ridge_mse", "segment_type": "all", "segment": "all", "diagnosis": "基准模型。"}]
    ).to_csv(matrix_card / "deltas_custom.csv", index=False)
    pd.DataFrame(
        [
            {
                "model": "ridge_mse",
                "segment_type": "all",
                "segment": "all",
                "best_relative_return_model": "ridge_mse",
                "worst_relative_return_model": "ridge_mse",
                "review_note": "基准模型。",
            }
        ]
    ).to_csv(matrix_card / "segment_matrix_custom.csv", index=False)
    pd.DataFrame(
        [
            {
                "model": "ridge_mse",
                "baseline_model": "ridge_mse",
                "segment_type": "regime",
                "segment": "bull",
                "review_signal": "strength",
                "decision_hint": "promote_candidate",
                "priority_rank": 1,
                "review_note": "ridge_mse 在 bull 强于基准。",
            }
        ]
    ).to_csv(matrix_card / "review_custom.csv", index=False)
    (matrix_card / "backtest_config_custom.json").write_text(
        json.dumps(
            {
                "schema_version": "backtest.config.v1",
                "start": "2026-01-01",
                "end": "2026-01-08",
                "cost_model": {"open_cost_bps": 1.0, "close_cost_bps": 1.0, "slippage_bps": 1.0},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "fold": 1,
                "train_start": "2026-01-01",
                "train_end": "2026-01-04",
                "test_start": "2026-01-05",
                "test_end": "2026-01-06",
                "schema_version": "backtest.walk_forward_split.v1",
            }
        ]
    ).to_csv(matrix_card / "walk_forward_custom.csv", index=False)
    pd.DataFrame([{"model": "ridge_mse", "paper_fidelity": "native_library"}]).to_csv(
        matrix_card / "capabilities_custom.csv", index=False
    )
    pd.DataFrame([{"model": "ridge_mse", "prediction": 0.1}]).to_csv(matrix_card / "predictions_private.csv", index=False)

    (matrix_card / "analysis_manifest.json").write_text(
        json.dumps(
            {
                "artifact_id": "manifest_first_card",
                "artifact_type": "analysis_report",
                "producer": "src.analysis.exports",
                "schema_version": "artifact.v1",
                "paths": {
                    "summary": "summary_custom.json",
                    "model_comparison": "comparison_custom.csv",
                    "regime_metrics": "regime_custom.csv",
                    "window_metrics": "window_custom.csv",
                    "qlib_risk_metrics": "risk_custom.csv",
                    "objective_scores": "objective_scores_custom.csv",
                    "promotion_scores": "promotion_scores_custom.csv",
                    "model_effect_deltas": "deltas_custom.csv",
                    "segment_effect_matrix": "segment_matrix_custom.csv",
                    "model_review_briefing": "review_custom.csv",
                    "backtest_config": "backtest_config_custom.json",
                    "walk_forward_splits": "walk_forward_custom.csv",
                    "model_capabilities": "capabilities_custom.csv",
                    "predictions": "predictions_private.csv",
                },
                "metadata": {"analysis_report_schema_version": "analysis.report.v1"},
                "manifest_hash": "x" * 64,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    payload = load_model_matrix_dashboard(matrix_card)

    assert payload["artifact_id"] == "manifest_first_card"
    assert payload["summary"]["status"] == "manifest_first_completed"
    assert payload["top_models"][0]["model"] == "ridge_mse"
    assert payload["regime_metrics"][0]["regime"] == "bull"
    assert payload["qlib_risk_metrics"][0]["annualized_return"] == 0.25
    assert payload["objective"]["scores"][0]["schema_version"] == "objective.evaluation.v1"
    assert payload["objective"]["promotion_scores"][0]["promotion_decision"] == "promote_candidate"
    assert payload["model_effect_deltas"][0]["diagnosis"] == "基准模型。"
    assert payload["segment_effect_matrix"][0]["best_relative_return_model"] == "ridge_mse"
    assert payload["model_review_briefing"][0]["review_signal"] == "strength"
    assert payload["backtest"]["config"]["schema_version"] == "backtest.config.v1"
    assert payload["backtest"]["walk_forward_splits"][0]["fold"] == 1
    assert payload["model_capabilities"][0]["paper_fidelity"] == "native_library"
    assert payload["artifacts"]["analysis_manifest"].endswith("analysis_manifest.json")
    assert payload["artifacts"]["model_comparison"].endswith("comparison_custom.csv")
    assert payload["artifacts"]["objective_scores"].endswith("objective_scores_custom.csv")
    assert payload["artifacts"]["promotion_scores"].endswith("promotion_scores_custom.csv")
    assert payload["artifacts"]["model_review_briefing"].endswith("review_custom.csv")
    assert payload["artifacts"]["backtest_config"].endswith("backtest_config_custom.json")
    assert payload["artifacts"]["walk_forward_splits"].endswith("walk_forward_custom.csv")
    assert "predictions" not in payload["artifacts"]
    assert "daily_backtest" not in payload["artifacts"]
    assert payload["manifest_source"]["kind"] == "analysis_manifest"
    assert payload["manifest_source"]["path"].endswith("analysis_manifest.json")
    assert payload["manifest_source"]["manifest_hash"] == "x" * 64


def test_dashboard_export_writes_manifest_style_payload(tmp_path):
    output_path = tmp_path / "dashboard_export.json"

    payload = write_model_matrix_dashboard_export(
        "research/factor_feature_set_matrix_v1",
        output_path=output_path,
        top_n=3,
    )

    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved["schema_version"] == "dashboard.model_matrix.v1"
    assert saved["artifact_type"] == "dashboard_export"
    assert saved["artifact_id"] == "factor_feature_set_matrix_v1"
    assert saved["top_models"] == payload["top_models"]
    assert len(saved["top_models"]) == 3


def test_dashboard_export_surfaces_window_regime_fidelity_and_attribution(tmp_path):
    matrix_card = tmp_path / "matrix_card"
    attribution_card = tmp_path / "attribution_card"
    matrix_card.mkdir()
    attribution_card.mkdir()

    (matrix_card / "analysis_summary.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "model_count": 2,
                "model_fidelity": {
                    "schema_version": "model.fidelity.v1",
                    "model_count": 2,
                    "fidelity_counts": {"native_library": 1, "proxy_feature_adapter": 1},
                    "proxy_models": ["gcn_mse"],
                    "native_models": ["ridge_mse"],
                    "promotion_blocker_count": 1,
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {"rank": 1, "model": "ridge_mse", "return_pct": 3.0, "stability_score": 2.0},
            {"rank": 2, "model": "gcn_mse", "return_pct": -2.0, "stability_score": -1.0},
        ]
    ).to_csv(matrix_card / "model_comparison.csv", index=False)
    pd.DataFrame([{"model": "ridge_mse", "regime": "bull", "return_pct": 3.0}]).to_csv(
        matrix_card / "regime_metrics.csv", index=False
    )
    pd.DataFrame([{"model": "ridge_mse", "window": "test", "return_pct": 3.0}]).to_csv(
        matrix_card / "window_metrics.csv", index=False
    )
    pd.DataFrame(
        [
            {
                "window": "test",
                "regime": "stress",
                "segment": "test::stress",
                "model": "gcn_mse",
                "return_pct": -4.0,
                "stability_score": -2.5,
            }
        ]
    ).to_csv(matrix_card / "window_regime_metrics.csv", index=False)
    pd.DataFrame(
        [
            {
                "model": "gcn_mse",
                "segment_type": "window_regime",
                "segment": "test::stress",
                "diagnosis": "弱于基准",
            }
        ]
    ).to_csv(matrix_card / "model_effect_deltas.csv", index=False)
    pd.DataFrame(
        [{"model": "gcn_mse", "paper_fidelity": "proxy_feature_adapter", "promotion_blocker": "graph data"}]
    ).to_csv(matrix_card / "model_capabilities.csv", index=False)

    (attribution_card / "analysis_summary.json").write_text(
        json.dumps(
            {
                "schema_version": "analysis.model_attribution.batch.v1",
                "status": "completed",
                "models": ["ridge_mse", "gcn_mse"],
                "proxy_models": ["gcn_mse"],
                "attribution_kinds": ["sensitivity"],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {"model": "gcn_mse", "feature": "soxx_return_5d", "normalized_importance": 0.6},
            {"model": "ridge_mse", "feature": "vix_return_1d", "normalized_importance": 0.4},
        ]
    ).to_csv(attribution_card / "top_features_by_model.csv", index=False)

    payload = load_model_matrix_dashboard(matrix_card, attribution_card_dir=attribution_card)

    assert payload["model_fidelity"]["fidelity_counts"]["proxy_feature_adapter"] == 1
    assert payload["model_capabilities"][0]["model"] == "gcn_mse"
    assert payload["window_regime_metrics"][0]["segment"] == "test::stress"
    assert payload["attribution"]["models"] == ["ridge_mse", "gcn_mse"]
    assert payload["attribution"]["attribution_kinds"] == {"sensitivity": None}
    assert payload["attribution"]["top_features_by_model"][0]["feature"] == "soxx_return_5d"
    assert "window_regime_metrics" in payload["artifacts"]
    assert "attribution_top_features" in payload["artifacts"]
    assert "predictions" not in payload["artifacts"]


def test_dashboard_export_limits_attribution_top_features_per_model(tmp_path):
    matrix_card = tmp_path / "matrix_card"
    attribution_card = tmp_path / "attribution_card"
    matrix_card.mkdir()
    attribution_card.mkdir()

    (matrix_card / "analysis_summary.json").write_text(
        json.dumps({"status": "completed", "model_count": 2}, ensure_ascii=False),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {"rank": 1, "model": "ridge_mse", "return_pct": 3.0, "stability_score": 2.0},
            {"rank": 2, "model": "gcn_mse", "return_pct": 1.0, "stability_score": 1.0},
        ]
    ).to_csv(matrix_card / "model_comparison.csv", index=False)
    pd.DataFrame([{"model": "ridge_mse", "regime": "bull", "return_pct": 3.0}]).to_csv(
        matrix_card / "regime_metrics.csv", index=False
    )
    pd.DataFrame([{"model": "ridge_mse", "window": "test", "return_pct": 3.0}]).to_csv(
        matrix_card / "window_metrics.csv", index=False
    )

    (attribution_card / "analysis_summary.json").write_text(
        json.dumps(
            {
                "schema_version": "analysis.model_attribution.batch.v1",
                "status": "completed",
                "models": ["gcn_mse", "ridge_mse"],
                "proxy_models": ["gcn_mse"],
                "attribution_kinds": {"sensitivity": 5},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {"model": "gcn_mse", "feature": "gcn_f1", "normalized_importance": 0.6},
            {"model": "gcn_mse", "feature": "gcn_f2", "normalized_importance": 0.5},
            {"model": "gcn_mse", "feature": "gcn_f3", "normalized_importance": 0.4},
            {"model": "ridge_mse", "feature": "ridge_f1", "normalized_importance": 0.7},
            {"model": "ridge_mse", "feature": "ridge_f2", "normalized_importance": 0.3},
        ]
    ).to_csv(attribution_card / "top_features_by_model.csv", index=False)

    payload = load_model_matrix_dashboard(matrix_card, top_n=2, attribution_card_dir=attribution_card)

    features_by_model = {
        model: group["feature"].tolist()
        for model, group in pd.DataFrame(payload["attribution"]["top_features_by_model"]).groupby("model")
    }
    assert features_by_model == {
        "gcn_mse": ["gcn_f1", "gcn_f2"],
        "ridge_mse": ["ridge_f1", "ridge_f2"],
    }


def test_schedule_run_job_once_uses_lock_and_reports_notification(tmp_path):
    events: list[str] = []
    calls = {"count": 0}

    def task():
        calls["count"] += 1
        return {"rows": 32}

    job = ScheduleJob(name="dashboard_export", task="refresh_dashboard", cron="manual")
    result = run_job_once(job, task=task, lock_dir=tmp_path, notify=events.append)

    assert result.status == "success"
    assert result.payload == {"rows": 32}
    assert calls["count"] == 1
    assert events == ["dashboard_export:success"]
    assert not (tmp_path / "dashboard_export.lock").exists()


def test_schedule_run_job_once_skips_when_lock_exists(tmp_path):
    (tmp_path / "dashboard_export.lock").write_text(json.dumps({"owner": "test"}), encoding="utf-8")
    job = ScheduleJob(name="dashboard_export", task="refresh_dashboard", cron="manual")
    result = run_job_once(job, task=lambda: {"should_not_run": True}, lock_dir=tmp_path)

    assert result.status == "locked"
    assert result.payload == {}


def test_paper_broker_places_target_position_and_reports_exposure():
    broker = PaperBrokerAdapter(cash=10000.0, shares=0)
    ok, detail = broker.healthcheck()

    snapshot = broker.place_target_position(target_position=0.35, price=100.0)

    assert ok is True
    assert "ready" in detail
    assert snapshot.shares == 35
    assert snapshot.cash == 6500.0
    assert abs(snapshot.position - 0.35) < 1e-12


def test_production_paper_sim_summary_records_review_gate_context():
    summary = json.loads(open("production/paper_sim_replay_v1/production_summary.json", encoding="utf-8").read())
    intent = json.loads(open("production/paper_sim_replay_v1/execution_intent.json", encoding="utf-8").read())

    assert summary["schema_version"] == "production.paper_sim_replay.v1"
    assert summary["live_broker_enabled"] is False
    assert summary["execution_gate"] in {"paper_review", "blocked_to_min_position"}
    assert summary["analysis_overview_schema"] == "analysis.overview.v1"
    assert summary["promotion_decision"]
    assert summary["backtest_schema_version"] == "backtest.config.v1"
    assert summary["backtest_walk_forward_fold_count"] >= 1
    assert "regime_gap=" in " ".join(summary["notifications"])
    assert intent["metadata"]["execution_gate"] == summary["execution_gate"]
    assert intent["metadata"]["analysis_overview_schema_version"] == summary["analysis_overview_schema"]
    assert intent["metadata"]["backtest_schema_version"] == summary["backtest_schema_version"]
