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
        [{"model": "ridge_mse", "segment_type": "all", "segment": "all", "diagnosis": "基准模型。"}]
    ).to_csv(matrix_card / "deltas_custom.csv", index=False)
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
                    "model_effect_deltas": "deltas_custom.csv",
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
    assert payload["model_effect_deltas"][0]["diagnosis"] == "基准模型。"
    assert payload["model_capabilities"][0]["paper_fidelity"] == "native_library"
    assert payload["artifacts"]["analysis_manifest"].endswith("analysis_manifest.json")
    assert payload["artifacts"]["model_comparison"].endswith("comparison_custom.csv")
    assert "predictions" not in payload["artifacts"]
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
