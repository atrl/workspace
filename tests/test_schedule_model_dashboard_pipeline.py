from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.schedule import ModelDashboardPipelineConfig, classify_market_calendar_day, run_model_dashboard_refresh_once


def test_model_dashboard_refresh_runs_matrix_then_dashboard_export_with_lock_and_notify(tmp_path):
    events: list[str] = []
    calls: list[tuple[str, Path]] = []

    def matrix_runner(*, card_dir: Path, start: str, end: str, train_end: str, model_names, top_n_factors: int):
        calls.append(("matrix", card_dir))
        card_dir.mkdir(parents=True, exist_ok=True)
        (card_dir / "analysis_summary.json").write_text(
            json.dumps({"status": "feature_set_matrix_completed", "model_count": len(tuple(model_names))}),
            encoding="utf-8",
        )
        return {"variant_count": 2}

    def dashboard_exporter(card_dir: Path, *, output_path: Path, top_n: int):
        calls.append(("dashboard", output_path))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": "dashboard.model_matrix.v1",
            "artifact_type": "dashboard_export",
            "artifact_id": card_dir.name,
            "top_models": [{"model": "ridge_mse__all_features", "rank": 1}],
        }
        output_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return payload

    config = ModelDashboardPipelineConfig(
        name="model_dashboard_refresh",
        card_dir=tmp_path / "matrix_card",
        dashboard_output_path=tmp_path / "dashboard" / "dashboard_export.json",
        start="2026-01-01",
        end="2026-06-09",
        train_end="2026-04-30",
        model_names=("ridge_mse", "random_forest_clf"),
        top_n_factors=3,
        retries=1,
    )

    result = run_model_dashboard_refresh_once(
        config,
        lock_dir=tmp_path / "locks",
        notify=events.append,
        matrix_runner=matrix_runner,
        dashboard_exporter=dashboard_exporter,
    )

    assert result.status == "success"
    assert result.payload["dashboard_export"].endswith("dashboard_export.json")
    assert result.payload["dashboard_schema_version"] == "dashboard.model_matrix.v1"
    assert result.payload["attempts"] == 1
    assert calls == [("matrix", config.card_dir), ("dashboard", config.dashboard_output_path)]
    assert events == ["model_dashboard_refresh:success"]
    assert not (tmp_path / "locks" / "model_dashboard_refresh.lock").exists()


def test_model_dashboard_refresh_skips_closed_market_day_before_matrix_or_export(tmp_path):
    events: list[str] = []
    calls = {"matrix": 0, "dashboard": 0}
    decision = classify_market_calendar_day(
        "2026-01-01",
        trading_calendar=("2026-01-02",),
        holiday_reasons={"2026-01-01": "New Year's Day"},
    )

    def matrix_runner(**kwargs):
        calls["matrix"] += 1
        Path(kwargs["card_dir"]).mkdir(parents=True, exist_ok=True)
        return {"variant_count": 1}

    def dashboard_exporter(card_dir: Path, *, output_path: Path, top_n: int):
        calls["dashboard"] += 1
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps({"schema_version": "dashboard.model_matrix.v1"}), encoding="utf-8")
        return {"schema_version": "dashboard.model_matrix.v1"}

    config = ModelDashboardPipelineConfig(
        name="model_dashboard_refresh",
        card_dir=tmp_path / "matrix_card",
        dashboard_output_path=tmp_path / "dashboard" / "dashboard_export.json",
        start="2026-01-01",
        end="2026-06-09",
        train_end="2026-04-30",
        market_calendar_decision=decision,
    )

    result = run_model_dashboard_refresh_once(
        config,
        lock_dir=tmp_path / "locks",
        notify=events.append,
        matrix_runner=matrix_runner,
        dashboard_exporter=dashboard_exporter,
    )

    assert result.status == "skipped"
    assert result.detail == "New Year's Day"
    assert result.payload["market_calendar_decision"]["market_status"] == "closed"
    assert calls == {"matrix": 0, "dashboard": 0}
    assert events == []
    assert not config.card_dir.exists()
    assert not config.dashboard_output_path.exists()
    assert not (tmp_path / "locks" / "model_dashboard_refresh.lock").exists()


def test_model_dashboard_refresh_reports_dashboard_manifest_source(tmp_path):
    def matrix_runner(**kwargs):
        Path(kwargs["card_dir"]).mkdir(parents=True, exist_ok=True)
        return {"variant_count": 1, "status": "feature_set_matrix_completed"}

    def dashboard_exporter(card_dir: Path, *, output_path: Path, top_n: int):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": "dashboard.model_matrix.v1",
            "artifact_id": "manifest_first_card",
            "top_models": [{"model": "ridge_mse", "rank": 1}],
            "manifest_source": {
                "kind": "analysis_manifest",
                "path": str(card_dir / "analysis_manifest.json"),
                "schema_version": "artifact.v1",
                "manifest_hash": "a" * 64,
                "producer": "src.analysis.exports",
            },
        }
        output_path.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    config = ModelDashboardPipelineConfig(
        name="model_dashboard_refresh",
        card_dir=tmp_path / "matrix_card",
        dashboard_output_path=tmp_path / "dashboard_export.json",
        start="2026-01-01",
        end="2026-06-09",
        train_end="2026-04-30",
        model_names=("ridge_mse",),
    )

    result = run_model_dashboard_refresh_once(
        config,
        lock_dir=tmp_path / "locks",
        matrix_runner=matrix_runner,
        dashboard_exporter=dashboard_exporter,
    )

    assert result.status == "success"
    assert result.payload["dashboard_artifact_id"] == "manifest_first_card"
    assert result.payload["dashboard_manifest_kind"] == "analysis_manifest"
    assert result.payload["dashboard_manifest_path"].endswith("analysis_manifest.json")
    assert result.payload["dashboard_manifest_hash"] == "a" * 64
    assert result.payload["dashboard_manifest_producer"] == "src.analysis.exports"


def test_model_dashboard_refresh_preflights_legacy_card_without_default_backfill(tmp_path):
    def matrix_runner(**kwargs):
        card_dir = Path(kwargs["card_dir"])
        card_dir.mkdir(parents=True, exist_ok=True)
        (card_dir / "analysis_summary.json").write_text(
            json.dumps(
                {
                    "schema_version": "analysis.report.v1",
                    "status": "legacy_completed",
                    "start": "2026-01-01",
                    "end": "2026-06-09",
                    "model_count": 1,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        pd.DataFrame([{"rank": 1, "model": "ridge_mse", "return_pct": 2.5, "stability_score": 2.0}]).to_csv(
            card_dir / "model_comparison.csv", index=False
        )
        pd.DataFrame([{"model": "ridge_mse", "regime": "bull", "return_pct": 2.5}]).to_csv(
            card_dir / "regime_metrics.csv", index=False
        )
        pd.DataFrame([{"model": "ridge_mse", "window": "test", "return_pct": 2.5}]).to_csv(
            card_dir / "window_metrics.csv", index=False
        )
        pd.DataFrame([{"model": "ridge_mse", "prediction": 0.1}]).to_csv(
            card_dir / "predictions.csv", index=False
        )
        return {"variant_count": 1, "status": "feature_set_matrix_completed"}

    def dashboard_exporter(card_dir: Path, *, output_path: Path, top_n: int):
        output_path.write_text(json.dumps({"schema_version": "dashboard.model_matrix.v1", "top_models": []}))
        return {"schema_version": "dashboard.model_matrix.v1", "top_models": []}

    config = ModelDashboardPipelineConfig(
        name="model_dashboard_refresh",
        card_dir=tmp_path / "matrix_card",
        dashboard_output_path=tmp_path / "dashboard_export.json",
        start="2026-01-01",
        end="2026-06-09",
        train_end="2026-04-30",
        model_names=("ridge_mse",),
        analysis_manifest_preflight=True,
        analysis_manifest_auto_backfill=False,
    )

    result = run_model_dashboard_refresh_once(
        config,
        lock_dir=tmp_path / "locks",
        matrix_runner=matrix_runner,
        dashboard_exporter=dashboard_exporter,
    )

    assert result.status == "success"
    assert result.payload["analysis_manifest_preflight_status"] == "needs_manifest"
    assert result.payload["analysis_manifest_preflight_action"] == "review_backfill"
    assert result.payload["analysis_manifest_preflight_public_sidecar_count"] >= 4
    assert result.payload["analysis_manifest_preflight_private_sidecar_count"] == 1
    assert not (config.card_dir / "analysis_manifest.json").exists()


def test_model_dashboard_refresh_can_auto_backfill_legacy_manifest_before_dashboard_export(tmp_path):
    def matrix_runner(**kwargs):
        card_dir = Path(kwargs["card_dir"])
        card_dir.mkdir(parents=True, exist_ok=True)
        (card_dir / "analysis_summary.json").write_text(
            json.dumps(
                {
                    "schema_version": "analysis.report.v1",
                    "status": "legacy_completed",
                    "start": "2026-01-01",
                    "end": "2026-06-09",
                    "model_count": 1,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        pd.DataFrame([{"rank": 1, "model": "ridge_mse", "return_pct": 2.5, "stability_score": 2.0}]).to_csv(
            card_dir / "model_comparison.csv", index=False
        )
        pd.DataFrame([{"model": "ridge_mse", "regime": "bull", "return_pct": 2.5}]).to_csv(
            card_dir / "regime_metrics.csv", index=False
        )
        pd.DataFrame([{"model": "ridge_mse", "window": "test", "return_pct": 2.5}]).to_csv(
            card_dir / "window_metrics.csv", index=False
        )
        pd.DataFrame(
            [{"model": "ridge_mse", "segment_type": "all", "segment": "all", "diagnosis": "基准模型。"}]
        ).to_csv(card_dir / "model_effect_deltas.csv", index=False)
        pd.DataFrame([{"model": "ridge_mse", "prediction": 0.1}]).to_csv(
            card_dir / "predictions.csv", index=False
        )
        return {"variant_count": 1, "status": "feature_set_matrix_completed"}

    config = ModelDashboardPipelineConfig(
        name="model_dashboard_refresh",
        card_dir=tmp_path / "matrix_card",
        dashboard_output_path=tmp_path / "dashboard_export.json",
        start="2026-01-01",
        end="2026-06-09",
        train_end="2026-04-30",
        model_names=("ridge_mse",),
        analysis_manifest_preflight=True,
        analysis_manifest_auto_backfill=True,
        analysis_manifest_dataset_name="unit_schedule_backfill",
        analysis_manifest_universe=("SOXL", "SOXX"),
        analysis_manifest_calendar_path="data/qlib/soxl/calendars/day.txt",
    )

    result = run_model_dashboard_refresh_once(
        config,
        lock_dir=tmp_path / "locks",
        matrix_runner=matrix_runner,
    )

    manifest_path = config.card_dir / "analysis_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dashboard = json.loads(config.dashboard_output_path.read_text(encoding="utf-8"))

    assert result.status == "success"
    assert result.payload["analysis_manifest_preflight_status"] == "backfilled"
    assert result.payload["analysis_manifest_preflight_action"] == "backfilled"
    assert result.payload["analysis_manifest_preflight_manifest_hash"] == manifest["manifest_hash"]
    assert result.payload["dashboard_manifest_kind"] == "analysis_manifest"
    assert result.payload["dashboard_manifest_hash"] == manifest["manifest_hash"]
    assert manifest["dataset"]["name"] == "unit_schedule_backfill"
    assert manifest["dataset"]["universe"] == ["SOXL", "SOXX"]
    assert "predictions" not in manifest["paths"]
    assert dashboard["manifest_source"]["kind"] == "analysis_manifest"
    assert dashboard["top_models"][0]["model"] == "ridge_mse"


def test_model_dashboard_refresh_defaults_to_manifest_preflight_and_backfill(tmp_path):
    def matrix_runner(**kwargs):
        card_dir = Path(kwargs["card_dir"])
        card_dir.mkdir(parents=True, exist_ok=True)
        (card_dir / "analysis_summary.json").write_text(
            json.dumps(
                {
                    "schema_version": "analysis.report.v1",
                    "status": "legacy_default_backfill_completed",
                    "start": "2026-01-01",
                    "end": "2026-06-09",
                    "model_count": 1,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        pd.DataFrame([{"rank": 1, "model": "ridge_mse", "return_pct": 2.5, "stability_score": 2.0}]).to_csv(
            card_dir / "model_comparison.csv", index=False
        )
        pd.DataFrame([{"model": "ridge_mse", "regime": "bull", "return_pct": 2.5}]).to_csv(
            card_dir / "regime_metrics.csv", index=False
        )
        pd.DataFrame([{"model": "ridge_mse", "window": "test", "return_pct": 2.5}]).to_csv(
            card_dir / "window_metrics.csv", index=False
        )
        pd.DataFrame([{"model": "ridge_mse", "prediction": 0.1}]).to_csv(
            card_dir / "predictions.csv", index=False
        )
        return {"variant_count": 1, "status": "feature_set_matrix_completed"}

    config = ModelDashboardPipelineConfig(
        name="model_dashboard_refresh",
        card_dir=tmp_path / "matrix_card",
        dashboard_output_path=tmp_path / "dashboard_export.json",
        start="2026-01-01",
        end="2026-06-09",
        train_end="2026-04-30",
        model_names=("ridge_mse",),
    )

    result = run_model_dashboard_refresh_once(
        config,
        lock_dir=tmp_path / "locks",
        matrix_runner=matrix_runner,
    )
    manifest = json.loads((config.card_dir / "analysis_manifest.json").read_text(encoding="utf-8"))
    dashboard = json.loads(config.dashboard_output_path.read_text(encoding="utf-8"))

    assert result.status == "success"
    assert result.payload["analysis_manifest_preflight_status"] == "backfilled"
    assert result.payload["analysis_manifest_preflight_action"] == "backfilled"
    assert result.payload["analysis_manifest_preflight_manifest_hash"] == manifest["manifest_hash"]
    assert result.payload["dashboard_manifest_kind"] == "analysis_manifest"
    assert dashboard["manifest_source"]["kind"] == "analysis_manifest"
    assert dashboard["top_models"][0]["model"] == "ridge_mse"
    assert manifest["dataset"]["name"] == "schedule_dashboard_refresh"
    assert manifest["dataset"]["universe"] == ["SOXL"]
    assert manifest["dataset"]["calendar_path"] == "data/qlib/soxl/calendars/day.txt"
    assert "predictions" not in manifest["paths"]


def test_model_dashboard_refresh_can_send_payload_aware_success_notification(tmp_path):
    events: list[str] = []

    def matrix_runner(**kwargs):
        card_dir = Path(kwargs["card_dir"])
        card_dir.mkdir(parents=True, exist_ok=True)
        (card_dir / "analysis_summary.json").write_text(
            json.dumps(
                {
                    "schema_version": "analysis.report.v1",
                    "status": "legacy_completed",
                    "start": "2026-01-01",
                    "end": "2026-06-09",
                    "model_count": 1,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        pd.DataFrame([{"rank": 1, "model": "ridge_mse", "return_pct": 2.5, "stability_score": 2.0}]).to_csv(
            card_dir / "model_comparison.csv", index=False
        )
        pd.DataFrame([{"model": "ridge_mse", "regime": "bull", "return_pct": 2.5}]).to_csv(
            card_dir / "regime_metrics.csv", index=False
        )
        pd.DataFrame([{"model": "ridge_mse", "window": "test", "return_pct": 2.5}]).to_csv(
            card_dir / "window_metrics.csv", index=False
        )
        return {"variant_count": 1, "status": "feature_set_matrix_completed"}

    config = ModelDashboardPipelineConfig(
        name="model_dashboard_refresh",
        card_dir=tmp_path / "matrix_card",
        dashboard_output_path=tmp_path / "dashboard_export.json",
        start="2026-01-01",
        end="2026-06-09",
        train_end="2026-04-30",
        model_names=("ridge_mse",),
        analysis_manifest_preflight=True,
        analysis_manifest_auto_backfill=True,
        analysis_manifest_dataset_name="unit_schedule_notify",
        analysis_manifest_universe=("SOXL",),
        analysis_manifest_calendar_path="data/qlib/soxl/calendars/day.txt",
        notify_details=True,
    )

    result = run_model_dashboard_refresh_once(
        config,
        lock_dir=tmp_path / "locks",
        notify=events.append,
        matrix_runner=matrix_runner,
    )

    assert result.status == "success"
    assert events[0].startswith(
        "model_dashboard_refresh:success | top_model=ridge_mse | dashboard_manifest=analysis_manifest "
        "| analysis_preflight=backfilled | attempts=1 | export="
        + str(config.dashboard_output_path)
    )
    assert "readiness=review pass=1 review=1 block=0" in events[0]


def test_model_dashboard_refresh_detailed_notification_includes_top_failure_mode(tmp_path):
    events: list[str] = []

    def matrix_runner(**kwargs):
        Path(kwargs["card_dir"]).mkdir(parents=True, exist_ok=True)
        return {"variant_count": 2, "status": "feature_set_matrix_completed"}

    def dashboard_exporter(card_dir: Path, *, output_path: Path, top_n: int):
        payload = {
            "schema_version": "dashboard.model_matrix.v1",
            "artifact_id": "failure_mode_card",
            "top_models": [{"model": "ridge_mse", "rank": 1}],
            "manifest_source": {
                "kind": "analysis_manifest",
                "path": str(card_dir / "analysis_manifest.json"),
                "schema_version": "artifact.v1",
                "manifest_hash": "b" * 64,
                "producer": "src.analysis.exports",
            },
            "failure_modes": [
                {
                    "model": "gcn_mse",
                    "segment": "stress_window::stress",
                    "failure_mode": "return_stability_lag",
                    "severity_score": 78.0,
                }
            ],
        }
        output_path.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    config = ModelDashboardPipelineConfig(
        name="model_dashboard_refresh",
        card_dir=tmp_path / "matrix_card",
        dashboard_output_path=tmp_path / "dashboard_export.json",
        start="2026-01-01",
        end="2026-06-09",
        train_end="2026-04-30",
        model_names=("ridge_mse", "gcn_mse"),
        notify_details=True,
    )

    result = run_model_dashboard_refresh_once(
        config,
        lock_dir=tmp_path / "locks",
        notify=events.append,
        matrix_runner=matrix_runner,
        dashboard_exporter=dashboard_exporter,
    )

    assert result.status == "success"
    assert result.payload["top_failure_mode"] == "return_stability_lag"
    assert result.payload["top_failure_segment"] == "stress_window::stress"
    assert "top_failure=return_stability_lag@stress_window::stress" in events[0]


def test_model_dashboard_refresh_detailed_notification_includes_top_review_briefing(tmp_path):
    events: list[str] = []

    def matrix_runner(**kwargs):
        Path(kwargs["card_dir"]).mkdir(parents=True, exist_ok=True)
        return {"variant_count": 2, "status": "feature_set_matrix_completed"}

    def dashboard_exporter(card_dir: Path, *, output_path: Path, top_n: int):
        payload = {
            "schema_version": "dashboard.model_matrix.v1",
            "artifact_id": "review_briefing_card",
            "top_models": [{"model": "ridge_mse", "rank": 1}],
            "manifest_source": {
                "kind": "analysis_manifest",
                "path": str(card_dir / "analysis_manifest.json"),
                "schema_version": "artifact.v1",
                "manifest_hash": "c" * 64,
                "producer": "src.analysis.exports",
            },
            "model_review_briefing": [
                {
                    "model": "gcn_mse",
                    "segment": "stress_window::stress",
                    "review_signal": "weakness",
                    "decision_hint": "block_or_rework",
                    "priority_score": 18.5,
                }
            ],
        }
        output_path.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    config = ModelDashboardPipelineConfig(
        name="model_dashboard_refresh",
        card_dir=tmp_path / "matrix_card",
        dashboard_output_path=tmp_path / "dashboard_export.json",
        start="2026-01-01",
        end="2026-06-09",
        train_end="2026-04-30",
        model_names=("ridge_mse", "gcn_mse"),
        notify_details=True,
    )

    result = run_model_dashboard_refresh_once(
        config,
        lock_dir=tmp_path / "locks",
        notify=events.append,
        matrix_runner=matrix_runner,
        dashboard_exporter=dashboard_exporter,
    )

    assert result.status == "success"
    assert result.payload["top_review_signal"] == "weakness"
    assert result.payload["top_review_decision_hint"] == "block_or_rework"
    assert result.payload["top_review_segment"] == "stress_window::stress"
    assert "top_review=weakness:block_or_rework@stress_window::stress" in events[0]


def test_model_dashboard_refresh_detailed_notification_includes_analysis_overview_regime_gap(tmp_path):
    events: list[str] = []

    def matrix_runner(**kwargs):
        Path(kwargs["card_dir"]).mkdir(parents=True, exist_ok=True)
        return {"variant_count": 2, "status": "feature_set_matrix_completed"}

    def dashboard_exporter(card_dir: Path, *, output_path: Path, top_n: int):
        payload = {
            "schema_version": "dashboard.model_matrix.v1",
            "artifact_id": "analysis_overview_card",
            "top_models": [{"model": "ridge_mse", "rank": 1}],
            "manifest_source": {
                "kind": "analysis_manifest",
                "path": str(card_dir / "analysis_manifest.json"),
                "schema_version": "artifact.v1",
                "manifest_hash": "e" * 64,
                "producer": "src.analysis.exports",
            },
            "analysis_overview": {
                "schema_version": "analysis.overview.v1",
                "baseline_model": "ridge_mse",
                "largest_relative_return_spread_segments": [
                    {
                        "segment": "stress_window::stress",
                        "best_relative_return_model": "ridge_mse",
                        "worst_relative_return_model": "gcn_mse",
                        "relative_return_pct_spread": 4.25,
                    }
                ],
                "top_weakness": {
                    "model": "gcn_mse",
                    "segment": "stress_window::stress",
                    "review_signal": "weakness",
                    "decision_hint": "block_or_rework",
                },
            },
        }
        output_path.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    config = ModelDashboardPipelineConfig(
        name="model_dashboard_refresh",
        card_dir=tmp_path / "matrix_card",
        dashboard_output_path=tmp_path / "dashboard_export.json",
        start="2026-01-01",
        end="2026-06-09",
        train_end="2026-04-30",
        model_names=("ridge_mse", "gcn_mse"),
        notify_details=True,
    )

    result = run_model_dashboard_refresh_once(
        config,
        lock_dir=tmp_path / "locks",
        notify=events.append,
        matrix_runner=matrix_runner,
        dashboard_exporter=dashboard_exporter,
    )

    assert result.status == "success"
    assert result.payload["analysis_overview_schema_version"] == "analysis.overview.v1"
    assert result.payload["analysis_overview_top_weakness_model"] == "gcn_mse"
    assert result.payload["analysis_overview_top_gap_segment"] == "stress_window::stress"
    assert result.payload["analysis_overview_top_gap_spread"] == 4.25
    assert "regime_gap=stress_window::stress:gcn_mse->ridge_mse/spread=4.25" in events[0]


def test_model_dashboard_refresh_detailed_notification_includes_model_difference_explanations(tmp_path):
    events: list[str] = []

    def matrix_runner(**kwargs):
        Path(kwargs["card_dir"]).mkdir(parents=True, exist_ok=True)
        return {"variant_count": 2, "status": "feature_set_matrix_completed"}

    def dashboard_exporter(card_dir: Path, *, output_path: Path, top_n: int):
        payload = {
            "schema_version": "dashboard.model_matrix.v1",
            "artifact_id": "difference_explanation_card",
            "top_models": [{"model": "ridge_mse", "rank": 1}],
            "manifest_source": {
                "kind": "analysis_manifest",
                "path": str(card_dir / "analysis_manifest.json"),
                "schema_version": "artifact.v1",
                "manifest_hash": "f" * 64,
                "producer": "src.analysis.exports",
            },
            "model_difference_explanations": [
                {
                    "schema_version": "analysis.model_difference_explanation.v1",
                    "priority_rank": 1,
                    "segment_type": "window_regime",
                    "segment": "stress_window::stress",
                    "window": "stress_window",
                    "regime": "stress",
                    "baseline_model": "ridge_mse",
                    "winner_model": "ridge_mse",
                    "loser_model": "gcn_mse",
                    "relative_return_pct_spread": 6.5,
                    "winner_vs_baseline_delta": 0.0,
                    "explanation_signal": "baseline_defends",
                    "decision_hint": "block_or_rework_loser",
                    "review_note": "中文 review：ridge_mse 在 stress 领先。",
                }
            ],
        }
        output_path.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    config = ModelDashboardPipelineConfig(
        name="model_dashboard_refresh",
        card_dir=tmp_path / "matrix_card",
        dashboard_output_path=tmp_path / "dashboard_export.json",
        start="2026-01-01",
        end="2026-06-09",
        train_end="2026-04-30",
        model_names=("ridge_mse", "gcn_mse"),
        notify_details=True,
    )

    result = run_model_dashboard_refresh_once(
        config,
        lock_dir=tmp_path / "locks",
        notify=events.append,
        matrix_runner=matrix_runner,
        dashboard_exporter=dashboard_exporter,
    )

    assert result.status == "success"
    assert result.payload["model_difference_schema_version"] == "analysis.model_difference_explanation.v1"
    assert result.payload["model_difference_count"] == 1
    assert result.payload["model_difference_top_signal"] == "baseline_defends"
    assert result.payload["model_difference_top_decision_hint"] == "block_or_rework_loser"
    assert result.payload["model_difference_top_segment"] == "stress_window::stress"
    assert result.payload["model_difference_top_winner_model"] == "ridge_mse"
    assert result.payload["model_difference_top_loser_model"] == "gcn_mse"
    assert result.payload["model_difference_top_spread"] == 6.5
    assert "model_difference=baseline_defends/block_or_rework_loser" in events[0]
    assert "stress_window::stress gcn_mse->ridge_mse spread=6.5" in events[0]


def test_model_dashboard_refresh_detailed_notification_includes_backtest_contract(tmp_path):
    events: list[str] = []

    def matrix_runner(**kwargs):
        Path(kwargs["card_dir"]).mkdir(parents=True, exist_ok=True)
        return {"variant_count": 2, "status": "feature_set_matrix_completed"}

    def dashboard_exporter(card_dir: Path, *, output_path: Path, top_n: int):
        payload = {
            "schema_version": "dashboard.model_matrix.v1",
            "artifact_id": "backtest_contract_card",
            "top_models": [{"model": "ridge_mse", "rank": 1}],
            "manifest_source": {
                "kind": "analysis_manifest",
                "path": str(card_dir / "analysis_manifest.json"),
                "schema_version": "artifact.v1",
                "manifest_hash": "d" * 64,
                "producer": "src.analysis.exports",
            },
            "backtest": {
                "config": {
                    "schema_version": "backtest.config.v1",
                    "start": "2026-05-01",
                    "end": "2026-06-08",
                    "cost_model": {
                        "open_cost_bps": 1.0,
                        "close_cost_bps": 1.0,
                        "slippage_bps": 1.5,
                    },
                },
                "walk_forward_splits": [
                    {
                        "fold": 1,
                        "train_start": "2026-01-01",
                        "train_end": "2026-04-30",
                        "test_start": "2026-05-01",
                        "test_end": "2026-05-13",
                    },
                    {
                        "fold": 2,
                        "train_start": "2026-01-14",
                        "train_end": "2026-05-13",
                        "test_start": "2026-05-14",
                        "test_end": "2026-05-26",
                    },
                ],
            },
        }
        output_path.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    config = ModelDashboardPipelineConfig(
        name="model_dashboard_refresh",
        card_dir=tmp_path / "matrix_card",
        dashboard_output_path=tmp_path / "dashboard_export.json",
        start="2026-01-01",
        end="2026-06-09",
        train_end="2026-04-30",
        model_names=("ridge_mse",),
        notify_details=True,
    )

    result = run_model_dashboard_refresh_once(
        config,
        lock_dir=tmp_path / "locks",
        notify=events.append,
        matrix_runner=matrix_runner,
        dashboard_exporter=dashboard_exporter,
    )

    assert result.status == "success"
    assert result.payload["backtest_schema_version"] == "backtest.config.v1"
    assert result.payload["backtest_slippage_bps"] == 1.5
    assert result.payload["walk_forward_fold_count"] == 2
    assert "backtest=backtest.config.v1 folds=2 slippage_bps=1.5" in events[0]


def test_model_dashboard_refresh_detailed_notification_includes_promotion_summary(tmp_path):
    events: list[str] = []

    def matrix_runner(**kwargs):
        Path(kwargs["card_dir"]).mkdir(parents=True, exist_ok=True)
        return {"variant_count": 2, "status": "feature_set_matrix_completed"}

    def dashboard_exporter(card_dir: Path, *, output_path: Path, top_n: int):
        payload = {
            "schema_version": "dashboard.model_matrix.v1",
            "artifact_id": "promotion_card",
            "top_models": [{"model": "ridge_mse", "rank": 1}],
            "manifest_source": {
                "kind": "analysis_manifest",
                "path": str(card_dir / "analysis_manifest.json"),
                "schema_version": "artifact.v1",
                "manifest_hash": "e" * 64,
                "producer": "src.analysis.exports",
            },
            "objective": {
                "promotion_scores": [
                    {
                        "rank": 1,
                        "model": "ridge_mse",
                        "promotion_score": 12.5,
                        "promotion_decision": "promote_candidate",
                        "promotion_blockers": "",
                    },
                    {
                        "rank": 2,
                        "model": "gcn_mse",
                        "promotion_score": -5.0,
                        "promotion_decision": "blocked",
                        "promotion_blockers": "max_drawdown_pct",
                    },
                ]
            },
        }
        output_path.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    config = ModelDashboardPipelineConfig(
        name="model_dashboard_refresh",
        card_dir=tmp_path / "matrix_card",
        dashboard_output_path=tmp_path / "dashboard_export.json",
        start="2026-01-01",
        end="2026-06-09",
        train_end="2026-04-30",
        model_names=("ridge_mse", "gcn_mse"),
        notify_details=True,
    )

    result = run_model_dashboard_refresh_once(
        config,
        lock_dir=tmp_path / "locks",
        notify=events.append,
        matrix_runner=matrix_runner,
        dashboard_exporter=dashboard_exporter,
    )

    assert result.status == "success"
    assert result.payload["top_promotion_model"] == "ridge_mse"
    assert result.payload["top_promotion_decision"] == "promote_candidate"
    assert result.payload["blocked_promotion_count"] == 1
    assert "promotion=promote_candidate:ridge_mse blocked=1" in events[0]


def test_model_dashboard_refresh_detailed_notification_includes_model_run_coverage(tmp_path):
    events: list[str] = []

    def matrix_runner(**kwargs):
        Path(kwargs["card_dir"]).mkdir(parents=True, exist_ok=True)
        return {"variant_count": 2, "status": "feature_set_matrix_completed"}

    def dashboard_exporter(card_dir: Path, *, output_path: Path, top_n: int):
        payload = {
            "schema_version": "dashboard.model_matrix.v1",
            "artifact_id": "coverage_card",
            "top_models": [{"model": "ridge_mse", "rank": 1}],
            "model_run_coverage": {
                "schema_version": "model.run_ledger_summary.v1",
                "expected_model_count": 2,
                "trained_model_count": 1,
                "backtested_model_count": 1,
                "all_models_completed": False,
                "missing_models": ["gcn_mse"],
            },
            "model_run_ledger": [
                {"model": "ridge_mse", "trained": True, "backtested": True, "analysis_recorded": True},
                {"model": "gcn_mse", "trained": False, "backtested": False, "analysis_recorded": False},
            ],
        }
        output_path.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    config = ModelDashboardPipelineConfig(
        name="model_dashboard_refresh",
        card_dir=tmp_path / "matrix_card",
        dashboard_output_path=tmp_path / "dashboard_export.json",
        start="2026-01-01",
        end="2026-06-09",
        train_end="2026-04-30",
        model_names=("ridge_mse", "gcn_mse"),
        notify_details=True,
    )

    result = run_model_dashboard_refresh_once(
        config,
        lock_dir=tmp_path / "locks",
        notify=events.append,
        matrix_runner=matrix_runner,
        dashboard_exporter=dashboard_exporter,
    )

    assert result.status == "success"
    assert result.payload["model_run_all_completed"] is False
    assert result.payload["model_run_expected_model_count"] == 2
    assert result.payload["model_run_trained_model_count"] == 1
    assert result.payload["model_run_backtested_model_count"] == 1
    assert result.payload["model_run_missing_model_count"] == 1
    assert result.payload["model_run_missing_models"] == "gcn_mse"
    assert "run_coverage=1/2 backtested=1/2 missing=1" in events[0]


def test_model_dashboard_refresh_detailed_notification_includes_requested_model_contract(tmp_path):
    events: list[str] = []

    def matrix_runner(**kwargs):
        Path(kwargs["card_dir"]).mkdir(parents=True, exist_ok=True)
        return {"variant_count": 32, "status": "matrix_completed"}

    def dashboard_exporter(card_dir: Path, *, output_path: Path, top_n: int):
        payload = {
            "schema_version": "dashboard.model_matrix.v1",
            "artifact_id": "requested_contract_card",
            "top_models": [{"model": "ridge_mse", "rank": 1}],
            "requested_model_contract": {
                "schema_version": "model.requested_contract_health.v1",
                "requested_model_count": 32,
                "registered_model_count": 32,
                "ledger_model_count": 31,
                "completed_requested_model_count": 31,
                "missing_from_registry": [],
                "missing_from_ledger": ["sthgcn_mse"],
                "incomplete_requested_models": ["sthgcn_mse"],
                "all_requested_models_completed": False,
                "adapter_statuses": ["paper_strict_adapter"],
            },
        }
        output_path.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    config = ModelDashboardPipelineConfig(
        name="model_dashboard_refresh",
        card_dir=tmp_path / "matrix_card",
        dashboard_output_path=tmp_path / "dashboard_export.json",
        start="2026-01-01",
        end="2026-06-09",
        train_end="2026-04-30",
        model_names=("ridge_mse",),
        notify_details=True,
    )

    result = run_model_dashboard_refresh_once(
        config,
        lock_dir=tmp_path / "locks",
        notify=events.append,
        matrix_runner=matrix_runner,
        dashboard_exporter=dashboard_exporter,
    )

    assert result.status == "success"
    assert result.payload["requested_model_count"] == 32
    assert result.payload["requested_model_completed_count"] == 31
    assert result.payload["requested_model_missing_count"] == 1
    assert result.payload["requested_model_incomplete_count"] == 1
    assert result.payload["requested_model_all_completed"] is False
    assert result.payload["requested_model_missing_models"] == "sthgcn_mse"
    assert "requested_models=31/32 missing=1 incomplete=1" in events[0]


def test_model_dashboard_refresh_detailed_notification_includes_readiness_overview(tmp_path):
    events: list[str] = []

    def matrix_runner(**kwargs):
        Path(kwargs["card_dir"]).mkdir(parents=True, exist_ok=True)
        return {"variant_count": 2, "status": "matrix_completed"}

    def dashboard_exporter(card_dir: Path, *, output_path: Path, top_n: int):
        payload = {
            "schema_version": "dashboard.model_matrix.v1",
            "artifact_id": "readiness_card",
            "manifest_source": {
                "kind": "analysis_manifest",
                "path": "matrix_card/analysis_manifest.json",
                "schema_version": "artifact.v1",
                "manifest_hash": "a" * 64,
                "producer": "src.analysis.exports",
            },
            "top_models": [{"model": "ridge_mse", "rank": 1}],
            "production_health": {
                "schema_version": "schedule.production_health.v1",
                "health_status": "healthy",
                "is_healthy": True,
                "missing_required_paths": [],
            },
            "notebook_card_readiness": {
                "schema_version": "dashboard.notebook_card_readiness.v1",
                "summary": {
                    "production_published_count": 1,
                    "production_template_count": 1,
                    "production_manifest_missing_count": 0,
                    "production_registry_mismatch_count": 0,
                },
                "production_cards": [],
            },
            "requested_model_contract": {
                "schema_version": "model.requested_contract_health.v1",
                "requested_model_count": 32,
                "registered_model_count": 32,
                "ledger_model_count": 31,
                "completed_requested_model_count": 31,
                "missing_from_registry": [],
                "missing_from_ledger": ["sthgcn_mse"],
                "incomplete_requested_models": ["sthgcn_mse"],
                "all_requested_models_completed": False,
            },
            "model_run_coverage": {
                "schema_version": "model.run_ledger_summary.v1",
                "expected_model_count": 2,
                "trained_model_count": 2,
                "backtested_model_count": 2,
                "all_models_completed": True,
                "missing_models": [],
            },
        }
        output_path.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    config = ModelDashboardPipelineConfig(
        name="model_dashboard_refresh",
        card_dir=tmp_path / "matrix_card",
        dashboard_output_path=tmp_path / "dashboard_export.json",
        start="2026-01-01",
        end="2026-06-09",
        train_end="2026-04-30",
        model_names=("ridge_mse",),
        notify_details=True,
    )

    result = run_model_dashboard_refresh_once(
        config,
        lock_dir=tmp_path / "locks",
        notify=events.append,
        matrix_runner=matrix_runner,
        dashboard_exporter=dashboard_exporter,
    )

    assert result.status == "success"
    assert result.payload["readiness_overview_schema_version"] == "dashboard.readiness_overview.v1"
    assert result.payload["readiness_overview_status"] == "review"
    assert result.payload["readiness_overview_gate_count"] == 5
    assert result.payload["readiness_overview_pass_count"] == 3
    assert result.payload["readiness_overview_review_count"] == 2
    assert result.payload["readiness_overview_block_count"] == 0
    assert (
        result.payload["readiness_overview_gates"]
        == "dashboard_manifest,production_health,notebook_cards,requested_models,model_runs"
    )
    assert result.payload["readiness_overview_block_gates"] == ""
    assert "readiness=review pass=3 review=2 block=0" in events[0]


def test_model_dashboard_refresh_passes_optional_attribution_card_to_dashboard_export(tmp_path):
    calls: list[tuple[str, Path | None]] = []
    attribution_card = tmp_path / "attribution_card"
    attribution_card.mkdir()
    (attribution_card / "analysis_summary.json").write_text(
        json.dumps({"status": "completed", "models": ["ridge_mse"]}),
        encoding="utf-8",
    )

    def matrix_runner(**kwargs):
        Path(kwargs["card_dir"]).mkdir(parents=True, exist_ok=True)
        return {"variant_count": 1, "status": "feature_set_matrix_completed"}

    def dashboard_exporter(card_dir: Path, *, output_path: Path, top_n: int, attribution_card_dir: Path | None = None):
        calls.append(("dashboard", attribution_card_dir))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": "dashboard.model_matrix.v1",
            "top_models": [{"model": "ridge_mse", "rank": 1}],
            "attribution": {"status": "completed", "models": ["ridge_mse"]},
        }
        output_path.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    config = ModelDashboardPipelineConfig(
        name="model_dashboard_refresh",
        card_dir=tmp_path / "matrix_card",
        dashboard_output_path=tmp_path / "dashboard_export.json",
        start="2026-01-01",
        end="2026-06-09",
        train_end="2026-04-30",
        model_names=("ridge_mse",),
        attribution_card_dir=attribution_card,
    )

    result = run_model_dashboard_refresh_once(
        config,
        lock_dir=tmp_path / "locks",
        matrix_runner=matrix_runner,
        dashboard_exporter=dashboard_exporter,
    )

    assert result.status == "success"
    assert calls == [("dashboard", attribution_card)]
    assert result.payload["attribution_card"] == str(attribution_card)
    assert result.payload["attribution_status"] == "completed"


def test_model_dashboard_refresh_passes_optional_production_health_to_dashboard_export(tmp_path):
    calls: list[Path | None] = []
    production_health_path = tmp_path / "production_health_export.json"
    production_health_path.write_text(
        json.dumps(
            {
                "schema_version": "schedule.production_health.v1",
                "strategy_id": "paper_sim_replay_v1",
                "health_status": "healthy",
                "is_healthy": True,
            }
        ),
        encoding="utf-8",
    )

    def matrix_runner(**kwargs):
        Path(kwargs["card_dir"]).mkdir(parents=True, exist_ok=True)
        return {"variant_count": 1, "status": "feature_set_matrix_completed"}

    def dashboard_exporter(
        card_dir: Path,
        *,
        output_path: Path,
        top_n: int,
        production_health_path: Path | None = None,
    ):
        calls.append(production_health_path)
        payload = {
            "schema_version": "dashboard.model_matrix.v1",
            "top_models": [{"model": "ridge_mse", "rank": 1}],
            "production_health": json.loads(Path(production_health_path).read_text(encoding="utf-8"))
            if production_health_path is not None
            else {},
        }
        output_path.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    config = ModelDashboardPipelineConfig(
        name="model_dashboard_refresh",
        card_dir=tmp_path / "matrix_card",
        dashboard_output_path=tmp_path / "dashboard_export.json",
        start="2026-01-01",
        end="2026-06-09",
        train_end="2026-04-30",
        model_names=("ridge_mse",),
        production_health_path=production_health_path,
    )

    result = run_model_dashboard_refresh_once(
        config,
        lock_dir=tmp_path / "locks",
        matrix_runner=matrix_runner,
        dashboard_exporter=dashboard_exporter,
    )

    assert result.status == "success"
    assert calls == [production_health_path]
    assert result.payload["dashboard_schema_version"] == "dashboard.model_matrix.v1"


def test_model_dashboard_refresh_passes_optional_notebook_card_manifest_to_dashboard_export(tmp_path):
    calls: list[Path | None] = []
    notebook_card_manifest_path = tmp_path / "notebook_card_manifest.json"
    notebook_card_manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "core.notebook_card_manifest.v1",
                "summary": {"production_published_count": 1, "production_template_count": 1},
                "cards": [],
                "manifest_hash": "n" * 64,
            }
        ),
        encoding="utf-8",
    )

    def matrix_runner(**kwargs):
        Path(kwargs["card_dir"]).mkdir(parents=True, exist_ok=True)
        return {"variant_count": 1, "status": "feature_set_matrix_completed"}

    def dashboard_exporter(
        card_dir: Path,
        *,
        output_path: Path,
        top_n: int,
        notebook_card_manifest_path: Path | None = None,
    ):
        calls.append(notebook_card_manifest_path)
        payload = {
            "schema_version": "dashboard.model_matrix.v1",
            "top_models": [{"model": "ridge_mse", "rank": 1}],
            "notebook_card_readiness": {
                "schema_version": "dashboard.notebook_card_readiness.v1",
                "manifest_hash": json.loads(Path(notebook_card_manifest_path).read_text(encoding="utf-8"))[
                    "manifest_hash"
                ],
                "summary": {"production_published_count": 1, "production_template_count": 1},
                "production_cards": [],
            }
            if notebook_card_manifest_path is not None
            else {},
        }
        output_path.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    config = ModelDashboardPipelineConfig(
        name="model_dashboard_refresh",
        card_dir=tmp_path / "matrix_card",
        dashboard_output_path=tmp_path / "dashboard_export.json",
        start="2026-01-01",
        end="2026-06-09",
        train_end="2026-04-30",
        model_names=("ridge_mse",),
        notebook_card_manifest_path=notebook_card_manifest_path,
    )

    result = run_model_dashboard_refresh_once(
        config,
        lock_dir=tmp_path / "locks",
        matrix_runner=matrix_runner,
        dashboard_exporter=dashboard_exporter,
    )

    assert result.status == "success"
    assert calls == [notebook_card_manifest_path]
    assert result.payload["dashboard_schema_version"] == "dashboard.model_matrix.v1"
    assert result.payload["notebook_card_production_published_count"] == 1
    assert result.payload["notebook_card_production_template_count"] == 1


def test_model_dashboard_refresh_optionally_writes_dashboard_html_page(tmp_path):
    calls: list[tuple[str, Path]] = []

    def matrix_runner(**kwargs):
        calls.append(("matrix", kwargs["card_dir"]))
        Path(kwargs["card_dir"]).mkdir(parents=True, exist_ok=True)
        return {"variant_count": 1, "status": "feature_set_matrix_completed"}

    def dashboard_exporter(card_dir: Path, *, output_path: Path, top_n: int):
        calls.append(("dashboard", output_path))
        payload = {
            "schema_version": "dashboard.model_matrix.v1",
            "artifact_id": "matrix_card",
            "summary": {"status": "feature_set_matrix_completed"},
            "model_fidelity": {
                "schema_version": "model.fidelity.v1",
                "proxy_models": [],
                "promotion_blocker_count": 0,
            },
            "model_capabilities": [{"model": "ridge_mse", "paper_fidelity": "native_library"}],
            "top_models": [{"model": "ridge_mse", "rank": 1, "return_pct": 1.0, "stability_score": 2.0}],
            "regime_metrics": [{"model": "ridge_mse", "regime": "bull", "return_pct": 1.0}],
            "window_metrics": [{"model": "ridge_mse", "window": "test", "return_pct": 1.0}],
            "window_regime_metrics": [
                {"model": "ridge_mse", "window": "test", "regime": "bull", "return_pct": 1.0}
            ],
            "model_effect_deltas": [],
            "attribution": {"status": "not_provided", "top_features_by_model": []},
        }
        output_path.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    config = ModelDashboardPipelineConfig(
        name="model_dashboard_refresh",
        card_dir=tmp_path / "matrix_card",
        dashboard_output_path=tmp_path / "dashboard_export.json",
        dashboard_page_output_path=tmp_path / "dashboard.html",
        dashboard_page_title="SOXL Refresh Page",
        start="2026-01-01",
        end="2026-06-09",
        train_end="2026-04-30",
        model_names=("ridge_mse",),
    )

    result = run_model_dashboard_refresh_once(
        config,
        lock_dir=tmp_path / "locks",
        matrix_runner=matrix_runner,
        dashboard_exporter=dashboard_exporter,
    )

    assert result.status == "success"
    assert calls == [("matrix", config.card_dir), ("dashboard", config.dashboard_output_path)]
    assert result.payload["dashboard_page"] == str(config.dashboard_page_output_path)
    assert result.payload["dashboard_page_schema_version"] == "dashboard.static_page.v1"
    assert result.payload["dashboard_view_model_schema_version"] == "dashboard.view_model.v1"
    assert result.payload["dashboard_page_model_card_count"] == 1
    assert config.dashboard_page_output_path.exists()
    assert "SOXL Refresh Page" in config.dashboard_page_output_path.read_text(encoding="utf-8")


def test_model_dashboard_refresh_retries_transient_failure_before_success(tmp_path):
    calls = {"count": 0}

    def flaky_matrix_runner(**kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("transient data source failure")
        Path(kwargs["card_dir"]).mkdir(parents=True, exist_ok=True)
        return {"variant_count": 1}

    def dashboard_exporter(card_dir: Path, *, output_path: Path, top_n: int):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"schema_version": "dashboard.model_matrix.v1", "top_models": []}
        output_path.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    config = ModelDashboardPipelineConfig(
        name="model_dashboard_refresh",
        card_dir=tmp_path / "matrix_card",
        dashboard_output_path=tmp_path / "dashboard_export.json",
        start="2026-01-01",
        end="2026-06-09",
        train_end="2026-04-30",
        model_names=("ridge_mse",),
        retries=2,
    )

    result = run_model_dashboard_refresh_once(
        config,
        lock_dir=tmp_path / "locks",
        matrix_runner=flaky_matrix_runner,
        dashboard_exporter=dashboard_exporter,
    )

    assert result.status == "success"
    assert result.payload["attempts"] == 2
    assert calls["count"] == 2


def test_model_dashboard_refresh_reports_failure_after_retry_budget(tmp_path):
    events: list[str] = []

    def failing_matrix_runner(**kwargs):
        raise RuntimeError("permanent failure")

    config = ModelDashboardPipelineConfig(
        name="model_dashboard_refresh",
        card_dir=tmp_path / "matrix_card",
        dashboard_output_path=tmp_path / "dashboard_export.json",
        start="2026-01-01",
        end="2026-06-09",
        train_end="2026-04-30",
        retries=1,
    )

    result = run_model_dashboard_refresh_once(
        config,
        lock_dir=tmp_path / "locks",
        notify=events.append,
        matrix_runner=failing_matrix_runner,
    )

    assert result.status == "failed"
    assert result.payload["attempts"] == 2
    assert result.payload["retry_schema_version"] == "schedule.retry_attempts.v1"
    assert result.payload["retry_error_count"] == 2
    assert result.payload["retry_errors"] == [
        {"attempt": 1, "error_type": "RuntimeError", "message": "permanent failure"},
        {"attempt": 2, "error_type": "RuntimeError", "message": "permanent failure"},
    ]
    assert "permanent failure" in result.detail
    assert events == ["model_dashboard_refresh:failed"]


def test_model_dashboard_refresh_detailed_failure_notification_includes_retry_metadata(tmp_path):
    events: list[str] = []

    def failing_matrix_runner(**kwargs):
        raise RuntimeError("permanent failure")

    config = ModelDashboardPipelineConfig(
        name="model_dashboard_refresh",
        card_dir=tmp_path / "matrix_card",
        dashboard_output_path=tmp_path / "dashboard_export.json",
        start="2026-01-01",
        end="2026-06-09",
        train_end="2026-04-30",
        retries=1,
        notify_details=True,
    )

    result = run_model_dashboard_refresh_once(
        config,
        lock_dir=tmp_path / "locks",
        notify=events.append,
        matrix_runner=failing_matrix_runner,
    )

    assert result.status == "failed"
    assert result.payload["retry_error_count"] == 2
    assert events == [
        "model_dashboard_refresh:failed | attempts=2 | retry_errors=2 | "
        "last_error=RuntimeError:permanent failure"
    ]
