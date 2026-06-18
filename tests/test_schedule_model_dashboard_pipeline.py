from __future__ import annotations

import json
from pathlib import Path

from src.schedule import ModelDashboardPipelineConfig, run_model_dashboard_refresh_once


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
    assert result.payload == {}
    assert "permanent failure" in result.detail
    assert events == ["model_dashboard_refresh:failed"]
