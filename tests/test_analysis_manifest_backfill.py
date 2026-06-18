from __future__ import annotations

import json

import pandas as pd

from src.analysis import backfill_analysis_manifest
from src.dashboard import load_model_matrix_dashboard


def _write_legacy_card(card):
    card.mkdir(parents=True, exist_ok=True)
    (card / "analysis_summary.json").write_text(
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
    pd.DataFrame([{"rank": 1, "model": "ridge_mse", "return_pct": 3.0, "stability_score": 2.0}]).to_csv(
        card / "model_comparison.csv", index=False
    )
    pd.DataFrame([{"model": "ridge_mse", "regime": "bull", "return_pct": 3.0}]).to_csv(
        card / "regime_metrics.csv", index=False
    )
    pd.DataFrame([{"model": "ridge_mse", "window": "test", "return_pct": 3.0}]).to_csv(
        card / "window_metrics.csv", index=False
    )
    pd.DataFrame([{"model": "ridge_mse", "return_kind": "strategy_return", "annualized_return": 0.2}]).to_csv(
        card / "qlib_risk_metrics.csv", index=False
    )
    pd.DataFrame([{"model": "ridge_mse", "segment_type": "all", "segment": "all", "diagnosis": "基准模型。"}]).to_csv(
        card / "model_effect_deltas.csv", index=False
    )
    pd.DataFrame([{"model": "ridge_mse", "paper_fidelity": "native_library"}]).to_csv(
        card / "model_capabilities.csv", index=False
    )
    pd.DataFrame([{"model": "ridge_mse", "prediction": 0.1}]).to_csv(card / "predictions.csv", index=False)
    pd.DataFrame([{"model": "ridge_mse", "nav": 1.0}]).to_csv(card / "daily_backtest.csv", index=False)


def test_backfill_analysis_manifest_from_legacy_sidecars_writes_public_paths_and_hash(tmp_path):
    card = tmp_path / "legacy_card"
    _write_legacy_card(card)

    payload = backfill_analysis_manifest(
        card,
        dataset_name="unit_legacy",
        universe=("SOXL", "SOXX"),
        calendar_path="data/qlib/soxl/calendars/day.txt",
        data_paths=("legacy_card",),
    )

    manifest_path = card / "analysis_manifest.json"
    assert manifest_path.exists()
    assert payload["artifact_id"] == "legacy_card"
    assert payload["artifact_type"] == "analysis_report"
    assert payload["producer"] == "src.analysis.manifest_backfill"
    assert payload["paths"]["summary"] == "analysis_summary.json"
    assert payload["paths"]["model_comparison"] == "model_comparison.csv"
    assert payload["paths"]["qlib_risk_metrics"] == "qlib_risk_metrics.csv"
    assert "predictions" not in payload["paths"]
    assert "daily_backtest" not in payload["paths"]
    assert payload["metadata"]["generated_from"] == "legacy_sidecars"
    assert payload["metadata"]["analysis_report_schema_version"] == "analysis.report.v1"
    assert payload["metadata"]["table_counts"]["model_comparison"] == 1
    assert len(payload["manifest_hash"]) == 64

    saved = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert saved == payload

    dashboard_payload = load_model_matrix_dashboard(card)
    assert dashboard_payload["manifest_source"]["kind"] == "analysis_manifest"
    assert dashboard_payload["manifest_source"]["manifest_hash"] == payload["manifest_hash"]
    assert dashboard_payload["top_models"][0]["model"] == "ridge_mse"
    assert "predictions" not in dashboard_payload["artifacts"]


def test_backfill_analysis_manifest_refuses_to_overwrite_without_flag(tmp_path):
    card = tmp_path / "legacy_card"
    _write_legacy_card(card)
    existing = {"schema_version": "artifact.v1", "manifest_hash": "existing"}
    (card / "analysis_manifest.json").write_text(json.dumps(existing), encoding="utf-8")

    try:
        backfill_analysis_manifest(
            card,
            dataset_name="unit_legacy",
            universe=("SOXL",),
            calendar_path="calendar.txt",
            data_paths=("legacy_card",),
        )
    except FileExistsError as exc:
        assert "analysis_manifest.json already exists" in str(exc)
    else:
        raise AssertionError("expected FileExistsError")
