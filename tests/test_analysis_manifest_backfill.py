from __future__ import annotations

import json

import pandas as pd

from src.analysis import backfill_analysis_manifest, backfill_legacy_analysis_cards, scan_analysis_cards
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


def test_scan_analysis_cards_classifies_legacy_manifested_and_private_only_dirs(tmp_path):
    legacy = tmp_path / "legacy_card"
    _write_legacy_card(legacy)
    manifested = tmp_path / "manifested_card"
    _write_legacy_card(manifested)
    backfill_analysis_manifest(
        manifested,
        dataset_name="unit_manifested",
        universe=("SOXL",),
        calendar_path="calendar.txt",
        data_paths=("manifested_card",),
    )
    private_only = tmp_path / "private_only"
    private_only.mkdir()
    pd.DataFrame([{"model": "ridge_mse", "prediction": 0.1}]).to_csv(private_only / "predictions.csv", index=False)

    scan = scan_analysis_cards(tmp_path)

    assert list(scan["card_name"]) == ["legacy_card", "manifested_card", "private_only"]
    by_name = {row["card_name"]: row for row in scan.to_dict("records")}
    assert by_name["legacy_card"]["status"] == "backfillable"
    assert by_name["legacy_card"]["migration_action"] == "backfill_analysis_manifest"
    assert by_name["legacy_card"]["has_manifest"] is False
    assert by_name["legacy_card"]["public_sidecar_count"] >= 6
    assert "predictions" not in by_name["legacy_card"]["public_artifact_keys"]
    assert "daily_backtest" not in by_name["legacy_card"]["public_artifact_keys"]
    assert by_name["manifested_card"]["status"] == "already_manifested"
    assert by_name["manifested_card"]["migration_action"] == "keep_manifest"
    assert by_name["private_only"]["status"] == "not_analysis_card"
    assert by_name["private_only"]["private_sidecar_count"] == 1


def test_backfill_legacy_analysis_cards_only_migrates_backfillable_cards(tmp_path):
    legacy = tmp_path / "legacy_card"
    _write_legacy_card(legacy)
    manifested = tmp_path / "manifested_card"
    _write_legacy_card(manifested)
    backfill_analysis_manifest(
        manifested,
        dataset_name="unit_manifested",
        universe=("SOXL",),
        calendar_path="calendar.txt",
        data_paths=("manifested_card",),
    )

    result = backfill_legacy_analysis_cards(
        tmp_path,
        dataset_name="unit_batch",
        universe=("SOXL", "SOXX"),
        calendar_path="calendar.txt",
    )

    assert result["schema_version"] == "analysis.manifest_backfill_batch.v1"
    assert result["backfilled"] == 1
    assert result["skipped"] == 1
    rows = {row["card_name"]: row for row in result["cards"]}
    assert rows["legacy_card"]["action"] == "backfilled"
    assert rows["legacy_card"]["manifest_hash"]
    assert (legacy / "analysis_manifest.json").exists()
    manifest = json.loads((legacy / "analysis_manifest.json").read_text(encoding="utf-8"))
    assert manifest["dataset"]["name"] == "unit_batch"
    assert manifest["dataset"]["data_paths"] == [str(legacy)]
    assert "predictions" not in manifest["paths"]
    assert rows["manifested_card"]["action"] == "skipped"
    assert rows["manifested_card"]["reason"] == "already_manifested"
