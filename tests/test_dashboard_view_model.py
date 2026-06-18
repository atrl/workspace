from __future__ import annotations

import json
from pathlib import Path

from src.dashboard import DashboardViewFilters, build_dashboard_view_model, load_dashboard_view_model


def _dashboard_payload() -> dict:
    payload = json.loads(Path("research/schedule_dashboard_attribution_refresh_v1/dashboard_export.json").read_text())
    payload["artifacts"] = {"model_comparison": "/path/that/must/not/be/read.csv"}
    return payload


def test_dashboard_view_model_builds_page_controls_and_model_cards_from_export_payload_only():
    payload = _dashboard_payload()

    view = build_dashboard_view_model(payload)

    assert view["schema_version"] == "dashboard.view_model.v1"
    assert view["source_schema_version"] == "dashboard.model_matrix.v1"
    assert view["controls"]["models"]
    assert {"bull", "downtrend", "neutral", "stress"}.issubset(set(view["controls"]["regimes"]))
    assert view["controls"]["windows"] == ["test"]
    assert {"native_library", "proxy_feature_adapter"}.issubset(set(view["controls"]["paper_fidelities"]))
    assert "soxx_return_5d" in view["controls"]["attribution_features"]
    assert "artifacts" not in view

    first_card = view["model_cards"][0]
    assert {
        "model",
        "base_model",
        "feature_set",
        "paper_fidelity",
        "proxy_model",
        "top_attribution_features",
        "regime_metrics",
        "window_regime_metrics",
        "diagnostics",
    }.issubset(first_card)
    assert first_card["top_attribution_features"]
    assert view["comparison_table"]


def test_dashboard_view_model_filters_by_model_regime_window_fidelity_and_attribution_feature():
    payload = _dashboard_payload()

    view = build_dashboard_view_model(
        payload,
        filters=DashboardViewFilters(
            models=("gcn_mse__top_ic_factors",),
            regimes=("stress",),
            windows=("test",),
            paper_fidelities=("proxy_feature_adapter",),
            attribution_features=("soxx_return_5d",),
        ),
    )

    assert view["filters_applied"] == {
        "models": ["gcn_mse__top_ic_factors"],
        "base_models": [],
        "feature_sets": [],
        "regimes": ["stress"],
        "windows": ["test"],
        "paper_fidelities": ["proxy_feature_adapter"],
        "attribution_features": ["soxx_return_5d"],
    }
    assert [card["model"] for card in view["model_cards"]] == ["gcn_mse__top_ic_factors"]
    card = view["model_cards"][0]
    assert card["paper_fidelity"] == "proxy_feature_adapter"
    assert card["proxy_model"] is True
    assert {row["regime"] for row in card["window_regime_metrics"]} == {"stress"}
    assert {row["window"] for row in card["window_regime_metrics"]} == {"test"}
    assert {row["feature"] for row in card["top_attribution_features"]} == {"soxx_return_5d"}
    assert all(row["model"] == "gcn_mse__top_ic_factors" for row in card["diagnostics"])


def test_load_dashboard_view_model_reads_dashboard_export_json(tmp_path):
    export_path = tmp_path / "dashboard_export.json"
    export_path.write_text(json.dumps(_dashboard_payload()), encoding="utf-8")

    view = load_dashboard_view_model(export_path, filters=DashboardViewFilters(regimes=("bull",)))

    assert view["source_path"] == str(export_path)
    assert view["filters_applied"]["regimes"] == ["bull"]
    assert all({row["regime"] for row in card["regime_metrics"]} <= {"bull"} for card in view["model_cards"])
