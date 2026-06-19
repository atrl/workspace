from __future__ import annotations

import json
from pathlib import Path

from src.dashboard import DashboardViewFilters, build_dashboard_view_model, load_dashboard_view_model


def _dashboard_payload() -> dict:
    payload = json.loads(Path("research/schedule_dashboard_attribution_refresh_v1/dashboard_export.json").read_text())
    payload["artifacts"] = {"model_comparison": "/path/that/must/not/be/read.csv"}
    return payload


def _dashboard_payload_with_failure_modes() -> dict:
    payload = _dashboard_payload()
    payload["failure_modes"] = [
        {
            "schema_version": "analysis.failure_modes.v1",
            "model": "gcn_mse__top_ic_factors",
            "baseline_model": "ridge_mse__all_features",
            "segment_type": "window_regime",
            "segment": "test::stress",
            "failure_mode": "return_stability_lag",
            "severity_score": 12.5,
            "review_note": "gcn_mse 在 stress window 弱于基准。",
        },
        {
            "schema_version": "analysis.failure_modes.v1",
            "model": "ridge_mse__top_ic_factors",
            "baseline_model": "ridge_mse__all_features",
            "segment_type": "window_regime",
            "segment": "test::downtrend",
            "failure_mode": "drawdown_expansion",
            "severity_score": 4.0,
            "review_note": "ridge_mse 在 downtrend window 回撤扩大。",
        },
        {
            "schema_version": "analysis.failure_modes.v1",
            "model": "ridge_mse__all_features",
            "baseline_model": "ridge_mse__all_features",
            "segment_type": "all",
            "segment": "all",
            "failure_mode": "baseline_note",
            "severity_score": 0.0,
            "review_note": "基准说明不应出现在 gcn 卡片。",
        },
    ]
    return payload


def _dashboard_payload_with_model_review_briefing() -> dict:
    payload = _dashboard_payload()
    payload["model_review_briefing"] = [
        {
            "schema_version": "analysis.model_review_briefing.v1",
            "priority_rank": 1,
            "model": "gcn_mse__top_ic_factors",
            "baseline_model": "ridge_mse__all_features",
            "segment_type": "window_regime",
            "segment": "test::stress",
            "window": "test",
            "regime": "stress",
            "review_signal": "weakness",
            "decision_hint": "block_or_rework",
            "priority_score": 12.5,
            "review_note": "gcn_mse 在 stress 弱于基准，需要重做信号适配。",
        },
        {
            "schema_version": "analysis.model_review_briefing.v1",
            "priority_rank": 2,
            "model": "ridge_mse__top_ic_factors",
            "baseline_model": "ridge_mse__all_features",
            "segment_type": "regime",
            "segment": "bull",
            "window": "",
            "regime": "bull",
            "review_signal": "strength",
            "decision_hint": "promote_candidate",
            "priority_score": 4.0,
            "review_note": "ridge_mse 在 bull 强于基准。",
        },
    ]
    return payload


def _dashboard_payload_with_model_run_ledger() -> dict:
    payload = _dashboard_payload()
    first_model = str(payload["top_models"][0]["model"])
    first_base = first_model.split("__", 1)[0]
    payload["model_run_coverage"] = {
        "schema_version": "model.run_ledger_summary.v1",
        "expected_model_count": 2,
        "trained_model_count": 2,
        "backtested_model_count": 2,
        "all_models_completed": True,
        "missing_models": [],
    }
    payload["model_run_ledger"] = [
        {
            "schema_version": "model.run_ledger.v1",
            "model": first_model,
            "base_model": first_base,
            "adapter_status": "paper_strict_adapter",
            "paper_fidelity": "native_library",
            "trained": True,
            "predicted": True,
            "backtested": True,
            "objective_evaluated": True,
            "promotion_evaluated": True,
            "analysis_recorded": True,
            "manifested": True,
        },
        {
            "schema_version": "model.run_ledger.v1",
            "model": "missing_model",
            "base_model": "missing_model",
            "adapter_status": "paper_strict_adapter",
            "paper_fidelity": "proxy_feature_adapter",
            "trained": False,
            "predicted": False,
            "backtested": False,
            "objective_evaluated": False,
            "promotion_evaluated": False,
            "analysis_recorded": False,
            "manifested": False,
        },
    ]
    return payload


def test_dashboard_view_model_builds_page_controls_and_model_cards_from_export_payload_only():
    payload = _dashboard_payload()

    view = build_dashboard_view_model(payload)

    assert view["schema_version"] == "dashboard.view_model.v1"
    assert view["source_schema_version"] == "dashboard.model_matrix.v1"
    assert view["controls"]["models"]
    assert {"bull", "downtrend", "neutral", "stress"}.issubset(set(view["controls"]["regimes"]))
    assert "test" in view["controls"]["windows"]
    assert {"trailing_1m", "trailing_3m", "trailing_6m", "trailing_12m", "ytd"}.issubset(
        set(view["controls"]["windows"])
    )
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
    attribution_cards = [card for card in view["model_cards"] if card["top_attribution_features"]]
    assert attribution_cards
    assert {
        row["model"] for row in attribution_cards[0]["top_attribution_features"]
    } == {attribution_cards[0]["base_model"]}
    assert view["comparison_table"]


def test_dashboard_view_model_attaches_model_run_ledger_to_cards_and_summary():
    payload = _dashboard_payload_with_model_run_ledger()

    view = build_dashboard_view_model(payload)

    assert view["model_run_coverage"]["schema_version"] == "model.run_ledger_summary.v1"
    assert view["model_run_coverage"]["all_models_completed"] is True
    assert "paper_strict_adapter" in view["controls"]["adapter_statuses"]
    assert {"complete", "incomplete"}.issubset(set(view["controls"]["run_statuses"]))
    first_card = view["model_cards"][0]
    assert first_card["run_record"]["schema_version"] == "model.run_ledger.v1"
    assert first_card["run_record"]["trained"] is True
    assert first_card["run_record"]["backtested"] is True
    assert first_card["run_record"]["analysis_recorded"] is True
    assert first_card["run_status"] == "complete"


def test_dashboard_view_model_preserves_production_health_from_export_payload_only():
    payload = _dashboard_payload()
    payload["artifacts"]["production_health"] = "/path/that/must/not/be_read.json"
    payload["production_health"] = {
        "schema_version": "schedule.production_health.v1",
        "strategy_id": "paper_sim_replay_v1",
        "health_status": "healthy",
        "is_healthy": True,
        "mode": "paper",
        "live_broker_enabled": False,
        "manifest_path": "production/paper_sim_replay_v1/production_manifest.json",
        "missing_required_paths": [],
        "checked_path_count": 7,
    }

    view = build_dashboard_view_model(payload)

    assert view["production_health"]["schema_version"] == "schedule.production_health.v1"
    assert view["production_health"]["strategy_id"] == "paper_sim_replay_v1"
    assert view["production_health"]["health_status"] == "healthy"
    assert view["production_health"]["is_healthy"] is True
    assert "artifacts" not in view


def test_dashboard_view_model_attaches_failure_modes_to_matching_model_cards():
    view = build_dashboard_view_model(
        _dashboard_payload_with_failure_modes(),
        filters=DashboardViewFilters(models=("gcn_mse__top_ic_factors",), regimes=("stress",), windows=("test",)),
    )

    assert [card["model"] for card in view["model_cards"]] == ["gcn_mse__top_ic_factors"]
    card = view["model_cards"][0]
    assert card["failure_modes"][0]["failure_mode"] == "return_stability_lag"
    assert card["failure_modes"][0]["segment"] == "test::stress"
    assert card["failure_modes"][0]["severity_score"] == 12.5
    assert "基准说明" not in json.dumps(card["failure_modes"], ensure_ascii=False)
    assert view["failure_mode_summary"]["count"] == 3
    assert view["failure_mode_summary"]["top_failure_mode"] == "return_stability_lag"


def test_dashboard_view_model_exposes_and_filters_failure_modes():
    view = build_dashboard_view_model(
        _dashboard_payload_with_failure_modes(),
        filters=DashboardViewFilters(failure_modes=("return_stability_lag",)),
    )

    assert view["controls"]["failure_modes"] == [
        "baseline_note",
        "drawdown_expansion",
        "return_stability_lag",
    ]
    assert view["filters_applied"]["failure_modes"] == ["return_stability_lag"]
    assert [card["model"] for card in view["model_cards"]] == ["gcn_mse__top_ic_factors"]
    assert {row["failure_mode"] for row in view["model_cards"][0]["failure_modes"]} == {"return_stability_lag"}


def test_dashboard_view_model_exposes_and_filters_model_review_briefing():
    view = build_dashboard_view_model(
        _dashboard_payload_with_model_review_briefing(),
        filters=DashboardViewFilters(review_signals=("weakness",), decision_hints=("block_or_rework",)),
    )

    assert view["controls"]["review_signals"] == ["strength", "weakness"]
    assert view["controls"]["decision_hints"] == ["block_or_rework", "promote_candidate"]
    assert view["filters_applied"]["review_signals"] == ["weakness"]
    assert view["filters_applied"]["decision_hints"] == ["block_or_rework"]
    assert [card["model"] for card in view["model_cards"]] == ["gcn_mse__top_ic_factors"]
    assert view["model_cards"][0]["model_review_briefing"][0]["review_signal"] == "weakness"
    assert view["model_cards"][0]["model_review_briefing"][0]["decision_hint"] == "block_or_rework"
    assert view["review_briefing_summary"]["count"] == 2
    assert view["review_briefing_summary"]["top_review_signal"] == "weakness"
    assert view["review_briefing_summary"]["top_decision_hint"] == "block_or_rework"


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
        "failure_modes": [],
        "review_signals": [],
        "decision_hints": [],
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
