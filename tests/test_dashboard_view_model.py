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


def _dashboard_payload_with_model_difference_explanations() -> dict:
    payload = _dashboard_payload()
    payload["model_difference_explanations"] = [
        {
            "schema_version": "analysis.model_difference_explanation.v1",
            "priority_rank": 1,
            "segment_type": "window_regime",
            "segment": "test::stress",
            "window": "test",
            "regime": "stress",
            "baseline_model": "ridge_mse__all_features",
            "winner_model": "ridge_mse__all_features",
            "loser_model": "gcn_mse__top_ic_factors",
            "relative_return_pct_spread": 6.5,
            "winner_vs_baseline_delta": 0.0,
            "explanation_signal": "baseline_defends",
            "decision_hint": "block_or_rework_loser",
            "review_note": "中文 review：基准 ridge_mse 在 stress 领先，gcn_mse 需要重做信号适配。",
        },
        {
            "schema_version": "analysis.model_difference_explanation.v1",
            "priority_rank": 2,
            "segment_type": "regime",
            "segment": "bull",
            "window": "",
            "regime": "bull",
            "baseline_model": "ridge_mse__all_features",
            "winner_model": "gcn_mse__top_ic_factors",
            "loser_model": "ridge_mse__all_features",
            "relative_return_pct_spread": 2.2,
            "winner_vs_baseline_delta": 2.2,
            "explanation_signal": "segment_winner",
            "decision_hint": "promote_segment_candidate",
            "review_note": "中文 review：gcn_mse 在 bull 强于基准，可作为分段候选。",
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


def _dashboard_payload_with_requested_model_contract() -> dict:
    payload = _dashboard_payload()
    payload["requested_model_contract"] = {
        "schema_version": "model.requested_contract_health.v1",
        "requested_model_count": 32,
        "registered_model_count": 32,
        "ledger_model_count": 31,
        "completed_requested_model_count": 31,
        "missing_from_registry": [],
        "missing_from_ledger": ["sthgcn_mse"],
        "incomplete_requested_models": ["sthgcn_mse"],
        "all_requested_models_completed": False,
    }
    return payload


def _dashboard_payload_with_readiness_sources() -> dict:
    payload = _dashboard_payload_with_requested_model_contract()
    payload["artifacts"]["production_health"] = "/path/that/must/not/be_read.json"
    payload["artifacts"]["notebook_card_manifest"] = "/path/that/must/not/be_read.json"
    payload["production_health"] = {
        "schema_version": "schedule.production_health.v1",
        "strategy_id": "paper_sim_replay_v1",
        "health_status": "healthy",
        "is_healthy": True,
        "mode": "paper",
        "missing_required_paths": [],
        "checked_path_count": 7,
    }
    payload["notebook_card_readiness"] = {
        "schema_version": "dashboard.notebook_card_readiness.v1",
        "manifest_hash": "n" * 64,
        "summary": {
            "card_count": 91,
            "production_card_count": 2,
            "production_published_count": 1,
            "production_template_count": 1,
            "production_manifest_missing_count": 0,
            "production_registry_mismatch_count": 0,
        },
        "production_cards": [
            {"card_id": "paper_reference_strategy", "production_status": "template_only"},
            {"card_id": "paper_sim_replay_v1", "production_status": "published"},
        ],
    }
    payload["model_run_coverage"] = {
        "schema_version": "model.run_ledger_summary.v1",
        "expected_model_count": 2,
        "trained_model_count": 2,
        "backtested_model_count": 2,
        "all_models_completed": True,
        "missing_models": [],
    }
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
    assert {"native_library", "native_graph"}.issubset(set(view["controls"]["paper_fidelities"]))
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


def test_dashboard_view_model_exposes_requested_model_contract_summary():
    payload = _dashboard_payload_with_requested_model_contract()

    view = build_dashboard_view_model(payload)

    assert view["requested_model_contract"]["schema_version"] == "model.requested_contract_health.v1"
    assert view["requested_model_contract"]["requested_model_count"] == 32
    assert view["requested_model_contract"]["completed_requested_model_count"] == 31
    assert view["requested_model_contract"]["missing_from_ledger"] == ["sthgcn_mse"]


def test_dashboard_view_model_builds_readiness_overview_from_export_payload_only():
    payload = _dashboard_payload_with_readiness_sources()

    view = build_dashboard_view_model(payload)

    overview = view["readiness_overview"]
    assert overview["schema_version"] == "dashboard.readiness_overview.v1"
    assert overview["status"] == "review"
    assert overview["gate_count"] == 4
    assert overview["pass_count"] == 2
    assert overview["review_count"] == 2
    assert overview["block_count"] == 0
    gates = {gate["gate"]: gate for gate in overview["gates"]}
    assert gates["production_health"]["status"] == "pass"
    assert gates["production_health"]["detail"] == "healthy / missing 0"
    assert gates["notebook_cards"]["status"] == "review"
    assert gates["notebook_cards"]["detail"] == "published 1 / template 1 / missing 0 / mismatch 0"
    assert gates["requested_models"]["status"] == "review"
    assert gates["requested_models"]["detail"] == "31 / 32 complete / missing 1 / incomplete 1"
    assert gates["model_runs"]["status"] == "pass"
    assert gates["model_runs"]["detail"] == "trained 2 / backtested 2 / expected 2"
    assert "artifacts" not in view


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


def test_dashboard_view_model_preserves_notebook_card_readiness_from_export_payload_only():
    payload = _dashboard_payload()
    payload["artifacts"]["notebook_card_manifest"] = "/path/that/must/not/be_read.json"
    payload["notebook_card_readiness"] = {
        "schema_version": "dashboard.notebook_card_readiness.v1",
        "manifest_hash": "n" * 64,
        "summary": {
            "production_card_count": 2,
            "production_published_count": 1,
            "production_template_count": 1,
        },
        "production_cards": [
            {"card_id": "paper_reference_strategy", "production_status": "template_only"},
            {"card_id": "paper_sim_replay_v1", "production_status": "published"},
        ],
    }

    view = build_dashboard_view_model(payload)

    assert view["notebook_card_readiness"]["schema_version"] == "dashboard.notebook_card_readiness.v1"
    assert view["notebook_card_readiness"]["manifest_hash"] == "n" * 64
    assert view["notebook_card_readiness"]["summary"]["production_published_count"] == 1
    assert [card["production_status"] for card in view["notebook_card_readiness"]["production_cards"]] == [
        "template_only",
        "published",
    ]
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


def test_dashboard_view_model_exposes_and_filters_model_difference_explanations():
    view = build_dashboard_view_model(
        _dashboard_payload_with_model_difference_explanations(),
        filters=DashboardViewFilters(
            explanation_signals=("baseline_defends",),
            difference_decisions=("block_or_rework_loser",),
        ),
    )

    assert view["controls"]["explanation_signals"] == ["baseline_defends", "segment_winner"]
    assert view["controls"]["difference_decisions"] == ["block_or_rework_loser", "promote_segment_candidate"]
    assert view["filters_applied"]["explanation_signals"] == ["baseline_defends"]
    assert view["filters_applied"]["difference_decisions"] == ["block_or_rework_loser"]
    assert [card["model"] for card in view["model_cards"]] == ["gcn_mse__top_ic_factors"]
    gcn = [card for card in view["model_cards"] if card["model"] == "gcn_mse__top_ic_factors"][0]
    assert gcn["model_difference_explanations"][0]["loser_model"] == "gcn_mse__top_ic_factors"
    assert gcn["model_difference_explanations"][0]["explanation_signal"] == "baseline_defends"
    assert view["model_difference_summary"]["count"] == 2
    assert view["model_difference_summary"]["top_explanation_signal"] == "baseline_defends"
    assert view["model_difference_summary"]["top_decision_hint"] == "block_or_rework_loser"


def test_dashboard_view_model_filters_by_model_regime_window_fidelity_and_attribution_feature():
    payload = _dashboard_payload()

    view = build_dashboard_view_model(
        payload,
        filters=DashboardViewFilters(
            models=("gcn_mse__top_ic_factors",),
            regimes=("stress",),
            windows=("test",),
            paper_fidelities=("native_graph",),
            attribution_features=("soxx_return_5d",),
        ),
    )

    assert view["filters_applied"] == {
        "models": ["gcn_mse__top_ic_factors"],
        "base_models": [],
        "feature_sets": [],
        "regimes": ["stress"],
        "windows": ["test"],
        "paper_fidelities": ["native_graph"],
        "attribution_features": ["soxx_return_5d"],
        "failure_modes": [],
        "review_signals": [],
        "decision_hints": [],
        "explanation_signals": [],
        "difference_decisions": [],
    }
    assert [card["model"] for card in view["model_cards"]] == ["gcn_mse__top_ic_factors"]
    card = view["model_cards"][0]
    assert card["paper_fidelity"] == "native_graph"
    assert card["proxy_model"] is False
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
