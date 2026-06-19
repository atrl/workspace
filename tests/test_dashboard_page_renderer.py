from __future__ import annotations

import json
from html.parser import HTMLParser
from pathlib import Path

from src.dashboard import build_dashboard_view_model, render_dashboard_html, write_dashboard_html


class _ElementCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: set[str] = set()
        self.data_attrs: list[dict[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = {name: value or "" for name, value in attrs}
        if "id" in attributes:
            self.ids.add(attributes["id"])
        if any(name.startswith("data-") for name in attributes):
            self.data_attrs.append(attributes)


def _view_model() -> dict:
    payload = json.loads(Path("research/schedule_dashboard_attribution_refresh_v1/dashboard_export.json").read_text())
    payload["artifacts"] = {"model_comparison": "/path/that/must/not/be/rendered.csv"}
    return build_dashboard_view_model(payload)


def _view_model_with_failure_modes() -> dict:
    payload = json.loads(Path("research/schedule_dashboard_attribution_refresh_v1/dashboard_export.json").read_text())
    payload["artifacts"] = {"failure_modes": "/path/that/must/not/be/rendered.csv"}
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
        }
    ]
    return build_dashboard_view_model(payload)


def _view_model_with_model_review_briefing() -> dict:
    payload = json.loads(Path("research/schedule_dashboard_attribution_refresh_v1/dashboard_export.json").read_text())
    payload["artifacts"] = {"model_review_briefing": "/path/that/must/not/be/rendered.csv"}
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
        }
    ]
    return build_dashboard_view_model(payload)


def _view_model_with_model_difference_explanations() -> dict:
    payload = json.loads(Path("research/schedule_dashboard_attribution_refresh_v1/dashboard_export.json").read_text())
    payload["artifacts"] = {"model_difference_explanations": "/path/that/must/not/be/rendered.csv"}
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
        }
    ]
    return build_dashboard_view_model(payload)


def _view_model_with_review_briefing() -> dict:
    payload = json.loads(Path("research/schedule_dashboard_attribution_refresh_v1/dashboard_export.json").read_text())
    payload["artifacts"] = {"model_review_briefing": "/path/that/must/not/be/rendered.csv"}
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
            "priority_score": 18.5,
            "review_note": "gcn_mse 在 stress window 弱于基准。",
        }
    ]
    return build_dashboard_view_model(payload)


def _view_model_with_analysis_overview() -> dict:
    payload = json.loads(Path("research/schedule_dashboard_attribution_refresh_v1/dashboard_export.json").read_text())
    payload["artifacts"] = {"analysis_overview": "/path/that/must/not/be/rendered.json"}
    payload["analysis_overview"] = {
        "schema_version": "analysis.overview.v1",
        "baseline_model": "ridge_mse__all_features",
        "model_count": 2,
        "regime_count": 4,
        "window_count": 1,
        "largest_relative_return_spread_segments": [
            {
                "segment": "test::stress",
                "best_relative_return_model": "ridge_mse__all_features",
                "worst_relative_return_model": "gcn_mse__top_ic_factors",
                "relative_return_pct_spread": 4.25,
            }
        ],
        "top_weakness": {
            "model": "gcn_mse__top_ic_factors",
            "segment": "test::stress",
            "review_signal": "weakness",
            "decision_hint": "block_or_rework",
        },
    }
    return build_dashboard_view_model(payload)


def _view_model_with_production_health() -> dict:
    payload = json.loads(Path("research/schedule_dashboard_attribution_refresh_v1/dashboard_export.json").read_text())
    payload["artifacts"] = {"production_health": "/path/that/must/not/be/rendered.json"}
    payload["production_health"] = {
        "schema_version": "schedule.production_health.v1",
        "strategy_id": "paper_sim_replay_v1",
        "health_status": "healthy",
        "is_healthy": True,
        "mode": "paper",
        "live_broker_enabled": False,
        "paper_execution_pipeline_schema_version": "schedule.paper_execution.v1",
        "paper_execution_job_status": "success",
        "paper_execution_objective_loss_schema_version": "objective.loss.v1",
        "paper_execution_objective_loss_status": "review",
        "paper_execution_objective_loss_reward_surface": "retired",
        "paper_execution_objective_loss_top_component": "ranking_loss",
        "paper_execution_objective_loss_top_value": 1.0,
        "manifest_path": "production/paper_sim_replay_v1/production_manifest.json",
        "missing_required_paths": [],
        "checked_path_count": 7,
    }
    return build_dashboard_view_model(payload)


def _view_model_with_requested_model_contract() -> dict:
    payload = json.loads(Path("research/schedule_dashboard_attribution_refresh_v1/dashboard_export.json").read_text())
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
    return build_dashboard_view_model(payload)


def _view_model_with_notebook_card_readiness() -> dict:
    payload = json.loads(Path("research/schedule_dashboard_attribution_refresh_v1/dashboard_export.json").read_text())
    payload["artifacts"] = {"notebook_card_manifest": "/path/that/must/not/be/rendered.json"}
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
    return build_dashboard_view_model(payload)


def _view_model_with_readiness_overview() -> dict:
    payload = json.loads(Path("research/schedule_dashboard_attribution_refresh_v1/dashboard_export.json").read_text())
    payload["artifacts"] = {
        "production_health": "/path/that/must/not/be/rendered.json",
        "notebook_card_manifest": "/path/that/must/not/be/rendered.json",
    }
    payload["production_health"] = {
        "schema_version": "schedule.production_health.v1",
        "strategy_id": "paper_sim_replay_v1",
        "health_status": "healthy",
        "is_healthy": True,
        "mode": "paper",
        "missing_required_paths": [],
        "checked_path_count": 7,
    }
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
    return build_dashboard_view_model(payload)


def _view_model_with_schedule_health() -> dict:
    payload = json.loads(Path("research/schedule_dashboard_attribution_refresh_v1/dashboard_export.json").read_text())
    payload["artifacts"] = {"schedule_health": "/path/that/must/not/be/rendered.json"}
    payload["schedule_health"] = {
        "schema_version": "dashboard.schedule_health.v1",
        "job_count": 2,
        "failed_job_count": 1,
        "total_attempts": 3,
        "total_retry_error_count": 2,
        "top_failure_job": "model_dashboard_refresh",
        "top_failure_last_error": "RuntimeError:permanent failure",
        "jobs": [
            {
                "name": "model_dashboard_refresh",
                "status": "failed",
                "attempts": 2,
                "retry_schema_version": "schedule.retry_attempts.v1",
                "retry_error_count": 2,
                "last_error_attempt": 2,
                "last_error_type": "RuntimeError",
                "last_error_message": "permanent failure",
                "detail": "permanent failure",
            },
            {
                "name": "data_update",
                "status": "success",
                "attempts": 1,
                "retry_schema_version": "schedule.retry_attempts.v1",
                "retry_error_count": 0,
                "last_error_attempt": 0,
                "last_error_type": "",
                "last_error_message": "",
                "detail": "",
            },
        ],
    }
    return build_dashboard_view_model(payload)


def _view_model_with_objective_losses() -> dict:
    payload = json.loads(Path("research/schedule_dashboard_attribution_refresh_v1/dashboard_export.json").read_text())
    payload["artifacts"] = {"objective_losses": "/path/that/must/not/be/rendered.json"}
    payload["objective"] = {
        **payload.get("objective", {}),
        "losses": {
            "schema_version": "objective.loss.v1",
            "metric_source_schema_version": "objective.evaluation.v1",
            "optimization_direction": "minimize",
            "reward_surface": "retired",
            "row_count": 78,
            "mse_loss": 0.0274,
            "classification_loss": 0.4359,
            "ic_loss": 0.9774,
            "ranking_loss": 1.0,
            "rank_group_count": 0,
        },
    }
    return build_dashboard_view_model(payload)


def test_render_dashboard_html_outputs_interactive_controls_and_model_sections_without_sidecar_paths():
    html = render_dashboard_html(_view_model(), title="SOXL 模型 Dashboard")
    parser = _ElementCollector()
    parser.feed(html)

    assert "<!doctype html>" in html.lower()
    assert "SOXL 模型 Dashboard" in html
    assert {"model-filter", "regime-filter", "fidelity-filter", "dashboard-view-model"}.issubset(parser.ids)
    assert "gcn_mse__top_ic_factors" in html
    assert "native_graph" in html
    assert "soxx_return_5d" in html
    assert "window-regime" in html
    assert "/path/that/must/not/be/rendered.csv" not in html
    assert any(attrs.get("data-model") == "gcn_mse__top_ic_factors" for attrs in parser.data_attrs)
    assert 'data-regime="stress"' in html


def test_render_dashboard_html_outputs_analysis_overview_without_sidecar_paths():
    html = render_dashboard_html(_view_model_with_analysis_overview(), title="SOXL Analysis Overview")

    assert "Analysis Overview" in html
    assert "test::stress" in html
    assert "gcn_mse__top_ic_factors -> ridge_mse__all_features" in html
    assert "4.25" in html
    assert "weakness / block_or_rework" in html
    assert "/path/that/must/not/be/rendered.json" not in html


def test_render_dashboard_html_outputs_production_health_panel_without_sidecar_paths():
    html = render_dashboard_html(_view_model_with_production_health(), title="SOXL Production Health")

    assert "Production Health" in html
    assert "paper_sim_replay_v1" in html
    assert "healthy" in html
    assert "paper" in html
    assert "missing 0" in html
    assert "live broker off" in html
    assert "Paper Execution" in html
    assert "schedule.paper_execution.v1" in html
    assert "success" in html
    assert "Objective Loss" in html
    assert "objective.loss.v1 / review" in html
    assert "ranking_loss: 1" in html
    assert "retired" in html
    assert "/path/that/must/not/be/rendered.json" not in html


def test_render_dashboard_html_outputs_requested_model_contract_summary():
    html = render_dashboard_html(_view_model_with_requested_model_contract(), title="SOXL Requested Contract")

    assert "Requested 模型" in html
    assert "31 / 32" in html
    assert "missing 1" in html
    assert "SOXL Requested Contract" in html


def test_render_dashboard_html_outputs_notebook_card_readiness_panel_without_sidecar_paths():
    html = render_dashboard_html(_view_model_with_notebook_card_readiness(), title="SOXL Notebook Readiness")

    assert "Notebook Readiness" in html
    assert "dashboard.notebook_card_readiness.v1" in html
    assert "91" in html
    assert "published 1" in html
    assert "template 1" in html
    assert "missing 0" in html
    assert "paper_reference_strategy" in html
    assert "paper_sim_replay_v1" in html
    assert "/path/that/must/not/be/rendered.json" not in html


def test_render_dashboard_html_outputs_readiness_overview_without_sidecar_paths():
    html = render_dashboard_html(_view_model_with_readiness_overview(), title="SOXL Readiness Overview")

    assert "Readiness Overview" in html
    assert "dashboard.readiness_overview.v1" in html
    assert "Overall" in html
    assert "review" in html
    assert "Dashboard Manifest" in html
    assert "analysis_manifest / complete" in html
    assert "Production Health" in html
    assert "Notebook Cards" in html
    assert "Requested Models" in html
    assert "Model Runs" in html
    assert "healthy / missing 0" in html
    assert "published 1 / template 1" in html
    assert "31 / 32 complete" in html
    assert "trained 2 / backtested 2 / expected 2" in html
    assert "/path/that/must/not/be/rendered.json" not in html


def test_render_dashboard_html_outputs_schedule_health_retry_panel_without_sidecar_paths():
    html = render_dashboard_html(_view_model_with_schedule_health(), title="SOXL Schedule Health")

    assert "Schedule Health" in html
    assert "dashboard.schedule_health.v1" in html
    assert "model_dashboard_refresh" in html
    assert "failed" in html
    assert "attempts 2" in html
    assert "retry_errors 2" in html
    assert "RuntimeError:permanent failure" in html
    assert "/path/that/must/not/be/rendered.json" not in html


def test_render_dashboard_html_outputs_objective_loss_panel_without_sidecar_paths():
    html = render_dashboard_html(_view_model_with_objective_losses(), title="SOXL Objective Loss")

    assert "Objective Loss" in html
    assert "objective.loss.v1" in html
    assert "retired" in html
    assert "minimize" in html
    assert "mse_loss" in html
    assert "classification_loss" in html
    assert "ic_loss" in html
    assert "ranking_loss" in html
    assert "0.0274" in html
    assert "1.0000" in html
    assert "/path/that/must/not/be/rendered.json" not in html


def test_render_dashboard_html_outputs_blocked_dashboard_manifest_readiness_gate():
    payload = json.loads(Path("research/schedule_dashboard_attribution_refresh_v1/dashboard_export.json").read_text())
    payload["manifest_source"] = {
        "kind": "fixed_sidecars",
        "path": "",
        "manifest_hash": "",
        "producer": "src.dashboard.model_matrix",
    }
    view = build_dashboard_view_model(payload)

    html = render_dashboard_html(view, title="SOXL Manifest Gate")

    assert "Readiness Overview" in html
    assert "blocked" in html
    assert "Dashboard Manifest" in html
    assert "fixed_sidecars / weak_kind" in html


def test_render_dashboard_html_outputs_failure_mode_sections_without_sidecar_paths():
    html = render_dashboard_html(_view_model_with_failure_modes(), title="SOXL Failure Modes")
    parser = _ElementCollector()
    parser.feed(html)

    assert "Failure Modes" in html
    assert "return_stability_lag" in html
    assert "12.50" in html
    assert "gcn_mse 在 stress window 弱于基准。" in html
    assert "failure-mode-filter" in parser.ids
    assert '<option value="return_stability_lag">return_stability_lag</option>' in html
    assert 'data-failure-mode="return_stability_lag"' in html
    assert 'data-failure-modes="return_stability_lag"' in html
    assert "/path/that/must/not/be/rendered.csv" not in html


def test_render_dashboard_html_outputs_model_review_briefing_controls_and_cards_without_sidecar_paths():
    html = render_dashboard_html(_view_model_with_model_review_briefing(), title="SOXL Review Briefing")
    parser = _ElementCollector()
    parser.feed(html)

    assert "Review Briefing" in html
    assert "weakness" in html
    assert "block_or_rework" in html
    assert "gcn_mse 在 stress 弱于基准，需要重做信号适配。" in html
    assert {"review-signal-filter", "decision-hint-filter"}.issubset(parser.ids)
    assert '<option value="weakness">weakness</option>' in html
    assert '<option value="block_or_rework">block_or_rework</option>' in html
    assert 'data-review-signals="weakness"' in html
    assert 'data-decision-hints="block_or_rework"' in html
    assert "/path/that/must/not/be/rendered.csv" not in html


def test_render_dashboard_html_outputs_model_difference_explanations_without_sidecar_paths():
    html = render_dashboard_html(_view_model_with_model_difference_explanations(), title="SOXL Difference Explanations")
    parser = _ElementCollector()
    parser.feed(html)

    assert "Model Difference Explanations" in html
    assert "baseline_defends" in html
    assert "block_or_rework_loser" in html
    assert "中文 review：基准 ridge_mse 在 stress 领先" in html
    assert {"explanation-signal-filter", "difference-decision-filter"}.issubset(parser.ids)
    assert '<option value="baseline_defends">baseline_defends</option>' in html
    assert '<option value="block_or_rework_loser">block_or_rework_loser</option>' in html
    assert 'data-explanation-signal="baseline_defends"' in html
    assert 'data-explanation-signals="baseline_defends"' in html
    assert 'data-difference-decisions="block_or_rework_loser"' in html
    assert "/path/that/must/not/be/rendered.csv" not in html


def test_render_dashboard_html_outputs_model_review_briefing_sections_without_sidecar_paths():
    html = render_dashboard_html(_view_model_with_review_briefing(), title="SOXL Review Briefing")
    parser = _ElementCollector()
    parser.feed(html)

    assert "Model Review" in html
    assert "weakness" in html
    assert "block_or_rework" in html
    assert "gcn_mse 在 stress window 弱于基准。" in html
    assert "review-signal-filter" in parser.ids
    assert "decision-hint-filter" in parser.ids
    assert 'data-review-signal="weakness"' in html
    assert 'data-review-signals="weakness"' in html
    assert 'data-decision-hints="block_or_rework"' in html
    assert "/path/that/must/not/be/rendered.csv" not in html


def test_write_dashboard_html_writes_page_and_keeps_embedded_view_model_json(tmp_path):
    output_path = tmp_path / "dashboard.html"

    page = write_dashboard_html(_view_model(), output_path=output_path)

    saved = output_path.read_text(encoding="utf-8")
    assert saved == page
    assert '<script type="application/json" id="dashboard-view-model">' in saved
    assert '"schema_version": "dashboard.view_model.v1"' in saved
