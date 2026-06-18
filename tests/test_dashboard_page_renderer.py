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


def test_render_dashboard_html_outputs_interactive_controls_and_model_sections_without_sidecar_paths():
    html = render_dashboard_html(_view_model(), title="SOXL 模型 Dashboard")
    parser = _ElementCollector()
    parser.feed(html)

    assert "<!doctype html>" in html.lower()
    assert "SOXL 模型 Dashboard" in html
    assert {"model-filter", "regime-filter", "fidelity-filter", "dashboard-view-model"}.issubset(parser.ids)
    assert "gcn_mse__top_ic_factors" in html
    assert "proxy_feature_adapter" in html
    assert "soxx_return_5d" in html
    assert "window-regime" in html
    assert "/path/that/must/not/be/rendered.csv" not in html
    assert any(attrs.get("data-model") == "gcn_mse__top_ic_factors" for attrs in parser.data_attrs)
    assert 'data-regime="stress"' in html


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
