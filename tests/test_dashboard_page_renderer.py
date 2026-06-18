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


def test_write_dashboard_html_writes_page_and_keeps_embedded_view_model_json(tmp_path):
    output_path = tmp_path / "dashboard.html"

    page = write_dashboard_html(_view_model(), output_path=output_path)

    saved = output_path.read_text(encoding="utf-8")
    assert saved == page
    assert '<script type="application/json" id="dashboard-view-model">' in saved
    assert '"schema_version": "dashboard.view_model.v1"' in saved
