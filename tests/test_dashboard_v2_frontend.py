from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_dashboard_v2_entrypoint_redirects_to_unified_vite_bundle():
    html = (ROOT / "dashboard_v2.html").read_text(encoding="utf-8")

    assert "/web/dashboard-v2/dist/index.html" in html
    assert "scripts/dashboard_server.py" in html
    assert "plotly" not in html.lower()


def test_dashboard_v2_dist_uses_relative_asset_paths():
    html = (ROOT / "web" / "dashboard-v2" / "dist" / "index.html").read_text(encoding="utf-8")

    assert 'src="/assets/' not in html
    assert 'href="/assets/' not in html
    assert 'src="./assets/' in html
    assert 'href="./assets/' in html


def test_dashboard_v2_main_mounts_full_width_buy_sell_feature_comparison():
    source = (ROOT / "web" / "dashboard-v2" / "src" / "main.ts").read_text(encoding="utf-8")

    assert 'id="feature-comparison-panel"' in source
    assert "renderFeatureComparisonPanel(" in source


def test_dashboard_v2_chart_commands_use_existing_chart_handles():
    source = (ROOT / "web" / "dashboard-v2" / "src" / "main.ts").read_text(encoding="utf-8")

    assert "computeChartCommandDateRange" not in source
    assert "applyChartRangeCommand(chartHandles, command)" in source


def test_dashboard_v2_model_picker_wraps_without_horizontal_scroll():
    styles = (ROOT / "web" / "dashboard-v2" / "src" / "styles.css").read_text(encoding="utf-8")
    profile_group = styles.split(".profile-group {", 1)[1].split("}", 1)[0]

    assert "flex-wrap: wrap;" in profile_group
    assert "overflow-x: auto;" not in profile_group
