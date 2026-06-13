from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_dashboard_v2_frontend_loads_v2_contract_and_feature_comparison():
    html = (ROOT / "dashboard_v2.html").read_text(encoding="utf-8")

    assert "dashboard_data/v2/manifest.json" in html
    assert "dashboard_data/v2/comparison/default.json" in html
    assert "featureActionTable" in html
    assert "feature_action_summary" in html
    assert "loadFeatureSummaryModels" in html
    assert "renderFeatureActionTable(featureModels)" in html
    assert "slice(0, 8)" in html
    assert "candlestick" in html
    assert "autoUpdateTasks" in html
