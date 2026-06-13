from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_dashboard_v2_entrypoint_redirects_to_unified_vite_bundle():
    html = (ROOT / "dashboard_v2.html").read_text(encoding="utf-8")

    assert "/web/dashboard-v2/dist/index.html" in html
    assert "scripts/dashboard_server.py" in html
    assert "plotly" not in html.lower()
