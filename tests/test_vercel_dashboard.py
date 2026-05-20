from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_vercel_serves_dashboard_as_static_root():
    config = json.loads((ROOT / "vercel.json").read_text(encoding="utf-8"))

    assert {"source": "/", "destination": "/dashboard.html"} in config["rewrites"]
    header_sources = {entry["source"]: entry for entry in config["headers"]}
    assert "/dashboard.html" in header_sources
    assert "/dashboard_data/:path*" in header_sources

    data_header = header_sources["/dashboard_data/:path*"]["headers"][0]
    assert data_header == {
        "key": "Cache-Control",
        "value": "public, max-age=0, must-revalidate",
    }


def test_dashboard_static_export_is_trackable():
    ignore_lines = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    active_ignore_lines = {
        line.strip()
        for line in ignore_lines
        if line.strip() and not line.lstrip().startswith("#")
    }

    assert "/dashboard.html" not in active_ignore_lines
    assert "/dashboard_data/" not in active_ignore_lines
    assert (ROOT / "dashboard.html").exists()
    assert (ROOT / "dashboard_data" / "index.json").exists()
    assert (ROOT / "dashboard_data" / "model_versions.json").exists()
