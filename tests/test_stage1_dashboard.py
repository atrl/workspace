from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient


ROOT = Path(__file__).resolve().parents[1]


def test_stage1_dashboard_payload_supports_custom_interval_and_benchmarks():
    from src.dashboard.stage1 import build_stage1_dashboard_payload

    payload = build_stage1_dashboard_payload(
        root=ROOT,
        start="2026-01-02",
        end="2026-03-31",
    )

    assert payload["source"] == "qlib_recorder"
    assert payload["selected_range"]["start"] == "2026-01-02"
    assert payload["selected_range"]["end"] == "2026-03-31"
    assert set(payload["metrics"]) == {"Stage1", "SOXL", "SOXX"}
    assert payload["rows"]
    assert {"soxl_weight", "soxx_weight", "cash_weight"} <= payload["rows"][0].keys()
    assert "非因果归因" in payload["factor_disclaimer"]
    assert all(event["factors"] for event in payload["trade_events"])


def test_stage1_dashboard_app_serves_ui_health_and_interval_api():
    from src.dashboard.stage1_app import create_stage1_app

    client = TestClient(create_stage1_app(ROOT))
    page = client.get("/")
    health = client.get("/health")
    interval = client.get("/api/stage1?start=2026-07-01&end=2026-07-22")

    assert page.status_code == 200
    assert "Stage1 · Qlib 决策台" in page.text
    assert "运行区间推理" in page.text
    assert health.json()["source"] == "qlib_recorder"
    assert interval.status_code == 200
    assert interval.json()["selected_range"]["start"] == "2026-07-01"


def test_stage1_dashboard_rejects_inverted_interval():
    from src.dashboard.stage1 import build_stage1_dashboard_payload

    try:
        build_stage1_dashboard_payload(root=ROOT, start="2026-07-20", end="2026-07-01")
    except ValueError as exc:
        assert "start date" in str(exc)
    else:
        raise AssertionError("inverted interval should fail closed")
