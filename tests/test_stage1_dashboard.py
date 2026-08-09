from __future__ import annotations

import json
import plistlib
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
    assert {"stage1_drawdown", "soxl_drawdown", "soxx_drawdown"} <= payload["rows"][0].keys()
    assert payload["window_summary"]["trade_event_count"] == len(payload["trade_events"])
    assert payload["window_summary"]["positive_day_rate_pct"] >= 0.0
    assert payload["last_completed"]["signal_date"] == payload["selected_range"]["end"]
    assert payload["artifact_policy"] == "slim_card_recorder_first"
    assert "非因果归因" in payload["factor_disclaimer"]
    assert all(event["factors"] for event in payload["trade_events"])
    assert payload["paper_simulation"]["source"] == "registered_stage1_runtime"
    assert payload["paper_simulation"]["status"] in {"ready", "repair_armed", "needs_data"}


def test_stage1_dashboard_builds_repair_and_execution_sync_panel(tmp_path):
    from src.dashboard.stage1 import build_stage1_paper_simulation_status

    runtime = tmp_path / "runtime" / "soxl" / "stage1_futu_sim"
    runtime.mkdir(parents=True)
    (runtime / "paper_cycle.json").write_text(
        json.dumps(
            {
                "cycle_id": "stage1_test_v3",
                "status": "active",
                "inception_date": "2026-08-03",
                "ledger_file": "paper_ledger_v3.json",
                "execution_state_file": "execution_state_v3.json",
                "execution_repair": {
                    "status": "armed",
                    "effective_session_date": "2026-08-10",
                    "applied_commit": "70f2e88e",
                    "history_policy": "preserve_observed_fills_without_rewriting",
                    "prior_recovery_slots_contaminated": ["2026-08-06:recovery"],
                },
            }
        ),
        encoding="utf-8",
    )
    (runtime / "paper_ledger_v3.json").write_text(
        json.dumps(
            {
                "cash": 37_897.395,
                "equity": 114_121.125,
                "peak_equity": 114_367.31,
                "drawdown": -0.00215,
                "total_inferred_fees": 14.94,
                "positions": {"SOXL": 160, "SOXX": 99},
                "prices": {"SOXL": 140.25, "SOXX": 543.27},
                "updated_at": "2026-08-08T10:03:25+08:00",
            }
        ),
        encoding="utf-8",
    )
    (runtime / "execution_state_v3.json").write_text(
        json.dumps(
            {
                "completed_slots": {
                    "2026-08-06:open": {
                        "slot": "open",
                        "signal_date": "2026-08-06",
                        "status": "filled_confirmed",
                        "recorded_at": "2026-08-07T21:31:30+08:00",
                        "target_quantities": {"SOXL": 156, "SOXX": 99},
                    },
                    "2026-08-06:recovery": {
                        "slot": "recovery",
                        "signal_date": "2026-08-06",
                        "status": "filled_confirmed",
                        "recorded_at": "2026-08-07T22:32:17+08:00",
                        "target_quantities": {"SOXL": 160, "SOXX": 99},
                    },
                },
                "pending_slots": {},
            }
        ),
        encoding="utf-8",
    )
    (runtime / "latest_run.json").write_text(
        json.dumps(
            {
                "status": "not_due",
                "slot": "not_due",
                "recorded_at": "2026-08-09T17:00:14+08:00",
            }
        ),
        encoding="utf-8",
    )
    summary = {
        "execution": {
            "cycle_manifest": "runtime/soxl/stage1_futu_sim/paper_cycle.json",
            "manual_start": {
                "supported": True,
                "entrypoint": "production/run_stage1_futu_sim.sh --slot start-now",
            },
        },
        "latest_target": {
            "status": "ready",
            "signal_date": "2026-08-07",
            "controller_state_source": "paper_ledger_cycle",
            "weights": {"SOXL": 0.1932668871, "SOXX": 0.4751834576, "cash": 0.3315496553},
            "recovery_action": False,
        },
    }

    status = build_stage1_paper_simulation_status(root=tmp_path, summary=summary)

    assert status["status"] == "repair_armed"
    assert status["ledger"]["positions"] == {"SOXL": 160, "SOXX": 99}
    assert status["target"]["approximate_quantities_at_last_mark"] == {
        "SOXL": 157,
        "SOXX": 99,
    }
    assert status["target"]["quantity_deltas_at_last_mark"] == {
        "SOXL": -3,
        "SOXX": 0,
    }
    assert status["execution_sync"]["completed_slot_count"] == 2
    assert status["execution_sync"]["pending_slot_count"] == 0
    assert status["execution_sync"]["contaminated_recovery_slot_count"] == 1
    assert status["policy"]["manual_start"]["supported"] is True
    assert status["policy"]["real_broker_order"] is False


def test_stage1_dashboard_never_presents_historical_target_as_executable(tmp_path):
    from src.dashboard.stage1 import build_stage1_paper_simulation_status

    runtime = tmp_path / "runtime" / "soxl" / "stage1_futu_sim"
    runtime.mkdir(parents=True)
    (runtime / "paper_cycle.json").write_text(
        json.dumps(
            {
                "cycle_id": "stage1_test_v3",
                "status": "active",
                "ledger_file": "paper_ledger_v3.json",
                "execution_state_file": "execution_state_v3.json",
            }
        ),
        encoding="utf-8",
    )
    (runtime / "paper_ledger_v3.json").write_text(
        json.dumps(
            {
                "cash": 37_897.395,
                "equity": 114_121.125,
                "positions": {"SOXL": 160, "SOXX": 99},
                "prices": {"SOXL": 140.25, "SOXX": 543.27},
            }
        ),
        encoding="utf-8",
    )
    (runtime / "execution_state_v3.json").write_text(
        json.dumps({"completed_slots": {}, "pending_slots": {}}),
        encoding="utf-8",
    )
    summary = {
        "execution": {
            "cycle_manifest": "runtime/soxl/stage1_futu_sim/paper_cycle.json"
        },
        "latest_target": {
            "status": "ready",
            "signal_date": "2026-08-07",
            "controller_state_source": "historical_model_replay",
            "weights": {"SOXL": 0.0, "SOXX": 0.3, "cash": 0.7},
            "recovery_action": False,
        },
    }

    target = build_stage1_paper_simulation_status(
        root=tmp_path, summary=summary
    )["target"]

    assert target["status"] == "pending_live_open_recompute"
    assert target["executable"] is False
    assert target["weights"] is None
    assert target["approximate_quantities_at_last_mark"] == {
        "SOXL": None,
        "SOXX": None,
    }
    assert target["quantity_deltas_at_last_mark"] == {
        "SOXL": None,
        "SOXX": None,
    }


def test_stage1_dashboard_app_serves_ui_health_and_interval_api():
    from src.dashboard.stage1_app import create_stage1_app

    client = TestClient(create_stage1_app(ROOT))
    page = client.get("/")
    health = client.get("/health")
    interval = client.get("/api/stage1?start=2026-07-01&end=2026-07-22")

    assert page.status_code == 200
    assert "Stage1 · Qlib 决策工作台" in page.text
    assert "近 3 月" in page.text
    assert "相对收益" in page.text
    assert "换仓诊断" in page.text
    assert "模拟盘修复与交易同步" in page.text
    assert 'id="sim-repair-status"' in page.text
    assert 'id="sim-start-mode"' in page.text
    assert "拖拽图表选择区间" in page.text
    assert 'id="apply-chart-selection"' in page.text
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


def test_stage1_dashboard_launchagents_keep_origin_local_and_tunnel_separate():
    dashboard_path = ROOT / "production" / "com.soxl.stage1.dashboard.plist"
    tunnel_path = ROOT / "production" / "com.soxl.stage1.cloudflare.plist"
    with dashboard_path.open("rb") as handle:
        dashboard = plistlib.load(handle)
    with tunnel_path.open("rb") as handle:
        tunnel = plistlib.load(handle)
    dashboard_script = (ROOT / "production" / "run_stage1_dashboard.sh").read_text(
        encoding="utf-8"
    )
    tunnel_script = (
        ROOT / "production" / "run_stage1_cloudflare_tunnel.sh"
    ).read_text(encoding="utf-8")
    tunnel_example = (
        ROOT / "production" / "cloudflared.stage1.example.yml"
    ).read_text(encoding="utf-8")

    assert dashboard["ProgramArguments"] == [
        "/Users/rziiiii/dev/workspace/production/run_stage1_dashboard.sh"
    ]
    assert tunnel["ProgramArguments"] == [
        "/Users/rziiiii/dev/workspace/production/run_stage1_cloudflare_tunnel.sh"
    ]
    assert "--host 127.0.0.1 --port 8765" in dashboard_script
    assert "soxl-stage1-dashboard.yml" in tunnel_script
    assert "hostname: soxl.atrl.me" in tunnel_example
    assert "service: http://127.0.0.1:8765" in tunnel_example
    assert tunnel_example.rstrip().endswith("service: http_status:404")


def test_stage1_dashboard_uses_muted_semantic_palette():
    page = (ROOT / "src" / "dashboard" / "stage1.html").read_text(encoding="utf-8")

    assert "--cyan: #6f95cf;" in page
    assert "--green: #4fb58a;" in page
    assert "--gold: #d2a85d;" in page
    assert "--blue: #907bd1;" in page
    assert "--red: #d96c75;" in page
    assert "#25f4c4" not in page
