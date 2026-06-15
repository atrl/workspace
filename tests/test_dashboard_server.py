from __future__ import annotations

import io
import json
from http.server import SimpleHTTPRequestHandler

import pandas as pd

import scripts.dashboard_server as dashboard_server
from scripts.dashboard_server import _build_daily_rows, _health_payload, _response_window_from_rows


def test_build_daily_rows_include_market_ohlc_for_rerun_kline():
    idx = pd.to_datetime(["2026-06-05", "2026-06-08"])
    account = pd.Series([1_000_000.0, 1_010_000.0], index=idx)
    price = pd.Series([182.54, 211.44], index=idx)
    market = {
        "2026-06-05": {
            "open": 180.0,
            "high": 184.0,
            "low": 178.0,
            "close": 182.54,
            "volume": 123,
            "ma180": 90.0,
        },
        "2026-06-08": {
            "open": 190.0,
            "high": 212.0,
            "low": 188.0,
            "close": 211.44,
            "volume": 456,
            "ma180": 91.0,
        },
    }

    rows = _build_daily_rows(
        account=account,
        price=price,
        position_history=[0.0, 0.5],
        trade_signals=[(idx[1], 211.44, 0.5, "buy")],
        market_lookup=market,
        feature_lookup={
            "2026-06-08": {
                "alpha_signal": 1.25,
                "shared_feature": 20.0,
            }
        },
    )

    assert rows[0]["open"] == 180.0
    assert rows[0]["high"] == 184.0
    assert rows[0]["low"] == 178.0
    assert rows[0]["close"] == 182.54
    assert rows[0]["volume"] == 123
    assert rows[1]["action"] == "BUY"
    assert rows[1]["position"] == 0.5
    assert rows[1]["feature_snapshot"] == {"alpha_signal": 1.25, "shared_feature": 20.0}


def test_health_payload_marks_dashboard_server_api_available():
    payload = _health_payload()

    assert payload["ok"] is True
    assert payload["service"] == "dashboard_server"
    assert payload["run_backtest_api"] == "/api/run-backtest"


def test_dashboard_server_disables_browser_cache_for_static_dashboard_payloads(monkeypatch):
    sent: list[tuple[str, str]] = []
    handler = object.__new__(dashboard_server.DashboardHandler)
    handler.send_header = lambda key, value: sent.append((key, value))
    monkeypatch.setattr(SimpleHTTPRequestHandler, "end_headers", lambda _self: None)

    handler.end_headers()

    assert ("Cache-Control", "no-store, max-age=0") in sent
    assert ("Pragma", "no-cache") in sent


def test_response_window_from_rows_uses_actual_row_dates():
    rows = [
        {"datetime": "2026-06-05"},
        {"datetime": "2026-06-10"},
    ]

    window = _response_window_from_rows(rows, "2026-06-05", "2026-06-11")

    assert window == {"start": "2026-06-05", "end": "2026-06-10", "rows": 2}


def test_profile_replay_start_reads_manifest_csv_start(tmp_path, monkeypatch):
    data_dir = tmp_path / "dashboard_data"
    replay_csv = tmp_path / "runtime" / "model" / "card" / "smoke" / "smoke_soxl_daily.csv"
    data_dir.mkdir()
    replay_csv.parent.mkdir(parents=True)
    replay_csv.write_text(
        "\n".join(
            [
                "datetime,strategy_nav,price,buyhold_nav,position,excess,excess_pct,action",
                "2020-01-02,1000000,10,1000000,0.3,0,0,BUY",
                "2026-06-10,2000000,20,3000000,0.5,0,0,HOLD",
            ]
        ),
        encoding="utf-8",
    )
    (data_dir / "model_replay_freshness.json").write_text(
        """
{
  "profiles": [
    {
      "profile": "smoke",
      "output": "runtime/model/card/smoke/smoke_soxl_daily.csv",
      "result": "current"
    }
  ]
}
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(dashboard_server, "ROOT", tmp_path)
    monkeypatch.setattr(dashboard_server, "DASHBOARD_DATA_DIR", data_dir)

    assert dashboard_server._profile_replay_start("smoke") == "2020-01-02"


def test_slice_daily_rows_preserves_full_replay_position_and_nav():
    rows = [
        {"datetime": "2020-01-02", "strategy_nav": 1_000_000.0, "position": 0.3},
        {"datetime": "2026-03-09", "strategy_nav": 12_795_559.84, "position": 1.0},
        {"datetime": "2026-03-10", "strategy_nav": 12_969_551.03, "position": 1.0},
        {"datetime": "2026-03-11", "strategy_nav": 13_253_960.55, "position": 0.3},
    ]

    sliced = dashboard_server._slice_daily_rows(rows, "2026-03-09", "2026-03-10")

    assert sliced == [
        {"datetime": "2026-03-09", "strategy_nav": 12_795_559.84, "position": 1.0},
        {"datetime": "2026-03-10", "strategy_nav": 12_969_551.03, "position": 1.0},
    ]


def test_backtest_request_scope_uses_full_test_seg_for_replay_aligned_runs(monkeypatch):
    monkeypatch.setattr(dashboard_server, "_profile_replay_start", lambda _profile: "2020-01-02")

    scope = dashboard_server._backtest_request_scope(
        "smoke",
        "2026-03-09",
        "2026-03-13",
        default_test_seg=("2020-01-01", "2026-12-31"),
    )

    assert scope == {
        "warmup_start": "2020-01-02",
        "feature_start": "2020-01-02",
        "feature_end": "",
        "test_seg": ("2020-01-01", "2026-12-31"),
        "replay_aligned": True,
    }


def test_run_backtest_runs_soxlbacktest_reference_profile(monkeypatch):
    body = json.dumps(
        {
            "profile": "soxl_backtest_policy_reimplementation_v1",
            "start_date": "2026-01-02",
            "end_date": "2026-01-31",
        }
    ).encode("utf-8")
    sent: dict[str, object] = {}
    handler = object.__new__(dashboard_server.DashboardHandler)
    handler.headers = {"Content-Length": str(len(body))}
    handler.rfile = io.BytesIO(body)
    handler._send_json = lambda data, status=200: sent.update({"data": data, "status": status})
    monkeypatch.setattr(dashboard_server, "_load_market_lookup", lambda: {
        "2026-01-02": {"open": 47.0, "high": 48.0, "low": 46.0, "close": 47.24, "volume": 1000}
    })
    monkeypatch.setattr(
        dashboard_server,
        "replay_legacy_soxl_policy",
        lambda **_kwargs: dashboard_server.LegacySoxlReplayResult(
            daily=pd.DataFrame(
                [
                    {
                        "date": "2026-01-02",
                        "price": 47.24,
                        "strategy_nav": 100000.0,
                        "buyhold_nav": 100000.0,
                        "position": 0.0,
                        "excess": 0.0,
                        "excess_pct": 0.0,
                        "action": "HOLD",
                        "panic_score": 0.0,
                        "HV_20": 1.0,
                        "HV_30": 1.0,
                        "VOL_COEF": 1.0,
                    },
                    {
                        "date": "2026-01-05",
                        "price": 49.22,
                        "strategy_nav": 100000.0,
                        "buyhold_nav": 104191.36,
                        "position": 0.0,
                        "excess": -4191.36,
                        "excess_pct": -4.02,
                        "action": "BUY",
                        "panic_score": -150.0,
                        "HV_20": 1.27,
                        "HV_30": 1.11,
                        "VOL_COEF": 1.27,
                    },
                ]
            ),
            trades=pd.DataFrame(
                [{"date": "2026-01-05", "type": "buy", "price": 49.22, "position": 0.3}]
            ),
            metrics={"rulebase_return_pct": 0.0, "trade_count": 1},
        ),
    )

    handler._handle_run_backtest()

    assert sent["status"] == 200
    assert sent["data"]["profile"] == "soxl_backtest_policy_reimplementation_v1"
    assert sent["data"]["source_kind"] == "soxlbacktest_qlib_buysell"
    assert sent["data"]["feature_columns"] == ["panic_score", "HV_20", "HV_30", "VOL_COEF"]
    assert sent["data"]["rows"][1]["action"] == "BUY"
    assert sent["data"]["rows"][1]["feature_snapshot"] == {
        "panic_score": -150.0,
        "HV_20": 1.27,
        "HV_30": 1.11,
        "VOL_COEF": 1.27,
    }


def test_run_backtest_accepts_frontend_camel_case_dates(monkeypatch):
    body = json.dumps(
        {
            "profile": "soxl_backtest_policy_reimplementation_v1",
            "startDate": "2026-01-02",
            "endDate": "2026-01-31",
        }
    ).encode("utf-8")
    sent: dict[str, object] = {}
    captured: dict[str, object] = {}
    handler = object.__new__(dashboard_server.DashboardHandler)
    handler.headers = {"Content-Length": str(len(body))}
    handler.rfile = io.BytesIO(body)
    handler._send_json = lambda data, status=200: sent.update({"data": data, "status": status})
    monkeypatch.setattr(dashboard_server, "_load_market_lookup", lambda: {})

    def fake_replay(**kwargs):
        captured.update(kwargs)
        return dashboard_server.LegacySoxlReplayResult(
            daily=pd.DataFrame(
                [
                    {
                        "date": "2026-01-02",
                        "price": 47.24,
                        "strategy_nav": 100000.0,
                        "buyhold_nav": 100000.0,
                        "position": 0.0,
                        "excess": 0.0,
                        "excess_pct": 0.0,
                        "action": "HOLD",
                        "panic_score": 0.0,
                        "HV_20": 1.0,
                        "HV_30": 1.0,
                        "VOL_COEF": 1.0,
                    }
                ]
            ),
            trades=pd.DataFrame(),
            metrics={"rulebase_return_pct": 0.0, "trade_count": 0},
        )

    monkeypatch.setattr(dashboard_server, "replay_legacy_soxl_policy", fake_replay)

    handler._handle_run_backtest()

    assert sent["status"] == 200
    assert captured["start"] == "2026-01-02"
    assert captured["end"] == "2026-01-31"


def test_run_backtest_returns_400_for_unknown_non_reference_profile(monkeypatch):
    body = json.dumps(
        {
            "profile": "missing_model",
            "start_date": "2026-01-02",
            "end_date": "2026-01-31",
        }
    ).encode("utf-8")
    sent: dict[str, object] = {}
    handler = object.__new__(dashboard_server.DashboardHandler)
    handler.headers = {"Content-Length": str(len(body))}
    handler.rfile = io.BytesIO(body)
    handler._send_json = lambda data, status=200: sent.update({"data": data, "status": status})
    monkeypatch.setattr(
        dashboard_server,
        "load_profile",
        lambda _profile: (_ for _ in ()).throw(FileNotFoundError("unknown profile")),
    )

    handler._handle_run_backtest()

    assert sent["status"] == 400
    assert sent["data"]["profile"] == "missing_model"
    assert sent["data"]["runnable"] is False
    assert "traceback" not in sent["data"]
