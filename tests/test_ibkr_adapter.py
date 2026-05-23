from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_ibkr_broker_lives_under_src_with_production_compatibility_import():
    from production.ib_broker import IBConnectionConfig as CompatIBConnectionConfig
    from production.ib_broker import IBBroker as CompatIBBroker
    from src.soxl_quant.app.ibkr import IBConnectionConfig, IBBroker

    assert CompatIBBroker is IBBroker
    assert CompatIBConnectionConfig is IBConnectionConfig


def test_production_ibkr_entrypoints_import_src_adapter():
    entrypoints = [
        ROOT / "production" / "soxl_paper_trading.py",
        ROOT / "production" / "ib_manual_session.py",
        ROOT / "production" / "check_ibkr_gateway.py",
    ]

    for path in entrypoints:
        assert path.exists()
        text = path.read_text(encoding="utf-8")
        assert "from production.ib_broker import IBBroker" not in text
        assert "from production.ib_client import IBConnectionConfig" not in text


def test_check_ibkr_gateway_entrypoint_runs_without_pythonpath():
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "production" / "check_ibkr_gateway.py"),
            "--port",
            "1",
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env={},
    )

    assert result.returncode == 1
    assert "FAIL: IBKR TWS/Gateway is NOT reachable" in result.stdout
    assert "ModuleNotFoundError" not in result.stderr


def test_ibkr_port_check_lives_under_src():
    from src.soxl_quant.app.ibkr import check_ibkr_port

    assert check_ibkr_port("127.0.0.1", 1, timeout=0.01) is False


def test_importing_ibkr_app_module_does_not_initialize_runtime_stack():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import src.soxl_quant.app.ibkr; print('ok')",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stdout.strip() == "ok"
    assert "Gym has been unmaintained" not in result.stderr


def test_ibkr_broker_connect_fails_fast_when_gateway_port_closed():
    from src.soxl_quant.app.ibkr import IBConnectionConfig, IBBroker

    broker = IBBroker(config=IBConnectionConfig(port=1, connect_timeout_sec=0.01))

    with pytest.raises(RuntimeError, match="IBKR TWS/Gateway is not reachable"):
        broker.connect()


def test_ibkr_submitted_order_does_not_assume_resulting_position_changed():
    from src.soxl_quant.app.ibkr import IBConnectionConfig, IBBroker

    class FakeClient:
        def account_summary(self):
            return {"NetLiquidation": 10000.0, "TotalCashValue": 10000.0}

        def positions(self):
            return []

        def place_target_order(self, symbol, quantity_delta, use_market_order, limit_price):
            assert symbol == "SOXL"
            assert quantity_delta == 100
            assert use_market_order is False
            assert limit_price == 50.0
            return "123", "Submitted"

    broker = IBBroker(config=IBConnectionConfig())
    broker.client = FakeClient()

    _commands, _statuses, result = broker.place_target_position(
        date="2026-05-20",
        symbol="SOXL",
        target_position=0.5,
        price=50.0,
        reason="unit-test",
    )

    assert result.submitted is True
    assert result.status == "Submitted"
    assert result.resulting_position == 0.0
    assert result.filled_qty == 0
    assert result.avg_price == 0.0


def test_paper_trading_accepts_ibkr_broker_arg(monkeypatch: pytest.MonkeyPatch):
    from production import soxl_paper_trading

    monkeypatch.setattr(sys, "argv", ["soxl_paper_trading.py", "--broker", "ibkr"])

    args = soxl_paper_trading.parse_args()

    assert args.broker == "ibkr"
    assert args.ibkr_host == "127.0.0.1"
    assert args.ibkr_port == 7497
    assert args.ibkr_client_id == 37


def test_daily_scheduler_accepts_ibkr_broker_arg(monkeypatch: pytest.MonkeyPatch):
    from production import soxl_daily_scheduler

    monkeypatch.setattr(sys, "argv", ["soxl_daily_scheduler.py", "--broker", "ibkr"])

    args = soxl_daily_scheduler.parse_args()

    assert args.broker == "ibkr"
