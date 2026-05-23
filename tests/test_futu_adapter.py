from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import time
import types

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_futu_broker_lives_under_src_with_production_compatibility_import():
    from production.futu_broker import FutuBroker as CompatFutuBroker
    from src.soxl_quant.app.futu import FutuBroker

    assert CompatFutuBroker is FutuBroker


def test_production_futu_entrypoints_import_src_adapter():
    entrypoints = [
        ROOT / "production" / "soxl_paper_trading.py",
        ROOT / "production" / "soxl_futu_runner.py",
        ROOT / "production" / "futu_manual_session.py",
        ROOT / "production" / "check_futu_opend.py",
        ROOT / "production" / "composed_session.py",
    ]

    for path in entrypoints:
        text = path.read_text(encoding="utf-8")
        assert "from production.futu_broker import FutuBroker" not in text


def test_check_futu_opend_entrypoint_runs_without_pythonpath():
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "production" / "check_futu_opend.py"),
            "--port",
            "1",
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env={},
    )

    assert result.returncode == 1
    assert "FAIL: OpenD is NOT reachable" in result.stdout
    assert "ModuleNotFoundError" not in result.stderr


def test_futu_opend_port_check_lives_under_src():
    from src.soxl_quant.app.futu import check_opend_port

    assert check_opend_port("127.0.0.1", 1, timeout=0.01) is False


def test_importing_futu_app_module_does_not_initialize_runtime_stack():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import src.soxl_quant.app.futu; print('ok')",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stdout.strip() == "ok"
    assert "Gym has been unmaintained" not in result.stderr


def test_futu_broker_connect_fails_fast_when_sdk_connect_blocks(monkeypatch: pytest.MonkeyPatch):
    import src.soxl_quant.app.futu as futu_app
    from src.soxl_quant.app.futu import FutuBroker

    class BlockingQuoteContext:
        def __init__(self, *args, **kwargs):
            time.sleep(10)

    fake_futu = types.SimpleNamespace(
        OpenQuoteContext=BlockingQuoteContext,
        OpenTradeContext=lambda *args, **kwargs: object(),
    )
    monkeypatch.setitem(sys.modules, "futu", fake_futu)
    monkeypatch.setattr(futu_app, "check_opend_port", lambda *args, **kwargs: True)

    broker = FutuBroker(connect_timeout_sec=0.1)

    with pytest.raises(RuntimeError, match="timed out"):
        broker.connect()


def test_futu_broker_uses_us_security_trade_context(monkeypatch: pytest.MonkeyPatch):
    import src.soxl_quant.app.futu as futu_app
    from src.soxl_quant.app.futu import FutuBroker

    calls: list[tuple[str, dict]] = []

    class FakeSysConfig:
        @staticmethod
        def set_all_thread_daemon(value):
            calls.append(("set_all_thread_daemon", {"value": value}))

    class FakeQuoteContext:
        def __init__(self, **kwargs):
            calls.append(("quote", kwargs))

    class FakeTradeContext:
        def __init__(self, **kwargs):
            calls.append(("trade", kwargs))

    fake_futu = types.SimpleNamespace(
        SysConfig=FakeSysConfig,
        OpenQuoteContext=FakeQuoteContext,
        OpenSecTradeContext=FakeTradeContext,
        TrdMarket=types.SimpleNamespace(US="US"),
    )
    monkeypatch.setitem(sys.modules, "futu", fake_futu)
    monkeypatch.setattr(futu_app, "check_opend_port", lambda *args, **kwargs: True)

    broker = FutuBroker(host="127.0.0.1", port=11111, connect_timeout_sec=1)
    broker.connect()

    assert ("set_all_thread_daemon", {"value": True}) in calls
    assert ("quote", {"host": "127.0.0.1", "port": 11111}) in calls
    assert ("trade", {"filter_trdmarket": "US", "host": "127.0.0.1", "port": 11111}) in calls
