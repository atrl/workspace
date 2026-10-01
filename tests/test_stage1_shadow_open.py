from datetime import datetime
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from src.schedule import stage1_shadow_open, us_open_quotes


NOW = datetime(2026, 10, 1, 9, 31, 30, tzinfo=ZoneInfo("America/New_York"))


@pytest.fixture
def quote_context(monkeypatch):
    class Context:
        state = "AFTERNOON"
        rows = [
            {"code": "US.SOXL", "last_price": 140.0, "update_time": "2026-10-01 09:31:20"},
            {"code": "US.SOXX", "last_price": 560.0, "update_time": "2026-10-01 09:31:21"},
        ]
        closed = False
        snapshots = 0

        def get_global_state(self):
            return 0, {"market_us": self.state}

        def get_market_snapshot(self, symbols):
            assert symbols == ["US.SOXL", "US.SOXX"]
            self.snapshots += 1
            return 0, pd.DataFrame(self.rows)

        def close(self):
            self.closed = True

    context = Context()
    monkeypatch.setitem(sys.modules, "futu", SimpleNamespace(
        OpenQuoteContext=lambda **kwargs: context, RET_OK=0))
    return context


def test_us_open_uses_afternoon_state_with_fresh_new_york_quotes(quote_context):
    prices, stamp = us_open_quotes.read_open_quotes(host="127.0.0.1", port=11111, now=NOW)
    assert prices == {"SOXL": 140.0, "SOXX": 560.0}
    assert stamp == "2026-10-01T09:31:20-04:00"
    assert quote_context.closed


@pytest.mark.parametrize("state", ["MORNING", "PRE_MARKET_BEGIN", "AFTER_HOURS_BEGIN", "CLOSED", "OVERNIGHT", ""])
def test_us_open_rejects_other_market_states_without_fetching_quotes(quote_context, state):
    quote_context.state = state
    with pytest.raises(ValueError, match="US regular session is not open"):
        us_open_quotes.read_open_quotes(host="127.0.0.1", port=11111, now=NOW)
    assert quote_context.closed and quote_context.snapshots == 0


@pytest.mark.parametrize("stamp", ["2026-09-30 09:31:20", "2026-10-01 09:29:29", "2026-10-01 09:31:36"])
def test_each_quote_must_be_fresh_and_causal(quote_context, stamp):
    quote_context.rows = [dict(row) for row in quote_context.rows]
    quote_context.rows[1]["update_time"] = stamp
    with pytest.raises(ValueError, match="stale, future, or off-session"):
        us_open_quotes.read_open_quotes(host="127.0.0.1", port=11111, now=NOW)
    assert quote_context.closed


def test_aware_quote_timestamps_are_converted_to_new_york(quote_context):
    quote_context.rows = [dict(row, update_time="2026-10-01T13:31:20+00:00") for row in quote_context.rows]
    assert us_open_quotes.read_open_quotes(host="127.0.0.1", port=11111, now=NOW)[1] == "2026-10-01T09:31:20-04:00"


def test_recovery_quotes_use_same_regular_session_and_freshness_contract(quote_context):
    recovery = NOW.replace(hour=10)
    quote_context.rows = [dict(row, update_time="2026-10-01 10:31:20") for row in quote_context.rows]
    prices, stamp = us_open_quotes.read_session_quotes(host="127.0.0.1", port=11111, slot="recovery", now=recovery)
    assert prices["SOXL"] == 140.0 and stamp == "2026-10-01T10:31:20-04:00"


@pytest.mark.parametrize("price", [float("nan"), float("inf"), 0, -1])
def test_nonpositive_or_nonfinite_price_is_rejected(quote_context, price):
    quote_context.rows = [dict(row) for row in quote_context.rows]
    quote_context.rows[0]["last_price"] = price
    with pytest.raises(ValueError, match="invalid opening quote price"):
        us_open_quotes.read_open_quotes(host="127.0.0.1", port=11111, now=NOW)
    assert quote_context.closed


@pytest.mark.parametrize("case", ["missing", "duplicate"])
def test_snapshot_requires_both_unique_symbols(quote_context, case):
    quote_context.rows = quote_context.rows[:1] if case == "missing" else [*quote_context.rows, quote_context.rows[0]]
    with pytest.raises(ValueError, match="incomplete|duplicate"):
        us_open_quotes.read_open_quotes(host="127.0.0.1", port=11111, now=NOW)


@pytest.mark.parametrize("now", [NOW.replace(tzinfo=None), NOW.replace(hour=8), NOW.replace(minute=37), NOW.replace(day=3)])
def test_clock_gate_prevents_quote_connections(monkeypatch, now):
    def forbidden(**kwargs):
        raise AssertionError("quote connection attempted outside observation window")
    monkeypatch.setitem(sys.modules, "futu", SimpleNamespace(OpenQuoteContext=forbidden, RET_OK=0))
    with pytest.raises(ValueError, match="timezone|outside"):
        us_open_quotes.read_open_quotes(host="127.0.0.1", port=11111, now=now)


def test_wrapper_injects_fixed_quote_reader_and_preserves_scheduler_result(monkeypatch, tmp_path):
    expected = {"status": "observed", "orders_submitted": 0, "execution_mode": "READ_ONLY_SHADOW"}
    seen = {}
    monkeypatch.setattr(stage1_shadow_open, "read_open_quotes", lambda **kwargs: seen.update(kwargs) or ("prices", "stamp"))

    def schedule(**kwargs):
        assert kwargs["root"] == tmp_path and kwargs["now"] == NOW
        assert kwargs["quote_reader"](host="127.0.0.1", port=11111) == ("prices", "stamp")
        return expected

    monkeypatch.setattr(stage1_shadow_open, "run_shadow_schedule", schedule)
    assert stage1_shadow_open.run_stage1_shadow_open(root=tmp_path, config_path=tmp_path / "config.json", now=NOW) is expected
    assert seen["now"] == NOW


def test_original_scheduler_still_matches_the_sealed_research_bundle():
    import json
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "research/stage1_economic_action_20260929/frozen_bundle_manifest.json").read_text())
    name = "src/schedule/stage1_economic_shadow.py"
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == manifest["code_sha256"][name]
