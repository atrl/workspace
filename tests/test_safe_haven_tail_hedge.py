from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


def test_safe_haven_diagnostics_fail_convexity_when_otm_puts_do_not_10x():
    from src.workflow.safe_haven_tail_hedge import SafeHavenTailHedgeSpec, build_safe_haven_diagnostics

    spec = SafeHavenTailHedgeSpec(
        root=Path("/tmp/unused"),
        start="2026-06-30",
        end="2026-07-02",
        as_of="2026-07-02",
        event_date="2026-07-02",
        pre_event_entry_start="2026-06-30",
        targets=("put_80", "put_70"),
        funding_budgets=(0.01,),
    )
    spot = pd.DataFrame(
        [
            {"date": "2026-06-30", "instrument": "SOXL", "open": 100.0, "close": 104.0},
            {"date": "2026-07-01", "instrument": "SOXL", "open": 104.0, "close": 103.0},
            {"date": "2026-07-02", "instrument": "SOXL", "open": 100.0, "close": 80.0},
        ]
    )
    selections = pd.DataFrame(
        [
            {
                "date": "2026-06-30",
                "target": "put_80",
                "option_ticker": "O:SOXL260731P00080000",
                "strike_price": 80.0,
                "strike_to_open": 0.8,
                "dte": 31,
            },
            {
                "date": "2026-07-01",
                "target": "put_80",
                "option_ticker": "O:SOXL260731P00082000",
                "strike_price": 82.0,
                "strike_to_open": 0.79,
                "dte": 30,
            },
            {
                "date": "2026-07-02",
                "target": "put_80",
                "option_ticker": "O:SOXL260731P00080000",
                "strike_price": 80.0,
                "strike_to_open": 0.8,
                "dte": 29,
            },
            {
                "date": "2026-07-02",
                "target": "put_70",
                "option_ticker": "O:SOXL260731P00070000",
                "strike_price": 70.0,
                "strike_to_open": 0.7,
                "dte": 29,
            },
        ]
    )
    bars = pd.DataFrame(
        [
            {
                "date": "2026-06-30",
                "option_ticker": "O:SOXL260731P00080000",
                "open": 10.0,
                "high": 11.0,
                "low": 9.0,
                "close": 9.5,
                "volume": 10,
                "transactions": 3,
            },
            {
                "date": "2026-07-02",
                "option_ticker": "O:SOXL260731P00080000",
                "open": 15.0,
                "high": 22.0,
                "low": 14.0,
                "close": 20.0,
                "volume": 20,
                "transactions": 8,
            },
            {
                "date": "2026-07-03",
                "option_ticker": "O:SOXL260731P00070000",
                "open": 30.0,
                "high": 33.0,
                "low": 29.0,
                "close": 32.0,
                "volume": 1,
                "transactions": 1,
            },
        ]
    )

    result = build_safe_haven_diagnostics(spec, spot=spot, selections=selections, bars=bars)

    screen = result["safe_haven_screen"]
    assert screen["promotion_decision"] == "not_promoted_convexity_insufficient"
    assert screen["passes_safe_haven_convexity"] is False
    assert screen["event_targets_with_same_day_quotes"] == ["put_80"]
    assert screen["max_pre_event_net_multiple_to_event"] == pytest.approx((0.97 * 20.0) / (1.03 * 10.0))
    risk = result["risk_analysis"].set_index("risk")
    assert "excluded 1 raw option bar rows after as_of" in risk.loc["future_leakage_control", "evidence"]


def test_return_metrics_compounds_and_reports_drawdown():
    from src.workflow.safe_haven_tail_hedge import _return_metrics

    metrics = _return_metrics(pd.Series([0.1, -0.2, 0.05]).to_numpy())

    assert metrics["total_return_pct"] == pytest.approx(((1.1 * 0.8 * 1.05) - 1.0) * 100.0)
    assert metrics["max_drawdown_pct"] == pytest.approx(-20.0)
