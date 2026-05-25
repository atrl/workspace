from __future__ import annotations

import pandas as pd

from src.soxl_quant.app.drawdown_budget_overlay import (
    DrawdownBudgetOverlayConfig,
    apply_drawdown_budget_overlay,
)


def test_drawdown_budget_overlay_caps_after_realized_equity_drawdown():
    dates = pd.date_range("2026-01-01", periods=4, freq="B")
    frame = pd.DataFrame({"datetime": dates, "position": [1.0, 1.0, 1.0, 1.0]})
    prices = pd.Series([100.0, 80.0, 80.0, 80.0], index=dates)

    out = apply_drawdown_budget_overlay(
        frame,
        prices,
        DrawdownBudgetOverlayConfig(
            start_drawdown=0.10,
            full_drawdown=0.20,
            min_position=0.25,
            trade_threshold=0.0,
            trade_lock_days=0,
        ),
    )

    assert out["base_position"].tolist() == [1.0, 1.0, 1.0, 1.0]
    assert out["position"].round(4).tolist() == [1.0, 0.25, 0.25, 0.25]
    assert out["equity_drawdown"].round(4).tolist() == [0.0, 0.2, 0.2, 0.2]
    assert out["drawdown_budget_active"].tolist() == [False, True, True, True]


def test_drawdown_budget_overlay_bypasses_lock_for_risk_reducing_sells_only():
    dates = pd.date_range("2026-01-01", periods=4, freq="B")
    frame = pd.DataFrame({"datetime": dates, "position": [1.0, 1.0, 1.0, 1.0]})
    prices = pd.Series([100.0, 80.0, 80.0, 90.0], index=dates)

    out = apply_drawdown_budget_overlay(
        frame,
        prices,
        DrawdownBudgetOverlayConfig(
            start_drawdown=0.10,
            full_drawdown=0.20,
            min_position=0.25,
            trade_threshold=0.20,
            trade_lock_days=5,
            sell_trade_threshold=0.0,
            sell_trade_lock_days=0,
        ),
    )

    assert out["position"].round(4).tolist() == [1.0, 0.25, 0.25, 0.25]
    assert out["effective_trade_lock_days"].tolist() == [5, 0, 5, 5]
    assert out["effective_trade_threshold"].round(4).tolist() == [0.2, 0.0, 0.2, 0.2]


def test_drawdown_budget_overlay_requires_single_position_column():
    dates = pd.date_range("2026-01-01", periods=2, freq="B")
    frame = pd.DataFrame([["2026-01-01", 0.5, 0.6], ["2026-01-02", 0.5, 0.6]], columns=["datetime", "position", "position"])
    prices = pd.Series([100.0, 101.0], index=dates)

    try:
        apply_drawdown_budget_overlay(frame, prices)
    except ValueError as exc:
        assert "duplicate `position`" in str(exc)
    else:
        raise AssertionError("duplicate position columns did not raise ValueError")


if __name__ == "__main__":
    test_drawdown_budget_overlay_caps_after_realized_equity_drawdown()
    test_drawdown_budget_overlay_bypasses_lock_for_risk_reducing_sells_only()
    test_drawdown_budget_overlay_requires_single_position_column()
    print("test_drawdown_budget_overlay: OK")
