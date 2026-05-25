from __future__ import annotations

import pandas as pd

from src.soxl_quant.app.drawdown_budget_overlay import (
    DrawdownBudgetOverlayConfig,
    apply_drawdown_budget_overlay,
    drawdown_budget_overlay_metrics,
    screen_drawdown_budget_overlay_configs,
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


def test_drawdown_budget_overlay_metrics_scores_relative_and_drawdown_gate():
    dates = pd.date_range("2026-01-01", periods=4, freq="B")
    prices = pd.Series([100.0, 80.0, 80.0, 120.0], index=dates)
    frame = apply_drawdown_budget_overlay(
        pd.DataFrame({"datetime": dates, "position": [1.0, 1.0, 1.0, 1.0]}),
        prices,
        DrawdownBudgetOverlayConfig(
            start_drawdown=0.10,
            full_drawdown=0.20,
            min_position=0.25,
            trade_threshold=0.0,
            trade_lock_days=0,
        ),
    )

    metrics = drawdown_budget_overlay_metrics(frame, prices, target_relative_return=30.0, target_max_dd=15.0)

    assert round(metrics["return"], 2) == -10.0
    assert round(metrics["buy_and_hold_return"], 2) == 20.0
    assert round(metrics["relative_return"], 2) == -30.0
    assert round(metrics["max_dd"], 2) == 20.0
    assert round(metrics["avg_pos"], 4) == 0.625
    assert metrics["active_days"] == 2
    assert metrics["changed_days"] == 2
    assert metrics["target_gate"] is False
    assert round(metrics["relative_gap"], 4) == 60.0
    assert round(metrics["drawdown_gap"], 4) == 5.0


def test_screen_drawdown_budget_overlay_configs_uses_train_window_only():
    dates = pd.date_range("2026-01-01", periods=5, freq="B")
    prices = pd.Series([100.0, 80.0, 60.0, 60.0, 180.0], index=dates)
    frame = pd.DataFrame({"datetime": dates, "position": [1.0, 1.0, 1.0, 1.0, 1.0]})
    configs = [
        DrawdownBudgetOverlayConfig(
            start_drawdown=0.10,
            full_drawdown=0.20,
            min_position=0.0,
            trade_threshold=0.0,
            trade_lock_days=0,
        ),
        DrawdownBudgetOverlayConfig(
            start_drawdown=0.10,
            full_drawdown=0.20,
            min_position=1.0,
            trade_threshold=0.0,
            trade_lock_days=0,
        ),
    ]

    scored = screen_drawdown_budget_overlay_configs(
        frame,
        prices,
        configs,
        fold="demo",
        train_start="2026-01-01",
        train_end="2026-01-06",
        target_relative_return=30.0,
        target_max_dd=15.0,
    )

    assert scored["fold"].tolist() == ["demo", "demo"]
    assert scored["candidate_id"].tolist() == [0, 1]
    assert scored.loc[0, "train_rows"] == 4
    assert scored.loc[0, "full_rows"] == 5
    assert scored.loc[0, "train_max_dd"] < scored.loc[1, "train_max_dd"]
    assert scored.loc[0, "robust_score"] < scored.loc[1, "robust_score"]


if __name__ == "__main__":
    test_drawdown_budget_overlay_caps_after_realized_equity_drawdown()
    test_drawdown_budget_overlay_bypasses_lock_for_risk_reducing_sells_only()
    test_drawdown_budget_overlay_requires_single_position_column()
    test_drawdown_budget_overlay_metrics_scores_relative_and_drawdown_gate()
    test_screen_drawdown_budget_overlay_configs_uses_train_window_only()
    print("test_drawdown_budget_overlay: OK")
