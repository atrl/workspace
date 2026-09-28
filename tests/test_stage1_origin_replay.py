import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from src.analysis.stage1_origin_replay import _buy_hold, build_origin_window
from src.workflow.stage1_open_action_value import Stage1OpenActionValueSpec, _simulate_on_policy_router
from src.workflow.intraday_recovery_direct_allocation import IntradayRecoveryDirectAllocationSpec


def test_origin_window_includes_last_open_to_close_without_a_future_open():
    dates = pd.to_datetime(["2026-09-24", "2026-09-25"])
    prices = pd.DataFrame(
        [[100, 200, 105, 205], [110, 220, 121, 231]], index=dates,
        columns=pd.MultiIndex.from_tuples([("open", "SOXL"), ("open", "SOXX"), ("close", "SOXL"), ("close", "SOXX")]),
    )
    inference = pd.DataFrame(
        {"trade_date": dates, "soxl_weight": 0.1, "soxx_weight": 0.2, "cash_weight": 0.7},
        index=pd.DatetimeIndex(["2026-09-23", "2026-09-24"], name="signal_date"),
    )
    first_hour = pd.DataFrame({"early_ret_soxl": 0.01, "early_ret_soxx": 0.01}, index=dates)
    frame = build_origin_window(inference, prices, first_hour, start="2026-09-24", end="2026-09-25")
    assert frame["soxl_return"].tolist() == pytest.approx([0.1, 0.1])
    assert frame.iloc[-1]["exit_basis"] == "terminal_close"
    assert frame.iloc[-1]["exit_date"] == pd.Timestamp("2026-09-25")
    assert frame.iloc[-1]["signal_date"] == pd.Timestamp("2026-09-24")


def test_buy_hold_charges_only_funded_entry_fee_and_never_rebalances():
    frame = pd.DataFrame({
        "signal_date": pd.to_datetime(["2026-09-23", "2026-09-24"]),
        "trade_date": pd.to_datetime(["2026-09-24", "2026-09-25"]),
        "exit_date": pd.to_datetime(["2026-09-25", "2026-09-25"]),
        "soxl_return": [0.1, 0.2], "soxx_return": [0, 0],
    })
    replay = _buy_hold(frame, "SOXL")
    assert replay.iloc[-1]["equity"] == pytest.approx(1.1 * 1.2 / 1.001)
    assert replay["execute_open_trade"].tolist() == [True, False]
    assert replay.iloc[-1]["fees"] == 0.0
    assert np.isclose(replay.iloc[-1]["ending_cash_weight"], 0.0)


def test_later_origin_resets_controller_instead_of_inheriting_previous_loss():
    frame = pd.DataFrame({
        "signal_date": pd.to_datetime(["2026-07-24", "2026-07-27"]),
        "trade_date": pd.to_datetime(["2026-07-27", "2026-07-28"]),
        "exit_date": pd.to_datetime(["2026-07-28", "2026-07-29"]),
        "soxl_weight": [0.1, 0.1], "soxx_weight": [0.2, 0.2], "cash_weight": [0.7, 0.7],
        "soxl_return": [-0.8, 0.0], "soxx_return": [0.0, 0.0],
        "tail_probability": [0.1, 0.1], "alpha_probability": [0.8, 0.8],
        "early_ret_soxl": [np.nan, np.nan], "early_ret_soxx": [np.nan, np.nan],
    })
    options = dict(models={}, recovery_models={}, source_spec=IntradayRecoveryDirectAllocationSpec(root=Path(".")),
                   spec=Stage1OpenActionValueSpec(root=Path(".")),
                   selected_model={"action_beta": 0.6, "probability_threshold": 0.825, "minimum_source_alpha_probability": 0.25},
                   segment="test_origin")
    full = _simulate_on_policy_router(frame, **options)
    restarted = _simulate_on_policy_router(frame.iloc[1:], **options)
    assert full.iloc[1]["portfolio_drawdown"] > 0.08
    assert restarted.iloc[0]["portfolio_drawdown"] == 0.0
    assert bool(restarted.iloc[0]["execute_open_trade"]) is True
