from __future__ import annotations

import numpy as np
import pandas as pd


def _qlib_frame() -> pd.DataFrame:
    dates = pd.date_range("2020-01-02", periods=90, freq="B")
    rows: list[dict[str, object]] = []
    for instrument, scale, weight in [
        ("AMD", 1.0, 60.0),
        ("NVDA", 1.2, 40.0),
    ]:
        prices = 100.0 * np.exp(
            np.cumsum(
                0.001
                + scale * 0.01 * np.sin(np.arange(len(dates)) / 7.0)
            )
        )
        for index, (date, price) in enumerate(zip(dates, prices)):
            rows.append(
                {
                    "datetime": date,
                    "instrument": instrument,
                    "$open": price * 0.995,
                    "$high": price * 1.01,
                    "$low": price * 0.99,
                    "$close": price,
                    "$adj_close": price,
                    "$volume": 1_000_000 + 1000 * index,
                    "$pit_weight_pct": weight,
                    "$pit_available": 1.0,
                    "$pit_member": 1.0,
                    "$instrument_type": 0.0,
                    "$leverage_multiple": 1.0,
                    "$is_actual_2x": 0.0,
                }
            )
    return pd.DataFrame(rows)


def test_constituent_2x_tail_training_frame_is_past_feature_future_label() -> None:
    from src.factor.soxx_pit_tail_factors import (
        CONSTITUENT_MODEL_FEATURES,
        build_constituent_2x_tail_training_frame,
    )

    frame = build_constituent_2x_tail_training_frame(
        _qlib_frame(),
        horizon=5,
    )

    assert set(CONSTITUENT_MODEL_FEATURES).issubset(frame.columns)
    assert frame["training_scope"].eq("pit_member").all()
    assert frame["training_eligible"].all()
    assert frame["synthetic_2x_forward_return"].notna().sum() > 0
    assert frame["synthetic_2x_upside_label"].dropna().isin(
        [0.0, 1.0]
    ).all()
    assert np.allclose(
        frame["synthetic_2x_excess_1x"],
        frame["synthetic_2x_forward_return"]
        - frame["underlying_forward_return"],
        equal_nan=True,
    )
    assert frame.groupby("instrument")[
        "synthetic_2x_forward_return"
    ].tail(5).isna().all()
    assert (
        frame["synthetic_2x_crash_label"].dropna().isin([0.0, 1.0]).all()
    )


def test_daily_tail_factor_frame_lags_cross_sectional_information() -> None:
    from src.factor.soxx_pit_tail_factors import (
        build_constituent_2x_tail_training_frame,
        build_daily_soxx_tail_factor_frame,
    )

    training = build_constituent_2x_tail_training_frame(_qlib_frame())
    factors = build_daily_soxx_tail_factor_frame(
        training,
        lag_days=1,
        systemic_window=10,
        tail_window=10,
    )

    assert all(
        column.endswith("_lag1") for column in factors.attrs["feature_cols"]
    )
    first = factors.iloc[0]
    assert first[factors.attrs["feature_cols"]].isna().all()
    assert "soxx_pit_tail_absorption20_lag1" in factors.columns
    assert "soxx_pit_tail_hill20_lag1" in factors.columns
