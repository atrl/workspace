from __future__ import annotations

import numpy as np
import pandas as pd
import pandas.testing as pdt


def _price_frame(periods: int = 420) -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-02", periods=periods)
    index = np.arange(periods)
    returns = 0.003 + 0.009 * np.sin(index / 8.0)
    returns[260:320] = 0.002 + 0.035 * np.sin(index[260:320] / 2.0)
    returns[320:390] = 0.032 * np.sin(index[320:390] / 1.7)
    close = 20.0 * np.exp(np.cumsum(returns))
    return pd.DataFrame({"instrument": "SOXL", "datetime": dates, "close": close})


def test_path_state_features_are_bounded_causal_and_keep_direction():
    from src.factor.soxl_path_state_risk import PATH_STATE_FEATURES, build_soxl_path_state_frame

    frame = _price_frame()
    result = build_soxl_path_state_frame(frame, normalization_window=120, min_periods=40)
    assert result[PATH_STATE_FEATURES].notna().any().all()
    assert result[PATH_STATE_FEATURES].apply(lambda values: values.dropna().between(0.0, 1.0).all()).all()
    assert result["signed_efficiency_20"].dropna().between(-1.0, 1.0).all()
    assert result["structural_chop_age"].ge(0).all()
    assert result["fast_tail_5d"].tail(5).isna().all()
    assert result["structural_break_20d"].tail(20).isna().all()

    cutoff = pd.Timestamp("2021-03-31")
    mutated = frame.copy()
    mutated.loc[mutated["datetime"] > cutoff, "close"] *= 3.0
    rebuilt = build_soxl_path_state_frame(mutated, normalization_window=120, min_periods=40)
    cols = ["instrument", "datetime", "signed_efficiency_20", "rv_rank_20_252", *PATH_STATE_FEATURES]
    pdt.assert_frame_equal(
        result.loc[result["datetime"] <= cutoff, cols].reset_index(drop=True),
        rebuilt.loc[rebuilt["datetime"] <= cutoff, cols].reset_index(drop=True),
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )


def test_structural_chop_age_requires_all_state_conditions_and_resets():
    from src.factor.soxl_path_state_risk import structural_state_age

    state = pd.Series([False, True, True, True, True, False, True, True])
    assert structural_state_age(state).tolist() == [0, 1, 2, 3, 4, 0, 1, 2]


def test_future_path_labels_start_after_the_current_close():
    from src.factor.soxl_path_state_risk import forward_path_metrics

    close = pd.Series([100.0, 95.0, 89.0, 96.0, 80.0, 94.0])
    metrics = forward_path_metrics(close, horizon=5, adverse_threshold=-0.10, terminal_threshold=-0.05)
    assert metrics.loc[0, "forward_mae_5"] == -0.20
    assert metrics.loc[0, "forward_terminal_5"] == -0.06
    assert metrics.loc[0, "forward_label_5"] == 1.0
    assert metrics["forward_label_5"].iloc[1:].isna().all()
