from __future__ import annotations

import numpy as np
import pandas as pd
import pandas.testing as pdt


def _synthetic_panel(periods: int = 96) -> pd.DataFrame:
    dates = pd.bdate_range("2025-01-02", periods=periods)
    rows: list[pd.DataFrame] = []
    curves = {
        "SOXL": 20.0 * np.exp(np.linspace(0.0, 0.18, periods) + 0.06 * np.sin(np.arange(periods) / 3.0)),
        "TQQQ": 50.0 * np.exp(np.linspace(0.0, 0.12, periods) + 0.04 * np.sin(np.arange(periods) / 4.0)),
        "UPRO": 40.0 * np.exp(np.linspace(0.0, 0.09, periods) + 0.03 * np.sin(np.arange(periods) / 5.0)),
    }
    for instrument, close in curves.items():
        underlying_close = 100.0 * np.exp(np.cumsum(np.r_[0.0, np.diff(np.log(close)) / 3.0]))
        rows.append(
            pd.DataFrame(
                {
                    "instrument": instrument,
                    "datetime": dates,
                    "open": close * 0.995,
                    "high": close * 1.02,
                    "low": close * 0.98,
                    "close": close,
                    "volume": 1_000_000 + np.arange(periods),
                    "underlying_close": underlying_close,
                    "vix": 17.0 + 0.5 * np.sin(np.arange(periods) / 8.0),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def test_build_frame_has_transferable_features_and_purged_path_labels():
    from src.factor.multi_3x_high_chop import PATH_LABELS, TRANSFERABLE_FEATURES, build_multi_3x_high_chop_frame

    horizon = 10
    frame = build_multi_3x_high_chop_frame(
        _synthetic_panel(),
        label_horizon=horizon,
        normalization_window=30,
        min_periods=10,
    )

    assert set(TRANSFERABLE_FEATURES).issubset(frame.columns)
    assert frame[TRANSFERABLE_FEATURES].notna().any().all()
    assert set(frame["path_label"].dropna().astype(int)).issubset(set(PATH_LABELS.values()))
    for _, asset in frame.groupby("instrument"):
        assert asset.sort_values("datetime")["path_label"].tail(horizon + 1).isna().all()
    assert frame["high_chop_intensity"].dropna().between(0.0, 1.0).all()


def test_transferable_features_are_invariant_to_future_mutation():
    from src.factor.multi_3x_high_chop import TRANSFERABLE_FEATURES, build_multi_3x_high_chop_frame

    panel = _synthetic_panel()
    cutoff = pd.Timestamp("2025-04-04")
    original = build_multi_3x_high_chop_frame(panel, label_horizon=10, normalization_window=30, min_periods=10)
    mutated = panel.copy()
    future = mutated["datetime"] > cutoff
    mutated.loc[future, ["open", "high", "low", "close", "underlying_close"]] *= 7.0
    mutated.loc[future, "vix"] = 80.0
    rebuilt = build_multi_3x_high_chop_frame(mutated, label_horizon=10, normalization_window=30, min_periods=10)

    cols = ["instrument", "datetime", *TRANSFERABLE_FEATURES]
    left = original.loc[original["datetime"] <= cutoff, cols].reset_index(drop=True)
    right = rebuilt.loc[rebuilt["datetime"] <= cutoff, cols].reset_index(drop=True)
    pdt.assert_frame_equal(left, right, check_exact=False, rtol=1e-12, atol=1e-12)


def test_transferable_feature_contract_excludes_constituent_and_asset_identity_fields():
    from src.factor.multi_3x_high_chop import TRANSFERABLE_FEATURES

    forbidden = {name for name in TRANSFERABLE_FEATURES if name.startswith("soxx_const_")}
    assert forbidden == set()
    assert "instrument" not in TRANSFERABLE_FEATURES


def test_purge_segment_tail_removes_labels_that_cross_boundary():
    from src.factor.multi_3x_high_chop import purge_segment_tail

    frame = pd.DataFrame(
        {
            "instrument": ["SOXL"] * 12 + ["TQQQ"] * 12,
            "datetime": list(pd.bdate_range("2025-01-02", periods=12)) * 2,
            "path_label": [0.0] * 24,
        }
    )
    purged = purge_segment_tail(frame, end="2025-01-15", horizon=3)

    in_segment = purged["datetime"] <= pd.Timestamp("2025-01-15")
    assert purged.loc[in_segment].groupby("instrument")["path_label"].apply(lambda s: s.tail(3).isna().all()).all()
