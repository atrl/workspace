from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _synthetic_frames(rows: int = 100) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.date_range("2024-01-02", periods=rows, freq="B")
    step = np.arange(rows, dtype="float64")
    soxx_returns = 0.002 * np.sin(step / 4.0) + 0.001 * np.cos(step / 9.0)
    soxx_open = 100.0 * np.cumprod(1.0 + soxx_returns)
    soxl_open = 100.0 * np.cumprod(1.0 + 3.0 * soxx_returns)
    index_frame = pd.DataFrame(
        {
            "datetime": dates,
            "soxx_open": soxx_open,
            "soxl_open": soxl_open,
            "vix": 18.0 + 0.15 * np.sin(step / 15.0),
        }
    )

    parts = []
    specs = [
        ("AAA", 1.3, 0.004, 35.0),
        ("BBB", 0.8, -0.003, 30.0),
        ("CCC", 1.0, 0.002, 20.0),
        ("DDD", 0.6, -0.002, 15.0),
    ]
    for offset, (instrument, beta, amplitude, weight) in enumerate(specs):
        returns = beta * soxx_returns + amplitude * np.sin(step / (3.0 + offset) + offset)
        open_price = 50.0 * np.cumprod(1.0 + returns)
        close_price = open_price * (1.0 + 0.001 * np.cos(step / 5.0 + offset))
        spread = 0.01 + 0.002 * np.sin(step / 7.0 + offset)
        parts.append(
            pd.DataFrame(
                {
                    "datetime": dates,
                    "instrument": instrument,
                    "open": open_price,
                    "high": np.maximum(open_price, close_price) * (1.0 + spread),
                    "low": np.minimum(open_price, close_price) * (1.0 - spread),
                    "adj_close": close_price,
                    "volume": 1_000_000.0 * (1.0 + 0.2 * np.sin(step / 6.0 + offset)),
                    "weight_pct": weight,
                }
            )
        )
    return pd.concat(parts, ignore_index=True), index_frame


def test_hidden_dispersion_panel_builds_lagged_features_and_future_labels():
    from src.analysis.soxx_hidden_dispersion import (
        HIDDEN_DISPERSION_FEATURES,
        build_hidden_dispersion_panel,
    )

    constituents, index_frame = _synthetic_frames()
    panel = build_hidden_dispersion_panel(
        constituents,
        index_frame,
        train=("2024-01-02", "2024-03-29"),
        lag_days=1,
        horizon=5,
    )

    assert set(HIDDEN_DISPERSION_FEATURES).issubset(panel.daily.columns)
    assert panel.daily["residual_dispersion_5"].notna().any()
    expected = panel.daily["soxx_open"].shift(-6) / panel.daily["soxx_open"].shift(-1) - 1.0
    assert np.allclose(panel.daily["soxx_5d"], expected, equal_nan=True)
    assert np.allclose(panel.daily["soxx_abs_5d"], expected.abs(), equal_nan=True)
    assert set(panel.betas) == {"AAA", "BBB", "CCC", "DDD"}


def test_hidden_dispersion_features_do_not_change_before_a_future_ohlcv_perturbation():
    from src.analysis.soxx_hidden_dispersion import HIDDEN_DISPERSION_FEATURES, build_hidden_dispersion_panel

    constituents, index_frame = _synthetic_frames()
    baseline = build_hidden_dispersion_panel(
        constituents,
        index_frame,
        train=("2024-01-02", "2024-03-29"),
        lag_days=1,
        horizon=5,
    )
    changed = constituents.copy()
    final_date = changed["datetime"].max()
    mask = changed["datetime"].eq(final_date) & changed["instrument"].eq("AAA")
    changed.loc[mask, ["open", "high", "low", "adj_close", "volume"]] *= 10.0
    perturbed = build_hidden_dispersion_panel(
        changed,
        index_frame,
        train=("2024-01-02", "2024-03-29"),
        lag_days=1,
        horizon=5,
    )

    before_final = baseline.daily["datetime"].lt(final_date)
    pd.testing.assert_frame_equal(
        baseline.daily.loc[before_final, list(HIDDEN_DISPERSION_FEATURES)].reset_index(drop=True),
        perturbed.daily.loc[before_final, list(HIDDEN_DISPERSION_FEATURES)].reset_index(drop=True),
    )


def test_hidden_dispersion_degenerate_cross_sections_remain_finite():
    from src.analysis.soxx_hidden_dispersion import build_hidden_dispersion_panel

    constituents, index_frame = _synthetic_frames(rows=50)
    for column in ["open", "high", "low", "adj_close", "volume"]:
        template = constituents.loc[constituents["instrument"].eq("AAA"), column].to_numpy()
        constituents[column] = np.tile(template, 4)
    panel = build_hidden_dispersion_panel(
        constituents,
        index_frame,
        train=("2024-01-02", "2024-02-29"),
        lag_days=1,
        horizon=5,
    )

    for column in ["cancellation_ratio_5", "breadth_entropy_5", "correlation_breakdown_20"]:
        values = panel.daily[column].dropna()
        assert len(values) > 0
        assert np.isfinite(values).all()


def test_hidden_dispersion_panel_fails_closed_when_vix_has_no_usable_values():
    from src.analysis.soxx_hidden_dispersion import build_hidden_dispersion_panel

    constituents, index_frame = _synthetic_frames()
    index_frame["vix"] = np.nan

    with pytest.raises(ValueError, match="vix has no usable values"):
        build_hidden_dispersion_panel(
            constituents,
            index_frame,
            train=("2024-01-02", "2024-03-29"),
            lag_days=1,
            horizon=5,
        )


def test_hidden_dispersion_evaluation_uses_train_orientation_and_nonoverlap_offsets():
    from src.analysis.soxx_hidden_dispersion import (
        build_hidden_dispersion_panel,
        evaluate_hidden_dispersion,
    )

    constituents, index_frame = _synthetic_frames()
    panel = build_hidden_dispersion_panel(
        constituents,
        index_frame,
        train=("2024-01-02", "2024-03-29"),
        lag_days=1,
        horizon=5,
    )
    metrics = evaluate_hidden_dispersion(
        panel,
        segments={
            "train": ("2024-01-02", "2024-03-29"),
            "validation": ("2024-04-01", "2024-04-30"),
            "test": ("2024-05-01", "2024-05-20"),
        },
    )

    assert set(metrics) == {"feature_metrics", "nonoverlap", "candidate_comparison", "stability"}
    assert metrics["feature_metrics"]["train_direction"].notna().all()
    assert set(metrics["nonoverlap"]["offset"]) == set(range(5))
    assert metrics["nonoverlap"]["overlap_purged"].all()
    assert set(metrics["candidate_comparison"]["candidate"]) == {
        "weighted_residual_baseline",
        "hidden_dispersion_composite",
        "baseline_plus_interaction",
    }
    assert set(metrics["candidate_comparison"]["target"]) == {"soxx_5d", "soxl_5d", "soxx_abs_5d"}
    assert set(metrics["stability"]["segment_pair"]) == {"train_validation", "validation_test"}
