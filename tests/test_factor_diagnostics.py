from __future__ import annotations

import pandas as pd

from src.factor import (
    analyze_factor_suite,
    build_composite_factor,
    compute_factor_correlation,
    compute_factor_ic,
    compute_factor_quantile_returns,
)


def _time_series_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=8, freq="D"),
            "factor_good": [-3, -2, -1, 0, 1, 2, 3, 4],
            "factor_bad": [3, 2, 1, 0, -1, -2, -3, -4],
            "factor_noisy": [1, -1, 1, -1, 1, -1, 1, -1],
            "target_return": [-0.06, -0.04, -0.02, 0.0, 0.02, 0.04, 0.06, 0.08],
        }
    )


def _cross_section_frame() -> pd.DataFrame:
    rows = []
    for date in pd.date_range("2026-01-01", periods=4, freq="D"):
        rows.extend(
            [
                {"date": date, "symbol": "A", "factor_good": 1.0, "factor_bad": 3.0, "target_return": 0.01},
                {"date": date, "symbol": "B", "factor_good": 2.0, "factor_bad": 2.0, "target_return": 0.02},
                {"date": date, "symbol": "C", "factor_good": 3.0, "factor_bad": 1.0, "target_return": 0.03},
            ]
        )
    return pd.DataFrame(rows)


def test_compute_factor_ic_supports_time_series_and_cross_sectional_modes():
    ts_ic = compute_factor_ic(
        _time_series_frame(),
        factor_columns=["factor_good", "factor_bad"],
        target_column="target_return",
    )
    good = ts_ic.loc[ts_ic["factor"] == "factor_good"].iloc[0]
    bad = ts_ic.loc[ts_ic["factor"] == "factor_bad"].iloc[0]

    assert good["scope"] == "time_series"
    assert good["observation_count"] == 8
    assert float(good["ic_mean"]) > 0.99
    assert float(bad["ic_mean"]) < -0.99

    xs_ic = compute_factor_ic(
        _cross_section_frame(),
        factor_columns=["factor_good", "factor_bad"],
        target_column="target_return",
        entity_column="symbol",
    )
    xs_good = xs_ic.loc[xs_ic["factor"] == "factor_good"].iloc[0]
    assert xs_good["scope"] == "cross_sectional"
    assert xs_good["period_count"] == 4
    assert float(xs_good["ic_mean"]) > 0.99


def test_compute_factor_correlation_and_composite_factor_are_stable():
    frame = _time_series_frame()
    corr = compute_factor_correlation(frame, factor_columns=["factor_good", "factor_bad", "factor_noisy"])

    pair = corr.loc[
        (corr["factor_a"] == "factor_good") & (corr["factor_b"] == "factor_bad")
    ].iloc[0]
    assert float(pair["correlation"]) < -0.99
    assert corr["abs_correlation"].max() <= 1.0

    composite = build_composite_factor(
        frame,
        factor_columns=["factor_good", "factor_bad"],
        weights={"factor_good": 1.0, "factor_bad": -1.0},
        output_column="combo",
    )
    assert "combo" in composite.columns
    assert abs(float(composite["combo"].mean())) < 1e-12
    assert float(composite["combo"].corr(composite["target_return"], method="spearman")) > 0.99


def test_compute_factor_quantile_returns_orders_factor_layers():
    quantiles = compute_factor_quantile_returns(
        _time_series_frame(),
        factor_column="factor_good",
        target_column="target_return",
        quantiles=4,
    )

    assert {"factor", "quantile", "mean_return", "spread_vs_bottom"}.issubset(quantiles.columns)
    bottom = quantiles.loc[quantiles["quantile"] == 1].iloc[0]
    top = quantiles.loc[quantiles["quantile"] == 4].iloc[0]
    assert float(top["mean_return"]) > float(bottom["mean_return"])
    assert float(top["spread_vs_bottom"]) > 0.0


def test_analyze_factor_suite_outputs_ic_correlation_quantiles_and_chinese_summary():
    report = analyze_factor_suite(
        _time_series_frame(),
        factor_columns=["factor_good", "factor_bad", "factor_noisy"],
        target_column="target_return",
        quantiles=4,
    )

    assert report.ic.loc[report.ic["factor"] == "factor_good", "ic_mean"].iloc[0] > 0.99
    assert not report.correlation.empty
    assert not report.quantile_returns.empty
    assert report.summary["top_factor"] == "factor_good"
    assert report.summary["factor_count"] == 3
    assert "中文结论" in report.summary["review_text"]
