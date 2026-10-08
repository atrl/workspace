"""Attribution contracts on synthetic frozen forecasts; no financial fits."""
import numpy as np
import pandas as pd
import pytest

from src.analysis.strategy_bottlenecks import analyze_forecast_increment


def _predictions():
    dates = pd.to_datetime(["2020-12-29", "2020-12-30", "2020-12-31", "2021-01-04", "2021-01-05"])
    y = np.array([.01, -.03, .01, .02, -.02])
    base = np.array([.001, -.002, .003, .004, .005])
    full = np.array([.003, -.001, .003, .009, .001])
    return pd.DataFrame({**{f"{head}{suffix}": value * multiplier
        for head, multiplier in (("cash5", 1), ("leverage5", 2))
        for suffix, value in (("", y), ("_base", base), ("_base_plus_earnings", full))}}, index=dates)


def test_mse_identity_annual_sse_contributions_and_input_preservation():
    frame = _predictions(); saved = frame.copy(deep=True)
    result = analyze_forecast_increment(frame)
    pd.testing.assert_frame_equal(frame, saved, check_exact=True)
    losses = result["mse_attribution"]
    assert len(losses) == 6 and len(result["event_calibration"]) == 24
    for head in ("cash5", "leverage5"):
        table = losses.loc[losses["head"].eq(head)].set_index("slice")
        target = frame[head].to_numpy(); base = frame[f"{head}_base"].to_numpy()
        full = frame[f"{head}_base_plus_earnings"].to_numpy()
        direct_sse = sum((f-y)**2 - (b-y)**2 for y, b, f in zip(target, base, full))
        assert table.loc["pooled", "SSE_difference"] == pytest.approx(direct_sse, abs=1e-18)
        annual = table.drop(index="pooled")
        assert annual["rows"].sum() == len(frame)
        assert annual["SSE_difference"].sum() == pytest.approx(direct_sse, abs=1e-18)
        assert annual["SSE_difference_fraction_of_pooled"].sum() == pytest.approx(1.)
        assert table["identity_within_roundoff"].all()
        np.testing.assert_allclose(table.extra_forecast_energy - table.twice_residual_alignment,
                                   table.full_minus_base_MSE, atol=1e-18, rtol=1e-12)
        np.testing.assert_allclose(table.increment_variance + table.increment_mean_square,
                                   table.extra_forecast_energy, atol=1e-18, rtol=1e-12)
        np.testing.assert_allclose(table.twice_increment_residual_covariance + table.twice_increment_residual_mean_product,
                                   table.twice_residual_alignment, atol=1e-18, rtol=1e-12)


def test_tied_rank_correlations_equal_independent_pandas_ranks():
    frame = _predictions()
    table = analyze_forecast_increment(frame)["mse_attribution"].set_index(["head", "slice"])
    for head in ("cash5", "leverage5"):
        for family, column in (("base", "base_ic"), ("base_plus_earnings", "full_ic")):
            expected = frame[head].rank(method="average").corr(frame[f"{head}_{family}"].rank(method="average"))
            assert table.loc[(head, "pooled"), column] == pytest.approx(expected)


def test_constant_increment_and_zero_pooled_difference_remain_undefined():
    frame = _predictions()
    frame["cash5_base_plus_earnings"] = frame.cash5_base
    result = analyze_forecast_increment(frame, target_heads=("cash5",))
    table = result["mse_attribution"]
    assert table.increment_residual_correlation.isna().all()
    assert table.increment_residual_correlation_undefined_reason.eq("constant_prediction").all()
    assert table.SSE_difference.eq(0).all()
    assert table.SSE_difference_fraction_of_pooled.isna().all()
    assert table.SSE_fraction_undefined_reason.eq("zero_pooled_SSE_difference").all()
    frame["cash5"] = .01
    constant = analyze_forecast_increment(frame, ("cash5",))["mse_attribution"]
    assert constant.base_ic.isna().all() and constant.full_ic.isna().all()
    assert constant.base_ic_undefined_reason.eq("constant_target").all()


@pytest.mark.parametrize("head,multiple", [("cash5", 1), ("leverage5", 2)])
@pytest.mark.parametrize("cost", [10., 25.])
def test_fee_events_strict_threshold_and_forecast_realized_gap(head, multiple, cost):
    frame = _predictions()
    c = cost / 10000.
    hurdle = multiple * (np.log1p(c) - np.log1p(-c))
    forecasts = np.array([np.nextafter(hurdle, -np.inf), hurdle, np.nextafter(hurdle, np.inf), .02, -.01])
    frame[f"{head}_base_plus_earnings"] = forecasts
    events = analyze_forecast_increment(frame, (head,))["event_calibration"]
    row = events.loc[events["slice"].eq("pooled") & events.cost_bps.eq(cost) & events.family.eq("base_plus_earnings")].iloc[0]
    selected_y = frame[head].to_numpy()[[2, 3]]
    assert row.events == 2 and row.coverage == 2/5
    assert row.forecast_mean_net_bp == pytest.approx(10000 * (forecasts[[2, 3]] - hurdle).mean())
    assert row.realized_mean_net_bp == pytest.approx(10000 * (selected_y - hurdle).mean())
    assert row.overestimation_bp == pytest.approx(row.forecast_mean_net_bp - row.realized_mean_net_bp)
    assert row.positive_event_fraction == pytest.approx((selected_y > hurdle).mean())


def test_empty_events_have_nan_conditional_means_and_single_row_correlation_is_undefined():
    frame = _predictions().iloc[:1].copy()
    frame.loc[:, ["cash5_base", "cash5_base_plus_earnings"]] = 0.
    result = analyze_forecast_increment(frame, ("cash5",))
    events = result["event_calibration"]
    assert events.events.eq(0).all() and events.coverage.eq(0).all()
    assert events[["forecast_mean_net_bp", "realized_mean_net_bp", "overestimation_bp", "positive_event_fraction"]].isna().all().all()
    assert events.conditional_mean_undefined_reason.eq("no_events").all()
    assert result["mse_attribution"].base_ic_undefined_reason.eq("insufficient_observations").all()


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_rows_fail_without_silent_deletion(bad):
    frame = _predictions(); frame.iloc[2, 0] = bad
    with pytest.raises(ValueError, match="all frozen prediction rows must be finite"):
        analyze_forecast_increment(frame)


@pytest.mark.parametrize("case", ["duplicate", "unsorted", "timezone", "nat", "empty", "missing", "duplicate_column"])
def test_invalid_axes_and_schema_fail_closed(case):
    frame = _predictions()
    if case == "duplicate": frame.index = [frame.index[0]] * len(frame)
    elif case == "unsorted": frame = frame.iloc[::-1]
    elif case == "timezone": frame.index = frame.index.tz_localize("UTC")
    elif case == "nat": frame.index = pd.DatetimeIndex([pd.NaT, *frame.index[1:]])
    elif case == "empty": frame = frame.iloc[:0]
    elif case == "missing": frame = frame.drop(columns="cash5_base")
    else: frame = pd.concat([frame, frame[["cash5"]]], axis=1)
    with pytest.raises(ValueError, match="blocked:"):
        analyze_forecast_increment(frame)


@pytest.mark.parametrize("heads", [(), ("unknown",), ("cash5", "cash5"), "cash5"])
def test_heads_must_have_known_unambiguous_event_semantics(heads):
    with pytest.raises(ValueError, match="blocked:"):
        analyze_forecast_increment(_predictions(), heads)


def test_finite_inputs_that_overflow_loss_arithmetic_fail_closed():
    frame = _predictions(); frame["cash5"] = 1e308
    with pytest.raises(ValueError, match="nonfinite forecast attribution arithmetic"):
        analyze_forecast_increment(frame, ("cash5",))
