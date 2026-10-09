"""The qualification contract uses synthetic paths, not market performance."""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from src.analysis import benchmark_relative_protocol as protocol


YEARS = tuple(range(2016, 2023))


@pytest.fixture
def paths():
    dates = pd.bdate_range("2016-01-04", "2022-12-30")
    baseline = np.sin(np.arange(len(dates))) * 0.004
    candidates, benchmarks = {}, {}
    for cost in (10.0, 25.0):
        reference = pd.DataFrame({
            "signal_date": dates, "trade_date": dates + pd.offsets.BDay(1),
            "exit_date": dates + pd.offsets.BDay(2), "strategy_return": baseline,
            "turnover": 0.0, "fees": 0.0, "cost_bps": cost,
        })
        benchmarks[cost] = reference
        candidates[cost] = reference.assign(strategy_return=np.expm1(np.log1p(baseline) + 0.0003))
    return candidates, benchmarks, dates


def evaluate(paths, **kwargs):
    candidates, benchmarks, dates = paths
    options = dict(expected_years=YEARS, minimum_positive_years=6,
                   bootstrap_resamples=200, expected_signal_dates=dates)
    options.update(kwargs)
    return protocol.evaluate_benchmark_relative(candidates, benchmarks, **options)


def test_complete_history_passes_statistics_but_never_certifies_forward_or_promotion(paths):
    original = deepcopy(paths)
    result = evaluate(paths, family_size=4, evidence_kind="independent_forward_verified")
    assert result["historical_gate_pass"] is True
    assert result["promotion_eligible"] is False
    assert result["new_forward_qualified"] is False
    assert result["forward_evidence_verified"] is False
    assert result["evidence_kind_is_unverified_caller_label"] is True
    assert "requires_independent_forward" in result["decision"]
    assert not any("seed" in column for column in result["gates"].columns)
    assert result["gates"].positive_years.eq(7).all()
    assert set(result["annual"].year) == set(YEARS)
    assert set(result["bootstrap"].mean_block) == {20, 60}
    assert result["bootstrap"].alpha.eq(0.05 / 4).all()
    assert result["rolling_summary"].positive_excess_rate.eq(1.0).all()
    assert result["rolling_summary"].overlapping_windows.all()
    assert not result["rolling_summary"].independent_observations.any()
    for cost in (10.0, 25.0):
        pd.testing.assert_frame_equal(paths[0][cost], original[0][cost])
        pd.testing.assert_frame_equal(paths[1][cost], original[1][cost])


def test_exact_benchmark_null_is_valid_but_cannot_pass_excess_gates(paths):
    _, benchmarks, dates = paths
    result = evaluate((deepcopy(benchmarks), benchmarks, dates))
    assert not result["historical_gate_pass"]
    assert result["gates"].anchored_max_drawdown_not_worse_than_soxx.all()
    assert result["gates"].cvar5_not_worse_than_soxx.all()
    assert not result["gates"].aggregate_positive_net_log_excess.any()
    assert not result["bootstrap"].lower_bound_positive.any()
    assert result["rolling_summary"].positive_excess_rate.eq(0.0).all()
    assert result["decision"] == "not_promoted_historical_gate_failed"


def test_six_of_seven_years_cannot_be_replaced_by_positive_aggregate(paths):
    candidates, benchmarks, dates = paths
    excess = np.where(np.isin(dates.year, [2019, 2020]), -0.00001, 0.0003)
    for cost in candidates:
        candidates[cost]["strategy_return"] = np.expm1(np.log1p(benchmarks[cost].strategy_return) + excess)
    result = evaluate(paths)
    assert result["gates"].aggregate_positive_net_log_excess.all()
    assert result["gates"].positive_years.eq(5).all()
    assert not result["gates"].minimum_positive_signal_years.any()
    assert not result["historical_gate_pass"]


def test_rolling_statistics_count_overlapping_complete_windows_only(paths):
    candidates, benchmarks, dates = paths
    excess = np.where(dates.year < 2020, 0.0003, -0.0002)
    for cost in candidates:
        candidates[cost]["strategy_return"] = np.expm1(np.log1p(benchmarks[cost].strategy_return) + excess)
    result = evaluate(paths, minimum_rolling_win_rate=0.9)
    expected = np.convolve(excess, np.ones(252), mode="valid")
    for row in result["rolling_summary"].itertuples():
        assert row.rolling_windows == len(dates) - 252 + 1
        assert row.positive_excess_rate == pytest.approx(np.mean(expected > 0))
        assert row.minimum_log_excess == pytest.approx(expected.min())
    assert not result["gates"].minimum_rolling_positive_excess_rate.any()


def test_return_and_risk_requirements_are_separate(paths):
    candidates, benchmarks, _ = paths
    for cost in candidates:
        candidates[cost]["strategy_return"] = benchmarks[cost].strategy_return * 3 + 0.001
    result = evaluate(paths)
    assert result["gates"].aggregate_positive_net_log_excess.all()
    assert not result["gates"].cvar5_not_worse_than_soxx.any()
    assert not result["historical_gate_pass"]


@pytest.mark.parametrize("which", ["candidate", "benchmark"])
def test_both_costs_are_required_for_both_books(paths, which):
    candidates, benchmarks, _ = paths
    (candidates if which == "candidate" else benchmarks).pop(25.0)
    with pytest.raises(ValueError, match="exactly both registered cost"):
        evaluate(paths)


@pytest.mark.parametrize("corruption", ["missing_cost", "wrong_cost", "missing_fee", "bad_fee", "nan_return", "string_return", "bool_return", "duplicate_column"])
def test_invalid_accounting_and_nonnumeric_data_fail_closed(paths, corruption):
    candidates, _, _ = paths
    path = candidates[10.0]
    if corruption == "missing_cost":
        candidates[10.0] = path.drop(columns="cost_bps")
    elif corruption == "wrong_cost":
        path["cost_bps"] = 25.0
    elif corruption == "missing_fee":
        candidates[10.0] = path.drop(columns="fees")
    elif corruption == "bad_fee":
        path.loc[0, "fees"] = 0.01
    elif corruption == "nan_return":
        path.loc[0, "strategy_return"] = np.nan
    elif corruption == "string_return":
        path["strategy_return"] = path.strategy_return.astype(str)
    elif corruption == "bool_return":
        path["strategy_return"] = True
    else:
        candidates[10.0] = pd.concat([path, path[["fees"]]], axis=1)
    with pytest.raises(ValueError, match="blocked"):
        evaluate(paths)


def test_matching_partial_year_books_are_rejected_against_calendar(paths):
    candidates, benchmarks, dates = paths
    for books in (candidates, benchmarks):
        for cost, path in books.items():
            books[cost] = path.loc[path.signal_date < "2022-12-01"]
    with pytest.raises(ValueError, match="complete expected signal calendar"):
        evaluate(paths)
    # A truncated calendar supplied alongside truncated books is not a full year.
    with pytest.raises(ValueError, match="partial expected calendar year"):
        evaluate((candidates, benchmarks, dates[dates < "2022-12-01"]))


def test_shared_missing_interior_session_cannot_be_hidden_by_relinking_dates(paths):
    candidates, benchmarks, dates = paths
    for books in (candidates, benchmarks):
        for cost, path in books.items():
            altered = path.drop(index=20).reset_index(drop=True)
            # Malformed books can claim continuous settlement over a missing day.
            altered.loc[18, "exit_date"] = altered.loc[20, "signal_date"]
            altered.loc[19, "trade_date"] = altered.loc[20, "signal_date"]
            altered.loc[19, "exit_date"] = altered.loc[21, "signal_date"]
            books[cost] = altered
    with pytest.raises(ValueError, match="complete expected signal calendar"):
        evaluate((candidates, benchmarks, dates))


def test_terminal_exit_must_match_across_cost_scenarios(paths):
    candidates, benchmarks, _ = paths
    for books in (candidates, benchmarks):
        books[25.0].loc[len(books[25.0]) - 1, "exit_date"] += pd.offsets.BDay(1)
    with pytest.raises(ValueError, match="cost scenarios must use identical"):
        evaluate(paths)


@pytest.mark.parametrize("options", [
    {"costs_bps": (10.0,)}, {"costs_bps": (25.0, 10.0)},
    {"expected_years": (2016, 2018)}, {"expected_years": ()},
    {"expected_years": (True,)}, {"minimum_positive_years": 8},
    {"minimum_positive_years": True}, {"rolling_sessions": 1},
    {"rolling_sessions": 2.5}, {"rolling_sessions": 10000},
    {"minimum_rolling_win_rate": float("nan")}, {"minimum_rolling_win_rate": 1.01},
    {"minimum_rolling_win_rate": True}, {"bootstrap_resamples": 199},
    {"bootstrap_resamples": 200.5}, {"family_size": 0},
    {"family_size": True}, {"evidence_kind": ""},
])
def test_invalid_protocol_parameters_fail_closed(paths, options):
    with pytest.raises(ValueError, match="blocked"):
        evaluate(paths, **options)


def test_default_calendar_calls_qlib_and_fails_closed_if_unavailable(paths, monkeypatch):
    from qlib.data import D

    def unavailable(**kwargs):
        assert kwargs == {"start_time": "2016-01-01", "end_time": "2022-12-31", "freq": "day"}
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(D, "calendar", unavailable, raising=False)
    with pytest.raises(ValueError, match="Qlib calendar unavailable"):
        evaluate(paths, expected_signal_dates=None)


def test_default_calendar_uses_exact_native_readback(paths, monkeypatch):
    from qlib.data import D

    monkeypatch.setattr(D, "calendar", lambda **kwargs: paths[2], raising=False)
    result = evaluate(paths, expected_signal_dates=None)
    assert result["calendar_source"] == "qlib.data.D.calendar"
    assert result["historical_gate_pass"]
