"""Deterministic, same-calendar historical qualification against SOXX.

The caller supplies Qlib-derived, independently reconciled portfolio paths.
This module evaluates those paths; it neither executes a strategy nor certifies
their provenance. Historical qualification cannot establish forward evidence or
authorize promotion, including when a caller labels the input as forward data.
"""
from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral, Real

import numpy as np
import pandas as pd

from src.analysis.integrated_allocation import (
    DATES,
    annual_paired_metrics,
    paired_metrics,
    stationary_bootstrap_excess,
    validate_replay,
)


COSTS_BPS = (10.0, 25.0)
BOOTSTRAP_BLOCKS = (20, 60)
QLIB_ANALYSIS_REFERENCE = "https://qlib.readthedocs.io/en/latest/component/report.html"


def _require(condition, message):
    if not condition:
        raise ValueError("blocked: " + message)


def _integer(value, *, name, minimum):
    _require(isinstance(value, Integral) and not isinstance(value, (bool, np.bool_))
             and value >= minimum, f"{name} must be an integer >= {minimum}")
    return int(value)


def _calendar(years, expected_signal_dates):
    """Use an explicit native calendar reference or read the active Qlib one.

    Explicit references must come from the caller's audited Qlib provider, not
    from the candidate path. Boundary checks reject partial calendar years even
    if the caller supplies the same truncated dates for all three inputs.
    """
    source = "caller_supplied_expected_signal_dates"
    if expected_signal_dates is None:
        from qlib.data import D

        source = "qlib.data.D.calendar"
        try:
            expected_signal_dates = D.calendar(
                start_time=f"{years[0]}-01-01", end_time=f"{years[-1]}-12-31", freq="day")
        except Exception as error:
            raise ValueError("blocked: Qlib calendar unavailable; supply an audited native calendar") from error
    try:
        dates = pd.DatetimeIndex(expected_signal_dates)
    except (TypeError, ValueError) as error:
        raise ValueError("blocked: invalid expected signal calendar") from error
    _require(len(dates) > 0 and dates.tz is None and not dates.hasnans
             and not dates.has_duplicates and dates.is_monotonic_increasing
             and dates.equals(dates.normalize()),
             "expected signal calendar must contain unique ordered daily sessions")
    _require(tuple(sorted(dates.year.unique())) == years,
             "expected calendar years differ from declared complete years")
    for year in years:
        annual = dates[dates.year == year]
        _require(annual[0] <= pd.Timestamp(year=year, month=1, day=7)
                 and annual[-1] >= pd.Timestamp(year=year, month=12, day=24)
                 and set(annual.month) == set(range(1, 13)),
                 f"partial expected calendar year {year}; full signal years are required")
    return dates, source


def _checked_path(frame, *, cost, years, expected_dates):
    _require(isinstance(frame, pd.DataFrame) and not frame.columns.has_duplicates,
             "portfolio path must be a DataFrame with unique columns")
    for column in ("strategy_return", "turnover", "cost_bps"):
        _require(column in frame and pd.api.types.is_numeric_dtype(frame[column])
                 and not pd.api.types.is_bool_dtype(frame[column]),
                 f"path requires numeric {column}")
    fee_columns = [name for name in ("fees", "fee") if name in frame]
    _require(bool(fee_columns), "path requires actual cash fees")
    for column in fee_columns:
        _require(pd.api.types.is_numeric_dtype(frame[column])
                 and not pd.api.types.is_bool_dtype(frame[column]), "cash fees must be numeric")
    path = validate_replay(frame, expected_years=years)
    _require(np.equal(path.cost_bps.to_numpy(), cost).all(),
             "path declared fee assumption differs from its cost scenario")
    signals = pd.DatetimeIndex(path.signal_date)
    _require(signals.equals(expected_dates),
             "path does not match the complete expected signal calendar; no missing sessions allowed")
    # Preserve every next-open interval, including held-state continuity.
    _require(np.array_equal(path.trade_date.to_numpy()[:-1], path.signal_date.to_numpy()[1:])
             and np.array_equal(path.exit_date.to_numpy()[:-2], path.signal_date.to_numpy()[2:]),
             "signal/trade/exit dates must follow consecutive expected sessions")
    return path


def evaluate_benchmark_relative(
    paths_by_cost: Mapping[float, pd.DataFrame],
    benchmarks_by_cost: Mapping[float, pd.DataFrame],
    *,
    expected_years,
    minimum_positive_years,
    rolling_sessions=252,
    minimum_rolling_win_rate=0.6,
    bootstrap_resamples=2000,
    family_size=1,
    evidence_kind="reused_history",
    costs_bps=COSTS_BPS,
    expected_signal_dates=None,
) -> dict:
    """Return historical evidence tables, never a production permission.

    Every deterministic candidate is evaluated at both 10 and 25 bp, against
    its matching independently funded benchmark. No artificial seed gate is
    applied. ``expected_signal_dates`` must be the full Qlib calendar for the
    declared signal years; omitted references use the active Qlib provider.
    Rolling windows are overlapping descriptive observations, not independent
    trials. The fixed family adjustment does not correct earlier human search.
    """
    try:
        years = tuple(expected_years)
        costs = tuple(costs_bps)
    except TypeError as error:
        raise ValueError("blocked: years and costs must be explicit sequences") from error
    _require(bool(years) and all(isinstance(year, Integral)
             and not isinstance(year, (bool, np.bool_)) and 1900 <= year <= 9998 for year in years),
             "expected_years must be explicit integer calendar years")
    years = tuple(int(year) for year in years)
    _require(years == tuple(range(years[0], years[-1] + 1)),
             "expected_years must be ordered, unique and contiguous")
    positive_years_required = _integer(minimum_positive_years, name="minimum_positive_years", minimum=1)
    _require(positive_years_required <= len(years), "positive-year requirement exceeds declared years")
    rolling_sessions = _integer(rolling_sessions, name="rolling_sessions", minimum=2)
    bootstrap_resamples = _integer(bootstrap_resamples, name="bootstrap_resamples", minimum=200)
    family_size = _integer(family_size, name="family_size", minimum=1)
    _require(isinstance(minimum_rolling_win_rate, Real)
             and not isinstance(minimum_rolling_win_rate, (bool, np.bool_))
             and np.isfinite(minimum_rolling_win_rate) and 0 < minimum_rolling_win_rate <= 1,
             "minimum_rolling_win_rate must be finite in (0, 1]")
    _require(costs == COSTS_BPS and all(isinstance(cost, Real)
             and not isinstance(cost, (bool, np.bool_)) for cost in costs),
             "registered costs must be exactly (10.0, 25.0) bps")
    _require(isinstance(paths_by_cost, Mapping) and isinstance(benchmarks_by_cost, Mapping)
             and set(paths_by_cost) == set(COSTS_BPS) and set(benchmarks_by_cost) == set(COSTS_BPS),
             "every candidate and benchmark must contain exactly both registered cost scenarios")
    _require(isinstance(evidence_kind, str) and bool(evidence_kind.strip()),
             "evidence_kind must be a nonempty descriptive label")
    expected_dates, calendar_source = _calendar(years, expected_signal_dates)
    _require(len(expected_dates) >= max(rolling_sessions, 2 * max(BOOTSTRAP_BLOCKS)),
             "insufficient complete observations for rolling and bootstrap analysis")

    # Validate the whole comparison before calculating any candidate statistic.
    checked = {}
    common_dates = None
    for cost in COSTS_BPS:
        candidate = _checked_path(paths_by_cost[cost], cost=cost, years=years, expected_dates=expected_dates)
        benchmark = _checked_path(benchmarks_by_cost[cost], cost=cost, years=years, expected_dates=expected_dates)
        dates = candidate.loc[:, list(DATES)]
        _require(dates.equals(benchmark.loc[:, list(DATES)]),
                 "candidate and benchmark must have identical signal/trade/exit windows")
        _require(common_dates is None or dates.equals(common_dates),
                 "cost scenarios must use identical signal/trade/exit windows")
        common_dates = dates
        checked[cost] = candidate, benchmark

    metric_rows, annual_tables, rolling_rows, bootstrap_rows, gate_rows = [], [], [], [], []
    for cost, (candidate, benchmark) in checked.items():
        metric = paired_metrics(candidate, benchmark)
        annual = annual_paired_metrics(candidate, benchmark)
        excess = np.log1p(candidate.strategy_return.to_numpy(float)) - np.log1p(benchmark.strategy_return.to_numpy(float))
        rolling = pd.Series(excess).rolling(rolling_sessions, min_periods=rolling_sessions).sum().dropna()
        _require(np.isfinite(rolling).all() and not rolling.empty, "invalid rolling excess statistics")
        win_rate = float(rolling.gt(0).mean())
        checks = {
            "aggregate_positive_net_log_excess": bool(metric["log_excess_sum"] > 0),
            "minimum_positive_signal_years": bool(annual.log_excess_sum.gt(0).sum() >= positive_years_required),
            "median_annual_log_excess_positive": bool(annual.log_excess_sum.median() > 0),
            "leave_any_one_year_out_log_excess_positive": bool(
                len(years) > 1 and (annual.log_excess_sum.sum() - annual.log_excess_sum).min() > 0),
            "anchored_max_drawdown_not_worse_than_soxx": bool(
                metric["max_drawdown_pct"] >= metric["benchmark_max_drawdown_pct"] - 1e-10),
            "cvar5_not_worse_than_soxx": bool(metric["cvar5_pct"] >= metric["benchmark_cvar5_pct"] - 1e-10),
            "minimum_rolling_positive_excess_rate": bool(win_rate >= minimum_rolling_win_rate),
        }
        for block in BOOTSTRAP_BLOCKS:
            stats = stationary_bootstrap_excess(candidate, benchmark, mean_block=block,
                resamples=bootstrap_resamples, alpha=0.05 / family_size, seed=1729)
            bootstrap_rows.append({"cost_bps": cost, "declared_family_size": family_size, **stats})
            checks[f"stationary_block_{block}_lower_bound_positive"] = bool(stats["lower_bound_positive"])
        metric_rows.append({"cost_bps": cost, **metric})
        annual_tables.append(annual.assign(cost_bps=cost))
        rolling_rows.append({
            "cost_bps": cost, "rolling_sessions": rolling_sessions, "rolling_windows": len(rolling),
            "positive_windows": int(rolling.gt(0).sum()), "positive_excess_rate": win_rate,
            "minimum_positive_excess_rate": float(minimum_rolling_win_rate),
            "minimum_log_excess": float(rolling.min()), "median_log_excess": float(rolling.median()),
            "maximum_log_excess": float(rolling.max()), "overlapping_windows": True,
            "independent_observations": False,
        })
        gate_rows.append({"cost_bps": cost, **checks, "all_pass": bool(all(checks.values())),
            "positive_years": int(annual.log_excess_sum.gt(0).sum()), "years": len(years),
            "minimum_positive_years": positive_years_required})

    gates = pd.DataFrame(gate_rows)
    passed = bool(gates.all_pass.all())
    return {
        "metrics": pd.DataFrame(metric_rows), "annual": pd.concat(annual_tables, ignore_index=True),
        "rolling_summary": pd.DataFrame(rolling_rows), "bootstrap": pd.DataFrame(bootstrap_rows),
        "gates": gates, "historical_gate_pass": passed,
        "promotion_eligible": False, "new_forward_qualified": False,
        "decision": "historical_gate_pass_requires_independent_forward_verification" if passed else "not_promoted_historical_gate_failed",
        "evidence_kind": evidence_kind, "evidence_kind_is_unverified_caller_label": True,
        "forward_evidence_verified": False,
        "calendar_source": calendar_source, "expected_signal_sessions": len(expected_dates),
        "expected_years": list(years), "costs_bps": list(COSTS_BPS),
        "seed_gate": "not_applicable_to_deterministic_candidates; no seed evidence manufactured",
        "qlib_analysis_reference": QLIB_ANALYSIS_REFERENCE,
        "limitations": [
            "Overlapping rolling windows are descriptive, not independent statistical samples.",
            "Bootstrap bounds are conditional on the retained path and do not include refit or adaptive research uncertainty.",
            "The declared family adjustment does not cover earlier human research or candidate selection.",
            "Calendar and portfolio provenance require upstream Qlib readback and native execution reconciliation.",
            "This evaluator cannot verify independent forward evidence; a caller label never enables promotion.",
        ],
    }
