"""Exclude numerically unresolved excess from positive-evidence decisions.

Returns, point estimates and retained confidence bounds are never modified.
The resolution floor is in cumulative log-excess units for each tested window;
it is not a daily hurdle, a transaction cost or a new significance calculation.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from numbers import Real

import numpy as np
import pandas as pd

from src.analysis.benchmark_relative_protocol import COSTS_BPS, BOOTSTRAP_BLOCKS, _calendar, _checked_path
from src.analysis.integrated_allocation import DATES, paired_metrics, annual_paired_metrics


CHECKS = (
    "aggregate_positive_net_log_excess", "minimum_positive_signal_years",
    "median_annual_log_excess_positive", "leave_any_one_year_out_log_excess_positive",
    "anchored_max_drawdown_not_worse_than_soxx", "cvar5_not_worse_than_soxx",
    "minimum_rolling_positive_excess_rate", "stationary_block_20_lower_bound_positive",
    "stationary_block_60_lower_bound_positive",
)


def _require(condition, message):
    if not condition:
        raise ValueError("blocked: " + message)


def _same(actual, expected, name):
    """Exact deterministic recomputation; never infer identity from a rounded display."""
    try:
        pd.testing.assert_frame_equal(actual.reset_index(drop=True), expected.reset_index(drop=True),
                                      check_exact=True, check_dtype=False)
    except (AssertionError, TypeError, ValueError) as error:
        raise ValueError(f"blocked: evidence/path {name} statistics differ") from error


def _single(table, cost, name):
    rows = table.loc[table.cost_bps.eq(cost)]
    _require(len(rows) == 1, f"{name} needs exactly one row per cost")
    return rows.index[0], rows.iloc[0]


def apply_excess_resolution(evidence, paths_by_cost, benchmarks_by_cost, *, log_excess_floor=1e-8):
    """Return a deep copy with resolved-positive counts and gates.

    Input must be the unmodified result of ``evaluate_benchmark_relative``.
    Alignment checks recompute deterministic path summaries, but do not sample
    or refit. Retained bootstrap intervals are checked for matching sample
    sizes, observed means and units; their bounds are reused without resampling.
    """
    _require(isinstance(log_excess_floor, Real) and not isinstance(log_excess_floor, (bool, np.bool_))
             and np.isfinite(log_excess_floor) and log_excess_floor > 0,
             "log_excess_floor must be a positive finite numeric scalar")
    floor = float(log_excess_floor)
    _require(isinstance(evidence, Mapping) and "excess_resolution" not in evidence,
             "unmodified benchmark evidence required; resolution cannot be applied twice")
    tables = ("metrics", "annual", "rolling_summary", "bootstrap", "gates")
    for name in tables:
        table = evidence.get(name)
        _require(isinstance(table, pd.DataFrame) and not table.empty and not table.columns.has_duplicates
                 and table.index.is_unique and "cost_bps" in table,
                 f"complete unambiguous {name} table required")
        _require(set(table.cost_bps) == set(COSTS_BPS), f"{name} cost inventory differs")
    _require(isinstance(paths_by_cost, Mapping) and isinstance(benchmarks_by_cost, Mapping)
             and set(paths_by_cost) == set(COSTS_BPS) and set(benchmarks_by_cost) == set(COSTS_BPS)
             and tuple(evidence.get("costs_bps", ())) == COSTS_BPS,
             "exactly the two registered candidate/benchmark costs are required")
    years = tuple(evidence.get("expected_years", ()))
    _require(bool(years) and all(isinstance(year, (int, np.integer)) and not isinstance(year, (bool, np.bool_)) for year in years)
             and years == tuple(range(years[0], years[-1] + 1)), "invalid evidence year coverage")
    required = {
        "gates": {"all_pass", "positive_years", "years", "minimum_positive_years"},
        "rolling_summary": {"rolling_sessions", "minimum_positive_excess_rate", "rolling_windows", "positive_windows",
            "positive_excess_rate", "minimum_log_excess", "median_log_excess", "maximum_log_excess"},
        "bootstrap": {"mean_block", "sessions", "daily_mean_log_excess", "daily_lower_bound", "daily_upper_bound",
            "annualized_log_excess", "annualized_lower_bound", "annualized_upper_bound", "lower_bound_positive"},
    }
    for name, columns in required.items():
        _require(columns.issubset(evidence[name]), f"{name} schema is incomplete")
    _require(isinstance(evidence.get("historical_gate_pass"), (bool, np.bool_))
             and pd.api.types.is_bool_dtype(evidence["bootstrap"].lower_bound_positive),
             "original qualification and bootstrap signs must be booleans")
    _require(set(CHECKS).issubset(evidence["gates"]), "original benchmark gate schema missing")
    for column in (*CHECKS, "all_pass"):
        _require(column in evidence["gates"] and pd.api.types.is_bool_dtype(evidence["gates"][column]),
                 "original gate decisions must be complete booleans")

    reference_dates = None
    checked = {}
    for cost in COSTS_BPS:
        _require(isinstance(paths_by_cost[cost], pd.DataFrame) and "signal_date" in paths_by_cost[cost],
                 "candidate path requires dated DataFrame rows")
        signals = pd.DatetimeIndex(paths_by_cost[cost]["signal_date"])
        dates, _ = _calendar(years, signals)
        candidate = _checked_path(paths_by_cost[cost], cost=cost, years=years, expected_dates=dates)
        benchmark = _checked_path(benchmarks_by_cost[cost], cost=cost, years=years, expected_dates=dates)
        axes = candidate.loc[:, list(DATES)]
        _require(axes.equals(benchmark.loc[:, list(DATES)])
                 and (reference_dates is None or axes.equals(reference_dates)), "paired cost/date coverage differs")
        reference_dates = axes
        _require(evidence.get("expected_signal_sessions") == len(candidate), "evidence sample size differs")
        metric = paired_metrics(candidate, benchmark)
        annual = annual_paired_metrics(candidate, benchmark)
        _require(set(metric).issubset(evidence["metrics"]) and set(annual).issubset(evidence["annual"]),
                 "retained paired statistics schema is incomplete")
        metric_index, stored_metric = _single(evidence["metrics"], cost, "metrics")
        _same(evidence["metrics"].loc[[metric_index], list(metric)], pd.DataFrame([metric]), "overall")
        stored_annual = evidence["annual"].loc[evidence["annual"].cost_bps.eq(cost)]
        _same(stored_annual.loc[:, list(annual)], annual, "annual")
        roll_index, roll_row = _single(evidence["rolling_summary"], cost, "rolling summary")
        width = roll_row.rolling_sessions
        _require(isinstance(width, (int, np.integer)) and not isinstance(width, (bool, np.bool_))
                 and 2 <= width <= len(candidate), "invalid retained rolling window")
        threshold = roll_row.minimum_positive_excess_rate
        _require(isinstance(threshold, Real) and not isinstance(threshold, (bool, np.bool_))
                 and np.isfinite(threshold) and 0 < threshold <= 1,
                 "invalid retained rolling threshold")
        excess = np.log1p(candidate.strategy_return.to_numpy(float)) - np.log1p(benchmark.strategy_return.to_numpy(float))
        rolling = pd.Series(excess).rolling(int(width), min_periods=int(width)).sum().dropna()
        expected_rolling = {"rolling_windows": len(rolling), "positive_windows": int(rolling.gt(0).sum()),
            "positive_excess_rate": float(rolling.gt(0).mean()), "minimum_log_excess": float(rolling.min()),
            "median_log_excess": float(rolling.median()), "maximum_log_excess": float(rolling.max())}
        _same(evidence["rolling_summary"].loc[[roll_index], list(expected_rolling)], pd.DataFrame([expected_rolling]), "rolling")
        gate_index, gate = _single(evidence["gates"], cost, "gates")
        minimum_years = gate.minimum_positive_years
        _require(isinstance(minimum_years, (int, np.integer)) and not isinstance(minimum_years, (bool, np.bool_))
                 and 1 <= minimum_years <= len(years), "invalid retained positive-year threshold")
        values = annual.log_excess_sum.to_numpy(float)
        leave_one = values.sum() - values
        raw_checks = {
            "aggregate_positive_net_log_excess": metric["log_excess_sum"] > 0,
            "minimum_positive_signal_years": int((values > 0).sum()) >= minimum_years,
            "median_annual_log_excess_positive": np.median(values) > 0,
            "leave_any_one_year_out_log_excess_positive": len(years) > 1 and leave_one.min() > 0,
            "anchored_max_drawdown_not_worse_than_soxx": metric["max_drawdown_pct"] >= metric["benchmark_max_drawdown_pct"] - 1e-10,
            "cvar5_not_worse_than_soxx": metric["cvar5_pct"] >= metric["benchmark_cvar5_pct"] - 1e-10,
            "minimum_rolling_positive_excess_rate": rolling.gt(0).mean() >= threshold,
        }
        boots = evidence["bootstrap"].loc[evidence["bootstrap"].cost_bps.eq(cost)]
        _require(len(boots) == len(BOOTSTRAP_BLOCKS) and not boots.mean_block.duplicated().any()
                 and set(boots.mean_block) == set(BOOTSTRAP_BLOCKS), "bootstrap block inventory differs")
        for row in boots.itertuples():
            _require(row.sessions == len(candidate) and np.isfinite([row.daily_mean_log_excess,
                row.daily_lower_bound, row.daily_upper_bound, row.annualized_log_excess,
                row.annualized_lower_bound, row.annualized_upper_bound]).all(), "invalid retained bootstrap sample or bounds")
            _require(row.daily_mean_log_excess == float(excess.mean())
                     and row.annualized_log_excess == float(252 * excess.mean())
                     and row.annualized_lower_bound == 252 * row.daily_lower_bound
                     and row.annualized_upper_bound == 252 * row.daily_upper_bound
                     and row.daily_lower_bound <= row.daily_upper_bound
                     and bool(row.lower_bound_positive) == bool(row.daily_lower_bound > 0),
                     "bootstrap observed mean, original sign or units differ from paths")
            raw_checks[f"stationary_block_{int(row.mean_block)}_lower_bound_positive"] = row.daily_lower_bound > 0
        _require(all(bool(gate[key]) == bool(value) for key, value in raw_checks.items())
                 and gate.positive_years == int((values > 0).sum()) and gate.years == len(years)
                 and bool(gate.all_pass) == all(raw_checks.values()), "original gate statistics differ from paths")
        checked[cost] = (metric_index, stored_annual.index, roll_index, gate_index, boots.index,
                         metric, values, leave_one, rolling, threshold, int(minimum_years))
    _require(bool(evidence.get("historical_gate_pass")) == bool(evidence["gates"].all_pass.all()),
             "original overall gate differs from its cost gates")

    result = deepcopy(evidence)
    for name in ("metrics", "annual"):
        for column in ("raw_positive_excess", "resolved_positive_excess", "within_resolution"):
            result[name][column] = False
    for column in ("within_resolution_years", "raw_positive_leave_one_year_out_count",
                   "within_resolution_leave_one_year_out_count", "resolved_positive_leave_one_year_out_count"):
        result["gates"][column] = 0
    result["rolling_summary"]["within_resolution_windows"] = 0
    result["bootstrap"]["cumulative_lower_bound"] = 0.
    result["bootstrap"]["within_resolution_cumulative_lower_bound"] = False
    for name, columns in (("gates", (*CHECKS, "all_pass", "positive_years")),
                          ("rolling_summary", ("positive_windows", "positive_excess_rate")),
                          ("bootstrap", ("lower_bound_positive",))):
        for column in columns:
            result[name]["raw_" + column] = result[name][column].copy()
    for cost, item in checked.items():
        mi, ai, ri, gi, bi, metric, annual, leave_one, rolling, threshold, minimum_years = item
        gates = result["gates"]
        gates.loc[gi, "aggregate_positive_net_log_excess"] = bool(metric["log_excess_sum"] > floor)
        gates.loc[gi, "positive_years"] = int((annual > floor).sum())
        gates.loc[gi, "minimum_positive_signal_years"] = bool((annual > floor).sum() >= minimum_years)
        gates.loc[gi, "median_annual_log_excess_positive"] = bool(np.median(annual) > floor)
        gates.loc[gi, "leave_any_one_year_out_log_excess_positive"] = bool(len(annual) > 1 and leave_one.min() > floor)
        rate = float(rolling.gt(floor).mean())
        gates.loc[gi, "minimum_rolling_positive_excess_rate"] = bool(rate >= threshold)
        gates.loc[gi, "within_resolution_years"] = int((np.abs(annual) <= floor).sum())
        gates.loc[gi, "raw_positive_leave_one_year_out_count"] = int((leave_one > 0).sum())
        gates.loc[gi, "within_resolution_leave_one_year_out_count"] = int((np.abs(leave_one) <= floor).sum())
        gates.loc[gi, "resolved_positive_leave_one_year_out_count"] = int((leave_one > floor).sum())
        result["metrics"].loc[mi, "raw_positive_excess"] = bool(metric["log_excess_sum"] > 0)
        result["metrics"].loc[mi, "resolved_positive_excess"] = bool(metric["log_excess_sum"] > floor)
        result["metrics"].loc[mi, "within_resolution"] = bool(abs(metric["log_excess_sum"]) <= floor)
        result["annual"].loc[ai, "raw_positive_excess"] = annual > 0
        result["annual"].loc[ai, "resolved_positive_excess"] = annual > floor
        result["annual"].loc[ai, "within_resolution"] = np.abs(annual) <= floor
        result["rolling_summary"].loc[ri, "positive_windows"] = int(rolling.gt(floor).sum())
        result["rolling_summary"].loc[ri, "positive_excess_rate"] = rate
        result["rolling_summary"].loc[ri, "within_resolution_windows"] = int(rolling.abs().le(floor).sum())
        for index in bi:
            row = result["bootstrap"].loc[index]
            cumulative_lower = float(row.sessions * row.daily_lower_bound)
            _require(np.isfinite(cumulative_lower), "nonfinite cumulative retained bootstrap bound")
            passed = bool(cumulative_lower > floor)
            result["bootstrap"].loc[index, "lower_bound_positive"] = passed
            result["bootstrap"].loc[index, "cumulative_lower_bound"] = cumulative_lower
            result["bootstrap"].loc[index, "within_resolution_cumulative_lower_bound"] = bool(abs(cumulative_lower) <= floor)
            gates.loc[gi, f"stationary_block_{int(row.mean_block)}_lower_bound_positive"] = passed
        gates.loc[gi, "all_pass"] = bool(gates.loc[gi, list(CHECKS)].all())

    result["raw_historical_gate_pass"] = evidence["historical_gate_pass"]
    result["historical_gate_pass"] = bool(result["gates"].all_pass.all())
    result["promotion_eligible"] = result["new_forward_qualified"] = result["forward_evidence_verified"] = False
    result["decision"] = ("historical_gate_pass_requires_independent_forward_verification"
        if result["historical_gate_pass"] else "not_promoted_historical_gate_failed")
    result["excess_resolution"] = {"log_excess_floor": floor, "units": "cumulative log excess per tested window",
        "strict_positive_rule": "effect > floor", "within_resolution_rule": "abs(effect) <= floor",
        "bootstrap_rule": "sessions * retained daily_lower_bound > floor",
        "returns_modified": False, "point_estimates_modified": False, "confidence_bounds_modified": False,
        "new_bootstrap_draws": 0, "risk_gates_modified": False,
        "evidence_alignment": "Exact paired/annual/rolling recomputation; retained bootstrap sample sizes, observed means and units checked without regenerating bounds.",
        "scope": "Exclude numerically indistinguishable floating-point excess; no return adjustment or new alpha claim."}
    return result
