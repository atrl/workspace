"""Paired, cost-after evidence for the continuous SOXL/SOXX/cash allocator.

These historical qualification statistics do not turn previously inspected
years into a fresh holdout. A stationary bootstrap preserves sampled serial
dependence under its assumptions; it does not correct adaptive research or
prove future performance. Qlib owns native execution and risk analysis.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd


DATES = ("signal_date", "trade_date", "exit_date")
ASSETS = ("soxl", "soxx", "cash")
STATIONARY_BOOTSTRAP_REFERENCE = "https://www.tandfonline.com/doi/abs/10.1080/01621459.1994.10476870"


@dataclass(frozen=True)
class StabilityGateSpec:
    """Preregister in the notebook; do not tune this contract to observed P/L."""

    expected_years: tuple[int, ...] = tuple(range(2016, 2023))
    minimum_positive_years: int = 5
    costs_bps: tuple[float, ...] = (10.0, 25.0)
    mean_block_lengths: tuple[int, ...] = (20, 60)
    bootstrap_resamples: int = 2000
    alpha: float = 0.05
    random_seed: int = 20261002
    family_size: int = 3
    expected_seeds: tuple[int, ...] = (17, 41, 71)
    minimum_positive_seeds: int = 2
    maximum_positive_year_share: float | None = None

    def __post_init__(self):
        if (not self.expected_years or tuple(sorted(set(self.expected_years))) != self.expected_years
                or not 1 <= self.minimum_positive_years <= len(self.expected_years)
                or not self.costs_bps or len(set(self.costs_bps)) != len(self.costs_bps)
                or not np.isfinite(self.costs_bps).all() or min(self.costs_bps) < 0
                or not self.mean_block_lengths or min(self.mean_block_lengths) < 1
                or len(set(self.mean_block_lengths)) != len(self.mean_block_lengths)
                or self.bootstrap_resamples < 200 or not 0 < self.alpha < .5
                or self.family_size < 1 or len(set(self.expected_seeds)) != len(self.expected_seeds)
                or not 1 <= self.minimum_positive_seeds <= len(self.expected_seeds)):
            raise ValueError("blocked: invalid preregistered stability gates")
        if self.maximum_positive_year_share is not None and not 0 < self.maximum_positive_year_share <= 1:
            raise ValueError("blocked: invalid concentration gate")


def _dates(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or set(DATES) - set(frame):
        raise ValueError("blocked: replay requires nonempty signal/trade/exit dates")
    result = frame.copy().reset_index(drop=True)
    for name in DATES:
        result[name] = pd.to_datetime(result[name])
        if result[name].isna().any() or result[name].dt.tz is not None:
            raise ValueError("blocked: daily replay requires finite timezone-naive session dates")
        if not result[name].eq(result[name].dt.normalize()).all():
            raise ValueError("blocked: daily replay dates must identify sessions, not intraday events")
    if (result.signal_date.duplicated().any() or result.trade_date.duplicated().any()
            or not result.signal_date.is_monotonic_increasing or not result.trade_date.is_monotonic_increasing
            or not result.signal_date.lt(result.trade_date).all() or not result.trade_date.lt(result.exit_date).all()):
        raise ValueError("blocked: duplicate, unordered, or noncausal replay dates")
    if len(result) > 1 and not result.exit_date.iloc[:-1].reset_index(drop=True).equals(result.trade_date.iloc[1:].reset_index(drop=True)):
        raise ValueError("blocked: noncontiguous replay sessions; missing returns cannot be silently dropped")
    return result


def validate_replay(frame: pd.DataFrame, *, expected_years=None) -> pd.DataFrame:
    result = _dates(frame)
    if "fees" not in result and "fee" in result:
        result["fees"] = result["fee"]
    elif "fees" in result and "fee" in result and not np.allclose(result.fees, result.fee, rtol=0, atol=1e-12):
        raise ValueError("blocked: fee aliases disagree")
    required = ("strategy_return", "turnover", "fees")
    if set(required) - set(result):
        raise ValueError("blocked: replay lacks net return, actual turnover, or cash fees")
    numeric = result.loc[:, list(required)].to_numpy(dtype=float)
    if (not np.isfinite(numeric).all() or (result.strategy_return <= -1).any()
            or (result.turnover < -1e-12).any() or (result.fees < -1e-12).any()):
        raise ValueError("blocked: invalid replay returns or accounting")
    if expected_years is not None and tuple(sorted(result.signal_date.dt.year.unique())) != tuple(expected_years):
        raise ValueError("blocked: replay annual coverage differs from preregistered signal years")
    if "cost_bps" in result:
        costs = result.cost_bps.to_numpy(dtype=float)
        if not np.isfinite(costs).all() or (costs < 0).any():
            raise ValueError("blocked: invalid replay cost assumptions")
        if not np.allclose(result.fees, result.turnover * costs / 10_000., rtol=1e-7, atol=1e-10):
            raise ValueError("blocked: replay cash fees do not match declared proportional costs")
    return result


def _paired(candidate: pd.DataFrame, benchmark: pd.DataFrame, *, expected_years=None):
    candidate = validate_replay(candidate, expected_years=expected_years)
    benchmark = validate_replay(benchmark, expected_years=expected_years)
    if not candidate.loc[:, list(DATES)].equals(benchmark.loc[:, list(DATES)]):
        raise ValueError("blocked: candidate and benchmark must have identical coverage; no inner join allowed")
    return candidate, benchmark


def _drawdown(log_returns) -> float:
    log_nav = np.r_[0., np.cumsum(np.asarray(log_returns, dtype=float))]
    return float(np.expm1((log_nav - np.maximum.accumulate(log_nav)).min()))


def _cvar(values, tail_fraction=.05) -> float:
    """Empirical lower expected shortfall with exactly the stated tail mass.

    Using every observation below an interpolated quantile overweights ties;
    a mostly-cash path can otherwise dilute one loss across all zero days.
    """
    ordered = np.sort(np.asarray(values, dtype=float))
    mass = len(ordered) * tail_fraction
    whole = int(np.floor(mass))
    fraction = mass - whole
    return float((ordered[:whole].sum() + fraction * ordered[whole]) / mass)


def _path_metrics(replay: pd.DataFrame) -> dict:
    from qlib.contrib.evaluate import risk_analysis

    values = replay.strategy_return.to_numpy(dtype=float)
    logs = np.log1p(values)
    # Qlib defaults to 238 daily observations. Specify the US convention and
    # retain its raw drawdown alongside an initial-capital-anchored supplement.
    native = risk_analysis(pd.Series(values, index=replay.trade_date), N=252, freq=None, mode="product")
    terminal = float(np.exp(logs.sum()))
    if not np.isfinite(terminal) or terminal <= 0 or not np.isfinite(native.loc["annualized_return", "risk"]):
        raise ValueError("blocked: portfolio wealth is outside the finite numerical range")
    worst5 = float(np.expm1(pd.Series(logs).rolling(5).sum().min())) if len(values) >= 5 else None
    return {
        "sessions": len(values), "start_signal": str(replay.signal_date.iloc[0].date()),
        "last_exit": str(replay.exit_date.iloc[-1].date()),
        "terminal_wealth": terminal, "return_pct": float(100 * np.expm1(logs.sum())),
        "cagr_pct": float(100 * native.loc["annualized_return", "risk"]),
        "annualized_log_volatility_pct": float(100 * np.std(logs, ddof=1) * np.sqrt(252)) if len(values) > 1 else None,
        "max_drawdown_pct": 100 * _drawdown(logs),
        "qlib_max_drawdown_without_initial_mark_pct": float(100 * native.loc["max_drawdown", "risk"]),
        "worst_day_pct": float(100 * values.min()), "cvar5_pct": 100 * _cvar(values),
        "worst_5d_pct": 100 * worst5 if worst5 is not None else None,
        "turnover": float(replay.turnover.sum()), "trade_days": int((replay.turnover > 1e-12).sum()),
        "sum_session_fee_rates": float(replay.fees.sum()),
    }


def paired_metrics(candidate: pd.DataFrame, benchmark: pd.DataFrame) -> dict:
    candidate, benchmark = _paired(candidate, benchmark)
    log_excess = np.log1p(candidate.strategy_return.to_numpy()) - np.log1p(benchmark.strategy_return.to_numpy())
    candidate_metrics, benchmark_metrics = _path_metrics(candidate), _path_metrics(benchmark)
    if not np.isfinite(np.exp([log_excess.sum(), 252 * log_excess.mean()])).all():
        raise ValueError("blocked: relative wealth is outside the finite numerical range")
    down = benchmark.strategy_return.to_numpy() < 0
    up = benchmark.strategy_return.to_numpy() > 0
    def conditional_excess(mask):
        return float(100 * np.expm1(log_excess[mask].sum())) if mask.any() else None
    return {
        **candidate_metrics, "benchmark_return_pct": benchmark_metrics["return_pct"],
        "benchmark_cagr_pct": benchmark_metrics["cagr_pct"],
        "benchmark_max_drawdown_pct": benchmark_metrics["max_drawdown_pct"],
        "benchmark_cvar5_pct": benchmark_metrics["cvar5_pct"],
        "benchmark_worst_day_pct": benchmark_metrics["worst_day_pct"],
        "benchmark_worst_5d_pct": benchmark_metrics["worst_5d_pct"],
        "log_excess_sum": float(log_excess.sum()),
        "relative_wealth": float(np.exp(log_excess.sum())),
        "relative_return_pct": float(100 * np.expm1(log_excess.sum())),
        "annualized_log_excess": float(252 * log_excess.mean()),
        "annualized_relative_return_pct": float(100 * np.expm1(252 * log_excess.mean())),
        "relative_max_drawdown_pct": 100 * _drawdown(log_excess),
        "up_day_conditional_relative_return_pct": conditional_excess(up),
        "down_day_conditional_relative_return_pct": conditional_excess(down),
        "conditional_returns_are_noncontiguous_statistics": True,
        "risk_policy": "reported against benchmark; no implicit fixed drawdown or exposure limit",
        "annualization_sessions": 252,
    }


def annual_paired_metrics(candidate: pd.DataFrame, benchmark: pd.DataFrame) -> pd.DataFrame:
    candidate, benchmark = _paired(candidate, benchmark)
    rows = []
    for year in sorted(candidate.signal_date.dt.year.unique()):
        mask = candidate.signal_date.dt.year.eq(year)
        rows.append({"year": int(year), "year_basis": "signal_year_fold_attribution",
                     **paired_metrics(candidate.loc[mask], benchmark.loc[mask])})
    return pd.DataFrame(rows)


def stationary_bootstrap_excess(candidate: pd.DataFrame, benchmark: pd.DataFrame, *, mean_block: int = 20,
                                resamples: int = 5000, alpha: float = .05, seed: int = 1729) -> dict:
    """One-sided percentile bound for paired mean log excess, geometric blocks.

    Indices continue circularly with probability 1-1/L, otherwise restart at a
    uniform observation. The candidate and benchmark use the same index path
    by resampling their paired difference. Seed repetitions are not extra days.
    """
    candidate, benchmark = _paired(candidate, benchmark)
    values = np.log1p(candidate.strategy_return.to_numpy()) - np.log1p(benchmark.strategy_return.to_numpy())
    count = len(values)
    if not isinstance(mean_block, (int, np.integer)) or mean_block < 1 or count < 2 * mean_block:
        raise ValueError("blocked: insufficient contiguous observations for the declared bootstrap block")
    if resamples < 200 or not 0 < alpha < .5:
        raise ValueError("blocked: invalid bootstrap precision or confidence")
    random = np.random.default_rng(seed)
    index = random.integers(count, size=resamples)
    totals = values[index].copy()
    for _ in range(1, count):
        restart = random.random(resamples) < 1. / mean_block
        index = (index + 1) % count
        index[restart] = random.integers(count, size=int(restart.sum()))
        totals += values[index]
    means = totals / count
    lower, upper = float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha))
    return {
        "method": "paired_stationary_bootstrap_percentile_mean_log_excess", "sessions": count,
        "mean_block": int(mean_block), "resamples": int(resamples), "random_seed": int(seed),
        "one_sided_confidence": 1 - alpha, "alpha": float(alpha),
        "daily_mean_log_excess": float(values.mean()), "daily_lower_bound": lower, "daily_upper_bound": upper,
        "annualized_log_excess": float(252 * values.mean()), "annualized_lower_bound": 252 * lower,
        "annualized_upper_bound": 252 * upper, "lower_bound_positive": lower > 0,
        "bootstrap_nonpositive_fraction_not_pvalue": float((means <= 0).mean()),
        "reference": STATIONARY_BOOTSTRAP_REFERENCE,
        "limitation": "Conditional on this reused path and weak stationarity; excludes model refit/selection uncertainty and is not new forward evidence.",
    }


def stable_excess_gates(paths_by_cost: Mapping[float, pd.DataFrame], benchmarks_by_cost: Mapping[float, pd.DataFrame],
                        *, spec: StabilityGateSpec = StabilityGateSpec(), seed_table: pd.DataFrame | None = None) -> dict:
    if set(paths_by_cost) != set(spec.costs_bps) or set(benchmarks_by_cost) != set(spec.costs_bps):
        raise ValueError("blocked: every preregistered cost scenario must have its own paired replay")
    # The same seeds must be positive at BOTH costs. Two different positive
    # seeds at each cost do not establish two reproducible positive fits.
    seed_counts = dict.fromkeys(spec.costs_bps, 0)
    seed_all_costs = 0
    if seed_table is not None:
        required = {"seed", "cost_bps", "log_excess_sum"}
        if not required.issubset(seed_table):
            raise ValueError("blocked: seed evidence requires seed, cost_bps, log_excess_sum")
        seeds = seed_table.loc[:, list(required)].copy()
        seeds["seed"] = seeds.seed.astype(str)
        if seeds.duplicated(["seed", "cost_bps"]).any() or not np.isfinite(seeds[["cost_bps", "log_excess_sum"]].to_numpy(dtype=float)).all():
            raise ValueError("blocked: duplicate or nonfinite seed evidence")
        expected = {(str(seed), cost) for seed in spec.expected_seeds for cost in spec.costs_bps}
        if set(zip(seeds.seed, seeds.cost_bps)) != expected:
            raise ValueError("blocked: incomplete declared seed-by-cost evidence; selecting seeds is prohibited")
        by_seed = seeds.pivot(index="seed", columns="cost_bps", values="log_excess_sum")
        seed_all_costs = int(by_seed.gt(0).all(axis=1).sum())
        seed_counts = {cost: int(by_seed[cost].gt(0).sum()) for cost in spec.costs_bps}
    metrics, annuals, bootstrap, rows = [], [], [], []
    reference_dates = None
    for cost in spec.costs_bps:
        candidate, benchmark = _paired(paths_by_cost[cost], benchmarks_by_cost[cost], expected_years=spec.expected_years)
        for path in (candidate, benchmark):
            if "cost_bps" not in path or not np.allclose(path.cost_bps, cost, rtol=0, atol=1e-12):
                raise ValueError("blocked: cost-stressed replay lacks its matching declared fee assumption")
        dates = candidate.loc[:, list(DATES)]
        if reference_dates is not None and not dates.equals(reference_dates):
            raise ValueError("blocked: cost scenarios use different observation windows")
        reference_dates = dates
        metric = paired_metrics(candidate, benchmark)
        annual = annual_paired_metrics(candidate, benchmark)
        excess = annual.log_excess_sum.to_numpy()
        positive = excess[excess > 0]
        largest_share = float(positive.max() / positive.sum()) if len(positive) else None
        leave_one_out = float((excess.sum() - excess).min())
        checks = {
            "aggregate_positive_net_log_excess": metric["log_excess_sum"] > 0,
            "minimum_positive_signal_years": int((excess > 0).sum()) >= spec.minimum_positive_years,
            "median_annual_log_excess_positive": float(np.median(excess)) > 0,
            "leave_any_one_year_out_log_excess_positive": leave_one_out > 0,
            "anchored_max_drawdown_not_worse_than_soxx": metric["max_drawdown_pct"] >= metric["benchmark_max_drawdown_pct"] - 1e-10,
            "cvar5_not_worse_than_soxx": metric["cvar5_pct"] >= metric["benchmark_cvar5_pct"] - 1e-10,
            "minimum_positive_seeds_at_both_costs": seed_all_costs >= spec.minimum_positive_seeds,
        }
        if spec.maximum_positive_year_share is not None:
            checks["positive_year_concentration_within_declared_limit"] = largest_share is not None and largest_share <= spec.maximum_positive_year_share
        for block in spec.mean_block_lengths:
            boot = stationary_bootstrap_excess(candidate, benchmark, mean_block=block,
                resamples=spec.bootstrap_resamples, alpha=spec.alpha / spec.family_size, seed=spec.random_seed)
            bootstrap.append({"cost_bps": cost, "declared_variant_family_size": spec.family_size, **boot})
            checks[f"stationary_block_{block}_lower_bound_positive"] = boot["lower_bound_positive"]
        rows.append({"cost_bps": cost, **checks, "all_pass": bool(all(checks.values())),
                     "positive_years": int((excess > 0).sum()), "years": len(excess),
                     "positive_seeds_at_this_cost": seed_counts[cost], "seeds_positive_at_both_costs": seed_all_costs,
                     "seed_evidence_complete": seed_table is not None,
                     "largest_positive_year_share": largest_share,
                     "minimum_leave_one_year_out_log_excess": leave_one_out})
        metrics.append({"cost_bps": cost, **metric})
        annuals.append(annual.assign(cost_bps=cost))
    gates = pd.DataFrame(rows)
    return {
        "spec": asdict(spec), "gates": gates, "metrics": pd.DataFrame(metrics),
        "annual": pd.concat(annuals, ignore_index=True), "bootstrap": pd.DataFrame(bootstrap),
        "historical_qualification_pass": bool(gates.all_pass.all()), "new_forward_qualified": False,
        "promotion_ready": False,
        "evidence_scope": "nested_historical_qualification_on_previously_inspected_years",
        "inference_scope": "A fixed family-size adjustment does not account for all earlier adaptive research; no unconditional significance claim.",
    }


# Short notebook-facing name; both names share the same strict contract.
stable_gates = stable_excess_gates


def seed_dispersion(paths: Mapping[str, pd.DataFrame], benchmark: pd.DataFrame) -> dict:
    if len(paths) < 2:
        raise ValueError("blocked: seed dispersion requires every declared repeated fit, not a selected best seed")
    table = pd.DataFrame([{"seed": str(seed), **paired_metrics(path, benchmark)} for seed, path in paths.items()])
    values = table.annualized_log_excess.to_numpy()
    return {"per_seed": table, "seed_count": len(table), "mean_annualized_log_excess": float(values.mean()),
            "std_annualized_log_excess": float(values.std(ddof=1)), "minimum_annualized_log_excess": float(values.min()),
            "maximum_annualized_log_excess": float(values.max()), "all_seeds_positive": bool((values > 0).all()),
            "best_seed_selected": False,
            "ensemble_requires_separate_own_holdings_replay": True,
            "seeds_are_not_independent_market_observations": True}


def native_allocation_reconciliation(frame: pd.DataFrame, replay: pd.DataFrame, *, provider_uri: str | Path,
                                      cost_bps: float = 10., account: float = 100_000., native_reconciler=None) -> dict:
    """Qlib daily execution of declared net targets, without legacy risk caps.

    Reuse only the generic fee-funded target/Exchange/Position reconciliation;
    no old controller, 30% SOXL ceiling, 15% cash floor, or recovery rule runs.
    Exact observed open marks are research conventions, not actual live fills.
    """
    from src.analysis.stage1_regime_validation import native_open_reconciliation
    from qlib.contrib.evaluate import risk_analysis

    schedule = validate_replay(replay)
    market = _dates(frame)
    if not schedule.loc[:, list(DATES)].equals(market.loc[:, list(DATES)]):
        raise ValueError("blocked: native schedule and market dates differ")
    if "recovery_action" in schedule and schedule.recovery_action.astype(bool).any():
        raise ValueError("blocked: integrated daily policy cannot include an intraday overlay")
    schedule["recovery_action"] = False
    if "execute_open_target" not in schedule:
        raise ValueError("blocked: native replay requires an explicit target-versus-HOLD schedule")
    if not schedule.execute_open_target.isin([True, False, 0, 1]).all():
        raise ValueError("blocked: target-versus-HOLD schedule is not boolean")
    for choices in ([f"target_{asset}_weight" for asset in ASSETS],
                    [f"open_{asset}_weight" for asset in ASSETS],
                    [f"{asset}_weight" for asset in ASSETS]):
        if set(choices).issubset(schedule):
            weights = schedule.loc[:, choices].to_numpy(dtype=float)
            break
    else:
        raise ValueError("blocked: native replay lacks its three declared post-fee target weights")
    if not np.isfinite(weights).all() or (weights < -1e-12).any() or not np.allclose(weights.sum(axis=1), 1, rtol=0, atol=1e-10):
        raise ValueError("blocked: integrated targets must be normalized finite long-only weights")
    schedule.loc[:, [f"open_{asset}_weight" for asset in ASSETS]] = weights
    reconcile = native_open_reconciliation if native_reconciler is None else native_reconciler
    native = reconcile(market, schedule, provider_uri=provider_uri, cost_bps=cost_bps, account=account)
    close_net = native["report"]["return"] - native["report"]["cost"]
    open_net = native["reconciliation"]["native_next_open_return"]
    # The generic older helper uses Qlib's regional default scaler. Explicitly
    # replace that report for this US experiment; do not silently relabel the
    # native close-mark return series as next-open performance.
    native["risk_analysis"] = risk_analysis(close_net, N=252, freq=None, mode="product")
    native["risk_analysis_open_to_open"] = risk_analysis(open_net, N=252, freq=None, mode="product")
    close_anchored = _drawdown(np.log1p(close_net.to_numpy()))
    open_anchored = _drawdown(np.log1p(open_net.to_numpy()))
    native["risk_analysis"].loc["max_drawdown_from_initial_capital", "risk"] = close_anchored
    native["risk_analysis_open_to_open"].loc["max_drawdown_from_initial_capital", "risk"] = open_anchored
    native["audit"].update({"policy_scope": "continuous_three_asset_targets_without_legacy_allocation_caps",
        "legacy_controller_executed": False, "scope": "complete prescribed daily target/HOLD schedule; actual Qlib Exchange, Position, report and next-open accounting",
        "risk_annualization_sessions": 252,
        "native_close_anchored_max_drawdown_pct": 100 * close_anchored,
        "native_open_to_open_anchored_max_drawdown_pct": 100 * open_anchored,
        "risk_table_mark_convention": "risk_analysis uses native close marks; risk_analysis_open_to_open uses native positions marked at the following open",
        "execution_limitation": "Observed historical opens and continuous adjusted shares; no claim of spread/impact/partial-fill evidence."})
    if native["audit"]["status"] != "passed":
        raise ValueError("blocked: native Qlib allocation reconciliation failed")
    return native
