"""Frozen Stage1 domain-head comparison from native Qlib account outputs."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.analysis.integrated_allocation import (
    DATES, paired_metrics, stationary_bootstrap_excess, validate_replay,
)


def _drawdown_pct(nav, initial_cash):
    values = np.r_[initial_cash, np.asarray(nav, dtype=float)]
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("blocked: invalid native NAV marks")
    return float(100 * (values / np.maximum.accumulate(values) - 1).min())


def _period_coverage(trade_dates, calendar, frequency):
    labels = pd.DatetimeIndex(trade_dates).to_period(frequency)
    source_labels = calendar.to_period(frequency)
    result = {}
    for period in labels.unique():
        expected = calendar[source_labels == period]
        observed = pd.DatetimeIndex(trade_dates)[labels == period]
        # Matching the truncated final year of a source is not full-year
        # evidence.  Require the frozen source to bracket the whole period.
        bracketed = calendar[0] < period.start_time and calendar[-1] > period.end_time
        covered = len(expected) > 0 and observed.equals(expected)
        result[period] = {
            "complete_period": bool(bracketed and covered),
            "calendar_expected_sessions": len(expected),
            "calendar_coverage_complete": bool(covered),
            "calendar_contains_full_period": bool(bracketed),
        }
    return labels, result


def analyze_accounts(books, daily_calendar_frame, plan):
    """Original fixed domain-head comparison contract."""
    return analyze_paired_accounts(books, daily_calendar_frame, plan,
        candidate_family="C_domain",
        required_comparators=("A_active_research_replay", "B_shared", "SOXX", "strong9097"),
        expected_families=("A_active_research_replay", "B_shared", "C_domain", "SOXX", "strong9097", "additional27baa"))


def analyze_paired_accounts(books, daily_calendar_frame, plan, *, candidate_family,
                            required_comparators, expected_families):
    """Small paired summaries plus all overlapping windows for Recorder storage.

    ``daily_calendar_frame`` is the pinned Qlib source calendar, not a new
    weekday or exchange-calendar reconstruction.  All native account rows are
    retained; no inner join or result-dependent window/seed selection occurs.
    """
    account = plan["accounts"]
    families, costs = account["families"], account["slippage_bps"]
    comparators = plan["gates"]["required_comparators"]
    expected_keys = {(family, cost) for family in families for cost in costs}
    if (len(families) != len(expected_families) or set(families) != set(expected_families)
            or list(costs) != [0, 2, 5] or set(books) != expected_keys):
        raise ValueError("blocked: frozen families and all three cost accounts are required")
    if len(comparators) != len(required_comparators) or set(comparators) != set(required_comparators):
        raise ValueError("blocked: frozen paired comparator contract differs")
    if candidate_family not in families or candidate_family in comparators:
        raise ValueError("blocked: candidate must be distinct from paired comparators")
    initial_cash = float(account["initial_cash"])
    calendar = pd.DatetimeIndex(pd.to_datetime(daily_calendar_frame.trade_date).dropna()).unique().sort_values()
    if calendar.empty or not calendar.equals(calendar.normalize()):
        raise ValueError("blocked: invalid frozen daily source calendar")
    paths = {key: validate_replay(book["authority"]) for key, book in books.items()}
    dates = paths[("SOXX", costs[0])][list(DATES)]
    if len(dates) != account["expected_sessions"]:
        raise ValueError("blocked: native account session count differs from frozen plan")
    if (str(dates.signal_date.iloc[0].date()) != account["first_signal"]
            or str(dates.trade_date.iloc[0].date()) != account["first_trade"]
            or dates.exit_date.iloc[-1].date() != pd.Timestamp(account["terminal"]).date()):
        raise ValueError("blocked: native account endpoints differ from frozen plan")
    trade_dates = pd.DatetimeIndex(dates.trade_date)
    expected_dates = calendar[(calendar >= trade_dates[0]) & (calendar <= trade_dates[-1])]
    if not trade_dates.equals(expected_dates):
        raise ValueError("blocked: native accounts omit a frozen source trading session")
    period_info = {kind: _period_coverage(trade_dates, calendar, frequency)
                   for frequency, kind in (("Y", "year"), ("Q", "quarter"))}
    metrics, periods, roll_summary, pairs, uncertainty, exposure = [], [], [], [], [], []
    rolling = {}
    point_grid = None
    for family in families:
        for cost in costs:
            key = (family, cost)
            book, path, bench = books[key], paths[key], paths[("SOXX", cost)]
            pd.testing.assert_frame_equal(path[list(DATES)], dates)
            if book.get("audit", {}).get("status") != "passed":
                raise ValueError("blocked: account has no passed native Qlib verification")
            np.testing.assert_allclose((1 + path.strategy_return).cumprod(), path.equity, rtol=1e-9, atol=1e-9)
            points = book["points"]
            stamps = pd.DatetimeIndex(pd.to_datetime(points.stamp))
            if stamps.has_duplicates or not stamps.is_monotonic_increasing:
                raise ValueError("blocked: native point NAV must be unique and chronological")
            if point_grid is None:
                point_grid = stamps
            elif not stamps.equals(point_grid):
                raise ValueError("blocked: native valuation grids differ across accounts")
            np.testing.assert_allclose(
                points.iloc[0][["cash", "receivable", "q_soxl", "q_soxx", "nav"]].to_numpy(float),
                [initial_cash, 0., 0., 0., initial_cash], rtol=0, atol=1e-7,
            )
            observed_close = pd.Series(points.nav.to_numpy(float), index=stamps).groupby(stamps.normalize()).last()
            closes = observed_close.reindex(trade_dates)
            if closes.isna().any():
                raise ValueError("blocked: native closing marks missing")
            meta = {"family": family, "slippage_bps": cost}
            metrics.append({
                **meta, **paired_metrics(path, bench),
                "max_drawdown_basis": "next_open_to_next_open_net_account_returns",
                "close_max_drawdown_pct": _drawdown_pct(closes, initial_cash),
                "sampled_point_max_drawdown_pct": _drawdown_pct(points.nav, initial_cash),
                "mean_ending_soxl_weight": float(path.ending_soxl_weight.mean()),
                "mean_ending_soxx_weight": float(path.ending_soxx_weight.mean()),
                "mean_ending_cash_weight": float(path.ending_cash_weight.mean()),
                "zero_SOXL_quantity_fraction": float(path.ending_soxl_quantity.eq(0).mean()),
                "fresh_fullminute_risk_qualified": False,
            })
            for kind, (labels, coverage) in period_info.items():
                for period in labels.unique():
                    mask = labels == period
                    periods.append({**meta, "kind": kind, "period": str(period),
                                    **coverage[period], **paired_metrics(path.loc[mask], bench.loc[mask])})
            excess = np.log1p(path.strategy_return) - np.log1p(bench.strategy_return)
            if len(excess) < 126:
                raise ValueError("blocked: insufficient observations for frozen 126-session windows")
            windows = pd.DataFrame({
                "first_trade": dates.trade_date.iloc[:-125].to_numpy(),
                "last_exit": dates.exit_date.iloc[125:].to_numpy(),
                "relative_log": excess.rolling(126).sum().iloc[125:].to_numpy(),
            })
            windows["relative_return_pct"] = 100 * np.expm1(windows.relative_log)
            rolling[f"{family}_{cost}"] = windows
            roll_summary.append({**meta, "comparator": "SOXX", "window_sessions": 126,
                                 "windows": len(windows), "overlapping_not_independent": True,
                                 "positive_fraction": float(windows.relative_log.gt(0).mean()),
                                 "minimum_relative_return_pct": float(windows.relative_return_pct.min()),
                                 "median_relative_return_pct": float(windows.relative_return_pct.median())})
            targets = book.get("targets")
            if targets is not None:
                pd.testing.assert_index_equal(pd.DatetimeIndex(targets.trade_date), trade_dates, check_names=False)
                exposure.append({
                    **meta, "target_basis": "controlled_or_hold_target; not original model proposal",
                    "mean_target_soxl_weight": float(targets.soxl_weight.mean()),
                    "mean_target_cash_weight": float(targets.cash_weight.mean()),
                    "mean_target_nominal_beta": float((3 * targets.soxl_weight + targets.soxx_weight).mean()),
                    "mean_ending_nominal_beta": float((3 * path.ending_soxl_weight + path.ending_soxx_weight).mean()),
                    "mean_ending_cash_weight": float(path.ending_cash_weight.mean()),
                    "open_execute_fraction": float(path.execute_open_target.mean()),
                    "recovery_fraction": float(path.recovery_action.mean()),
                })

    bootstrap = plan["gates"]["paired_uncertainty"]
    bootstrap_args = {name: bootstrap[name] for name in ("mean_block", "resamples", "alpha", "seed")}
    for cost in costs:
        candidate = paths[(candidate_family, cost)]
        for comparator in [*comparators, "additional27baa"]:
            other = paths[(comparator, cost)]
            meta = {"family": candidate_family, "comparator": comparator, "slippage_bps": cost,
                    "required_comparator": comparator in comparators}
            pairs.append({**meta, **paired_metrics(candidate, other)})
            if comparator in comparators:
                uncertainty.append({**meta, **stationary_bootstrap_excess(candidate, other, **bootstrap_args)})
    metrics, periods, pairs = pd.DataFrame(metrics), pd.DataFrame(periods), pd.DataFrame(pairs)
    roll_summary, uncertainty = pd.DataFrame(roll_summary), pd.DataFrame(uncertainty)
    candidate = metrics.loc[metrics.family.eq(candidate_family)]
    annual = periods.loc[periods.family.eq(candidate_family) & periods.kind.eq("year") & periods.complete_period]
    annual_counts = annual.groupby("slippage_bps").size().reindex(costs, fill_value=0)
    annual_rates = annual.groupby("slippage_bps").relative_wealth.apply(lambda wealth: wealth.gt(1).mean()).reindex(costs)
    gates = {
        "net_relative_wealth_all_costs": bool(pairs.loc[pairs.required_comparator, "relative_wealth"].gt(1 + 1e-8).all()),
        "sampled_drawdown_30pct": bool(candidate.close_max_drawdown_pct.ge(-30).all()
                                       and candidate.sampled_point_max_drawdown_pct.ge(-30).all()),
        "cvar_no_worse_soxx": bool(candidate.cvar5_pct.ge(candidate.benchmark_cvar5_pct).all()),
        "minimum_complete_years": bool(annual_counts.ge(plan["gates"]["minimum_complete_years"]).all()),
        "complete_annual_70pct": bool(annual_rates.notna().all()
                                     and annual_rates.ge(plan["gates"]["complete_annual_positive_fraction"]).all()),
        "rolling126_70pct": bool(roll_summary.loc[roll_summary.family.eq(candidate_family), "positive_fraction"]
                                 .ge(plan["gates"]["rolling126_positive_fraction"]).all()),
        "paired_block_lower_positive": bool(len(uncertainty) == len(comparators) * len(costs)
                                            and uncertainty.lower_bound_positive.all()),
        "fresh_fullminute_risk_qualified": False,
        "qualified_independent_forward": False,
    }
    return {"metrics": metrics, "paired_comparison": pairs,
            "annual_metrics": periods.loc[periods.kind.eq("year")].reset_index(drop=True),
            "quarterly_metrics": periods.loc[periods.kind.eq("quarter")].reset_index(drop=True),
            "rolling_summary": roll_summary, "uncertainty": uncertainty,
            "rolling": rolling, "gates": gates, "exposure_summary": pd.DataFrame(exposure)}


def information_metrics(predictions, ensembles, dataset, *, as_of="2026-10-09T09:47:00-04:00"):
    """Primary task diagnostics only; classification never gates account runs."""
    labels = dataset.prepare("predict", col_set=["label", "meta"]).copy()
    if isinstance(labels.index, pd.MultiIndex):
        labels.index = labels.index.get_level_values("datetime")
    if labels.index.has_duplicates or not labels.index.is_monotonic_increasing:
        raise ValueError("blocked: primary information labels have ambiguous dates")
    cutoff = pd.Timestamp(as_of)
    if cutoff.tzinfo is None:
        raise ValueError("blocked: information cutoff needs an explicit timezone")
    maturity = pd.to_datetime(labels[("meta", "label_available_at")])
    if maturity.dt.tz is None:
        raise ValueError("blocked: label maturity must retain its timezone")
    matured = maturity.notna() & maturity.le(cutoff)
    reports = []
    all_predictions = [(family, str(seed), value) for (family, seed), value in predictions.items()]
    all_predictions += [(family, "ensemble", value) for family, value in ensembles.items()]
    if not all_predictions:
        raise ValueError("blocked: no primary predictions supplied")
    for family, seed, prediction in all_predictions:
        pd.testing.assert_index_equal(pd.DatetimeIndex(prediction.index), pd.DatetimeIndex(labels.index), check_names=False)
        for task, label_name, probability_name in (
            ("tail", "tail_event_5d", "tail_probability"),
            ("alpha", "relative_alpha_5d", "alpha_probability"),
        ):
            actual = labels[("label", label_name)]
            valid = matured & np.isfinite(actual.to_numpy(float))
            if task == "tail" and not actual.loc[valid].isin([0., 1.]).all():
                raise ValueError("blocked: matured primary tail event is not binary")
            observed = actual.loc[valid].gt(0).to_numpy(dtype=float)
            probability = prediction.loc[valid, probability_name].to_numpy(dtype=float)
            if not np.isfinite(probability).all() or (probability < 0).any() or (probability > 1).any():
                raise ValueError("blocked: invalid matured primary probability")
            clipped = np.clip(probability, 1e-15, 1 - 1e-15)
            count = len(observed)
            reports.append({
                "family": family, "seed": seed, "task": task, "as_of": str(cutoff),
                "prediction_rows": len(prediction), "observations": count,
                "excluded_unmatured_or_unknown_maturity": int((~matured).sum()),
                "excluded_matured_missing_label": int((matured & ~valid).sum()),
                "positive_events": int(observed.sum()),
                "event_rate": float(observed.mean()) if count else np.nan,
                "mean_probability": float(probability.mean()) if count else np.nan,
                "bce": float(-(observed * np.log(clipped) + (1 - observed) * np.log1p(-clipped)).mean()) if count else np.nan,
                "brier": float(np.mean((probability - observed) ** 2)) if count else np.nan,
                "auc": float(roc_auc_score(observed, probability)) if count and len(np.unique(observed)) == 2 else np.nan,
                "history_status": "reused_out_of_fit_primary_data_not_independent_validation",
                "account_execution_gate": False,
            })
    return pd.DataFrame(reports)
