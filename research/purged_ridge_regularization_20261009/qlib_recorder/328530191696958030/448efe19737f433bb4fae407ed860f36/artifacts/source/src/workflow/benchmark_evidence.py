"""Shared reporting for registered Qlib-derived benchmark-relative paths."""
from __future__ import annotations

import pandas as pd

from src.analysis.benchmark_relative_protocol import evaluate_benchmark_relative
from src.analysis.integrated_allocation import paired_metrics, annual_paired_metrics


def evaluate_registered_paths(paths, *, plan, calendar, candidate_controls):
    """Report every path; only preregistered candidates enter qualification.

    This function never selects a model or changes parameters after results.
    Whole post-development history is explicitly a reused diagnostic window.
    """
    costs, years = plan["costs_bps"], plan["development_years"]
    names = {name for name, cost in paths}
    if set(paths) != {(name, cost) for name in names for cost in costs} or "soxx" not in names:
        raise ValueError("blocked: incomplete strategy-by-cost inventory")
    if names != {"soxx", *[v["name"] for v in plan["variants"]]}:
        raise ValueError("blocked: paths differ from registered strategy inventory")
    for (name, cost), path in paths.items():
        if "cost_bps" not in path or not path.cost_bps.eq(cost).all():
            raise ValueError("blocked: path cost differs from inventory key")
        as_of = plan["as_of"] if "as_of" in plan else plan["source"]["as_of"]
        if path.exit_date.max() != pd.Timestamp(as_of) or (path.exit_date > pd.Timestamp(as_of)).any():
            raise ValueError("blocked: path does not end at the registered market cutoff")
    if set(candidate_controls) - names or set(candidate_controls.values()) - names:
        raise ValueError("blocked: registered candidate/control is missing")
    if len(candidate_controls) != plan["candidate_budget"]:
        raise ValueError("blocked: registered candidate budget differs")
    if any(name == control for name, control in candidate_controls.items()):
        raise ValueError("blocked: candidate cannot serve as its own control")
    comparisons, annual, ablations, gate_tables, rolling, bootstraps, evidence = [], [], [], [], [], [], {}
    def dev(path):
        return path.loc[path.signal_date.dt.year.isin(years)]
    calendar = pd.DatetimeIndex(calendar)
    for name in candidate_controls:
        result = evaluate_benchmark_relative({cost: dev(paths[(name, cost)]) for cost in costs},
            {cost: dev(paths[("soxx", cost)]) for cost in costs}, expected_years=tuple(years),
            expected_signal_dates=calendar[calendar.year.isin(years)], **plan["gates"])
        evidence[name] = result
        for target, key in ((gate_tables, "gates"), (rolling, "rolling_summary"), (bootstraps, "bootstrap")):
            target.append(result[key].assign(strategy=name))
    for (name, cost), path in paths.items():
        reference = paths[("soxx", cost)]
        annual.append(annual_paired_metrics(path, reference).assign(strategy=name, cost_bps=cost))
        for window, mask in (("development", path.signal_date.dt.year.isin(years)),
                             ("reused_diagnostic", path.signal_date.dt.year.gt(max(years)))):
            if not mask.any():
                raise ValueError("blocked: registered comparison window is empty")
            comparisons.append({"strategy": name, "cost_bps": cost, "window": window,
                **paired_metrics(path.loc[mask], reference.loc[mask])})
            if name in candidate_controls:
                control = candidate_controls[name]
                ablations.append({"strategy": name, "control": control, "cost_bps": cost, "window": window,
                    **paired_metrics(path.loc[mask], paths[(control, cost)].loc[mask])})
    tables = {"backtest_benchmark_comparison.csv": pd.DataFrame(comparisons),
        "annual_metrics.csv": pd.concat(annual, ignore_index=True),
        "ablation_comparison.csv": pd.DataFrame(ablations), "gates.csv": pd.concat(gate_tables, ignore_index=True),
        "rolling_summary.csv": pd.concat(rolling, ignore_index=True), "bootstrap_analysis.csv": pd.concat(bootstraps, ignore_index=True)}
    return tables, evidence
