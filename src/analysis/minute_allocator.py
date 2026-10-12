"""Analysis of native account outputs for a fixed minute-feature comparison."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.analysis.integrated_allocation import (
    _path_metrics, annual_paired_metrics, paired_metrics, stationary_bootstrap_excess,
)


def feature_information_audit(features, labels, timing, *, groups, years, cutoff) -> pd.DataFrame:
    matured = timing.label_available_at.notna() & timing.label_available_at.lt(pd.Timestamp(cutoff))
    rows = []
    for group, names in groups.items():
        for name in names:
            for year in years:
                for target in ("tail_event_5d", "relative_alpha_5d"):
                    keep = matured & (features.index.year == year)
                    paired = pd.concat([features.loc[keep, name], labels.loc[keep, target]], axis=1).dropna()
                    correlation = (float(spearmanr(paired.iloc[:, 0], paired.iloc[:, 1]).statistic)
                                   if len(paired) > 20 and paired.nunique().gt(1).all() else np.nan)
                    rows.append({"group": group, "feature": name, "year": year, "target": target,
                                 "observations": len(paired), "rank_ic": correlation,
                                 "selection_role": "diagnostic; no individual feature cherry-picking"})
    return pd.DataFrame(rows)


def compare_native_books(books: dict, *, window: str, costs: list[int], families: list[str],
                         references: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, annual = [], []
    for fee in costs:
        for family in families:
            path = books[f"{window}_{fee}_{family}"]["authority"]
            metrics = _path_metrics(path)
            values = {"window": window, "cost_bps": fee, "family": family, **metrics}
            for reference in references:
                benchmark = books[f"{window}_{fee}_{reference}"]["authority"]
                pair = paired_metrics(path, benchmark)
                values["wealth_ratio_vs_" + reference] = pair["relative_wealth"]
                values["mean_log_excess_vs_" + reference] = pair["log_excess_sum"] / len(path)
                benchmark_metrics = _path_metrics(benchmark)
                values["mdd_no_worse_than_" + reference] = metrics["max_drawdown_pct"] >= benchmark_metrics["max_drawdown_pct"]
                values["cvar_no_worse_than_" + reference] = metrics["cvar5_pct"] >= benchmark_metrics["cvar5_pct"]
                years = annual_paired_metrics(path, benchmark)
                years["window"], years["cost_bps"], years["family"], years["reference"] = window, fee, family, reference
                annual.append(years)
            values["native_reconciliation_maximum_error"] = books[f"{window}_{fee}_{family}"]["native"]["audit"]["maximum_session_return_error"]
            rows.append(values)
    return pd.DataFrame(rows), pd.concat(annual, ignore_index=True)


def development_selection(books, *, candidate_families, costs, bootstrap) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    records, uncertainty = [], []
    for family in candidate_families:
        scores, positive, risk, lower = [], [], [], []
        for fee in costs:
            path = books[f"development_{fee}_{family}"]["authority"]
            metrics = _path_metrics(path)
            for reference in ("base", "availability_control"):
                comparator = books[f"development_{fee}_{reference}"]["authority"]
                pair = paired_metrics(path, comparator)
                scores.append(float(annual_paired_metrics(path, comparator).annualized_log_excess.mean()) / 252.)
                positive.append(pair["relative_wealth"] > 1.)
                base = _path_metrics(comparator)
                risk.append(metrics["max_drawdown_pct"] >= base["max_drawdown_pct"] and metrics["cvar5_pct"] >= base["cvar5_pct"])
                for length in bootstrap["mean_block_lengths"]:
                    bound = stationary_bootstrap_excess(path, comparator, mean_block=length,
                        resamples=bootstrap["resamples"], alpha=bootstrap["alpha"] / bootstrap["family_size"],
                        seed=bootstrap["seed"])
                    uncertainty.append({**bound, "bootstrap_reference": bound["reference"],
                                        "family": family, "cost_bps": fee, "reference": reference})
                    lower.append(bound["lower_bound_positive"])
        records.append({"family": family, "worst_cost_and_reference_mean_log_excess": min(scores),
                        "positive_both_costs_and_references": all(positive), "risk_no_worse_than_matched_controls": all(risk),
                        "family_adjusted_bounds_positive": all(lower), "uses_outer_diagnostic": False,
                        "selection_role": "fixed development rank; risk/uncertainty gates cannot be overridden"})
    table = pd.DataFrame(records).sort_values("worst_cost_and_reference_mean_log_excess", ascending=False)
    return str(table.family.iloc[0]), table, pd.DataFrame(uncertainty)
