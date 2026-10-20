"""Historical-only SOXX hard-gate assessment for paired account tables."""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Mapping

import numpy as np
import pandas as pd


CONTRACT_VERSION = 2
COSTS = (0, 2, 5)
EXPECTED_STATISTICS = {"mean_block": 63, "resamples": 10000, "random_seed": 20261014,
                       "alpha": 0.025, "family_size": 15}


def _table(analysis: Mapping[str, Any], name: str, family: str, **filters) -> pd.DataFrame:
    value = analysis.get(name)
    if not isinstance(value, pd.DataFrame) or "family" not in value or "slippage_bps" not in value:
        return pd.DataFrame()
    result = value.loc[value.family.eq(family)].copy()
    for column, expected in filters.items():
        if column not in result:
            return pd.DataFrame()
        result = result.loc[result[column].eq(expected)]
    return result


def _one_per_cost(frame: pd.DataFrame) -> bool:
    if "slippage_bps" not in frame or len(frame) != len(COSTS):
        return False
    costs = pd.to_numeric(frame.slippage_bps, errors="coerce")
    return bool(np.isfinite(costs.to_numpy(float)).all() and
                all(int(costs.eq(cost).sum()) == 1 for cost in COSTS))


def _finite(frame: pd.DataFrame, columns: tuple[str, ...]) -> bool:
    return bool(not frame.empty and set(columns) <= set(frame.columns) and
                np.isfinite(frame.loc[:, list(columns)].to_numpy(dtype=float)).all())


def _soxx_wealth_aligned(metrics: pd.DataFrame, pairs: pd.DataFrame) -> bool:
    fields = ("slippage_bps", "relative_wealth")
    if not _one_per_cost(metrics) or not _one_per_cost(pairs) or not _finite(metrics, fields) or not _finite(pairs, fields):
        return False
    left, right = metrics.loc[:, list(fields)].copy(), pairs.loc[:, list(fields)].copy()
    left["slippage_bps"] = pd.to_numeric(left.slippage_bps)
    right["slippage_bps"] = pd.to_numeric(right.slippage_bps)
    merged = left.merge(right, on="slippage_bps",
        suffixes=("_metric", "_pair"), validate="one_to_one")
    return bool(len(merged) == len(COSTS) and np.allclose(merged.relative_wealth_metric,
        merged.relative_wealth_pair, rtol=0, atol=1e-8))


def _policy_status(provenance: Mapping[str, Any], candidate: str) -> str:
    policy = provenance.get("objective_policy")
    if not isinstance(policy, Mapping):
        return "needs_policy"
    try:
        digest = hashlib.sha256(json.dumps(policy, sort_keys=True, separators=(",", ":"),
                                         allow_nan=False).encode()).hexdigest()
    except (TypeError, ValueError):
        return "policy_mismatch"
    proof = provenance.get("frozen_plan_policy_sha256")
    frozen = provenance.get("frozen_plan_sha256")
    if (not isinstance(frozen, str) or re.fullmatch(r"[0-9a-f]{64}", frozen) is None or digest != proof
            or policy.get("contract_version") != CONTRACT_VERSION
            or policy.get("candidate_family") != candidate
            or not provenance.get("phase") or policy.get("phase") != provenance.get("phase")
            or policy.get("statistical_settings") != EXPECTED_STATISTICS):
        return "policy_mismatch"
    return "verified_from_frozen_plan_identity"


def evaluate_soxx_objective(analysis: Mapping[str, Any], *, candidate_family: str,
                            provenance: Mapping[str, Any]) -> dict[str, Any]:
    """Apply contract-v2 historical gates; receipt refs/policy are caller-bound.

    ``provenance`` requires ``receipt_refs`` entries with path/SHA-256,
    ``phase``, ``candidate_family``, ``frozen_plan_sha256``, the policy object,
    and its ``frozen_plan_policy_sha256`` recorded by that frozen plan.
    """
    metrics = _table(analysis, "metrics", candidate_family)
    pairs = _table(analysis, "paired_comparison", candidate_family, comparator="SOXX")
    annual = _table(analysis, "annual_metrics", candidate_family, kind="year")
    rolling = _table(analysis, "rolling_summary", candidate_family, comparator="SOXX")
    uncertainty = _table(analysis, "uncertainty", candidate_family, comparator="SOXX")
    metric_fields = ("relative_wealth", "close_max_drawdown_pct", "sampled_point_max_drawdown_pct",
                     "cvar5_pct", "benchmark_cvar5_pct")
    complete = annual.loc[annual.get("complete_period", pd.Series(False, index=annual.index)).eq(True)].copy()
    duplicate_years = ("slippage_bps" in complete and "period" in complete and
                       complete.duplicated(["slippage_bps", "period"]).any())
    cost_year_counts = (complete.groupby("slippage_bps").period.nunique()
                        if {"slippage_bps", "period"} <= set(complete) else pd.Series(dtype=int))
    annual_rates = (complete.assign(positive=complete.relative_wealth.gt(1)).groupby("slippage_bps").positive.mean()
                    if {"relative_wealth", "slippage_bps"} <= set(complete) else pd.Series(dtype=float))
    annual_coverage = (not duplicate_years and {"slippage_bps", "period"} <= set(complete) and
                       all(int(cost_year_counts.get(cost, 0)) >= 2 for cost in COSTS))
    annual_stability = annual_coverage and all(float(annual_rates.get(cost, 0)) >= .70 for cost in COSTS)
    counts = pd.to_numeric(rolling.get("windows", pd.Series(dtype=float)), errors="coerce")
    fractions = pd.to_numeric(rolling.get("positive_fraction", pd.Series(dtype=float)), errors="coerce")
    rolling_ok = (_one_per_cost(rolling) and {"window_sessions", "windows", "positive_fraction"} <= set(rolling) and
                  rolling.window_sessions.eq(126).all() and np.isfinite(counts.to_numpy(float)).all() and
                  counts.gt(0).all() and counts.mod(1).eq(0).all() and np.isfinite(fractions.to_numpy(float)).all() and
                  fractions.between(0, 1).all() and fractions.ge(.70).all())
    finite_required = (_finite(metrics, metric_fields) and _finite(pairs, ("relative_wealth",)) and
                       _finite(complete, ("relative_wealth",)) and _finite(uncertainty, ("daily_lower_bound",)))
    ci_settings = True
    for field, expected in {"mean_block": 63, "resamples": 10000, "random_seed": 20261014,
                            "alpha": .025 / 15, "family_size": 15}.items():
        if field in uncertainty and not pd.to_numeric(uncertainty[field], errors="coerce").eq(expected).all():
            ci_settings = False
    receipt_refs = provenance.get("receipt_refs", [])
    receipt_ok = (isinstance(receipt_refs, list) and bool(receipt_refs) and
                  all(isinstance(ref, Mapping) and ref.get("path") and
                      isinstance(ref.get("sha256"), str) and re.fullmatch(r"[0-9a-f]{64}", ref["sha256"]) for ref in receipt_refs) and
                  provenance.get("candidate_family") == candidate_family)
    policy_status = _policy_status(provenance, candidate_family)
    gates = {
        "receipt_provenance": receipt_ok,
        "one_candidate_row_per_cost": _one_per_cost(metrics),
        "finite_required_metrics": finite_required,
        "soxx_pair_per_cost": _one_per_cost(pairs),
        "candidate_soxx_wealth_consistent": _soxx_wealth_aligned(metrics, pairs),
        "soxx_relative_wealth": (_finite(pairs, ("relative_wealth",)) and pairs.relative_wealth.gt(1 + 1e-8).all()),
        "two_complete_years_each_cost": annual_coverage,
        "annual_positive_fraction_70pct": annual_stability,
        "soxx_rolling_126_positive_70pct": rolling_ok,
        "drawdown_limit_30pct": (_finite(metrics, ("close_max_drawdown_pct", "sampled_point_max_drawdown_pct")) and
            metrics.close_max_drawdown_pct.ge(-30).all() and metrics.sampled_point_max_drawdown_pct.ge(-30).all()),
        "cvar_no_worse_than_soxx": (_finite(metrics, ("cvar5_pct", "benchmark_cvar5_pct")) and
            metrics.cvar5_pct.ge(metrics.benchmark_cvar5_pct).all()),
        "soxx_ci_per_cost": _one_per_cost(uncertainty),
        "soxx_ci_lower_positive": (_finite(uncertainty, ("daily_lower_bound",)) and
            uncertainty.daily_lower_bound.gt(0).all() and uncertainty.get("lower_bound_positive", pd.Series(True, index=uncertainty.index)).eq(True).all()),
        "statistics_match_frozen_policy": ci_settings and policy_status == "verified_from_frozen_plan_identity",
    }
    hard_pass = all(gates.values())
    status = "historical_pass" if hard_pass else (
        "needs_policy" if policy_status == "needs_policy" and all(v for k, v in gates.items() if k != "statistics_match_frozen_policy")
        else "historical_fail")
    return {"contract_version": CONTRACT_VERSION, "candidate_family": candidate_family,
            "status": status, "historical_pass": hard_pass, "gates": gates,
            "policy_status": policy_status, "expected_statistics": dict(EXPECTED_STATISTICS),
            "legacy_analysis_gates": analysis.get("gates"),
            "diagnostic_comparisons": _table(analysis, "paired_comparison", candidate_family),
            "assessment_scope": "historical_only_reused_data", "independent_forward_qualified": False,
            "promotion_eligible": False, "goal_complete": False, "provenance": dict(provenance)}
