import copy
import hashlib
import json

import numpy as np
import pandas as pd

from src.analysis.soxx_objective import EXPECTED_STATISTICS, evaluate_soxx_objective


def _evidence():
    family = "candidate"
    costs = [0, 2, 5]
    metrics = pd.DataFrame([{"family": family, "slippage_bps": cost, "relative_wealth": 1.03,
        "close_max_drawdown_pct": -25., "sampled_point_max_drawdown_pct": -29.,
        "cvar5_pct": -5., "benchmark_cvar5_pct": -6.} for cost in costs])
    pairs = pd.DataFrame([{"family": family, "comparator": comp, "slippage_bps": cost,
        "relative_wealth": (1.03 if comp == "SOXX" else .85)}
        for cost in costs for comp in ("SOXX", "strong_reference")])
    annual = pd.DataFrame([{"family": family, "kind": "year", "slippage_bps": cost,
        "period": str(year), "complete_period": True,
        "relative_wealth": (1.01 if year < 7 else .99)}
        for cost in costs for year in range(10)])
    rolling = pd.DataFrame([{"family": family, "comparator": "SOXX", "slippage_bps": cost,
        "window_sessions": 126, "windows": 100, "positive_fraction": .70} for cost in costs])
    uncertainty = pd.DataFrame([{"family": family, "comparator": "SOXX", "slippage_bps": cost,
        "daily_lower_bound": .0001, "lower_bound_positive": True, "mean_block": 63,
        "resamples": 10000, "random_seed": 20261014, "alpha": .025 / 15,
        } for cost in costs])
    policy = {"contract_version": 2, "candidate_family": family, "phase": "P1",
              "statistical_settings": dict(EXPECTED_STATISTICS)}
    policy_hash = hashlib.sha256(json.dumps(policy, sort_keys=True, separators=(",", ":"),
                                      allow_nan=False).encode()).hexdigest()
    provenance = {"candidate_family": family, "phase": "P1",
        "receipt_refs": [{"path": "P1/summary.json", "sha256": "a" * 64}],
        "frozen_plan_sha256": "b" * 64, "frozen_plan_policy_sha256": policy_hash,
        "objective_policy": policy}
    analysis = {"metrics": metrics, "paired_comparison": pairs, "annual_metrics": annual,
        "rolling_summary": rolling, "uncertainty": uncertainty,
        "gates": {"legacy_strong_reference": False, "legacy_rule": "unchanged"}}
    return analysis, provenance


def test_v2_historical_soxx_gates_are_cost_complete_and_never_promotion():
    analysis, provenance = _evidence()
    result = evaluate_soxx_objective(analysis, candidate_family="candidate", provenance=provenance)
    assert result["status"] == "historical_pass"
    assert result["historical_pass"]
    assert result["diagnostic_comparisons"].loc[
        result["diagnostic_comparisons"].comparator.eq("strong_reference"), "relative_wealth"].lt(1).all()
    assert result["legacy_analysis_gates"] is analysis["gates"]
    assert not result["independent_forward_qualified"]
    assert not result["promotion_eligible"] and not result["goal_complete"]

    no_policy = copy.deepcopy(provenance)
    no_policy.pop("objective_policy")
    needs_policy = evaluate_soxx_objective(analysis, candidate_family="candidate", provenance=no_policy)
    assert needs_policy["status"] == "needs_policy" and not needs_policy["historical_pass"]

    missing_cost = copy.deepcopy(analysis)
    missing_cost["metrics"] = missing_cost["metrics"].loc[missing_cost["metrics"].slippage_bps.ne(2)]
    assert not evaluate_soxx_objective(missing_cost, candidate_family="candidate",
                                      provenance=provenance)["gates"]["one_candidate_row_per_cost"]

    nonfinite = copy.deepcopy(analysis)
    nonfinite["metrics"].loc[2, "sampled_point_max_drawdown_pct"] = np.inf
    assert not evaluate_soxx_objective(nonfinite, candidate_family="candidate",
                                      provenance=provenance)["gates"]["finite_required_metrics"]

    no_complete_years = copy.deepcopy(analysis)
    no_complete_years["annual_metrics"]["complete_period"] = False
    assert not evaluate_soxx_objective(no_complete_years, candidate_family="candidate",
                                      provenance=provenance)["gates"]["two_complete_years_each_cost"]

    duplicate_year = copy.deepcopy(analysis)
    duplicate_year["annual_metrics"] = pd.concat([duplicate_year["annual_metrics"],
        duplicate_year["annual_metrics"].iloc[[0]]], ignore_index=True)
    assert not evaluate_soxx_objective(duplicate_year, candidate_family="candidate",
        provenance=provenance)["gates"]["two_complete_years_each_cost"]

    negative_ci = copy.deepcopy(analysis)
    negative_ci["uncertainty"].loc[0, "daily_lower_bound"] = -.0001
    negative_ci["uncertainty"].loc[0, "lower_bound_positive"] = False
    assert not evaluate_soxx_objective(negative_ci, candidate_family="candidate",
        provenance=provenance)["gates"]["soxx_ci_lower_positive"]
    missing_ci = copy.deepcopy(analysis)
    missing_ci["uncertainty"] = missing_ci["uncertainty"].drop(columns="daily_lower_bound")
    assert not evaluate_soxx_objective(missing_ci, candidate_family="candidate",
        provenance=provenance)["gates"]["soxx_ci_lower_positive"]

    inconsistent_wealth = copy.deepcopy(analysis)
    inconsistent_wealth["paired_comparison"].loc[
        inconsistent_wealth["paired_comparison"].comparator.eq("SOXX") &
        inconsistent_wealth["paired_comparison"].slippage_bps.eq(2), "relative_wealth"] = 1.02
    assert not evaluate_soxx_objective(inconsistent_wealth, candidate_family="candidate",
        provenance=provenance)["gates"]["candidate_soxx_wealth_consistent"]

    invalid_rolling = copy.deepcopy(analysis)
    invalid_rolling["rolling_summary"].loc[0, "windows"] = 0
    assert not evaluate_soxx_objective(invalid_rolling, candidate_family="candidate",
        provenance=provenance)["gates"]["soxx_rolling_126_positive_70pct"]
