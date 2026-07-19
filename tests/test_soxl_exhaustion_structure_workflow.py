from __future__ import annotations

import numpy as np
import pandas as pd


def test_split_keeps_all_2023_plus_soxl_out_of_development():
    from src.workflow.soxl_exhaustion_structure_fragility import split_family_development_outer

    frame = pd.DataFrame(
        {
            "instrument": ["SOXL", "SOXL", "TQQQ", "TQQQ"],
            "datetime": pd.to_datetime(["2022-12-30", "2023-01-03", "2022-12-30", "2023-01-03"]),
        }
    )
    development, structure_development, outer = split_family_development_outer(frame)
    assert development["instrument"].eq("SOXL").sum() == 0
    assert pd.to_datetime(development["datetime"]).max() <= pd.Timestamp("2022-12-31")
    assert structure_development["instrument"].eq("SOXL").all()
    assert pd.to_datetime(structure_development["datetime"]).max() <= pd.Timestamp("2022-12-31")
    assert outer["instrument"].eq("SOXL").all()
    assert pd.to_datetime(outer["datetime"]).min() >= pd.Timestamp("2023-01-01")


def test_conditional_lift_measures_increment_inside_price_quintiles():
    from src.workflow.soxl_exhaustion_structure_fragility import conditional_incremental_lift

    price = np.repeat(np.arange(5, dtype=float), 40)
    factor = np.tile(np.r_[np.zeros(20), np.ones(20)], 5)
    label = factor.copy()
    result = conditional_incremental_lift(
        pd.Series(label), pd.Series(price), pd.Series(factor), min_bin_observations=10
    )
    assert result["eligible_price_bins"] == 5
    assert result["conditional_event_rate_lift"] == 1.0


def test_intraday_coverage_gate_blocks_short_external_history_but_keeps_soxl_soxx():
    from src.workflow.soxl_exhaustion_structure_fragility import intraday_coverage_decision

    result = intraday_coverage_decision(
        core_start="2018-12-31",
        external_complete_start="2025-01-02",
        development_end="2022-12-31",
        outer_start="2023-01-01",
    )
    assert result["soxl_soxx_intraday"]["status"] == "eligible"
    assert result["external_bellwether_intraday"]["status"] == "blocked_insufficient_history"
    assert result["external_bellwether_intraday"]["included_in_selection"] is False


def test_family_decision_requires_increment_over_price_only_and_conditional_lift():
    from src.workflow.soxl_exhaustion_structure_fragility import evaluate_family_decision

    price = {"auc": 0.60, "average_precision_lift": 1.3, "quintile_lift": 0.08}
    stronger = {"auc": 0.64, "average_precision_lift": 1.5, "quintile_lift": 0.11}
    weaker = {"auc": 0.58, "average_precision_lift": 1.2, "quintile_lift": 0.06}
    kept = evaluate_family_decision(
        selected_factors=["x"], price_metrics=price, family_metrics=stronger, conditional_lift=0.02
    )
    rejected = evaluate_family_decision(
        selected_factors=["x"], price_metrics=price, family_metrics=weaker, conditional_lift=-0.01
    )
    assert kept["decision"] == "retained_incremental_factor_evidence"
    assert rejected["decision"] == "not_retained_no_incremental_factor_evidence"


def test_family_decision_rejects_an_eligible_outer_year_direction_reversal():
    from src.workflow.soxl_exhaustion_structure_fragility import evaluate_family_decision

    annual = pd.DataFrame(
        {
            "observations": [100, 100],
            "breakdown_events": [10, 10],
            "auc": [0.61, 0.44],
            "quintile_lift": [0.08, -0.03],
        }
    )
    result = evaluate_family_decision(
        selected_factors=["x"],
        price_metrics={"auc": 0.60, "average_precision_lift": 1.3, "quintile_lift": 0.08},
        family_metrics={"auc": 0.64, "average_precision_lift": 1.5, "quintile_lift": 0.11},
        conditional_lift=0.02,
        annual_metrics=annual,
    )
    assert result["gates"]["annual_direction_consistency"]["passed"] is False
    assert result["decision"] == "not_retained_no_incremental_factor_evidence"
