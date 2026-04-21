from __future__ import annotations

import pandas as pd


def _build_transition_v23_pack(context: dict[str, object]) -> dict[str, pd.Series]:
    anchor_gap_180 = context["anchor_gap_180"]
    below_anchor_ratio_63 = context["below_anchor_ratio_63"]
    relative_strength_21 = context["relative_strength_21"]
    regime_risk = context["regime_risk"]
    vix_zscore = context["vix_zscore"]
    downside_ratio_10 = context["downside_ratio_10"]
    trend_slope_10 = context["trend_slope_10"]

    vix_zscore_delta_5 = vix_zscore - vix_zscore.shift(5)
    downside_ratio_10_delta_5 = downside_ratio_10 - downside_ratio_10.shift(5)
    stress_day_mask = (
        (regime_risk > 0.60)
        | (vix_zscore > 0.75)
        | (downside_ratio_10 > 0.55)
    )
    days_above_stress_threshold_21 = stress_day_mask.astype(float).rolling(21).sum()
    stress_then_relief_flag = (
        (anchor_gap_180 < -0.02)
        & (days_above_stress_threshold_21 >= 5.0)
        & (vix_zscore_delta_5 < -0.35)
        & (downside_ratio_10_delta_5 < -0.05)
    ).astype(float)
    trend_restart_flag = (
        (anchor_gap_180 < -0.02)
        & (below_anchor_ratio_63 > 0.55)
        & (trend_slope_10 > 0.01)
        & (relative_strength_21 > 0.05)
        & (regime_risk < 0.60)
    ).astype(float)
    return {
        "vix_zscore_delta_5": vix_zscore_delta_5,
        "downside_ratio_10_delta_5": downside_ratio_10_delta_5,
        "days_above_stress_threshold_21": days_above_stress_threshold_21,
        "trend_slope_10": trend_slope_10,
        "stress_then_relief_flag": stress_then_relief_flag,
        "trend_restart_flag": trend_restart_flag,
    }


FEATURE_PACKS = {
    "transition_v23": {
        "columns": (
            "vix_zscore_delta_5",
            "downside_ratio_10_delta_5",
            "days_above_stress_threshold_21",
            "trend_slope_10",
            "stress_then_relief_flag",
            "trend_restart_flag",
        ),
        "builder": _build_transition_v23_pack,
    }
}
