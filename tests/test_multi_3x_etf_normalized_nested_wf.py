from __future__ import annotations

import pandas as pd

from src.workflow.multi_3x_etf_normalized_nested_wf import (
    EXPERIMENT,
    _attach_normalized_features,
    _overfit_audit_summary,
    _select_candidate_from_inner_metrics,
)


def test_normalized_features_are_past_only_and_do_not_encode_asset_identity():
    frame = pd.DataFrame(
        {
            "instrument": ["SOXL"] * 8 + ["TQQQ"] * 8,
            "datetime": list(pd.date_range("2024-01-02", periods=8, freq="B")) * 2,
            "open": [10, 11, 12, 13, 12, 13, 14, 15, 20, 21, 22, 23, 24, 23, 24, 25],
            "high": [11, 12, 13, 14, 13, 14, 15, 16, 21, 22, 23, 24, 25, 24, 25, 26],
            "low": [9, 10, 11, 12, 11, 12, 13, 14, 19, 20, 21, 22, 23, 22, 23, 24],
            "close": [10, 11, 12, 13, 12, 13, 14, 15, 20, 21, 22, 23, 24, 23, 24, 25],
            "volume": [100, 110, 120, 130, 140, 150, 160, 170, 200, 210, 220, 230, 240, 250, 260, 270],
            "underlying_close": [5, 5.1, 5.2, 5.3, 5.2, 5.3, 5.4, 5.5, 10, 10.2, 10.4, 10.6, 10.8, 10.7, 10.9, 11.0],
            "vix": [18, 17, 16, 15, 20, 19, 18, 17, 18, 17, 16, 15, 20, 19, 18, 17],
        }
    )

    normalized = _attach_normalized_features(frame, windows=(3,), min_periods=2)
    early = normalized.loc[
        (normalized["instrument"] == "SOXL") & (normalized["datetime"] == pd.Timestamp("2024-01-09")),
        "ret_3_z3",
    ].iloc[0]

    modified = frame.copy()
    modified.loc[modified["datetime"] > pd.Timestamp("2024-01-09"), "close"] *= 10.0
    modified_normalized = _attach_normalized_features(modified, windows=(3,), min_periods=2)
    modified_early = modified_normalized.loc[
        (modified_normalized["instrument"] == "SOXL")
        & (modified_normalized["datetime"] == pd.Timestamp("2024-01-09")),
        "ret_3_z3",
    ].iloc[0]

    assert early == modified_early
    assert {"ret_3_z3", "ret_3_pctile3", "tracking_error_3_z3"}.issubset(normalized.columns)
    feature_cols = [column for column in normalized.columns if column.endswith("_z3") or column.endswith("_pctile3")]
    assert "instrument" not in feature_cols
    assert all("symbol" not in column.lower() and "asset" not in column.lower() for column in feature_cols)


def test_select_candidate_from_inner_metrics_requires_leave_one_asset_out_and_no_outer_test():
    metrics = pd.DataFrame(
        [
            {
                "candidate": "lgbm_loose",
                "inner_year": 2021,
                "heldout_asset": "SOXL",
                "selection_return_pct": 12.0,
                "selection_max_drawdown_pct": -12.0,
                "selection_total_turnover": 4.0,
            },
            {
                "candidate": "lgbm_loose",
                "inner_year": 2021,
                "heldout_asset": "TQQQ",
                "selection_return_pct": 8.0,
                "selection_max_drawdown_pct": -9.0,
                "selection_total_turnover": 5.0,
            },
            {
                "candidate": "xgb_overfit",
                "inner_year": 2021,
                "heldout_asset": "SOXL",
                "selection_return_pct": 30.0,
                "selection_max_drawdown_pct": -35.0,
                "selection_total_turnover": 2.0,
            },
            {
                "candidate": "xgb_overfit",
                "inner_year": 2021,
                "heldout_asset": "TQQQ",
                "selection_return_pct": 25.0,
                "selection_max_drawdown_pct": -40.0,
                "selection_total_turnover": 2.0,
            },
        ]
    )

    selected = _select_candidate_from_inner_metrics(metrics, outer_test_year=2024, required_heldout_assets=("SOXL", "TQQQ"))

    assert selected["candidate"] == "lgbm_loose"
    assert selected["selection_uses_outer_test"] is False
    assert selected["required_heldout_assets"] == ["SOXL", "TQQQ"]
    assert selected["outer_test_year"] == 2024


def test_overfit_audit_rejects_manual_templates_and_accepts_nested_multi_asset_design():
    rejected = _overfit_audit_summary(
        experiment=EXPERIMENT,
        manual_template_selection=True,
        nested_walk_forward=False,
        leave_one_asset_out_assets=("SOXL",),
        feature_identity_columns=("instrument_code",),
        selection_uses_outer_test=True,
    )
    accepted = _overfit_audit_summary(
        experiment=EXPERIMENT,
        manual_template_selection=False,
        nested_walk_forward=True,
        leave_one_asset_out_assets=("SOXL", "TQQQ", "UPRO", "SPXL", "TECL"),
        feature_identity_columns=(),
        selection_uses_outer_test=False,
    )

    assert rejected["overfit_risk_status"] == "fail"
    assert "manual_template_selection" in rejected["blocking_reasons"]
    assert "asset_identity_feature_leakage" in rejected["blocking_reasons"]
    assert accepted["overfit_risk_status"] == "mitigated"
    assert accepted["blocking_reasons"] == ["ok"]
