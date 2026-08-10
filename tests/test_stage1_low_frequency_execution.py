from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_materiality_band_suppresses_only_small_non_risk_reducing_changes():
    from src.workflow.stage1_low_frequency_execution import apply_materiality_band

    current = np.array([0.20, 0.50, 0.30])
    desired = np.array([0.21, 0.51, 0.28])
    effective, suppressed = apply_materiality_band(
        desired,
        current,
        beta_cap=1.2,
        minimum_rebalance_turnover=0.05,
        minimum_cash_weight=0.15,
        maximum_soxl_weight=0.30,
    )

    assert suppressed is True
    assert effective == pytest.approx(current)


def test_materiality_band_is_symmetric_but_never_suppresses_illegal_holdings():
    from src.workflow.stage1_low_frequency_execution import apply_materiality_band

    current = np.array([0.20, 0.50, 0.30])
    lower_risk = np.array([0.19, 0.49, 0.32])
    effective, suppressed = apply_materiality_band(
        lower_risk,
        current,
        beta_cap=1.2,
        minimum_rebalance_turnover=0.05,
        minimum_cash_weight=0.15,
        maximum_soxl_weight=0.30,
    )
    assert suppressed is True
    assert effective == pytest.approx(current)

    illegal = np.array([0.31, 0.54, 0.15])
    legal = np.array([0.30, 0.55, 0.15])
    effective, suppressed = apply_materiality_band(
        legal,
        illegal,
        beta_cap=1.45,
        minimum_rebalance_turnover=0.05,
        minimum_cash_weight=0.15,
        maximum_soxl_weight=0.30,
    )
    assert suppressed is False
    assert effective == pytest.approx(legal)


def test_low_frequency_candidate_selection_excludes_outer_and_requires_real_reduction():
    from src.workflow.stage1_low_frequency_execution import (
        select_low_frequency_candidate,
    )

    candidates = pd.DataFrame(
        [
            {
                "minimum_rebalance_turnover": 0.03,
                "selection_uses_outer_test": False,
                "selection_gate_pass": True,
                "trade_day_reduction_vs_baseline": 0.36,
                "terminal_wealth_ratio_vs_baseline": 1.00,
            },
            {
                "minimum_rebalance_turnover": 0.05,
                "selection_uses_outer_test": False,
                "selection_gate_pass": True,
                "trade_day_reduction_vs_baseline": 0.42,
                "terminal_wealth_ratio_vs_baseline": 0.995,
            },
            {
                "minimum_rebalance_turnover": 0.10,
                "selection_uses_outer_test": True,
                "selection_gate_pass": True,
                "trade_day_reduction_vs_baseline": 0.70,
                "terminal_wealth_ratio_vs_baseline": 1.20,
            },
        ]
    )

    selected = select_low_frequency_candidate(candidates)

    assert selected["minimum_rebalance_turnover"] == pytest.approx(0.05)
    assert bool(selected["selection_uses_outer_test"]) is False


def test_low_frequency_card_rejects_fake_execution_and_predictive_promotions():
    card = ROOT / "research" / "stage1_low_frequency_execution_v1"
    summary = json.loads((card / "summary.json").read_text(encoding="utf-8"))
    selected = json.loads(
        (card / "selected_model.json").read_text(encoding="utf-8")
    )
    notebook = json.loads(
        (card / "stage1_low_frequency_execution_v1.ipynb").read_text(
            encoding="utf-8"
        )
    )

    assert summary["selection_uses_outer_test"] is False
    assert summary["decision"] == (
        "not_promoted_no_low_frequency_candidate_preserves_performance"
    )
    assert summary["selected_execution_policy"] is None
    assert summary["development"]["trade_day_reduction"] < 0.35
    assert summary["development"]["terminal_wealth_ratio"] >= 0.99
    assert selected["predictive_model_changed"] is False
    assert selected["minimum_rebalance_turnover"] is None
    assert selected["diagnostic_minimum_rebalance_turnover"] == pytest.approx(
        0.0025
    )
    assert selected["required_risk_reduction_bypass"] is True
    assert selected["deployment"] == "rejected_not_wired_to_simulator"
    assert notebook["nbformat"] == 4
    assert "真实交易频率" in json.dumps(notebook, ensure_ascii=False)
