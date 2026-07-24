from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.workflow.intraday_staged_recovery_direct_allocation import (
    EXPERIMENT,
    IntradayStagedRecoveryDirectAllocationSpec,
    apply_open_no_trade_band,
    time_limited_stage2_path,
)

ROOT = Path(__file__).resolve().parents[1]


def test_staged_spec_keeps_outer_out_of_selection():
    spec = IntradayStagedRecoveryDirectAllocationSpec(root=Path("/tmp/repo"))

    assert spec.experiment == EXPERIMENT
    assert spec.intraday_selection_years == (2019, 2020, 2021, 2022)
    assert spec.outer_years == (2023, 2024, 2025, 2026)
    assert spec.second_cut_bar_candidates == (6, 8)


def test_time_limited_stage2_path_restores_weights_before_overnight():
    value_at_cut = 1.02
    drifted_at_cut = np.array([0.10, 0.30, 0.60])
    action_weights = np.array([0.30, 0.30, 0.40])
    restore_weights = np.array([0.10, 0.30, 0.60])
    cut_to_close = np.array([0.12, 0.04, 0.0])
    overnight = np.array([-0.09, -0.03, 0.0])

    gross_return, turnover = time_limited_stage2_path(
        value_at_cut=value_at_cut,
        drifted_at_cut=drifted_at_cut,
        action_weights=action_weights,
        restore_weights=restore_weights,
        cut_to_close_returns=cut_to_close,
        overnight_returns=overnight,
    )

    close_growth = 1.0 + action_weights @ cut_to_close
    drifted_at_close = (
        action_weights * (1.0 + cut_to_close) / close_growth
    )
    expected_gross = (
        value_at_cut
        * close_growth
        * (1.0 + restore_weights @ overnight)
        - 1.0
    )
    expected_turnover = np.abs(
        action_weights[:2] - drifted_at_cut[:2]
    ).sum() + np.abs(
        restore_weights[:2] - drifted_at_close[:2]
    ).sum()

    assert gross_return == pytest.approx(expected_gross)
    assert turnover == pytest.approx(expected_turnover)


def test_time_limited_stage2_path_has_no_hidden_cash_turnover():
    weights = np.array([0.10, 0.30, 0.60])

    gross_return, turnover = time_limited_stage2_path(
        value_at_cut=1.0,
        drifted_at_cut=weights,
        action_weights=weights,
        restore_weights=weights,
        cut_to_close_returns=np.zeros(3),
        overnight_returns=np.zeros(3),
    )

    assert gross_return == pytest.approx(0.0)
    assert turnover == pytest.approx(0.0)


def test_open_no_trade_band_suppresses_only_small_legal_changes():
    previous = np.array([0.10, 0.30, 0.60])
    desired = np.array([0.102, 0.298, 0.60])

    weights, suppressed = apply_open_no_trade_band(
        desired,
        previous,
        beta_cap=0.65,
        no_trade_band=0.01,
    )
    unchanged, zero_delta_suppressed = apply_open_no_trade_band(
        previous,
        previous,
        beta_cap=0.65,
        no_trade_band=0.01,
    )

    assert suppressed is True
    assert weights == pytest.approx(previous)
    assert zero_delta_suppressed is False
    assert unchanged == pytest.approx(previous)


def test_open_no_trade_band_never_blocks_required_risk_reduction():
    previous = np.array([0.10, 0.30, 0.60])
    desired = np.array([0.098, 0.298, 0.604])

    weights, suppressed = apply_open_no_trade_band(
        desired,
        previous,
        beta_cap=0.55,
        no_trade_band=0.01,
    )

    assert suppressed is False
    assert weights == pytest.approx(desired)


def test_staged_recovery_card_is_executed_recorder_first_and_outer_safe():
    card = ROOT / "research" / EXPERIMENT
    summary = json.loads((card / "summary.json").read_text(encoding="utf-8"))
    notebook = (card / f"{EXPERIMENT}.ipynb").read_text(encoding="utf-8")

    assert summary["decision"] == (
        "promoted_research_candidate_staged_intraday_recovery"
    )
    assert summary["selection_uses_outer_test"] is False
    assert summary["outer_status"] == (
        "reused_diagnostic_not_fresh_unseen_test"
    )
    assert summary["selected_model"]["second_cut_time"] == "11:00"
    assert summary["selected_model"]["stage2_action_beta"] == 1.2
    assert summary["outer_diagnostic_metrics_10bps"][
        "max_drawdown_pct"
    ] >= -30.0
    assert summary["fixed_decision_stress_metrics_25bps"][
        "max_drawdown_pct"
    ] >= -30.0
    assert summary["fixed_decision_stress_metrics_25bps"][
        "worst_rolling_5d_pct"
    ] >= -12.0
    assert summary["stage2_effect"]["outer_stage2_action_count"] == 7
    assert summary["artifact_policy"]["policy"] == (
        "slim_card_recorder_first"
    )
    assert summary["recorder_id"] in notebook
    recorder_dir = Path(summary["recorder_local_dir"])
    artifact_dir = recorder_dir / "artifacts" / EXPERIMENT
    for name in (
        "positions.csv",
        "strategy_returns.csv",
        "stage1_recovery_training_frame.csv",
        "stage2_recovery_training_frame.csv",
        "model_selection_full.csv",
    ):
        assert (artifact_dir / name).exists(), name
    for root_only in (
        "positions.csv",
        "strategy_returns.csv",
        "stage1_recovery_training_frame.csv",
        "stage2_recovery_training_frame.csv",
        "model_selection_full.csv",
    ):
        assert not (card / root_only).exists(), root_only
