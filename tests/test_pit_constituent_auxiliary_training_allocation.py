from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.workflow.pit_constituent_auxiliary_training_allocation import (
    DRAWDOWN_CANDIDATES,
    CANDIDATES,
    AuxiliaryTrainingCandidate,
    PitAuxiliaryAllocationSpec,
    _apply_drawdown_budget,
    _allocation_loss,
    _project_legal_weights,
    _selection_table,
    _smooth_weights,
)


ROOT = Path(__file__).resolve().parents[1]


def test_projection_emits_direct_legal_weights_with_cash_floor():
    logits = torch.tensor(
        [[10.0, -10.0, -10.0], [-4.0, 3.0, -2.0]],
        dtype=torch.float32,
    )
    weights = _project_legal_weights(
        logits,
        maximum_soxl_weight=0.30,
        minimum_cash_weight=0.15,
    ).detach().numpy()

    assert np.allclose(weights.sum(axis=1), 1.0)
    assert (weights >= 0.0).all()
    assert (weights[:, 0] <= 0.300001).all()
    assert (weights[:, 2] >= 0.149999).all()


def test_smoothing_preserves_legal_simplex_and_reduces_small_changes():
    spec = PitAuxiliaryAllocationSpec(
        root=ROOT,
        smoothing_span=1,
        rebalance_threshold=0.05,
    )
    prediction = pd.DataFrame(
        {
            "soxl_weight": [0.20, 0.21, 0.28],
            "soxx_weight": [0.60, 0.59, 0.52],
            "cash_weight": [0.20, 0.20, 0.20],
            "tail_probability": [0.1, 0.2, 0.3],
        },
        index=pd.date_range("2020-01-01", periods=3),
    )

    result = _smooth_weights(prediction, spec=spec)

    assert np.allclose(
        result[["soxl_weight", "soxx_weight", "cash_weight"]].sum(
            axis=1
        ),
        1.0,
    )
    assert result.iloc[1]["soxl_weight"] == result.iloc[0]["soxl_weight"]
    assert result.iloc[2]["soxl_weight"] != result.iloc[1]["soxl_weight"]


def test_selection_is_development_only_and_rewards_matched_aux_increment():
    comparison_rows = []
    annual_rows = []
    for candidate in CANDIDATES:
        ratio10 = 1.10
        ratio25 = 1.06
        if candidate.name == "pit_plus_union_aux_005":
            ratio10 = 1.25
            ratio25 = 1.15
        for cost, ratio in [(10, ratio10), (25, ratio25)]:
            comparison_rows.append(
                {
                    "segment": "development_walk_forward",
                    "candidate": candidate.name,
                    "cost_bps": cost,
                    "selection_uses_outer_test": False,
                    "terminal_wealth_ratio_vs_soxx": ratio,
                    "max_drawdown_pct": -18.0,
                    "worst_day_pct": -10.0,
                    "cvar5_pct": -6.0,
                    "worst_rolling_5d_pct": -11.0,
                    "avg_cash_weight_pct": 20.0,
                    "upside_capture_pct": 65.0,
                }
            )
        for year in range(2016, 2023):
            annual_rows.append(
                {
                    "candidate": candidate.name,
                    "cost_bps": 10,
                    "fold_year": year,
                    "terminal_wealth_ratio_vs_soxx": (
                        1.02
                        if candidate.name == "pit_plus_union_aux_005"
                        or year < 2020
                        else 0.98
                    ),
                }
            )

    selection = _selection_table(
        pd.DataFrame(comparison_rows), pd.DataFrame(annual_rows)
    )
    selected = selection.loc[selection["selected"]].iloc[0]

    assert selected["candidate"] == "pit_plus_union_aux_005"
    assert bool(selected["relative_objective_pass"])
    assert not bool(selected["selection_uses_outer_test"])


def test_drawdown_loss_penalizes_path_breach_and_low_cash_on_tail_days():
    weights = torch.tensor(
        [[0.30, 0.55, 0.15]] * 10,
        dtype=torch.float32,
    )
    returns = torch.tensor(
        [[-0.15, -0.06]] * 5 + [[0.02, 0.01]] * 5,
        dtype=torch.float32,
    )
    years = torch.tensor([2020] * 10, dtype=torch.int64)
    cash_target = torch.tensor([0.55] * 5 + [0.0] * 5)
    tail_event = torch.tensor([1.0] * 5 + [0.0] * 5)
    candidate = AuxiliaryTrainingCandidate(
        name="risk",
        auxiliary_weight=0.05,
        include_union_backfill=False,
        absolute_drawdown_penalty=0.10,
        conditional_drawdown_penalty=0.08,
        rolling_5d_penalty=0.20,
        cash_target_penalty=0.10,
        tail_event_weight_multiplier=8.0,
        stress_cash_ceiling=0.85,
    )

    _, _, _, components = _allocation_loss(
        weights=weights,
        returns=returns,
        years=years,
        cash_target=cash_target,
        tail_event=tail_event,
        candidate=candidate,
        drawdown_target_pct=-30.0,
        worst_5d_target_pct=-15.0,
        minimum_cash_weight=0.15,
        cost_bps=10,
        turnover_weight=0.006,
    )

    assert float(components["max_drawdown_loss"]) > 0.0
    assert float(components["conditional_drawdown_loss"]) > 0.0
    assert float(components["rolling_5d_loss"]) > 0.0
    assert float(components["cash_target_loss"]) > 0.0


def test_drawdown_research_selection_prioritizes_minus_30pct_gate():
    comparison_rows = []
    annual_rows = []
    for candidate in DRAWDOWN_CANDIDATES:
        is_safe = candidate.name == "pit_cdar_cash_fast_010"
        for cost in [10, 25]:
            comparison_rows.append(
                {
                    "segment": "development_walk_forward",
                    "candidate": candidate.name,
                    "cost_bps": cost,
                    "selection_uses_outer_test": False,
                    "terminal_wealth_ratio_vs_soxx": (
                        1.05 if is_safe else 1.40
                    ),
                    "max_drawdown_pct": -28.0 if is_safe else -45.0,
                    "worst_day_pct": -9.0,
                    "cvar5_pct": -6.0,
                    "worst_rolling_5d_pct": (
                        -13.0 if is_safe else -20.0
                    ),
                    "avg_cash_weight_pct": 30.0,
                    "upside_capture_pct": 50.0,
                }
            )
        for year in range(2016, 2023):
            annual_rows.append(
                {
                    "candidate": candidate.name,
                    "cost_bps": 10,
                    "fold_year": year,
                    "terminal_wealth_ratio_vs_soxx": 1.01,
                }
            )

    selection = _selection_table(
        pd.DataFrame(comparison_rows),
        pd.DataFrame(annual_rows),
        candidates=DRAWDOWN_CANDIDATES,
        research_max_drawdown_target_pct=-30.0,
        research_worst_5d_target_pct=-15.0,
    )
    selected = selection.loc[selection["selected"]].iloc[0]

    assert selected["candidate"] == "pit_cdar_cash_fast_010"
    assert bool(selected["research_target_pass"])


def test_drawdown_budget_is_causal_and_preserves_legal_weights():
    index = pd.date_range("2020-01-01", periods=6)
    prediction = pd.DataFrame(
        {
            "soxl_weight": [0.30] * 6,
            "soxx_weight": [0.55] * 6,
            "cash_weight": [0.15] * 6,
        },
        index=index,
    )
    market = pd.DataFrame(
        {
            "soxl_forward_open_return": [0.0, -0.20, -0.20, 0.0, 0.0, 0.0],
            "soxx_forward_open_return": [0.0, -0.08, -0.08, 0.0, 0.0, 0.0],
        },
        index=index,
    )
    candidate = AuxiliaryTrainingCandidate(
        name="budget",
        auxiliary_weight=0.05,
        include_union_backfill=False,
        drawdown_floor=0.70,
        drawdown_budget_multiplier=2.0,
    )

    result = _apply_drawdown_budget(
        market=market,
        prediction=prediction,
        candidate=candidate,
        cost_bps=10,
    )
    weights = result[
        ["soxl_weight", "soxx_weight", "cash_weight"]
    ].to_numpy()

    assert np.allclose(weights.sum(axis=1), 1.0)
    assert (weights >= 0.0).all()
    assert np.isclose(
        weights[0, :2].sum(),
        weights[1, :2].sum(),
        atol=0.002,
    )
    assert weights[3, :2].sum() < weights[2, :2].sum()

    changed_future = market.copy()
    changed_future.iloc[-1, :] = 0.50
    future_result = _apply_drawdown_budget(
        market=changed_future,
        prediction=prediction,
        candidate=candidate,
        cost_bps=10,
    )
    assert np.allclose(
        result[["soxl_weight", "soxx_weight", "cash_weight"]],
        future_result[["soxl_weight", "soxx_weight", "cash_weight"]],
    )
