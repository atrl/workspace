from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.workflow.pit_constituent_auxiliary_training_allocation import (
    CANDIDATES,
    PitAuxiliaryAllocationSpec,
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
