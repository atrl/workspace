from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.factor import (
    EMPIRICAL_TAIL_FEATURES,
    IMPLIED_TAIL_FEATURES,
    STRUCTURAL_FRAGILITY_FEATURES,
)
from src.workflow.role_separated_direct_allocation import (
    CANDIDATES,
    HEAD_COLUMNS,
    DirectHeadAllocator,
    _attach_role_targets_and_deltas,
    _factorized_weights,
    _feature_roles,
)


ROOT = Path(__file__).resolve().parents[1]
CARD = ROOT / "research" / "role_separated_direct_allocation_v1"


def _frame(rows: int = 120) -> pd.DataFrame:
    index = pd.date_range("2020-01-02", periods=rows, freq="B")
    phase = np.arange(rows, dtype=float)
    soxx = 0.0005 + 0.01 * np.sin(phase / 9.0)
    soxl = 3.0 * soxx - 3.0 * soxx**2
    frame = pd.DataFrame(
        {
            "soxx_forward_open_return": soxx,
            "soxl_forward_open_return": soxl,
        },
        index=index,
    )
    for feature in {
        *EMPIRICAL_TAIL_FEATURES,
        *IMPLIED_TAIL_FEATURES,
        *STRUCTURAL_FRAGILITY_FEATURES,
    }:
        frame[feature] = np.sin(phase / 17.0)
    return frame


def test_role_targets_do_not_change_before_appended_future_boundary() -> None:
    full = _frame()
    short = full.iloc[:100].copy()
    built_short = _attach_role_targets_and_deltas(short)
    built_full = _attach_role_targets_and_deltas(full)
    columns = [
        "future_downside_risk_20d",
        "future_relative_utility_20d",
    ]
    pd.testing.assert_frame_equal(
        built_short.iloc[:-20].loc[:, columns],
        built_full.loc[built_short.index[:-20], columns],
    )


def test_feature_roles_keep_slow_tail_state_out_of_alpha() -> None:
    frame = _frame()
    for feature in (
        "soxx_vol20",
        "soxx_trend60",
        "vix_z_60",
        "soxl_ret20",
    ):
        frame[feature] = np.linspace(-1.0, 1.0, len(frame))
    frame = _attach_role_targets_and_deltas(frame)
    roles = _feature_roles(frame)
    assert set(EMPIRICAL_TAIL_FEATURES).issubset(roles["hazard"])
    assert set(STRUCTURAL_FRAGILITY_FEATURES).issubset(roles["hazard"])
    assert set(EMPIRICAL_TAIL_FEATURES).isdisjoint(roles["alpha"])
    assert set(IMPLIED_TAIL_FEATURES).isdisjoint(roles["alpha"])
    assert any("delta_" in feature for feature in roles["recovery"])


def test_allocator_candidates_are_fixed_role_ablation() -> None:
    assert [candidate.name for candidate in CANDIDATES] == [
        "crossfit_heads_direct_mse",
        "crossfit_alpha_recovery_direct_mse",
        "crossfit_heads_router_mse",
        "crossfit_alpha_recovery_router_mse",
        "crossfit_alpha_recovery_router_utility",
    ]
    assert CANDIDATES[0].router is False
    assert CANDIDATES[1].router is False
    assert all(candidate.router for candidate in CANDIDATES[2:])
    assert CANDIDATES[-1].objective == "relative_wealth_cvar"
    assert all(
        not candidate.use_hazard
        for candidate in (CANDIDATES[1], CANDIDATES[3], CANDIDATES[4])
    )


def test_factorized_weights_are_legal_and_direct() -> None:
    logits = torch.tensor(
        [[-20.0, -20.0], [0.0, 0.0], [20.0, 20.0]]
    )
    weights = _factorized_weights(logits, 0.30).detach().numpy()
    assert np.allclose(weights.sum(axis=1), 1.0)
    assert np.all(weights >= 0.0)
    assert np.all(weights[:, 0] <= 0.30 + 1e-8)


def test_direct_allocator_only_accepts_three_crossfit_heads() -> None:
    model = DirectHeadAllocator(hidden_units=8)
    features = torch.zeros((5, len(HEAD_COLUMNS)))
    weights, _ = model(features, 0.30)
    assert weights.shape == (5, 3)
    assert torch.allclose(weights.sum(dim=1), torch.ones(5))


def test_role_separated_card_contract_after_execution() -> None:
    if not (CARD / "summary.json").exists():
        return
    summary = json.loads(
        (CARD / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["artifact_policy"]["policy"] == (
        "slim_card_recorder_first"
    )
    assert summary["architecture"]["base_position"] is None
    assert summary["architecture"]["candidate_weight_grid"] is False
    assert summary["development"]["selection_uses_outer_test"] is False
    assert summary["architecture"]["head_training"] == (
        "expanding_window_cross_fitted"
    )
    sensitivity = summary["development"]["seed_sensitivity"]
    assert len(sensitivity) == 5
    assert all(
        row["selection_uses_outer_test"] is False
        for row in sensitivity
    )
    root_files = {
        path.name for path in CARD.iterdir() if path.is_file()
    }
    assert "predictions.csv" not in root_files
    assert "positions.csv" not in root_files
    assert "strategy_returns.csv" not in root_files
