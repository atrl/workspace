from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.workflow.intraday_recovery_direct_allocation import (
    apply_fixed_decision_cost_stress,
    project_legal_weights,
    select_development_candidate,
)


def test_project_legal_weights_enforces_three_asset_constraints():
    weights = project_legal_weights(0.30, 0.55, 0.60)

    assert weights.sum() == pytest.approx(1.0)
    assert weights[0] <= 0.30
    assert weights[2] >= 0.15
    assert np.all(weights >= 0.0)
    assert 3.0 * weights[0] + weights[1] <= 0.60 + 1e-12


def test_selection_ignores_outer_even_when_outer_return_is_higher():
    candidates = pd.DataFrame(
        [
            {
                "segment": "development_walk_forward",
                "selection_uses_outer_test": False,
                "risk_window_pass_count": 7,
                "positive_relative_window_count": 6,
                "relative_ratio_p25": 1.01,
                "relative_ratio_median": 1.05,
                "terminal_wealth": 5.0,
                "candidate": "development_only",
            },
            {
                "segment": "outer_diagnostic",
                "selection_uses_outer_test": True,
                "risk_window_pass_count": 7,
                "positive_relative_window_count": 7,
                "relative_ratio_p25": 2.0,
                "relative_ratio_median": 2.0,
                "terminal_wealth": 50.0,
                "candidate": "outer_leak",
            },
        ]
    )

    selected = select_development_candidate(candidates)

    assert selected["candidate"] == "development_only"


def test_fixed_decision_stress_changes_cost_not_decisions():
    gross = np.array([0.02, -0.01, 0.03])
    turnover = np.array([0.5, 0.0, 0.25])

    stressed = apply_fixed_decision_cost_stress(gross, turnover, cost_bps=25.0)

    np.testing.assert_allclose(stressed, gross - turnover * 0.0025)
    with pytest.raises(ValueError):
        apply_fixed_decision_cost_stress(gross, turnover[:2], cost_bps=25.0)
