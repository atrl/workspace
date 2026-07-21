from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.workflow.smart_leverage_soxl_wealth_control import (
    asymmetric_wealth_multiplier,
    beta_preserving_soxl_soxx_ladder,
    simulate_wealth_control,
)


ROOT = Path(__file__).resolve().parents[1]


def _frame() -> pd.DataFrame:
    dates = pd.date_range("2023-01-03", periods=4, freq="B")
    return pd.DataFrame(
        {
            "signal_date": dates,
            "trade_date": dates + pd.offsets.BDay(1),
            "exit_date": dates + pd.offsets.BDay(2),
            "soxl_return": [0.10, -0.05, 0.02, -0.01],
            "soxx_return": [0.03, -0.02, 0.01, -0.005],
        }
    )


def test_asymmetric_wealth_multiplier_only_de_risks_when_ahead() -> None:
    assert asymmetric_wealth_multiplier(-0.20, target_surplus=0.0, strength=2.0, floor=0.4) == 1.0
    assert asymmetric_wealth_multiplier(0.0, target_surplus=0.0, strength=2.0, floor=0.4) == 1.0
    assert 0.4 <= asymmetric_wealth_multiplier(0.30, target_surplus=0.0, strength=2.0, floor=0.4) < 1.0
    assert asymmetric_wealth_multiplier(100.0, target_surplus=0.0, strength=2.0, floor=0.4) == 0.4


@pytest.mark.parametrize("target", [0.0, 0.1, 1.0 / 3.0, 0.5, 0.8, 1.0])
def test_beta_ladder_is_feasible_and_preserves_effective_beta(target: float) -> None:
    soxl, soxx = beta_preserving_soxl_soxx_ladder(target)
    assert soxl >= 0.0
    assert soxx >= 0.0
    assert soxl + soxx <= 1.0 + 1e-12
    assert 3.0 * soxl + soxx == pytest.approx(3.0 * target)


def test_wealth_control_uses_prior_wealth_and_replays_clean_returns() -> None:
    result = simulate_wealth_control(
        _frame(),
        np.full(4, 0.5),
        cost_bps=10,
        allocation_mode="soxl_only",
        wealth_lock=True,
        benchmark_soxx_weight=0.7,
        wealth_lock_strength=1.0,
        wealth_lock_floor=0.5,
        target_surplus=0.0,
    )
    assert result.loc[0, "wealth_multiplier"] == 1.0
    assert result.loc[0, "strategy_return"] == pytest.approx(0.5 * 0.10 - 0.5 * 0.001)
    assert result.loc[1, "wealth_multiplier"] < 1.0
    assert (result["wealth_multiplier"] <= 1.0).all()
    assert (result["cash_weight"] >= -1e-12).all()


def test_research_card_is_slim_recorder_first_and_outer_is_evaluation_only() -> None:
    card = ROOT / "research" / "smart_leverage_soxl_wealth_control_v1"
    summary = json.loads((card / "summary.json").read_text(encoding="utf-8"))
    notebook = (card / "smart_leverage_soxl_wealth_control_v1.ipynb").read_text(encoding="utf-8")

    assert summary["selection_uses_outer_test"] is False
    assert summary["source_opportunity_audit"]["selection_uses_outer_test"] is False
    assert summary["source_opportunity_audit"]["ignored_source_return_fields"] == [
        "strategy_return",
        "open_to_open_return",
    ]
    assert summary["artifact_policy"]["policy"] == "slim_card_recorder_first"
    assert summary["required_cost_bps"] == 10
    assert summary["stress_cost_bps"] == 25
    assert summary["required_cost_pass_count"] == 0
    assert summary["primary_candidate"]["all_required_gates_pass"] is False
    assert not (card / "positions.csv").exists()
    assert not (card / "strategy_returns.csv").exists()
    assert summary["recorder_id"] in notebook
    assert "run_smart_leverage_soxl_wealth_control_card" in notebook
    assert "2023–2026 outer test 只评估" in notebook
