import numpy as np
import pytest

from src.backtest.position_accounting import settle_target_session


def test_open_fee_reduces_cash_before_next_weights_are_normalized():
    result = settle_target_session(
        [0, 0, 1], open_target_weights=[0.3, 0.5, 0.2],
        asset_returns=[0, 0, 0], cost_bps=10,
    )
    assert result["ending_values"] == pytest.approx([0.3, 0.5, 0.1992])
    assert result["net_equity"] == pytest.approx(0.9992)
    assert result["ending_weights"] == pytest.approx(np.array([0.3, 0.5, 0.1992]) / 0.9992)


def test_two_stage_fees_use_each_stages_actual_net_asset_values():
    early = np.array([0.10, -0.05, 0.0])
    post = np.array([-0.03, 0.02, 0.0])
    values = np.array([0.3, 0.5, 0.2])
    values[2] -= 0.0008
    values *= 1 + early
    target_values = values.sum() * np.array([0.2, 0.5, 0.3])
    recovery_fee = np.abs(target_values[:2] - values[:2]).sum() * 0.001
    target_values[2] -= recovery_fee
    expected = target_values * (1 + post)
    result = settle_target_session(
        [0, 0, 1], open_target_weights=[0.3, 0.5, 0.2],
        asset_returns=(1 + early) * (1 + post) - 1, cost_bps=10,
        recovery_target_weights=[0.2, 0.5, 0.3], early_returns=early,
    )
    assert result["ending_values"] == pytest.approx(expected)
    assert result["fees"] == pytest.approx(0.0008 + recovery_fee)
    assert result["ending_weights"].sum() == pytest.approx(1.0)
    assert result["net_return"] == pytest.approx(expected.sum() - 1)


def test_hold_has_zero_fees_and_only_passive_value_drift():
    result = settle_target_session(
        [0.3, 0.5, 0.2], open_target_weights=None,
        asset_returns=[0.1, -0.05, 0], cost_bps=25,
    )
    assert result["ending_values"] == pytest.approx([0.33, 0.475, 0.2])
    assert result["fees"] == 0
    assert result["turnover"] == 0
