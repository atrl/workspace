from __future__ import annotations

import numpy as np

from src.soxl_quant.data import FEATURE_COLUMNS, STATIC_INDEX
from src.soxl_quant.rl import ContinuousReward, EXTRA_INDEX
from src.soxl_quant.settings import EXTRA_OBS_DIM, RewardConfig, StrategyConfig


def make_profile() -> StrategyConfig:
    return StrategyConfig(
        name="test",
        max_iters=1,
        episode_per_iter=1,
        episode_len=120,
        hidden_sizes=(32, 16),
        lr=3e-4,
        update_batch_size=16,
        update_repeat=1,
        reward=RewardConfig(
            return_scale=1.0,
            transaction_cost=0.1,
            turnover_penalty=0.7,
            drawdown_penalty=3.5,
            downside_penalty=1.6,
            risk_budget_penalty=2.1,
            high_exposure_penalty=1.1,
            smoothness_penalty=0.45,
            activity_bonus=0.8,
            inactivity_penalty=0.4,
        ),
    )


def make_state(**kwargs):
    obs = np.zeros(len(FEATURE_COLUMNS) + EXTRA_OBS_DIM, dtype=np.float32)
    obs[STATIC_INDEX["downside_semivol_21"]] = kwargs.get("downside_semivol", 0.2)
    obs[STATIC_INDEX["realized_vol_10"]] = kwargs.get("realized_vol", 0.55)
    obs[STATIC_INDEX["trend_gap_21"]] = kwargs.get("trend_gap_21", 0.03)
    obs[STATIC_INDEX["anchor_gap_180"]] = kwargs.get("anchor_gap_180", 0.01)
    obs[STATIC_INDEX["cum_drop_30"]] = kwargs.get("cum_drop_30", -0.08)
    obs[EXTRA_INDEX["holding_age"]] = kwargs.get("holding_age", 0.10)
    obs[EXTRA_INDEX["avg_entry_pnl"]] = kwargs.get("avg_entry_pnl", 0.01)
    obs[EXTRA_INDEX["high_gap"]] = kwargs.get("high_gap", -0.01)
    return {
        "obs": obs,
        "pos": kwargs.get("pos", 0.8),
        "val": kwargs.get("val", 101_000.0),
        "pval": kwargs.get("pval", 100_000.0),
        "turnover": kwargs.get("turnover", 0.2),
        "drawdown": kwargs.get("drawdown", 0.1),
        "regime_risk": kwargs.get("regime_risk", 0.8),
        "regime_delta": kwargs.get("regime_delta", 0.05),
        "flat_steps": kwargs.get("flat_steps", 0.0),
    }


def test_reward_penalizes_riskier_state():
    reward = ContinuousReward(make_profile())
    safer = reward.reward(make_state(pos=0.3, turnover=0.05, drawdown=0.02, regime_risk=0.2))
    riskier = reward.reward(make_state(pos=0.95, turnover=0.4, drawdown=0.18, regime_risk=0.9))
    assert safer > riskier


def test_reward_improves_with_better_pnl_when_other_terms_equal():
    reward = ContinuousReward(make_profile())
    weak = reward.reward(make_state(val=100_300.0, pval=100_000.0, pos=0.4, turnover=0.05, drawdown=0.03, regime_risk=0.3))
    strong = reward.reward(make_state(val=101_500.0, pval=100_000.0, pos=0.4, turnover=0.05, drawdown=0.03, regime_risk=0.3))
    assert strong > weak


def test_reward_penalizes_staying_idle_too_long():
    reward = ContinuousReward(make_profile())
    short_idle = reward.reward(make_state(turnover=0.0, flat_steps=0.0, pos=0.5, regime_risk=0.6))
    long_idle = reward.reward(make_state(turnover=0.0, flat_steps=10.0, pos=0.5, regime_risk=0.6))
    assert short_idle > long_idle


def test_reward_penalizes_dead_flat_more_than_controlled_pullback():
    reward = ContinuousReward(make_profile())
    controlled_pullback = reward.reward(
        make_state(
            pos=0.65,
            turnover=0.03,
            drawdown=0.08,
            regime_risk=0.28,
            anchor_gap_180=0.05,
            cum_drop_30=-0.12,
            holding_age=0.18,
            avg_entry_pnl=0.03,
            high_gap=-0.02,
            flat_steps=1.0,
        )
    )
    dead_flat = reward.reward(
        make_state(
            pos=0.18,
            turnover=0.0,
            drawdown=0.01,
            regime_risk=0.28,
            anchor_gap_180=0.05,
            cum_drop_30=-0.12,
            holding_age=0.55,
            avg_entry_pnl=0.002,
            high_gap=-0.005,
            flat_steps=10.0,
        )
    )
    assert controlled_pullback > dead_flat


def test_reward_penalizes_stale_losing_position():
    reward = ContinuousReward(make_profile())
    healthy_hold = reward.reward(
        make_state(
            pos=0.55,
            turnover=0.01,
            drawdown=0.07,
            regime_risk=0.35,
            holding_age=0.28,
            avg_entry_pnl=0.04,
            high_gap=-0.01,
        )
    )
    trapped_hold = reward.reward(
        make_state(
            pos=0.55,
            turnover=0.01,
            drawdown=0.07,
            regime_risk=0.35,
            holding_age=0.60,
            avg_entry_pnl=-0.05,
            high_gap=-0.08,
        )
    )
    assert healthy_hold > trapped_hold
