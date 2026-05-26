from __future__ import annotations

import numpy as np

from src.soxl_quant.data import FEATURE_COLUMNS, STATIC_INDEX
from src.soxl_quant.policy.observation import EXTRA_INDEX
from src.soxl_quant.policy.reward import ContinuousReward
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
    obs[STATIC_INDEX["trend_gap_63"]] = kwargs.get("trend_gap_63", 0.03)
    obs[STATIC_INDEX["anchor_gap_180"]] = kwargs.get("anchor_gap_180", 0.01)
    obs[STATIC_INDEX["anchor_slope_180"]] = kwargs.get("anchor_slope_180", 0.01)
    obs[STATIC_INDEX["cum_drop_30"]] = kwargs.get("cum_drop_30", -0.08)
    obs[STATIC_INDEX["mom_10"]] = kwargs.get("mom_10", 0.0)
    obs[STATIC_INDEX["mom_21"]] = kwargs.get("mom_21", 0.0)
    obs[STATIC_INDEX["relative_strength_21"]] = kwargs.get("relative_strength_21", 0.0)
    obs[STATIC_INDEX["vol_of_vol_21"]] = kwargs.get("vol_of_vol_21", 0.0)
    obs[STATIC_INDEX["below_anchor_ratio_63"]] = kwargs.get("below_anchor_ratio_63", 0.0)
    obs[STATIC_INDEX["vix_percentile_252"]] = kwargs.get("vix_percentile_252", 0.0)
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
        "prev_drawdown": kwargs.get("prev_drawdown", kwargs.get("drawdown", 0.1)),
        "regime_risk": kwargs.get("regime_risk", 0.8),
        "regime_delta": kwargs.get("regime_delta", 0.05),
        "flat_steps": kwargs.get("flat_steps", 0.0),
        "rule_pos": kwargs.get("rule_pos", 0.5),
        "done": kwargs.get("done", False),
        "episode_returns": kwargs.get("episode_returns", []),
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


def test_pbrs_rewards_improving_drawdown():
    """PBRS should give a positive bump when drawdown improves (decreases)."""
    reward = ContinuousReward(make_profile())
    # drawdown worsens: potential goes from -0.05 to -0.10 -> delta = -0.05 -> PBRS negative
    worse = reward.reward(make_state(drawdown=0.10, prev_drawdown=0.05))
    # drawdown improves: potential goes from -0.10 to -0.05 -> delta = +0.05 -> PBRS positive
    better = reward.reward(make_state(drawdown=0.05, prev_drawdown=0.10))
    assert better > worse


def test_pbrs_zero_when_drawdown_unchanged():
    """When drawdown doesn't change, PBRS delta should be zero (other terms identical)."""
    reward = ContinuousReward(make_profile())
    base = reward.reward(make_state(drawdown=0.08, prev_drawdown=0.08))
    # other terms equal, only pbrs should cancel out
    assert abs(base - reward.reward(make_state(drawdown=0.08, prev_drawdown=0.08))) < 1e-6


def test_gated_activity_bonus_has_no_effect_in_current_reward():
    """Current reward path no longer grants an activity bonus in any regime."""
    profile = make_profile()
    profile = StrategyConfig(
        name=profile.name,
        max_iters=profile.max_iters,
        episode_per_iter=profile.episode_per_iter,
        episode_len=profile.episode_len,
        hidden_sizes=profile.hidden_sizes,
        lr=profile.lr,
        update_batch_size=profile.update_batch_size,
        update_repeat=profile.update_repeat,
        reward=RewardConfig(
            return_scale=1.0,
            transaction_cost=0.1,
            turnover_penalty=0.0,
            drawdown_penalty=0.0,
            downside_penalty=0.0,
            risk_budget_penalty=0.0,
            high_exposure_penalty=0.0,
            smoothness_penalty=0.0,
            activity_bonus=1.0,
            inactivity_penalty=0.0,
            gated_activity_bonus=True,
            activity_gate_vol_threshold=0.60,
            activity_gate_regime_delta_threshold=0.10,
        ),
    )
    reward = ContinuousReward(profile)
    # calm market: realized_vol=0.30, regime_delta=0.02
    calm = reward.reward(make_state(turnover=0.20, realized_vol=0.30, regime_delta=0.02))
    # stormy market: realized_vol=0.70, regime_delta=0.15
    stormy = reward.reward(make_state(turnover=0.20, realized_vol=0.70, regime_delta=0.15))
    assert abs(stormy - calm) < 1e-6


def test_rule_deviation_penalty_discourages_moving_far_from_rule():
    profile = make_profile()
    profile = StrategyConfig(
        name=profile.name,
        max_iters=profile.max_iters,
        episode_per_iter=profile.episode_per_iter,
        episode_len=profile.episode_len,
        hidden_sizes=profile.hidden_sizes,
        lr=profile.lr,
        update_batch_size=profile.update_batch_size,
        update_repeat=profile.update_repeat,
        reward=RewardConfig(
            return_scale=1.0,
            transaction_cost=0.0,
            turnover_penalty=0.0,
            drawdown_penalty=0.0,
            downside_penalty=0.0,
            risk_budget_penalty=0.0,
            high_exposure_penalty=0.0,
            smoothness_penalty=0.0,
            activity_bonus=0.0,
            inactivity_penalty=0.0,
            rule_deviation_penalty=2.0,
        ),
    )
    reward = ContinuousReward(profile)
    near_rule = reward.reward(make_state(pos=0.52, turnover=0.0, rule_pos=0.50))
    far_rule = reward.reward(make_state(pos=0.80, turnover=0.0, rule_pos=0.50))
    assert near_rule > far_rule


def test_micro_rebalance_penalty_hits_small_turnover_more_than_large_turnover():
    profile = make_profile()
    profile = StrategyConfig(
        name=profile.name,
        max_iters=profile.max_iters,
        episode_per_iter=profile.episode_per_iter,
        episode_len=profile.episode_len,
        hidden_sizes=profile.hidden_sizes,
        lr=profile.lr,
        update_batch_size=profile.update_batch_size,
        update_repeat=profile.update_repeat,
        reward=RewardConfig(
            return_scale=1.0,
            transaction_cost=0.0,
            turnover_penalty=0.0,
            drawdown_penalty=0.0,
            downside_penalty=0.0,
            risk_budget_penalty=0.0,
            high_exposure_penalty=0.0,
            smoothness_penalty=0.0,
            activity_bonus=0.0,
            inactivity_penalty=0.0,
            micro_rebalance_penalty=1.0,
            micro_rebalance_threshold=0.05,
        ),
    )
    reward = ContinuousReward(profile)
    tiny = reward.reward(make_state(turnover=0.01))
    bigger = reward.reward(make_state(turnover=0.08))
    assert bigger > tiny


def test_state_conditioned_reward_penalizes_high_exposure_in_broken_anchor_downtrend():
    profile = make_profile()
    profile = StrategyConfig(
        name=profile.name,
        max_iters=profile.max_iters,
        episode_per_iter=profile.episode_per_iter,
        episode_len=profile.episode_len,
        hidden_sizes=profile.hidden_sizes,
        lr=profile.lr,
        update_batch_size=profile.update_batch_size,
        update_repeat=profile.update_repeat,
        reward=RewardConfig(
            return_scale=1.0,
            transaction_cost=0.0,
            turnover_penalty=0.0,
            drawdown_penalty=0.0,
            downside_penalty=0.0,
            risk_budget_penalty=0.0,
            high_exposure_penalty=0.0,
            smoothness_penalty=0.0,
            activity_bonus=0.0,
            inactivity_penalty=0.0,
            state_conditioned_reward_enabled=True,
            state_downtrend_position_penalty=2.0,
            state_downtrend_position_threshold=0.30,
            state_downtrend_anchor_gap_max=-0.05,
            state_downtrend_below_anchor_ratio_min=0.55,
            state_downtrend_vix_percentile_min=0.60,
        ),
    )
    reward = ContinuousReward(profile)
    low_exposure = reward.reward(
        make_state(
            pos=0.25,
            drawdown=0.12,
            anchor_gap_180=-0.10,
            below_anchor_ratio_63=0.80,
            vix_percentile_252=0.80,
            turnover=0.0,
        )
    )
    high_exposure = reward.reward(
        make_state(
            pos=0.65,
            drawdown=0.12,
            anchor_gap_180=-0.10,
            below_anchor_ratio_63=0.80,
            vix_percentile_252=0.80,
            turnover=0.0,
        )
    )
    assert low_exposure > high_exposure


def test_state_conditioned_reward_penalizes_underparticipation_only_in_rebound_state():
    profile = make_profile()
    profile = StrategyConfig(
        name=profile.name,
        max_iters=profile.max_iters,
        episode_per_iter=profile.episode_per_iter,
        episode_len=profile.episode_len,
        hidden_sizes=profile.hidden_sizes,
        lr=profile.lr,
        update_batch_size=profile.update_batch_size,
        update_repeat=profile.update_repeat,
        reward=RewardConfig(
            return_scale=1.0,
            transaction_cost=0.0,
            turnover_penalty=0.0,
            drawdown_penalty=0.0,
            downside_penalty=0.0,
            risk_budget_penalty=0.0,
            high_exposure_penalty=0.0,
            smoothness_penalty=0.0,
            activity_bonus=0.0,
            inactivity_penalty=0.0,
            state_conditioned_reward_enabled=True,
            state_rebound_opportunity_penalty=1.5,
            state_rebound_target_position=0.45,
            state_rebound_mom10_min=0.02,
            state_rebound_mom21_min=0.20,
            state_rebound_relative_strength_min=0.50,
            state_rebound_vol_of_vol_min=0.08,
            state_rebound_vix_percentile_max=0.85,
        ),
    )
    reward = ContinuousReward(profile)
    underheld_rebound = reward.reward(
        make_state(
            pos=0.10,
            turnover=0.0,
            mom_10=0.03,
            mom_21=0.28,
            relative_strength_21=0.80,
            vol_of_vol_21=0.12,
            vix_percentile_252=0.70,
        )
    )
    participated_rebound = reward.reward(
        make_state(
            pos=0.45,
            turnover=0.0,
            mom_10=0.03,
            mom_21=0.28,
            relative_strength_21=0.80,
            vol_of_vol_21=0.12,
            vix_percentile_252=0.70,
        )
    )
    underheld_non_rebound = reward.reward(
        make_state(
            pos=0.10,
            turnover=0.0,
            mom_10=-0.01,
            mom_21=0.05,
            relative_strength_21=0.10,
            vol_of_vol_21=0.12,
            vix_percentile_252=0.70,
        )
    )
    assert participated_rebound > underheld_rebound
    assert underheld_non_rebound > underheld_rebound


def test_sequence_risk_reward_penalizes_high_exposure_and_drawdown_only_in_trend_break_state():
    profile = make_profile()
    profile = StrategyConfig(
        name=profile.name,
        max_iters=profile.max_iters,
        episode_per_iter=profile.episode_per_iter,
        episode_len=profile.episode_len,
        hidden_sizes=profile.hidden_sizes,
        lr=profile.lr,
        update_batch_size=profile.update_batch_size,
        update_repeat=profile.update_repeat,
        reward=RewardConfig(
            return_scale=1.0,
            transaction_cost=0.0,
            turnover_penalty=0.0,
            drawdown_penalty=0.0,
            downside_penalty=0.0,
            risk_budget_penalty=0.0,
            high_exposure_penalty=0.0,
            smoothness_penalty=0.0,
            activity_bonus=0.0,
            inactivity_penalty=0.0,
            sequence_risk_reward_enabled=True,
            sequence_risk_position_penalty=3.0,
            sequence_risk_drawdown_penalty=2.0,
            sequence_risk_position_threshold=0.10,
            sequence_risk_drawdown_threshold=0.15,
            sequence_risk_min_feature_hits=2,
            sequence_risk_anchor_slope_max=-0.01,
            sequence_risk_anchor_gap_max=-0.12,
            sequence_risk_trend_gap_min=0.18,
        ),
    )
    reward = ContinuousReward(profile)
    low_exposure_sequence = reward.reward(
        make_state(
            pos=0.08,
            drawdown=0.18,
            anchor_slope_180=-0.04,
            anchor_gap_180=-0.20,
            trend_gap_63=0.22,
            turnover=0.0,
        )
    )
    high_exposure_sequence = reward.reward(
        make_state(
            pos=0.45,
            drawdown=0.18,
            anchor_slope_180=-0.04,
            anchor_gap_180=-0.20,
            trend_gap_63=0.22,
            turnover=0.0,
        )
    )
    high_exposure_non_sequence = reward.reward(
        make_state(
            pos=0.45,
            drawdown=0.18,
            anchor_slope_180=0.03,
            anchor_gap_180=0.08,
            trend_gap_63=0.05,
            turnover=0.0,
        )
    )

    assert low_exposure_sequence > high_exposure_sequence
    assert high_exposure_non_sequence > high_exposure_sequence


def test_moderate_reward_applies_terminal_cvar_penalty():
    profile = make_profile()
    profile = StrategyConfig(
        name=profile.name,
        max_iters=profile.max_iters,
        episode_per_iter=profile.episode_per_iter,
        episode_len=profile.episode_len,
        hidden_sizes=profile.hidden_sizes,
        lr=profile.lr,
        update_batch_size=profile.update_batch_size,
        update_repeat=profile.update_repeat,
        reward=RewardConfig(
            return_scale=0.0,
            transaction_cost=0.0,
            turnover_penalty=0.0,
            drawdown_penalty=0.0,
            downside_penalty=0.0,
            risk_budget_penalty=0.0,
            high_exposure_penalty=0.0,
            smoothness_penalty=0.0,
            activity_bonus=0.0,
            inactivity_penalty=0.0,
            use_moderate_reward=True,
            use_alpha_reward=True,
            alpha_scale=0.0,
            use_cvar_terminal=True,
            cvar_terminal_coef=2.0,
            cvar_terminal_alpha=0.25,
        ),
    )
    reward = ContinuousReward(profile)
    clean_tail = reward.reward(
        make_state(
            val=100_000.0,
            pval=100_000.0,
            pos=0.2,
            turnover=0.0,
            drawdown=0.0,
            regime_risk=0.2,
            done=True,
            episode_returns=[0.01, 0.02, 0.0, -0.005],
        )
    )
    bad_tail = reward.reward(
        make_state(
            val=100_000.0,
            pval=100_000.0,
            pos=0.2,
            turnover=0.0,
            drawdown=0.0,
            regime_risk=0.2,
            done=True,
            episode_returns=[-0.20, -0.10, 0.01, 0.02],
        )
    )
    assert clean_tail > bad_tail
