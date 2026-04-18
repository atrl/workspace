from __future__ import annotations

import math

from qlib.rl.reward import Reward

from ..config import StrategyConfig
from ..data import STATIC_INDEX
from .observation import EXTRA_INDEX


class ContinuousReward(Reward):
    """连续型奖励函数：评估交易质量，而不仅是收益率。"""

    def __init__(self, profile: StrategyConfig):
        super().__init__()
        self.profile = profile

    def reward(self, state):
        cfg = self.profile.reward
        pos = float(state["pos"])
        prev_val = float(state["pval"])
        pnl_ret = (float(state["val"]) - prev_val) / prev_val if prev_val > 0 else 0.0
        turnover = float(state.get("turnover", 0.0))
        drawdown = float(state.get("drawdown", 0.0))
        regime_risk = float(state.get("regime_risk", 0.0))
        regime_delta = abs(float(state.get("regime_delta", 0.0)))
        flat_steps = float(state.get("flat_steps", 0.0))
        downside_signal = max(float(state["obs"][STATIC_INDEX["downside_semivol_21"]]), 0.0)
        realized_vol = max(float(state["obs"][STATIC_INDEX["realized_vol_10"]]), 0.0)
        trend_strength = abs(float(state["obs"][STATIC_INDEX["trend_gap_21"]]))
        anchor_gap = float(state["obs"][STATIC_INDEX["anchor_gap_180"]])
        cum_drop_30 = float(state["obs"][STATIC_INDEX["cum_drop_30"]])
        holding_age = max(float(state["obs"][EXTRA_INDEX["holding_age"]]), 0.0)
        avg_entry_pnl = float(state["obs"][EXTRA_INDEX["avg_entry_pnl"]])
        high_gap = float(state["obs"][EXTRA_INDEX["high_gap"]])
        activity_signal = max(realized_vol - 0.45, 0.0) + 1.8 * regime_delta + 1.1 * trend_strength
        trend_support = max(anchor_gap, 0.0) + max(-cum_drop_30 - 0.10, 0.0) * 0.4
        tolerated_drawdown = max(0.06 + min(trend_support, 0.12) - regime_risk * 0.03, 0.02)
        excess_drawdown = max(drawdown - tolerated_drawdown, 0.0)
        stale_position = max(holding_age - 0.30, 0.0) * max(0.025 - abs(avg_entry_pnl), 0.0)
        trapped_loser = max(holding_age - 0.22, 0.0) * max(-avg_entry_pnl - 0.02, 0.0) * max(-high_gap - 0.03, 0.0)

        if cfg.use_log_reward:
            safe_ret = max(pnl_ret, -0.999)
            reward = math.log(1.0 + safe_ret) * 100.0 * cfg.return_scale * cfg.log_reward_scale
        elif cfg.use_marginal_reward:
            period_alpha = float(state.get("period_alpha", 0.0))
            is_decision_day = float(state.get("is_decision_day", 1.0))
            reward = period_alpha * 100.0 * cfg.return_scale * cfg.marginal_return_scale if is_decision_day > 0.5 else 0.0
        elif cfg.use_alpha_reward:
            step_return = float(state.get("step_return", 0.0))
            rule_return = float(state.get("rule_return", 0.0))
            alpha = step_return - rule_return
            if cfg.use_kelly_alpha:
                safe_alpha = max(alpha, -0.999)
                reward = math.log(1.0 + safe_alpha) * 100.0 * cfg.alpha_scale
            else:
                reward = alpha * 100.0 * cfg.alpha_scale
        elif cfg.use_quality_reward:
            reward = 0.0
            if anchor_gap > cfg.trend_alignment_threshold and pos > 0.55:
                reward += 1.0
            elif anchor_gap < -cfg.trend_alignment_threshold and pos < 0.35:
                reward += 1.0
            else:
                reward -= 0.3
            if drawdown > 0.15 and pos < 0.40:
                reward += 0.5
            elif drawdown > 0.15 and pos > 0.60:
                reward -= 1.0
            reward -= turnover * 0.5
            if flat_steps >= 3:
                reward += 0.2
            reward *= cfg.quality_reward_scale
        elif cfg.use_nday_reward:
            nday_reward = float(state.get("nday_reward", 0.0))
            reward = nday_reward * 100.0 * cfg.return_scale * cfg.nday_reward_scale
        elif cfg.use_sparse_reward:
            if state.get("done", False):
                episode_return = (float(state["val"]) - float(state["episode_start_val"])) / float(state["episode_start_val"])
                reward = episode_return * 100.0 * cfg.return_scale * cfg.sparse_reward_scale
            else:
                reward = 0.0
        else:
            reward = pnl_ret * 100.0 * cfg.return_scale

        prev_drawdown = float(state.get("prev_drawdown", drawdown))
        pbrs_delta = (-drawdown) - (-prev_drawdown)
        reward += pbrs_delta * cfg.pbrs_coef

        if not cfg.use_marginal_reward and not cfg.use_log_reward:
            reward -= turnover * cfg.transaction_cost * 100.0
        if turnover < 0.15:
            effective_turnover_penalty = cfg.turnover_penalty * 3.0
            effective_smoothness = cfg.smoothness_penalty * 3.0
        else:
            effective_turnover_penalty = cfg.turnover_penalty
            effective_smoothness = cfg.smoothness_penalty
        reward -= turnover * effective_turnover_penalty
        reward -= (turnover**2) * effective_smoothness

        if flat_steps >= 3 and cfg.stability_bonus > 0:
            stability_bonus = min((flat_steps - 2.0) / 8.0, 1.0) * cfg.stability_bonus
            reward += stability_bonus

        if cfg.trend_alignment_bonus > 0:
            if anchor_gap > cfg.trend_alignment_threshold and pos > 0.55:
                reward += cfg.trend_alignment_bonus
            if anchor_gap < -cfg.trend_alignment_threshold and pos < 0.35:
                reward += cfg.trend_alignment_bonus

        reward -= excess_drawdown * cfg.drawdown_penalty
        reward -= downside_signal * cfg.downside_penalty
        reward -= max(pos - (1.0 - 0.65 * regime_risk), 0.0) * cfg.risk_budget_penalty
        reward -= max(pos - 0.85, 0.0) * cfg.high_exposure_penalty
        reward += min(drawdown, tolerated_drawdown) * min(pos, 0.80) * min(trend_support * 2.5, 0.18)
        reward -= stale_position * (6.0 + 4.0 * activity_signal)
        reward -= trapped_loser * (18.0 + 6.0 * regime_risk)

        trend_gap = float(state["obs"][STATIC_INDEX["trend_gap_21"]])
        if anchor_gap > cfg.opportunity_threshold_anchor and pos < cfg.opportunity_threshold_pos:
            reward -= cfg.opportunity_cost_penalty
        if trend_gap > cfg.opportunity_threshold_trend and realized_vol < 0.45 and pos < cfg.opportunity_threshold_pos + 0.05:
            reward -= cfg.opportunity_cost_penalty * 0.40
        if anchor_gap > 0.02 and regime_risk < 0.45 and pos < 0.30:
            reward -= 0.12
        if cum_drop_30 < -0.18 and regime_risk < 0.80 and pos < 0.20:
            reward -= 0.10

        last_tranche_pnl = float(state["obs"][EXTRA_INDEX["last_tranche_pnl"]])
        prev_pos = float(state.get("prev_pos", pos))
        if turnover > 0.02 and pos < prev_pos - 1e-6:
            if last_tranche_pnl > cfg.tranche_profit_threshold:
                reward += cfg.profit_taking_bonus * min(last_tranche_pnl * 12.0, 2.0)
            elif last_tranche_pnl < -0.05:
                reward -= cfg.underwater_sell_penalty * min(abs(last_tranche_pnl) * 8.0, 1.5)

        if turnover > 0.02 and pos > prev_pos + 1e-6:
            if cum_drop_30 < cfg.panic_buy_cum_drop_threshold and regime_risk > cfg.panic_buy_regime_risk_threshold:
                panic_depth = min(abs(cum_drop_30) - abs(cfg.panic_buy_cum_drop_threshold), 0.15)
                reward += cfg.panic_buy_bonus * (1.0 + panic_depth * 5.0)

        if realized_vol > 0.50 and turnover > 0.03:
            reward += cfg.vol_trading_bonus * turnover

        if flat_steps >= 4:
            flat_drag = min((flat_steps - 3.0) / 6.0, 1.6)
            reward -= flat_drag * (0.20 + activity_signal * 0.9) * cfg.inactivity_penalty
        if flat_steps >= 8 and activity_signal > 0.12:
            reward -= min(flat_steps / 8.0, 1.5) * cfg.inactivity_penalty
        if holding_age >= 0.35 and turnover < 0.02 and abs(avg_entry_pnl) < 0.02:
            reward -= (0.16 + holding_age * 0.14) * cfg.inactivity_penalty

        return float(reward)


__all__ = ["ContinuousReward", "EXTRA_INDEX"]
