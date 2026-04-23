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
        flat_steps = float(state.get("flat_steps", 0.0))
        realized_vol = max(float(state["obs"][STATIC_INDEX["realized_vol_10"]]), 0.0)
        anchor_gap = float(state["obs"][STATIC_INDEX["anchor_gap_180"]])
        cum_drop_30 = float(state["obs"][STATIC_INDEX["cum_drop_30"]])

        # ============================================================
        # MINIMAL REWARD BRANCH — major adjustment
        # ============================================================
        if getattr(cfg, "use_minimal_reward", False):
            return self._minimal_reward(state, cfg, pos, turnover, drawdown, regime_risk)

        # ============================================================
        # SIMPLIFIED REWARD BRANCH (deprecated)
        # ============================================================
        if getattr(cfg, "use_simplified_reward", False):
            return self._simplified_reward(state, cfg, pos, pnl_ret, turnover, drawdown, regime_risk, realized_vol, anchor_gap, cum_drop_30, flat_steps)

        # ============================================================
        # MODERATE REWARD BRANCH (curated 12-term + imitation bonus)
        # ============================================================
        if getattr(cfg, "use_moderate_reward", False):
            return self._moderate_reward(state, cfg, pos, prev_val, pnl_ret, turnover, drawdown, regime_risk, flat_steps, realized_vol, anchor_gap, cum_drop_30)

        # ============================================================
        # KELLY REWARD BRANCH (v10: 3-term minimal + Kelly alignment)
        # ============================================================
        if getattr(cfg, "use_kelly_reward", False):
            return self._kelly_reward(state, cfg, pos, turnover, realized_vol)

        # ============================================================
        # FULL LEGACY REWARD (backward compatibility)
        # ============================================================
        return self._legacy_reward(state, cfg, pos, prev_val, pnl_ret, turnover, drawdown, regime_risk, flat_steps, realized_vol, anchor_gap, cum_drop_30)

    def _simplified_reward(self, state, cfg, pos, pnl_ret, turnover, drawdown, regime_risk, realized_vol, anchor_gap, cum_drop_30, flat_steps):
        """Minimal reward: alpha + drawdown control + transaction cost.

        Hypothesis: the 200-line legacy reward creates conflicting gradients.
        A clean 3-term reward lets the model learn cash-holding organically.
        """
        step_return = float(state.get("step_return", 0.0))
        rule_return = float(state.get("rule_return", 0.0))
        alpha = step_return - rule_return

        # 1. Core alpha signal (Kelly-scaled)
        effective_alpha_scale = float(cfg.alpha_scale)
        if regime_risk < 0.35:
            effective_alpha_scale *= 1.3
        elif regime_risk > 0.60:
            effective_alpha_scale *= 0.7
        safe_alpha = max(alpha, -0.999)
        reward = math.log(1.0 + safe_alpha) * 100.0 * effective_alpha_scale

        # 2. Transaction cost
        reward -= turnover * cfg.transaction_cost * 100.0
        reward -= turnover * cfg.turnover_penalty
        reward -= (turnover ** 2) * cfg.smoothness_penalty

        # 3. Drawdown control — the ONLY risk term
        # Immediate penalty when drawdown worsens (proportional to increase)
        if cfg.episode_drawdown_penalty > 0:
            episode_max_dd = float(state.get("episode_max_dd", 0.0))
            prev_episode_max_dd = float(state.get("prev_episode_max_dd", 0.0))
            if episode_max_dd > prev_episode_max_dd:
                increase = episode_max_dd - prev_episode_max_dd
                reward -= increase * cfg.episode_drawdown_penalty * 100.0

        # Terminal penalty on excess drawdown
        if state.get("done", False) and cfg.episode_drawdown_penalty > 0:
            episode_max_dd = float(state.get("episode_max_dd", 0.0))
            excess_episode_dd = max(episode_max_dd - cfg.episode_drawdown_threshold, 0.0)
            reward -= excess_episode_dd * cfg.episode_drawdown_penalty * 100.0

        # 4. Excess position penalty (only when drawdown is already elevated)
        # This replaces the complex risk_budget_penalty + high_exposure_penalty
        if drawdown > 0.10 and pos > 0.50:
            reward -= (drawdown - 0.10) * (pos - 0.50) * 2.0

        # 5. Minimal inactivity penalty (to prevent permanent cash, but weak)
        if flat_steps >= 10 and realized_vol < 0.30 and anchor_gap > 0.01:
            reward -= 0.05

        return float(reward)

    def _kelly_reward(self, state, cfg, pos, turnover, realized_vol):
        """v10 Kelly-centric minimal reward: 3 terms only.

        Hypothesis: v4/v5-v9 failed because the reward function had too many
        conflicting terms.  A pure 3-term reward (alpha + Kelly alignment +
        transaction cost) gives the model a single clear signal to follow.
        """
        # 1. Core alpha (Kelly-log scaled)
        step_return = float(state.get("step_return", 0.0))
        rule_return = float(state.get("rule_return", 0.0))
        alpha = step_return - rule_return
        safe_alpha = max(alpha, -0.999)
        reward = math.log(1.0 + safe_alpha) * 100.0 * float(cfg.alpha_scale)

        # 2. Kelly alignment bonus (the ONLY new term)
        kelly_bonus = getattr(cfg, "kelly_tracking_bonus", 0.0)
        if kelly_bonus > 0.0:
            kelly_pos = float(state.get("kelly_position_21", 0.0))
            win_rate = float(state.get("win_rate_21", 0.5))
            # Activate when Kelly shows positive edge
            if kelly_pos > getattr(cfg, "kelly_min_signal", 0.15) and win_rate > 0.45:
                target = min(kelly_pos, 1.0)
                # Signal quality: require decent win-rate + non-negative Sharpe
                sharpe = float(state.get("sharpe_21", 0.0))
                signal_quality = max(0.0, (win_rate - 0.45) * 3.0) * (1.0 if sharpe > 0 else 0.5)
                if signal_quality > 0.0:
                    alignment = max(0.0, 1.0 - abs(pos - target))
                    # Volatility attenuation
                    attenuation = 1.0
                    if getattr(cfg, "kelly_vol_attenuation", True) and realized_vol > 1.0:
                        attenuation = max(0.2, 1.0 - (realized_vol - 1.0) * 0.8)
                    reward += alignment * kelly_bonus * signal_quality * attenuation

        # 3. Transaction cost (keep turnover reasonable)
        reward -= turnover * cfg.transaction_cost * 100.0

        return float(reward)

    def _minimal_reward(self, state, cfg, pos, turnover, drawdown, regime_risk):
        """Truly minimal reward: 5 terms, ~10 parameters.

        Hypothesis: the 200-line legacy reward creates conflicting gradients.
        A clean 5-term reward + regime-conditional budget lets the model learn
        cash-holding organically while the budget provides structural drawdown control.
        """
        # 1. Core alpha signal (Kelly-scaled, regime-modulated)
        step_return = float(state.get("step_return", 0.0))
        rule_return = float(state.get("rule_return", 0.0))
        alpha = step_return - rule_return
        alpha_scale = float(cfg.alpha_scale)
        if regime_risk < 0.35:
            alpha_scale *= 1.3
        elif regime_risk > 0.60:
            alpha_scale *= 0.7
        safe_alpha = max(alpha, -0.999)
        reward = math.log(1.0 + safe_alpha) * 100.0 * alpha_scale

        # 2. Transaction costs (linear + quadratic for smoothness)
        reward -= turnover * cfg.transaction_cost * 100.0
        reward -= turnover * cfg.turnover_penalty
        reward -= (turnover ** 2) * cfg.smoothness_penalty

        # 3. Regime-conditional budget violation penalty
        # When regime_risk >= threshold, position is capped at budget_cap.
        # This integrates the proven rt0.6 overlay directly into training.
        if getattr(cfg, "use_regime_budget", False):
            budget = 1.0 if regime_risk < cfg.regime_budget_threshold else cfg.regime_budget_cap
            if pos > budget + 1e-6:
                reward -= (pos - budget) * cfg.regime_budget_penalty

        # 4. Episode drawdown penalty (step-level)
        if cfg.episode_drawdown_penalty > 0:
            episode_max_dd = float(state.get("episode_max_dd", 0.0))
            prev_episode_max_dd = float(state.get("prev_episode_max_dd", 0.0))
            if episode_max_dd > prev_episode_max_dd:
                increase = episode_max_dd - prev_episode_max_dd
                reward -= increase * cfg.episode_drawdown_penalty * 100.0

        # 5. Terminal drawdown penalty
        if state.get("done", False) and cfg.episode_drawdown_penalty > 0:
            episode_max_dd = float(state.get("episode_max_dd", 0.0))
            excess = max(episode_max_dd - cfg.episode_drawdown_threshold, 0.0)
            reward -= excess * cfg.episode_drawdown_penalty * 100.0

        return float(reward)

    def _moderate_reward(self, state, cfg, pos, prev_val, pnl_ret, turnover, drawdown, regime_risk, flat_steps, realized_vol, anchor_gap, cum_drop_30):
        """Moderate reward: curated 12-term + rulebase imitation bonus.

        Design principles (from minimal-v1 failure analysis):
        1. Core alpha (Kelly-log) is non-negotiable.
        2. Episodic drawdown penalty > step-level for structural control.
        3. Weak step-risk signals prevent over-exposure without micro-management.
        4. Participation signals (trend_alignment, opportunity_cost) are essential
           to prevent permanent conservatism.
        5. Tactical bonuses (profit_taking, panic_buy) capture edge cases.
        6. Cost controls keep turnover reasonable.
        7. Imitation bonus provides a positive baseline for learning rulebase timing.
        """
        # --- 1. Core alpha (Kelly-log, regime-modulated) ---
        step_return = float(state.get("step_return", 0.0))
        rule_return = float(state.get("rule_return", 0.0))
        alpha = step_return - rule_return
        alpha_scale = float(cfg.alpha_scale)
        if regime_risk < 0.35:
            alpha_scale *= 1.3
        elif regime_risk > 0.60:
            alpha_scale *= 0.7
        safe_alpha = max(alpha, -0.999)
        reward = math.log(1.0 + safe_alpha) * 100.0 * alpha_scale

        # --- 2. PBRS (potential-based reward shaping for drawdown) ---
        prev_drawdown = float(state.get("prev_drawdown", drawdown))
        pbrs_delta = (-drawdown) - (-prev_drawdown)
        reward += pbrs_delta * cfg.pbrs_coef

        # --- 3. Cost controls ---
        reward -= turnover * cfg.transaction_cost * 100.0
        if turnover < 0.15:
            effective_turnover_penalty = cfg.turnover_penalty * 3.0
            effective_smoothness = cfg.smoothness_penalty * 3.0
        else:
            effective_turnover_penalty = cfg.turnover_penalty
            effective_smoothness = cfg.smoothness_penalty
        reward -= turnover * effective_turnover_penalty
        reward -= (turnover ** 2) * effective_smoothness

        # --- 4. Weak step-level risk controls ---
        reward -= max(drawdown - 0.06, 0.0) * cfg.drawdown_penalty
        downside_signal = max(float(state["obs"][STATIC_INDEX["downside_semivol_21"]]), 0.0)
        reward -= downside_signal * cfg.downside_penalty
        reward -= max(pos - (1.0 - 0.65 * regime_risk), 0.0) * cfg.risk_budget_penalty
        reward -= max(pos - 0.85, 0.0) * cfg.high_exposure_penalty

        # --- 5. Participation signals (prevent permanent cash) ---
        if cfg.trend_alignment_bonus > 0:
            if anchor_gap > cfg.trend_alignment_threshold and pos > 0.55:
                reward += cfg.trend_alignment_bonus
            if anchor_gap < -cfg.trend_alignment_threshold and pos < 0.35:
                reward += cfg.trend_alignment_bonus
        if cfg.opportunity_cost_penalty > 0:
            if anchor_gap > cfg.opportunity_threshold_anchor and pos < cfg.opportunity_threshold_pos:
                reward -= cfg.opportunity_cost_penalty
            trend_gap = float(state["obs"][STATIC_INDEX["trend_gap_21"]])
            if trend_gap > cfg.opportunity_threshold_trend and realized_vol < 0.45 and pos < cfg.opportunity_threshold_pos + 0.05:
                reward -= cfg.opportunity_cost_penalty * 0.40

        # --- 5b. Kelly criterion tracking (prevent fixed low-position collapse) ---
        # v5/v6 failure: penalty-based tracking (deviation * penalty) was too weak
        # to overcome episode_drawdown fear. Switched to BONUS-based tracking:
        # when Kelly shows positive edge, REWARD alignment instead of punishing deviation.
        kelly_bonus = getattr(cfg, "kelly_tracking_bonus", 0.0)
        if kelly_bonus > 0.0:
            kelly_pos = float(state.get("kelly_position_21", 0.0))
            win_rate = float(state.get("win_rate_21", 0.5))
            sharpe = float(state.get("sharpe_21", 0.0))
            # Only activate when Kelly signal is clearly positive
            if kelly_pos > getattr(cfg, "kelly_min_signal", 0.1):
                # Signal quality: require decent win-rate and non-negative Sharpe
                signal_quality = max(0.0, (win_rate - 0.45) * 3.0) * (1.0 if sharpe > 0 else 0.5)
                if signal_quality > 0.0:
                    target = min(kelly_pos, 1.0)
                    # Reward increases as position approaches Kelly target
                    alignment = max(0.0, 1.0 - abs(pos - target))
                    # Volatility attenuation: dampen bonus in extreme vol regimes
                    attenuation = 1.0
                    if getattr(cfg, "kelly_vol_attenuation", True) and realized_vol > 1.0:
                        attenuation = max(0.2, 1.0 - (realized_vol - 1.0) * 0.8)
                    reward += alignment * kelly_bonus * signal_quality * attenuation

        # --- 6. Tactical bonuses ---
        holding_age = max(float(state["obs"][EXTRA_INDEX["holding_age"]]), 0.0)
        avg_entry_pnl = float(state["obs"][EXTRA_INDEX["avg_entry_pnl"]])
        prev_pos = float(state.get("prev_pos", pos))
        last_tranche_pnl = float(state["obs"][EXTRA_INDEX["last_tranche_pnl"]])
        if turnover > 0.02 and pos < prev_pos - 1e-6:
            if last_tranche_pnl > cfg.tranche_profit_threshold:
                reward += cfg.profit_taking_bonus * min(last_tranche_pnl * 12.0, 2.0)
            elif last_tranche_pnl < -0.05:
                reward -= cfg.underwater_sell_penalty * min(abs(last_tranche_pnl) * 8.0, 1.5)
        if turnover > 0.02 and pos > prev_pos + 1e-6:
            if cum_drop_30 < cfg.panic_buy_cum_drop_threshold and regime_risk > cfg.panic_buy_regime_risk_threshold:
                panic_depth = min(abs(cum_drop_30) - abs(cfg.panic_buy_cum_drop_threshold), 0.15)
                reward += cfg.panic_buy_bonus * (1.0 + panic_depth * 5.0)

        # --- 6b. Bottomless pit protection (NEW) ---
        # When market is in free-fall (deep cumulative drop + extreme volatility),
        # penalize ANY increase in position. This prevents the model from
        # catching a falling knife like in Oct 2022.
        if getattr(cfg, "bottomless_pit_penalty", 0.0) > 0.0:
            if turnover > 1e-6 and pos > prev_pos + 1e-6:
                pit_depth = 0.0
                if cum_drop_30 < -0.25:
                    pit_depth += 0.5
                if cum_drop_30 < -0.35:
                    pit_depth += 0.5
                if realized_vol > 1.0:
                    pit_depth += 0.3
                if realized_vol > 1.5:
                    pit_depth += 0.3
                if regime_risk > 0.75:
                    pit_depth += 0.2
                if pit_depth > 0.0:
                    reward -= cfg.bottomless_pit_penalty * pit_depth

        # --- 7. Micro-rebalance penalty ---
        if cfg.micro_rebalance_penalty > 0 and 0.0 < turnover < cfg.micro_rebalance_threshold:
            micro_ratio = turnover / max(cfg.micro_rebalance_threshold, 1e-6)
            reward -= (1.0 - micro_ratio) * cfg.micro_rebalance_penalty

        # --- 8. Episodic drawdown penalty ---
        if cfg.episode_drawdown_penalty > 0:
            episode_max_dd = float(state.get("episode_max_dd", 0.0))
            prev_episode_max_dd = float(state.get("prev_episode_max_dd", 0.0))
            if episode_max_dd > prev_episode_max_dd:
                increase = episode_max_dd - prev_episode_max_dd
                reward -= increase * cfg.episode_drawdown_penalty * 100.0
        if state.get("done", False) and cfg.episode_drawdown_penalty > 0:
            episode_max_dd = float(state.get("episode_max_dd", 0.0))
            excess = max(episode_max_dd - cfg.episode_drawdown_threshold, 0.0)
            reward -= excess * cfg.episode_drawdown_penalty * 100.0

        # --- 9. Downside asymmetry ---
        if cfg.downside_asymmetry_coef > 1.0 and pnl_ret < 0:
            reward *= cfg.downside_asymmetry_coef

        # --- 10. Rulebase imitation bonus (NEW) ---
        if getattr(cfg, "rule_imitation_bonus", 0.0) > 0.0:
            rule_pos = float(state.get("rule_pos", pos))
            prev_rule_pos = float(state.get("prev_rule_pos", rule_pos))
            # Position match bonus: dense positive baseline
            position_similarity = 1.0 - min(abs(pos - rule_pos), 1.0)
            imitation_reward = position_similarity * cfg.rule_imitation_bonus
            # Decision point bonus: extra when rulebase trades and model follows
            rule_turnover = abs(rule_pos - prev_rule_pos)
            if rule_turnover > 0.03:
                model_direction = pos - prev_pos
                rule_direction = rule_pos - prev_rule_pos
                if model_direction * rule_direction > 1e-12:
                    decision_bonus = getattr(cfg, "rule_decision_bonus", 0.0) * min(1.0, rule_turnover / 0.10)
                    imitation_reward += decision_bonus
            reward += imitation_reward

        return float(reward)

    def _legacy_reward(self, state, cfg, pos, prev_val, pnl_ret, turnover, drawdown, regime_risk, flat_steps, realized_vol, anchor_gap, cum_drop_30):
        """Original 200-line reward function — kept for backward compatibility."""
        regime_delta = abs(float(state.get("regime_delta", 0.0)))
        downside_signal = max(float(state["obs"][STATIC_INDEX["downside_semivol_21"]]), 0.0)
        trend_strength = abs(float(state["obs"][STATIC_INDEX["trend_gap_21"]]))
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
            effective_alpha_scale = float(cfg.alpha_scale)
            if cfg.use_kelly_alpha:
                if regime_risk < 0.35:
                    effective_alpha_scale = float(cfg.alpha_scale) * 1.3
                elif regime_risk > 0.60:
                    effective_alpha_scale = float(cfg.alpha_scale) * 0.7
                safe_alpha = max(alpha, -0.999)
                reward = math.log(1.0 + safe_alpha) * 100.0 * effective_alpha_scale
            else:
                if regime_risk < 0.35:
                    effective_alpha_scale = float(cfg.alpha_scale) * 1.3
                elif regime_risk > 0.60:
                    effective_alpha_scale = float(cfg.alpha_scale) * 0.7
                reward = alpha * 100.0 * effective_alpha_scale
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

        rule_pos = float(state.get("rule_pos", pos))
        if cfg.rule_deviation_penalty > 0:
            reward -= abs(pos - rule_pos) * cfg.rule_deviation_penalty
        if cfg.micro_rebalance_penalty > 0 and 0.0 < turnover < cfg.micro_rebalance_threshold:
            micro_ratio = turnover / max(cfg.micro_rebalance_threshold, 1e-6)
            reward -= (1.0 - micro_ratio) * cfg.micro_rebalance_penalty
        if cfg.flat_trade_penalty > 0.0 and turnover > 0.0:
            reward -= cfg.flat_trade_penalty
        if cfg.short_hold_turnover_penalty > 0 and turnover > 0.0 and holding_age < cfg.short_hold_threshold:
            reward -= (1.0 - holding_age / max(cfg.short_hold_threshold, 1e-6)) * turnover * cfg.short_hold_turnover_penalty

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

        if cfg.episode_drawdown_penalty > 0:
            episode_max_dd = float(state.get("episode_max_dd", 0.0))
            prev_episode_max_dd = float(state.get("prev_episode_max_dd", 0.0))
            if episode_max_dd > prev_episode_max_dd:
                increase = episode_max_dd - prev_episode_max_dd
                reward -= increase * cfg.episode_drawdown_penalty * 100.0

        if state.get("done", False) and cfg.episode_drawdown_penalty > 0:
            episode_max_dd = float(state.get("episode_max_dd", 0.0))
            excess_episode_dd = max(episode_max_dd - cfg.episode_drawdown_threshold, 0.0)
            reward -= excess_episode_dd * cfg.episode_drawdown_penalty * 100.0

        if (
            state.get("done", False)
            and cfg.use_cvar_terminal
            and cfg.cvar_terminal_coef > 0
        ):
            returns = state.get("episode_returns") or []
            if returns:
                sorted_returns = sorted(float(r) for r in returns)
                alpha = max(min(float(cfg.cvar_terminal_alpha), 0.5), 1e-3)
                tail_count = max(1, int(len(sorted_returns) * alpha))
                cvar_tail = sum(sorted_returns[:tail_count]) / tail_count
                reward -= max(-cvar_tail, 0.0) * cfg.cvar_terminal_coef * 100.0

        if cfg.downside_asymmetry_coef > 1.0 and pnl_ret < 0:
            reward *= cfg.downside_asymmetry_coef

        if flat_steps >= 4:
            flat_drag = min((flat_steps - 3.0) / 6.0, 1.6)
            reward -= flat_drag * (0.20 + activity_signal * 0.9) * cfg.inactivity_penalty
        if flat_steps >= 8 and activity_signal > 0.12:
            reward -= min(flat_steps / 8.0, 1.5) * cfg.inactivity_penalty
        if holding_age >= 0.35 and turnover < 0.02 and abs(avg_entry_pnl) < 0.02:
            reward -= (0.16 + holding_age * 0.14) * cfg.inactivity_penalty

        return float(reward)


__all__ = ["ContinuousReward", "EXTRA_INDEX"]
