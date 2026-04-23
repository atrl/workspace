from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
from tianshou.utils.net.discrete import Actor

from ..config import ExecutionConfig, PositionBudgetConfig, PositionCapConfig
from ..data import FEATURE_COLUMNS
from .execution import (
    DEFAULT_EXECUTION,
    execution_buy_cooldown_days,
    execution_regime,
    execution_trade_lock_days,
    rebalance_threshold,
)
from .modes import ModeState, mode_band, route_mode
from .observation import make_observation
from .positioning import target_position_for_profile
from ..settings import LOOKBACK
from .environment import TrancheLedger, build_price_path
from .policy_builder import actor_action


def evaluate_split(
    actor: Actor,
    df: pd.DataFrame,
    cash0: float = 100_000.0,
    position_cap_config: Optional[PositionCapConfig] = None,
    position_budget_config: Optional[PositionBudgetConfig] = None,
    execution_config: Optional[ExecutionConfig] = None,
    asset_idx: int = 0,
    stress_threshold: float = 0.65,
    action_levels: Optional[np.ndarray] = None,
    observation_window: int = 1,
    hard_floor: Optional[float] = None,
    use_hybrid_overlay: bool = False,
    delta_levels: Optional[np.ndarray] = None,
    feature_columns: list[str] | tuple[str, ...] | None = None,
    mode_router_config=None,
) -> Dict[str, float]:
    cash = cash0
    peak = cash0
    pos = 0.0
    ledger = TrancheLedger()
    exec_cfg = execution_config or DEFAULT_EXECUTION
    buy_cooldown = 0
    trade_lock = 0
    price_path = build_price_path(df)
    values = [cash]
    positions = []
    stress_positions = []
    mode_state = ModeState()
    for idx in range(LOOKBACK, len(df) - 1):
        obs = make_observation(
            df,
            idx,
            pos,
            cash,
            peak,
            ledger,
            price_path,
            asset_idx=asset_idx,
            observation_window=observation_window,
            feature_columns=feature_columns or FEATURE_COLUMNS,
        )
        mode_state = route_mode(obs, mode_state, mode_router_config) if mode_router_config is not None else ModeState(name="flat_wait", days_in_mode=1)
        action = actor_action(actor, obs)
        row = df.iloc[idx]
        # --- Kelly-aware dynamic hard floor (execution-layer) ---
        effective_hard_floor = hard_floor
        if hard_floor is not None and "kelly_position_21" in row.index:
            kelly_pos = float(np.nan_to_num(row["kelly_position_21"]))
            win_rate = float(np.nan_to_num(row["win_rate_21"]))
            if kelly_pos > 0.20 and win_rate > 0.48:
                realized_vol = float(np.nan_to_num(row["realized_vol_10"]))
                vol_factor = 1.0
                if realized_vol > 1.0:
                    vol_factor = max(0.3, 1.0 - (realized_vol - 1.0))
                kelly_floor = min(kelly_pos * 0.7, 0.45)
                effective_hard_floor = max(hard_floor, kelly_floor * vol_factor)
        target_pos = target_position_for_profile(
            action,
            obs,
            config=position_cap_config,
            action_levels=action_levels,
            hard_floor=effective_hard_floor,
            use_hybrid_overlay=use_hybrid_overlay,
            delta_levels=delta_levels,
            mode_band=mode_band(mode_state.name, mode_router_config) if mode_router_config and getattr(mode_router_config, "enabled", False) else None,
            position_budget_config=position_budget_config,
            current_drawdown=max(1.0 - cash / max(peak, 1e-8), 0.0),
        )
        regime_risk = float(np.nan_to_num(row["regime_risk"]))
        realized_vol = float(np.nan_to_num(row["realized_vol_10"]))
        trend_strength = abs(float(np.nan_to_num(row["trend_gap_21"])))
        cum_drop_30 = float(np.nan_to_num(row["cum_drop_30"]))
        relative_strength = float(np.nan_to_num(row["relative_strength_21"]))
        vix_zscore = float(np.nan_to_num(row["vix_zscore"]))
        downside_ratio_10 = float(np.nan_to_num(row["downside_ratio_10"]))
        regime = execution_regime(
            regime_risk,
            realized_vol,
            trend_strength,
            cum_drop_30=cum_drop_30,
            relative_strength=relative_strength,
            vix_zscore=vix_zscore,
            downside_ratio_10=downside_ratio_10,
            config=exec_cfg,
        )
        direction = "buy" if target_pos > pos else ("sell" if target_pos < pos else "neutral")
        min_rebalance = rebalance_threshold(
            regime_risk,
            realized_vol,
            trend_strength,
            cum_drop_30=cum_drop_30,
            relative_strength=relative_strength,
            vix_zscore=vix_zscore,
            downside_ratio_10=downside_ratio_10,
            direction=direction,
            mode=mode_state.name if mode_router_config and getattr(mode_router_config, "enabled", False) else None,
            config=exec_cfg,
            mode_router_config=mode_router_config,
        )
        if trade_lock > 0:
            if abs(target_pos - pos) >= min_rebalance:
                target_pos = pos
            trade_lock -= 1
        if buy_cooldown > 0:
            if target_pos > pos:
                target_pos = pos
            buy_cooldown -= 1
        if abs(target_pos - pos) < min_rebalance:
            target_pos = pos
        next_ret = float(df.iloc[idx + 1]["return"])
        turnover = abs(target_pos - pos)
        cash *= 1.0 + target_pos * next_ret - turnover * 0.001
        peak = max(peak, cash)
        ledger.update(pos, target_pos, float(price_path[idx]))
        if target_pos > pos + 1e-6:
            buy_cooldown = execution_buy_cooldown_days(cum_drop_30, exec_cfg, regime)
        if turnover > exec_cfg.min_trade_turnover:
            trade_lock = execution_trade_lock_days(exec_cfg, regime)
        pos = target_pos
        positions.append(pos)
        if regime_risk >= stress_threshold:
            stress_positions.append(pos)
        values.append(cash)
    account = np.array(values)
    daily_ret = np.diff(account) / account[:-1]
    peak_curve = np.maximum.accumulate(account)
    return {
        "return": (account[-1] / account[0] - 1) * 100,
        "sharpe": float(np.mean(daily_ret) / (np.std(daily_ret) + 1e-8) * np.sqrt(252)),
        "max_dd": float(np.max((peak_curve - account) / peak_curve) * 100),
        "avg_pos": float(np.mean(positions)) if positions else 0.0,
        "stress_avg_pos": float(np.mean(stress_positions)) if stress_positions else 0.0,
        "stress_days": float(len(stress_positions)),
    }


__all__ = ["evaluate_split"]
