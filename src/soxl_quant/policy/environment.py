from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import List, NamedTuple, Optional, Tuple

import numpy as np
import pandas as pd
from gymnasium import spaces
from qlib.rl.interpreter import ActionInterpreter, StateInterpreter
from qlib.rl.simulator import Simulator

from ..config import ExecutionConfig, PositionCapConfig, StrategyConfig
from ..data import FEATURE_COLUMNS
from .execution import (
    DEFAULT_EXECUTION,
    execution_buy_cooldown_days,
    execution_regime,
    execution_trade_lock_days,
    rebalance_threshold,
)
from .observation import EXTRA_OBS_DIM, make_observation
from .positioning import (
    BASE_ACTION_LEVELS,
    action_to_position,
    rule_based_position,
)
from ..settings import LOOKBACK, N_ACTIONS, N_ASSETS


@dataclass(frozen=True)
class EpisodeSummary:
    account: np.ndarray
    positions: List[float]


@dataclass
class TrancheLedger:
    """持仓账本（带分段成本栈），与 backtest/runtime 侧共享。"""

    tranches: List[dict[str, float]] = None
    high_water_price: float = 0.0
    days_in_position: int = 0

    def __post_init__(self) -> None:
        if self.tranches is None:
            self.tranches = []

    def reset(self) -> None:
        self.tranches = []
        self.high_water_price = 0.0
        self.days_in_position = 0

    def update(self, prev_pos: float, new_pos: float, price: float) -> None:
        price = float(price)
        if new_pos <= 1e-6:
            self.reset()
            return
        if prev_pos <= 1e-6:
            self.tranches = [{"price": price, "shares": new_pos}]
            self.high_water_price = price
            self.days_in_position = 1
            return

        self.days_in_position += 1
        self.high_water_price = max(self.high_water_price, price)
        if new_pos > prev_pos + 1e-6:
            add_size = max(new_pos - prev_pos, 0.0)
            self.tranches.append({"price": price, "shares": add_size})
        elif new_pos < prev_pos - 1e-6:
            sell_size = prev_pos - new_pos
            remaining = sell_size
            while remaining > 1e-9 and self.tranches:
                last = self.tranches[-1]
                shares_sold = min(remaining, last["shares"])
                last["shares"] -= shares_sold
                if last["shares"] <= 1e-9:
                    self.tranches.pop()
                remaining -= shares_sold

    def snapshot(self, price: float) -> Tuple[float, float, float, float, float, float]:
        if self.days_in_position <= 0 or price <= 0 or not self.tranches:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        total_shares = sum(t["shares"] for t in self.tranches)
        total_cost = sum(t["price"] * t["shares"] for t in self.tranches)
        avg_entry_price = total_cost / total_shares if total_shares > 0 else 0.0
        avg_pnl = price / avg_entry_price - 1.0 if avg_entry_price > 0 else 0.0
        last_add_price = self.tranches[-1]["price"] if self.tranches else 0.0
        add_pnl = price / last_add_price - 1.0 if last_add_price > 0 else 0.0
        high_gap = price / self.high_water_price - 1.0 if self.high_water_price > 0 else 0.0
        age = min(self.days_in_position / 63.0, 1.0)
        last_tranche_pnl = add_pnl
        n_tranches_ratio = min(len(self.tranches) / 10.0, 1.0)
        return (
            float(age),
            float(avg_pnl),
            float(add_pnl),
            float(high_gap),
            float(last_tranche_pnl),
            float(n_tranches_ratio),
        )


class EpisodeConfig(NamedTuple):
    pkl_path: str
    start_idx: int
    end_idx: int
    lookback: int = LOOKBACK
    cash: float = 100_000.0
    asset_idx: int = 0
    observation_window: int = 1
    decision_frequency: int = 1
    nday_window: int = 5
    execution_config: Optional[ExecutionConfig] = None


def build_price_path(df: pd.DataFrame, base_price: float = 100.0) -> np.ndarray:
    returns = df["return"].fillna(0.0).to_numpy(dtype=np.float32)
    return np.cumprod(1.0 + returns, dtype=np.float32) * np.float32(base_price)


class SoxlSimulator(Simulator):
    """轻量级训练模拟器。"""

    def __init__(self, initial: EpisodeConfig, **kwargs):
        super().__init__(initial, **kwargs)
        self.df = pd.read_pickle(initial.pkl_path).iloc[initial.start_idx : initial.end_idx].copy()
        self.price_path = build_price_path(self.df)
        self.lookback = initial.lookback
        self.cash0 = initial.cash
        self.idx = self.lookback
        self.pos = 0.0
        self.val = self.cash0
        self.prev_val = self.cash0
        self.max_val = self.cash0
        self.last_turnover = 0.0
        self.last_step_return = 0.0
        self.last_market_return = 0.0
        self.flat_steps = 0
        self.prev_regime_risk = 0.0
        self.prev_drawdown = 0.0
        self.prev_pos = 0.0
        self.buy_cooldown = 0
        self.trade_lock = 0
        self.asset_idx = getattr(initial, "asset_idx", 0)
        self.observation_window = max(int(getattr(initial, "observation_window", 1)), 1)
        self.obs_dim = len(FEATURE_COLUMNS) * self.observation_window + EXTRA_OBS_DIM + N_ASSETS
        self.ledger = TrancheLedger()
        self.decision_frequency = max(int(getattr(initial, "decision_frequency", 1)), 1)
        self.nday_window = max(int(getattr(initial, "nday_window", 5)), 1)
        self.exec_cfg = initial.execution_config or DEFAULT_EXECUTION
        self.period_start_val = self.cash0
        self.period_baseline_val = self.cash0
        self.period_start_pos = 0.0
        self._is_decision_day = True
        self.nday_start_val = self.cash0
        self.nday_step_counter = 0
        self.nday_reward_val = 0.0
        self.episode_start_val = self.cash0

    def step(self, action: float) -> None:
        action = float(np.clip(action, 0, 1))
        if self.idx >= len(self.df) - 1:
            return

        is_decision_day = True
        if self.decision_frequency > 1:
            days_since_start = self.idx - self.lookback
            if days_since_start % self.decision_frequency != 0:
                is_decision_day = False
                action = self.pos

        if is_decision_day:
            self.period_start_val = self.val
            self.period_start_pos = self.pos
            self.period_baseline_val = self.val
        self._is_decision_day = is_decision_day

        row = self.df.iloc[self.idx]
        price = float(self.price_path[self.idx])
        cum_drop_30 = float(np.nan_to_num(row["cum_drop_30"]))
        direction = "buy" if action > self.pos else ("sell" if action < self.pos else "neutral")
        min_rebalance = rebalance_threshold(
            float(np.nan_to_num(row["regime_risk"])),
            float(np.nan_to_num(row["realized_vol_10"])),
            abs(float(np.nan_to_num(row["trend_gap_21"]))),
            direction=direction,
            config=self.exec_cfg,
        )
        if self.trade_lock > 0:
            if abs(action - self.pos) >= min_rebalance:
                action = self.pos
            self.trade_lock -= 1
        if self.buy_cooldown > 0:
            if action > self.pos:
                action = self.pos
            self.buy_cooldown -= 1
        if abs(action - self.pos) < min_rebalance:
            action = self.pos

        self.prev_drawdown = abs(self.val / max(self.max_val, 1e-8) - 1.0)
        self.prev_pos = self.pos
        next_ret = float(self.df.iloc[self.idx + 1]["return"])
        turnover = abs(action - self.pos)
        gross_ret = action * next_ret
        net_ret = gross_ret - turnover * 0.001
        self.prev_val = self.val
        self.val *= 1.0 + net_ret
        self.max_val = max(self.max_val, self.val)
        self.last_turnover = turnover
        self.last_step_return = net_ret
        self.last_market_return = next_ret
        self.flat_steps = 0 if turnover > 0.02 else self.flat_steps + 1
        self.period_baseline_val *= 1.0 + self.period_start_pos * next_ret
        self.ledger.update(self.pos, action, price)

        if action > self.pos + 1e-6:
            regime = execution_regime(
                float(np.nan_to_num(row["regime_risk"])),
                float(np.nan_to_num(row["realized_vol_10"])),
                abs(float(np.nan_to_num(row["trend_gap_21"]))),
                config=self.exec_cfg,
            )
            self.buy_cooldown = execution_buy_cooldown_days(cum_drop_30, self.exec_cfg, regime)

        turnover = abs(action - self.pos)
        if turnover > self.exec_cfg.min_trade_turnover:
            regime = execution_regime(
                float(np.nan_to_num(row["regime_risk"])),
                float(np.nan_to_num(row["realized_vol_10"])),
                abs(float(np.nan_to_num(row["trend_gap_21"]))),
                config=self.exec_cfg,
            )
            self.trade_lock = execution_trade_lock_days(self.exec_cfg, regime)

        self.pos = action
        self.idx += 1
        self.nday_step_counter += 1
        if self.nday_step_counter % self.nday_window == 0:
            nday_return = (self.val - self.nday_start_val) / max(self.nday_start_val, 1e-8)
            self.nday_reward_val = nday_return
            self.nday_start_val = self.val
        else:
            self.nday_reward_val = 0.0

    def get_state(self):
        if self.idx >= len(self.df):
            obs = np.zeros(self.obs_dim, dtype=np.float32)
            return {
                "obs": obs,
                "pos": self.pos,
                "val": self.prev_val,
                "pval": self.prev_val,
                "turnover": 0.0,
                "drawdown": 0.0,
                "step_return": 0.0,
                "regime_risk": 0.0,
                "regime_delta": 0.0,
                "flat_steps": 0.0,
                "done": True,
                "nday_reward": 0.0,
                "episode_start_val": self.episode_start_val,
            }
        row = self.df.iloc[self.idx]
        underwater = self.val / max(self.max_val, 1e-8) - 1.0
        regime_risk = float(np.nan_to_num(row["regime_risk"]))
        regime_delta = regime_risk - self.prev_regime_risk
        self.prev_regime_risk = regime_risk
        obs = make_observation(
            self.df,
            self.idx,
            self.pos,
            self.val,
            self.max_val,
            self.ledger,
            self.price_path,
            asset_idx=self.asset_idx,
            observation_window=self.observation_window,
        )
        rule_pos = rule_based_position(obs)
        return {
            "obs": obs,
            "pos": self.pos,
            "val": self.val,
            "pval": self.prev_val,
            "turnover": self.last_turnover,
            "drawdown": abs(underwater),
            "prev_drawdown": self.prev_drawdown,
            "step_return": self.last_step_return,
            "regime_risk": regime_risk,
            "regime_delta": regime_delta,
            "flat_steps": float(self.flat_steps),
            "prev_pos": self.prev_pos,
            "market_return": self.last_market_return,
            "trade_lock": float(self.trade_lock),
            "buy_cooldown": float(self.buy_cooldown),
            "period_alpha": (self.val - self.period_baseline_val) / self.period_start_val if self.period_start_val > 0 else 0.0,
            "is_decision_day": 1.0 if self._is_decision_day else 0.0,
            "rule_pos": rule_pos,
            "rule_return": rule_pos * self.last_market_return,
            "done": False,
            "nday_reward": self.nday_reward_val,
            "episode_start_val": self.episode_start_val,
        }

    def done(self) -> bool:
        return self.idx >= len(self.df) - 1


class SoxlState(StateInterpreter):
    def __init__(self, obs_dim: int):
        super().__init__()
        self.obs_dim = obs_dim

    @property
    def observation_space(self):
        return spaces.Box(-np.inf, np.inf, (self.obs_dim,), dtype=np.float32)

    def interpret(self, state):
        return state["obs"]


class PositionAction(ActionInterpreter):
    def __init__(
        self,
        n_actions: int = N_ACTIONS,
        position_cap_config: Optional[PositionCapConfig] = None,
        action_levels: Optional[np.ndarray] = None,
        hard_floor: Optional[float] = None,
        use_hybrid_overlay: bool = False,
        delta_levels: Optional[np.ndarray] = None,
    ):
        super().__init__()
        self.n_actions = n_actions
        self.position_cap_config = position_cap_config
        self.action_levels = action_levels if action_levels is not None else BASE_ACTION_LEVELS
        self.hard_floor = hard_floor
        self.use_hybrid_overlay = use_hybrid_overlay
        self.delta_levels = np.asarray(
            delta_levels if delta_levels is not None else [-0.15, 0.0, 0.15],
            dtype=np.float32,
        )

    @property
    def action_space(self):
        return spaces.Discrete(self.n_actions)

    def interpret(self, state, action):
        if self.use_hybrid_overlay:
            rule_pos = rule_based_position(state["obs"])
            clipped = int(np.clip(action, 0, len(self.delta_levels) - 1))
            delta = float(self.delta_levels[clipped])
            return float(np.clip(rule_pos + delta, 0.0, 1.0))
        return action_to_position(
            int(action),
            state["obs"],
            config=self.position_cap_config,
            action_levels=self.action_levels,
            hard_floor=self.hard_floor,
        )


def build_initial_states(
    sources: List[Tuple[Path, pd.DataFrame, int]],
    profile: StrategyConfig,
    smoke: bool,
) -> List[EpisodeConfig]:
    episode_len = profile.episode_len
    all_states: List[EpisodeConfig] = []
    sampling = profile.sampling
    observation_window = max(int(profile.model.observation_window), 1)
    decision_frequency = max(int(profile.decision_frequency), 1)
    for pkl_path, df, asset_idx in sources:
        stride = int(sampling.episode_stride) if int(sampling.episode_stride) > 0 else episode_len
        max_start = max(len(df) - episode_len, 0)
        start_indices = list(range(0, max_start + 1, stride))
        if not start_indices:
            start_indices = [0]
        if smoke:
            start_indices = start_indices[:4]
        for start_idx in start_indices:
            end_idx = min(start_idx + episode_len, len(df))
            if (end_idx - start_idx) <= LOOKBACK + 2:
                continue
            nday_window = max(int(profile.reward.nday_window), 1) if profile.reward.use_nday_reward else 1
            state = EpisodeConfig(
                str(pkl_path),
                start_idx,
                end_idx,
                asset_idx=asset_idx,
                observation_window=observation_window,
                decision_frequency=decision_frequency,
                nday_window=nday_window,
                execution_config=profile.execution,
            )
            all_states.append(state)
            segment = df.iloc[start_idx:end_idx]
            risk_score = float(segment["regime_risk"].mean())
            drawdown_score = abs(float(segment["cum_drop_30"].min()))
            realized_vol = float(segment["realized_vol_10"].mean())
            anchor_gap = float(segment["anchor_gap_180"].mean())
            trend_gap = float(segment["trend_gap_21"].mean())
            risk_head = float(segment["regime_risk"].head(min(len(segment), 20)).mean())
            risk_tail = float(segment["regime_risk"].tail(min(len(segment), 20)).mean())
            risk_release = risk_head - risk_tail
            persistent_stress = max(
                float(segment["drawdown_streak_63"].max()),
                float(segment["below_ma63_days"].max()),
                float(segment["rolling_new_low_ratio_63"].mean()),
                float(segment["negative_gap_frequency_21"].mean()),
            )
            is_bull_trend = (
                anchor_gap > sampling.bull_anchor_threshold
                and trend_gap > sampling.bull_trend_threshold
                and realized_vol < sampling.bull_vol_threshold
                and risk_score < sampling.bull_risk_threshold
            )
            is_rebound = (
                anchor_gap < sampling.rebound_anchor_ceiling
                and trend_gap > sampling.rebound_trend_floor
                and risk_score < sampling.rebound_risk_ceiling
                and risk_release > sampling.rebound_risk_release
                and drawdown_score > sampling.mild_drawdown_threshold
            )
            repeats = 1
            if sampling.regime_stratified:
                if (
                    risk_score > sampling.severe_risk_threshold
                    or drawdown_score > sampling.severe_drawdown_threshold
                    or persistent_stress > sampling.persistent_stress_days
                ):
                    repeats = max(repeats, sampling.severe_repeat)
                elif (
                    risk_score > sampling.mild_risk_threshold
                    or drawdown_score > sampling.mild_drawdown_threshold
                ):
                    repeats = max(repeats, sampling.mild_repeat)
            if sampling.phase_balanced:
                if is_rebound:
                    repeats = max(repeats, sampling.rebound_repeat)
                elif is_bull_trend:
                    repeats = max(repeats, sampling.trend_repeat)
            else:
                if risk_score > 0.58 or drawdown_score > 0.18:
                    repeats += 1
                if risk_score > 0.42 and drawdown_score > 0.10:
                    repeats += 1
            if not smoke:
                all_states.extend([state] * max(repeats - 1, 0))
    if not smoke:
        random.shuffle(all_states)
    return all_states


__all__ = [
    "EpisodeConfig",
    "EpisodeSummary",
    "PositionAction",
    "SoxlSimulator",
    "SoxlState",
    "TrancheLedger",
    "build_initial_states",
    "build_price_path",
]
