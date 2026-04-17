from __future__ import annotations

"""Daily SOXL reinforcement-learning environment and policy helpers.

【设计核心】actor 不直接输出股数，而是输出一个离散的“意图”，
再由 action_to_position 将其映射到一个受市场状态约束的仓位区间。
这个区间由两部分共同决定：
  1) 动态 floor/cap（基于 regime_risk、anchor_gap、vol_coef 等）
  2) 硬性的 drawdown cap（基于 cum_drop_30 和 PositionCapConfig）
这样即使策略漂移，风险控制系统仍然可以把仓位强行压在安全范围内。

【已知优化方向】
  1. 踏空惩罚：当前模型在强趋势中仓位偏低（最大仅 0.59），需要加入
     对低仓位的 opportunity-cost 惩罚。
  2. 动态 cap：drawdown_position_cap 的 min_cap 目前是固定值 0.4，
     未来可以改为根据 trend_strength / realized_vol 动态调整。
  3. 多资产训练：当前只在 SOXL 上训练，泛化到 TQQQ/TECL 时跑输 B&H。
  4. 网络升级：当前是两层 MLP，未来可替换为 LSTM/Transformer 以利用
     60 天的 LOOKBACK 时序信息。
"""

from dataclasses import dataclass
import random
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from gymnasium import spaces
from qlib.rl.interpreter import ActionInterpreter, StateInterpreter
from qlib.rl.reward import Reward
from qlib.rl.simulator import Simulator
from tianshou.policy import PPOPolicy
from tianshou.utils.net.common import Net
from tianshou.utils.net.discrete import Actor, Critic

from .data import FEATURE_COLUMNS, STATIC_INDEX
from .settings import EXTRA_OBS_DIM, LOOKBACK, N_ACTIONS, N_ASSETS, PositionCapConfig, StrategyConfig

BASE_ACTION_LEVELS = np.linspace(0.0, 1.0, N_ACTIONS, dtype=np.float32)
DEFAULT_POSITION_CAP = PositionCapConfig()
EXTRA_INDEX = {
    "lookback_mean_return": len(FEATURE_COLUMNS) + 0,
    "lookback_return_skew": len(FEATURE_COLUMNS) + 1,
    "lookback_downside": len(FEATURE_COLUMNS) + 2,
    "episode_progress": len(FEATURE_COLUMNS) + 3,
    "position": len(FEATURE_COLUMNS) + 4,
    "underwater": len(FEATURE_COLUMNS) + 5,
    "holding_age": len(FEATURE_COLUMNS) + 6,
    "avg_entry_pnl": len(FEATURE_COLUMNS) + 7,
    "last_add_pnl": len(FEATURE_COLUMNS) + 8,
    "high_gap": len(FEATURE_COLUMNS) + 9,
    "last_tranche_pnl": len(FEATURE_COLUMNS) + 10,
    "n_tranches": len(FEATURE_COLUMNS) + 11,
    "drawdown_velocity_5": len(FEATURE_COLUMNS) + 12,
    "regime_risk_max_60d": len(FEATURE_COLUMNS) + 13,
    "vol_velocity_5": len(FEATURE_COLUMNS) + 14,
}


@dataclass(frozen=True)
class EpisodeSummary:
    account: np.ndarray
    positions: List[float]


@dataclass
class TrancheLedger:
    """持仓账本（带分段成本栈）：模拟 soxl_backtest 的 LIFO 获利了结逻辑。

    每次加仓生成一个新的 tranche；减仓时优先卖出最近一笔 tranche。
    这让 reward 能够识别“最后一笔买入已获利”的反弹卖出机会，
    即使整体仓位仍然套牢。
    """

    tranches: List[Dict[str, float]] = None
    high_water_price: float = 0.0
    days_in_position: int = 0

    def __post_init__(self):
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
        return float(age), float(avg_pnl), float(add_pnl), float(high_gap), float(last_tranche_pnl), float(n_tranches_ratio)


class EpisodeConfig(NamedTuple):
    pkl_path: str
    start_idx: int
    end_idx: int
    lookback: int = LOOKBACK
    cash: float = 100_000.0
    asset_idx: int = 0


def rebalance_threshold(regime_risk: float, realized_vol: float, trend_strength: float, direction: str = "neutral") -> float:
    """抑制无效调仓：对称高阈值，减少 whipsaw 与摩擦成本。

    基于 A+B 实验（symmetric 0.04/0.08）的回滚修复： asymmetric
    分支 (buy 0.02 / sell 0.06) 已被证明会显著降低收益（461.7%）。
    """
    threshold = 0.04
    if regime_risk < 0.35 and realized_vol < 0.45 and trend_strength < 0.04:
        threshold = 0.08
    if regime_risk > 0.60 or realized_vol > 0.70 or trend_strength > 0.10:
        threshold = 0.012
    return threshold


def action_band_from_obs(obs: np.ndarray) -> Tuple[float, float]:
    """将市场状态映射为可交易的仓位区间 [floor, cap]。

    actor 只决定在该区间内的具体位置。区间本身编码了硬先验：
      - 健康上升趋势中保持一定底仓（floor 提升）
      - 高风险或高波动时压缩天花板（cap 降低）
    
    【优化方向 1】当前 cap 最高只到 1.0，但 drawdown_position_cap 会进一步压低。
    在泛化测试中发现模型最大仓位仅 ~0.59，牛市踏空严重。
    建议：当 anchor_gap>0、trend_support 强、realized_vol<0.4 时，
    允许 cap 提升到 0.85 ~ 1.0，同时放松 drawdown cap。
    
    【优化方向 2】如果做多资产训练，需要把 ETF 的波动率画像作为额外输入，
    让模型自动调整 floor/cap 的敏感度。
    """
    regime_risk = float(np.nan_to_num(obs[STATIC_INDEX["regime_risk"]]))
    anchor_gap = float(np.nan_to_num(obs[STATIC_INDEX["anchor_gap_180"]]))
    anchor_slope = float(np.nan_to_num(obs[STATIC_INDEX["anchor_slope_180"]]))
    cum_drop_30 = float(np.nan_to_num(obs[STATIC_INDEX["cum_drop_30"]]))
    vix_zscore = float(np.nan_to_num(obs[STATIC_INDEX["vix_zscore"]]))
    vol_coef = float(np.nan_to_num(obs[STATIC_INDEX["vol_coef_21"]]))

    floor = 0.30
    cap = 0.95

    if anchor_gap > 0.02 and anchor_slope >= 0:
        floor = 0.40
        cap = 1.00
    if regime_risk > 0.72 or (anchor_gap < -0.06 and anchor_slope < 0):
        floor = 0.30
        cap = 0.80
    if cum_drop_30 < -0.18 and vix_zscore < 2.8:
        floor = max(floor, 0.30)
        cap = 1.00
    if vol_coef > 1.45:
        floor = max(0.20, floor - 0.05)
        cap = min(cap, 0.90)

    return floor, cap


def drawdown_position_cap(
    cumulative_drop: float,
    obs: Optional[np.ndarray] = None,
    config: Optional[PositionCapConfig] = None,
) -> float:
    """硬性仓位上限：在持续回撤时强制压低仓位，防止死扛。

    当前使用幂函数 (scale^curve_power) 进行 shaping：
      - drop <= start_drawdown (默认 0.10) 时 cap = 1.0
      - drop >= full_drawdown (默认 0.30) 时 cap = effective_min_cap
    
    【P0 优化】effective_min_cap 不再是固定值，而是根据市场状态动态调整：
      - 牛市（realized_vol < 0.40 且 anchor_gap > 0.01）：min_cap 提升到 0.65
      - 高风险市（regime_risk > 0.65）：min_cap 收紧到 0.30
    这样在不同市场体制下获得更好的收益-回撤平衡。
    """
    cfg = config or DEFAULT_POSITION_CAP
    effective_min_cap = float(cfg.min_cap)

    # 动态调整 min_cap（P0 优化）
    # 【第二轮调参】牛市判定增加 trend_gap>0 条件，防止在震荡市中误放宽 cap。
    if obs is not None and len(obs) > 0:
        realized_vol = float(np.nan_to_num(obs[STATIC_INDEX["realized_vol_10"]]))
        anchor_gap = float(np.nan_to_num(obs[STATIC_INDEX["anchor_gap_180"]]))
        trend_gap = float(np.nan_to_num(obs[STATIC_INDEX["trend_gap_21"]]))
        regime_risk = float(np.nan_to_num(obs[STATIC_INDEX["regime_risk"]]))
        if (realized_vol < cfg.bull_vol_threshold
                and anchor_gap > cfg.bull_anchor_threshold
                and trend_gap > 0.0):
            effective_min_cap = max(effective_min_cap, float(cfg.bull_min_cap))
        if regime_risk > cfg.bear_regime_threshold:
            effective_min_cap = min(effective_min_cap, float(cfg.bear_min_cap))

    drop = max(float(cumulative_drop), 0.0)
    if drop <= cfg.start_drawdown:
        return 1.0
    if drop >= cfg.full_drawdown:
        return float(effective_min_cap)
    span = max(cfg.full_drawdown - cfg.start_drawdown, 1e-6)
    scale = (drop - cfg.start_drawdown) / span
    curve_power = max(float(getattr(cfg, "curve_power", 1.0)), 1e-6)
    shaped_scale = scale**curve_power
    return float(1.0 - shaped_scale * (1.0 - effective_min_cap))


def action_to_position(action: int, obs: np.ndarray, config: Optional[PositionCapConfig] = None) -> float:
    """将离散的 action（0~10）映射为连续仓位 [0, 1]。

    映射公式：position = floor + base * (cap - floor)
    其中 base = action / 10，cap 同时受 action_band 和 drawdown_position_cap 约束。
    
    【P0 优化】drawdown_position_cap 现在接收 obs，支持根据市场状态动态调整 cap。
    在牛市中 effective_min_cap 提升， actor 即使选择最大 action 也可以达到更高仓位。
    
    【注意】N_ACTIONS=11 意味着最小粒度为 10%，这既是量化交易中的手数约束，
    也起到了自然正则化作用，避免过度频繁的微调。
    """
    base = float(BASE_ACTION_LEVELS[int(np.clip(action, 0, N_ACTIONS - 1))])
    floor, cap = action_band_from_obs(obs)
    cumulative_drop = max(-float(np.nan_to_num(obs[STATIC_INDEX["cum_drop_30"]])), 0.0)
    cap = min(cap, drawdown_position_cap(cumulative_drop, obs=obs, config=config))
    return float(np.clip(floor + base * (cap - floor), 0.0, 1.0))


def build_price_path(df: pd.DataFrame, base_price: float = 100.0) -> np.ndarray:
    returns = df["return"].fillna(0.0).to_numpy(dtype=np.float32)
    return np.cumprod(1.0 + returns, dtype=np.float32) * np.float32(base_price)


class SoxlSimulator(Simulator):
    """轻量级日内模拟器，用于 PPO 训练阶段快速 rollout。

    与 heavyweight 的 qlib backtest 不同，这里只做最简单的数学模拟：
      nav_{t+1} = nav_t * (1 + position_t * return_{t+1} - turnover * 0.001)
    0.1% 的假设交易成本已经内嵌其中。
    
    【优化方向】当前模拟器没有考虑滑点、开盘缺口、涨停跌停等微观结构。
    如果未来要上线实盘，需要在 qlib backtest 中做更严格的冲击成本测试。
    """
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
        self.flat_steps = 0
        self.prev_regime_risk = 0.0
        self.prev_drawdown = 0.0
        self.prev_pos = 0.0
        self.buy_cooldown = 0
        self.trade_lock = 0
        self.nf = len(FEATURE_COLUMNS)
        self.asset_idx = getattr(initial, "asset_idx", 0)
        self.ledger = TrancheLedger()

    def step(self, action: float) -> None:
        """执行一步模拟。

        逻辑顺序：
          1. 用 rebalance_threshold 过滤掉过小的调仓
          2. 用次日收益计算净值变化（含 0.1% 换手成本）
          3. 更新账本和内部状态
        
        【优化方向】当前成本假设为固定 0.1%。如果交易次数继续优化降低，
        这个假设是偏保守的；但如果未来换手率上升，需要考虑滑点模型。
        """
        action = float(np.clip(action, 0, 1))
        if self.idx >= len(self.df) - 1:
            return
        row = self.df.iloc[self.idx]
        price = float(self.price_path[self.idx])
        vol_coef = float(np.nan_to_num(row["vol_coef_21"]))
        cum_drop_30 = float(np.nan_to_num(row["cum_drop_30"]))
        direction = "buy" if action > self.pos else ("sell" if action < self.pos else "neutral")
        min_rebalance = rebalance_threshold(
            float(np.nan_to_num(row["regime_risk"])),
            float(np.nan_to_num(row["realized_vol_10"])),
            abs(float(np.nan_to_num(row["trend_gap_21"]))),
            direction=direction,
        )
        # Trade lock takes precedence over any action change
        if self.trade_lock > 0:
            if abs(action - self.pos) >= min_rebalance:
                action = self.pos
            self.trade_lock -= 1
        # Buy cooldown: post-buy cooldown period
        if self.buy_cooldown > 0:
            if action > self.pos:
                action = self.pos
            self.buy_cooldown -= 1
        if abs(action - self.pos) < min_rebalance:
            action = self.pos
        # PBRS: record drawdown *before* value update for potential shaping
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
        self.flat_steps = 0 if turnover > 0.02 else self.flat_steps + 1
        self.ledger.update(self.pos, action, price)
        # Set cooldown if increasing position (rule-based inspired)
        if action > self.pos + 1e-6:
            if cum_drop_30 < -0.15:
                # Deep cumulative drop => long cooldown (up to ~20 days)
                self.buy_cooldown = min(20, max(3, int(abs(cum_drop_30) * 30)))
            else:
                self.buy_cooldown = 1
        # Trade lock: freeze position for N days after any material trade
        turnover = abs(action - self.pos)
        if turnover > 0.02:
            self.trade_lock = 3
        self.pos = action
        self.idx += 1

    def get_state(self):
        """构造当前状态向量（observation）。

        观测向量由三部分拼接而成（总维度 = nf + EXTRA_OBS_DIM = 48）：
          1. 38 维 engineered market features（来自 data.py 的 FEATURE_COLUMNS）
          2. 4 维 lookback 统计量：均值 return、偏度、下行波动、进度百分比
          3. 6 维 portfolio 状态：当前仓位、水下幅度、ledger（age, avg_pnl, add_pnl, high_gap）
        
        【优化方向】当前只使用了 lb["return"].mean/skew，没有利用完整的 60 天时序。
        如果要升级网络为 LSTM，可在此返回 (60, nf) 的序列矩阵，而非展平向量。
        """
        if self.idx >= len(self.df):
            obs = np.zeros(self.nf + EXTRA_OBS_DIM + N_ASSETS, dtype=np.float32)
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
            }
        row = self.df.iloc[self.idx]
        lb = self.df.iloc[self.idx - self.lookback : self.idx]
        underwater = self.val / max(self.max_val, 1e-8) - 1.0
        downside_lookback = lb["return"].where(lb["return"] < 0, 0.0).pow(2).mean()
        regime_risk = float(np.nan_to_num(row["regime_risk"]))
        regime_delta = regime_risk - self.prev_regime_risk
        self.prev_regime_risk = regime_risk
        ledger_snapshot = self.ledger.snapshot(float(self.price_path[self.idx]))
        obs = make_observation(
            self.df,
            self.idx,
            self.pos,
            self.val,
            self.max_val,
            self.ledger,
            self.price_path,
            asset_idx=self.asset_idx,
        )
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
            "trade_lock": float(self.trade_lock),
            "buy_cooldown": float(self.buy_cooldown),
            "done": False,
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
    def __init__(self, n_actions: int = N_ACTIONS, position_cap_config: Optional[PositionCapConfig] = None):
        super().__init__()
        self.n_actions = n_actions
        self.position_cap_config = position_cap_config

    @property
    def action_space(self):
        return spaces.Discrete(self.n_actions)

    def interpret(self, state, action):
        return action_to_position(int(action), state["obs"], config=self.position_cap_config)


class ContinuousReward(Reward):
    """连续型奖励函数：评估交易质量，而不仅是收益率。

    设计原则是有意不对称的：
      - 在健康趋势中，允许一定程度的受控回撤
      - 但对长期空仓无所事事（stale inactivity）和长期套牢仓位惩罚更重
    
    【泛化测试发现的问题】
      1. 牛市踏空：TQQQ/TECL 在 2020-2021、2023-2024 大涨中仓位偏低，
         当前 reward 对低仓位没有足够惩罚。
      2. 换手率偏高：SOXL 训练出 578 次交易，成本侵蚀收益。
    
    【建议优化】
      - 加入 opportunity_cost_penalty：当 anchor_gap>0.03 且 pos<0.30 时扣分
      - 提高 turnover_penalty / smoothness_penalty 权重以降低交易频率
      - 根据 ETF 品种动态调整 risk_budget_penalty（科技股容忍度 vs 宽基指数）
    """
    def __init__(self, profile: StrategyConfig):
        super().__init__()
        self.profile = profile

    def reward(self, state):
        """计算单步奖励。"""
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

        # === 核心收益项 ===
        # === 核心收益项 ===
        reward = pnl_ret * 100.0 * cfg.return_scale

        # === PBRS (Potential-Based Reward Shaping) ===
        # 以 -drawdown 为 potential，理论上不改变最优策略，但能加速学习
        prev_drawdown = float(state.get("prev_drawdown", drawdown))
        pbrs_delta = (-drawdown) - (-prev_drawdown)
        reward += pbrs_delta * cfg.pbrs_coef

        # === 交易成本与换手率惩罚 ===
        reward -= turnover * cfg.transaction_cost * 100.0          # 直接交易成本
        reward -= turnover * cfg.turnover_penalty                  # 额外换手惩罚
        reward -= (turnover**2) * cfg.smoothness_penalty           # 鼓励平滑、减少日内反复
        # 动态 activity bonus：只在市场 regime 明显切换或高波动时才奖励换手
        if cfg.gated_activity_bonus:
            regime_switch_flag = 1.0 if (regime_delta > cfg.activity_gate_regime_delta_threshold or realized_vol > cfg.activity_gate_vol_threshold) else 0.0
            activity_bonus_eff = min(turnover, 0.30) * min(activity_signal, 2.2) * cfg.activity_bonus * regime_switch_flag
        else:
            activity_bonus_eff = min(turnover, 0.30) * min(activity_signal, 2.2) * cfg.activity_bonus
        reward += activity_bonus_eff

        # === 回撤与下行风险控制 ===
        reward -= excess_drawdown * cfg.drawdown_penalty           # 超过容忍度的回撤重罚
        reward -= downside_signal * cfg.downside_penalty           # 高下行波动环境惩罚
        reward -= max(pos - (1.0 - 0.65 * regime_risk), 0.0) * cfg.risk_budget_penalty    # 风险预算超支
        reward -= max(pos - 0.85, 0.0) * cfg.high_exposure_penalty # 硬性的高仓位惩罚

        # === 趋势保护奖励 ===
        # 在健康趋势中，即使有小回撤，只要不超过容忍度且仓位不过低，给予微弱正奖励
        reward += min(drawdown, tolerated_drawdown) * min(pos, 0.80) * min(trend_support * 2.5, 0.18)
        
        # === 持仓质量惩罚 ===
        reward -= stale_position * (6.0 + 4.0 * activity_signal)   # 长期空仓且在震荡中无所事事
        reward -= trapped_loser * (18.0 + 6.0 * regime_risk)       # 长期套牢且从高点回落


        # === 低仓位惩罚（防止踏空）===
        trend_gap = float(state["obs"][STATIC_INDEX["trend_gap_21"]])
        if anchor_gap > cfg.opportunity_threshold_anchor and pos < cfg.opportunity_threshold_pos:
            reward -= cfg.opportunity_cost_penalty
        if (trend_gap > cfg.opportunity_threshold_trend and realized_vol < 0.45
                and pos < cfg.opportunity_threshold_pos + 0.05):
            reward -= cfg.opportunity_cost_penalty * 0.40
        if anchor_gap > 0.02 and regime_risk < 0.45 and pos < 0.30:
            reward -= 0.12
        if cum_drop_30 < -0.18 and regime_risk < 0.80 and pos < 0.20:
            reward -= 0.10

        # === Tranche-aware profit-taking / underwater-sell guardrails ===
        last_tranche_pnl = float(state["obs"][EXTRA_INDEX["last_tranche_pnl"]])
        prev_pos = float(state.get("prev_pos", pos))
        if turnover > 0.02 and pos < prev_pos - 1e-6:
            # Selling: strongly reward taking profit on the last tranche,
            # penalize dumping an underwater last tranche.
            if last_tranche_pnl > cfg.tranche_profit_threshold:
                reward += cfg.profit_taking_bonus * min(last_tranche_pnl * 12.0, 2.0)
            elif last_tranche_pnl < -0.05:
                reward -= cfg.underwater_sell_penalty * min(abs(last_tranche_pnl) * 8.0, 1.5)

        # === Panic-buy bonus ===
        if turnover > 0.02 and pos > prev_pos + 1e-6:
            if (cum_drop_30 < cfg.panic_buy_cum_drop_threshold
                    and regime_risk > cfg.panic_buy_regime_risk_threshold):
                panic_depth = min(abs(cum_drop_30) - abs(cfg.panic_buy_cum_drop_threshold), 0.15)
                reward += cfg.panic_buy_bonus * (1.0 + panic_depth * 5.0)

        # === Volatility T-trading bonus ===
        if realized_vol > 0.50 and turnover > 0.03:
            reward += cfg.vol_trading_bonus * turnover
        
        # === 不活跃惩罚 ===
        if flat_steps >= 4:
            flat_drag = min((flat_steps - 3.0) / 6.0, 1.6)
            reward -= flat_drag * (0.20 + activity_signal * 0.9) * cfg.inactivity_penalty
        if flat_steps >= 8 and activity_signal > 0.12:
            reward -= min(flat_steps / 8.0, 1.5) * cfg.inactivity_penalty
        if holding_age >= 0.35 and turnover < 0.02 and abs(avg_entry_pnl) < 0.02:
            reward -= (0.16 + holding_age * 0.14) * cfg.inactivity_penalty
        
        return float(reward)


def build_initial_states(
    sources: List[Tuple[Path, pd.DataFrame, int]],
    profile: StrategyConfig,
    smoke: bool,
) -> List[EpisodeConfig]:
    """生成训练用的 episode 起止点，并对高压力区间做过采样。

    过采样只基于训练窗口内部可见的状态（regime_risk、cum_drop_30），
    不偷看未来标签，符合仓库的 non-leak 规则。
    
    现在支持从多只 ETF 的 DataFrame 中混合采样，并在 EpisodeConfig 中
    携带 asset_idx，实现跨品种联合训练。
    """
    episode_len = profile.episode_len
    all_states: List[EpisodeConfig] = []
    for pkl_path, df, asset_idx in sources:
        frame_count = len(df) // episode_len
        if smoke:
            frame_count = min(frame_count, 4)
        for i in range(frame_count):
            start_idx = i * episode_len
            end_idx = min((i + 1) * episode_len, len(df))
            if (end_idx - start_idx) <= LOOKBACK + 2:
                continue
            state = EpisodeConfig(str(pkl_path), start_idx, end_idx, asset_idx=asset_idx)
            all_states.append(state)
            segment = df.iloc[start_idx:end_idx]
            risk_score = float(segment["regime_risk"].mean())
            drawdown_score = abs(float(segment["cum_drop_30"].min()))
            repeats = 1
            if risk_score > 0.58 or drawdown_score > 0.18:
                repeats += 1
            if risk_score > 0.42 and drawdown_score > 0.10:
                repeats += 1
            if not smoke:
                all_states.extend([state] * max(repeats - 1, 0))
    if not smoke:
        random.shuffle(all_states)
    return all_states


class RegimeMoENet(torch.nn.Module):
    """轻量级 Mixture-of-Experts 网络，支持 Noisy Top-1 稀疏路由与 load-balancing loss.

    升级点（基于 arXiv 2503.07137 / Switch Transformer）：
      - Noisy Top-1 gating：训练时在 gate logits 上加高斯噪声，防止专家坍塌。
      - Load-balancing auxiliary loss：标准 Switch-Transformer balance loss。
      - Shared expert：始终激活一个共享专家，保留跨 regime 的公共知识。
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_size: int = 64,
        n_experts: int = 3,
        device: str = "cpu",
        top_k: int = 1,
        noise_std: float = 0.1,
        shared_hidden_size: int = 16,
        regime_risk_idx: Optional[int] = None,
        hard_regime_routing: bool = False,
    ):
        super().__init__()
        self.n_experts = n_experts
        self.hidden_size = hidden_size
        self.output_dim = hidden_size  # required by tianshou Actor/Critic
        self.device = device
        self.top_k = top_k
        self.noise_std = noise_std
        self.regime_risk_idx = regime_risk_idx
        self.hard_regime_routing = hard_regime_routing
        self.gate = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, n_experts),
        )
        self.experts = torch.nn.ModuleList(
            [torch.nn.Sequential(torch.nn.Linear(obs_dim, hidden_size), torch.nn.ReLU()) for _ in range(n_experts)]
        )
        self.shared_expert = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, shared_hidden_size),
            torch.nn.ReLU(),
        )
        self.output = torch.nn.Linear(hidden_size + shared_hidden_size, hidden_size)
        self._last_lb_loss: Optional[torch.Tensor] = None

    def _load_balance_loss(self, gate_probs: torch.Tensor) -> torch.Tensor:
        """Switch-Transformer style auxiliary balance loss."""
        # gate_probs: (batch, n_experts) after softmax over all experts
        f = gate_probs.mean(dim=0)  # fraction of batch routed to each expert
        p = gate_probs.mean(dim=0)  # mean gate probability for each expert
        loss = self.n_experts * (f * p).sum()
        return loss

    def forward(self, obs, state: Any = None, info: Dict[str, Any] = None) -> Tuple[torch.Tensor, Any]:
        if not isinstance(obs, torch.Tensor):
            obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        orig_leading = obs.shape[:-1]
        if obs.ndim != 2:
            obs = obs.reshape(-1, obs.shape[-1])
        gate_logits = self.gate(obs)  # (batch, n_experts)

        # noisy gating during training
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(gate_logits) * self.noise_std
            gate_logits = gate_logits + noise

        # compute full softmax for load-balance loss
        gate_probs_full = torch.softmax(gate_logits, dim=-1)
        if self.training:
            self._last_lb_loss = self._load_balance_loss(gate_probs_full)
        else:
            self._last_lb_loss = None

        # top-k sparse routing
        if self.top_k >= self.n_experts:
            gate_weights = gate_probs_full
        else:
            topk_vals, topk_indices = torch.topk(gate_logits, self.top_k, dim=-1)
            gate_weights = torch.zeros_like(gate_logits)
            # scatter softmax of topk values back into zero tensor
            softmax_topk = torch.softmax(topk_vals, dim=-1)
            gate_weights.scatter_(-1, topk_indices, softmax_topk)

        # Hard regime routing: override gate in extreme regimes
        if self.hard_regime_routing and self.regime_risk_idx is not None:
            regime_risk = obs[:, self.regime_risk_idx]
            hard_weights = torch.zeros_like(gate_weights)
            for b in range(gate_weights.shape[0]):
                rr = float(regime_risk[b])
                if rr < 0.35:
                    hard_weights[b, 0] = 1.0
                elif rr > 0.65:
                    hard_weights[b, 2] = 1.0
                else:
                    # keep original top-1 expert
                    expert_idx = int(gate_weights[b].argmax())
                    hard_weights[b, expert_idx] = 1.0
            gate_weights = hard_weights

        expert_outs = torch.stack([expert(obs) for expert in self.experts], dim=1)  # (batch, n_experts, hidden)
        routed = (gate_weights.unsqueeze(-1) * expert_outs).sum(dim=1)  # (batch, hidden)
        shared = self.shared_expert(obs)
        out = self.output(torch.cat([routed, shared], dim=-1))
        if orig_leading:
            out = out.reshape(*orig_leading, -1)
        return out, state

    def get_load_balance_loss(self) -> Optional[torch.Tensor]:
        return self._last_lb_loss


class SoxlPPOPolicy(PPOPolicy):
    """PPOPolicy 子类，自动将 RegimeMoENet 的 load-balance loss 加入总损失。"""

    def __init__(self, *args, moe_lb_coef: float = 0.01, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._moe_lb_coef = moe_lb_coef

    def learn(  # type: ignore[override]
        self, batch: "Batch", batch_size: int, repeat: int, **kwargs: Any
    ) -> Dict[str, List[float]]:
        losses, clip_losses, vf_losses, ent_losses = [], [], [], []
        for step in range(repeat):
            if self._recompute_adv and step > 0:
                batch = self._compute_returns(batch, self._buffer, self._indices)
            for minibatch in batch.split(batch_size, merge_last=True):
                dist = self(minibatch).dist
                if self._norm_adv:
                    mean, std = minibatch.adv.mean(), minibatch.adv.std()
                    minibatch.adv = (minibatch.adv - mean) / (std + self._eps)
                ratio = (dist.log_prob(minibatch.act) - minibatch.logp_old).exp().float()
                ratio = ratio.reshape(ratio.size(0), -1).transpose(0, 1)
                surr1 = ratio * minibatch.adv
                surr2 = ratio.clamp(1.0 - self._eps_clip, 1.0 + self._eps_clip) * minibatch.adv
                if self._dual_clip:
                    clip1 = torch.min(surr1, surr2)
                    clip2 = torch.max(clip1, self._dual_clip * minibatch.adv)
                    clip_loss = -torch.where(minibatch.adv < 0, clip2, clip1).mean()
                else:
                    clip_loss = -torch.min(surr1, surr2).mean()
                value = self.critic(minibatch.obs).flatten()
                if self._value_clip:
                    v_clip = minibatch.v_s + (value - minibatch.v_s).clamp(-self._eps_clip, self._eps_clip)
                    vf1 = (minibatch.returns - value).pow(2)
                    vf2 = (minibatch.returns - v_clip).pow(2)
                    vf_loss = torch.max(vf1, vf2).mean()
                else:
                    vf_loss = (minibatch.returns - value).pow(2).mean()
                ent_loss = dist.entropy().mean()
                loss = clip_loss + self._weight_vf * vf_loss - self._weight_ent * ent_loss

                # add MoE load-balance auxiliary loss
                lb = None
                if hasattr(self.actor, "net"):
                    net = self.actor.net
                    if hasattr(net, "get_load_balance_loss"):
                        lb = net.get_load_balance_loss()
                if lb is not None and isinstance(lb, torch.Tensor):
                    loss = loss + self._moe_lb_coef * lb

                self.optim.zero_grad()
                loss.backward()
                if self._grad_norm:
                    nn.utils.clip_grad_norm_(
                        self._actor_critic.parameters(), max_norm=self._grad_norm
                    )
                self.optim.step()
                clip_losses.append(clip_loss.item())
                vf_losses.append(vf_loss.item())
                ent_losses.append(ent_loss.item())
                losses.append(loss.item())
        return {
            "loss": losses,
            "loss/clip": clip_losses,
            "loss/vf": vf_losses,
            "loss/ent": ent_losses,
        }


def build_policy(obs_dim: int, profile: StrategyConfig, use_moe: bool = False, hard_regime_routing: bool = False, **kwargs) -> Tuple[PPOPolicy, Actor]:
    if use_moe:
        net = RegimeMoENet(obs_dim, hidden_size=64, n_experts=3, device="cpu", top_k=1, noise_std=0.1, regime_risk_idx=STATIC_INDEX["regime_risk"], hard_regime_routing=hard_regime_routing)
    else:
        net = Net((obs_dim,), hidden_sizes=list(profile.hidden_sizes), device="cpu")
    actor = Actor(net, N_ACTIONS, device="cpu")
    critic = Critic(net, device="cpu")
    optim = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=profile.lr)
    policy_cls = SoxlPPOPolicy if use_moe else PPOPolicy
    policy_kwargs = dict(
        actor=actor,
        critic=critic,
        optim=optim,
        dist_fn=lambda logits: torch.distributions.Categorical(logits=logits),
        discount_factor=0.99,
        gae_lambda=0.95,
        max_grad_norm=0.5,
        vf_coef=0.25,
        ent_coef=0.01,
        reward_normalization=True,
        recompute_advantage=True,
        action_space=spaces.Discrete(N_ACTIONS),
    )
    if use_moe:
        policy_kwargs["moe_lb_coef"] = kwargs.get("moe_lb_coef", 0.01)
    policy = policy_cls(**policy_kwargs)
    return policy, actor
    return policy, actor


def actor_action(actor: Actor, obs: np.ndarray) -> int:
    obs = adapt_observation_for_actor(obs, actor)
    with torch.no_grad():
        out = actor(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
        logits = out[0] if isinstance(out, tuple) else out
        return int(logits.argmax(-1).item())


def infer_actor_obs_dim(actor: Actor) -> int | None:
    """Best-effort detection of the actor's expected observation width."""

    seen = set()
    stack = [actor]
    while stack:
        module = stack.pop()
        if id(module) in seen:
            continue
        seen.add(id(module))
        in_features = getattr(module, "in_features", None)
        if isinstance(in_features, int) and in_features > 0:
            return in_features
        children = list(getattr(module, "children", lambda: [])())
        stack.extend(reversed(children))
    return None


def adapt_observation(obs: np.ndarray, expected_obs_dim: int | None) -> np.ndarray:
    """Trim or pad observations so old actor bundles remain replay-compatible."""

    if expected_obs_dim is None or obs.shape[0] == expected_obs_dim:
        return obs
    if obs.shape[0] > expected_obs_dim:
        return obs[:expected_obs_dim]
    pad = np.zeros(expected_obs_dim - obs.shape[0], dtype=obs.dtype)
    return np.concatenate([obs, pad], axis=0)


def adapt_observation_for_actor(obs: np.ndarray, actor: Actor) -> np.ndarray:
    return adapt_observation(obs, infer_actor_obs_dim(actor))


def make_observation(
    df: pd.DataFrame,
    idx: int,
    pos: float,
    value: float,
    peak_value: float,
    ledger: TrancheLedger,
    price_path: np.ndarray,
    asset_idx: int = 0,
) -> np.ndarray:
    """Build the actor input vector.

    Observation layout is:

    1. engineered market/regime features from `FEATURE_COLUMNS`
    2. recent path statistics from the lookback window
    3. portfolio state such as position, underwater, and entry context
    """
    row = df.iloc[idx]
    lb = df.iloc[idx - LOOKBACK : idx]
    underwater = value / max(peak_value, 1e-8) - 1.0
    downside_lookback = lb["return"].where(lb["return"] < 0, 0.0).pow(2).mean()
    ledger_snapshot = ledger.snapshot(float(price_path[idx]))
    one_hot = np.zeros(N_ASSETS, dtype=np.float32)
    if 0 <= asset_idx < N_ASSETS:
        one_hot[asset_idx] = 1.0
    drawdown_velocity_5 = (
        float(row["cum_drop_30"] - lb["cum_drop_30"].iloc[-6]) if len(lb) >= 6 else 0.0
    )
    regime_risk_max_60d = float(lb["regime_risk"].max()) if len(lb) > 0 else 0.0
    vol_velocity_5 = (
        float(row["realized_vol_10"] - lb["realized_vol_10"].iloc[-6]) if len(lb) >= 6 else 0.0
    )
    obs = np.array(
        list(row[FEATURE_COLUMNS].values.astype(np.float32))
        + [
            float(np.nan_to_num(lb["return"].mean())),
            float(np.nan_to_num(lb["return"].skew())) if len(lb) > 3 else 0.0,
            float(np.nan_to_num(downside_lookback**0.5)),
            float(idx / len(df)),
            float(pos),
            float(underwater),
            *ledger_snapshot,
            drawdown_velocity_5,
            regime_risk_max_60d,
            vol_velocity_5,
        ]
        + list(one_hot),
        dtype=np.float32,
    )
    return np.nan_to_num(obs)


def evaluate_split(
    actor: Actor,
    df: pd.DataFrame,
    cash0: float = 100_000.0,
    position_cap_config: Optional[PositionCapConfig] = None,
    asset_idx: int = 0,
    stress_threshold: float = 0.65,
) -> Dict[str, float]:
    """Fast actor evaluation outside the heavyweight qlib backtest loop."""
    cash = cash0
    peak = cash0
    pos = 0.0
    ledger = TrancheLedger()
    price_path = build_price_path(df)
    values = [cash]
    positions = []
    stress_positions = []
    for idx in range(LOOKBACK, len(df) - 1):
        obs = make_observation(df, idx, pos, cash, peak, ledger, price_path, asset_idx=asset_idx)
        action = actor_action(actor, obs)
        target_pos = action_to_position(action, obs, config=position_cap_config)
        row = df.iloc[idx]
        regime_risk = float(np.nan_to_num(row["regime_risk"]))
        direction = "buy" if target_pos > pos else ("sell" if target_pos < pos else "neutral")
        min_rebalance = rebalance_threshold(
            regime_risk,
            float(np.nan_to_num(row["realized_vol_10"])),
            abs(float(np.nan_to_num(row["trend_gap_21"]))),
            direction=direction,
        )
        if abs(target_pos - pos) < min_rebalance:
            target_pos = pos
        next_ret = float(df.iloc[idx + 1]["return"])
        turnover = abs(target_pos - pos)
        cash *= 1.0 + target_pos * next_ret - turnover * 0.001
        peak = max(peak, cash)
        ledger.update(pos, target_pos, float(price_path[idx]))
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
