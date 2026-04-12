from __future__ import annotations

"""Daily SOXL reinforcement-learning environment and policy helpers.

The critical design choice here is that the actor does not output raw share
counts. It outputs a coarse discrete intent, which is then translated into a
position band constrained by regime-aware floors/caps and a hard drawdown cap.
That keeps risk control enforceable even when the policy drifts.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from gymnasium import spaces
from qlib.rl.interpreter import ActionInterpreter, StateInterpreter
from qlib.rl.reward import Reward
from qlib.rl.simulator import Simulator
from tianshou.policy import PPOPolicy
from tianshou.utils.net.common import Net
from tianshou.utils.net.discrete import Actor, Critic

from .data import FEATURE_COLUMNS, STATIC_INDEX
from .settings import EXTRA_OBS_DIM, LOOKBACK, N_ACTIONS, PositionCapConfig, StrategyConfig

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
}


@dataclass(frozen=True)
class EpisodeSummary:
    account: np.ndarray
    positions: List[float]


@dataclass
class PositionLedger:
    """Track entry context that is not visible in raw market features.

    These fields let the reward distinguish between:

    - a fresh add into strength
    - a stale flat position doing nothing
    - a long-held losing position that should likely be de-risked
    """

    avg_entry_price: float = 0.0
    last_add_price: float = 0.0
    high_water_price: float = 0.0
    days_in_position: int = 0

    def reset(self) -> None:
        self.avg_entry_price = 0.0
        self.last_add_price = 0.0
        self.high_water_price = 0.0
        self.days_in_position = 0

    def update(self, prev_pos: float, new_pos: float, price: float) -> None:
        price = float(price)
        if new_pos <= 1e-6:
            self.reset()
            return
        if prev_pos <= 1e-6:
            self.avg_entry_price = price
            self.last_add_price = price
            self.high_water_price = price
            self.days_in_position = 1
            return

        self.days_in_position += 1
        self.high_water_price = max(self.high_water_price, price)
        if new_pos > prev_pos + 1e-6:
            add_size = max(new_pos - prev_pos, 0.0)
            base_size = max(prev_pos, 1e-6)
            self.avg_entry_price = (self.avg_entry_price * base_size + price * add_size) / max(base_size + add_size, 1e-6)
            self.last_add_price = price

    def snapshot(self, price: float) -> Tuple[float, float, float, float]:
        if self.days_in_position <= 0 or price <= 0:
            return 0.0, 0.0, 0.0, 0.0
        avg_pnl = price / max(self.avg_entry_price, 1e-6) - 1.0 if self.avg_entry_price > 0 else 0.0
        add_pnl = price / max(self.last_add_price, 1e-6) - 1.0 if self.last_add_price > 0 else 0.0
        high_gap = price / max(self.high_water_price, 1e-6) - 1.0 if self.high_water_price > 0 else 0.0
        age = min(self.days_in_position / 63.0, 1.0)
        return float(age), float(avg_pnl), float(add_pnl), float(high_gap)


class EpisodeConfig(NamedTuple):
    pkl_path: str
    start_idx: int
    end_idx: int
    lookback: int = LOOKBACK
    cash: float = 100_000.0


def rebalance_threshold(regime_risk: float, realized_vol: float, trend_strength: float) -> float:
    """Suppress tiny churn when the market state does not justify a rebalance."""
    threshold = 0.015
    if regime_risk < 0.35 and realized_vol < 0.45 and trend_strength < 0.04:
        threshold = 0.03
    if regime_risk > 0.60 or realized_vol > 0.70 or trend_strength > 0.10:
        threshold = 0.008
    return threshold


def action_band_from_obs(obs: np.ndarray) -> Tuple[float, float]:
    """Map market regime features into a tradable position band.

    The actor chooses where inside the band to sit. The band itself encodes the
    hard prior that SOXL should keep some core exposure in healthy uptrends and
    should compress its ceiling when risk or volatility becomes abnormal.
    """
    regime_risk = float(np.nan_to_num(obs[STATIC_INDEX["regime_risk"]]))
    anchor_gap = float(np.nan_to_num(obs[STATIC_INDEX["anchor_gap_180"]]))
    anchor_slope = float(np.nan_to_num(obs[STATIC_INDEX["anchor_slope_180"]]))
    cum_drop_30 = float(np.nan_to_num(obs[STATIC_INDEX["cum_drop_30"]]))
    vix_zscore = float(np.nan_to_num(obs[STATIC_INDEX["vix_zscore"]]))
    vol_coef = float(np.nan_to_num(obs[STATIC_INDEX["vol_coef_21"]]))

    floor = 0.15
    cap = 0.95

    if anchor_gap > 0.02 and anchor_slope >= 0:
        floor = 0.30
        cap = 1.00
    if regime_risk > 0.72 or (anchor_gap < -0.06 and anchor_slope < 0):
        floor = 0.05
        cap = 0.80
    if cum_drop_30 < -0.18 and vix_zscore < 2.8:
        floor = max(floor, 0.18)
        cap = 1.00
    if vol_coef > 1.45:
        floor = max(0.08, floor - 0.05)
        cap = min(cap, 0.90)

    return floor, cap


def drawdown_position_cap(cumulative_drop: float, config: Optional[PositionCapConfig] = None) -> float:
    """Hard cap that overrides the actor during persistent drawdown regimes."""
    cfg = config or DEFAULT_POSITION_CAP
    drop = max(float(cumulative_drop), 0.0)
    if drop <= cfg.start_drawdown:
        return 1.0
    if drop >= cfg.full_drawdown:
        return float(cfg.min_cap)
    span = max(cfg.full_drawdown - cfg.start_drawdown, 1e-6)
    scale = (drop - cfg.start_drawdown) / span
    curve_power = max(float(getattr(cfg, "curve_power", 1.0)), 1e-6)
    shaped_scale = scale**curve_power
    return float(1.0 - shaped_scale * (1.0 - cfg.min_cap))


def action_to_position(action: int, obs: np.ndarray, config: Optional[PositionCapConfig] = None) -> float:
    base = float(BASE_ACTION_LEVELS[int(np.clip(action, 0, N_ACTIONS - 1))])
    floor, cap = action_band_from_obs(obs)
    cumulative_drop = max(-float(np.nan_to_num(obs[STATIC_INDEX["cum_drop_30"]])), 0.0)
    cap = min(cap, drawdown_position_cap(cumulative_drop, config=config))
    return float(np.clip(floor + base * (cap - floor), 0.0, 1.0))


def build_price_path(df: pd.DataFrame, base_price: float = 100.0) -> np.ndarray:
    returns = df["return"].fillna(0.0).to_numpy(dtype=np.float32)
    return np.cumprod(1.0 + returns, dtype=np.float32) * np.float32(base_price)


class SoxlSimulator(Simulator):
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
        self.nf = len(FEATURE_COLUMNS)
        self.ledger = PositionLedger()

    def step(self, action: float) -> None:
        action = float(np.clip(action, 0, 1))
        if self.idx >= len(self.df) - 1:
            return
        row = self.df.iloc[self.idx]
        price = float(self.price_path[self.idx])
        min_rebalance = rebalance_threshold(
            float(np.nan_to_num(row["regime_risk"])),
            float(np.nan_to_num(row["realized_vol_10"])),
            abs(float(np.nan_to_num(row["trend_gap_21"]))),
        )
        if abs(action - self.pos) < min_rebalance:
            action = self.pos
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
        self.pos = action
        self.idx += 1

    def get_state(self):
        if self.idx >= len(self.df):
            obs = np.zeros(self.nf + EXTRA_OBS_DIM, dtype=np.float32)
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
        obs_list = list(row[FEATURE_COLUMNS].values.astype(np.float32))
        obs_list.extend(
            [
                float(np.nan_to_num(lb["return"].mean())),
                float(np.nan_to_num(lb["return"].skew())) if len(lb) > 3 else 0.0,
                float(np.nan_to_num(downside_lookback**0.5)),
                float(self.idx / len(self.df)),
                float(self.pos),
                float(underwater),
                *ledger_snapshot,
            ]
        )
        return {
            "obs": np.nan_to_num(np.array(obs_list, dtype=np.float32)),
            "pos": self.pos,
            "val": self.val,
            "pval": self.prev_val,
            "turnover": self.last_turnover,
            "drawdown": abs(underwater),
            "step_return": self.last_step_return,
            "regime_risk": regime_risk,
            "regime_delta": regime_delta,
            "flat_steps": float(self.flat_steps),
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
    def __init__(self, profile: StrategyConfig):
        super().__init__()
        self.profile = profile

    def reward(self, state):
        """Reward trade quality, not just raw return.

        The reward is intentionally asymmetric: it tolerates controlled pullback
        in healthy trend regimes, but it punishes stale inactivity and long-held
        losing exposure much more aggressively.
        """
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

        reward = pnl_ret * 100.0 * cfg.return_scale
        reward -= turnover * cfg.transaction_cost * 100.0
        reward -= turnover * cfg.turnover_penalty
        reward -= excess_drawdown * cfg.drawdown_penalty
        reward -= downside_signal * cfg.downside_penalty
        reward -= max(pos - (1.0 - 0.65 * regime_risk), 0.0) * cfg.risk_budget_penalty
        reward -= max(pos - 0.85, 0.0) * cfg.high_exposure_penalty
        reward -= (turnover**2) * cfg.smoothness_penalty
        reward += min(turnover, 0.30) * min(activity_signal, 2.2) * cfg.activity_bonus
        reward += min(drawdown, tolerated_drawdown) * min(pos, 0.80) * min(trend_support * 2.5, 0.18)
        reward -= stale_position * (6.0 + 4.0 * activity_signal)
        reward -= trapped_loser * (18.0 + 6.0 * regime_risk)
        if anchor_gap > 0.02 and regime_risk < 0.45 and pos < 0.30:
            reward -= 0.12
        if cum_drop_30 < -0.18 and regime_risk < 0.80 and pos < 0.20:
            reward -= 0.10
        if flat_steps >= 4:
            flat_drag = min((flat_steps - 3.0) / 6.0, 1.6)
            reward -= flat_drag * (0.20 + activity_signal * 0.9) * cfg.inactivity_penalty
        if flat_steps >= 8 and activity_signal > 0.12:
            reward -= min(flat_steps / 8.0, 1.5) * cfg.inactivity_penalty
        if holding_age >= 0.35 and turnover < 0.02 and abs(avg_entry_pnl) < 0.02:
            reward -= (0.16 + holding_age * 0.14) * cfg.inactivity_penalty
        return float(reward)


def build_initial_states(pkl_path: Path, df: pd.DataFrame, profile: StrategyConfig, smoke: bool) -> List[EpisodeConfig]:
    """Create episode start/end pairs with mild stress oversampling.

    Oversampling is based only on state visible inside the training window. That
    preserves the repo's non-leak rule while still exposing the policy to more
    stressful paths than naive uniform sampling would.
    """
    episode_len = profile.episode_len
    frame_count = len(df) // episode_len
    if smoke:
        frame_count = min(frame_count, 4)
    states: List[EpisodeConfig] = []
    for i in range(frame_count):
        start_idx = i * episode_len
        end_idx = min((i + 1) * episode_len, len(df))
        if (end_idx - start_idx) <= LOOKBACK + 2:
            continue
        state = EpisodeConfig(str(pkl_path), start_idx, end_idx)
        states.append(state)
        segment = df.iloc[start_idx:end_idx]
        risk_score = float(segment["regime_risk"].mean())
        drawdown_score = abs(float(segment["cum_drop_30"].min()))
        repeats = 1
        if risk_score > 0.58 or drawdown_score > 0.18:
            repeats += 1
        if risk_score > 0.42 and drawdown_score > 0.10:
            repeats += 1
        if not smoke:
            states.extend([state] * max(repeats - 1, 0))
    return states


def build_policy(obs_dim: int, profile: StrategyConfig) -> Tuple[PPOPolicy, Actor]:
    net = Net((obs_dim,), hidden_sizes=list(profile.hidden_sizes), device="cpu")
    actor = Actor(net, N_ACTIONS, device="cpu")
    critic = Critic(net, device="cpu")
    optim = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=profile.lr)
    policy = PPOPolicy(
        actor=actor,
        critic=critic,
        optim=optim,
        dist_fn=lambda logits: torch.distributions.Categorical(logits=logits),
        discount_factor=0.99,
        gae_lambda=0.95,
        max_grad_norm=0.5,
        vf_coef=0.25,
        ent_coef=0.0,
        action_space=spaces.Discrete(N_ACTIONS),
    )
    return policy, actor


def actor_action(actor: Actor, obs: np.ndarray) -> int:
    with torch.no_grad():
        out = actor(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
        logits = out[0] if isinstance(out, tuple) else out
        return int(logits.argmax(-1).item())


def make_observation(
    df: pd.DataFrame,
    idx: int,
    pos: float,
    value: float,
    peak_value: float,
    ledger: PositionLedger,
    price_path: np.ndarray,
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
        ],
        dtype=np.float32,
    )
    return np.nan_to_num(obs)


def evaluate_split(
    actor: Actor,
    df: pd.DataFrame,
    cash0: float = 100_000.0,
    position_cap_config: Optional[PositionCapConfig] = None,
) -> Dict[str, float]:
    """Fast actor evaluation outside the heavyweight qlib backtest loop."""
    cash = cash0
    peak = cash0
    pos = 0.0
    ledger = PositionLedger()
    price_path = build_price_path(df)
    values = [cash]
    positions = []
    for idx in range(LOOKBACK, len(df) - 1):
        obs = make_observation(df, idx, pos, cash, peak, ledger, price_path)
        action = actor_action(actor, obs)
        target_pos = action_to_position(action, obs, config=position_cap_config)
        row = df.iloc[idx]
        min_rebalance = rebalance_threshold(
            float(np.nan_to_num(row["regime_risk"])),
            float(np.nan_to_num(row["realized_vol_10"])),
            abs(float(np.nan_to_num(row["trend_gap_21"]))),
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
        values.append(cash)
    account = np.array(values)
    daily_ret = np.diff(account) / account[:-1]
    peak_curve = np.maximum.accumulate(account)
    return {
        "return": (account[-1] / account[0] - 1) * 100,
        "sharpe": float(np.mean(daily_ret) / (np.std(daily_ret) + 1e-8) * np.sqrt(252)),
        "max_dd": float(np.max((peak_curve - account) / peak_curve) * 100),
        "avg_pos": float(np.mean(positions)) if positions else 0.0,
    }
