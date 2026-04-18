from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from tianshou.utils.net.discrete import Actor

from ..config import StrategyConfig
from ..data import FEATURE_COLUMNS
from ..settings import EXTRA_OBS_DIM, LOOKBACK, N_ASSETS

if TYPE_CHECKING:
    from .environment import TrancheLedger


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


def observation_dim_for_profile(profile: StrategyConfig) -> int:
    window = max(int(profile.model.observation_window), 1)
    return len(FEATURE_COLUMNS) * window + EXTRA_OBS_DIM + N_ASSETS


def infer_actor_obs_dim(actor: Actor) -> int | None:
    seen = set()
    stack = [actor]
    while stack:
        module = stack.pop()
        if id(module) in seen:
            continue
        seen.add(id(module))
        input_dim = getattr(module, "input_dim", None)
        if isinstance(input_dim, int) and input_dim > 0:
            return input_dim
        in_features = getattr(module, "in_features", None)
        if isinstance(in_features, int) and in_features > 0:
            return in_features
        children = list(getattr(module, "children", lambda: [])())
        stack.extend(reversed(children))
    return None


def adapt_observation(obs: np.ndarray, expected_obs_dim: int | None) -> np.ndarray:
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
    observation_window: int = 1,
) -> np.ndarray:
    row = df.iloc[idx]
    lb = df.iloc[idx - LOOKBACK : idx]
    underwater = value / max(peak_value, 1e-8) - 1.0
    downside_lookback = lb["return"].where(lb["return"] < 0, 0.0).pow(2).mean()
    ledger_snapshot = ledger.snapshot(float(price_path[idx]))
    one_hot = np.zeros(N_ASSETS, dtype=np.float32)
    if 0 <= asset_idx < N_ASSETS:
        one_hot[asset_idx] = 1.0
    window = max(int(observation_window), 1)
    history_features: list[float] = []
    if window > 1:
        history = df.iloc[idx - window + 1 : idx][FEATURE_COLUMNS]
        if len(history) < window - 1:
            pad_rows = np.zeros((window - 1 - len(history), len(FEATURE_COLUMNS)), dtype=np.float32)
            history_values = np.vstack([pad_rows, history.to_numpy(dtype=np.float32)])
        else:
            history_values = history.to_numpy(dtype=np.float32)
        history_features = history_values.reshape(-1).tolist()
    drawdown_velocity_5 = float(row["cum_drop_30"] - lb["cum_drop_30"].iloc[-6]) if len(lb) >= 6 else 0.0
    regime_risk_max_60d = float(lb["regime_risk"].max()) if len(lb) > 0 else 0.0
    vol_velocity_5 = float(row["realized_vol_10"] - lb["realized_vol_10"].iloc[-6]) if len(lb) >= 6 else 0.0
    obs = np.array(
        list(row[FEATURE_COLUMNS].values.astype(np.float32))
        + history_features
        + [
            float(np.nan_to_num(lb["return"].mean())),
            float(np.nan_to_num(lb["return"].skew())) if len(lb) > 3 else 0.0,
            float(np.nan_to_num(downside_lookback ** 0.5)),
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


__all__ = [
    "EXTRA_INDEX",
    "adapt_observation",
    "adapt_observation_for_actor",
    "infer_actor_obs_dim",
    "make_observation",
    "observation_dim_for_profile",
]
