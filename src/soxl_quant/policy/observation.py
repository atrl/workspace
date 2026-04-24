from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from tianshou.utils.net.discrete import Actor

from ..config import StrategyConfig
from ..data import FEATURE_COLUMNS, feature_columns_for_packs
from ..settings import EXTRA_OBS_DIM, LOOKBACK, N_ASSETS

if TYPE_CHECKING:
    from .environment import TrancheLedger


def extra_index(feature_columns: list[str] | tuple[str, ...] | None = None) -> dict[str, int]:
    cols = list(feature_columns or FEATURE_COLUMNS)
    return {
        "lookback_mean_return": len(cols) + 0,
        "lookback_return_skew": len(cols) + 1,
        "lookback_downside": len(cols) + 2,
        "episode_progress": len(cols) + 3,
        "position": len(cols) + 4,
        "underwater": len(cols) + 5,
        "holding_age": len(cols) + 6,
        "avg_entry_pnl": len(cols) + 7,
        "last_add_pnl": len(cols) + 8,
        "high_gap": len(cols) + 9,
        "last_tranche_pnl": len(cols) + 10,
        "n_tranches": len(cols) + 11,
        "drawdown_velocity_5": len(cols) + 12,
        "regime_risk_max_60d": len(cols) + 13,
        "vol_velocity_5": len(cols) + 14,
    }


EXTRA_INDEX = extra_index(FEATURE_COLUMNS)


def observation_dim_for_profile(profile: StrategyConfig) -> int:
    window = max(int(profile.model.observation_window), 1)
    feature_columns = feature_columns_for_packs(profile.feature.packs, profile.name)
    base = len(feature_columns) * window + EXTRA_OBS_DIM + N_ASSETS
    if getattr(profile.model, "use_mode_conditioned_input", False):
        from .mode_conditioning import MODE_ONE_HOT_ORDER
        base += len(MODE_ONE_HOT_ORDER)
    return base


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
    feature_columns: list[str] | tuple[str, ...] | None = None,
) -> np.ndarray:
    selected_columns = list(feature_columns or FEATURE_COLUMNS)
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
        history = df.iloc[idx - window + 1 : idx][selected_columns]
        if len(history) < window - 1:
            pad_rows = np.zeros((window - 1 - len(history), len(selected_columns)), dtype=np.float32)
            history_values = np.vstack([pad_rows, history.to_numpy(dtype=np.float32)])
        else:
            history_values = history.to_numpy(dtype=np.float32)
        history_features = history_values.reshape(-1).tolist()
    drawdown_velocity_5 = float(row["cum_drop_30"] - lb["cum_drop_30"].iloc[-6]) if len(lb) >= 6 else 0.0
    regime_risk_max_60d = float(lb["regime_risk"].max()) if len(lb) > 0 else 0.0
    vol_velocity_5 = float(row["realized_vol_10"] - lb["realized_vol_10"].iloc[-6]) if len(lb) >= 6 else 0.0
    obs = np.array(
        list(row[selected_columns].values.astype(np.float32))
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
