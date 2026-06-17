from __future__ import annotations

import random

import numpy as np
import pandas as pd

from ..config import StrategyConfig
from ..settings import LOOKBACK
from ..core.runtime_state import TrancheLedger, build_price_path
from .modes import mode_band
from .observation import make_observation
from ..data import feature_columns_for_packs


def usage_target_from_position(position: float, mode: str, profile: StrategyConfig) -> float:
    if profile.mode_router and getattr(profile.mode_router, "enabled", False):
        floor, cap = mode_band(str(mode), profile.mode_router)
    else:
        floor, cap = (0.0, 1.0)
    floor = float(floor)
    cap = float(cap)
    if cap <= floor:
        return 0.0
    usage = (float(position) - floor) / (cap - floor)
    return float(np.clip(usage, 0.0, 1.0))


def build_budget_usage_regression_dataset_from_frame(
    df: pd.DataFrame,
    replay_daily: pd.DataFrame,
    profile: StrategyConfig,
    sample_size: int = 2000,
    rng_seed: int = 0,
    asset_idx: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    if len(df) <= LOOKBACK + 2:
        raise ValueError("budget-usage regression dataset requires a frame longer than LOOKBACK")
    required = {"datetime", "position", "mode"}
    missing = required.difference(replay_daily.columns)
    if missing:
        raise ValueError(f"replay daily missing required columns: {sorted(missing)}")

    label_frame = replay_daily[list(required)].copy()
    label_frame["datetime"] = pd.to_datetime(label_frame["datetime"])
    label_by_date = {
        pd.Timestamp(row["datetime"]).normalize(): (float(row["position"]), str(row["mode"]))
        for _, row in label_frame.iterrows()
    }

    def row_date(idx: int) -> pd.Timestamp:
        if "datetime" in df.columns:
            return pd.Timestamp(df.iloc[idx]["datetime"]).normalize()
        return pd.Timestamp(df.index[idx]).normalize()

    feature_columns = feature_columns_for_packs(profile.feature.packs, profile.name)
    price_path = build_price_path(df)
    cash = 100_000.0
    peak = cash
    pos = 0.0
    ledger = TrancheLedger()
    obs_list: list[np.ndarray] = []
    target_list: list[float] = []

    for idx in range(LOOKBACK, len(df) - 1):
        date = row_date(idx)
        if date not in label_by_date:
            continue
        oracle_pos, oracle_mode = label_by_date[date]
        obs = make_observation(
            df,
            idx,
            pos,
            cash,
            peak,
            ledger,
            price_path,
            asset_idx=asset_idx,
            observation_window=profile.model.observation_window,
            feature_columns=feature_columns,
        )
        obs_list.append(obs)
        target_list.append(usage_target_from_position(oracle_pos, oracle_mode, profile))

        next_ret = float(df.iloc[idx + 1]["return"])
        cash *= 1.0 + oracle_pos * next_ret
        peak = max(peak, cash)
        ledger.update(pos, oracle_pos, float(price_path[idx]))
        pos = oracle_pos

    if not obs_list:
        raise RuntimeError("budget-usage regression dataset produced 0 samples")

    if sample_size > 0 and len(obs_list) > sample_size:
        rng = random.Random(rng_seed)
        indices = sorted(rng.sample(range(len(obs_list)), int(sample_size)))
        obs_list = [obs_list[i] for i in indices]
        target_list = [target_list[i] for i in indices]

    return np.stack(obs_list, axis=0), np.asarray(target_list, dtype=np.float32)


__all__ = ["build_budget_usage_regression_dataset_from_frame", "usage_target_from_position"]
