from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
import pandas as pd


STARTING_NAV = 1_000_000.0


@dataclass(frozen=True)
class StateConditionedRouterConfig:
    upside_cap: float = 0.45
    rebound_mom21: float = 0.40
    rebound_volq: float = 0.40
    risk_cap: float = 0.15
    risk_dd63: float = -0.20
    risk_mom21: float = 0.0
    use_semis_stress: bool = True
    require_semis_widen: bool = False
    rebound_mom5: float = 0.02
    rebound_min_dd63: float = -0.35
    semis_ratio_z: float = -0.75
    semis_ratio_widen: float = -0.05
    semis_soxx_mom21: float = 0.0


def _coalesce_feature(df: pd.DataFrame, names: Iterable[str], default: float = 0.0) -> pd.Series:
    for name in names:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce")
    return pd.Series(default, index=df.index, dtype=float)


def _safe_quantile(values: np.ndarray, q: float) -> float:
    clean = values[np.isfinite(values)]
    if len(clean) == 0:
        return float("nan")
    return float(np.quantile(clean, float(q)))


def _recompute_nav(price: np.ndarray, position: np.ndarray, *, starting_nav: float = STARTING_NAV) -> np.ndarray:
    returns = np.zeros_like(price, dtype=float)
    returns[1:] = price[1:] / price[:-1] - 1.0
    applied_position = np.r_[position[0], position[:-1]]
    return float(starting_nav) * np.cumprod(1.0 + applied_position * returns)


def _max_drawdown(nav: pd.Series) -> float:
    values = pd.to_numeric(nav, errors="coerce").dropna().astype(float)
    if len(values) < 2:
        return 0.0
    peak = values.cummax()
    return abs(float((values / peak - 1.0).min()) * 100.0)


def state_conditioned_router_metrics(frame: pd.DataFrame) -> dict[str, float]:
    required = {"strategy_nav", "buyhold_nav", "router_position", "base_position"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"router frame missing columns: {missing}")
    strategy = pd.to_numeric(frame["strategy_nav"], errors="coerce")
    buyhold = pd.to_numeric(frame["buyhold_nav"], errors="coerce")
    position = pd.to_numeric(frame["router_position"], errors="coerce")
    base = pd.to_numeric(frame["base_position"], errors="coerce")
    if len(frame) < 2:
        strategy_return = buyhold_return = 0.0
    else:
        strategy_return = float((strategy.iloc[-1] / strategy.iloc[0] - 1.0) * 100.0)
        buyhold_return = float((buyhold.iloc[-1] / buyhold.iloc[0] - 1.0) * 100.0)
    return {
        "rows": int(len(frame)),
        "return": strategy_return,
        "buy_and_hold_return": buyhold_return,
        "relative_return": float(strategy_return - buyhold_return),
        "max_dd": _max_drawdown(strategy),
        "avg_pos": float(position.mean()) if len(position) else 0.0,
        "turnover": float(position.diff().abs().fillna(0.0).sum()),
        "rebound_days": int(frame.get("rebound_state", pd.Series(False, index=frame.index)).sum()),
        "risk_days": int(frame.get("risk_state", pd.Series(False, index=frame.index)).sum()),
        "changed_days": int((position.sub(base).abs() > 1e-9).sum()),
    }


def _prepare_router_inputs(base_daily: pd.DataFrame, upside_daily: pd.DataFrame) -> dict[str, object]:
    base = base_daily.copy()
    upside = upside_daily.copy()
    for frame in (base, upside):
        if "datetime" not in frame.columns or "position" not in frame.columns:
            raise ValueError("daily paths require datetime and position columns")
        frame["datetime"] = pd.to_datetime(frame["datetime"])
    merged = base.sort_values("datetime").merge(
        upside.sort_values("datetime")[["datetime", "position"]].rename(
            columns={"position": "upside_position"}
        ),
        on="datetime",
        how="inner",
        validate="one_to_one",
    )
    merged = merged.rename(columns={"position": "base_position"})
    merged = merged.loc[:, ~merged.columns.duplicated()].reset_index(drop=True)
    if merged.empty:
        raise ValueError("no overlapping dates for base/upside daily paths")
    if "price" not in merged.columns:
        raise ValueError("base daily path requires price column")

    price_s = pd.to_numeric(merged["price"], errors="coerce")
    returns_s = price_s.pct_change()
    computed_mom5 = price_s / price_s.shift(5) - 1.0
    computed_mom21 = price_s / price_s.shift(21) - 1.0
    computed_dd63 = price_s / price_s.rolling(63, min_periods=2).max() - 1.0
    computed_rv21 = returns_s.rolling(21, min_periods=5).std() * np.sqrt(252.0)
    computed_vov21 = returns_s.rolling(21, min_periods=5).std()
    return {
        "merged": merged,
        "price": price_s.to_numpy(dtype=float),
        "base_position": pd.to_numeric(merged["base_position"], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=float),
        "upside_position": pd.to_numeric(merged["upside_position"], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=float),
        "mom5": _coalesce_feature(merged, ["mom_5", "price_mom_5", "lag_mom_5"], default=np.nan)
        .fillna(computed_mom5)
        .fillna(0.0)
        .to_numpy(dtype=float),
        "mom21": _coalesce_feature(merged, ["mom_21", "price_mom_21", "lag_mom_21"], default=np.nan)
        .fillna(computed_mom21)
        .fillna(0.0)
        .to_numpy(dtype=float),
        "dd63": _coalesce_feature(
            merged,
            ["drawdown_63", "price_drawdown_63", "lag_drawdown_63"],
            default=np.nan,
        )
        .fillna(computed_dd63)
        .fillna(0.0)
        .to_numpy(dtype=float),
        "rv21": _coalesce_feature(merged, ["rv_21", "realized_vol_21", "lag_rv_21"], default=np.nan)
        .fillna(computed_rv21)
        .fillna(0.0)
        .to_numpy(dtype=float),
        "vol_of_vol_21": _coalesce_feature(
            merged,
            ["vol_of_vol_21", "lag_vol_of_vol_21"],
            default=np.nan,
        )
        .fillna(computed_vov21)
        .fillna(0.0)
        .to_numpy(dtype=float),
        "ratio_widen_21": _coalesce_feature(
            merged,
            ["soxl_soxx_ratio_widen_21", "lag_soxl_soxx_ratio_widen_21"],
            default=np.nan,
        )
        .fillna(0.0)
        .to_numpy(dtype=float),
        "ratio_z_252": _coalesce_feature(
            merged,
            ["soxl_soxx_ratio_z_252", "lag_soxl_soxx_ratio_z_252"],
            default=np.nan,
        )
        .fillna(0.0)
        .to_numpy(dtype=float),
        "soxx_mom21": _coalesce_feature(merged, ["soxx_mom_21", "lag_soxx_mom_21"], default=np.nan)
        .fillna(0.0)
        .to_numpy(dtype=float),
    }


def _route_position_arrays(
    inputs: dict[str, object],
    config: StateConditionedRouterConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    base_position = inputs["base_position"]
    upside_position = inputs["upside_position"]
    rv21 = inputs["rv21"]
    vol_of_vol_21 = inputs["vol_of_vol_21"]
    assert isinstance(base_position, np.ndarray)
    assert isinstance(upside_position, np.ndarray)
    assert isinstance(rv21, np.ndarray)
    assert isinstance(vol_of_vol_21, np.ndarray)
    volatility_signal = np.where(np.isfinite(rv21) & (rv21 > 0), rv21, vol_of_vol_21)
    vol_threshold = _safe_quantile(volatility_signal, config.rebound_volq)

    ratio_widen_21 = inputs["ratio_widen_21"]
    ratio_z_252 = inputs["ratio_z_252"]
    mom21 = inputs["mom21"]
    mom5 = inputs["mom5"]
    dd63 = inputs["dd63"]
    soxx_mom21 = inputs["soxx_mom21"]
    assert isinstance(ratio_widen_21, np.ndarray)
    assert isinstance(ratio_z_252, np.ndarray)
    assert isinstance(mom21, np.ndarray)
    assert isinstance(mom5, np.ndarray)
    assert isinstance(dd63, np.ndarray)
    assert isinstance(soxx_mom21, np.ndarray)

    semis_widen_ok = np.ones_like(base_position, dtype=bool)
    if config.require_semis_widen:
        semis_widen_ok = (ratio_widen_21 >= 0.0) | (ratio_z_252 >= -0.25)
    rebound = (
        (mom21 >= float(config.rebound_mom21))
        & (mom5 >= float(config.rebound_mom5))
        & (dd63 >= float(config.rebound_min_dd63))
        & (volatility_signal >= vol_threshold)
        & semis_widen_ok
    )
    broken_anchor_risk = (dd63 <= float(config.risk_dd63)) & (mom21 <= float(config.risk_mom21))
    semis_stress = (
        ((ratio_z_252 <= float(config.semis_ratio_z)) | (ratio_widen_21 <= float(config.semis_ratio_widen)))
        & (soxx_mom21 <= float(config.semis_soxx_mom21))
    )
    risk = broken_anchor_risk | (semis_stress if config.use_semis_stress else False)
    position = base_position.copy()
    position[risk] = np.minimum(position[risk], float(config.risk_cap))
    lift = np.maximum(position, np.minimum(upside_position, float(config.upside_cap)))
    position[rebound & ~risk] = lift[rebound & ~risk]
    return position, rebound, risk, vol_threshold


def _router_frame_from_inputs(
    inputs: dict[str, object],
    config: StateConditionedRouterConfig,
    *,
    starting_nav: float,
) -> pd.DataFrame:
    merged = inputs["merged"]
    price = inputs["price"]
    assert isinstance(merged, pd.DataFrame)
    assert isinstance(price, np.ndarray)
    position, rebound, risk, vol_threshold = _route_position_arrays(inputs, config)
    result = merged.copy()
    result["buyhold_nav"] = float(starting_nav) * price / price[0]
    result["strategy_nav"] = _recompute_nav(price, position, starting_nav=starting_nav)
    result["router_position"] = position
    result["rebound_state"] = rebound
    result["risk_state"] = risk
    result["mom_5"] = inputs["mom5"]
    result["mom_21"] = inputs["mom21"]
    result["drawdown_63"] = inputs["dd63"]
    result["rv_21"] = inputs["rv21"]
    result["vol_of_vol_21"] = inputs["vol_of_vol_21"]
    result["soxl_soxx_ratio_widen_21"] = inputs["ratio_widen_21"]
    result["soxl_soxx_ratio_z_252"] = inputs["ratio_z_252"]
    result["soxx_mom_21"] = inputs["soxx_mom21"]
    result["vol_threshold"] = vol_threshold
    return result


def _router_metrics_from_arrays(
    *,
    price: np.ndarray,
    position: np.ndarray,
    base_position: np.ndarray,
    rebound: np.ndarray,
    risk: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    sub_price = price[mask]
    sub_position = position[mask]
    sub_base = base_position[mask]
    sub_rebound = rebound[mask]
    sub_risk = risk[mask]
    if len(sub_price) == 0:
        return {
            "rows": 0,
            "return": 0.0,
            "buy_and_hold_return": 0.0,
            "relative_return": 0.0,
            "max_dd": 0.0,
            "avg_pos": 0.0,
            "turnover": 0.0,
            "rebound_days": 0,
            "risk_days": 0,
            "changed_days": 0,
        }
    nav = _recompute_nav(sub_price, sub_position)
    buyhold = STARTING_NAV * sub_price / sub_price[0]
    if len(sub_price) < 2:
        strategy_return = buyhold_return = 0.0
    else:
        strategy_return = float((nav[-1] / nav[0] - 1.0) * 100.0)
        buyhold_return = float((buyhold[-1] / buyhold[0] - 1.0) * 100.0)
    max_dd = 0.0
    if len(nav) >= 2:
        max_dd = abs(float((nav / np.maximum.accumulate(nav) - 1.0).min()) * 100.0)
    return {
        "rows": int(len(sub_price)),
        "return": strategy_return,
        "buy_and_hold_return": buyhold_return,
        "relative_return": float(strategy_return - buyhold_return),
        "max_dd": max_dd,
        "avg_pos": float(np.nanmean(sub_position)),
        "turnover": float(np.abs(np.diff(sub_position, prepend=sub_position[0])).sum()),
        "rebound_days": int(sub_rebound.sum()),
        "risk_days": int(sub_risk.sum()),
        "changed_days": int((np.abs(sub_position - sub_base) > 1e-9).sum()),
    }


def build_state_conditioned_router_frame(
    base_daily: pd.DataFrame,
    upside_daily: pd.DataFrame,
    config: StateConditionedRouterConfig,
    *,
    starting_nav: float = STARTING_NAV,
) -> pd.DataFrame:
    """Merge base/upside daily paths and apply a state-conditioned router."""
    return _router_frame_from_inputs(
        _prepare_router_inputs(base_daily, upside_daily),
        config,
        starting_nav=starting_nav,
    )


def score_state_conditioned_router_metrics(
    metrics: dict[str, float],
    *,
    target_relative_return: float = 30.0,
    target_max_dd: float = 15.0,
) -> dict[str, float | bool]:
    relative_gap = max(0.0, float(target_relative_return) - float(metrics["relative_return"]))
    drawdown_gap = max(0.0, float(metrics["max_dd"]) - float(target_max_dd))
    robust_score = relative_gap**2 + 4.0 * drawdown_gap**2 + 0.02 * float(metrics.get("turnover", 0.0))
    return {
        "robust_score": float(robust_score),
        "relative_gap": float(relative_gap),
        "drawdown_gap": float(drawdown_gap),
        "target_gate": bool(relative_gap == 0.0 and drawdown_gap == 0.0),
    }


def screen_state_conditioned_router_configs(
    base_daily: pd.DataFrame,
    upside_daily: pd.DataFrame,
    configs: Iterable[StateConditionedRouterConfig],
    *,
    fold: str,
    train_start: str | pd.Timestamp,
    train_end: str | pd.Timestamp,
) -> pd.DataFrame:
    """Score router configs using only a fold's train-window rows."""

    train_start_ts = pd.Timestamp(train_start).normalize()
    train_end_ts = pd.Timestamp(train_end).normalize()
    if train_end_ts < train_start_ts:
        raise ValueError("train_end is before train_start")

    rows: list[dict[str, object]] = []
    inputs = _prepare_router_inputs(base_daily, upside_daily)
    merged = inputs["merged"]
    price = inputs["price"]
    base_position = inputs["base_position"]
    assert isinstance(merged, pd.DataFrame)
    assert isinstance(price, np.ndarray)
    assert isinstance(base_position, np.ndarray)
    dates = pd.to_datetime(merged["datetime"]).dt.normalize()
    train_mask = ((dates >= train_start_ts) & (dates <= train_end_ts)).to_numpy(dtype=bool)
    for candidate_id, config in enumerate(configs):
        position, rebound, risk, _ = _route_position_arrays(inputs, config)
        metrics = _router_metrics_from_arrays(
            price=price,
            position=position,
            base_position=base_position,
            rebound=rebound,
            risk=risk,
            mask=train_mask,
        )
        if int(metrics["rows"]) == 0:
            continue
        score = score_state_conditioned_router_metrics(metrics)
        row = {
            "fold": str(fold),
            "candidate_id": int(candidate_id),
            **asdict(config),
            "full_rows": int(len(merged)),
            **{f"train_{key}": value for key, value in metrics.items()},
            **score,
        }
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["target_gate", "robust_score", "candidate_id"],
        ascending=[False, True, True],
    ).reset_index(drop=True)


__all__ = [
    "STARTING_NAV",
    "StateConditionedRouterConfig",
    "build_state_conditioned_router_frame",
    "score_state_conditioned_router_metrics",
    "screen_state_conditioned_router_configs",
    "state_conditioned_router_metrics",
]
