from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from ..policy.two_speed_budget import (
    TwoSpeedBudgetConfig,
    compose_budget_usage_policy_frame,
    compose_budget_usage_position,
    infer_mode_from_slow_position,
    maybe_rebalance_position,
)


DEFAULT_WINDOWS: dict[str, tuple[str, str]] = {
    "test_2020": ("2020-01-01", "2020-12-31"),
    "test_2022": ("2022-01-01", "2022-12-31"),
    "test_2023_2024": ("2023-01-01", "2024-12-31"),
}


@dataclass(frozen=True)
class BudgetUsageCounterfactual:
    payload: dict
    daily: pd.DataFrame


@dataclass(frozen=True)
class BullParticipationGuardConfig:
    enabled: bool = False
    ma_window: int = 50
    momentum_window: int = 20
    high_window: int = 50
    min_periods: int = 20
    min_momentum: float = 0.03
    max_drawdown: float = 0.08


def _position_frame(frame: pd.DataFrame, position_col: str = "position") -> pd.DataFrame:
    if "datetime" not in frame.columns:
        raise ValueError("policy frame missing `datetime` column")
    if position_col not in frame.columns:
        raise ValueError(f"policy frame missing `{position_col}` column")
    out = frame[["datetime", position_col]].rename(columns={position_col: "position"}).copy()
    out["datetime"] = pd.to_datetime(out["datetime"])
    out["position"] = out["position"].astype(float).clip(0.0, 1.0)
    return out.sort_values("datetime").reset_index(drop=True)


def align_price(frame: pd.DataFrame, price_series: pd.Series) -> pd.DataFrame:
    out = frame.copy()
    out["datetime"] = pd.to_datetime(out["datetime"])
    if "price" in out.columns:
        out = out.drop(columns=["price"])

    price = price_series.copy()
    if not isinstance(price.index, pd.DatetimeIndex):
        price.index = pd.to_datetime(price.index)
    price_df = price.rename("price").reset_index()
    price_df.columns = ["datetime", "price"]
    price_df["datetime"] = pd.to_datetime(price_df["datetime"])

    out = out.merge(price_df, on="datetime", how="inner")
    out = out.dropna(subset=["price"])
    if len(out) < 2:
        raise ValueError("not enough overlapping non-null prices to score policy frame")
    return out.sort_values("datetime").reset_index(drop=True)


def compute_metrics(frame: pd.DataFrame, price_series: pd.Series) -> dict:
    scored = align_price(frame, price_series)
    positions = scored["position"].to_numpy(dtype=float)
    prices = scored["price"].to_numpy(dtype=float)
    returns = np.diff(prices) / prices[:-1]
    strat_returns = positions[:-1] * returns

    nav = np.cumprod(1.0 + strat_returns)
    total_return = float(nav[-1] - 1.0) * 100.0
    sharpe = (
        float(np.mean(strat_returns) / np.std(strat_returns) * np.sqrt(252))
        if len(strat_returns) > 1 and float(np.std(strat_returns)) > 0.0
        else 0.0
    )
    peak = np.maximum.accumulate(nav)
    drawdowns = (nav - peak) / peak
    max_dd = abs(float(np.min(drawdowns)) * 100.0)
    stress_mask = drawdowns < -0.10
    stress_avg_pos = float(np.mean(positions[1:][stress_mask])) if np.any(stress_mask) else 0.0

    buy_hold_nav = np.cumprod(1.0 + returns)
    changes = np.abs(np.diff(positions))
    return {
        "rows": int(len(scored)),
        "start": scored["datetime"].iloc[0].date().isoformat(),
        "end": scored["datetime"].iloc[-1].date().isoformat(),
        "return": total_return,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "avg_pos": float(np.mean(positions)),
        "stress_avg_pos": stress_avg_pos,
        "stress_days": int(np.sum(stress_mask)),
        "buy_and_hold_return": float(buy_hold_nav[-1] - 1.0) * 100.0,
        "relative_return": total_return - float(buy_hold_nav[-1] - 1.0) * 100.0,
        "trades": int(np.sum(changes > 0.001)),
    }


def slice_metrics(
    frame: pd.DataFrame,
    price_series: pd.Series,
    windows: Mapping[str, tuple[str, str]] | None = None,
) -> dict[str, dict]:
    scored = frame.copy()
    scored["datetime"] = pd.to_datetime(scored["datetime"])
    result: dict[str, dict] = {}
    for name, (start, end) in (windows or DEFAULT_WINDOWS).items():
        sub = scored[(scored["datetime"] >= start) & (scored["datetime"] <= end)].copy()
        if len(sub) < 2:
            continue
        result[name] = compute_metrics(sub, price_series.loc[start:end])
    return result


def _variant_payload(frame: pd.DataFrame, price_series: pd.Series, windows: Mapping[str, tuple[str, str]] | None) -> dict:
    return {
        "test_proxy": compute_metrics(frame, price_series),
        "slices": slice_metrics(frame, price_series, windows),
    }


def _comparison(composed: dict, baseline: dict) -> dict[str, dict]:
    out: dict[str, dict] = {}
    labels = ["test_proxy", *sorted(set(composed.get("slices", {})) | set(baseline.get("slices", {})))]
    for label in labels:
        comp = composed["test_proxy"] if label == "test_proxy" else composed.get("slices", {}).get(label)
        base = baseline["test_proxy"] if label == "test_proxy" else baseline.get("slices", {}).get(label)
        if not comp or not base:
            continue
        out[label] = {
            "delta_return": comp["return"] - base["return"],
            "delta_sharpe": comp["sharpe"] - base["sharpe"],
            "delta_max_dd": comp["max_dd"] - base["max_dd"],
            "delta_avg_pos": comp["avg_pos"] - base["avg_pos"],
            "delta_trades": comp["trades"] - base["trades"],
        }
    return out


def _bull_guard_config_payload(cfg: BullParticipationGuardConfig | None) -> dict:
    guard = cfg or BullParticipationGuardConfig()
    return {
        "enabled": bool(guard.enabled),
        "ma_window": int(guard.ma_window),
        "momentum_window": int(guard.momentum_window),
        "high_window": int(guard.high_window),
        "min_periods": int(guard.min_periods),
        "min_momentum": float(guard.min_momentum),
        "max_drawdown": float(guard.max_drawdown),
        "action": "floor_raw_target_to_fast_position",
    }


def _compose_budget_usage_policy_frame_with_bull_guard(
    slow: pd.DataFrame,
    fast: pd.DataFrame,
    price_series: pd.Series,
    config: TwoSpeedBudgetConfig,
    guard: BullParticipationGuardConfig,
) -> pd.DataFrame:
    frame = slow[["datetime", "slow_position"]].merge(
        fast[["datetime", "fast_position"]],
        on="datetime",
        how="inner",
    )
    if frame.empty:
        raise ValueError("slow/fast policy frames have no overlapping datetimes")
    frame["datetime"] = pd.to_datetime(frame["datetime"])

    price = price_series.copy()
    if not isinstance(price.index, pd.DatetimeIndex):
        price.index = pd.to_datetime(price.index)
    price_frame = price.rename("price").reset_index()
    price_frame.columns = ["datetime", "price"]
    price_frame["datetime"] = pd.to_datetime(price_frame["datetime"])
    frame = frame.merge(price_frame, on="datetime", how="inner").sort_values("datetime").reset_index(drop=True)
    if frame.empty:
        raise ValueError("policy frames and price series have no overlapping datetimes")

    prices = frame["price"].astype(float)
    ma_window = max(int(guard.ma_window), 1)
    momentum_window = max(int(guard.momentum_window), 1)
    high_window = max(int(guard.high_window), 1)
    min_periods = max(int(guard.min_periods), 1)
    moving_average = prices.rolling(ma_window, min_periods=min(min_periods, ma_window)).mean()
    momentum = prices / prices.shift(momentum_window) - 1.0
    drawdown = prices / prices.rolling(high_window, min_periods=min(min_periods, high_window)).max() - 1.0

    current = 0.0
    days_since_trade = config.trade_lock_days
    modes: list[str] = []
    base_targets: list[float] = []
    raw_targets: list[float] = []
    targets: list[float] = []
    fast_usage: list[float] = []
    guard_flags: list[bool] = []

    for idx, row in frame.iterrows():
        slow_pos = float(row["slow_position"])
        usage = max(0.0, min(float(row["fast_position"]), 1.0))
        mode = infer_mode_from_slow_position(slow_pos)
        base_raw = compose_budget_usage_position(usage, mode, config)
        guard_ready = bool(
            prices.iloc[idx] > moving_average.iloc[idx]
            and momentum.iloc[idx] >= float(guard.min_momentum)
            and drawdown.iloc[idx] >= -float(guard.max_drawdown)
        )
        raw = max(base_raw, usage) if guard_ready else base_raw
        target = maybe_rebalance_position(current, raw, days_since_trade, config)
        if abs(target - current) >= config.trade_threshold:
            days_since_trade = 0
        else:
            days_since_trade += 1
        current = target
        modes.append(mode)
        base_targets.append(base_raw)
        raw_targets.append(raw)
        targets.append(target)
        fast_usage.append(usage)
        guard_flags.append(guard_ready)

    out = frame.drop(columns=["price"]).copy()
    out["mode"] = modes
    out["fast_usage"] = fast_usage
    out["base_two_speed_position"] = base_targets
    out["raw_two_speed_position"] = raw_targets
    out["position"] = targets
    out["bull_participation_guard"] = guard_flags
    out["bull_guard_ma"] = moving_average
    out["bull_guard_momentum"] = momentum
    out["bull_guard_drawdown"] = drawdown
    return out


def build_budget_usage_counterfactual(
    slow: pd.DataFrame,
    fast: pd.DataFrame,
    price_series: pd.Series,
    *,
    slow_label: str,
    fast_label: str,
    composed_label: str,
    description: str,
    profile: str | None = None,
    config: TwoSpeedBudgetConfig | None = None,
    bull_guard: BullParticipationGuardConfig | None = None,
    windows: Mapping[str, tuple[str, str]] | None = None,
) -> BudgetUsageCounterfactual:
    cfg = config or TwoSpeedBudgetConfig()
    slow_frame = _position_frame(slow).rename(columns={"position": "slow_position"})
    fast_frame = _position_frame(fast).rename(columns={"position": "fast_position"})
    guard_cfg = bull_guard or BullParticipationGuardConfig()
    daily = (
        _compose_budget_usage_policy_frame_with_bull_guard(slow_frame, fast_frame, price_series, cfg, guard_cfg)
        if guard_cfg.enabled
        else compose_budget_usage_policy_frame(slow_frame, fast_frame, cfg)
    )

    variants = {
        slow_label: _variant_payload(slow_frame.rename(columns={"slow_position": "position"}), price_series, windows),
        fast_label: _variant_payload(fast_frame.rename(columns={"fast_position": "position"}), price_series, windows),
        composed_label: _variant_payload(daily, price_series, windows),
    }
    payload = {
        "profile": profile or composed_label,
        "composition": "budget_usage",
        "description": description,
        "slow_label": slow_label,
        "fast_label": fast_label,
        "composed_label": composed_label,
        "config": {
            "mode_bands": dict(cfg.mode_bands),
            "mode_slack": dict(cfg.mode_slack),
            "trade_threshold": cfg.trade_threshold,
            "trade_lock_days": cfg.trade_lock_days,
            "bull_participation_guard": _bull_guard_config_payload(guard_cfg),
        },
        "variants": variants,
        "comparison": {
            "composed_vs_slow": _comparison(variants[composed_label], variants[slow_label]),
            "composed_vs_fast": _comparison(variants[composed_label], variants[fast_label]),
        },
    }
    return BudgetUsageCounterfactual(payload=payload, daily=daily)


def dashboard_daily_frame(daily: pd.DataFrame, price_series: pd.Series) -> pd.DataFrame:
    scored = align_price(daily, price_series)
    positions = scored["position"].to_numpy(dtype=float)
    prices = scored["price"].to_numpy(dtype=float)
    returns = np.diff(prices) / prices[:-1]
    strategy_nav = np.concatenate([[1.0], np.cumprod(1.0 + positions[:-1] * returns)]) * 1_000_000.0
    buyhold_nav = np.concatenate([[1.0], np.cumprod(1.0 + returns)]) * 1_000_000.0
    actions = ["HOLD"]
    for delta in np.diff(positions):
        actions.append("BUY" if delta > 0.001 else "SELL" if delta < -0.001 else "HOLD")

    out = scored.copy()
    out["strategy_nav"] = strategy_nav
    out["buyhold_nav"] = buyhold_nav
    out["excess"] = out["strategy_nav"] - out["buyhold_nav"]
    out["excess_pct"] = out["excess"] / out["buyhold_nav"] * 100.0
    out["action"] = actions
    leading = ["datetime", "strategy_nav", "price", "buyhold_nav", "position", "excess", "excess_pct", "action"]
    return out[leading + [col for col in out.columns if col not in leading]]


def _fmt_metric(metrics: dict) -> str:
    return (
        f"{metrics['return']:.2f}% / {metrics['sharpe']:.3f} / "
        f"{metrics['max_dd']:.2f}% / pos {metrics['avg_pos']:.3f} / trades {int(metrics['trades'])}"
    )


def render_budget_usage_review(payload: dict) -> str:
    lines = [
        f"# {payload['profile']}",
        "",
        payload["description"],
        "",
        f"- Slow path: `{payload['slow_label']}`",
        f"- Fast usage path: `{payload['fast_label']}`",
        f"- Composed path: `{payload['composed_label']}`",
        "",
        "| variant | return | sharpe | max_dd | avg_pos | trades |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, variant in payload["variants"].items():
        metric = variant["test_proxy"]
        lines.append(
            f"| {label} | {metric['return']:.2f}% | {metric['sharpe']:.3f} | "
            f"{metric['max_dd']:.2f}% | {metric['avg_pos']:.3f} | {int(metric['trades'])} |"
        )

    slice_names = sorted(
        {
            name
            for variant in payload["variants"].values()
            for name in variant.get("slices", {}).keys()
        }
    )
    if slice_names:
        lines.extend(["", "## Slices", "", "| slice | " + " | ".join(payload["variants"].keys()) + " |"])
        lines.append("| --- | " + " | ".join(["---"] * len(payload["variants"])) + " |")
        for slice_name in slice_names:
            row = [slice_name]
            for variant in payload["variants"].values():
                metric = variant.get("slices", {}).get(slice_name)
                row.append(_fmt_metric(metric) if metric else "")
            lines.append("| " + " | ".join(row) + " |")

    lines.extend(["", "## Deltas", ""])
    for name, comparison in payload.get("comparison", {}).items():
        lines.extend([f"### {name}", "", "| window | delta_return | delta_sharpe | delta_max_dd | delta_avg_pos |"])
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        for window, metric in comparison.items():
            lines.append(
                f"| {window} | {metric.get('delta_return', 0.0):.2f}pp | "
                f"{metric.get('delta_sharpe', 0.0):.3f} | "
                f"{metric.get('delta_max_dd', 0.0):.2f}pp | "
                f"{metric.get('delta_avg_pos', 0.0):.3f} |"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_counterfactual_artifacts(
    result: BudgetUsageCounterfactual,
    output_dir: Path,
    price_series: pd.Series,
) -> dict[str, Path]:
    import json

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = str(result.payload["profile"])
    json_path = output_dir / f"{stem}.json"
    daily_path = output_dir / f"{stem}_soxl_daily.csv"
    review_path = output_dir / f"{stem}.md"
    json_path.write_text(json.dumps(result.payload, indent=2), encoding="utf-8")
    dashboard_daily_frame(result.daily, price_series).to_csv(daily_path, index=False)
    review_path.write_text(render_budget_usage_review(result.payload), encoding="utf-8")
    return {"json": json_path, "daily": daily_path, "review": review_path}


__all__ = [
    "BudgetUsageCounterfactual",
    "BullParticipationGuardConfig",
    "DEFAULT_WINDOWS",
    "align_price",
    "build_budget_usage_counterfactual",
    "compute_metrics",
    "dashboard_daily_frame",
    "render_budget_usage_review",
    "slice_metrics",
    "write_counterfactual_artifacts",
]
