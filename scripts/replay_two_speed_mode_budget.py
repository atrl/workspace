#!/usr/bin/env python3
from __future__ import annotations

"""Replay a fixed two-speed composition from existing daily SOXL paths."""

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd

from src.soxl_quant.policy.two_speed_budget import (
    TwoSpeedBudgetConfig,
    compose_budget_usage_policy_frame,
    compose_budget_usage_position,
    compose_two_speed_position,
    infer_mode_from_slow_position,
    maybe_rebalance_position,
)

SLICES: Dict[str, Tuple[str, str]] = {
    "test_2020": ("2020-01-01", "2020-12-31"),
    "test_2022": ("2022-01-01", "2022-12-31"),
    "test_2023_2024": ("2023-01-01", "2024-12-31"),
    "test_proxy": ("2020-01-01", "2026-03-31"),
}


def _read_daily(path: Path, label: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["datetime"])
    required = {"datetime", "position"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")
    keep = ["datetime", "position"]
    if "price" in df.columns:
        keep.append("price")
    return df[keep].rename(columns={"position": f"{label}_position", "price": f"{label}_price"})


def _price_from_frame(frame: pd.DataFrame) -> pd.Series:
    price_cols = [col for col in frame.columns if col.endswith("_price")]
    for col in price_cols:
        series = pd.to_numeric(frame[col], errors="coerce")
        if series.notna().sum() > len(series) * 0.95:
            return series
    raise ValueError("no usable price column found")


def _action(prev_pos: float, pos: float, threshold: float) -> str:
    delta = pos - prev_pos
    if delta > threshold:
        return "BUY"
    if delta < -threshold:
        return "SELL"
    return "HOLD"


def replay_positions(
    frame: pd.DataFrame,
    position_column: str,
    price: pd.Series,
    trade_threshold: float,
) -> pd.DataFrame:
    nav = 1_000_000.0
    buyhold = 1_000_000.0
    prev_price = float(price.iloc[0])
    first_price = prev_price
    prev_pos = float(frame[position_column].iloc[0])
    rows = []
    for i, row in frame.iterrows():
        current_price = float(price.iloc[i])
        if i > 0 and prev_price > 0:
            pct = current_price / prev_price - 1.0
            nav *= 1.0 + prev_pos * pct
            buyhold = 1_000_000.0 * current_price / first_price
        pos = float(row[position_column])
        action = "HOLD" if i == 0 else _action(prev_pos, pos, trade_threshold)
        rows.append(
            {
                "datetime": row["datetime"],
                "strategy_nav": nav,
                "price": current_price,
                "buyhold_nav": buyhold,
                "position": pos,
                "excess": nav - buyhold,
                "excess_pct": ((nav / buyhold) - 1.0) * 100.0 if buyhold else 0.0,
                "action": action,
            }
        )
        prev_price = current_price
        prev_pos = pos
    return pd.DataFrame(rows)


def compose_two_speed_frame(
    slow: pd.DataFrame,
    fast: pd.DataFrame,
    config: TwoSpeedBudgetConfig,
    composition: str = "slack",
) -> pd.DataFrame:
    frame = slow.merge(fast, on="datetime", how="inner")
    price = _price_from_frame(frame)
    if composition == "budget_usage":
        policy_frame = compose_budget_usage_policy_frame(
            frame[["datetime", "slow_position"]],
            frame[["datetime", "fast_position"]],
            config,
        )
        out = replay_positions(policy_frame, "position", price, config.trade_threshold)
        out["mode"] = policy_frame["mode"].to_numpy()
        out["slow_position"] = policy_frame["slow_position"].to_numpy()
        out["fast_position"] = policy_frame["fast_position"].to_numpy()
        out["raw_two_speed_position"] = policy_frame["raw_two_speed_position"].to_numpy()
        return out

    current = 0.0
    days_since_trade = config.trade_lock_days
    targets = []
    modes = []
    raw_targets = []
    for _, row in frame.iterrows():
        slow_pos = float(row["slow_position"])
        fast_pos = float(row["fast_position"])
        mode = infer_mode_from_slow_position(slow_pos)
        if composition == "budget_usage":
            raw = compose_budget_usage_position(fast_pos, mode, config)
        else:
            raw = compose_two_speed_position(slow_pos, fast_pos, mode, config)
        target = maybe_rebalance_position(current, raw, days_since_trade, config)
        if abs(target - current) >= config.trade_threshold:
            days_since_trade = 0
        else:
            days_since_trade += 1
        current = target
        modes.append(mode)
        raw_targets.append(raw)
        targets.append(target)
    frame["mode"] = modes
    frame["raw_two_speed_position"] = raw_targets
    frame["two_speed_position"] = targets
    out = replay_positions(frame, "two_speed_position", price, config.trade_threshold)
    out["mode"] = modes
    out["slow_position"] = frame["slow_position"].to_numpy()
    out["fast_position"] = frame["fast_position"].to_numpy()
    out["raw_two_speed_position"] = raw_targets
    return out


def metrics_for_daily(df: pd.DataFrame) -> Dict[str, float]:
    nav = pd.to_numeric(df["strategy_nav"], errors="coerce")
    pos = pd.to_numeric(df["position"], errors="coerce")
    daily = nav.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    sharpe = 0.0
    if len(daily) > 1 and float(daily.std()) > 0:
        sharpe = float(daily.mean() / daily.std() * np.sqrt(252))
    peak = nav.cummax()
    max_dd = float(((peak - nav) / peak).max() * 100.0)
    trades = int((df["action"].isin(["BUY", "SELL"])).sum())
    return {
        "return": float((nav.iloc[-1] / nav.iloc[0] - 1.0) * 100.0),
        "sharpe": sharpe,
        "max_dd": max_dd,
        "avg_pos": float(pos.mean()),
        "trades": trades,
    }


def slice_daily(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    mask = (df["datetime"] >= pd.Timestamp(start)) & (df["datetime"] <= pd.Timestamp(end))
    return df.loc[mask].reset_index(drop=True)


def summarize_daily(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    return {name: metrics_for_daily(slice_daily(df, *window)) for name, window in SLICES.items()}


def _format_metrics(metrics: Dict[str, float]) -> str:
    return (
        f"{metrics['return']:.1f}% / {metrics['sharpe']:.3f} / "
        f"{metrics['max_dd']:.1f}% / pos {metrics['avg_pos']:.3f} / trades {metrics['trades']}"
    )


def write_markdown(path: Path, payload: Dict[str, object]) -> None:
    rows: Iterable[Tuple[str, Dict[str, Dict[str, float]]]] = payload["variants"].items()  # type: ignore[assignment]
    lines = [
        f"# {payload['profile']}",
        "",
        str(payload["description"]),
        "",
        "| variant | test_2020 ret/sh/dd | test_2022 ret/sh/dd/pos | test_2023_2024 ret/sh/dd | proxy ret/sh/dd/trades |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name, metrics in rows:
        test_2020 = metrics["test_2020"]
        test_2022 = metrics["test_2022"]
        test_2023_2024 = metrics["test_2023_2024"]
        proxy = metrics["test_proxy"]
        lines.append(
            f"| {name} | {_format_metrics(test_2020)} | {_format_metrics(test_2022)} | "
            f"{_format_metrics(test_2023_2024)} | {_format_metrics(proxy)} |"
        )
    decision = payload["decision"]
    lines.extend(["", "## Decision", "", str(decision), ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slow-daily", required=True, type=Path)
    parser.add_argument("--fast-daily", required=True, type=Path)
    parser.add_argument("--v19-daily", type=Path)
    parser.add_argument("--profile-name", default="two_speed_mode_budget_v1")
    parser.add_argument(
        "--composition",
        choices=("slack", "budget_usage"),
        default="slack",
        help="slack caps the fast path above the slow anchor; budget_usage treats the fast path as usage ratio inside mode bands",
    )
    parser.add_argument("--output-json", default=Path("runtime/post_train_review/two_speed_mode_budget_v1.json"), type=Path)
    parser.add_argument("--output-md", default=Path("runtime/post_train_review/two_speed_mode_budget_v1.md"), type=Path)
    parser.add_argument("--output-daily", default=Path("runtime/post_train_review/two_speed_mode_budget_v1_soxl_daily.csv"), type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = TwoSpeedBudgetConfig()
    slow = _read_daily(args.slow_daily, "slow")
    fast = _read_daily(args.fast_daily, "fast")
    composed = compose_two_speed_frame(slow, fast, cfg, composition=args.composition)
    price_frame = slow.merge(fast, on="datetime", how="inner")
    price = _price_from_frame(price_frame)

    variants: Dict[str, Dict[str, Dict[str, float]]] = {
        "mode_v1_slow": summarize_daily(replay_positions(price_frame, "slow_position", price, cfg.trade_threshold)),
        "imitation_v4_fast": summarize_daily(replay_positions(price_frame, "fast_position", price, cfg.trade_threshold)),
        args.profile_name: summarize_daily(composed),
    }
    if args.v19_daily is not None:
        v19 = _read_daily(args.v19_daily, "v19").merge(price_frame[["datetime"]], on="datetime", how="inner")
        variants["v19_reference"] = summarize_daily(replay_positions(v19, "v19_position", price, cfg.trade_threshold))

    two = variants[args.profile_name]
    fast_metrics = variants["imitation_v4_fast"]
    passes = {
        "test_2022_max_dd_le_30": two["test_2022"]["max_dd"] <= 30.0,
        "test_2022_avg_pos_le_035": two["test_2022"]["avg_pos"] <= 0.35,
        "test_proxy_sharpe_ge_090": two["test_proxy"]["sharpe"] >= 0.90,
        "test_2023_2024_return_ge_468": two["test_2023_2024"]["return"] >= 46.8,
        "proxy_dd_improves_fast_by_2pt": two["test_proxy"]["max_dd"] <= fast_metrics["test_proxy"]["max_dd"] - 2.0,
    }
    payload: Dict[str, object] = {
        "profile": args.profile_name,
        "description": (
            "Fixed-signal counterfactual combining the slow `mode_v1` risk governor "
            "with the fast `imitation_v4` path as a budget-usage signal."
            if args.composition == "budget_usage"
            else "Fixed-signal counterfactual combining the slow `mode_v1` risk anchor "
            "with the fast `imitation_v4` participation path."
        ),
        "config": {
            "composition": args.composition,
            "mode_slack": dict(cfg.mode_slack),
            "mode_bands": {k: list(v) for k, v in cfg.mode_bands.items()},
            "trade_threshold": cfg.trade_threshold,
            "trade_lock_days": cfg.trade_lock_days,
        },
        "variants": variants,
        "acceptance": passes,
        "decision": "candidate" if all(passes.values()) else "dropped",
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    composed.to_csv(args.output_daily, index=False)
    write_markdown(args.output_md, payload)
    print(f"[replay] decision={payload['decision']}")
    print(f"[replay] json={args.output_json}")
    print(f"[replay] md={args.output_md}")
    print(f"[replay] daily={args.output_daily}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
