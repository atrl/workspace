#!/usr/bin/env python3
"""Cross-ETF diagnostics for the risk-budget usage mechanism."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.replay_two_speed_mode_budget import replay_positions, summarize_daily
from src.soxl_quant.backtest import backtest_profile
from src.soxl_quant.data import build_dataset, feature_columns_for_packs, init_qlib, refresh_qlib_binary, segment_frame
from src.soxl_quant.policy.policy_builder import build_policy
from src.soxl_quant.settings import (
    ACCOUNT,
    ETF_UNIVERSE,
    MODEL_DIR,
    TEST_SEG,
    TRAIN_SEG,
    VALID_SEG,
    load_profile,
)

DATA_DIR = ROOT / "data" / "generalization"
RUNTIME_DIR = ROOT / "runtime" / "generalization"
QLIB_RUNTIME = RUNTIME_DIR / "qlib_stores"
DEFAULT_FIXED_DAILY = ROOT / "runtime" / "post_train_review" / "risk_budget_usage_v1_soxl_daily.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="risk_budget_usage_v1")
    parser.add_argument("--model", default="soxl_balanced_rule_residual_imitation_v4_actor.pt")
    parser.add_argument("--fixed-daily", type=Path, default=DEFAULT_FIXED_DAILY)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=ROOT / "runtime" / "post_train_review" / "risk_budget_usage_v1_cross_etf.json",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=ROOT / "runtime" / "post_train_review" / "risk_budget_usage_v1_cross_etf.md",
    )
    parser.add_argument(
        "--daily-dir",
        type=Path,
        default=ROOT / "runtime" / "post_train_review" / "risk_budget_usage_v1_cross_etf_daily",
    )
    return parser.parse_args()


def load_actor(model_path: Path, profile_name: str):
    profile = load_profile(profile_name)
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    _, actor = build_policy(
        int(ckpt["obs_dim"]),
        profile,
        use_moe=bool(ckpt.get("use_moe", False)),
        hard_regime_routing=bool(ckpt.get("use_moe", False)),
    )
    actor.load_state_dict(ckpt["state_dict"])
    actor.eval()
    return actor, profile


def build_etf_dataset(etf: str, profile) -> tuple[pd.DataFrame, pd.Series]:
    meta = ETF_UNIVERSE[etf]
    csv_path = DATA_DIR / f"{etf.lower()}_with_underlying_vix.csv"
    qlib_dir = QLIB_RUNTIME / etf.lower()
    refresh_qlib_binary(
        csv_path=csv_path,
        qlib_dir=qlib_dir,
        leveraged_etf=etf,
        underlying_etf=str(meta["underlying"]),
        leverage_ratio=float(meta["leverage"]),
    )
    init_qlib(qlib_dir)
    dataset = build_dataset(
        leveraged_etf=etf,
        underlying_etf=str(meta["underlying"]),
        leverage_ratio=float(meta["leverage"]),
        train_seg=TRAIN_SEG,
        valid_seg=VALID_SEG,
        test_seg=TEST_SEG,
        feature_packs=tuple(profile.feature.packs),
        profile_name=profile.name,
    )
    test_df = segment_frame(dataset, "test")
    price = dataset.handler.price_frame.droplevel("instrument")["price"].loc[test_df.index]
    return test_df, price


def _action(prev_pos: float, pos: float, threshold: float = 0.04) -> str:
    delta = pos - prev_pos
    if delta > threshold:
        return "BUY"
    if delta < -threshold:
        return "SELL"
    return "HOLD"


def _daily_from_backtest(result: dict[str, Any], price: pd.Series) -> pd.DataFrame:
    portfolio = result["portfolio"]
    account = portfolio["account"].astype(float)
    aligned_price = price.reindex(account.index).ffill().bfill()
    first_price = float(aligned_price.iloc[0])
    buyhold_nav = ACCOUNT * aligned_price / first_price
    positions = list(result.get("position_history", []))
    rows: list[dict[str, Any]] = []
    prev_pos = 0.0
    for idx, (dt, nav) in enumerate(account.items()):
        pos = float(positions[idx]) if idx < len(positions) else prev_pos
        action = "HOLD" if idx == 0 else _action(prev_pos, pos)
        bh = float(buyhold_nav.loc[dt])
        rows.append(
            {
                "datetime": pd.Timestamp(dt),
                "strategy_nav": float(nav),
                "price": float(aligned_price.loc[dt]),
                "buyhold_nav": bh,
                "position": pos,
                "excess": float(nav) - bh,
                "excess_pct": ((float(nav) / bh) - 1.0) * 100.0 if bh else 0.0,
                "action": action,
            }
        )
        prev_pos = pos
    return pd.DataFrame(rows)


def _write_daily(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = frame.copy()
    out["datetime"] = pd.to_datetime(out["datetime"]).dt.strftime("%Y-%m-%d")
    out.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def _buyhold_metrics(price: pd.Series) -> dict[str, float]:
    price = price.ffill().bfill().astype(float)
    nav = ACCOUNT * price / float(price.iloc[0])
    daily = nav.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    sharpe = 0.0
    if len(daily) > 1 and float(daily.std()) > 0:
        sharpe = float(daily.mean() / daily.std() * np.sqrt(252))
    peak = nav.cummax()
    return {
        "return": float((nav.iloc[-1] / nav.iloc[0] - 1.0) * 100.0),
        "sharpe": sharpe,
        "max_dd": float(((peak - nav) / peak).max() * 100.0),
        "avg_pos": 1.0,
        "trades": 1,
    }


def _format(metrics: dict[str, float]) -> str:
    return (
        f"{metrics['return']:.1f}% / {metrics['sharpe']:.3f} / "
        f"{metrics['max_dd']:.1f}% / pos {metrics['avg_pos']:.3f}"
    )


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# risk_budget_usage_v1 cross-ETF diagnostic",
        "",
        "This diagnostic freezes the merged risk-budget usage mechanism and tests two separable capabilities:",
        "",
        "- `fixed_signal_transfer`: use the existing SOXL fixed replay position path on each ETF's own price series.",
        "- `actor_budget_runtime`: run the retained `imitation_v4` actor under the `risk_budget_usage_v1` budget-usage mapping on each ETF.",
        "",
        "The retained `balanced_rule_residual_mode_v1` actor bundle is absent, so this is not a full two-actor zero-shot composer.",
        "",
        "| ETF | fixed proxy ret/sh/dd/pos | fixed 2022 ret/sh/dd/pos | actor proxy ret/sh/dd/pos | actor 2022 ret/sh/dd/pos | buyhold proxy ret/sh/dd | decision |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for etf, metrics in payload["results"].items():
        fixed = metrics["fixed_signal_transfer"]
        actor = metrics["actor_budget_runtime"]
        buyhold = metrics["buyhold"]
        decision = metrics["decision"]
        lines.append(
            f"| {etf} | {_format(fixed['test_proxy'])} | {_format(fixed['test_2022'])} | "
            f"{_format(actor['test_proxy'])} | {_format(actor['test_2022'])} | "
            f"{_format(buyhold['test_proxy'])} | {decision} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            str(payload["interpretation"]),
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.daily_dir.mkdir(parents=True, exist_ok=True)
    profile_name = str(args.profile)
    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = MODEL_DIR / model_path
    actor, profile = load_actor(model_path, profile_name)
    feature_columns = feature_columns_for_packs(profile.feature.packs, profile.name)

    fixed_path = args.fixed_daily
    if not fixed_path.is_absolute():
        fixed_path = ROOT / fixed_path
    fixed_daily = pd.read_csv(fixed_path, parse_dates=["datetime"])
    if "position" not in fixed_daily.columns:
        raise ValueError(f"{fixed_path} missing `position` column")
    fixed_positions = fixed_daily[["datetime", "position"]].rename(columns={"position": "fixed_position"})

    results: dict[str, Any] = {}
    for asset_idx, etf in enumerate(ETF_UNIVERSE):
        test_df, price = build_etf_dataset(etf, profile)
        price_frame = pd.DataFrame({"datetime": price.index, "etf_price": price.to_numpy(dtype=float)})
        fixed_frame = fixed_positions.merge(price_frame, on="datetime", how="inner")
        fixed_replay = replay_positions(
            fixed_frame.rename(columns={"fixed_position": "position"}),
            "position",
            fixed_frame["etf_price"],
            trade_threshold=0.04,
        )
        _write_daily(args.daily_dir / f"{etf.lower()}_fixed_signal_transfer.csv", fixed_replay)

        bt = backtest_profile(
            actor=actor,
            feature_df=test_df,
            price_series=price,
            position_cap_config=profile.position_cap,
            position_budget_config=profile.position_budget,
            execution_config=profile.execution,
            mode_router_config=profile.mode_router,
            symbol=etf,
            asset_idx=asset_idx,
            feature_columns=feature_columns,
            test_seg=TEST_SEG,
            model_config=profile.model,
        )
        actor_daily = _daily_from_backtest(bt, price)
        _write_daily(args.daily_dir / f"{etf.lower()}_actor_budget_runtime.csv", actor_daily)

        buyhold_daily = pd.DataFrame(
            {
                "datetime": price.index,
                "strategy_nav": ACCOUNT * price.to_numpy(dtype=float) / float(price.iloc[0]),
                "price": price.to_numpy(dtype=float),
                "buyhold_nav": ACCOUNT * price.to_numpy(dtype=float) / float(price.iloc[0]),
                "position": 1.0,
                "action": "HOLD",
            }
        )
        fixed_metrics = summarize_daily(fixed_replay)
        actor_metrics = summarize_daily(actor_daily)
        buyhold_metrics = {"test_proxy": _buyhold_metrics(price)}
        actor_proxy = actor_metrics["test_proxy"]
        fixed_proxy = fixed_metrics["test_proxy"]
        decision = "diagnostic"
        if actor_proxy["max_dd"] < buyhold_metrics["test_proxy"]["max_dd"] and actor_proxy["return"] > 0:
            decision = "risk transfers, return weak" if actor_proxy["return"] < fixed_proxy["return"] else "runtime transfers"
        if actor_proxy["return"] < 0 or actor_proxy["sharpe"] < 0:
            decision = "fails runtime transfer"
        results[etf] = {
            "fixed_signal_transfer": fixed_metrics,
            "actor_budget_runtime": actor_metrics,
            "buyhold": buyhold_metrics,
            "decision": decision,
        }
        print(
            f"{etf}: fixed proxy {_format(fixed_proxy)} | "
            f"actor proxy {_format(actor_proxy)} | decision={decision}"
        )

    interpretation = (
        "A result only counts as promotable if actor_budget_runtime, not just fixed_signal_transfer, "
        "keeps drawdown controlled and preserves enough participation across ETFs. Fixed signal transfer "
        "is a timing-path stress test and remains exposed to SOXL test-path selection bias."
    )
    payload = {
        "profile": profile_name,
        "model": str(model_path),
        "fixed_daily": str(fixed_path),
        "test_seg": TEST_SEG,
        "results": results,
        "interpretation": interpretation,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown(args.output_md, payload)
    print(f"[done] wrote {args.output_json}")
    print(f"[done] wrote {args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
