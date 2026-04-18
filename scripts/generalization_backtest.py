#!/usr/bin/env python3
"""Zero-shot generalization backtest for the SOXL-trained actor on other 3x ETFs."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Silence warnings
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.soxl_quant.backtest import backtest_profile
from src.soxl_quant.data import (
    build_dataset,
    flatten_feature_frame,
    init_qlib,
    refresh_qlib_binary,
    segment_frame,
)
from src.soxl_quant.policy.policy_builder import build_policy
from src.soxl_quant.settings import (
    ACCOUNT,
    ETF_UNIVERSE,
    MODEL_DIR,
    TEST_SEG,
    TRAIN_SEG,
    VALID_SEG,
    ExecutionConfig,
    PositionCapConfig,
    StrategyConfig,
    load_profile,
)

DATA_DIR = ROOT / "data" / "generalization"
RUNTIME_DIR = ROOT / "runtime" / "generalization"
QLIB_RUNTIME = RUNTIME_DIR / "qlib_stores"


def _export_detailed_csv(
    out_path: Path,
    portfolio_df: pd.DataFrame,
    price_series: pd.Series,
    position_history: list[float],
    trade_signals: list[tuple],
) -> None:
    """Export daily rows compatible with dashboard detailed_trades.csv format."""
    account = portfolio_df["account"].astype(float)
    dates = account.index.strftime("%Y-%m-%d").tolist()
    prices = price_series.reindex(account.index).fillna(method="ffill").values
    # buy-and-hold NAV
    initial_price = float(prices[0]) if len(prices) > 0 else 1.0
    buyhold_nav = ACCOUNT * (prices / initial_price)

    # action map from trade_signals
    action_map: dict[str, str] = {}
    for dt, _price, _pos, stype in trade_signals:
        key = pd.Timestamp(dt).strftime("%Y-%m-%d")
        action_map[key] = stype.upper()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["datetime", "strategy_nav", "price", "buyhold_nav", "position", "excess", "excess_pct", "action"])
        for i, dt in enumerate(dates):
            strat_nav = float(account.iloc[i])
            price = float(prices[i])
            bh_nav = float(buyhold_nav[i])
            pos = float(position_history[i]) if i < len(position_history) else 0.0
            excess = strat_nav - bh_nav
            excess_pct = (excess / bh_nav * 100) if bh_nav != 0 else 0.0
            action = action_map.get(dt, "HOLD")
            writer.writerow([dt, strat_nav, price, bh_nav, pos, excess, excess_pct, action])




def load_actor(checkpoint_path: Path, profile: StrategyConfig, use_moe: bool = False):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    obs_dim = int(ckpt["obs_dim"])
    policy, actor = build_policy(obs_dim, profile, use_moe=use_moe)
    actor.load_state_dict(ckpt["state_dict"])
    actor.eval()
    return actor


def run_etf_backtest(
    etf: str,
    underlying: str,
    leverage: float,
    actor,
    position_cap_config: PositionCapConfig,
    test_seg: tuple[str, str],
    execution_config: ExecutionConfig | None = None,
) -> dict:
    csv_path = DATA_DIR / f"{etf.lower()}_with_underlying_vix.csv"
    qlib_dir = QLIB_RUNTIME / etf.lower()

    print(f"\n=== {etf} ({underlying}) ===")
    print(f"Refreshing qlib binary at {qlib_dir} ...")
    refresh_qlib_binary(
        csv_path=csv_path,
        qlib_dir=qlib_dir,
        leveraged_etf=etf,
        underlying_etf=underlying,
        leverage_ratio=leverage,
    )

    print(f"Initializing qlib ...")
    init_qlib(qlib_dir)

    print(f"Building dataset ...")
    # Use the same train/valid/test splits for comparability.
    dataset = build_dataset(
        leveraged_etf=etf,
        underlying_etf=underlying,
        leverage_ratio=leverage,
        train_seg=TRAIN_SEG,
        valid_seg=VALID_SEG,
        test_seg=test_seg,
    )

    test_df = segment_frame(dataset, "test")
    price = dataset.handler.price_frame.droplevel("instrument")["price"].loc[test_df.index]

    asset_idx = list(ETF_UNIVERSE.keys()).index(etf)
    print(f"Running backtest on {len(test_df)} rows ({test_df.index[0].date()} ~ {test_df.index[-1].date()}) ...")
    result = backtest_profile(
        actor=actor,
        feature_df=test_df,
        price_series=price,
        position_cap_config=position_cap_config,
        execution_config=execution_config,
        symbol=etf,
        asset_idx=asset_idx,
        test_seg=test_seg,
    )

    # Compute buy-and-hold summary for dashboard summary.json compatibility
    pdf = result["portfolio"]
    if isinstance(pdf, tuple):
        pdf = pdf[0]
    account = pdf["account"].astype(float)
    strategy_return = account.iloc[-1] / ACCOUNT - 1
    buyhold_price = price.reindex(account.index).fillna(method="ffill")
    buyhold_nav = ACCOUNT * (buyhold_price / buyhold_price.iloc[0])
    buyhold_return = buyhold_nav.iloc[-1] / ACCOUNT - 1
    excess_return = strategy_return - buyhold_return

    peak_s = account.cummax()
    max_dd_strategy = ((peak_s - account) / peak_s).max()
    peak_b = buyhold_nav.cummax()
    max_dd_buyhold = ((peak_b - buyhold_nav) / peak_b).max()

    return {
        "return": float(result["return"]),
        "sharpe": float(result["sharpe"]),
        "max_dd": float(result["max_dd"]),
        "trades": int(result["trades"]),
        "_portfolio": pdf,
        "_position_history": result["position_history"],
        "_trade_signals": result.get("trade_signals", []),
        "_price_series": price,
        "_summary": {
            "etf": etf,
            "strategy_return": float(strategy_return),
            "buyhold_return": float(buyhold_return),
            "excess_return": float(excess_return),
            "max_dd_strategy": float(max_dd_strategy),
            "max_dd_buyhold": float(max_dd_buyhold),
            "trades": int(result["trades"]),
            "buy_count": sum(1 for _, _, _, t in result.get("trade_signals", []) if t == "buy"),
            "sell_count": sum(1 for _, _, _, t in result.get("trade_signals", []) if t == "sell"),
            "max_position": max(result["position_history"]) if result["position_history"] else 0.0,
            "min_position": min(result["position_history"]) if result["position_history"] else 0.0,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="balanced", help="Profile name in configs/soxl/")
    parser.add_argument("--model", default="soxl_balanced_actor.pt", help="Model checkpoint filename")
    parser.add_argument("--moe", action="store_true", help="Model uses Mixture-of-Experts architecture")
    parser.add_argument("--etf", help="Run a single ETF (default: all)")
    parser.add_argument("--tag", help="Version tag for saving detailed results (e.g. v2.1.0). If omitted, only aggregate results are saved.")
    args = parser.parse_args()

    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)

    profile = load_profile(args.profile)
    model_path = MODEL_DIR / args.model
    print(f"Loading actor from {model_path} ...")
    actor = load_actor(model_path, profile, use_moe=args.moe)

    targets = {args.etf.upper(): ETF_UNIVERSE[args.etf.upper()]} if args.etf else ETF_UNIVERSE

    raw_results: dict[str, dict] = {}
    for etf, meta in targets.items():
        try:
            result = run_etf_backtest(
                etf=etf,
                underlying=meta["underlying"],
                leverage=meta["leverage"],
                actor=actor,
                position_cap_config=profile.position_cap,
                test_seg=TEST_SEG,
                execution_config=profile.execution,
            )
            raw_results[etf] = result
            print(f"Result: return={result['return']:.2f}% sharpe={result['sharpe']:.3f} max_dd={result['max_dd']:.2f}% trades={result['trades']}")
        except Exception as exc:
            print(f"ERROR running {etf}: {exc}")
            import traceback
            traceback.print_exc()

    # Aggregate results (backward compatible)
    agg_results = {etf: {k: v for k, v in r.items() if not k.startswith("_")} for etf, r in raw_results.items()}
    results_path = RUNTIME_DIR / "results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(agg_results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Versioned detailed export
    if args.tag:
        version_dir = RUNTIME_DIR / "versions" / args.tag
        detail_dir = version_dir / "detailed"
        detail_dir.mkdir(parents=True, exist_ok=True)
        summary_items = []
        for etf, r in raw_results.items():
            csv_path = detail_dir / f"{etf.lower()}_detailed_trades.csv"
            _export_detailed_csv(
                csv_path,
                r["_portfolio"],
                r["_price_series"],
                r["_position_history"],
                r["_trade_signals"],
            )
            print(f"[version] {csv_path}")
            summary_items.append(r["_summary"])
        # summary.json compatible with dashboard
        summary_path = detail_dir / "summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_items, f, indent=2, ensure_ascii=False)
        print(f"[version] {summary_path}")
        # also save aggregate results under version
        version_results_path = version_dir / "results.json"
        with open(version_results_path, "w", encoding="utf-8") as f:
            json.dump(agg_results, f, indent=2)
        print(f"[version] {version_results_path}")

    # Text report
    lines = ["Generalization Backtest Report", "=" * 60, ""]
    lines.append(f"{'ETF':<8} {'Return':>10} {'Sharpe':>8} {'MaxDD':>10} {'Trades':>8}")
    lines.append("-" * 60)
    for etf, r in agg_results.items():
        lines.append(
            f"{etf:<8} {r['return']:>10.2f}% {r['sharpe']:>8.3f} {r['max_dd']:>10.2f}% {r['trades']:>8d}"
        )
    report_path = RUNTIME_DIR / "report.txt"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
