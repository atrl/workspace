from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import pandas as pd

from ..settings import ACCOUNT, EVIDENCE_DIR, ROOT

POST_TRAIN_REVIEW_DIR = EVIDENCE_DIR / "post_train_review"
DASHBOARD_BUILD_SCRIPT = ROOT / "scripts" / "build_backtest_dashboard.py"


def review_daily_csv_path(profile_name: str) -> Path:
    return POST_TRAIN_REVIEW_DIR / f"{profile_name}_soxl_daily.csv"


def write_review_daily_csv(
    out_path: Path,
    portfolio_df: pd.DataFrame,
    price_series: pd.Series,
    position_history: list[float],
    trade_signals: list[tuple],
) -> Path:
    account = portfolio_df["account"].astype(float)
    dates = account.index.strftime("%Y-%m-%d").tolist()
    prices = price_series.reindex(account.index).ffill().values
    initial_price = float(prices[0]) if len(prices) > 0 else 1.0
    buyhold_nav = ACCOUNT * (prices / initial_price)

    action_map: dict[str, str] = {}
    for dt, _price, _pos, signal_type in trade_signals:
        action_map[pd.Timestamp(dt).strftime("%Y-%m-%d")] = str(signal_type).upper()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["datetime", "strategy_nav", "price", "buyhold_nav", "position", "excess", "excess_pct", "action"])
        for idx, dt in enumerate(dates):
            strat_nav = float(account.iloc[idx])
            price = float(prices[idx])
            bh_nav = float(buyhold_nav[idx])
            pos = float(position_history[idx]) if idx < len(position_history) else 0.0
            excess = strat_nav - bh_nav
            excess_pct = (excess / bh_nav * 100.0) if bh_nav != 0 else 0.0
            action = action_map.get(dt, "HOLD")
            writer.writerow([dt, strat_nav, price, bh_nav, pos, excess, excess_pct, action])
    return out_path


def export_review_daily_csv(profile_name: str, bt_result: dict, price_series: pd.Series) -> Path | None:
    portfolio_df = bt_result.get("portfolio")
    if portfolio_df is None:
        return None
    return write_review_daily_csv(
        review_daily_csv_path(profile_name),
        portfolio_df,
        price_series,
        bt_result.get("position_history", []),
        bt_result.get("trade_signals", []),
    )


def rebuild_dashboard() -> bool:
    if not DASHBOARD_BUILD_SCRIPT.exists():
        return False
    proc = subprocess.run([sys.executable, str(DASHBOARD_BUILD_SCRIPT)], cwd=str(ROOT), check=False)
    return proc.returncode == 0


__all__ = [
    "export_review_daily_csv",
    "rebuild_dashboard",
    "review_daily_csv_path",
    "write_review_daily_csv",
]
