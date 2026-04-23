"""Independent signal backtest for SOXL single-asset features.

Simulates a simple long/flat strategy based on feature thresholds:
    feature > threshold_long  → long (100% position)
    feature < threshold_short → flat (0% position)
    otherwise                 → hold previous position

Usage:
    qlib_venv/bin/python .codex/skills/soxl-factor-mining/scripts/backtest_signal.py \
      --feature RB_PANIC_SCORE_PROXY --horizon 5 --threshold-long 0.0 --threshold-short -0.5
    qlib_venv/bin/python .codex/skills/soxl-factor-mining/scripts/backtest_signal.py \
      --feature RB_PANIC_SCORE_PROXY --sweep --backfill
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from soxl_quant.data import build_feature_frame, init_qlib
from pathlib import Path

DB_PATH = ROOT / "data" / "soxl_factor_library.db"


def _ensure_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS feature_backtests (
            feature_name TEXT,
            horizon_days INTEGER,
            threshold_long REAL,
            threshold_short REAL,
            hold_days INTEGER,
            signal_accuracy REAL,
            profit_factor REAL,
            avg_return REAL,
            max_drawdown REAL,
            sharpe REAL,
            win_rate REAL,
            num_signals INTEGER,
            PRIMARY KEY (feature_name, horizon_days, threshold_long, hold_days)
        );
        """
    )
    return conn


def backtest_signal(
    factor: pd.Series,
    price: pd.Series,
    horizon: int,
    threshold_long: float,
    threshold_short: float,
    hold_days: int = 1,
    transaction_cost: float = 0.001,
) -> dict:
    """Run a simple long/flat backtest based on feature thresholds."""
    df = pd.DataFrame({"factor": factor, "price": price}).dropna()
    if len(df) < 30:
        return {}

    df["return"] = df["price"].pct_change().shift(-1).fillna(0.0)
    df["forward_return"] = (df["price"].shift(-horizon) / df["price"] - 1.0).fillna(0.0)

    # Generate signal: explicit state machine
    # 1=long, 0=flat. Start flat, switch on threshold crosses.
    signals = []
    pos = 0.0
    for f in df["factor"]:
        if f > threshold_long:
            pos = 1.0
        elif f < threshold_short:
            pos = 0.0
        # else: keep previous pos
        signals.append(pos)
    df["signal"] = signals

    # Apply hold_days minimum holding period
    if hold_days > 1:
        held_signals = []
        current = 0.0
        countdown = 0
        for s in df["signal"]:
            if countdown <= 0:
                current = s
                countdown = hold_days
            held_signals.append(current)
            countdown -= 1
        df["signal"] = held_signals

    # Strategy returns
    df["strat_ret"] = df["signal"].shift(1).fillna(0.0) * df["return"]

    # Transaction cost on signal changes
    df["signal_change"] = df["signal"].diff().abs()
    df["strat_ret"] -= df["signal_change"] * transaction_cost

    df["cum"] = (1.0 + df["strat_ret"]).cumprod()
    df["bench_cum"] = (1.0 + df["return"]).cumprod()

    # Metrics
    returns = df["strat_ret"].dropna()
    if len(returns) == 0 or returns.std() == 0:
        return {}

    total_ret = float(df["cum"].iloc[-1] - 1.0) if len(df["cum"]) > 0 else 0.0
    ann_factor = 252 / len(returns)
    ann_ret = (1.0 + total_ret) ** ann_factor - 1.0
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0

    cummax = df["cum"].cummax()
    drawdown = (df["cum"] - cummax) / cummax
    max_dd = float(drawdown.min())

    # Signal quality: when signal=1, was forward_return positive?
    long_mask = df["signal"] > 0.5
    if long_mask.sum() > 0:
        signal_accuracy = (df.loc[long_mask, "forward_return"] > 0).mean()
        win_rate = (df.loc[long_mask, "return"] > 0).mean()
        wins = df.loc[long_mask & (df["return"] > 0), "return"].sum()
        losses = abs(df.loc[long_mask & (df["return"] < 0), "return"].sum())
        profit_factor = wins / losses if losses > 0 else float("inf")
        avg_return = df.loc[long_mask, "return"].mean()
    else:
        signal_accuracy = 0.0
        win_rate = 0.0
        profit_factor = 0.0
        avg_return = 0.0

    num_signals = int((df["signal_change"] > 0.5).sum())
    turnover = float(df["signal_change"].sum()) / len(df)

    return {
        "feature_name": factor.name,
        "horizon_days": horizon,
        "threshold_long": threshold_long,
        "threshold_short": threshold_short,
        "hold_days": hold_days,
        "signal_accuracy": float(signal_accuracy),
        "profit_factor": float(profit_factor),
        "avg_return": float(avg_return),
        "max_drawdown": float(max_dd),
        "sharpe": float(sharpe),
        "win_rate": float(win_rate),
        "num_signals": num_signals,
        "annualized_return": float(ann_ret),
        "turnover": float(turnover),
    }


def sweep_thresholds(
    factor: pd.Series,
    price: pd.Series,
    horizon: int,
    hold_days: int = 1,
) -> list[dict]:
    """Sweep threshold_long and threshold_short combinations.

    Uses percentile-based thresholds adapted to the factor distribution.
    For sparse factors (many zeros), uses higher percentiles to avoid
    always-long or never-long scenarios.

    Exit threshold is set to a lower percentile than entry to create
    a natural hysteresis band and ensure trades can actually complete.
    """
    # Compute percentiles
    p10, p25, p50, p75, p90, p95 = factor.quantile([0.10, 0.25, 0.50, 0.75, 0.90, 0.95]).values

    # Determine if factor is sparse (many identical low values)
    unique_ratio = factor.nunique() / len(factor)
    is_sparse = unique_ratio < 0.3 or (factor == factor.min()).mean() > 0.5

    if is_sparse:
        # For sparse factors: enter at high percentile, exit when below median
        long_thresholds = [p75, p90, p95]
        short_thresholds = [p25, p50]
    else:
        long_thresholds = [p50, p75, p90]
        short_thresholds = [p10, p25, p50]

    results = []
    fmin = factor.min()
    for tlong in long_thresholds:
        for tshort in short_thresholds:
            # If tshort equals the factor minimum, it will never trigger exit.
            # Bump it slightly above minimum to allow actual trades.
            if abs(tshort - fmin) < 1e-9:
                tshort = fmin + 1e-6
            if tlong <= tshort:
                continue
            res = backtest_signal(factor, price, horizon, tlong, tshort, hold_days)
            if res:
                results.append(res)
    return results


def backfill_db(conn: sqlite3.Connection, results: list[dict]) -> None:
    for r in results:
        conn.execute(
            """INSERT INTO feature_backtests
               (feature_name, horizon_days, threshold_long, threshold_short, hold_days,
                signal_accuracy, profit_factor, avg_return, max_drawdown, sharpe, win_rate, num_signals)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(feature_name, horizon_days, threshold_long, hold_days) DO UPDATE SET
                 threshold_short=excluded.threshold_short,
                 signal_accuracy=excluded.signal_accuracy,
                 profit_factor=excluded.profit_factor,
                 avg_return=excluded.avg_return,
                 max_drawdown=excluded.max_drawdown,
                 sharpe=excluded.sharpe,
                 win_rate=excluded.win_rate,
                 num_signals=excluded.num_signals""",
            (
                r["feature_name"],
                r["horizon_days"],
                r["threshold_long"],
                r["threshold_short"],
                r["hold_days"],
                r["signal_accuracy"],
                r["profit_factor"],
                r["avg_return"],
                r["max_drawdown"],
                r["sharpe"],
                r["win_rate"],
                r["num_signals"],
            ),
        )
    conn.commit()
    print(f"[Backfill] Wrote {len(results)} backtest records to DB")


def main():
    parser = argparse.ArgumentParser(description="Independent signal backtest for SOXL features")
    parser.add_argument("--feature", required=True, help="Feature name")
    parser.add_argument("--horizon", type=int, default=5, help="Forward return horizon")
    parser.add_argument("--threshold-long", type=float, default=0.0, help="Long threshold")
    parser.add_argument("--threshold-short", type=float, default=-0.5, help="Short/flat threshold")
    parser.add_argument("--hold-days", type=int, default=1, help="Hold days after signal")
    parser.add_argument("--sweep", action="store_true", help="Sweep thresholds")
    parser.add_argument("--backfill", action="store_true", help="Write to DB")
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default="2026-04-01")
    args = parser.parse_args()

    init_qlib(Path("qlib_soxl_binary"))

    print(f"Loading SOXL data ({args.start} ~ {args.end})...")
    feature_df, price = build_feature_frame(args.start, args.end, feature_packs=("rulebase_signals",))
    feature_df = feature_df.droplevel("instrument")
    price = price.droplevel("instrument")

    if args.feature not in feature_df.columns:
        print(f"ERROR: Feature '{args.feature}' not found. Available: {list(feature_df.columns)}")
        sys.exit(1)

    factor = feature_df[args.feature]

    if args.sweep:
        results = sweep_thresholds(factor, price, args.horizon, args.hold_days)
        print(f"\n{'='*100}")
        print(f"  SWEEP RESULTS: {args.feature}  horizon={args.horizon}d")
        print(f"{'='*100}")
        print(
            f"{'thr_long':>8} {'thr_short':>9} {'acc':>6} {'PF':>5} {'avg_ret':>8} "
            f"{'sharpe':>6} {'win%':>5} {'MDD':>7} {'n_sig':>6}"
        )
        for r in results:
            print(
                f"{r['threshold_long']:>8.3f} {r['threshold_short']:>9.3f} "
                f"{r['signal_accuracy']:>6.2%} {r['profit_factor']:>5.2f} "
                f"{r['avg_return']:>8.4f} {r['sharpe']:>6.2f} {r['win_rate']:>5.1%} "
                f"{r['max_drawdown']:>7.2%} {r['num_signals']:>6d}"
            )
    else:
        result = backtest_signal(
            factor, price, args.horizon,
            args.threshold_long, args.threshold_short, args.hold_days,
        )
        results = [result] if result else []
        print(f"\nBacktest: {args.feature}  horizon={args.horizon}d")
        for k, v in result.items():
            if isinstance(v, float):
                print(f"  {k:20s}: {v:.4f}")
            else:
                print(f"  {k:20s}: {v}")

    if args.backfill and results:
        conn = _ensure_db()
        try:
            backfill_db(conn, results)
        finally:
            conn.close()

    # Save CSV
    out_dir = ROOT / "docs" / "factor_research"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"backtest_{args.feature}.csv"
    pd.DataFrame(results).to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
