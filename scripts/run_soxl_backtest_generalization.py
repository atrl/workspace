"""Run the rule-based soxl_backtest on all 3x ETFs for cross-ETF comparison."""
from __future__ import annotations

import json
import sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from soxl_backtest.backtest import SOXLBacktest

ETFS = {
    "SOXL": "SOXL",
    "TQQQ": "TQQQ",
    "UPRO": "UPRO",
    "SPXL": "SPXL",
    "TECL": "TECL",
}


def run_backtest(etf: str):
    params = {
        "USE_LONGBRIDGE": False,
        "STOCK_CODE": etf,
        "START_DATE": "2020-01-02",
        "END_DATE": "2026-03-31",
        "INITIAL_CAPITAL": 100_000,
    }
    bt = SOXLBacktest(params)
    bt.run()

    capital = np.array(bt.capital_history)
    daily_ret = np.diff(capital) / capital[:-1]
    peak = np.maximum.accumulate(capital)
    max_dd = float(np.max((peak - capital) / peak) * 100)
    total_ret = float((capital[-1] / capital[0] - 1) * 100)
    sharpe = float(np.mean(daily_ret) / (np.std(daily_ret) + 1e-8) * np.sqrt(252))
    avg_pos = float(np.mean(bt.position_history))
    trades = len(bt.trades)

    return {
        "return": total_ret,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "trades": trades,
        "avg_pos": avg_pos,
    }


def main():
    results = {}
    for etf in ETFS:
        print(f"\n{'='*60}")
        print(f"Running rule-based backtest for {etf}")
        print(f"{'='*60}")
        try:
            result = run_backtest(etf)
            results[etf] = result
            print(f"[RESULT] {etf}: return={result['return']:.1f}% sharpe={result['sharpe']:.2f} maxdd={result['max_dd']:.1f}% trades={result['trades']} avg_pos={result['avg_pos']:.2f}")
        except Exception as e:
            print(f"ERROR running {etf}: {e}")
            import traceback
            traceback.print_exc()

    out_dir = ROOT / "runtime" / "generalization"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "rule_based_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print("Rule-based backtest generalization summary")
    print(f"{'='*60}")
    print(f"{'ETF':<8} {'Return':>10} {'Sharpe':>8} {'MaxDD':>10} {'Trades':>8} {'AvgPos':>8}")
    print("-" * 60)
    for etf, r in results.items():
        print(f"{etf:<8} {r['return']:>10.1f}% {r['sharpe']:>8.2f} {r['max_dd']:>10.1f}% {r['trades']:>8d} {r['avg_pos']:>8.2f}")


if __name__ == "__main__":
    main()
