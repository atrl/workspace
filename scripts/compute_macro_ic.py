#!/usr/bin/env python3
"""
Compute Information Coefficient (IC) for macro signals vs SOXL forward returns.

Usage:
    python scripts/compute_macro_ic.py

Output:
    runtime/model/card/macro_ic_analysis/ic_results.json
    runtime/model/card/macro_ic_analysis/ic_summary.txt
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUT_DIR = PROJECT_ROOT / "runtime" / "model" / "card" / "macro_ic_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_soxl() -> pd.Series:
    """Load SOXL daily close prices, return as daily series."""
    df = pd.read_csv(DATA_DIR / "symbols" / "SOXL.csv", parse_dates=["date"])
    df = df.sort_values("date").set_index("date")
    return df["close"]


def load_macro_series(name: str, value_col: str) -> pd.Series:
    """Load a macro CSV and return daily series."""
    df = pd.read_csv(
        DATA_DIR / "external" / f"{name}.csv",
        parse_dates=["observation_date"],
    )
    df = df.sort_values("observation_date").set_index("observation_date")
    df.columns = [c.lower().strip() for c in df.columns]
    if value_col.lower() not in df.columns:
        # Fallback: use the only non-date column
        cols = [c for c in df.columns if c != "observation_date"]
        value_col = cols[0]
    return df[value_col.lower()].dropna()


def align_to_business_days(series: pd.Series, business_index: pd.DatetimeIndex) -> pd.Series:
    """Reindex macro series to SOXL trading days, forward-fill."""
    # Ensure daily frequency first
    series = series.resample("D").last()
    # Forward fill
    series = series.ffill()
    # Reindex to business days
    return series.reindex(business_index).ffill().bfill()


# ---------------------------------------------------------------------------
# Signal builders (mirror data.py)
# ---------------------------------------------------------------------------

def build_macro_signals(index: pd.DatetimeIndex) -> dict[str, pd.Series]:
    """Build all macro signals aligned to the given index."""
    credit_spread = align_to_business_days(
        load_macro_series("credit_spread_baa", "credit_spread"), index
    )
    yc_10y2y = align_to_business_days(
        load_macro_series("yield_curve_10y2y", "value"), index
    )
    yc_10y3m = align_to_business_days(
        load_macro_series("yield_curve_10y3m", "value"), index
    )
    dxy = align_to_business_days(
        load_macro_series("dxy_broad", "value"), index
    )
    oil = align_to_business_days(
        load_macro_series("oil_wti", "value"), index
    )

    signals = {}

    # Raw levels
    signals["MACRO_CREDIT_SPREAD"] = credit_spread
    signals["MACRO_YC_10Y2Y"] = yc_10y2y
    signals["MACRO_YC_10Y3M"] = yc_10y3m

    # Z-scores (252-day)
    signals["MACRO_CREDIT_SPREAD_ZSCORE_252"] = (
        (credit_spread - credit_spread.rolling(252).mean())
        / (credit_spread.rolling(252).std() + 1e-8)
    ).fillna(0.0)

    signals["MACRO_YC_10Y2Y_ZSCORE_252"] = (
        (yc_10y2y - yc_10y2y.rolling(252).mean())
        / (yc_10y2y.rolling(252).std() + 1e-8)
    ).fillna(0.0)

    signals["MACRO_YC_10Y3M_ZSCORE_252"] = (
        (yc_10y3m - yc_10y3m.rolling(252).mean())
        / (yc_10y3m.rolling(252).std() + 1e-8)
    ).fillna(0.0)

    signals["MACRO_DXY_ZSCORE_252"] = (
        (dxy - dxy.rolling(252).mean())
        / (dxy.rolling(252).std() + 1e-8)
    ).fillna(0.0)

    signals["MACRO_OIL_ZSCORE_252"] = (
        (oil - oil.rolling(252).mean())
        / (oil.rolling(252).std() + 1e-8)
    ).fillna(0.0)

    # Deltas (21-day)
    signals["MACRO_CREDIT_SPREAD_DELTA_21"] = (credit_spread - credit_spread.shift(21)).fillna(0.0)
    signals["MACRO_YC_10Y2Y_DELTA_21"] = (yc_10y2y - yc_10y2y.shift(21)).fillna(0.0)

    # Additional: rate of change (momentum)
    signals["MACRO_CREDIT_SPREAD_ROC_63"] = credit_spread.pct_change(63).fillna(0.0)
    signals["MACRO_YC_10Y2Y_ROC_63"] = yc_10y2y.pct_change(63).fillna(0.0)
    signals["MACRO_DXY_ROC_63"] = dxy.pct_change(63).fillna(0.0)
    signals["MACRO_OIL_ROC_63"] = oil.pct_change(63).fillna(0.0)

    return signals


# ---------------------------------------------------------------------------
# IC computation
# ---------------------------------------------------------------------------

def _spearman_ic(x, y):
    """Compute Spearman correlation between two aligned series."""
    if len(x) < 10:
        return np.nan
    return spearmanr(x, y)[0]


def compute_ic(signal: pd.Series, returns: pd.DataFrame) -> dict:
    """Compute IC statistics for a signal against multiple forward returns."""
    results = {}
    for col in returns.columns:
        # Align and drop NaN
        df = pd.DataFrame({"signal": signal, "ret": returns[col]}).dropna()
        if len(df) < 30:
            results[col] = {"ic_mean": np.nan, "ic_std": np.nan, "ic_ir": np.nan, "pos_rate": np.nan, "t_stat": np.nan, "n": len(df)}
            continue
        # Rolling IC (252-day) using custom apply
        rolling_ic = df["signal"].rolling(252).apply(
            lambda x: _spearman_ic(x, df["ret"].loc[x.index]), raw=False
        )
        ic_mean = rolling_ic.mean()
        ic_std = rolling_ic.std()
        ic_ir = ic_mean / (ic_std + 1e-8) if ic_std > 0 else np.nan
        pos_rate = (rolling_ic > 0).mean()
        t_stat = ic_mean / (ic_std / np.sqrt(len(df)) + 1e-8) if ic_std > 0 else np.nan
        results[col] = {
            "ic_mean": round(ic_mean, 4),
            "ic_std": round(ic_std, 4),
            "ic_ir": round(ic_ir, 4),
            "pos_rate": round(pos_rate, 4),
            "t_stat": round(t_stat, 4),
            "n": len(df),
        }
    return results


def compute_ic_by_regime(signal: pd.Series, returns: pd.DataFrame, regime: pd.Series) -> dict:
    """Compute IC separately for bull and bear regimes."""
    results = {}
    for reg_name, mask in [("bull", regime == 1), ("bear", regime == -1)]:
        reg_results = {}
        for col in returns.columns:
            df = pd.DataFrame({"signal": signal[mask], "ret": returns[col][mask]}).dropna()
            if len(df) < 30:
                reg_results[col] = {"ic_mean": np.nan, "n": len(df)}
                continue
            ic = spearmanr(df["signal"], df["ret"])[0]
            reg_results[col] = {"ic_mean": round(ic, 4), "n": len(df)}
        results[reg_name] = reg_results
    return results


def classify_regime(returns: pd.Series, window: int = 63) -> pd.Series:
    """Simple bull/bear regime: +1 if rolling return > 0, -1 otherwise."""
    rolling_ret = returns.rolling(window).mean()
    return np.where(rolling_ret > 0, 1, -1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading SOXL data...")
    soxl_close = load_soxl()
    soxl_ret = soxl_close.pct_change().dropna()

    print(f"SOXL data: {soxl_close.index[0].date()} to {soxl_close.index[-1].date()}")

    # Forward returns
    horizons = [1, 5, 10, 21]
    forward_returns = pd.DataFrame({
        f"ret_{h}d": soxl_ret.shift(-h) for h in horizons
    }, index=soxl_ret.index)

    print("Building macro signals...")
    signals = build_macro_signals(soxl_ret.index)

    print(f"Signals built: {list(signals.keys())}")

    # Regime classification
    regime = pd.Series(classify_regime(soxl_ret), index=soxl_ret.index)

    print("\nComputing IC...")
    all_results = {}
    for sig_name, sig_series in signals.items():
        print(f"  {sig_name}...")
        ic = compute_ic(sig_series, forward_returns)
        ic_regime = compute_ic_by_regime(sig_series, forward_returns, regime)
        all_results[sig_name] = {
            "overall": ic,
            "by_regime": ic_regime,
        }

    # Save JSON
    json_path = OUT_DIR / "ic_results.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved JSON: {json_path}")

    # Save human-readable summary
    summary_path = OUT_DIR / "ic_summary.txt"
    with open(summary_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("MACRO SIGNAL IC ANALYSIS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"SOXL period: {soxl_ret.index[0].date()} to {soxl_ret.index[-1].date()}\n")
        f.write(f"Total trading days: {len(soxl_ret)}\n\n")

        for sig_name in signals:
            f.write("-" * 80 + "\n")
            f.write(f"Signal: {sig_name}\n")
            f.write("-" * 80 + "\n")
            ic_data = all_results[sig_name]["overall"]
            for h_col, stats in ic_data.items():
                f.write(f"  {h_col}:\n")
                if np.isnan(stats["ic_mean"]):
                    f.write(f"    (insufficient data, n={stats['n']})\n")
                else:
                    f.write(f"    IC mean:     {stats['ic_mean']:+.4f}\n")
                    f.write(f"    IC std:      {stats['ic_std']:.4f}\n")
                    f.write(f"    IC IR:       {stats['ic_ir']:+.4f}\n")
                    f.write(f"    Pos rate:    {stats['pos_rate']:.2%}\n")
                    f.write(f"    t-stat:      {stats['t_stat']:+.4f}\n")
                    f.write(f"    n:           {stats['n']}\n")

            f.write("  By regime (bull / bear):\n")
            for reg_name in ["bull", "bear"]:
                f.write(f"    {reg_name}:\n")
                reg_data = all_results[sig_name]["by_regime"][reg_name]
                for h_col, stats in reg_data.items():
                    ic_val = stats["ic_mean"]
                    if np.isnan(ic_val):
                        f.write(f"      {h_col}: n/a (n={stats['n']})\n")
                    else:
                        f.write(f"      {h_col}: {ic_val:+.4f} (n={stats['n']})\n")
            f.write("\n")

    print(f"Saved summary: {summary_path}")

    # Print key highlights
    print("\n" + "=" * 80)
    print("KEY HIGHLIGHTS")
    print("=" * 80)
    for sig_name in signals:
        ic_1d = all_results[sig_name]["overall"].get("ret_1d", {}).get("ic_mean", np.nan)
        ic_21d = all_results[sig_name]["overall"].get("ret_21d", {}).get("ic_mean", np.nan)
        if not np.isnan(ic_1d):
            print(f"  {sig_name:40s}  IC(1d)={ic_1d:+.4f}  IC(21d)={ic_21d:+.4f}")


if __name__ == "__main__":
    main()
