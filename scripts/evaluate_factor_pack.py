#!/usr/bin/env python3
"""Generic factor-pack evaluator for SOXL harness.

Reusable across any feature pack registered in BUILTIN_FEATURE_PACKS or
experiment extensions.  Computes time-series IC, long/short-flat backtests,
correlation budget vs CORE_FEATURE_COLUMNS, and writes cards + DB records.

Examples:
    python scripts/evaluate_factor_pack.py --pack cicc_price_volume
    python scripts/evaluate_factor_pack.py --pack transition_signals \
        --features TRANS_VIX_DIVERGENCE SIGNED_TE_REVERSION_10
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sqlite3
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.soxl_quant.data import (
    CORE_FEATURE_COLUMNS,
    BUILTIN_FEATURE_PACKS,
    _feature_pack_registry,
    engineer_feature_frame,
)
from src.soxl_quant.settings import CSV_PATH


@dataclass(frozen=True)
class ICResult:
    feature_name: str
    horizon: int
    n_days: int
    ic_mean: float
    ic_std: float
    ic_t: float
    ic_positive_ratio: float
    pearson_corr: float


@dataclass(frozen=True)
class BacktestResult:
    feature_name: str
    direction: str  # "long" or "short"
    threshold: float
    total_return: float
    sharpe: float
    max_drawdown: float
    profit_factor: float
    signal_accuracy: float
    num_signals: int
    trades: int
    equity: pd.Series


def _load_ohlcv_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(CSV_PATH, parse_dates=["date"]).set_index("date")
    soxl = pd.DataFrame(
        {
            "$close": df["soxl_close"],
            "$open": df["soxl_open"],
            "$high": df["soxl_high"],
            "$low": df["soxl_low"],
            "$volume": df["soxl_volume"],
            "$vix": df["vix"],
            "Mean($close, 5)": df["soxl_close"].rolling(5).mean(),
            "Mean($close, 21)": df["soxl_close"].rolling(21).mean(),
            "Mean($close, 63)": df["soxl_close"].rolling(63).mean(),
            "Mean($close, 180)": df["soxl_close"].rolling(180).mean(),
            "Std($close / Ref($close, 1) - 1, 10)": df["soxl_close"].pct_change().rolling(10).std(),
            "Std($close / Ref($close, 1) - 1, 21)": df["soxl_close"].pct_change().rolling(21).std(),
            "Std($close / Ref($close, 1) - 1, 63)": df["soxl_close"].pct_change().rolling(63).std(),
            "Mean($volume, 21)": df["soxl_volume"].rolling(21).mean(),
        }
    )
    soxx = pd.DataFrame(
        {
            "$close": df["soxx_close"],
            "$open": df["soxx_open"],
            "$high": df["soxx_high"],
            "$low": df["soxx_low"],
            "$volume": df["soxx_volume"],
        }
    )
    return soxl, soxx


def compute_time_series_ic(feature: pd.Series, returns: pd.Series, horizon: int) -> ICResult:
    forward_return = (1.0 + returns).rolling(horizon).apply(np.prod, raw=True).shift(-horizon) - 1.0
    aligned = pd.concat({"feature": feature, "forward_return": forward_return}, axis=1).dropna()
    signal_return = aligned["feature"] * aligned["forward_return"]
    n_days = int(len(aligned))
    ic_mean = float(signal_return.mean()) if n_days else 0.0
    ic_std = float(signal_return.std(ddof=1)) if n_days > 1 else 0.0
    ic_t = float(ic_mean / (ic_std + 1e-12) * np.sqrt(n_days)) if n_days > 1 else 0.0
    positive_ratio = float((signal_return > 0.0).mean()) if n_days else 0.0
    pearson_corr = float(aligned["feature"].corr(aligned["forward_return"])) if n_days > 1 else 0.0
    return ICResult(
        feature_name=str(feature.name or "<unnamed>"),
        horizon=horizon,
        n_days=n_days,
        ic_mean=ic_mean,
        ic_std=ic_std,
        ic_t=ic_t,
        ic_positive_ratio=positive_ratio,
        pearson_corr=pearson_corr,
    )


def run_long_flat_backtest(signal: pd.Series, returns: pd.Series, threshold: float) -> BacktestResult:
    raw_position = (signal > threshold).astype(float)
    position = raw_position.shift(1).fillna(0.0)
    daily_return = position * returns.fillna(0.0)
    equity = (1.0 + daily_return).cumprod()
    total_return = float(equity.iloc[-1] - 1.0) if len(equity) else 0.0
    sharpe = float(daily_return.mean() / (daily_return.std(ddof=1) + 1e-12) * np.sqrt(252)) if len(daily_return) > 1 else 0.0
    drawdown = equity / equity.cummax() - 1.0
    max_drawdown = float(drawdown.min()) if len(drawdown) else 0.0
    gains = daily_return[daily_return > 0.0].sum()
    losses = -daily_return[daily_return < 0.0].sum()
    profit_factor = float(gains / (losses + 1e-12))
    active = raw_position > 0.0
    next_return = returns.shift(-1)
    signal_accuracy = float((next_return[active] > 0.0).mean()) if active.any() else 0.0
    trades = int(position.diff().abs().fillna(0.0).sum())
    return BacktestResult(
        feature_name=str(signal.name or "<unnamed>"),
        direction="long",
        threshold=threshold,
        total_return=total_return,
        sharpe=sharpe,
        max_drawdown=max_drawdown,
        profit_factor=profit_factor,
        signal_accuracy=signal_accuracy,
        num_signals=int(active.sum()),
        trades=trades,
        equity=equity,
    )


def run_short_flat_backtest(signal: pd.Series, returns: pd.Series, threshold: float) -> BacktestResult:
    raw_position = -(signal < threshold).astype(float)
    position = raw_position.shift(1).fillna(0.0)
    daily_return = position * returns.fillna(0.0)
    equity = (1.0 + daily_return).cumprod()
    total_return = float(equity.iloc[-1] - 1.0) if len(equity) else 0.0
    sharpe = float(daily_return.mean() / (daily_return.std(ddof=1) + 1e-12) * np.sqrt(252)) if len(daily_return) > 1 else 0.0
    drawdown = equity / equity.cummax() - 1.0
    max_drawdown = float(drawdown.min()) if len(drawdown) else 0.0
    gains = daily_return[daily_return > 0.0].sum()
    losses = -daily_return[daily_return < 0.0].sum()
    profit_factor = float(gains / (losses + 1e-12))
    active = raw_position < 0.0
    next_return = returns.shift(-1)
    signal_accuracy = float((next_return[active] < 0.0).mean()) if active.any() else 0.0
    trades = int(position.diff().abs().fillna(0.0).sum())
    return BacktestResult(
        feature_name=str(signal.name or "<unnamed>"),
        direction="short",
        threshold=threshold,
        total_return=total_return,
        sharpe=sharpe,
        max_drawdown=max_drawdown,
        profit_factor=profit_factor,
        signal_accuracy=signal_accuracy,
        num_signals=int(active.sum()),
        trades=trades,
        equity=equity,
    )


def _result_rows(results: list[ICResult]) -> list[dict]:
    return [
        {
            "feature_name": r.feature_name,
            "horizon": r.horizon,
            "n_days": r.n_days,
            "ic_mean": r.ic_mean,
            "ic_std": r.ic_std,
            "ic_t": r.ic_t,
            "ic_positive_ratio": r.ic_positive_ratio,
            "pearson_corr": r.pearson_corr,
        }
        for r in results
    ]


def _backtest_rows(results: list[BacktestResult]) -> list[dict]:
    return [
        {
            "feature_name": r.feature_name,
            "direction": r.direction,
            "threshold": r.threshold,
            "total_return": r.total_return,
            "sharpe": r.sharpe,
            "max_drawdown": r.max_drawdown,
            "profit_factor": r.profit_factor,
            "signal_accuracy": r.signal_accuracy,
            "num_signals": r.num_signals,
            "trades": r.trades,
        }
        for r in results
    ]


def _write_card(
    output_dir: Path,
    pack_name: str,
    feature_name: str,
    best_ic: ICResult,
    best_backtest: BacktestResult,
    max_correlations: dict[str, float],
) -> None:
    card_dir = output_dir / "card" / pack_name
    card_dir.mkdir(parents=True, exist_ok=True)
    max_corr = max(abs(v) for v in max_correlations.values())
    best_feature_corr = abs(max_correlations[feature_name])
    passed_gate = (
        abs(best_ic.ic_mean) > 0.03
        and best_ic.ic_positive_ratio > 0.55
        and best_backtest.profit_factor > 1.2
        and best_backtest.signal_accuracy > 0.55
        and best_feature_corr <= 0.80
    )
    status = "✅ Strong" if passed_gate else "🟡 Candidate"
    card = f"""# {feature_name}

**方向**: factor
**类别**: {pack_name}
**状态**: {status}

## 结果

| 指标 | 数值 |
|------|------|
| Best IC Horizon | {best_ic.horizon}d |
| IC Mean | {best_ic.ic_mean:+.6f} |
| IC Positive Ratio | {best_ic.ic_positive_ratio:.3f} |
| IC t-stat | {best_ic.ic_t:.3f} |
| Best Backtest Direction | {best_backtest.direction} |
| Threshold | {best_backtest.threshold:.2f} |
| Sharpe | {best_backtest.sharpe:+.3f} |
| Profit Factor | {best_backtest.profit_factor:.3f} |
| Signal Accuracy | {best_backtest.signal_accuracy:.3f} |
| Max Core Correlation | {best_feature_corr:.3f} |

## 结论

{'通过全部门槛，建议作为 Candidate 保留。' if passed_gate else '未通过全部门槛，需进一步 regime-gated 测试或放弃。'}
"""
    (card_dir / f"card_{feature_name}.md").write_text(card, encoding="utf-8")


def _write_to_database(
    db_path: Path,
    pack_name: str,
    ic_results: list[ICResult],
    backtest_results: list[BacktestResult],
) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    for r in ic_results:
        cursor.execute(
            """INSERT OR REPLACE INTO feature_ic_tests
            (feature_name, horizon_days, test_start, test_end, n_days, ic_mean, ic_std, ic_t, ic_positive_ratio)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (r.feature_name, r.horizon, "", "", r.n_days, r.ic_mean, r.ic_std, r.ic_t, r.ic_positive_ratio),
        )

    for r in backtest_results:
        cursor.execute(
            """INSERT OR REPLACE INTO feature_backtests
            (feature_name, horizon_days, threshold_long, threshold_short, hold_days, signal_accuracy, profit_factor, avg_return, max_drawdown, sharpe, win_rate, num_signals)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                r.feature_name, 0,
                r.threshold if r.direction == "long" else 0.0,
                r.threshold if r.direction == "short" else 0.0,
                0, r.signal_accuracy, r.profit_factor, 0.0, r.max_drawdown, r.sharpe, 0.0, r.num_signals,
            ),
        )

    conn.commit()
    conn.close()


def evaluate_pack(
    pack_name: str,
    feature_names: tuple[str, ...] | None,
    output_dir: Path,
    horizons: tuple[int, ...],
    thresholds: tuple[float, ...],
    write: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    registry = _feature_pack_registry()
    payload = registry.get(pack_name)
    if payload is None:
        raise KeyError(f"unknown feature pack '{pack_name}'; available: {list(registry.keys())}")

    pack_columns = list(payload["columns"])
    if feature_names is None:
        feature_names = tuple(pack_columns)
    else:
        invalid = set(feature_names) - set(pack_columns)
        if invalid:
            raise ValueError(f"features not in pack '{pack_name}': {invalid}")

    soxl, soxx = _load_ohlcv_frames()
    feature_df, _ = engineer_feature_frame(soxl, soxx, feature_packs=(pack_name,))
    flat = feature_df.droplevel("instrument")
    returns = flat["return"]

    ic_results: list[ICResult] = []
    for fname in feature_names:
        for h in horizons:
            ic_results.append(compute_time_series_ic(flat[fname], returns, h))
    ic_df = pd.DataFrame(_result_rows(ic_results))

    backtest_results: list[BacktestResult] = []
    for fname in feature_names:
        feature = flat[fname]
        for thresh in thresholds:
            long_bt = run_long_flat_backtest(feature, returns, thresh)
            short_bt = run_short_flat_backtest(feature, returns, thresh)
            backtest_results.append(long_bt)
            backtest_results.append(short_bt)
    backtest_df = pd.DataFrame(_backtest_rows(backtest_results))

    correlations = []
    for fname in feature_names:
        corr = flat[[*CORE_FEATURE_COLUMNS, fname]].corr()[fname].drop(fname)
        correlations.append(
            {
                "feature_name": fname,
                "max_abs_core_correlation": float(corr.abs().max()),
                "max_abs_core_correlation_feature": corr.abs().idxmax(),
            }
        )
    corr_df = pd.DataFrame(correlations)

    if write:
        output_dir.mkdir(parents=True, exist_ok=True)
        card_dir = output_dir / "card" / pack_name
        card_dir.mkdir(parents=True, exist_ok=True)
        ic_df.to_csv(output_dir / f"{pack_name}_ic.csv", index=False)
        backtest_df.to_csv(card_dir / f"backtest_{pack_name}.csv", index=False)
        corr_df.to_csv(card_dir / f"correlation_{pack_name}.csv", index=False)

        for fname in feature_names:
            feat_ic = [r for r in ic_results if r.feature_name == fname]
            feat_bt = [r for r in backtest_results if r.feature_name == fname]
            best_ic = max(feat_ic, key=lambda r: abs(r.ic_mean))
            best_bt = max(feat_bt, key=lambda r: r.sharpe)
            _write_card(
                output_dir,
                pack_name=pack_name,
                feature_name=fname,
                best_ic=best_ic,
                best_backtest=best_bt,
                max_correlations={row["feature_name"]: row["max_abs_core_correlation"] for row in correlations},
            )

        _write_to_database(
            ROOT / "data" / "soxl_factor_library.db",
            pack_name=pack_name,
            ic_results=ic_results,
            backtest_results=backtest_results,
        )

    return ic_df, backtest_df, corr_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", required=True, help="feature pack name (e.g. cicc_price_volume)")
    parser.add_argument("--features", nargs="+", default=None, help="subset of features to evaluate (default: all in pack)")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runtime" / "factor")
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 5, 10, 20])
    parser.add_argument("--thresholds", nargs="+", type=float, default=[-1.0, -0.5, 0.0, 0.5, 1.0, 1.5])
    parser.add_argument("--no-write", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ic_df, backtest_df, corr_df = evaluate_pack(
        pack_name=args.pack,
        feature_names=tuple(args.features) if args.features else None,
        output_dir=args.output_dir,
        horizons=tuple(args.horizons),
        thresholds=tuple(args.thresholds),
        write=not args.no_write,
    )

    print(f"[{args.pack}] top IC (by abs ic_mean)")
    print(ic_df.reindex(ic_df["ic_mean"].abs().sort_values(ascending=False).index).head(10).to_string(index=False))
    print(f"\n[{args.pack}] top backtest (by sharpe)")
    print(backtest_df.head(10).to_string(index=False))
    print(f"\n[{args.pack}] correlation budget")
    print(corr_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
