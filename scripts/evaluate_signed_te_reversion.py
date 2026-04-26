#!/usr/bin/env python3
from __future__ import annotations

"""Evaluate signed tracking-error reversion as an independent SOXL factor."""

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.soxl_quant.data import CORE_FEATURE_COLUMNS, engineer_feature_frame
from src.soxl_quant.settings import CSV_PATH

FACTOR_COLUMNS = (
    "SIGNED_TE_REVERSION_5",
    "SIGNED_TE_REVERSION_10",
    "SIGNED_TE_REVERSION_21",
)
HORIZONS = (1, 5, 10, 20)


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
    threshold: float
    total_return: float
    sharpe: float
    max_drawdown: float
    profit_factor: float
    signal_accuracy: float
    num_signals: int
    trades: int
    equity: pd.Series


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


def _result_rows(results: list[ICResult]) -> list[dict[str, object]]:
    return [
        {
            "feature_name": result.feature_name,
            "horizon": result.horizon,
            "n_days": result.n_days,
            "ic_mean": result.ic_mean,
            "ic_std": result.ic_std,
            "ic_t": result.ic_t,
            "ic_positive_ratio": result.ic_positive_ratio,
            "pearson_corr": result.pearson_corr,
        }
        for result in results
    ]


def _backtest_rows(results: list[BacktestResult]) -> list[dict[str, object]]:
    return [
        {
            "feature_name": result.feature_name,
            "threshold": result.threshold,
            "total_return": result.total_return,
            "sharpe": result.sharpe,
            "max_drawdown": result.max_drawdown,
            "profit_factor": result.profit_factor,
            "signal_accuracy": result.signal_accuracy,
            "num_signals": result.num_signals,
            "trades": result.trades,
        }
        for result in results
    ]


def _write_card(
    output_dir: Path,
    best_ic: ICResult,
    best_backtest: BacktestResult,
    max_correlations: dict[str, float],
) -> None:
    card_dir = output_dir / "card" / "signed_te_reversion"
    card_dir.mkdir(parents=True, exist_ok=True)
    max_corr = max(abs(value) for value in max_correlations.values())
    best_feature_corr = abs(max_correlations[best_backtest.feature_name])
    passed_gate = (
        best_ic.ic_mean > 0.03
        and best_ic.ic_positive_ratio > 0.55
        and best_backtest.profit_factor > 1.2
        and best_backtest.signal_accuracy > 0.55
        and best_feature_corr <= 0.80
    )
    card = f"""# SIGNED_TE_REVERSION

**方向**: factor
**类别**: transition / mean-reversion
**状态**: {'✅ Strong' if passed_gate else '🟡 Candidate / blocked'}

## 假设

SOXL 相对 `3 * SOXX` 的有符号 tracking error 类似币圈 futures basis / funding dislocation。
当 SOXL 显著跑输 3x underlying path 时，`SIGNED_TE_REVERSION_*` 转正，预期后续存在均值回归反弹。

## 方法/配置

公式: `-zscore(rolling_sum(SOXL_return - 3 * SOXX_return, window), 252d)`

测试窗口: 5d / 10d / 21d

## 结果

| 指标 | 数值 |
|------|------|
| Best IC Feature | {best_ic.feature_name} |
| Best IC Horizon | {best_ic.horizon}d |
| IC Mean | {best_ic.ic_mean:+.6f} |
| IC Positive Ratio | {best_ic.ic_positive_ratio:.3f} |
| IC t-stat | {best_ic.ic_t:.3f} |
| Best Backtest Feature | {best_backtest.feature_name} |
| Threshold | {best_backtest.threshold:.2f} |
| Sharpe | {best_backtest.sharpe:+.3f} |
| Profit Factor | {best_backtest.profit_factor:.3f} |
| Signal Accuracy | {best_backtest.signal_accuracy:.3f} |
| Best Feature Core Correlation | {best_feature_corr:.3f} |
| Max Core Correlation | {max_corr:.3f} |

## 结论

该因子 standalone backtest 有正收益，但 IC mean / positive ratio 未过门槛，方向稳定性不足。
`SIGNED_TE_REVERSION_5` 还超过 core correlation budget；本轮不加入 core RL feature schema。
后续若继续，应只围绕 `SIGNED_TE_REVERSION_10` 做 regime-gated ablation，而不是直接推广整组信号。

## 参考

- Crypto futures basis / funding-rate dislocation analogue
- LETF tracking-error mean reversion: compare with `TRANS_TE_ZSCORE`
"""
    (card_dir / "card.md").write_text(card, encoding="utf-8")


def evaluate(output_dir: Path, write: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    soxl, soxx = _load_ohlcv_frames()
    feature_df, _ = engineer_feature_frame(soxl, soxx, feature_packs=("transition_signals",))
    flat = feature_df.droplevel("instrument")
    returns = flat["return"]

    ic_results: list[ICResult] = []
    for feature_name in FACTOR_COLUMNS:
        for horizon in HORIZONS:
            ic_results.append(compute_time_series_ic(flat[feature_name], returns, horizon))
    ic_df = pd.DataFrame(_result_rows(ic_results)).sort_values(
        ["ic_mean", "ic_positive_ratio"], ascending=[False, False]
    )

    backtest_results: list[BacktestResult] = []
    for feature_name in FACTOR_COLUMNS:
        for threshold in (0.0, 0.5, 1.0, 1.5):
            backtest_results.append(run_long_flat_backtest(flat[feature_name], returns, threshold))
    backtest_df = pd.DataFrame(_backtest_rows(backtest_results)).sort_values(
        ["sharpe", "profit_factor"], ascending=[False, False]
    )

    correlations = []
    for feature_name in FACTOR_COLUMNS:
        corr = flat[[*CORE_FEATURE_COLUMNS, feature_name]].corr()[feature_name].drop(feature_name)
        max_core = corr.abs().max()
        max_core_name = corr.abs().idxmax()
        correlations.append(
            {
                "feature_name": feature_name,
                "max_abs_core_correlation": float(max_core),
                "max_abs_core_correlation_feature": max_core_name,
            }
        )
    corr_df = pd.DataFrame(correlations).sort_values("max_abs_core_correlation")

    if write:
        output_dir.mkdir(parents=True, exist_ok=True)
        card_dir = output_dir / "card" / "signed_te_reversion"
        card_dir.mkdir(parents=True, exist_ok=True)
        ic_df.to_csv(output_dir / "signed_te_reversion_ic.csv", index=False)
        backtest_df.to_csv(card_dir / "backtest_SIGNED_TE_REVERSION.csv", index=False)
        corr_df.to_csv(card_dir / "correlation_SIGNED_TE_REVERSION.csv", index=False)
        _write_card(
            output_dir,
            best_ic=ic_results[int(ic_df.index[0])],
            best_backtest=backtest_results[int(backtest_df.index[0])],
            max_correlations={
                row["feature_name"]: row["max_abs_core_correlation"]
                for row in correlations
            },
        )
    return ic_df, backtest_df, corr_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runtime" / "factor")
    parser.add_argument("--no-write", action="store_true", help="compute only; do not write runtime artifacts")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ic_df, backtest_df, corr_df = evaluate(args.output_dir, write=not args.no_write)
    print("[signed_te_reversion] top IC")
    print(ic_df.head(5).to_string(index=False))
    print("[signed_te_reversion] top backtest")
    print(backtest_df.head(5).to_string(index=False))
    print("[signed_te_reversion] correlation budget")
    print(corr_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
