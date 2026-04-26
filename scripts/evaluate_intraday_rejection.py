#!/usr/bin/env python3
from __future__ import annotations

"""Evaluate intraday rejection wick as an independent SOXL factor."""

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_signed_te_reversion import (
    HORIZONS,
    _backtest_rows,
    _load_ohlcv_frames,
    _result_rows,
    compute_time_series_ic,
    run_long_flat_backtest,
)
from src.soxl_quant.data import CORE_FEATURE_COLUMNS, engineer_feature_frame

FACTOR_COLUMNS = ("INTRADAY_REJECTION_EVENT",)
THRESHOLDS = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0)


def _write_card(
    output_dir: Path,
    ic_df: pd.DataFrame,
    backtest_df: pd.DataFrame,
    corr_df: pd.DataFrame,
) -> None:
    card_dir = output_dir / "card" / "intraday_rejection_event"
    card_dir.mkdir(parents=True, exist_ok=True)
    best_ic = ic_df.iloc[0]
    best_backtest = backtest_df.iloc[0]
    best_feature_corr = float(corr_df.loc[
        corr_df["feature_name"] == best_backtest["feature_name"],
        "max_abs_core_correlation",
    ].iloc[0])
    passed_gate = (
        float(best_ic["ic_mean"]) > 0.03
        and float(best_ic["ic_positive_ratio"]) > 0.55
        and float(best_backtest["profit_factor"]) > 1.2
        and float(best_backtest["signal_accuracy"]) > 0.55
        and best_feature_corr <= 0.80
    )
    card = f"""# INTRADAY_REJECTION_EVENT

**方向**: factor
**类别**: transition / intraday-reversal
**状态**: {'✅ Strong' if passed_gate else '🟡 Candidate / blocked'}

## 假设

币圈单资产中常见的 liquidation wick / panic rejection 可以映射到 SOXL：
大振幅日出现长下影、收盘靠近高位且放量，代表盘中卖压被吸收，后续可能反弹。

## 方法/配置

公式: `lower_wick_ratio * close_location * positive_intraday_range_zscore * (1 + positive_relative_volume_zscore)`

## 结果

| 指标 | 数值 |
|------|------|
| Best IC Horizon | {int(best_ic['horizon'])}d |
| IC Mean | {float(best_ic['ic_mean']):+.6f} |
| IC Positive Ratio | {float(best_ic['ic_positive_ratio']):.3f} |
| IC t-stat | {float(best_ic['ic_t']):.3f} |
| Threshold | {float(best_backtest['threshold']):.2f} |
| Sharpe | {float(best_backtest['sharpe']):+.3f} |
| Profit Factor | {float(best_backtest['profit_factor']):.3f} |
| Signal Accuracy | {float(best_backtest['signal_accuracy']):.3f} |
| Num Signals | {int(best_backtest['num_signals'])} |
| Max Core Correlation | {best_feature_corr:.3f} |

## 结论

该因子先作为 standalone candidate 留存。只有在 IC、方向准确率、profit factor
和相关性预算同时通过后，才允许进入 RL feature promotion 讨论。

## 参考

- Crypto single-asset intraday reversal / liquidation wick analogue
- 与 `ACAD_INTRADAY_REVERSAL_21` 区分：本因子是事件型 wick rejection，不是 21d 平滑均值。
"""
    (card_dir / "card.md").write_text(card, encoding="utf-8")


def evaluate(output_dir: Path, write: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    soxl, soxx = _load_ohlcv_frames()
    feature_df, _ = engineer_feature_frame(soxl, soxx, feature_packs=("transition_signals",))
    flat = feature_df.droplevel("instrument")
    returns = flat["return"]

    ic_results = [
        compute_time_series_ic(flat[feature_name], returns, horizon)
        for feature_name in FACTOR_COLUMNS
        for horizon in HORIZONS
    ]
    ic_df = pd.DataFrame(_result_rows(ic_results)).sort_values(
        ["ic_mean", "ic_positive_ratio"], ascending=[False, False]
    )

    backtest_results = [
        run_long_flat_backtest(flat[feature_name], returns, threshold)
        for feature_name in FACTOR_COLUMNS
        for threshold in THRESHOLDS
    ]
    backtest_df = pd.DataFrame(_backtest_rows(backtest_results)).sort_values(
        ["sharpe", "profit_factor"], ascending=[False, False]
    )

    correlations = []
    for feature_name in FACTOR_COLUMNS:
        corr = flat[[*CORE_FEATURE_COLUMNS, feature_name]].corr()[feature_name].drop(feature_name)
        correlations.append(
            {
                "feature_name": feature_name,
                "max_abs_core_correlation": float(corr.abs().max()),
                "max_abs_core_correlation_feature": corr.abs().idxmax(),
            }
        )
    corr_df = pd.DataFrame(correlations).sort_values("max_abs_core_correlation")

    if write:
        output_dir.mkdir(parents=True, exist_ok=True)
        card_dir = output_dir / "card" / "intraday_rejection_event"
        card_dir.mkdir(parents=True, exist_ok=True)
        ic_df.to_csv(output_dir / "intraday_rejection_event_ic.csv", index=False)
        backtest_df.to_csv(card_dir / "backtest_INTRADAY_REJECTION_EVENT.csv", index=False)
        corr_df.to_csv(card_dir / "correlation_INTRADAY_REJECTION_EVENT.csv", index=False)
        _write_card(output_dir, ic_df, backtest_df, corr_df)
    return ic_df, backtest_df, corr_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runtime" / "factor")
    parser.add_argument("--no-write", action="store_true", help="compute only; do not write runtime artifacts")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ic_df, backtest_df, corr_df = evaluate(args.output_dir, write=not args.no_write)
    print("[intraday_rejection_event] top IC")
    print(ic_df.head(5).to_string(index=False))
    print("[intraday_rejection_event] top backtest")
    print(backtest_df.head(5).to_string(index=False))
    print("[intraday_rejection_event] correlation budget")
    print(corr_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
