#!/usr/bin/env python3
from __future__ import annotations

"""Evaluate SIGNED_TE_REVERSION_10 under stress-release gates."""

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

FACTOR_NAME = "SIGNED_TE_REVERSION_10_STRESS_GATE"
TE_THRESHOLDS = (0.0, 0.5, 1.0)
VIX_DIVERGENCE_THRESHOLDS = (0.02, 0.05, 0.10)
REGIME_RISK_THRESHOLDS = (0.55, 0.60, 0.65)
REGIME_RISK_DROP_THRESHOLDS = (0.03, 0.05, 0.08)


def build_gated_signal(
    signed_te: pd.Series,
    vix_divergence: pd.Series,
    regime_risk: pd.Series,
    te_threshold: float,
    vix_divergence_threshold: float,
    regime_risk_threshold: float,
    regime_risk_drop_5_threshold: float,
) -> pd.Series:
    """Keep signed TE only when stress is releasing.

    Gate has two legs:
    - VIX-divergence leg: sentiment has started fading while price is still low.
    - Regime-risk relief leg: recent regime risk was elevated and has dropped.
    """
    recent_stress = regime_risk.rolling(10).max() >= regime_risk_threshold
    risk_drop_5 = regime_risk.shift(5) - regime_risk
    vix_release = vix_divergence >= vix_divergence_threshold
    risk_release = recent_stress & (risk_drop_5 >= regime_risk_drop_5_threshold)
    gate = (vix_release | risk_release) & (signed_te > te_threshold)
    return signed_te.where(gate, 0.0).fillna(0.0).rename(FACTOR_NAME)


def _write_card(
    output_dir: Path,
    ic_df: pd.DataFrame,
    backtest_df: pd.DataFrame,
    corr_df: pd.DataFrame,
) -> None:
    card_dir = output_dir / "card" / "signed_te_stress_gate"
    card_dir.mkdir(parents=True, exist_ok=True)
    best_ic = ic_df.iloc[0]
    best_backtest = backtest_df.iloc[0]
    best_feature_corr = float(corr_df["max_abs_core_correlation"].iloc[0])
    passed_gate = (
        float(best_ic["ic_mean"]) > 0.03
        and float(best_ic["ic_positive_ratio"]) > 0.55
        and float(best_backtest["profit_factor"]) > 1.2
        and float(best_backtest["signal_accuracy"]) > 0.55
        and best_feature_corr <= 0.80
    )
    card = f"""# SIGNED_TE_REVERSION_10_STRESS_GATE

**方向**: factor
**类别**: transition / gated mean-reversion
**状态**: {'✅ Strong' if passed_gate else '🟡 Candidate / blocked'}

## 假设

`SIGNED_TE_REVERSION_10` 裸信号有 standalone 正收益但方向不稳。
只在 stress-release gate 下启用，可能过滤掉非反弹阶段的 tracking-error noise。

## 方法/配置

基础信号: `SIGNED_TE_REVERSION_10`

Gate: `(TRANS_VIX_DIVERGENCE >= threshold) OR (recent regime_risk elevated AND regime_risk drops over 5d)`

## 结果

| 指标 | 数值 |
|------|------|
| Best IC Horizon | {int(best_ic['horizon'])}d |
| IC Mean | {float(best_ic['ic_mean']):+.6f} |
| IC Positive Ratio | {float(best_ic['ic_positive_ratio']):.3f} |
| IC t-stat | {float(best_ic['ic_t']):.3f} |
| TE Threshold | {float(best_backtest['te_threshold']):.2f} |
| VIX Divergence Threshold | {float(best_backtest['vix_divergence_threshold']):.2f} |
| Regime Risk Threshold | {float(best_backtest['regime_risk_threshold']):.2f} |
| Regime Risk Drop 5d | {float(best_backtest['regime_risk_drop_5_threshold']):.2f} |
| Sharpe | {float(best_backtest['sharpe']):+.3f} |
| Profit Factor | {float(best_backtest['profit_factor']):.3f} |
| Signal Accuracy | {float(best_backtest['signal_accuracy']):.3f} |
| Num Signals | {int(best_backtest['num_signals'])} |
| Max Core Correlation | {best_feature_corr:.3f} |

## 结论

该 gated signal 仅作为 standalone candidate 留存。若 gate 后仍不能同时通过
IC、profit factor、direction accuracy 和 correlation budget，不进入 RL feature promotion。

## 参考

- Crypto futures basis / funding-rate dislocation analogue
- Stress-release family: compare with `TRANS_VIX_DIVERGENCE`
"""
    (card_dir / "card.md").write_text(card, encoding="utf-8")


def evaluate(output_dir: Path, write: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    soxl, soxx = _load_ohlcv_frames()
    feature_df, _ = engineer_feature_frame(soxl, soxx, feature_packs=("transition_signals",))
    flat = feature_df.droplevel("instrument")
    returns = flat["return"]

    records: list[dict[str, object]] = []
    signal_map: dict[str, pd.Series] = {}
    for te_threshold in TE_THRESHOLDS:
        for vix_threshold in VIX_DIVERGENCE_THRESHOLDS:
            for risk_threshold in REGIME_RISK_THRESHOLDS:
                for drop_threshold in REGIME_RISK_DROP_THRESHOLDS:
                    signal = build_gated_signal(
                        signed_te=flat["SIGNED_TE_REVERSION_10"],
                        vix_divergence=flat["TRANS_VIX_DIVERGENCE"],
                        regime_risk=flat["regime_risk"],
                        te_threshold=te_threshold,
                        vix_divergence_threshold=vix_threshold,
                        regime_risk_threshold=risk_threshold,
                        regime_risk_drop_5_threshold=drop_threshold,
                    )
                    key = (
                        f"te{te_threshold:.2f}_vix{vix_threshold:.2f}_"
                        f"risk{risk_threshold:.2f}_drop{drop_threshold:.2f}"
                    )
                    signal_map[key] = signal
                    result = run_long_flat_backtest(signal, returns, threshold=0.0)
                    row = _backtest_rows([result])[0]
                    row.update(
                        {
                            "variant": key,
                            "te_threshold": te_threshold,
                            "vix_divergence_threshold": vix_threshold,
                            "regime_risk_threshold": risk_threshold,
                            "regime_risk_drop_5_threshold": drop_threshold,
                        }
                    )
                    records.append(row)

    backtest_df = pd.DataFrame(records).sort_values(
        ["sharpe", "profit_factor"], ascending=[False, False]
    )

    best_variant = str(backtest_df.iloc[0]["variant"])
    best_signal = signal_map[best_variant]
    ic_results = [compute_time_series_ic(best_signal, returns, horizon) for horizon in HORIZONS]
    ic_df = pd.DataFrame(_result_rows(ic_results)).sort_values(
        ["ic_mean", "ic_positive_ratio"], ascending=[False, False]
    )

    corr = flat[[*CORE_FEATURE_COLUMNS]].corrwith(best_signal).dropna()
    corr_df = pd.DataFrame(
        [
            {
                "feature_name": FACTOR_NAME,
                "variant": best_variant,
                "max_abs_core_correlation": float(corr.abs().max()),
                "max_abs_core_correlation_feature": corr.abs().idxmax(),
            }
        ]
    )

    if write:
        output_dir.mkdir(parents=True, exist_ok=True)
        card_dir = output_dir / "card" / "signed_te_stress_gate"
        card_dir.mkdir(parents=True, exist_ok=True)
        ic_df.to_csv(output_dir / "signed_te_stress_gate_ic.csv", index=False)
        backtest_df.to_csv(card_dir / "backtest_SIGNED_TE_STRESS_GATE.csv", index=False)
        corr_df.to_csv(card_dir / "correlation_SIGNED_TE_STRESS_GATE.csv", index=False)
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
    print("[signed_te_stress_gate] top IC for best backtest variant")
    print(ic_df.head(5).to_string(index=False))
    print("[signed_te_stress_gate] top backtest")
    print(backtest_df.head(10).to_string(index=False))
    print("[signed_te_stress_gate] correlation budget")
    print(corr_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
