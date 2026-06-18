from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import pandas as pd

from .reports import (
    _prepare,
    compare_models,
    compute_regime_metrics,
    compute_window_metrics,
    compute_window_regime_metrics,
)


QLIB_RISK_SOURCE = "qlib.contrib.evaluate.risk_analysis"


@dataclass(frozen=True)
class RegimeWindowEffectAnalysis:
    qlib_return_frame: pd.DataFrame
    qlib_risk_metrics: pd.DataFrame
    segment_metrics: pd.DataFrame
    window_regime_metrics: pd.DataFrame
    model_effect_deltas: pd.DataFrame
    summary: dict[str, object]


def build_qlib_return_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Build a qlib-style model return table from daily backtest paths."""

    prepared = _prepare(frame)
    rows: list[pd.DataFrame] = []
    for model, group in prepared.groupby("model", sort=True):
        ordered = group.sort_values("date").copy()
        ordered["strategy_return"] = ordered["nav"].astype(float).pct_change().fillna(0.0)
        if "benchmark_nav" in ordered.columns:
            ordered["benchmark_return"] = ordered["benchmark_nav"].astype(float).pct_change().fillna(0.0)
        else:
            ordered["benchmark_return"] = 0.0
        ordered["excess_return"] = ordered["strategy_return"] - ordered["benchmark_return"]
        keep = [
            "date",
            "model",
            "regime",
            "strategy_return",
            "benchmark_return",
            "excess_return",
            "nav",
        ]
        if "benchmark_nav" in ordered.columns:
            keep.append("benchmark_nav")
        if "position" in ordered.columns:
            keep.append("position")
        if "turnover" in ordered.columns:
            keep.append("turnover")
        rows.append(ordered.loc[:, keep])
    if not rows:
        return pd.DataFrame(
            columns=[
                "date",
                "model",
                "regime",
                "strategy_return",
                "benchmark_return",
                "excess_return",
                "nav",
            ]
        )
    return pd.concat(rows, ignore_index=True).sort_values(["model", "date"]).reset_index(drop=True)


def _clean_metric(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(result):
        return 0.0
    return result


def _fallback_risk_analysis(returns: pd.Series, *, annualization: int = 238) -> dict[str, float]:
    values = returns.astype(float).fillna(0.0)
    if values.empty:
        return {
            "mean": 0.0,
            "std": 0.0,
            "annualized_return": 0.0,
            "information_ratio": 0.0,
            "max_drawdown": 0.0,
        }
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    cumulative = (1.0 + values).cumprod()
    annualized = float(cumulative.iloc[-1] ** (annualization / max(len(values), 1)) - 1.0)
    drawdown = float((cumulative / cumulative.cummax() - 1.0).min())
    ir = mean / std * math.sqrt(float(annualization)) if std > 1e-12 else 0.0
    return {
        "mean": mean,
        "std": std,
        "annualized_return": annualized,
        "information_ratio": ir,
        "max_drawdown": drawdown,
    }


def _qlib_risk_analysis(returns: pd.Series, *, freq: str) -> tuple[dict[str, float], str]:
    values = returns.astype(float).fillna(0.0)
    if values.empty:
        return _fallback_risk_analysis(values), "fallback.empty"
    try:
        from qlib.contrib.evaluate import risk_analysis

        result = risk_analysis(values, freq=freq, mode="product")["risk"].to_dict()
        return {key: _clean_metric(value) for key, value in result.items()}, QLIB_RISK_SOURCE
    except Exception:
        return _fallback_risk_analysis(values), "fallback.internal"


def _window_filter(frame: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    return frame.loc[(frame["date"] >= start_ts) & (frame["date"] <= end_ts)].copy()


def _risk_rows(
    frame: pd.DataFrame,
    *,
    segment_type: str,
    segment: str,
    return_columns: Sequence[str],
    freq: str,
    extra_columns: Mapping[str, object] | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model, group in frame.groupby("model", sort=True):
        for return_kind in return_columns:
            metrics, source = _qlib_risk_analysis(group[return_kind], freq=freq)
            rows.append(
                {
                    "model": model,
                    "segment_type": segment_type,
                    "segment": segment,
                    "return_kind": return_kind,
                    "row_count": int(len(group)),
                    "metric_source": source,
                    **dict(extra_columns or {}),
                    **metrics,
                }
            )
    return rows


def compute_qlib_risk_metrics(
    frame: pd.DataFrame,
    *,
    windows: Mapping[str, tuple[str, str]] | None = None,
    return_columns: Sequence[str] = ("strategy_return", "excess_return"),
    freq: str = "day",
) -> pd.DataFrame:
    returns = build_qlib_return_frame(frame)
    rows: list[dict[str, object]] = []
    rows.extend(
        _risk_rows(
            returns,
            segment_type="all",
            segment="all",
            return_columns=return_columns,
            freq=freq,
        )
    )
    for regime, group in returns.groupby("regime", sort=True):
        rows.extend(
            _risk_rows(
                group,
                segment_type="regime",
                segment=str(regime),
                return_columns=return_columns,
                freq=freq,
            )
        )
    for window, (start, end) in (windows or {}).items():
        window_frame = _window_filter(returns, start, end)
        if window_frame.empty:
            continue
        rows.extend(
            _risk_rows(
                window_frame,
                segment_type="window",
                segment=str(window),
                return_columns=return_columns,
                freq=freq,
            )
        )
        for regime, group in window_frame.groupby("regime", sort=True):
            rows.extend(
                _risk_rows(
                    group,
                    segment_type="window_regime",
                    segment=f"{window}::{regime}",
                    return_columns=return_columns,
                    freq=freq,
                    extra_columns={"window": str(window), "regime": str(regime)},
                )
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["model", "segment_type", "segment", "return_kind"]).reset_index(drop=True)


def _segment_metrics(frame: pd.DataFrame, *, windows: Mapping[str, tuple[str, str]] | None) -> pd.DataFrame:
    all_metrics = compare_models(frame).copy()
    all_metrics["segment_type"] = "all"
    all_metrics["segment"] = "all"

    regime_metrics = compute_regime_metrics(frame).copy()
    regime_metrics["segment"] = regime_metrics["regime"]
    regime_metrics["segment_type"] = "regime"

    metric_frames = [all_metrics, regime_metrics]
    if windows:
        window_metrics = compute_window_metrics(frame, windows=windows).copy()
        window_metrics["segment"] = window_metrics["window"]
        window_metrics["segment_type"] = "window"
        metric_frames.append(window_metrics)
        window_regime_metrics = compute_window_regime_metrics(frame, windows=windows)
        window_regime_metrics["segment_type"] = "window_regime"
        metric_frames.append(window_regime_metrics)

    result = pd.concat(metric_frames, ignore_index=True, sort=False)
    desired = [
        "segment_type",
        "segment",
        "window",
        "regime",
        "model",
        "rank",
        "return_pct",
        "benchmark_return_pct",
        "relative_return_pct",
        "max_drawdown_pct",
        "sharpe",
        "avg_position",
        "turnover",
        "stability_score",
    ]
    for column in desired:
        if column not in result.columns:
            result[column] = pd.NA
    return result.loc[:, desired].sort_values(["segment_type", "segment", "model"]).reset_index(drop=True)


def _diagnose(row: pd.Series, *, baseline_model: str) -> str:
    if row["model"] == baseline_model:
        return "基准模型。"
    return_delta = float(row["relative_return_pct_delta"])
    score_delta = float(row["stability_score_delta"])
    drawdown_delta = float(row["max_drawdown_pct_delta"])
    if return_delta < -0.5 and score_delta < 0.0:
        return "弱于基准：分段相对收益和稳定性同时落后，需要检查信号、仓位或 regime 适配。"
    if return_delta > 0.5 and score_delta > 0.0:
        return "强于基准：分段相对收益和稳定性同时改善，可作为晋升候选。"
    if drawdown_delta > 2.0:
        return "风险更高：收益接近但回撤扩大，需要增加风险约束或降权。"
    if return_delta > 0.0:
        return "略强于基准：收益有改善，但仍需看稳定性和回撤。"
    if return_delta < 0.0:
        return "略弱于基准：收益不足，需确认是否有低相关性或组合价值。"
    return "接近基准：没有明显分段优势。"


def compare_segment_model_effects(
    frame: pd.DataFrame,
    *,
    windows: Mapping[str, tuple[str, str]] | None = None,
    baseline_model: str | None = None,
) -> pd.DataFrame:
    metrics = _segment_metrics(frame, windows=windows)
    if metrics.empty:
        return pd.DataFrame()
    baseline = baseline_model or str(compare_models(frame).iloc[0]["model"])
    rows: list[dict[str, object]] = []
    for (segment_type, segment), group in metrics.groupby(["segment_type", "segment"], sort=True):
        base_rows = group.loc[group["model"] == baseline]
        if base_rows.empty:
            continue
        base = base_rows.iloc[0]
        for _, row in group.iterrows():
            payload = row.to_dict()
            payload["baseline_model"] = baseline
            for column in [
                "return_pct",
                "benchmark_return_pct",
                "relative_return_pct",
                "max_drawdown_pct",
                "sharpe",
                "turnover",
                "stability_score",
            ]:
                payload[f"{column}_baseline"] = _clean_metric(base[column])
                payload[f"{column}_delta"] = _clean_metric(row[column]) - _clean_metric(base[column])
            payload["diagnosis"] = _diagnose(pd.Series(payload), baseline_model=baseline)
            rows.append(payload)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["segment_type", "segment", "relative_return_pct_delta", "stability_score_delta"],
        ascending=[True, True, False, False],
    ).reset_index(drop=True)


def analyze_model_effects_by_regime_and_window(
    frame: pd.DataFrame,
    *,
    windows: Mapping[str, tuple[str, str]] | None = None,
    baseline_model: str | None = None,
    freq: str = "day",
) -> RegimeWindowEffectAnalysis:
    returns = build_qlib_return_frame(frame)
    risk = compute_qlib_risk_metrics(frame, windows=windows, freq=freq)
    segment_metrics = _segment_metrics(frame, windows=windows)
    window_regime_metrics = compute_window_regime_metrics(frame, windows=windows or {})
    baseline = baseline_model or str(compare_models(frame).iloc[0]["model"])
    deltas = compare_segment_model_effects(frame, windows=windows, baseline_model=baseline)
    weakness_frame = deltas.loc[deltas["model"] != baseline].sort_values(
        ["relative_return_pct_delta", "stability_score_delta"],
        ascending=[True, True],
    )
    weakest_segments = [
        {
            "model": str(row["model"]),
            "segment_type": str(row["segment_type"]),
            "segment": str(row["segment"]),
            "relative_return_pct_delta": _clean_metric(row["relative_return_pct_delta"]),
            "stability_score_delta": _clean_metric(row["stability_score_delta"]),
            "diagnosis": str(row["diagnosis"]),
        }
        for _, row in weakness_frame.head(5).iterrows()
    ]
    summary = {
        "schema_version": "analysis.regime_window_effect.v1",
        "baseline_model": baseline,
        "model_count": int(returns["model"].nunique()) if not returns.empty else 0,
        "regime_count": int(returns["regime"].nunique()) if not returns.empty else 0,
        "window_count": int(len(windows or {})),
        "window_regime_count": int(window_regime_metrics["segment"].nunique()) if not window_regime_metrics.empty else 0,
        "segment_types": sorted(str(value) for value in segment_metrics["segment_type"].dropna().unique().tolist()),
        "qlib_metric_source": QLIB_RISK_SOURCE,
        "weakest_segments": weakest_segments,
    }
    return RegimeWindowEffectAnalysis(
        qlib_return_frame=returns,
        qlib_risk_metrics=risk,
        segment_metrics=segment_metrics,
        window_regime_metrics=window_regime_metrics,
        model_effect_deltas=deltas,
        summary=summary,
    )
