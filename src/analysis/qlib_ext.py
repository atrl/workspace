from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import pandas as pd

from .reports import (
    _prepare,
    compare_models,
    compute_regime_metrics,
    compute_window_metrics,
    compute_window_regime_metrics,
)
from .windows import build_analysis_window_table


QLIB_RISK_SOURCE = "qlib.contrib.evaluate.risk_analysis"
SEGMENT_EFFECT_MATRIX_SCHEMA_VERSION = "analysis.segment_effect_matrix.v1"
MODEL_DIFFERENCE_EXPLANATION_SCHEMA_VERSION = "analysis.model_difference_explanation.v1"
MODEL_REGIME_WINDOW_DIAGNOSTIC_SCHEMA_VERSION = "analysis.model_regime_window_diagnostic.v1"
MODEL_REVIEW_BRIEFING_SCHEMA_VERSION = "analysis.model_review_briefing.v1"
ANALYSIS_OVERVIEW_SCHEMA_VERSION = "analysis.overview.v1"

MODEL_DIFFERENCE_EXPLANATION_COLUMNS = [
    "schema_version",
    "priority_rank",
    "segment_type",
    "segment",
    "window",
    "regime",
    "baseline_model",
    "winner_model",
    "loser_model",
    "winner_relative_return_pct",
    "loser_relative_return_pct",
    "relative_return_pct_spread",
    "winner_vs_baseline_delta",
    "winner_stability_model",
    "winner_stability_vs_baseline_delta",
    "highest_drawdown_model",
    "highest_drawdown_pct",
    "drawdown_pct_spread",
    "explanation_signal",
    "decision_hint",
    "priority_score",
    "review_note",
]

MODEL_REVIEW_BRIEFING_COLUMNS = [
    "schema_version",
    "priority_rank",
    "model",
    "baseline_model",
    "segment_type",
    "segment",
    "window",
    "regime",
    "review_signal",
    "decision_hint",
    "priority_score",
    "relative_return_pct_delta",
    "stability_score_delta",
    "max_drawdown_pct_delta",
    "turnover_delta",
    "diagnosis",
    "review_note",
]

MODEL_REGIME_WINDOW_DIAGNOSTIC_COLUMNS = [
    "schema_version",
    "model",
    "baseline_model",
    "evaluated_segment_count",
    "positive_segment_count",
    "negative_segment_count",
    "weakness_segment_count",
    "strength_segment_count",
    "risk_segment_count",
    "window_regime_negative_count",
    "stress_negative_count",
    "positive_segment_share",
    "mean_relative_return_pct_delta",
    "mean_stability_score_delta",
    "mean_max_drawdown_pct_delta",
    "consistency_score",
    "best_segment_type",
    "best_segment",
    "best_window",
    "best_regime",
    "best_relative_return_pct_delta",
    "best_stability_score_delta",
    "worst_segment_type",
    "worst_segment",
    "worst_window",
    "worst_regime",
    "worst_relative_return_pct_delta",
    "worst_stability_score_delta",
    "worst_max_drawdown_pct_delta",
    "diagnostic_signal",
    "review_note",
]


@dataclass(frozen=True)
class RegimeWindowEffectAnalysis:
    qlib_return_frame: pd.DataFrame
    qlib_risk_metrics: pd.DataFrame
    segment_metrics: pd.DataFrame
    segment_effect_matrix: pd.DataFrame
    window_regime_metrics: pd.DataFrame
    model_effect_deltas: pd.DataFrame
    model_regime_window_diagnostics: pd.DataFrame
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


def _metric_extreme(group: pd.DataFrame, *, column: str, highest: bool) -> pd.Series:
    ordered = group.assign(_metric_value=group[column].map(_clean_metric)).sort_values(
        ["_metric_value", "model"],
        ascending=[not highest, True],
    )
    return ordered.iloc[0]


def _matrix_review_note(row: dict[str, object]) -> str:
    segment = str(row["segment"])
    baseline = str(row["baseline_model"])
    best_return_model = str(row["best_relative_return_model"])
    worst_return_model = str(row["worst_relative_return_model"])
    best_delta = _clean_metric(row["best_relative_return_vs_baseline_delta"])
    best_stability_delta = _clean_metric(row["best_stability_vs_baseline_delta"])
    spread = _clean_metric(row["relative_return_pct_spread"])
    drawdown_spread = _clean_metric(row["drawdown_pct_spread"])

    if not baseline:
        return f"{segment} 缺少基准模型，当前只做模型间横向排序。"
    if best_return_model != baseline and best_delta > 0.5 and best_stability_delta >= 0.0:
        return f"{best_return_model} 在 {segment} 强于基准 {baseline}，可作为该 regime/window 的候选优势模型。"
    if best_return_model == baseline and spread > 0.5:
        return f"{baseline} 在 {segment} 领先；{worst_return_model} 明显弱于基准，需检查该分段适配。"
    if drawdown_spread > 2.0:
        return f"{segment} 模型回撤差异较大，应优先核查高回撤模型的仓位和风险暴露。"
    return f"{segment} 模型差异较小，需结合回撤、换手和组合互补性判断。"


def build_segment_effect_matrix(
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
        if group.empty:
            continue
        best_return = _metric_extreme(group, column="relative_return_pct", highest=True)
        worst_return = _metric_extreme(group, column="relative_return_pct", highest=False)
        best_stability = _metric_extreme(group, column="stability_score", highest=True)
        worst_stability = _metric_extreme(group, column="stability_score", highest=False)
        highest_drawdown = _metric_extreme(group, column="max_drawdown_pct", highest=True)
        lowest_drawdown = _metric_extreme(group, column="max_drawdown_pct", highest=False)
        highest_turnover = _metric_extreme(group, column="turnover", highest=True)
        baseline_rows = group.loc[group["model"] == baseline]
        baseline_row = baseline_rows.iloc[0] if not baseline_rows.empty else None
        baseline_return = _clean_metric(baseline_row["relative_return_pct"]) if baseline_row is not None else 0.0
        baseline_stability = _clean_metric(baseline_row["stability_score"]) if baseline_row is not None else 0.0

        payload = {
            "schema_version": SEGMENT_EFFECT_MATRIX_SCHEMA_VERSION,
            "segment_type": str(segment_type),
            "segment": str(segment),
            "window": str(group["window"].dropna().iloc[0]) if "window" in group.columns and group["window"].notna().any() else "",
            "regime": str(group["regime"].dropna().iloc[0]) if "regime" in group.columns and group["regime"].notna().any() else "",
            "baseline_model": baseline if baseline_row is not None else "",
            "model_count": int(group["model"].nunique()),
            "best_relative_return_model": str(best_return["model"]),
            "best_relative_return_pct": _clean_metric(best_return["relative_return_pct"]),
            "worst_relative_return_model": str(worst_return["model"]),
            "worst_relative_return_pct": _clean_metric(worst_return["relative_return_pct"]),
            "relative_return_pct_spread": _clean_metric(best_return["relative_return_pct"])
            - _clean_metric(worst_return["relative_return_pct"]),
            "baseline_relative_return_pct": baseline_return,
            "best_relative_return_vs_baseline_delta": _clean_metric(best_return["relative_return_pct"]) - baseline_return,
            "best_stability_model": str(best_stability["model"]),
            "best_stability_score": _clean_metric(best_stability["stability_score"]),
            "worst_stability_model": str(worst_stability["model"]),
            "worst_stability_score": _clean_metric(worst_stability["stability_score"]),
            "stability_score_spread": _clean_metric(best_stability["stability_score"])
            - _clean_metric(worst_stability["stability_score"]),
            "baseline_stability_score": baseline_stability,
            "best_stability_vs_baseline_delta": _clean_metric(best_stability["stability_score"]) - baseline_stability,
            "highest_drawdown_model": str(highest_drawdown["model"]),
            "highest_drawdown_pct": _clean_metric(highest_drawdown["max_drawdown_pct"]),
            "lowest_drawdown_model": str(lowest_drawdown["model"]),
            "lowest_drawdown_pct": _clean_metric(lowest_drawdown["max_drawdown_pct"]),
            "drawdown_pct_spread": _clean_metric(highest_drawdown["max_drawdown_pct"])
            - _clean_metric(lowest_drawdown["max_drawdown_pct"]),
            "highest_turnover_model": str(highest_turnover["model"]),
            "highest_turnover": _clean_metric(highest_turnover["turnover"]),
            "average_turnover": float(group["turnover"].map(_clean_metric).mean()),
        }
        payload["review_note"] = _matrix_review_note(payload)
        rows.append(payload)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["segment_type", "segment"]).reset_index(drop=True)


def _difference_signal(row: pd.Series) -> tuple[str, str, float, str]:
    segment = _clean_string(row.get("segment", ""))
    baseline = _clean_string(row.get("baseline_model", ""))
    winner = _clean_string(row.get("best_relative_return_model", ""))
    loser = _clean_string(row.get("worst_relative_return_model", ""))
    spread = _clean_metric(row.get("relative_return_pct_spread"))
    winner_delta = _clean_metric(row.get("best_relative_return_vs_baseline_delta"))
    winner_stability_delta = _clean_metric(row.get("best_stability_vs_baseline_delta"))
    drawdown_spread = _clean_metric(row.get("drawdown_pct_spread"))

    priority_score = max(0.0, spread) * 2.0 + max(0.0, drawdown_spread) * 0.5 + max(0.0, abs(winner_delta))
    if baseline and winner != baseline and winner_delta >= 0.5 and winner_stability_delta >= 0.0:
        return (
            "segment_winner",
            "promote_segment_candidate",
            priority_score,
            f"中文 review：{winner} 在 {segment} 强于基准 {baseline}，可作为该 regime/window 的分段候选；仍需复核回撤、换手和组合互补性。",
        )
    if baseline and winner == baseline and spread > 0.5:
        return (
            "baseline_defends",
            "block_or_rework_loser",
            priority_score,
            f"中文 review：基准 {baseline} 在 {segment} 领先，{loser} 是当前 loser；晋升前应重做该分段信号、仓位或风险约束。",
        )
    if drawdown_spread > 2.0:
        return (
            "risk_dispersion",
            "risk_review",
            priority_score,
            f"中文 review：{segment} 的模型回撤差异较大，需优先检查 {row.get('highest_drawdown_model', '')} 的压力期暴露。",
        )
    return (
        "small_gap",
        "monitor",
        priority_score,
        f"中文 review：{segment} 的模型差异较小，暂不单独晋升；继续结合成本、换手和稳定性观察。",
    )


def build_model_difference_explanations(
    analysis: RegimeWindowEffectAnalysis | pd.DataFrame,
    *,
    max_rows: int | None = None,
) -> pd.DataFrame:
    """Build a segment-level Chinese explanation table for model differences."""

    matrix = analysis.segment_effect_matrix if isinstance(analysis, RegimeWindowEffectAnalysis) else analysis
    if matrix.empty:
        return pd.DataFrame(columns=MODEL_DIFFERENCE_EXPLANATION_COLUMNS)

    rows: list[dict[str, object]] = []
    for _, row in matrix.iterrows():
        signal, decision_hint, priority_score, review_note = _difference_signal(row)
        rows.append(
            {
                "schema_version": MODEL_DIFFERENCE_EXPLANATION_SCHEMA_VERSION,
                "priority_rank": 0,
                "segment_type": _clean_string(row.get("segment_type", "")),
                "segment": _clean_string(row.get("segment", "")),
                "window": _clean_string(row.get("window", "")),
                "regime": _clean_string(row.get("regime", "")),
                "baseline_model": _clean_string(row.get("baseline_model", "")),
                "winner_model": _clean_string(row.get("best_relative_return_model", "")),
                "loser_model": _clean_string(row.get("worst_relative_return_model", "")),
                "winner_relative_return_pct": _clean_metric(row.get("best_relative_return_pct")),
                "loser_relative_return_pct": _clean_metric(row.get("worst_relative_return_pct")),
                "relative_return_pct_spread": _clean_metric(row.get("relative_return_pct_spread")),
                "winner_vs_baseline_delta": _clean_metric(row.get("best_relative_return_vs_baseline_delta")),
                "winner_stability_model": _clean_string(row.get("best_stability_model", "")),
                "winner_stability_vs_baseline_delta": _clean_metric(row.get("best_stability_vs_baseline_delta")),
                "highest_drawdown_model": _clean_string(row.get("highest_drawdown_model", "")),
                "highest_drawdown_pct": _clean_metric(row.get("highest_drawdown_pct")),
                "drawdown_pct_spread": _clean_metric(row.get("drawdown_pct_spread")),
                "explanation_signal": signal,
                "decision_hint": decision_hint,
                "priority_score": round(priority_score, 6),
                "review_note": review_note,
            }
        )

    result = pd.DataFrame(rows, columns=MODEL_DIFFERENCE_EXPLANATION_COLUMNS).sort_values(
        ["priority_score", "segment_type", "segment"],
        ascending=[False, True, True],
    )
    if max_rows is not None:
        result = result.head(max_rows)
    result = result.reset_index(drop=True)
    result["priority_rank"] = range(1, len(result) + 1)
    return result


def _briefing_signal(row: pd.Series) -> tuple[str, str, float, str]:
    model = str(row.get("model", ""))
    baseline = str(row.get("baseline_model", ""))
    segment = str(row.get("segment", ""))
    relative_return = _clean_metric(row.get("relative_return_pct_delta"))
    stability = _clean_metric(row.get("stability_score_delta"))
    drawdown = _clean_metric(row.get("max_drawdown_pct_delta"))
    turnover = _clean_metric(row.get("turnover_delta"))

    weakness_score = max(0.0, -relative_return) * 2.0 + max(0.0, -stability) + max(0.0, drawdown) * 0.5
    strength_score = max(0.0, relative_return) * 2.0 + max(0.0, stability)

    if relative_return <= -0.5 and stability < 0.0:
        return (
            "weakness",
            "block_or_rework",
            weakness_score + max(0.0, turnover) * 0.1,
            f"{model} 在 {segment} 弱于基准 {baseline}：相对收益和稳定性同时落后，应先重做该 regime/window 的信号或仓位适配。",
        )
    if drawdown > 2.0:
        return (
            "risk",
            "reduce_or_guardrail",
            weakness_score,
            f"{model} 在 {segment} 相比基准 {baseline} 回撤扩大，晋升前需要增加风险约束或降低权重。",
        )
    if relative_return >= 0.5 and stability >= 0.0:
        return (
            "strength",
            "promote_candidate",
            strength_score,
            f"{model} 在 {segment} 强于基准 {baseline}：相对收益和稳定性改善，可作为该分段的晋升候选。",
        )
    if turnover > 1.0 and relative_return <= 0.25:
        return (
            "efficiency_check",
            "cost_review",
            max(0.0, turnover) + max(0.0, -relative_return),
            f"{model} 在 {segment} 换手高但收益改善不足，需要检查交易成本、滑点和信号噪声。",
        )
    return "", "", 0.0, ""


def build_model_review_briefing(
    analysis: RegimeWindowEffectAnalysis | pd.DataFrame,
    *,
    max_rows: int | None = None,
) -> pd.DataFrame:
    """Build a Chinese review briefing from regime/window model effect deltas."""

    deltas = analysis.model_effect_deltas if isinstance(analysis, RegimeWindowEffectAnalysis) else analysis
    if deltas.empty:
        return pd.DataFrame(columns=MODEL_REVIEW_BRIEFING_COLUMNS)

    rows: list[dict[str, object]] = []
    for _, row in deltas.iterrows():
        model = str(row.get("model", ""))
        baseline = str(row.get("baseline_model", ""))
        if baseline and model == baseline:
            continue
        signal, decision_hint, priority_score, review_note = _briefing_signal(row)
        if not signal:
            continue
        rows.append(
            {
                "schema_version": MODEL_REVIEW_BRIEFING_SCHEMA_VERSION,
                "priority_rank": 0,
                "model": model,
                "baseline_model": baseline,
                "segment_type": str(row.get("segment_type", "")),
                "segment": str(row.get("segment", "")),
                "window": str(row.get("window", "")) if pd.notna(row.get("window", "")) else "",
                "regime": str(row.get("regime", "")) if pd.notna(row.get("regime", "")) else "",
                "review_signal": signal,
                "decision_hint": decision_hint,
                "priority_score": round(priority_score, 6),
                "relative_return_pct_delta": _clean_metric(row.get("relative_return_pct_delta")),
                "stability_score_delta": _clean_metric(row.get("stability_score_delta")),
                "max_drawdown_pct_delta": _clean_metric(row.get("max_drawdown_pct_delta")),
                "turnover_delta": _clean_metric(row.get("turnover_delta")),
                "diagnosis": str(row.get("diagnosis", "")),
                "review_note": review_note,
            }
        )
    if not rows:
        return pd.DataFrame(columns=MODEL_REVIEW_BRIEFING_COLUMNS)

    result = pd.DataFrame(rows, columns=MODEL_REVIEW_BRIEFING_COLUMNS).sort_values(
        ["priority_score", "segment_type", "segment", "model"],
        ascending=[False, True, True, True],
    )
    if max_rows is not None:
        result = result.head(max_rows)
    result = result.reset_index(drop=True)
    result["priority_rank"] = range(1, len(result) + 1)
    return result


def _clean_string(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def _briefing_payload(row: pd.Series | None) -> dict[str, object]:
    if row is None:
        return {}
    return {
        "model": _clean_string(row.get("model", "")),
        "baseline_model": _clean_string(row.get("baseline_model", "")),
        "segment_type": _clean_string(row.get("segment_type", "")),
        "segment": _clean_string(row.get("segment", "")),
        "window": _clean_string(row.get("window", "")),
        "regime": _clean_string(row.get("regime", "")),
        "review_signal": _clean_string(row.get("review_signal", "")),
        "decision_hint": _clean_string(row.get("decision_hint", "")),
        "priority_rank": int(_clean_metric(row.get("priority_rank", 0))),
        "priority_score": _clean_metric(row.get("priority_score", 0.0)),
        "relative_return_pct_delta": _clean_metric(row.get("relative_return_pct_delta", 0.0)),
        "stability_score_delta": _clean_metric(row.get("stability_score_delta", 0.0)),
        "max_drawdown_pct_delta": _clean_metric(row.get("max_drawdown_pct_delta", 0.0)),
        "review_note": _clean_string(row.get("review_note", "")),
    }


def _top_briefing(briefing: pd.DataFrame, *, signal: str | None = None) -> dict[str, object]:
    if briefing.empty:
        return {}
    rows = briefing
    if signal is not None and "review_signal" in rows.columns:
        rows = rows.loc[rows["review_signal"].astype(str) == signal]
    if rows.empty:
        return {}
    ordered = rows.assign(
        _priority_rank=rows.get("priority_rank", pd.Series([999999] * len(rows))).map(_clean_metric),
        _priority_score=rows.get("priority_score", pd.Series([0.0] * len(rows))).map(_clean_metric),
    ).sort_values(["_priority_rank", "_priority_score"], ascending=[True, False])
    return _briefing_payload(ordered.iloc[0])


def _value_counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    if frame.empty or column not in frame.columns:
        return {}
    counts = frame[column].dropna().astype(str).value_counts().sort_index()
    return {str(key): int(value) for key, value in counts.items() if str(key)}


def _segment_gap_payloads(segment_effect_matrix: pd.DataFrame, *, max_items: int) -> list[dict[str, object]]:
    if segment_effect_matrix.empty or "relative_return_pct_spread" not in segment_effect_matrix.columns:
        return []
    ordered = segment_effect_matrix.assign(
        _spread=segment_effect_matrix["relative_return_pct_spread"].map(_clean_metric)
    ).sort_values(["_spread", "segment_type", "segment"], ascending=[False, True, True])
    rows: list[dict[str, object]] = []
    for _, row in ordered.head(max_items).iterrows():
        rows.append(
            {
                "segment_type": _clean_string(row.get("segment_type", "")),
                "segment": _clean_string(row.get("segment", "")),
                "window": _clean_string(row.get("window", "")),
                "regime": _clean_string(row.get("regime", "")),
                "baseline_model": _clean_string(row.get("baseline_model", "")),
                "best_relative_return_model": _clean_string(row.get("best_relative_return_model", "")),
                "worst_relative_return_model": _clean_string(row.get("worst_relative_return_model", "")),
                "relative_return_pct_spread": _clean_metric(row.get("relative_return_pct_spread", 0.0)),
                "best_relative_return_vs_baseline_delta": _clean_metric(
                    row.get("best_relative_return_vs_baseline_delta", 0.0)
                ),
                "best_stability_model": _clean_string(row.get("best_stability_model", "")),
                "worst_stability_model": _clean_string(row.get("worst_stability_model", "")),
                "stability_score_spread": _clean_metric(row.get("stability_score_spread", 0.0)),
                "highest_drawdown_model": _clean_string(row.get("highest_drawdown_model", "")),
                "highest_drawdown_pct": _clean_metric(row.get("highest_drawdown_pct", 0.0)),
                "review_note": _clean_string(row.get("review_note", "")),
            }
        )
    return rows


def _series_or_default(frame: pd.DataFrame, column: str, default: object) -> pd.Series:
    if column in frame.columns:
        return frame[column]
    return pd.Series([default] * len(frame), index=frame.index)


def _ordered_delta_row(group: pd.DataFrame, *, highest: bool) -> pd.Series:
    ordered = group.assign(_return_delta=group["relative_return_pct_delta"].map(_clean_metric)).sort_values(
        ["_return_delta", "segment_type", "segment", "model"],
        ascending=[not highest, True, True, True],
    )
    return ordered.iloc[0]


def _segment_payload_value(row: pd.Series, column: str) -> str:
    return _clean_string(row.get(column, ""))


def _stress_context_mask(frame: pd.DataFrame) -> pd.Series:
    context = (
        _series_or_default(frame, "segment", "").astype(str)
        + " "
        + _series_or_default(frame, "regime", "").astype(str)
        + " "
        + _series_or_default(frame, "window", "").astype(str)
    ).str.lower()
    return context.str.contains("stress|downtrend|bear|crash|drawdown", regex=True)


def _diagnostic_signal(
    *,
    model: str,
    baseline_model: str,
    strength_count: int,
    weakness_count: int,
    risk_count: int,
    negative_count: int,
    window_regime_negative_count: int,
    stress_negative_count: int,
) -> str:
    if model == baseline_model:
        return "baseline"
    if stress_negative_count > 0 and weakness_count > 0:
        return "stress_fragile"
    if weakness_count >= 2 or window_regime_negative_count >= 2:
        return "regime_fragile"
    if strength_count > 0 and negative_count == 0 and risk_count == 0:
        return "broad_strength"
    if strength_count > 0 and strength_count >= weakness_count:
        return "selective_strength"
    return "mixed"


def _diagnostic_review_note(
    *,
    model: str,
    baseline_model: str,
    signal: str,
    best_segment: str,
    worst_segment: str,
    strength_count: int,
    weakness_count: int,
    stress_negative_count: int,
) -> str:
    if signal == "baseline":
        return f"{model} 是当前分析基准；其他模型的 regime/window 差异均相对它计算。"
    if signal == "stress_fragile":
        return (
            f"{model} 在压力或下跌分段弱于基准 {baseline_model}，最弱分段为 {worst_segment}；"
            "晋升前应先检查压力期信号、仓位和风险约束。"
        )
    if signal == "regime_fragile":
        return (
            f"{model} 的弱点集中在多个 regime/window 分段，最弱分段为 {worst_segment}；"
            "需要拆分区间和 regime 重新验证适配性。"
        )
    if signal == "broad_strength":
        return f"{model} 在已评估分段中表现较稳定，最佳分段为 {best_segment}；可继续进入组合和成本复核。"
    if signal == "selective_strength":
        return (
            f"{model} 有 {strength_count} 个优势分段，也有 {weakness_count} 个弱势分段；"
            f"最佳分段为 {best_segment}，最弱分段为 {worst_segment}，适合作为分段候选而非全局替代。"
        )
    return (
        f"{model} 相对 {baseline_model} 的分段表现混合，压力弱势分段数为 {stress_negative_count}；"
        "需要结合因子暴露、换手和回撤来源继续 review。"
    )


def build_model_regime_window_diagnostics(
    analysis: RegimeWindowEffectAnalysis | pd.DataFrame,
) -> pd.DataFrame:
    """Summarize each model's stability across all/regime/window/window_regime segments."""

    deltas = analysis.model_effect_deltas if isinstance(analysis, RegimeWindowEffectAnalysis) else analysis
    if deltas.empty:
        return pd.DataFrame(columns=MODEL_REGIME_WINDOW_DIAGNOSTIC_COLUMNS)

    frame = deltas.copy()
    text_columns = {"segment_type", "segment", "window", "regime", "diagnosis", "baseline_model"}
    for column in [
        "relative_return_pct_delta",
        "stability_score_delta",
        "max_drawdown_pct_delta",
        "segment_type",
        "segment",
        "window",
        "regime",
        "diagnosis",
        "baseline_model",
    ]:
        if column not in frame.columns:
            frame[column] = "" if column in text_columns else 0.0

    baseline_values = frame["baseline_model"].dropna().astype(str)
    baseline_model = str(baseline_values.iloc[0]) if not baseline_values.empty else ""
    rows: list[dict[str, object]] = []
    for model, group in frame.groupby("model", sort=True):
        model_name = str(model)
        returns = group["relative_return_pct_delta"].map(_clean_metric)
        stability = group["stability_score_delta"].map(_clean_metric)
        drawdown = group["max_drawdown_pct_delta"].map(_clean_metric)
        diagnosis = group["diagnosis"].astype(str)
        positive = returns > 0.0
        negative = returns < 0.0
        weakness = diagnosis.str.contains("弱于基准", regex=False) | ((returns <= -0.5) & (stability < 0.0))
        strength = diagnosis.str.contains("强于基准", regex=False) | ((returns >= 0.5) & (stability > 0.0))
        risk = diagnosis.str.contains("风险更高", regex=False) | (drawdown > 2.0)
        window_regime_negative = (group["segment_type"].astype(str) == "window_regime") & negative
        stress_negative = _stress_context_mask(group) & (negative | weakness)
        evaluated_count = int(len(group))
        positive_count = int(positive.sum())
        negative_count = int(negative.sum())
        weakness_count = int(weakness.sum())
        strength_count = int(strength.sum())
        risk_count = int(risk.sum())
        window_regime_negative_count = int(window_regime_negative.sum())
        stress_negative_count = int(stress_negative.sum())
        positive_share = positive_count / evaluated_count if evaluated_count else 0.0
        consistency_score = (
            positive_share
            - (weakness_count / evaluated_count if evaluated_count else 0.0)
            - 0.5 * (risk_count / evaluated_count if evaluated_count else 0.0)
            - (0.25 if stress_negative_count else 0.0)
        )
        best = _ordered_delta_row(group, highest=True)
        worst = _ordered_delta_row(group, highest=False)
        signal = _diagnostic_signal(
            model=model_name,
            baseline_model=baseline_model,
            strength_count=strength_count,
            weakness_count=weakness_count,
            risk_count=risk_count,
            negative_count=negative_count,
            window_regime_negative_count=window_regime_negative_count,
            stress_negative_count=stress_negative_count,
        )
        best_segment = _segment_payload_value(best, "segment")
        worst_segment = _segment_payload_value(worst, "segment")
        rows.append(
            {
                "schema_version": MODEL_REGIME_WINDOW_DIAGNOSTIC_SCHEMA_VERSION,
                "model": model_name,
                "baseline_model": baseline_model,
                "evaluated_segment_count": evaluated_count,
                "positive_segment_count": positive_count,
                "negative_segment_count": negative_count,
                "weakness_segment_count": weakness_count,
                "strength_segment_count": strength_count,
                "risk_segment_count": risk_count,
                "window_regime_negative_count": window_regime_negative_count,
                "stress_negative_count": stress_negative_count,
                "positive_segment_share": round(float(positive_share), 6),
                "mean_relative_return_pct_delta": round(float(returns.mean()), 6) if evaluated_count else 0.0,
                "mean_stability_score_delta": round(float(stability.mean()), 6) if evaluated_count else 0.0,
                "mean_max_drawdown_pct_delta": round(float(drawdown.mean()), 6) if evaluated_count else 0.0,
                "consistency_score": round(float(consistency_score), 6),
                "best_segment_type": _segment_payload_value(best, "segment_type"),
                "best_segment": best_segment,
                "best_window": _segment_payload_value(best, "window"),
                "best_regime": _segment_payload_value(best, "regime"),
                "best_relative_return_pct_delta": _clean_metric(best.get("relative_return_pct_delta", 0.0)),
                "best_stability_score_delta": _clean_metric(best.get("stability_score_delta", 0.0)),
                "worst_segment_type": _segment_payload_value(worst, "segment_type"),
                "worst_segment": worst_segment,
                "worst_window": _segment_payload_value(worst, "window"),
                "worst_regime": _segment_payload_value(worst, "regime"),
                "worst_relative_return_pct_delta": _clean_metric(worst.get("relative_return_pct_delta", 0.0)),
                "worst_stability_score_delta": _clean_metric(worst.get("stability_score_delta", 0.0)),
                "worst_max_drawdown_pct_delta": _clean_metric(worst.get("max_drawdown_pct_delta", 0.0)),
                "diagnostic_signal": signal,
                "review_note": _diagnostic_review_note(
                    model=model_name,
                    baseline_model=baseline_model,
                    signal=signal,
                    best_segment=best_segment,
                    worst_segment=worst_segment,
                    strength_count=strength_count,
                    weakness_count=weakness_count,
                    stress_negative_count=stress_negative_count,
                ),
            }
        )

    result = pd.DataFrame(rows, columns=MODEL_REGIME_WINDOW_DIAGNOSTIC_COLUMNS)
    return result.sort_values(
        ["diagnostic_signal", "consistency_score", "model"],
        ascending=[True, True, True],
    ).reset_index(drop=True)


def _top_fragile_model_payloads(diagnostics: pd.DataFrame, *, max_items: int) -> list[dict[str, object]]:
    if diagnostics.empty:
        return []
    fragile = diagnostics.loc[
        diagnostics["diagnostic_signal"].astype(str).isin({"stress_fragile", "regime_fragile"})
    ].copy()
    if fragile.empty:
        return []
    ordered = fragile.assign(
        _weakness=fragile["weakness_segment_count"].map(_clean_metric),
        _stress=fragile["stress_negative_count"].map(_clean_metric),
        _consistency=fragile["consistency_score"].map(_clean_metric),
    ).sort_values(["_stress", "_weakness", "_consistency", "model"], ascending=[False, False, True, True])
    rows: list[dict[str, object]] = []
    for _, row in ordered.head(max_items).iterrows():
        rows.append(
            {
                "model": _clean_string(row.get("model", "")),
                "baseline_model": _clean_string(row.get("baseline_model", "")),
                "diagnostic_signal": _clean_string(row.get("diagnostic_signal", "")),
                "worst_segment": _clean_string(row.get("worst_segment", "")),
                "weakness_segment_count": int(_clean_metric(row.get("weakness_segment_count", 0))),
                "stress_negative_count": int(_clean_metric(row.get("stress_negative_count", 0))),
                "consistency_score": _clean_metric(row.get("consistency_score", 0.0)),
                "review_note": _clean_string(row.get("review_note", "")),
            }
        )
    return rows


def build_analysis_overview(
    analysis_or_summary: RegimeWindowEffectAnalysis | Mapping[str, Any],
    *,
    segment_effect_matrix: pd.DataFrame | None = None,
    model_review_briefing: pd.DataFrame | None = None,
    max_items: int = 5,
) -> dict[str, object]:
    """Build a compact regime/window model-difference summary for dashboard and schedule."""

    if isinstance(analysis_or_summary, RegimeWindowEffectAnalysis):
        summary = dict(analysis_or_summary.summary)
        matrix = segment_effect_matrix if segment_effect_matrix is not None else analysis_or_summary.segment_effect_matrix
        briefing = model_review_briefing if model_review_briefing is not None else build_model_review_briefing(analysis_or_summary)
    else:
        summary = dict(analysis_or_summary)
        matrix = segment_effect_matrix if segment_effect_matrix is not None else pd.DataFrame()
        briefing = model_review_briefing if model_review_briefing is not None else pd.DataFrame()

    gap_rows = _segment_gap_payloads(matrix, max_items=max_items)
    if not gap_rows:
        gap_rows = list(summary.get("largest_relative_return_spread_segments", []))[:max_items]

    return {
        "schema_version": ANALYSIS_OVERVIEW_SCHEMA_VERSION,
        "baseline_model": _clean_string(summary.get("baseline_model", "")),
        "model_count": int(_clean_metric(summary.get("model_count", 0))),
        "regime_count": int(_clean_metric(summary.get("regime_count", 0))),
        "window_count": int(_clean_metric(summary.get("window_count", 0))),
        "window_regime_count": int(_clean_metric(summary.get("window_regime_count", 0))),
        "segment_effect_matrix_count": int(_clean_metric(summary.get("segment_effect_matrix_count", len(matrix)))),
        "model_regime_window_diagnostic_count": int(
            _clean_metric(summary.get("model_regime_window_diagnostic_count", 0))
        ),
        "segment_types": [str(value) for value in summary.get("segment_types", [])],
        "qlib_metric_source": _clean_string(summary.get("qlib_metric_source", "")),
        "review_signal_counts": _value_counts(briefing, "review_signal"),
        "decision_hint_counts": _value_counts(briefing, "decision_hint"),
        "top_review": _top_briefing(briefing),
        "top_weakness": _top_briefing(briefing, signal="weakness"),
        "top_strength": _top_briefing(briefing, signal="strength"),
        "top_regime_fragile_models": list(summary.get("top_regime_fragile_models", []))[:max_items],
        "weakest_segments": list(summary.get("weakest_segments", []))[:max_items],
        "largest_relative_return_spread_segments": gap_rows,
        "dashboard_note": "统一 regime/window 模型差异摘要，供 dashboard 总览和 schedule 通知读取。",
    }


def render_model_review_briefing_markdown(
    briefing: pd.DataFrame,
    *,
    title: str = "模型差异 Review Briefing",
    max_rows: int = 12,
) -> str:
    lines = [f"# {title}", "", "## 中文结论", ""]
    if briefing.empty:
        lines.extend(
            [
                "- 当前没有触发显著的模型差异 review 项。",
                "- 是否晋升：不能仅凭空 briefing 晋升，仍需检查全区间收益、压力期回撤和交易成本。",
            ]
        )
        return "\n".join(lines) + "\n"

    top = briefing.head(max_rows)
    signal_counts = {str(key): int(value) for key, value in briefing["review_signal"].value_counts().sort_index().items()}
    lines.extend(
        [
            f"- 共生成 {len(briefing)} 条 review 项，信号分布：{signal_counts}。",
            "- 是否晋升：`promote_candidate` 只能作为分段候选；若同一模型存在 `block_or_rework`，应先处理压力区间弱点。",
            "- Review 顺序按相对收益差、稳定性差、回撤差和换手效率综合排序。",
            "",
            "| rank | model | segment | signal | decision | note |",
            "| ---: | --- | --- | --- | --- | --- |",
        ]
    )
    for _, row in top.iterrows():
        lines.append(
            "| {rank} | {model} | {segment} | {signal} | {decision} | {note} |".format(
                rank=int(row["priority_rank"]),
                model=str(row["model"]),
                segment=str(row["segment"]),
                signal=str(row["review_signal"]),
                decision=str(row["decision_hint"]),
                note=str(row["review_note"]),
            )
        )
    return "\n".join(lines) + "\n"


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
    segment_effect_matrix = build_segment_effect_matrix(frame, windows=windows, baseline_model=baseline)
    deltas = compare_segment_model_effects(frame, windows=windows, baseline_model=baseline)
    model_regime_window_diagnostics = build_model_regime_window_diagnostics(deltas)
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
    window_table = build_analysis_window_table(windows or {}, source="analysis_input")
    window_definitions = [
        {"window": str(row["window"]), "start": str(row["start"]), "end": str(row["end"])}
        for _, row in window_table.iterrows()
    ]
    summary = {
        "schema_version": "analysis.regime_window_effect.v1",
        "baseline_model": baseline,
        "model_count": int(returns["model"].nunique()) if not returns.empty else 0,
        "regime_count": int(returns["regime"].nunique()) if not returns.empty else 0,
        "window_count": int(len(windows or {})),
        "window_names": [item["window"] for item in window_definitions],
        "window_definitions": window_definitions,
        "window_regime_count": int(window_regime_metrics["segment"].nunique()) if not window_regime_metrics.empty else 0,
        "segment_effect_matrix_count": int(len(segment_effect_matrix)),
        "model_regime_window_diagnostic_count": int(len(model_regime_window_diagnostics)),
        "segment_types": sorted(str(value) for value in segment_metrics["segment_type"].dropna().unique().tolist()),
        "qlib_metric_source": QLIB_RISK_SOURCE,
        "weakest_segments": weakest_segments,
        "top_regime_fragile_models": _top_fragile_model_payloads(model_regime_window_diagnostics, max_items=5),
        "largest_relative_return_spread_segments": [
            {
                "segment_type": str(row["segment_type"]),
                "segment": str(row["segment"]),
                "relative_return_pct_spread": _clean_metric(row["relative_return_pct_spread"]),
                "best_relative_return_model": str(row["best_relative_return_model"]),
                "worst_relative_return_model": str(row["worst_relative_return_model"]),
            }
            for _, row in segment_effect_matrix.sort_values("relative_return_pct_spread", ascending=False).head(5).iterrows()
        ]
        if not segment_effect_matrix.empty
        else [],
    }
    return RegimeWindowEffectAnalysis(
        qlib_return_frame=returns,
        qlib_risk_metrics=risk,
        segment_metrics=segment_metrics,
        segment_effect_matrix=segment_effect_matrix,
        window_regime_metrics=window_regime_metrics,
        model_effect_deltas=deltas,
        model_regime_window_diagnostics=model_regime_window_diagnostics,
        summary=summary,
    )
