from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import pandas as pd


@dataclass(frozen=True)
class FactorSuiteReport:
    ic: pd.DataFrame
    correlation: pd.DataFrame
    quantile_returns: pd.DataFrame
    summary: dict[str, object]


def _validate_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"factor frame missing columns: {missing}")


def _clean_float(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(result):
        return 0.0
    return result


def _corr(left: pd.Series, right: pd.Series, *, method: str) -> float:
    pair = pd.concat([left.astype(float), right.astype(float)], axis=1).dropna()
    if len(pair) < 2:
        return 0.0
    return _clean_float(pair.iloc[:, 0].corr(pair.iloc[:, 1], method=method))


def compute_factor_ic(
    frame: pd.DataFrame,
    *,
    factor_columns: Sequence[str],
    target_column: str,
    date_column: str = "date",
    entity_column: str | None = None,
    method: str = "spearman",
    min_periods: int = 3,
) -> pd.DataFrame:
    _validate_columns(frame, [*factor_columns, target_column])
    data = frame.copy()
    rows: list[dict[str, object]] = []
    if entity_column and entity_column in data.columns and date_column in data.columns:
        scope = "cross_sectional"
        for factor in factor_columns:
            values: list[float] = []
            observations = 0
            for _, group in data.groupby(date_column, sort=True):
                valid = group[[factor, target_column, entity_column]].dropna()
                if len(valid) < min_periods:
                    continue
                observations += int(len(valid))
                values.append(_corr(valid[factor], valid[target_column], method=method))
            series = pd.Series(values, dtype=float)
            rows.append(
                {
                    "factor": factor,
                    "scope": scope,
                    "method": method,
                    "ic_mean": _clean_float(series.mean()) if not series.empty else 0.0,
                    "ic_std": _clean_float(series.std(ddof=0)) if not series.empty else 0.0,
                    "ic_positive_rate": _clean_float((series > 0).mean()) if not series.empty else 0.0,
                    "ic_negative_rate": _clean_float((series < 0).mean()) if not series.empty else 0.0,
                    "observation_count": observations,
                    "period_count": int(len(series)),
                }
            )
    else:
        scope = "time_series"
        for factor in factor_columns:
            valid = data[[factor, target_column]].dropna()
            ic = _corr(valid[factor], valid[target_column], method=method)
            rows.append(
                {
                    "factor": factor,
                    "scope": scope,
                    "method": method,
                    "ic_mean": ic,
                    "ic_std": 0.0,
                    "ic_positive_rate": 1.0 if ic > 0 else 0.0,
                    "ic_negative_rate": 1.0 if ic < 0 else 0.0,
                    "observation_count": int(len(valid)),
                    "period_count": 1 if len(valid) >= 2 else 0,
                }
            )
    return pd.DataFrame(rows).sort_values(["ic_mean", "factor"], ascending=[False, True]).reset_index(drop=True)


def compute_factor_correlation(
    frame: pd.DataFrame,
    *,
    factor_columns: Sequence[str],
    method: str = "spearman",
) -> pd.DataFrame:
    _validate_columns(frame, factor_columns)
    rows: list[dict[str, object]] = []
    for factor_a in factor_columns:
        for factor_b in factor_columns:
            value = _corr(frame[factor_a], frame[factor_b], method=method)
            rows.append(
                {
                    "factor_a": factor_a,
                    "factor_b": factor_b,
                    "method": method,
                    "correlation": value,
                    "abs_correlation": abs(value),
                }
            )
    return pd.DataFrame(rows).sort_values(["factor_a", "factor_b"]).reset_index(drop=True)


def build_composite_factor(
    frame: pd.DataFrame,
    *,
    factor_columns: Sequence[str],
    weights: Mapping[str, float] | None = None,
    output_column: str = "composite_factor",
) -> pd.DataFrame:
    _validate_columns(frame, factor_columns)
    data = frame.copy()
    weighted = pd.Series(0.0, index=data.index, dtype=float)
    for factor in factor_columns:
        values = data[factor].astype(float)
        std = float(values.std(ddof=0))
        zscore = (values - float(values.mean())) / std if std > 1e-12 else values * 0.0
        weighted = weighted.add(zscore * float((weights or {}).get(factor, 1.0)), fill_value=0.0)
    data[output_column] = weighted
    return data


def compute_factor_quantile_returns(
    frame: pd.DataFrame,
    *,
    factor_column: str,
    target_column: str,
    quantiles: int = 5,
) -> pd.DataFrame:
    if quantiles < 2:
        raise ValueError("quantiles must be at least 2")
    _validate_columns(frame, [factor_column, target_column])
    valid = frame[[factor_column, target_column]].dropna().copy()
    if valid.empty:
        return pd.DataFrame(
            columns=["factor", "quantile", "count", "mean_return", "median_return", "spread_vs_bottom"]
        )
    rank = valid[factor_column].rank(method="first")
    valid["quantile"] = pd.qcut(rank, q=min(quantiles, len(valid)), labels=False, duplicates="drop") + 1
    grouped = valid.groupby("quantile", sort=True)[target_column]
    result = grouped.agg(count="count", mean_return="mean", median_return="median").reset_index()
    result.insert(0, "factor", factor_column)
    bottom = float(result.loc[result["quantile"] == result["quantile"].min(), "mean_return"].iloc[0])
    result["spread_vs_bottom"] = result["mean_return"].astype(float) - bottom
    return result.reset_index(drop=True)


def analyze_factor_suite(
    frame: pd.DataFrame,
    *,
    factor_columns: Sequence[str],
    target_column: str,
    date_column: str = "date",
    entity_column: str | None = None,
    quantiles: int = 5,
    method: str = "spearman",
) -> FactorSuiteReport:
    ic = compute_factor_ic(
        frame,
        factor_columns=factor_columns,
        target_column=target_column,
        date_column=date_column,
        entity_column=entity_column,
        method=method,
    )
    correlation = compute_factor_correlation(frame, factor_columns=factor_columns, method=method)
    quantile_frames = [
        compute_factor_quantile_returns(
            frame,
            factor_column=factor,
            target_column=target_column,
            quantiles=quantiles,
        )
        for factor in factor_columns
    ]
    quantile_returns = pd.concat(quantile_frames, ignore_index=True) if quantile_frames else pd.DataFrame()
    top_factor = str(ic.iloc[0]["factor"]) if not ic.empty else ""
    top_ic = _clean_float(ic.iloc[0]["ic_mean"]) if not ic.empty else 0.0
    high_corr = correlation.loc[
        (correlation["factor_a"] != correlation["factor_b"]) & (correlation["abs_correlation"] >= 0.8)
    ]
    review_text = (
        "## 中文结论\n\n"
        f"- 本次评估 {len(factor_columns)} 个因子，目标列为 `{target_column}`。\n"
        f"- IC 最强因子为 `{top_factor}`，IC 均值为 {top_ic:.4f}。\n"
        f"- 高相关因子对数量为 {len(high_corr)}，用于提示多因子冗余风险。\n"
        "- 分层收益已输出 quantile 表，用于判断因子方向和单调性。"
    )
    summary: dict[str, object] = {
        "schema_version": "factor_suite_report.v1",
        "factor_count": int(len(factor_columns)),
        "target_column": target_column,
        "top_factor": top_factor,
        "top_ic_mean": top_ic,
        "high_correlation_pair_count": int(len(high_corr)),
        "review_text": review_text,
    }
    return FactorSuiteReport(ic=ic, correlation=correlation, quantile_returns=quantile_returns, summary=summary)
