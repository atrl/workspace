from __future__ import annotations

import pandas as pd


def factor_ic(frame: pd.DataFrame, factor_col: str, forward_return_col: str) -> float:
    return round(float(frame[factor_col].corr(frame[forward_return_col], method="pearson")), 10)


def factor_correlation_matrix(frame: pd.DataFrame, factor_cols: list[str]) -> pd.DataFrame:
    return frame[factor_cols].corr().round(10)


def quantile_forward_returns(
    frame: pd.DataFrame,
    factor_col: str,
    forward_return_col: str,
    *,
    quantiles: int,
) -> pd.DataFrame:
    labels = range(1, quantiles + 1)
    bucketed = frame.copy()
    bucketed["quantile"] = pd.qcut(bucketed[factor_col], q=quantiles, labels=labels)
    result = (
        bucketed.groupby("quantile", observed=True)[forward_return_col]
        .agg(mean_forward_return="mean", count="count")
        .reset_index()
    )
    result["quantile"] = result["quantile"].astype(int)
    return result
