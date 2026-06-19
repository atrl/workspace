from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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


def combine_factor_scores(
    frame: pd.DataFrame,
    *,
    weights: dict[str, float],
    output_col: str = "factor_score",
    standardize: bool = True,
) -> pd.DataFrame:
    if not weights:
        raise ValueError("weights must include at least one factor")
    missing = [factor for factor in weights if factor not in frame.columns]
    if missing:
        raise ValueError(f"missing factor columns: {missing}")

    result = frame.copy()
    score = pd.Series(0.0, index=result.index, dtype="float64")
    for factor, weight in weights.items():
        values = result[factor].astype(float)
        if standardize:
            std = float(values.std(ddof=0))
            values = values - float(values.mean())
            if std > 0:
                values = values / std
        score = score + values * float(weight)
    result[output_col] = score
    return result


@dataclass(frozen=True)
class FactorCombinationReport:
    ic_table: pd.DataFrame
    correlation_matrix: pd.DataFrame
    quantile_returns: pd.DataFrame

    def write_sidecars(self, directory: Path) -> dict[str, Path]:
        directory.mkdir(parents=True, exist_ok=True)
        paths = {
            "ic": directory / "factor_ic.csv",
            "correlation": directory / "factor_correlation.csv",
            "quantiles": directory / "factor_quantiles.csv",
        }
        self.ic_table.to_csv(paths["ic"], index=False)
        self.correlation_matrix.to_csv(paths["correlation"])
        self.quantile_returns.to_csv(paths["quantiles"], index=False)
        return paths


def factor_combination_diagnostics(
    frame: pd.DataFrame,
    *,
    factor_cols: list[str],
    score_col: str,
    forward_return_col: str,
    quantiles: int,
) -> FactorCombinationReport:
    diagnostics_cols = factor_cols + [score_col]
    ic_table = pd.DataFrame(
        [
            {"factor": factor, "ic": factor_ic(frame, factor, forward_return_col)}
            for factor in diagnostics_cols
        ]
    )
    return FactorCombinationReport(
        ic_table=ic_table,
        correlation_matrix=factor_correlation_matrix(frame, factor_cols),
        quantile_returns=quantile_forward_returns(frame, score_col, forward_return_col, quantiles=quantiles),
    )
