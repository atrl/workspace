from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


def _factor_ic(frame: pd.DataFrame, factor_col: str, label_col: str) -> float:
    clean = frame[[factor_col, label_col]].dropna()
    if clean.empty or clean[factor_col].nunique() <= 1 or clean[label_col].nunique() <= 1:
        return 0.0
    return round(float(clean[factor_col].corr(clean[label_col], method="pearson")), 10)


@dataclass(frozen=True)
class FactorDiagnosticsReport:
    source: str
    segment: str
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


def qlib_factor_diagnostics(
    dataset,
    *,
    segment: str,
    factor_cols: list[str],
    label_col: str,
    quantiles: int,
) -> FactorDiagnosticsReport:
    frame = dataset.prepare(segment).copy()
    missing = [column for column in [*factor_cols, label_col] if column not in frame.columns]
    if missing:
        raise ValueError(f"missing qlib dataset columns: {missing}")

    ic_table = pd.DataFrame(
        [{"factor": factor, "ic": _factor_ic(frame, factor, label_col)} for factor in factor_cols]
    )
    correlation_matrix = frame[factor_cols].corr().fillna(0.0).round(10)
    score = frame[factor_cols].mean(axis=1)
    ranked = pd.DataFrame({"score": score, "forward_return": frame[label_col]}).dropna()
    ranked["quantile"] = pd.qcut(ranked["score"].rank(method="first"), q=quantiles, labels=range(1, quantiles + 1))
    quantile_returns = (
        ranked.groupby("quantile", observed=True)["forward_return"]
        .agg(mean_forward_return="mean", count="count")
        .reset_index()
    )
    quantile_returns["quantile"] = quantile_returns["quantile"].astype(int)
    return FactorDiagnosticsReport(
        source="qlib_dataset",
        segment=segment,
        ic_table=ic_table,
        correlation_matrix=correlation_matrix,
        quantile_returns=quantile_returns,
    )
