from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import pandas as pd

from .diagnostics import FactorSuiteReport, analyze_factor_suite


@dataclass(frozen=True)
class FactorFeatureSetSpec:
    name: str
    feature_columns: tuple[str, ...]
    description: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class FactorFeatureSetPlan:
    train: pd.DataFrame
    test: pd.DataFrame
    feature_sets: tuple[FactorFeatureSetSpec, ...]
    diagnostics: FactorSuiteReport
    summary: dict[str, object]


def _zscore_with_train_params(train: pd.Series, values: pd.Series) -> pd.Series:
    train_values = train.astype(float)
    mean = float(train_values.mean())
    std = float(train_values.std(ddof=0))
    if std <= 1e-12:
        return values.astype(float) * 0.0
    return (values.astype(float) - mean) / std


def _add_ic_weighted_composite(
    *,
    train: pd.DataFrame,
    test: pd.DataFrame,
    weights: dict[str, float],
    output_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_out = train.copy()
    test_out = test.copy()
    train_composite = pd.Series(0.0, index=train_out.index, dtype=float)
    test_composite = pd.Series(0.0, index=test_out.index, dtype=float)
    for factor, weight in weights.items():
        train_z = _zscore_with_train_params(train_out[factor], train_out[factor])
        test_z = _zscore_with_train_params(train_out[factor], test_out[factor])
        train_composite = train_composite.add(train_z * float(weight), fill_value=0.0)
        test_composite = test_composite.add(test_z * float(weight), fill_value=0.0)
    train_out[output_column] = train_composite
    test_out[output_column] = test_composite
    return train_out, test_out


def prepare_factor_feature_sets(
    *,
    train: pd.DataFrame,
    test: pd.DataFrame,
    factor_columns: Sequence[str],
    target_column: str,
    top_n: int = 5,
    min_abs_ic: float = 0.0,
    composite_column: str = "ic_weighted_composite",
) -> FactorFeatureSetPlan:
    if train.empty or test.empty:
        raise ValueError("train and test frames must be non-empty")
    if top_n <= 0:
        raise ValueError("top_n must be positive")
    diagnostics = analyze_factor_suite(
        train,
        factor_columns=tuple(factor_columns),
        target_column=target_column,
    )
    ranked = diagnostics.ic.assign(abs_ic=diagnostics.ic["ic_mean"].astype(float).abs()).sort_values(
        ["abs_ic", "ic_mean", "factor"],
        ascending=[False, False, True],
    )
    selected = ranked.loc[ranked["abs_ic"] >= float(min_abs_ic), "factor"].head(top_n).astype(str).tolist()
    if not selected:
        selected = ranked["factor"].head(min(top_n, len(ranked))).astype(str).tolist()
    weights = {
        str(row.factor): float(row.ic_mean)
        for row in ranked.loc[ranked["factor"].isin(selected)].itertuples(index=False)
    }
    train_out, test_out = _add_ic_weighted_composite(
        train=train,
        test=test,
        weights=weights,
        output_column=composite_column,
    )
    feature_sets = (
        FactorFeatureSetSpec(
            name="all_features",
            feature_columns=tuple(factor_columns),
            description="原始 supervised 特征全集。",
        ),
        FactorFeatureSetSpec(
            name="top_ic_factors",
            feature_columns=tuple(selected),
            description="仅使用训练集 IC 绝对值筛选出的候选因子。",
        ),
        FactorFeatureSetSpec(
            name="ic_weighted_composite",
            feature_columns=(composite_column,),
            description="使用训练集 IC 作为权重、训练集均值方差标准化后的组合因子。",
        ),
    )
    summary: dict[str, object] = {
        "schema_version": "factor_feature_set_plan.v1",
        "selection_source": "train_only_factor_ic",
        "top_n": int(top_n),
        "min_abs_ic": float(min_abs_ic),
        "factor_count": int(len(factor_columns)),
        "selected_factors": selected,
        "composite_column": composite_column,
    }
    return FactorFeatureSetPlan(
        train=train_out,
        test=test_out,
        feature_sets=feature_sets,
        diagnostics=diagnostics,
        summary=summary,
    )
