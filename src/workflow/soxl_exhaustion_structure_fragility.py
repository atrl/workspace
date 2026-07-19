"""Qlib-native factor-only SOXL exhaustion, structure, and fragility experiment."""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data import QlibProviderConfig, build_qlib_dataset, build_qlib_store_from_frame, load_qlib_feature_frame
from src.factor.intraday_15m_factors import build_intraday_15m_factor_frame
from src.factor.multi_3x_price_volume import (
    PRICE_ONLY_BASELINE_FEATURES,
    build_multi_3x_price_volume_frame,
)
from src.factor.soxl_exhaustion_structure import (
    FIXED_BELLWETHERS,
    LEVERAGE_FRAGILITY_FEATURES,
    PRICE_PATH_EXHAUSTION_FEATURES,
    STRUCTURE_FEATURES,
    build_exhaustion_fragility_frame,
    build_fixed_bellwether_structure_frame,
)
from src.workflow.multi_3x_price_volume_transfer import (
    _diagnostic_metrics,
    factor_group_diagnostics,
    select_transferable_factors,
)


EXPERIMENT = "soxl_exhaustion_structure_fragility_v1"
CARD_TYPE = "soxl_factor_only_exhaustion_structure_fragility_research_card"
QLIB_DOCS = {
    "data": "https://qlib.readthedocs.io/en/latest/component/data.html",
    "workflow": "https://qlib.readthedocs.io/en/latest/component/workflow.html",
    "recorder": "https://qlib.readthedocs.io/en/latest/component/recorder.html",
}

INTRADAY_PATH_FEATURES = [
    "intraday_late_selloff_pressure",
    "intraday_close_location_pressure",
    "intraday_leverage_residual_pressure",
    "intraday_range_ratio_pressure",
    "intraday_realized_vol_ratio_pressure",
]

FAMILY_FEATURES = {
    "price_path_exhaustion": PRICE_PATH_EXHAUSTION_FEATURES,
    "leverage_fragility": LEVERAGE_FRAGILITY_FEATURES,
    "fixed_bellwether_structure": STRUCTURE_FEATURES,
    "soxl_soxx_intraday_path": INTRADAY_PATH_FEATURES,
}


@dataclass(frozen=True)
class SoxlExhaustionStructureFragilitySpec:
    root: Path
    experiment: str = EXPERIMENT
    start: str = "2011-01-03"
    as_of: str = "2026-07-02"
    development_end: str = "2022-12-31"
    outer_start: str = "2023-01-01"
    normalization_window: int = 252
    min_periods: int = 60
    label_horizon: int = 20

    @property
    def card_dir(self) -> Path:
        return self.root / "research" / self.experiment

    @property
    def provider_uri(self) -> Path:
        return self.root / "data" / "qlib" / self.experiment

    @property
    def constituent_provider_uri(self) -> Path:
        return self.root / "data" / "qlib" / "soxx_constituents"

    @property
    def daily_soxl_provider_uri(self) -> Path:
        return self.root / "data" / "qlib" / "soxl"

    @property
    def intraday_provider_uri(self) -> Path:
        return self.root / "data" / "qlib" / "soxl_intraday_15min"

    @property
    def external_intraday_provider_uri(self) -> Path:
        return self.root / "data" / "qlib" / "soxl_external_intraday_15min"


def split_family_development_outer(
    frame: pd.DataFrame,
    *,
    development_end: str = "2022-12-31",
    outer_start: str = "2023-01-01",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = pd.to_datetime(frame["datetime"])
    development = frame.loc[frame["instrument"].ne("SOXL") & dates.le(development_end)].copy()
    structure_development = frame.loc[frame["instrument"].eq("SOXL") & dates.le(development_end)].copy()
    outer = frame.loc[frame["instrument"].eq("SOXL") & dates.ge(outer_start)].copy()
    return tuple(piece.reset_index(drop=True) for piece in (development, structure_development, outer))


def conditional_incremental_lift(
    labels: pd.Series,
    price_scores: pd.Series,
    factor_scores: pd.Series,
    *,
    min_bin_observations: int = 20,
) -> dict[str, Any]:
    """Compare high versus low factor scores within fixed price-score quintiles."""

    work = pd.DataFrame(
        {"label": labels.astype(float), "price": price_scores.astype(float), "factor": factor_scores.astype(float)}
    ).dropna()
    if len(work) < 10 or work["price"].nunique() < 2 or work["factor"].nunique() < 2:
        return {"conditional_event_rate_lift": np.nan, "eligible_price_bins": 0, "observations": int(len(work))}
    work["price_bin"] = pd.qcut(work["price"].rank(method="first"), q=5, labels=False)
    lifts: list[float] = []
    weights: list[int] = []
    for _, group in work.groupby("price_bin", sort=True):
        if len(group) < min_bin_observations or group["factor"].nunique() < 2:
            continue
        median = float(group["factor"].median())
        low = group.loc[group["factor"].le(median), "label"]
        high = group.loc[group["factor"].gt(median), "label"]
        if low.empty or high.empty:
            continue
        lifts.append(float(high.mean() - low.mean()))
        weights.append(int(min(len(low), len(high))))
    return {
        "conditional_event_rate_lift": float(np.average(lifts, weights=weights)) if lifts else np.nan,
        "eligible_price_bins": int(len(lifts)),
        "observations": int(len(work)),
    }


def intraday_coverage_decision(
    *,
    core_start: str,
    external_complete_start: str,
    development_end: str,
    outer_start: str,
) -> dict[str, Any]:
    core_eligible = pd.Timestamp(core_start) < pd.Timestamp(development_end)
    external_eligible = pd.Timestamp(external_complete_start) < pd.Timestamp(development_end)
    return {
        "soxl_soxx_intraday": {
            "status": "eligible" if core_eligible else "blocked_insufficient_history",
            "start": core_start,
            "included_in_selection": bool(core_eligible),
        },
        "external_bellwether_intraday": {
            "status": "eligible" if external_eligible else "blocked_insufficient_history",
            "complete_start": external_complete_start,
            "required_development_end": development_end,
            "outer_start": outer_start,
            "included_in_selection": bool(external_eligible),
        },
    }


def evaluate_family_decision(
    *,
    selected_factors: list[str],
    price_metrics: dict[str, float] | pd.Series,
    family_metrics: dict[str, float] | pd.Series,
    conditional_lift: float,
    annual_metrics: pd.DataFrame | None = None,
) -> dict[str, Any]:
    selected_gate = bool(selected_factors)
    gates = {
        "development_selection": {"passed": selected_gate, "value": len(selected_factors), "threshold": 1},
        "outer_auc_increment": {
            "passed": selected_gate and float(family_metrics["auc"]) > float(price_metrics["auc"]),
            "value": float(family_metrics["auc"] - price_metrics["auc"]),
            "threshold": 0.0,
        },
        "outer_ap_lift_increment": {
            "passed": selected_gate
            and float(family_metrics["average_precision_lift"]) > float(price_metrics["average_precision_lift"]),
            "value": float(family_metrics["average_precision_lift"] - price_metrics["average_precision_lift"]),
            "threshold": 0.0,
        },
        "outer_quintile_lift_increment": {
            "passed": selected_gate and float(family_metrics["quintile_lift"]) > float(price_metrics["quintile_lift"]),
            "value": float(family_metrics["quintile_lift"] - price_metrics["quintile_lift"]),
            "threshold": 0.0,
        },
        "conditional_increment": {
            "passed": selected_gate and np.isfinite(conditional_lift) and conditional_lift > 0.0,
            "value": float(conditional_lift),
            "threshold": 0.0,
        },
    }
    if annual_metrics is not None:
        eligible = annual_metrics.loc[
            annual_metrics["observations"].ge(50)
            & annual_metrics["breakdown_events"].ge(5)
            & annual_metrics["observations"].sub(annual_metrics["breakdown_events"]).ge(5)
            & annual_metrics["auc"].notna()
            & annual_metrics["quintile_lift"].notna()
        ]
        positive = eligible["auc"].gt(0.50) & eligible["quintile_lift"].gt(0.0)
        gates["annual_direction_consistency"] = {
            "passed": selected_gate and len(eligible) > 0 and bool(positive.all()),
            "positive_years": int(positive.sum()),
            "eligible_years": int(len(eligible)),
            "threshold": "all_eligible_years",
        }
    all_passed = all(bool(gate["passed"]) for gate in gates.values())
    return {
        "decision": "retained_incremental_factor_evidence" if all_passed else "not_retained_no_incremental_factor_evidence",
        "all_passed": all_passed,
        "gates": gates,
    }


def _rolling_percentile(series: pd.Series, window: int = 252, min_periods: int = 60) -> pd.Series:
    def rank_last(values: np.ndarray) -> float:
        finite = values[np.isfinite(values)]
        return float(np.mean(finite <= values[-1])) if np.isfinite(values[-1]) and len(finite) else np.nan

    return series.rolling(window, min_periods=min_periods).apply(rank_last, raw=True)


def _build_intraday_path_frame(spec: SoxlExhaustionStructureFragilitySpec) -> pd.DataFrame:
    frame = build_intraday_15m_factor_frame(
        QlibProviderConfig(
            provider_uri=spec.intraday_provider_uri,
            market="all",
            region="us",
            freq="15min",
            kernels=1,
        ),
        instruments=["SOXL", "SOXX"],
        start="2018-12-31",
        end=spec.as_of,
        lag_days=1,
    )
    raw = {
        "intraday_late_selloff_pressure": -pd.to_numeric(frame["intraday15m_soxl_late_ret_lag1"], errors="coerce"),
        "intraday_close_location_pressure": 1.0
        - pd.to_numeric(frame["intraday15m_soxl_close_location_lag1"], errors="coerce"),
        "intraday_leverage_residual_pressure": -pd.to_numeric(
            frame["intraday15m_soxl_minus_3x_soxx_ret_lag1"], errors="coerce"
        ),
        "intraday_range_ratio_pressure": pd.to_numeric(
            frame["intraday15m_range_ratio_soxl_soxx_lag1"], errors="coerce"
        ),
        "intraday_realized_vol_ratio_pressure": pd.to_numeric(
            frame["intraday15m_realized_vol_ratio_soxl_soxx_lag1"], errors="coerce"
        ),
    }
    out = pd.DataFrame({"datetime": pd.to_datetime(frame["datetime"]).dt.normalize()})
    for name, values in raw.items():
        # The raw aggregate already describes the prior completed session.
        out[name] = _rolling_percentile(values, spec.normalization_window, spec.min_periods).clip(0.0, 1.0)
    return out


def _load_fixed_structure(spec: SoxlExhaustionStructureFragilitySpec) -> tuple[pd.DataFrame, dict[str, Any]]:
    constituents = load_qlib_feature_frame(
        QlibProviderConfig(provider_uri=spec.constituent_provider_uri, market="all", region="us", kernels=1),
        instruments=list(FIXED_BELLWETHERS),
        fields=["$close"],
        start=spec.start,
        end=spec.as_of,
    ).reset_index()
    constituents = constituents.rename(columns={"$close": "close"})
    soxx = load_qlib_feature_frame(
        QlibProviderConfig(provider_uri=spec.daily_soxl_provider_uri, market="all", region="us", kernels=1),
        instruments=["SOXX"],
        fields=["$close"],
        start=spec.start,
        end=spec.as_of,
    ).reset_index().rename(columns={"$close": "soxx_close"})
    structure = build_fixed_bellwether_structure_frame(
        constituents[["instrument", "datetime", "close"]],
        soxx[["datetime", "soxx_close"]],
        normalization_window=spec.normalization_window,
        min_periods=spec.min_periods,
    )
    audit = {
        "provider_uri": spec.constituent_provider_uri.resolve().as_posix(),
        "source_api": "qlib.data.D.features",
        "fixed_bellwethers": list(FIXED_BELLWETHERS),
        "membership_contract": structure.attrs["membership_contract"],
        "row_count": int(len(constituents)),
        "duplicate_instrument_datetime_rows": int(constituents.duplicated(["instrument", "datetime"]).sum()),
        "required_close_null_rows": int(constituents["close"].isna().sum()),
        "min_datetime": pd.Timestamp(constituents["datetime"].min()).date().isoformat(),
        "max_datetime": pd.Timestamp(constituents["datetime"].max()).date().isoformat(),
        "limitation": "fixed current bellwether proxy, not point-in-time historical SOXX membership",
    }
    return structure, audit


def _selection_for_family(
    development: pd.DataFrame,
    *,
    factors: list[str],
    group_column: str,
    min_observations: int,
    min_events: int,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    work = development.copy()
    work["underlying_group"] = work[group_column].astype(str)
    group_metrics = factor_group_diagnostics(
        work,
        factors=factors,
        min_observations=min_observations,
        min_events=min_events,
    )
    high = work.loc[work["high_level_state"].eq(1.0)]
    correlation = high[factors].corr(method="spearman")
    selection = select_transferable_factors(
        group_metrics,
        correlation=correlation,
        min_direction_consistency=0.70,
        min_median_auc=0.52,
        min_median_quintile_lift=0.02,
    )
    return selection, group_metrics, correlation


def _window_metrics(frame: pd.DataFrame, score_column: str) -> dict[str, Any]:
    sample = frame.loc[frame["high_level_state"].eq(1.0), ["breakdown_label", score_column]].dropna()
    return {
        "observations": int(len(sample)),
        "breakdown_events": int(sample["breakdown_label"].sum()) if len(sample) else 0,
    } | _diagnostic_metrics(sample["breakdown_label"], sample[score_column])


def _evaluate_outer(
    outer: pd.DataFrame,
    *,
    selections: dict[str, list[str]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any], pd.DataFrame]:
    scored = outer.copy()
    scored["price_only_score"] = scored[PRICE_ONLY_BASELINE_FEATURES].mean(axis=1, skipna=False)
    price_metrics = _window_metrics(scored, "price_only_score")
    individual_rows: list[dict[str, Any]] = []
    for family, factors in FAMILY_FEATURES.items():
        for factor in factors:
            metrics = _window_metrics(scored, factor)
            conditional = conditional_incremental_lift(
                scored.loc[scored["high_level_state"].eq(1.0), "breakdown_label"],
                scored.loc[scored["high_level_state"].eq(1.0), "price_only_score"],
                scored.loc[scored["high_level_state"].eq(1.0), factor],
            )
            individual_rows.append(
                {"family": family, "factor": factor, "development_selected": factor in selections[family]}
                | metrics
                | conditional
            )

    family_rows = [{"window": "overall", "family": "price_only", "selected_factors": ""} | price_metrics]
    decisions: dict[str, Any] = {}
    annual_rows: list[dict[str, Any]] = []
    dates = pd.to_datetime(scored["datetime"])
    for family, selected in selections.items():
        if not selected:
            decisions[family] = evaluate_family_decision(
                selected_factors=[],
                price_metrics=price_metrics,
                family_metrics=price_metrics,
                conditional_lift=np.nan,
            )
            continue
        score_column = f"{family}_score"
        scored[score_column] = scored[selected].mean(axis=1, skipna=False)
        family_metrics = _window_metrics(scored, score_column)
        high = scored["high_level_state"].eq(1.0)
        conditional = conditional_incremental_lift(
            scored.loc[high, "breakdown_label"], scored.loc[high, "price_only_score"], scored.loc[high, score_column]
        )
        family_rows.append(
            {"window": "overall", "family": family, "selected_factors": ",".join(selected)}
            | family_metrics
            | conditional
        )
        family_annual: list[dict[str, Any]] = []
        for year in sorted(dates.dt.year.unique()):
            year_frame = scored.loc[dates.dt.year.eq(year)]
            row = {"window": str(year), "family": family} | _window_metrics(year_frame, score_column)
            family_annual.append(row)
            annual_rows.append(row)
        decisions[family] = evaluate_family_decision(
            selected_factors=selected,
            price_metrics=price_metrics,
            family_metrics=family_metrics,
            conditional_lift=float(conditional["conditional_event_rate_lift"]),
            annual_metrics=pd.DataFrame(family_annual),
        )
    return (
        pd.DataFrame(individual_rows),
        pd.DataFrame(family_rows),
        pd.DataFrame(annual_rows),
        decisions,
        scored,
    )


def _resolve_root(root: Path | str | None) -> Path:
    if root is not None:
        return Path(root).resolve()
    current = Path.cwd().resolve()
    while current != current.parent:
        if (current / "AGENTS.md").exists() and (current / "SPEC.md").exists():
            return current
        current = current.parent
    raise ValueError("could not resolve repository root")


def run_soxl_exhaustion_structure_fragility_card(*, root: Path | str | None = None) -> dict[str, Any]:
    from src.workflow.multi_3x_price_volume_transfer import _load_source_panel

    spec = SoxlExhaustionStructureFragilitySpec(root=_resolve_root(root))
    panel, daily_source_audit, universe, underlying_groups = _load_source_panel(spec)
    baseline = build_multi_3x_price_volume_frame(
        panel,
        label_horizon=spec.label_horizon,
        normalization_window=spec.normalization_window,
        min_periods=spec.min_periods,
    )
    added = build_exhaustion_fragility_frame(
        panel,
        normalization_window=spec.normalization_window,
        min_periods=spec.min_periods,
    )
    feature_frame = baseline[
        ["instrument", "datetime", *PRICE_ONLY_BASELINE_FEATURES, "high_level_state", "breakdown_label"]
    ].merge(
        added[["instrument", "datetime", *PRICE_PATH_EXHAUSTION_FEATURES, *LEVERAGE_FRAGILITY_FEATURES]],
        on=["instrument", "datetime"],
        how="left",
        validate="one_to_one",
    )
    structure, structure_audit = _load_fixed_structure(spec)
    intraday = _build_intraday_path_frame(spec)
    feature_frame = feature_frame.merge(structure, on="datetime", how="left", validate="many_to_one")
    feature_frame = feature_frame.merge(intraday, on="datetime", how="left", validate="many_to_one")
    all_features = [feature for features in FAMILY_FEATURES.values() for feature in features]

    provider_manifest = build_qlib_store_from_frame(
        feature_frame[
            ["instrument", "datetime", *PRICE_ONLY_BASELINE_FEATURES, *all_features, "high_level_state", "breakdown_label"]
        ],
        provider_uri=spec.provider_uri,
        as_of=spec.as_of,
        source="qlib D.features daily pair, fixed bellwether, and lagged SOXL/SOXX 15min providers",
    )
    fields = [
        *[f"${feature}" for feature in [*PRICE_ONLY_BASELINE_FEATURES, *all_features]],
        "$high_level_state",
        "$breakdown_label",
    ]
    readback = load_qlib_feature_frame(
        QlibProviderConfig(provider_uri=spec.provider_uri, market="all", region="us", kernels=1),
        instruments=universe,
        fields=fields,
        start=spec.start,
        end=spec.as_of,
    ).reset_index()
    readback = readback.rename(columns={field: field.removeprefix("$") for field in fields})
    readback["datetime"] = pd.to_datetime(readback["datetime"]).dt.normalize()
    readback["underlying_group"] = readback["instrument"].map(underlying_groups)
    development, soxl_development, outer = split_family_development_outer(
        readback,
        development_end=spec.development_end,
        outer_start=spec.outer_start,
    )
    soxl_development["development_year"] = pd.to_datetime(soxl_development["datetime"]).dt.year.astype(str)

    selections: dict[str, list[str]] = {}
    summaries: list[pd.DataFrame] = []
    group_tables: list[pd.DataFrame] = []
    correlations: list[pd.DataFrame] = []
    for family in ("price_path_exhaustion", "leverage_fragility"):
        selection, groups, correlation = _selection_for_family(
            development,
            factors=FAMILY_FEATURES[family],
            group_column="underlying_group",
            min_observations=120,
            min_events=10,
        )
        selections[family] = selection["selected_factors"]
        summaries.append(selection["factor_summary"].assign(family=family, selection_uses_soxl=False))
        group_tables.append(groups.assign(family=family))
        correlations.append(correlation.rename_axis("factor").reset_index().assign(family=family))
    for family in ("fixed_bellwether_structure", "soxl_soxx_intraday_path"):
        selection, groups, correlation = _selection_for_family(
            soxl_development,
            factors=FAMILY_FEATURES[family],
            group_column="development_year",
            min_observations=50,
            min_events=5,
        )
        selections[family] = selection["selected_factors"]
        summaries.append(selection["factor_summary"].assign(family=family, selection_uses_soxl=True))
        group_tables.append(groups.assign(family=family))
        correlations.append(correlation.rename_axis("factor").reset_index().assign(family=family))

    outer_factor_metrics, outer_family_metrics, annual_metrics, decisions, outer_scored = _evaluate_outer(
        outer,
        selections=selections,
    )
    dataset = build_qlib_dataset(
        QlibProviderConfig(provider_uri=spec.provider_uri, market="all", region="us", kernels=1),
        instruments=universe,
        features=[f"${feature}" for feature in [*PRICE_ONLY_BASELINE_FEATURES, *all_features]],
        label="$breakdown_label",
        train=(spec.start, "2020-12-31"),
        valid=("2021-01-01", spec.development_end),
        test=(spec.outer_start, spec.as_of),
    )
    dataset_evidence = {
        "api": dataset.qlib_dataset_api,
        "segments": dataset.qlib_segments,
        "prepared_rows": {segment: int(len(dataset.prepare(segment))) for segment in ("train", "valid", "test")},
    }
    coverage = intraday_coverage_decision(
        core_start="2018-12-31",
        external_complete_start="2025-01-02",
        development_end=spec.development_end,
        outer_start=spec.outer_start,
    )
    return _write_card(
        spec,
        universe=universe,
        underlying_groups=underlying_groups,
        daily_source_audit=daily_source_audit,
        structure_audit=structure_audit,
        coverage=coverage,
        provider_manifest=provider_manifest,
        dataset_evidence=dataset_evidence,
        selections=selections,
        factor_summary=pd.concat(summaries, ignore_index=True),
        group_metrics=pd.concat(group_tables, ignore_index=True),
        correlations=pd.concat(correlations, ignore_index=True),
        outer_factor_metrics=outer_factor_metrics,
        outer_family_metrics=outer_family_metrics,
        annual_metrics=annual_metrics,
        decisions=decisions,
        development=development,
        soxl_development=soxl_development,
        outer_scored=outer_scored,
    )


def _write_card(spec: SoxlExhaustionStructureFragilitySpec, **artifacts: Any) -> dict[str, Any]:
    card = spec.card_dir
    card.mkdir(parents=True, exist_ok=True)
    for name in (
        "factor_summary",
        "group_metrics",
        "correlations",
        "outer_factor_metrics",
        "outer_family_metrics",
        "annual_metrics",
    ):
        artifacts[name].to_csv(card / f"{name}.csv", index=False)
    coverage_rows = []
    for source, values in artifacts["coverage"].items():
        coverage_rows.append({"source": source, **values})
    pd.DataFrame(coverage_rows).to_csv(card / "coverage_audit.csv", index=False)

    (card / "workflow.yaml").write_text(
        f"experiment: {spec.experiment}\n"
        f"provider_uri: {spec.provider_uri.resolve().as_posix()}\n"
        "method: transparent_factor_only_no_predictive_model\n"
        "factor_direction: higher_means_more_breakdown_pressure\n"
        f"development_end: {spec.development_end}\n"
        f"outer_soxl_start: {spec.outer_start}\n"
        "qlib_api:\n  - qlib.data.D.features\n  - DatasetH\n  - qlib.workflow.R\n  - Recorder\n"
        "artifact_policy: slim_card_recorder_first\n",
        encoding="utf-8",
    )
    overall = artifacts["outer_family_metrics"].set_index("family")
    price = overall.loc["price_only"]
    family_lines = []
    for family in FAMILY_FEATURES:
        selected = artifacts["selections"][family]
        if family in overall.index:
            row = overall.loc[family]
            family_lines.append(
                f"- {family}: selected={','.join(selected)}; AUC={row['auc']:.3f}; AP lift={row['average_precision_lift']:.2f}x; "
                f"Q5-Q1={row['quintile_lift']:.1%}; conditional={row['conditional_event_rate_lift']:.1%}; "
                f"decision={artifacts['decisions'][family]['decision']}"
            )
        else:
            family_lines.append(
                f"- {family}: selected=none; decision={artifacts['decisions'][family]['decision']}"
            )
    (card / "conclusion.md").write_text(
        "# 结论\n\n"
        "- 执行内容：不用预测模型，分别检验价格路径耗竭、杠杆脆弱性、固定五龙头结构代理和SOXL/SOXX盘中路径。\n"
        f"- 基线：2023至2026 SOXL高位样本 price-only AUC {price['auc']:.3f}、AP lift {price['average_precision_lift']:.2f}x、Q5-Q1 {price['quintile_lift']:.1%}。\n"
        + "\n".join(family_lines)
        + "\n- 晋升决定：只有同时改善基线AUC、AP lift、Q5-Q1并在price-only分层内仍有正增量的家族才保留。\n"
        "- 风险：固定五龙头是代理组合而非历史成分股；外部龙头15分钟历史从2025起，已阻断其长期筛选；2025/2026事件较少时年度值仅作诊断。\n"
        "- 下一步：只把通过增量门槛的因子接入降仓/波动控制；失败家族停止继续调模型。\n",
        encoding="utf-8",
    )
    (card / "failure_modes.md").write_text(
        "# Failure Modes\n\n"
        "- fixed_bellwether_proxy: 当前固定五龙头回填历史存在生存偏差，不代表点时SOXX持仓。\n"
        "- external_intraday: 2025-01-02前没有完整五龙头15分钟覆盖，状态为blocked_insufficient_history。\n"
        "- sparse_outer_years: 年度事件少于5个时不用于方向一致性晋升。\n"
        "- factor_only: 结果衡量breakdown辨识力，不等同于含成本仓位收益。\n",
        encoding="utf-8",
    )

    recorder_uri = card / "qlib_recorder"
    recorder_uri.mkdir(parents=True, exist_ok=True)
    from qlib.workflow import R

    R.set_uri(recorder_uri.resolve().as_posix())
    with tempfile.TemporaryDirectory(prefix=f"{spec.experiment}_") as temp_dir:
        temp = Path(temp_dir)
        recorder_frames = {
            "development_non_soxl_rows.csv": artifacts["development"],
            "development_soxl_rows.csv": artifacts["soxl_development"],
            "outer_soxl_scored_rows.csv": artifacts["outer_scored"],
            "factor_summary.csv": artifacts["factor_summary"],
            "outer_factor_metrics.csv": artifacts["outer_factor_metrics"],
            "outer_family_metrics.csv": artifacts["outer_family_metrics"],
        }
        for name, frame in recorder_frames.items():
            frame.to_csv(temp / name, index=False)
        with R.start(experiment_name=spec.experiment):
            recorder = R.get_recorder()
            R.log_params(
                method="transparent_factor_only_no_predictive_model",
                development_end=spec.development_end,
                outer_soxl_start=spec.outer_start,
                factor_direction="higher_means_more_breakdown_pressure",
                qlib_docs=QLIB_DOCS["data"],
            )
            for path in temp.iterdir():
                recorder.log_artifact(path.as_posix(), artifact_path=spec.experiment)
            summary = {
                "experiment": spec.experiment,
                "card_type": CARD_TYPE,
                "status": "completed",
                "as_of": spec.as_of,
                "method": "transparent_factor_only_no_predictive_model",
                "research_universe": artifacts["universe"],
                "underlying_groups": artifacts["underlying_groups"],
                "qlib_contract": {
                    "docs": QLIB_DOCS,
                    "data_api": ["qlib.init", "qlib.data.D.features"],
                    "dataset_api": ["DatasetH", "DataHandlerLP", "QlibDataLoader"],
                    "workflow_api": ["qlib.workflow.R", "Recorder"],
                    "dataset_evidence": artifacts["dataset_evidence"],
                },
                "data_lineage": {
                    "provider_manifest": artifacts["provider_manifest"],
                    "daily_source_audit": artifacts["daily_source_audit"],
                    "fixed_bellwether_audit": artifacts["structure_audit"],
                },
                "data_quality": {
                    "daily": artifacts["daily_source_audit"]["quality"],
                    "fixed_bellwether": artifacts["structure_audit"],
                    "intraday_coverage": artifacts["coverage"],
                },
                "factor_contract": {
                    "direction": "higher_means_more_breakdown_pressure",
                    "families": FAMILY_FEATURES,
                    "selected_factors": artifacts["selections"],
                    "label": "20_session_volatility_scaled_breakdown",
                    "subset": "range_pos_252_gte_0_70",
                    "fixed_bellwether_membership": "fixed_bellwether_proxy_not_historical_constituents",
                    "external_intraday_selection": "blocked_insufficient_history",
                },
                "benchmark": "price_only_equal_weight_composite",
                "family_decisions": artifacts["decisions"],
                "promotion_decision": (
                    "retained_incremental_factor_evidence"
                    if any(value["all_passed"] for value in artifacts["decisions"].values())
                    else "not_retained_no_incremental_factor_evidence"
                ),
                "recorder": {
                    "experiment": spec.experiment,
                    "recorder_id": recorder.id,
                    "recorder_uri": recorder_uri.resolve().as_posix(),
                    "local_dir": Path(recorder.get_local_dir()).resolve().as_posix(),
                },
                "artifact_policy": {
                    "policy": "slim_card_recorder_first",
                    "workflow_config": "workflow.yaml",
                    "recorder_only_artifacts": list(recorder_frames),
                },
                "compact_sidecars": [
                    "factor_summary.csv",
                    "group_metrics.csv",
                    "correlations.csv",
                    "outer_factor_metrics.csv",
                    "outer_family_metrics.csv",
                    "annual_metrics.csv",
                    "coverage_audit.csv",
                    "failure_modes.md",
                ],
            }
            (card / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
            for name in summary["compact_sidecars"] + ["workflow.yaml", "summary.json", "conclusion.md"]:
                recorder.log_artifact((card / name).as_posix(), artifact_path=spec.experiment)
    return json.loads((card / "summary.json").read_text(encoding="utf-8"))
