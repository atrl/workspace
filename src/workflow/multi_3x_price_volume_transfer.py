"""Qlib-native factor-only transfer diagnostics for multi-3x price-volume pressure."""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.data import (
    QlibProviderConfig,
    build_qlib_dataset,
    build_qlib_store_from_frame,
    load_qlib_feature_frame,
)
from src.factor import (
    ALL_FACTOR_FEATURES,
    PRICE_ONLY_BASELINE_FEATURES,
    PRICE_VOLUME_PRESSURE_FEATURES,
    build_multi_3x_price_volume_frame,
)


EXPERIMENT = "multi_3x_price_volume_transfer_v1"
CARD_TYPE = "multi_3x_price_volume_transfer_research_card"
QLIB_DOCS = "https://qlib.readthedocs.io/en/latest/"


@dataclass(frozen=True)
class Multi3xPriceVolumeTransferSpec:
    root: Path
    experiment: str = EXPERIMENT
    as_of: str = "2026-07-02"
    start: str = "2011-01-03"
    development_end: str = "2022-12-31"
    outer_start: str = "2023-01-01"
    label_horizon: int = 20
    normalization_window: int = 252
    min_periods: int = 60
    min_group_observations: int = 120
    min_group_events: int = 10

    @property
    def card_dir(self) -> Path:
        return self.root / "research" / self.experiment

    @property
    def provider_uri(self) -> Path:
        return self.root / "data" / "qlib" / self.experiment


def _quintile_metrics(labels: pd.Series, scores: pd.Series) -> dict[str, float]:
    work = pd.DataFrame({"label": labels.astype(float), "score": scores.astype(float)}).dropna()
    if len(work) < 5 or work["score"].nunique() < 2:
        return {
            "quintile_1_rate": np.nan,
            "quintile_5_rate": np.nan,
            "quintile_lift": np.nan,
            "quintile_monotonicity": np.nan,
        }
    bins = pd.qcut(work["score"].rank(method="first"), q=5, labels=False) + 1
    rates = work.assign(quintile=bins).groupby("quintile")["label"].mean().reindex(range(1, 6))
    return {
        "quintile_1_rate": float(rates.loc[1]),
        "quintile_5_rate": float(rates.loc[5]),
        "quintile_lift": float(rates.loc[5] - rates.loc[1]),
        "quintile_monotonicity": float(rates.corr(pd.Series(range(1, 6), index=rates.index), method="spearman")),
    }


def _diagnostic_metrics(labels: pd.Series, scores: pd.Series) -> dict[str, float]:
    work = pd.DataFrame({"label": labels, "score": scores}).dropna()
    prevalence = float(work["label"].mean()) if len(work) else np.nan
    if len(work) == 0 or work["label"].nunique() < 2 or work["score"].nunique() < 2:
        return {
            "auc": np.nan,
            "average_precision": np.nan,
            "prevalence": prevalence,
            "average_precision_lift": np.nan,
        } | _quintile_metrics(work["label"], work["score"])
    auc = float(roc_auc_score(work["label"], work["score"]))
    average_precision = float(average_precision_score(work["label"], work["score"]))
    return {
        "auc": auc,
        "average_precision": average_precision,
        "prevalence": prevalence,
        "average_precision_lift": float(average_precision / prevalence) if prevalence > 0.0 else np.nan,
    } | _quintile_metrics(work["label"], work["score"])


def factor_group_diagnostics(
    frame: pd.DataFrame,
    *,
    factors: list[str],
    min_observations: int = 120,
    min_events: int = 10,
) -> pd.DataFrame:
    """Evaluate fixed-direction factors separately for every underlying group."""

    required = {"underlying_group", "breakdown_label", "high_level_state", *factors}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"diagnostic frame missing columns: {missing}")
    rows: list[dict[str, Any]] = []
    high_level = frame.loc[frame["high_level_state"].eq(1.0)].copy()
    for group, group_frame in high_level.groupby("underlying_group", sort=True):
        for factor in factors:
            sample = group_frame[["breakdown_label", factor]].dropna()
            observations = int(len(sample))
            events = int(sample["breakdown_label"].sum())
            eligible = bool(
                observations >= min_observations
                and events >= min_events
                and events <= observations - min_events
                and sample[factor].nunique() >= 2
            )
            base = {
                "underlying_group": str(group),
                "factor": factor,
                "observations": observations,
                "breakdown_events": events,
                "eligible": eligible,
            }
            if not eligible:
                rows.append(
                    base
                    | {
                        "status": "blocked_insufficient_group_sample",
                        "auc": np.nan,
                        "average_precision": np.nan,
                        "prevalence": float(sample["breakdown_label"].mean()) if observations else np.nan,
                        "average_precision_lift": np.nan,
                        "quintile_1_rate": np.nan,
                        "quintile_5_rate": np.nan,
                        "quintile_lift": np.nan,
                        "quintile_monotonicity": np.nan,
                        "direction_pass": False,
                    }
                )
                continue
            metrics = _diagnostic_metrics(sample["breakdown_label"], sample[factor])
            direction_pass = bool(metrics["auc"] > 0.50 and metrics["quintile_lift"] > 0.0)
            rows.append(base | {"status": "ok"} | metrics | {"direction_pass": direction_pass})
    return pd.DataFrame(rows)


def select_transferable_factors(
    group_metrics: pd.DataFrame,
    *,
    correlation: pd.DataFrame,
    min_direction_consistency: float = 0.70,
    min_median_auc: float = 0.52,
    min_median_quintile_lift: float = 0.02,
    max_abs_correlation: float = 0.85,
) -> dict[str, Any]:
    """Apply fixed transfer gates and a deterministic redundancy filter."""

    required = {"factor", "eligible", "direction_pass", "auc", "quintile_lift"}
    missing = sorted(required.difference(group_metrics.columns))
    if missing:
        raise ValueError(f"group metrics missing columns: {missing}")
    summaries: list[dict[str, Any]] = []
    for factor, rows in group_metrics.groupby("factor", sort=True):
        eligible = rows.loc[rows["eligible"].astype(bool)]
        group_count = int(len(eligible))
        consistency = float(eligible["direction_pass"].mean()) if group_count else 0.0
        median_auc = float(eligible["auc"].median()) if group_count else np.nan
        median_lift = float(eligible["quintile_lift"].median()) if group_count else np.nan
        passed = bool(
            group_count > 0
            and consistency >= min_direction_consistency
            and median_auc >= min_median_auc
            and median_lift >= min_median_quintile_lift
        )
        summaries.append(
            {
                "factor": factor,
                "eligible_group_count": group_count,
                "direction_consistency": consistency,
                "median_group_auc": median_auc,
                "median_group_quintile_lift": median_lift,
                "development_gate_passed": passed,
                "selection_status": "failed_development_gate",
            }
        )
    summary = pd.DataFrame(summaries)
    candidates = summary.loc[summary["development_gate_passed"]].sort_values(
        ["median_group_auc", "median_group_quintile_lift", "factor"],
        ascending=[False, False, True],
        kind="mergesort",
    )
    selected: list[str] = []
    for factor in candidates["factor"]:
        redundant = any(
            selected_factor in correlation.index
            and factor in correlation.index
            and np.isfinite(correlation.loc[factor, selected_factor])
            and abs(float(correlation.loc[factor, selected_factor])) > max_abs_correlation
            for selected_factor in selected
        )
        status = "redundant" if redundant else "selected"
        summary.loc[summary["factor"].eq(factor), "selection_status"] = status
        if not redundant:
            selected.append(str(factor))
    return {
        "selected_factors": selected,
        "factor_summary": summary.sort_values("factor").reset_index(drop=True),
        "selection_uses_soxl": False,
        "selection_method": "fixed_direction_group_gate_then_spearman_redundancy",
    }


def split_development_outer(
    frame: pd.DataFrame,
    *,
    development_end: str = "2022-12-31",
    outer_start: str = "2023-01-01",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.to_datetime(frame["datetime"])
    development = frame.loc[
        frame["instrument"].ne("SOXL") & dates.le(pd.Timestamp(development_end))
    ].copy()
    outer = frame.loc[
        frame["instrument"].eq("SOXL") & dates.ge(pd.Timestamp(outer_start))
    ].copy()
    return development.reset_index(drop=True), outer.reset_index(drop=True)


def _window_composite_metrics(frame: pd.DataFrame, *, score_column: str) -> dict[str, Any]:
    sample = frame.loc[frame["high_level_state"].eq(1.0), ["breakdown_label", score_column]].dropna()
    return {
        "observations": int(len(sample)),
        "breakdown_events": int(sample["breakdown_label"].sum()) if len(sample) else 0,
    } | _diagnostic_metrics(sample["breakdown_label"], sample[score_column])


def evaluate_outer_composites(
    frame: pd.DataFrame,
    *,
    selected_factors: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate frozen equal-weight composites on outer SOXL rows."""

    if not selected_factors:
        raise ValueError("selected_factors must not be empty for outer evaluation")
    missing = sorted(set(PRICE_ONLY_BASELINE_FEATURES + selected_factors).difference(frame.columns))
    if missing:
        raise ValueError(f"outer frame missing composite columns: {missing}")
    scored = frame.copy()
    scored["price_only_score"] = scored[PRICE_ONLY_BASELINE_FEATURES].mean(axis=1, skipna=False)
    scored["price_volume_score"] = scored[selected_factors].mean(axis=1, skipna=False)
    rows: list[dict[str, Any]] = []
    windows: list[tuple[str, pd.DataFrame]] = [("overall", scored)]
    dates = pd.to_datetime(scored["datetime"])
    windows.extend(
        (str(year), scored.loc[dates.dt.year.eq(year)])
        for year in sorted(dates.dt.year.unique())
    )
    for window, window_frame in windows:
        for composite, score_column in [
            ("price_only", "price_only_score"),
            ("price_volume", "price_volume_score"),
        ]:
            rows.append(
                {"window": window, "composite": composite}
                | _window_composite_metrics(window_frame, score_column=score_column)
            )
    return pd.DataFrame(rows), scored


def evaluate_transfer_decision(
    *,
    selected_factors: list[str],
    outer_metrics: pd.DataFrame,
    eligible_development_groups: int,
) -> dict[str, Any]:
    if eligible_development_groups <= 0:
        raise ValueError("eligible development groups must be positive")
    if not selected_factors:
        return {
            "decision": "not_retained_no_transferable_price_volume_evidence",
            "all_passed": False,
            "gates": {"selected_factor_count": {"passed": False, "value": 0, "threshold": 1}},
        }
    if outer_metrics.empty:
        return {
            "decision": "not_retained_no_transferable_price_volume_evidence",
            "all_passed": False,
            "gates": {"outer_evidence": {"passed": False}},
        }
    overall = outer_metrics.loc[outer_metrics["window"].eq("overall")].set_index("composite")
    if not {"price_only", "price_volume"}.issubset(overall.index):
        raise ValueError("outer metrics missing price_only or price_volume overall rows")
    price = overall.loc["price_only"]
    volume = overall.loc["price_volume"]
    annual_volume = outer_metrics.loc[
        outer_metrics["composite"].eq("price_volume") & outer_metrics["window"].ne("overall")
    ]
    eligible_years = annual_volume.loc[
        annual_volume["auc"].notna()
        & annual_volume["quintile_lift"].notna()
        & annual_volume["observations"].ge(50)
        & annual_volume["breakdown_events"].ge(5)
        & annual_volume["observations"].sub(annual_volume["breakdown_events"]).ge(5)
    ]
    positive_years = int(
        (eligible_years["auc"].gt(0.50) & eligible_years["quintile_lift"].gt(0.0)).sum()
    )
    required_years = min(3, int(len(eligible_years))) if len(eligible_years) else 1
    gates = {
        "selected_factor_count": {"passed": len(selected_factors) >= 1, "value": len(selected_factors), "threshold": 1},
        "outer_auc": {"passed": float(volume["auc"]) > 0.50, "value": float(volume["auc"]), "threshold": 0.50},
        "outer_ap_lift_increment": {
            "passed": float(volume["average_precision_lift"] - price["average_precision_lift"]) > 0.0,
            "value": float(volume["average_precision_lift"] - price["average_precision_lift"]),
            "threshold": 0.0,
        },
        "outer_quintile_lift_increment": {
            "passed": float(volume["quintile_lift"] - price["quintile_lift"]) >= 0.02,
            "value": float(volume["quintile_lift"] - price["quintile_lift"]),
            "threshold": 0.02,
        },
        "annual_direction_consistency": {
            "passed": positive_years >= required_years,
            "positive_years": positive_years,
            "eligible_years": int(len(eligible_years)),
            "threshold": required_years,
        },
    }
    all_passed = all(bool(item["passed"]) for item in gates.values())
    return {
        "decision": (
            "retained_for_risk_control_followup"
            if all_passed
            else "not_retained_no_transferable_price_volume_evidence"
        ),
        "all_passed": all_passed,
        "gates": gates,
    }


def build_price_volume_dataset(
    provider_uri: Path | str,
    *,
    instruments: list[str],
    features: list[str],
    train: tuple[str, str],
    valid: tuple[str, str],
    test: tuple[str, str],
):
    return build_qlib_dataset(
        QlibProviderConfig(provider_uri=Path(provider_uri), market="all", region="us", kernels=1),
        instruments=instruments,
        features=[name if name.startswith("$") else f"${name}" for name in features],
        label="$breakdown_label",
        train=train,
        valid=valid,
        test=test,
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


def run_multi_3x_price_volume_transfer_card(*, root: Path | str | None = None) -> dict[str, Any]:
    spec = Multi3xPriceVolumeTransferSpec(root=_resolve_root(root))
    panel, source_audit, universe, underlying_groups = _load_source_panel(spec)
    featured = build_multi_3x_price_volume_frame(
        panel,
        label_horizon=spec.label_horizon,
        normalization_window=spec.normalization_window,
        min_periods=spec.min_periods,
    )
    derived_fields = [
        "instrument",
        "datetime",
        *ALL_FACTOR_FEATURES,
        "breakdown_label",
        "high_level_state",
    ]
    provider_manifest = build_qlib_store_from_frame(
        featured[derived_fields],
        provider_uri=spec.provider_uri,
        as_of=spec.as_of,
        source="qlib.data.D.features paired 3x ETF and underlying providers",
    )
    return _execute_card(
        spec,
        provider_manifest=provider_manifest,
        source_audit=source_audit,
        universe=universe,
        underlying_groups=underlying_groups,
    )


def _load_source_panel(
    spec: Multi3xPriceVolumeTransferSpec,
) -> tuple[pd.DataFrame, dict[str, Any], list[str], dict[str, str]]:
    from src.workflow.multi_3x_etf_normalized_nested_wf import _load_pair_frame
    from src.workflow.multi_3x_high_chop_risk_control import RESEARCH_ETFS, UNDERLYING_GROUPS

    frames: list[pd.DataFrame] = []
    audits: dict[str, Any] = {}
    for etf in RESEARCH_ETFS:
        provider_uri = spec.root / "data" / "qlib" / etf.lower()
        frame = _load_pair_frame(
            QlibProviderConfig(provider_uri=provider_uri, market="all", region="us", kernels=1),
            etf=etf,
            underlying=UNDERLYING_GROUPS[etf],
            start=spec.start,
            end=spec.as_of,
        )
        frames.append(frame)
        audits[etf] = {
            "provider_uri": provider_uri.resolve().as_posix(),
            "underlying": UNDERLYING_GROUPS[etf],
            "row_count": int(len(frame)),
            "min_datetime": pd.Timestamp(frame["datetime"].min()).date().isoformat(),
            "max_datetime": pd.Timestamp(frame["datetime"].max()).date().isoformat(),
            "source_api": "qlib.data.D.features",
        }
    panel = pd.concat(frames, ignore_index=True)
    common_end = pd.Timestamp(panel.groupby("instrument")["datetime"].max().min())
    panel = panel.loc[pd.to_datetime(panel["datetime"]).le(common_end)].copy()
    nonpositive_volume = panel["volume"].le(0.0)
    source_quality = {
        "grain": "one ETF trading session per instrument/datetime",
        "row_count": int(len(panel)),
        "instrument_count": int(panel["instrument"].nunique()),
        "duplicate_instrument_datetime_rows": int(panel.duplicated(["instrument", "datetime"]).sum()),
        "required_field_null_count": int(
            panel[["open", "high", "low", "close", "volume", "underlying_close"]].isna().sum().sum()
        ),
        "nonpositive_volume_rows": int(nonpositive_volume.sum()),
        "nonpositive_volume_share": float(nonpositive_volume.mean()),
        "nonpositive_volume_by_instrument": {
            str(key): int(value)
            for key, value in panel.loc[nonpositive_volume].groupby("instrument").size().items()
        },
        "nonpositive_volume_outer_soxl_rows": int(
            (
                nonpositive_volume
                & panel["instrument"].eq("SOXL")
                & pd.to_datetime(panel["datetime"]).ge(pd.Timestamp(spec.outer_start))
            ).sum()
        ),
        "common_end": common_end.date().isoformat(),
    }
    return (
        panel.sort_values(["datetime", "instrument"]).reset_index(drop=True),
        {"instruments": audits, "quality": source_quality},
        list(RESEARCH_ETFS),
        dict(UNDERLYING_GROUPS),
    )


def _read_derived_provider(
    spec: Multi3xPriceVolumeTransferSpec,
    *,
    universe: list[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    fields = [
        *[f"${name}" for name in ALL_FACTOR_FEATURES],
        "$breakdown_label",
        "$high_level_state",
    ]
    loaded = load_qlib_feature_frame(
        QlibProviderConfig(provider_uri=spec.provider_uri, market="all", region="us", kernels=1),
        instruments=universe,
        fields=fields,
        start=spec.start,
        end=spec.as_of,
    )
    frame = loaded.reset_index().rename(columns={field: field.removeprefix("$") for field in fields})
    frame["datetime"] = pd.to_datetime(frame["datetime"]).dt.normalize()
    missing_features = [name for name in ALL_FACTOR_FEATURES if frame[name].dropna().empty]
    if missing_features:
        raise ValueError(f"qlib readback missing factors: {missing_features}")
    return frame, {
        "provider_uri": spec.provider_uri.resolve().as_posix(),
        "source_api": loaded.attrs["qlib_data_api"],
        "fields": fields,
        "row_count": int(len(frame)),
        "max_datetime": frame["datetime"].max().date().isoformat(),
        "missing_features": missing_features,
    }


def _execute_card(
    spec: Multi3xPriceVolumeTransferSpec,
    *,
    provider_manifest: dict[str, Any],
    source_audit: dict[str, Any],
    universe: list[str],
    underlying_groups: dict[str, str],
) -> dict[str, Any]:
    frame, readback_audit = _read_derived_provider(spec, universe=universe)
    frame["underlying_group"] = frame["instrument"].map(underlying_groups)
    development, outer = split_development_outer(
        frame,
        development_end=spec.development_end,
        outer_start=spec.outer_start,
    )
    group_metrics = factor_group_diagnostics(
        development,
        factors=list(PRICE_VOLUME_PRESSURE_FEATURES),
        min_observations=spec.min_group_observations,
        min_events=spec.min_group_events,
    )
    development_high = development.loc[development["high_level_state"].eq(1.0)]
    correlation = development_high[PRICE_VOLUME_PRESSURE_FEATURES].corr(method="spearman")
    selection = select_transferable_factors(group_metrics, correlation=correlation)
    eligible_groups = int(group_metrics.loc[group_metrics["eligible"], "underlying_group"].nunique())
    selected = selection["selected_factors"]
    if selected:
        outer_metrics, outer_scored = evaluate_outer_composites(outer, selected_factors=selected)
    else:
        outer_metrics = pd.DataFrame()
        outer_scored = outer.copy()
    decision = evaluate_transfer_decision(
        selected_factors=selected,
        outer_metrics=outer_metrics,
        eligible_development_groups=eligible_groups,
    )

    dataset = build_price_volume_dataset(
        spec.provider_uri,
        instruments=universe,
        features=list(ALL_FACTOR_FEATURES),
        train=(spec.start, "2020-12-31"),
        valid=("2021-01-01", spec.development_end),
        test=(spec.outer_start, spec.as_of),
    )
    dataset_evidence = {
        "qlib_dataset_api": dataset.qlib_dataset_api,
        "segments": dataset.qlib_segments,
        "prepared_rows": {name: int(len(dataset.prepare(name))) for name in ("train", "valid", "test")},
    }
    return _write_and_record_card(
        spec,
        universe=universe,
        underlying_groups=underlying_groups,
        provider_manifest=provider_manifest,
        source_audit=source_audit,
        readback_audit=readback_audit,
        dataset_evidence=dataset_evidence,
        development=development,
        outer_scored=outer_scored,
        group_metrics=group_metrics,
        factor_summary=selection["factor_summary"],
        correlation=correlation,
        outer_metrics=outer_metrics,
        selected_factors=selected,
        decision=decision,
    )


def _write_and_record_card(spec: Multi3xPriceVolumeTransferSpec, **artifacts: Any) -> dict[str, Any]:
    card = spec.card_dir
    card.mkdir(parents=True, exist_ok=True)
    factor_summary: pd.DataFrame = artifacts["factor_summary"]
    group_metrics: pd.DataFrame = artifacts["group_metrics"]
    correlation: pd.DataFrame = artifacts["correlation"]
    outer_metrics: pd.DataFrame = artifacts["outer_metrics"]
    selected_factors: list[str] = artifacts["selected_factors"]
    decision: dict[str, Any] = artifacts["decision"]

    factor_summary.to_csv(card / "factor_summary.csv", index=False)
    group_metrics.to_csv(card / "group_metrics.csv", index=False)
    correlation.rename_axis("factor").reset_index().to_csv(card / "factor_correlation.csv", index=False)
    outer_metrics.to_csv(card / "outer_test_metrics.csv", index=False)
    (card / "workflow.yaml").write_text(
        f"experiment: {spec.experiment}\n"
        f"provider_uri: {spec.provider_uri.resolve().as_posix()}\n"
        "data_api: qlib.data.D.features\n"
        "dataset_api:\n  - DatasetH\n  - DataHandlerLP\n  - QlibDataLoader\n"
        "workflow_api:\n  - qlib.workflow.R\n  - Recorder\n"
        f"development_end: {spec.development_end}\n"
        f"outer_soxl_start: {spec.outer_start}\n"
        "selection_uses_soxl: false\n"
        "factor_direction: higher_means_more_breakdown_pressure\n"
        "artifact_policy: slim_card_recorder_first\n",
        encoding="utf-8",
    )
    failed_gates = [name for name, gate in decision["gates"].items() if not gate["passed"]]
    (card / "failure_modes.md").write_text(
        "# Failure Modes\n\n"
        f"- decision: {decision['decision']}\n"
        f"- failed_gates: {', '.join(failed_gates) if failed_gates else 'none'}\n"
        "- factor_risk: 日频成交量无法识别盘中吸筹和派发顺序，拆股或数据源调整也可能改变量能分布。\n"
        "- data_risk: 早期低流动性产品存在少量零成交量记录；因子将其视为零相对量，且SOXL外层样本没有该问题。\n"
        "- transfer_risk: 3x ETF 的产品年龄、行业集中度和再平衡摩擦不同，跨标的一致不保证半导体周期有效。\n"
        "- target_risk: breakdown 标签是波动率缩放的20日路径结果，不等同于可交易收益或择时胜率。\n",
        encoding="utf-8",
    )
    if not outer_metrics.empty:
        overall = outer_metrics.loc[outer_metrics["window"].eq("overall")].set_index("composite")
        price = overall.loc["price_only"]
        volume = overall.loc["price_volume"]
        metric_text = (
            f"SOXL样本外 price-only AUC {price['auc']:.3f}、Q5-Q1 {price['quintile_lift']:.1%}；"
            f"price-volume AUC {volume['auc']:.3f}、Q5-Q1 {volume['quintile_lift']:.1%}、"
            f"AP lift {volume['average_precision_lift']:.2f}x。"
        )
    else:
        metric_text = "开发样本没有因子通过迁移门槛，因此按失败关闭规则未形成SOXL组合。"
    conclusion = (
        "# 结论\n\n"
        "- 执行内容：用非SOXL的长期3x ETF逐underlying检验8个点时量价压力因子，冻结后在2023至2026年SOXL上零样本揭盲。\n"
        f"- 关键指标：{metric_text}\n"
        f"- 保留因子：{', '.join(selected_factors) if selected_factors else '无'}。\n"
        f"- 晋升决定：{decision['decision']}；该结论只决定是否进入下一轮风险控制研究，不是交易策略晋升。\n"
        "- 风险：日频量价、ETF上市期差异、早期少量零成交量记录和20日路径标签可能限制泛化，未评估仓位收益与交易成本。\n"
        "- 下一步：只对跨组且SOXL增量门槛同时通过的因子做仓位节流实验；其余因子停止调模型。\n"
    )
    (card / "conclusion.md").write_text(conclusion, encoding="utf-8")

    recorder_uri = card / "qlib_recorder"
    recorder_uri.mkdir(parents=True, exist_ok=True)
    from qlib.workflow import R

    R.set_uri(recorder_uri.resolve().as_posix())
    with tempfile.TemporaryDirectory(prefix="multi_3x_price_volume_") as tmp:
        temp = Path(tmp)
        recorder_files = {
            "development_factor_rows.csv": artifacts["development"],
            "outer_soxl_scored_rows.csv": artifacts["outer_scored"],
            "group_metrics.csv": group_metrics,
            "factor_summary.csv": factor_summary,
            "outer_test_metrics.csv": outer_metrics,
        }
        for name, frame in recorder_files.items():
            frame.to_csv(temp / name, index=False)
        with R.start(experiment_name=spec.experiment):
            recorder = R.get_recorder()
            R.log_params(
                qlib_docs=QLIB_DOCS,
                provider_uri=spec.provider_uri.resolve().as_posix(),
                selection_uses_soxl="false",
                development_end=spec.development_end,
                outer_soxl_start=spec.outer_start,
                selected_factors=",".join(selected_factors),
                factor_direction="higher_means_more_breakdown_pressure",
            )
            for path in temp.iterdir():
                recorder.log_artifact(path.as_posix(), artifact_path=spec.experiment)
            summary = {
                "experiment": spec.experiment,
                "card_type": CARD_TYPE,
                "status": "completed",
                "as_of": spec.as_of,
                "research_universe": artifacts["universe"],
                "underlying_groups": artifacts["underlying_groups"],
                "qlib_contract": {
                    "docs": QLIB_DOCS,
                    "data_api": ["qlib.init", "qlib.data.D.features"],
                    "dataset_api": ["DatasetH", "DataHandlerLP", "QlibDataLoader"],
                    "workflow_api": ["qlib.workflow.R", "Recorder"],
                    "dataset_evidence": artifacts["dataset_evidence"],
                    "provider_readback": artifacts["readback_audit"],
                },
                "data_lineage": {
                    "provider_manifest": artifacts["provider_manifest"],
                    "source_audit": artifacts["source_audit"],
                },
                "data_quality": artifacts["source_audit"]["quality"],
                "factor_contract": {
                    "direction": "higher_means_more_breakdown_pressure",
                    "price_only_baseline": list(PRICE_ONLY_BASELINE_FEATURES),
                    "price_volume_candidates": list(PRICE_VOLUME_PRESSURE_FEATURES),
                    "selected_factors": selected_factors,
                    "selection_uses_soxl": False,
                    "label": "20_session_volatility_scaled_breakdown",
                    "subset": "range_pos_252_gte_0_70",
                },
                "benchmark": "price_only_equal_weight_composite",
                "decision": decision,
                "promotion_decision": decision["decision"],
                "recorder": {
                    "experiment": spec.experiment,
                    "recorder_id": recorder.id,
                    "recorder_uri": recorder_uri.resolve().as_posix(),
                    "local_dir": Path(recorder.get_local_dir()).resolve().as_posix(),
                },
                "artifact_policy": {
                    "policy": "slim_card_recorder_first",
                    "workflow_config": "workflow.yaml",
                    "recorder_only_artifacts": list(recorder_files),
                },
                "compact_sidecars": [
                    "factor_summary.csv",
                    "group_metrics.csv",
                    "outer_test_metrics.csv",
                    "factor_correlation.csv",
                    "failure_modes.md",
                ],
            }
            (card / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
            for name in summary["compact_sidecars"] + ["workflow.yaml", "summary.json", "conclusion.md"]:
                path = card / name
                if path.exists():
                    recorder.log_artifact(path.as_posix(), artifact_path=spec.experiment)
    return json.loads((card / "summary.json").read_text(encoding="utf-8"))
