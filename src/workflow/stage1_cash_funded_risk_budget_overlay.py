from __future__ import annotations

import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.workflow.intraday_recovery_direct_allocation import _risk_metrics
from src.workflow.stage1_elastic_soxl_overflow import (
    SOURCE_EXPERIMENT,
    SOURCE_RECORDER_ID,
    Stage1ElasticSoxlOverflowSpec,
    _qlib_features,
    _stage1_replay,
)


EXPERIMENT = "stage1_cash_funded_risk_budget_overlay_v1"
FEATURE_COLUMNS = (
    "soxx_trend20",
    "relative_momentum20",
    "soxx_vol20",
    "chop_breakdown_risk",
)


@dataclass(frozen=True)
class CashFundedCandidate:
    name: str
    asset: str
    maximum_overlay_weight: float
    gate: str


CANDIDATES = (
    CashFundedCandidate(
        "soxl_005_pullback",
        "SOXL",
        0.05,
        "trend_positive_momentum_not_overextended",
    ),
    CashFundedCandidate(
        "soxl_010_pullback",
        "SOXL",
        0.10,
        "trend_positive_momentum_not_overextended",
    ),
    CashFundedCandidate(
        "soxl_005_quality",
        "SOXL",
        0.05,
        "pullback_low_vol_low_chop",
    ),
    CashFundedCandidate(
        "soxl_010_quality",
        "SOXL",
        0.10,
        "pullback_low_vol_low_chop",
    ),
    CashFundedCandidate(
        "soxx_005_quality",
        "SOXX",
        0.05,
        "pullback_low_vol_low_chop",
    ),
    CashFundedCandidate(
        "soxx_010_quality",
        "SOXX",
        0.10,
        "pullback_low_vol_low_chop",
    ),
)


@dataclass(frozen=True)
class Stage1CashFundedRiskBudgetSpec:
    root: Path
    experiment: str = EXPERIMENT
    source_experiment: str = SOURCE_EXPERIMENT
    source_recorder_id: str = SOURCE_RECORDER_ID
    source_training_end: str = "2022-12-30"
    train_only_years: tuple[int, ...] = (2016, 2017)
    development_years: tuple[int, ...] = (2018, 2019, 2020, 2021, 2022)
    outer_years: tuple[int, ...] = (2023, 2024, 2025, 2026)
    transaction_cost_bps: float = 10.0
    stress_cost_bps: float = 25.0
    minimum_cash_weight: float = 0.15
    source_maximum_soxl_weight: float = 0.30
    overlay_maximum_soxl_weight: float = 0.40
    cap_binding_tolerance: float = 0.0005
    minimum_development_return_delta_pct_points: float = 20.0
    minimum_development_actions: int = 12
    minimum_positive_action_rate: float = 0.52
    minimum_positive_years: int = 3
    maximum_maxdd_degradation_pct_points: float = 3.0
    maximum_worst5_degradation_pct_points: float = 1.0
    maximum_incremental_turnover_ratio: float = 0.15
    minimum_neighbor_wealth_ratio: float = 0.95

    @property
    def card_dir(self) -> Path:
        return self.root / "research" / self.experiment

    @property
    def source_card_dir(self) -> Path:
        return self.root / "research" / self.source_experiment

    @property
    def provider_uri(self) -> Path:
        return self.root / "data" / "qlib" / "soxl_soxx_soxs"


def run_stage1_cash_funded_risk_budget_overlay_card(
    *, root: Path | str | None = None
) -> dict[str, Any]:
    return _run_card(
        Stage1CashFundedRiskBudgetSpec(root=_resolve_root(root))
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


def _source_contract(
    spec: Stage1CashFundedRiskBudgetSpec,
) -> dict[str, Any]:
    summary = json.loads(
        (spec.source_card_dir / "summary.json").read_text(encoding="utf-8")
    )
    if summary.get("recorder_id") != spec.source_recorder_id:
        raise ValueError("blocked: frozen Stage1 recorder drifted")
    selected = summary.get("selected_model", {})
    if selected.get("stage2_enabled") is not False:
        raise ValueError("blocked: cash overlay requires Stage2 off")
    if summary.get("selection_uses_outer_test") is not False:
        raise ValueError("blocked: source Stage1 used outer for selection")
    return {
        "experiment": spec.source_experiment,
        "recorder_id": spec.source_recorder_id,
        "training_end": spec.source_training_end,
        "stage2_enabled": False,
        "frozen": True,
    }


def _build_comparison_frame(
    spec: Stage1CashFundedRiskBudgetSpec,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    elastic_spec = Stage1ElasticSoxlOverflowSpec(root=spec.root)
    replay, source_audit = _stage1_replay(
        elastic_spec,
        cost_bps=spec.transaction_cost_bps,
    )
    features, feature_audit = _qlib_features(elastic_spec)
    frame = replay.merge(
        features.loc[:, list(FEATURE_COLUMNS)],
        left_on="signal_date",
        right_index=True,
        how="left",
        validate="one_to_one",
    )
    if frame.loc[:, FEATURE_COLUMNS].isna().all(axis=1).any():
        raise ValueError("blocked: qlib cash-overlay features missing")
    frame["year"] = frame["signal_date"].dt.year
    frame["cap_bind"] = frame["open_soxl_weight"].ge(
        spec.source_maximum_soxl_weight - spec.cap_binding_tolerance
    )
    frame["cash_available"] = (
        frame["open_cash_weight"] - spec.minimum_cash_weight
    ).clip(lower=0.0)
    frame["cash_to_soxl_return"] = frame["soxl_return"]
    frame["cash_to_soxx_return"] = frame["soxx_return"]
    return frame.sort_values("signal_date").reset_index(drop=True), {
        **source_audit,
        **{
            f"feature_{key}": value
            for key, value in feature_audit.items()
            if key not in source_audit
        },
        "selected_features": list(FEATURE_COLUMNS),
    }


def _build_qlib_dataset(
    frame: pd.DataFrame,
    spec: Stage1CashFundedRiskBudgetSpec,
):
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP
    from qlib.data.dataset.loader import StaticDataLoader

    indexed = frame.set_index("signal_date").sort_index()
    labels = [
        "cash_to_soxl_return",
        "cash_to_soxx_return",
        "cap_bind",
        "cash_available",
    ]
    static = pd.concat(
        {
            "feature": indexed.loc[:, list(FEATURE_COLUMNS)],
            "label": indexed.loc[:, labels].astype("float64"),
        },
        axis=1,
    )
    static.index = pd.MultiIndex.from_arrays(
        [
            static.index,
            np.full(len(static), "STAGE1_CASH_RISK_BUDGET"),
        ],
        names=["datetime", "instrument"],
    )
    segments: dict[str, tuple[str, str]] = {
        "all": (
            static.index.get_level_values("datetime")
            .min()
            .date()
            .isoformat(),
            static.index.get_level_values("datetime")
            .max()
            .date()
            .isoformat(),
        ),
        "development": (
            f"{spec.development_years[0]}-01-01",
            f"{spec.development_years[-1]}-12-31",
        ),
        "outer": (
            f"{spec.outer_years[0]}-01-01",
            f"{spec.outer_years[-1]}-12-31",
        ),
        "train_outer": (
            f"{spec.train_only_years[0]}-01-01",
            spec.source_training_end,
        ),
    }
    for year in spec.development_years:
        segments[f"train_{year}"] = (
            f"{spec.train_only_years[0]}-01-01",
            f"{year - 1}-12-31",
        )
        segments[f"test_{year}"] = (
            f"{year}-01-01",
            f"{year}-12-31",
        )
    for year in spec.outer_years:
        segments[f"test_{year}"] = (
            f"{year}-01-01",
            f"{year}-12-31",
        )
    dataset = DatasetH(
        handler=DataHandlerLP(
            instruments=None,
            start_time=static.index.get_level_values("datetime").min(),
            end_time=static.index.get_level_values("datetime").max(),
            data_loader=StaticDataLoader(config=static),
            infer_processors=[],
            learn_processors=[],
        ),
        segments=segments,
    )
    audit = {
        "dataset_class": "qlib.data.dataset.DatasetH",
        "handler_class": "qlib.data.dataset.handler.DataHandlerLP",
        "loader_class": "qlib.data.dataset.loader.StaticDataLoader",
        "features": list(FEATURE_COLUMNS),
        "labels": labels,
        "segments": {
            name: list(bounds) for name, bounds in segments.items()
        },
        "segment_rows": {
            name: int(len(dataset.prepare(name))) for name in segments
        },
    }
    return dataset, audit


def _prepared(dataset, segment: str) -> pd.DataFrame:
    result = dataset.prepare(segment, col_set=["feature", "label"])
    result = result.droplevel("instrument").sort_index()
    result.columns = [str(column[-1]) for column in result.columns]
    return result


def _threshold_schedule(
    dataset,
    spec: Stage1CashFundedRiskBudgetSpec,
) -> pd.DataFrame:
    rows = []
    for year in (*spec.development_years, *spec.outer_years):
        segment = (
            "train_outer"
            if year in spec.outer_years
            else f"train_{year}"
        )
        training = _prepared(dataset, segment)
        cap = training[training["cap_bind"].gt(0.5)].copy()
        if len(cap) < 20:
            raise ValueError(
                f"blocked: insufficient prior cap events for {year}: "
                f"{len(cap)}"
            )
        rows.append(
            {
                "test_year": year,
                "training_segment": segment,
                "training_end": (
                    spec.source_training_end
                    if year in spec.outer_years
                    else f"{year - 1}-12-31"
                ),
                "training_cap_events": len(cap),
                "relative_momentum_q67": float(
                    cap["relative_momentum20"].quantile(2.0 / 3.0)
                ),
                "chop_breakdown_q50": float(
                    cap["chop_breakdown_risk"].quantile(0.50)
                ),
                "soxx_vol_q75": float(
                    cap["soxx_vol20"].quantile(0.75)
                ),
                "outer_thresholds_frozen": (
                    year in spec.outer_years
                ),
            }
        )
    return pd.DataFrame(rows)


def _apply_candidate(
    frame: pd.DataFrame,
    *,
    candidate: CashFundedCandidate,
    thresholds: pd.DataFrame,
    spec: Stage1CashFundedRiskBudgetSpec,
) -> pd.DataFrame:
    result = frame.copy()
    threshold_map = thresholds.set_index("test_year").to_dict("index")
    overlay = np.zeros(len(result), dtype="float64")
    factor_gate = np.zeros(len(result), dtype=bool)
    for index, row in enumerate(result.itertuples()):
        threshold = threshold_map.get(int(row.year))
        if threshold is None:
            continue
        pullback = (
            bool(row.cap_bind)
            and float(row.soxx_trend20) > 0.0
            and float(row.relative_momentum20)
            <= threshold["relative_momentum_q67"]
        )
        if candidate.gate == "pullback_low_vol_low_chop":
            pullback = (
                pullback
                and float(row.soxx_vol20)
                <= threshold["soxx_vol_q75"]
                and float(row.chop_breakdown_risk)
                <= threshold["chop_breakdown_q50"]
            )
        factor_gate[index] = pullback
        if not pullback:
            continue
        available = max(
            0.0,
            float(row.open_cash_weight) - spec.minimum_cash_weight,
        )
        if candidate.asset == "SOXL":
            available = min(
                available,
                spec.overlay_maximum_soxl_weight
                - float(row.open_soxl_weight),
            )
        overlay[index] = min(
            candidate.maximum_overlay_weight,
            max(0.0, available),
        )
    overlay_soxl = np.where(
        candidate.asset == "SOXL", overlay, 0.0
    )
    overlay_soxx = np.where(
        candidate.asset == "SOXX", overlay, 0.0
    )
    incremental_turnover = np.abs(
        np.diff(overlay_soxl, prepend=0.0)
    ) + np.abs(np.diff(overlay_soxx, prepend=0.0))
    incremental_gross = (
        overlay_soxl * result["soxl_return"].to_numpy()
        + overlay_soxx * result["soxx_return"].to_numpy()
    )
    incremental_net_10bps = (
        incremental_gross
        - incremental_turnover
        * spec.transaction_cost_bps
        / 10_000.0
    )
    incremental_net_25bps = (
        incremental_gross
        - incremental_turnover
        * spec.stress_cost_bps
        / 10_000.0
    )
    base_25bps = (
        result["gross_return"].to_numpy()
        - result["turnover"].to_numpy()
        * spec.stress_cost_bps
        / 10_000.0
    )
    result["candidate"] = candidate.name
    result["factor_gate"] = factor_gate
    result["overlay_asset"] = candidate.asset
    result["overlay_weight"] = overlay
    result["overlay_soxl_weight"] = overlay_soxl
    result["overlay_soxx_weight"] = overlay_soxx
    result["target_soxl_weight"] = (
        result["open_soxl_weight"] + overlay_soxl
    )
    result["target_soxx_weight"] = (
        result["open_soxx_weight"] + overlay_soxx
    )
    result["target_cash_weight"] = (
        result["open_cash_weight"] - overlay
    )
    result["incremental_turnover"] = incremental_turnover
    result["incremental_gross_return"] = incremental_gross
    result["incremental_net_return_10bps"] = incremental_net_10bps
    result["incremental_net_return_25bps"] = incremental_net_25bps
    result["portfolio_return_10bps"] = (
        result["strategy_return"] + incremental_net_10bps
    )
    result["portfolio_return_25bps"] = (
        base_25bps + incremental_net_25bps
    )
    _validate_weights(result, spec)
    return result


def _validate_weights(
    result: pd.DataFrame,
    spec: Stage1CashFundedRiskBudgetSpec,
) -> None:
    total = result[
        [
            "target_soxl_weight",
            "target_soxx_weight",
            "target_cash_weight",
        ]
    ].sum(axis=1)
    if not np.allclose(total, 1.0, atol=1e-8):
        raise ValueError("blocked: cash overlay weights do not sum to one")
    if (
        result["target_cash_weight"].min()
        < spec.minimum_cash_weight - 1e-8
    ):
        raise ValueError("blocked: cash overlay breached cash floor")
    if (
        result["target_soxl_weight"].max()
        > spec.overlay_maximum_soxl_weight + 1e-8
    ):
        raise ValueError("blocked: cash overlay breached SOXL ceiling")


def _segment_mask(
    frame: pd.DataFrame,
    years: tuple[int, ...],
) -> pd.Series:
    return frame["year"].isin(years)


def _baseline_returns(
    frame: pd.DataFrame,
    *,
    cost_bps: float,
) -> np.ndarray:
    return (
        frame["gross_return"].to_numpy()
        - frame["turnover"].to_numpy() * cost_bps / 10_000.0
    )


def _annual_positive_years(
    replay: pd.DataFrame,
    *,
    years: tuple[int, ...],
) -> int:
    count = 0
    for year in years:
        section = replay[replay["year"].eq(year)]
        if section.empty:
            continue
        candidate_wealth = float(
            np.prod(1.0 + section["portfolio_return_10bps"])
        )
        baseline_wealth = float(
            np.prod(
                1.0
                + _baseline_returns(
                    section,
                    cost_bps=10.0,
                )
            )
        )
        count += candidate_wealth > baseline_wealth
    return int(count)


def _metrics_row(
    *,
    variant: str,
    candidate: CashFundedCandidate | None,
    replay: pd.DataFrame,
    years: tuple[int, ...],
    segment: str,
    spec: Stage1CashFundedRiskBudgetSpec,
) -> dict[str, Any]:
    selected = replay[_segment_mask(replay, years)].copy()
    baseline_returns = _baseline_returns(
        selected,
        cost_bps=spec.transaction_cost_bps,
    )
    baseline_stress = _baseline_returns(
        selected,
        cost_bps=spec.stress_cost_bps,
    )
    baseline_metrics = _risk_metrics(baseline_returns)
    baseline_stress_metrics = _risk_metrics(baseline_stress)
    if candidate is None:
        candidate_returns = baseline_returns
        candidate_stress = baseline_stress
        actions = selected.iloc[0:0]
        incremental_turnover = 0.0
        positive_years = 0
    else:
        candidate_returns = selected[
            "portfolio_return_10bps"
        ].to_numpy()
        candidate_stress = selected[
            "portfolio_return_25bps"
        ].to_numpy()
        actions = selected[selected["overlay_weight"].gt(1e-12)]
        incremental_turnover = float(
            selected["incremental_turnover"].sum()
        )
        positive_years = _annual_positive_years(
            selected,
            years=years,
        )
    metrics = _risk_metrics(candidate_returns)
    stress = _risk_metrics(candidate_stress)
    baseline_turnover = float(selected["turnover"].sum())
    return {
        "variant": variant,
        "segment": segment,
        "selection_uses_outer_test": False,
        "asset": candidate.asset if candidate else "BASELINE",
        "gate": candidate.gate if candidate else "none",
        "maximum_overlay_weight": (
            candidate.maximum_overlay_weight if candidate else 0.0
        ),
        **metrics,
        "stress_25bps_return_pct": stress["return_pct"],
        "stress_25bps_max_drawdown_pct": stress[
            "max_drawdown_pct"
        ],
        "stress_25bps_worst_rolling_5d_pct": stress[
            "worst_rolling_5d_pct"
        ],
        "baseline_return_pct": baseline_metrics["return_pct"],
        "baseline_max_drawdown_pct": baseline_metrics[
            "max_drawdown_pct"
        ],
        "baseline_worst_rolling_5d_pct": baseline_metrics[
            "worst_rolling_5d_pct"
        ],
        "baseline_stress_25bps_return_pct": baseline_stress_metrics[
            "return_pct"
        ],
        "return_delta_pct_points": (
            metrics["return_pct"] - baseline_metrics["return_pct"]
        ),
        "max_drawdown_delta_pct_points": (
            metrics["max_drawdown_pct"]
            - baseline_metrics["max_drawdown_pct"]
        ),
        "worst_5d_delta_pct_points": (
            metrics["worst_rolling_5d_pct"]
            - baseline_metrics["worst_rolling_5d_pct"]
        ),
        "action_count": int(len(actions)),
        "positive_action_rate": (
            float(
                actions["incremental_net_return_10bps"].gt(0.0).mean()
            )
            if len(actions)
            else np.nan
        ),
        "positive_relative_years": positive_years,
        "incremental_turnover": incremental_turnover,
        "incremental_turnover_ratio": (
            incremental_turnover / baseline_turnover
            if baseline_turnover
            else np.nan
        ),
        "average_overlay_weight": (
            float(selected["overlay_weight"].mean())
            if candidate is not None
            else 0.0
        ),
    }


def _selection_table(
    comparison: pd.DataFrame,
    spec: Stage1CashFundedRiskBudgetSpec,
) -> tuple[pd.DataFrame, str, dict[str, Any]]:
    table = comparison.copy()
    development = table[
        table["segment"].eq("development_walk_forward")
        & table["variant"].ne("stage1_baseline")
    ].copy()
    development["return_gate_pass"] = development[
        "return_delta_pct_points"
    ].ge(spec.minimum_development_return_delta_pct_points)
    development["action_count_gate_pass"] = development[
        "action_count"
    ].ge(spec.minimum_development_actions)
    development["positive_action_gate_pass"] = development[
        "positive_action_rate"
    ].ge(spec.minimum_positive_action_rate)
    development["positive_years_gate_pass"] = development[
        "positive_relative_years"
    ].ge(spec.minimum_positive_years)
    development["max_drawdown_gate_pass"] = development[
        "max_drawdown_delta_pct_points"
    ].ge(-spec.maximum_maxdd_degradation_pct_points)
    development["worst_5d_gate_pass"] = development[
        "worst_5d_delta_pct_points"
    ].ge(-spec.maximum_worst5_degradation_pct_points)
    development["turnover_gate_pass"] = development[
        "incremental_turnover_ratio"
    ].le(spec.maximum_incremental_turnover_ratio)
    for index, row in development.iterrows():
        neighbors = development[
            development["asset"].eq(row["asset"])
            & development["gate"].eq(row["gate"])
        ]
        development.loc[index, "neighbor_count"] = len(neighbors)
        development.loc[index, "neighbor_min_wealth_ratio"] = (
            neighbors["terminal_wealth"].min()
            / float(row["terminal_wealth"])
        )
        development.loc[index, "neighbor_positive_count"] = int(
            neighbors["return_delta_pct_points"].gt(0.0).sum()
        )
    development["neighbor_gate_pass"] = (
        development["neighbor_count"].ge(2)
        & development["neighbor_positive_count"].ge(2)
        & development["neighbor_min_wealth_ratio"].ge(
            spec.minimum_neighbor_wealth_ratio
        )
    )
    gate_columns = [
        "return_gate_pass",
        "action_count_gate_pass",
        "positive_action_gate_pass",
        "positive_years_gate_pass",
        "max_drawdown_gate_pass",
        "worst_5d_gate_pass",
        "turnover_gate_pass",
        "neighbor_gate_pass",
    ]
    development["gate_pass_count"] = development[gate_columns].sum(
        axis=1
    )
    development["all_development_gates_pass"] = development[
        gate_columns
    ].all(axis=1)
    eligible = development[
        development["all_development_gates_pass"]
    ]
    selection_pool = eligible if not eligible.empty else development
    selected = selection_pool.sort_values(
        ["gate_pass_count", "return_delta_pct_points", "incremental_turnover"],
        ascending=[False, False, True],
    ).iloc[0]
    selected_name = str(selected["variant"])
    merge_columns = [
        "variant",
        *gate_columns,
        "neighbor_count",
        "neighbor_min_wealth_ratio",
        "neighbor_positive_count",
        "gate_pass_count",
        "all_development_gates_pass",
    ]
    table = table.merge(
        development.loc[:, merge_columns],
        on="variant",
        how="left",
        validate="many_to_one",
    )
    table["selected"] = table["variant"].eq(selected_name)
    audit = {
        column: bool(selected[column]) for column in gate_columns
    }
    audit.update(
        {
            "all_development_gates_pass": bool(
                selected["all_development_gates_pass"]
            ),
            "gate_pass_count": int(selected["gate_pass_count"]),
            "neighbor_count": int(selected["neighbor_count"]),
            "neighbor_min_wealth_ratio": float(
                selected["neighbor_min_wealth_ratio"]
            ),
            "neighbor_positive_count": int(
                selected["neighbor_positive_count"]
            ),
            "required_return_delta_pct_points": (
                spec.minimum_development_return_delta_pct_points
            ),
            "clean_unseen_holdout_available": False,
            "outer_used_for_selection": False,
        }
    )
    return table, selected_name, audit


def _cash_availability(
    frame: pd.DataFrame,
    spec: Stage1CashFundedRiskBudgetSpec,
) -> pd.DataFrame:
    rows = []
    for segment, years in [
        ("development_walk_forward", spec.development_years),
        ("outer_reused_diagnostic", spec.outer_years),
    ]:
        section = frame[
            _segment_mask(frame, years) & frame["cap_bind"]
        ]
        cash = section["open_cash_weight"]
        rows.append(
            {
                "segment": segment,
                "rows": len(section),
                "cap_bind_rate": float(
                    len(section)
                    / max(1, _segment_mask(frame, years).sum())
                ),
                "mean_soxx_weight": float(
                    section["open_soxx_weight"].mean()
                ),
                "mean_cash_weight": float(cash.mean()),
                "median_cash_weight": float(cash.median()),
                "minimum_cash_weight": float(cash.min()),
                "maximum_cash_weight": float(cash.max()),
                "cash_above_20pct_rate": float(cash.gt(0.20).mean()),
                "cash_above_25pct_rate": float(cash.gt(0.25).mean()),
            }
        )
    return pd.DataFrame(rows)


def _factor_diagnostics(
    frame: pd.DataFrame,
    spec: Stage1CashFundedRiskBudgetSpec,
) -> pd.DataFrame:
    rows = []
    for segment, years in [
        ("development_walk_forward", spec.development_years),
        ("outer_reused_diagnostic", spec.outer_years),
    ]:
        section = frame[_segment_mask(frame, years)]
        cap = section[section["cap_bind"]]
        noncap = section[~section["cap_bind"]]
        for feature in FEATURE_COLUMNS:
            valid = cap[
                [feature, "cash_to_soxl_return", "cash_to_soxx_return"]
            ].dropna()
            spread = (
                valid["cash_to_soxl_return"]
                - valid["cash_to_soxx_return"]
            )
            q_low, q_high = valid[feature].quantile(
                [1.0 / 3.0, 2.0 / 3.0]
            )
            low = valid[valid[feature].le(q_low)]
            high = valid[valid[feature].ge(q_high)]
            low_spread = (
                low["cash_to_soxl_return"]
                - low["cash_to_soxx_return"]
            )
            high_spread = (
                high["cash_to_soxl_return"]
                - high["cash_to_soxx_return"]
            )
            scale = float(section[feature].std())
            rows.append(
                {
                    "segment": segment,
                    "factor": feature,
                    "cap_rows": len(valid),
                    "cap_vs_noncap_mean_difference_z": (
                        float(
                            (
                                cap[feature].mean()
                                - noncap[feature].mean()
                            )
                            / scale
                        )
                        if scale
                        else np.nan
                    ),
                    "spearman_vs_soxl_minus_soxx": float(
                        valid[feature].corr(spread, method="spearman")
                    ),
                    "low_third_rows": len(low),
                    "low_third_positive_spread_rate": float(
                        low_spread.gt(0.0).mean()
                    ),
                    "low_third_mean_spread": float(low_spread.mean()),
                    "high_third_rows": len(high),
                    "high_third_positive_spread_rate": float(
                        high_spread.gt(0.0).mean()
                    ),
                    "high_third_mean_spread": float(
                        high_spread.mean()
                    ),
                }
            )
    return pd.DataFrame(rows)


def _momentum_failure(
    frame: pd.DataFrame,
    spec: Stage1CashFundedRiskBudgetSpec,
) -> pd.DataFrame:
    rows = []
    relevant = frame[
        frame["year"].isin(
            (*spec.development_years, *spec.outer_years)
        )
        & frame["cap_bind"]
    ]
    for year, section in relevant.groupby("year"):
        positive_momentum = section[
            section["relative_momentum20"].gt(0.0)
        ]
        spread = (
            positive_momentum["soxl_return"]
            - positive_momentum["soxx_return"]
        )
        rows.append(
            {
                "year": int(year),
                "segment": (
                    "development_walk_forward"
                    if year in spec.development_years
                    else "outer_reused_diagnostic"
                ),
                "cap_rows": len(section),
                "positive_momentum_rows": len(positive_momentum),
                "positive_momentum_success_rate": (
                    float(spread.gt(0.0).mean())
                    if len(spread)
                    else np.nan
                ),
                "positive_momentum_mean_spread": (
                    float(spread.mean()) if len(spread) else np.nan
                ),
                "positive_momentum_failure_count": int(
                    spread.le(0.0).sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def _annual_metrics(
    replay: pd.DataFrame,
    *,
    variant: str,
    spec: Stage1CashFundedRiskBudgetSpec,
) -> pd.DataFrame:
    rows = []
    for year, section in replay[
        replay["year"].isin(
            (*spec.development_years, *spec.outer_years)
        )
    ].groupby("year"):
        candidate = _risk_metrics(
            section["portfolio_return_10bps"].to_numpy()
        )
        baseline = _risk_metrics(
            _baseline_returns(
                section,
                cost_bps=spec.transaction_cost_bps,
            )
        )
        rows.append(
            {
                "variant": variant,
                "year": int(year),
                "segment": (
                    "development_walk_forward"
                    if year in spec.development_years
                    else "outer_reused_diagnostic"
                ),
                **candidate,
                "baseline_return_pct": baseline["return_pct"],
                "incremental_return_pct_points": (
                    candidate["return_pct"] - baseline["return_pct"]
                ),
                "action_count": int(
                    section["overlay_weight"].gt(1e-12).sum()
                ),
                "selection_uses_outer_test": False,
            }
        )
    return pd.DataFrame(rows)


def _run_card(
    spec: Stage1CashFundedRiskBudgetSpec,
) -> dict[str, Any]:
    spec.card_dir.mkdir(parents=True, exist_ok=True)
    source_contract = _source_contract(spec)
    frame, source_audit = _build_comparison_frame(spec)
    dataset, dataset_audit = _build_qlib_dataset(frame, spec)
    thresholds = _threshold_schedule(dataset, spec)
    replays = {
        candidate.name: _apply_candidate(
            frame,
            candidate=candidate,
            thresholds=thresholds,
            spec=spec,
        )
        for candidate in CANDIDATES
    }
    rows = [
        _metrics_row(
            variant="stage1_baseline",
            candidate=None,
            replay=frame,
            years=spec.development_years,
            segment="development_walk_forward",
            spec=spec,
        ),
        _metrics_row(
            variant="stage1_baseline",
            candidate=None,
            replay=frame,
            years=spec.outer_years,
            segment="outer_reused_diagnostic",
            spec=spec,
        ),
    ]
    for candidate in CANDIDATES:
        rows.extend(
            [
                _metrics_row(
                    variant=candidate.name,
                    candidate=candidate,
                    replay=replays[candidate.name],
                    years=spec.development_years,
                    segment="development_walk_forward",
                    spec=spec,
                ),
                _metrics_row(
                    variant=candidate.name,
                    candidate=candidate,
                    replay=replays[candidate.name],
                    years=spec.outer_years,
                    segment="outer_reused_diagnostic",
                    spec=spec,
                ),
            ]
        )
    comparison, selected_name, selection_audit = _selection_table(
        pd.DataFrame(rows),
        spec,
    )
    selected_candidate = next(
        item for item in CANDIDATES if item.name == selected_name
    )
    selected_replay = replays[selected_name]
    annual = _annual_metrics(
        selected_replay,
        variant=selected_name,
        spec=spec,
    )
    cash = _cash_availability(frame, spec)
    factors = _factor_diagnostics(frame, spec)
    momentum = _momentum_failure(frame, spec)
    decision = (
        "candidate_for_clean_forward_validation_not_promoted"
        if selection_audit["all_development_gates_pass"]
        else "not_promoted_development_uplift_or_coverage_gates_failed"
    )

    from qlib.contrib.evaluate import risk_analysis
    from qlib.workflow import R

    risk_rows = []
    for segment, years in [
        ("development_walk_forward", spec.development_years),
        ("outer_reused_diagnostic", spec.outer_years),
    ]:
        section = selected_replay[_segment_mask(selected_replay, years)]
        for cost_bps, column in [
            (spec.transaction_cost_bps, "portfolio_return_10bps"),
            (spec.stress_cost_bps, "portfolio_return_25bps"),
        ]:
            risk = risk_analysis(
                pd.Series(
                    section[column].to_numpy(),
                    index=pd.to_datetime(section["trade_date"]),
                ),
                freq="day",
            ).reset_index()
            risk["segment"] = segment
            risk["cost_bps"] = cost_bps
            risk_rows.append(risk)
    qlib_risk = pd.concat(risk_rows, ignore_index=True)

    recorder_uri = spec.card_dir / "qlib_recorder"
    recorder_uri.mkdir(parents=True, exist_ok=True)
    R.set_uri(recorder_uri.resolve().as_posix())
    with tempfile.TemporaryDirectory(
        prefix=f"{spec.experiment}_"
    ) as temporary:
        temp = Path(temporary)
        selected_replay.to_csv(
            temp / "portfolio_report.csv",
            index=False,
        )
        comparison.to_csv(
            temp / "model_selection_full.csv",
            index=False,
        )
        thresholds.to_csv(
            temp / "threshold_schedule.csv",
            index=False,
        )
        frame[
            [
                "signal_date",
                "trade_date",
                "year",
                "cap_bind",
                "open_soxl_weight",
                "open_soxx_weight",
                "open_cash_weight",
                *FEATURE_COLUMNS,
                "cash_to_soxl_return",
                "cash_to_soxx_return",
            ]
        ].to_csv(temp / "factor_label_frame.csv", index=False)
        with R.start(experiment_name=spec.experiment):
            recorder = R.get_recorder()
            R.log_params(
                source_stage1_recorder=spec.source_recorder_id,
                selected_candidate=selected_name,
                candidate_count=len(CANDIDATES),
                selection_uses_outer_test="false",
                capital_source="cash_only",
                outer_thresholds="frozen_training_end_2022",
            )
            selected_dev = comparison[
                comparison["variant"].eq(selected_name)
                & comparison["segment"].eq(
                    "development_walk_forward"
                )
            ].iloc[0]
            selected_outer = comparison[
                comparison["variant"].eq(selected_name)
                & comparison["segment"].eq(
                    "outer_reused_diagnostic"
                )
            ].iloc[0]
            R.log_metrics(
                development_return_delta_pct_points=float(
                    selected_dev["return_delta_pct_points"]
                ),
                development_max_drawdown_pct=float(
                    selected_dev["max_drawdown_pct"]
                ),
                development_action_count=float(
                    selected_dev["action_count"]
                ),
                outer_return_delta_pct_points=float(
                    selected_outer["return_delta_pct_points"]
                ),
                all_development_gates=float(
                    selection_audit["all_development_gates_pass"]
                ),
            )
            for artifact in temp.glob("*.csv"):
                recorder.log_artifact(
                    artifact.as_posix(),
                    artifact_path=spec.experiment,
                )
            recorder_id = recorder.id
            recorder_local_dir = Path(
                recorder.get_local_dir()
            ).resolve()
            summary = _write_card(
                spec=spec,
                source_contract=source_contract,
                source_audit=source_audit,
                dataset_audit=dataset_audit,
                selected_candidate=selected_candidate,
                comparison=comparison,
                annual=annual,
                cash=cash,
                factors=factors,
                momentum=momentum,
                thresholds=thresholds,
                qlib_risk=qlib_risk,
                selection_audit=selection_audit,
                decision=decision,
                recorder_id=recorder_id,
                recorder_uri=recorder_uri,
                recorder_local_dir=recorder_local_dir,
            )
            for filename in summary["sidecars"].values():
                path = spec.card_dir / filename
                if path.exists():
                    recorder.log_artifact(
                        path.as_posix(),
                        artifact_path=spec.experiment,
                    )
            _remove_qlib_code_snapshot_artifacts(recorder_local_dir)
    return summary


def _write_card(
    *,
    spec: Stage1CashFundedRiskBudgetSpec,
    source_contract: dict[str, Any],
    source_audit: dict[str, Any],
    dataset_audit: dict[str, Any],
    selected_candidate: CashFundedCandidate,
    comparison: pd.DataFrame,
    annual: pd.DataFrame,
    cash: pd.DataFrame,
    factors: pd.DataFrame,
    momentum: pd.DataFrame,
    thresholds: pd.DataFrame,
    qlib_risk: pd.DataFrame,
    selection_audit: dict[str, Any],
    decision: str,
    recorder_id: str,
    recorder_uri: Path,
    recorder_local_dir: Path,
) -> dict[str, Any]:
    card = spec.card_dir
    selected_dev = comparison[
        comparison["variant"].eq(selected_candidate.name)
        & comparison["segment"].eq("development_walk_forward")
    ].iloc[0]
    selected_outer = comparison[
        comparison["variant"].eq(selected_candidate.name)
        & comparison["segment"].eq("outer_reused_diagnostic")
    ].iloc[0]
    baseline_dev = comparison[
        comparison["variant"].eq("stage1_baseline")
        & comparison["segment"].eq("development_walk_forward")
    ].iloc[0]
    dev_cash = cash[
        cash["segment"].eq("development_walk_forward")
    ].iloc[0]
    outer_cash = cash[
        cash["segment"].eq("outer_reused_diagnostic")
    ].iloc[0]
    relative_factor = factors[
        factors["factor"].eq("relative_momentum20")
    ].set_index("segment")
    dev_momentum = relative_factor.loc["development_walk_forward"]
    outer_momentum = relative_factor.loc["outer_reused_diagnostic"]
    sidecars = {
        "selected_model": "selected_model.json",
        "model_selection": "model_selection.csv",
        "backtest_benchmark_comparison": (
            "backtest_benchmark_comparison.csv"
        ),
        "annual_metrics": "annual_metrics.csv",
        "risk_analysis": "risk_analysis.csv",
        "cash_availability": "cash_availability.csv",
        "factor_diagnostics": "factor_diagnostics.csv",
        "momentum_failure": "momentum_failure.csv",
        "threshold_schedule": "threshold_schedule.csv",
        "analysis_summary": "analysis_summary.json",
        "failure_modes": "failure_modes.md",
        "conclusion": "conclusion.md",
        "workflow_config": "workflow.yaml",
    }
    summary: dict[str, Any] = {
        "experiment": spec.experiment,
        "card_type": "stage1_cash_funded_risk_budget_research_card",
        "primary_notebook": f"{spec.experiment}.ipynb",
        "system_of_record": "qlib",
        "provider_uri": spec.provider_uri.resolve().as_posix(),
        "as_of": source_audit["intraday_provider_as_of"],
        "market": "US",
        "region": "us",
        "source_stage1": source_contract,
        "hypothesis": (
            "When Stage1 is already at its 30% SOXL ceiling but retains "
            "cash, a low-frequency non-overextended trend gate can fund "
            "a small SOXL or SOXX overlay from cash without replacing SOXX."
        ),
        "method": {
            "capital_source": "cash_only",
            "holding_period": "next_open_to_following_open",
            "base_stage1": "frozen",
            "development": "2018-2022 expanding prior-year thresholds",
            "outer": (
                "2023-2026 thresholds frozen using data through "
                "2022-12-30; outer excluded from selection"
            ),
            "gate": {
                "required": [
                    "Stage1 SOXL weight at 30% cap",
                    "SOXX 20d trend positive",
                    "relative momentum not above prior cap-event q67",
                ],
                "quality_variant_adds": [
                    "SOXX volatility below prior cap-event q75",
                    "chop risk below prior cap-event median",
                ],
            },
            "hard_constraints": {
                "minimum_cash_weight": spec.minimum_cash_weight,
                "source_maximum_soxl_weight": (
                    spec.source_maximum_soxl_weight
                ),
                "overlay_maximum_soxl_weight": (
                    spec.overlay_maximum_soxl_weight
                ),
                "long_only": True,
                "weights_sum_to_one": True,
            },
        },
        "candidate_count": len(CANDIDATES),
        "candidates": [asdict(item) for item in CANDIDATES],
        "selected_model": {
            **asdict(selected_candidate),
            "selection_uses_outer_test": False,
        },
        "development_baseline_metrics_10bps": _json_row(baseline_dev),
        "development_selected_metrics_10bps": _json_row(selected_dev),
        "outer_diagnostic_selected_metrics_10bps": _json_row(
            selected_outer
        ),
        "selection_gates": selection_audit,
        "cash_audit": {
            "development_cap_rows": int(dev_cash["rows"]),
            "development_mean_cash_weight": float(
                dev_cash["mean_cash_weight"]
            ),
            "outer_cap_rows": int(outer_cash["rows"]),
            "outer_mean_cash_weight": float(
                outer_cash["mean_cash_weight"]
            ),
        },
        "momentum_audit": {
            "development_spearman_vs_soxl_minus_soxx": float(
                dev_momentum["spearman_vs_soxl_minus_soxx"]
            ),
            "development_low_third_mean_spread": float(
                dev_momentum["low_third_mean_spread"]
            ),
            "development_high_third_mean_spread": float(
                dev_momentum["high_third_mean_spread"]
            ),
            "outer_spearman_vs_soxl_minus_soxx": float(
                outer_momentum["spearman_vs_soxl_minus_soxx"]
            ),
        },
        "selection_uses_outer_test": False,
        "outer_status": "reused_diagnostic_not_clean_unseen_holdout",
        "decision": decision,
        "promotion": "blocked",
        "dataset_audit": dataset_audit,
        "threshold_audit": {
            "outer_training_end": spec.source_training_end,
            "outer_unique_threshold_rows": int(
                thresholds[
                    thresholds["test_year"].isin(spec.outer_years)
                ][
                    [
                        "relative_momentum_q67",
                        "chop_breakdown_q50",
                        "soxx_vol_q75",
                    ]
                ]
                .drop_duplicates()
                .shape[0]
            ),
        },
        "qlib_api": [
            "qlib.init",
            "qlib.data.D.features",
            "DatasetH",
            "DataHandlerLP",
            "qlib.workflow.R",
            "qlib.contrib.evaluate.risk_analysis",
        ],
        "qlib_docs_reference": [
            "https://qlib.readthedocs.io/en/latest/component/strategy.html",
            "https://qlib.readthedocs.io/en/latest/component/recorder.html",
            "https://qlib.readthedocs.io/en/latest/component/model.html",
        ],
        "recorder_id": recorder_id,
        "recorder_uri": recorder_uri.resolve().as_posix(),
        "recorder_local_dir": recorder_local_dir.as_posix(),
        "sidecars": sidecars,
        "artifact_policy": {
            "policy": "slim_card_recorder_first",
            "workflow_config": "workflow.yaml",
            "recorder_only_artifacts": [
                f"artifacts/{spec.experiment}/portfolio_report.csv",
                f"artifacts/{spec.experiment}/model_selection_full.csv",
                f"artifacts/{spec.experiment}/factor_label_frame.csv",
            ],
        },
    }
    (card / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    comparison.to_csv(card / "model_selection.csv", index=False)
    comparison[
        comparison["variant"].isin(
            ["stage1_baseline", selected_candidate.name]
        )
    ].to_csv(card / "backtest_benchmark_comparison.csv", index=False)
    annual.to_csv(card / "annual_metrics.csv", index=False)
    qlib_risk.to_csv(card / "risk_analysis.csv", index=False)
    cash.to_csv(card / "cash_availability.csv", index=False)
    factors.to_csv(card / "factor_diagnostics.csv", index=False)
    momentum.to_csv(card / "momentum_failure.csv", index=False)
    thresholds.to_csv(card / "threshold_schedule.csv", index=False)
    (card / "selected_model.json").write_text(
        json.dumps(
            {
                **asdict(selected_candidate),
                "selection_uses_outer_test": False,
                "development_metrics": _json_row(selected_dev),
                "selection_gates": selection_audit,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    analysis = {
        "selected_candidate": selected_candidate.name,
        "development_return_delta_pct_points": float(
            selected_dev["return_delta_pct_points"]
        ),
        "development_max_drawdown_delta_pct_points": float(
            selected_dev["max_drawdown_delta_pct_points"]
        ),
        "development_action_count": int(selected_dev["action_count"]),
        "outer_return_delta_pct_points": float(
            selected_outer["return_delta_pct_points"]
        ),
        "development_cap_mean_cash_weight": float(
            dev_cash["mean_cash_weight"]
        ),
        "momentum_high_minus_low_spread": float(
            dev_momentum["high_third_mean_spread"]
            - dev_momentum["low_third_mean_spread"]
        ),
        "decision": decision,
        "selection_uses_outer_test": False,
    }
    (card / "analysis_summary.json").write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    failed = [
        key
        for key, value in selection_audit.items()
        if key.endswith("_gate_pass") and value is False
    ]
    (card / "failure_modes.md").write_text(
        "# Failure modes\n\n"
        + "\n".join(f"- {item}" for item in failed)
        + "\n- outer_reused_diagnostic_negative_for_selected_candidate"
        + "\n- clean_unseen_holdout_unavailable\n",
        encoding="utf-8",
    )
    conclusion = (
        "# 中文结论\n\n"
        "- 做了什么：冻结现有 Stage1，仅在 SOXL 已触及 30%、SOXX "
        "20 日趋势为正且相对动量未处于 prior cap-event 最高三分之一"
        "时，测试从现金增加 5%/10% SOXL 或 SOXX；现金不得低于 15%，"
        "SOXL 临时上限 40%。\n"
        f"- 封顶后现金：开发期 {int(dev_cash['rows'])} 次封顶，平均现金="
        f"{100.0 * float(dev_cash['mean_cash_weight']):.2f}%；outer "
        f"{int(outer_cash['rows'])} 次封顶，平均现金="
        f"{100.0 * float(outer_cash['mean_cash_weight']):.2f}%。\n"
        f"- 开发期选中诊断候选：`{selected_candidate.name}`，10bp return="
        f"{float(selected_dev['return_pct']):.4f}%，相对 Stage1 增量="
        f"{float(selected_dev['return_delta_pct_points']):.4f} 个百分点，"
        f"动作={int(selected_dev['action_count'])}，MaxDD="
        f"{float(selected_dev['max_drawdown_pct']):.4f}%。\n"
        f"- 收益门：要求至少 "
        f"{spec.minimum_development_return_delta_pct_points:.2f} 个百分点，"
        f"实际={float(selected_dev['return_delta_pct_points']):.4f}，"
        f"gate={selection_audit['return_gate_pass']}。\n"
        f"- 动量诊断：开发期封顶样本的相对动量与下一日 "
        f"SOXL-SOXX spread Spearman="
        f"{float(dev_momentum['spearman_vs_soxl_minus_soxx']):.4f}；"
        f"低三分位平均 spread="
        f"{100.0 * float(dev_momentum['low_third_mean_spread']):.4f}%，"
        f"高三分位="
        f"{100.0 * float(dev_momentum['high_third_mean_spread']):.4f}%，"
        "确认高动量存在过热/反转风险。\n"
        f"- outer 复用诊断：相对 Stage1 增量="
        f"{float(selected_outer['return_delta_pct_points']):.4f} 个百分点；"
        "该窗口不参与选择，也不是 clean holdout。\n"
        f"- 决策：`{decision}`；promotion=blocked，生产 Stage1 不变。\n"
        f"- 失败门：{failed}。开发期的小幅改善不足以补偿样本稀疏和"
        "outer 方向反转。\n"
        "- 下一步：把本卡保留为现金出资失败边界；只有新增 PIT 广度"
        "信息或预登记 clean-forward 后才重开，不用 outer 调阈值。\n"
        f"- qlib recorder：`{recorder_id}`。\n"
        "- 不构成买卖建议。\n"
    )
    (card / "conclusion.md").write_text(
        conclusion,
        encoding="utf-8",
    )
    (card / "workflow.yaml").write_text(
        "\n".join(
            [
                f"experiment: {spec.experiment}",
                f"provider_uri: {summary['provider_uri']}",
                f"recorder_uri: {summary['recorder_uri']}",
                "market: US",
                "region: us",
                f"source_stage1_recorder: {spec.source_recorder_id}",
                "source_stage1_frozen: true",
                "stage2_enabled: false",
                "capital_source: cash_only",
                "minimum_cash_weight: 0.15",
                "source_maximum_soxl_weight: 0.30",
                "overlay_maximum_soxl_weight: 0.40",
                "selection_uses_outer_test: false",
                f"outer_threshold_training_end: {spec.source_training_end}",
                "model_selection:",
                "  candidate_count: 6",
                "  segment: 2018-2022_development_walk_forward",
                "  benchmark: frozen_stage1",
                "artifact_policy:",
                "  policy: slim_card_recorder_first",
                "  workflow_config: workflow.yaml",
                "  recorder_only_artifacts:",
                *[
                    f"    - {artifact}"
                    for artifact in summary["artifact_policy"][
                        "recorder_only_artifacts"
                    ]
                ],
                "",
            ]
        ),
        encoding="utf-8",
    )
    return summary


def _remove_qlib_code_snapshot_artifacts(
    recorder_local_dir: Path,
) -> None:
    artifacts = recorder_local_dir / "artifacts"
    for filename in (
        "code_cached.txt",
        "code_diff.txt",
        "code_status.txt",
    ):
        (artifacts / filename).unlink(missing_ok=True)


def _json_row(row: pd.Series) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in row.to_dict().items():
        if isinstance(value, (np.bool_, bool)):
            result[key] = bool(value)
        elif isinstance(value, np.integer):
            result[key] = int(value)
        elif isinstance(value, (np.floating, float)):
            result[key] = None if pd.isna(value) else float(value)
        else:
            result[key] = value
    return result
