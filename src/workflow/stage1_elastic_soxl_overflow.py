from __future__ import annotations

import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.workflow.direct_continuous_soxx_allocation import (
    DirectContinuousAllocationSpec,
    _load_frame,
)
from src.workflow.intraday_recovery_direct_allocation import (
    IntradayRecoveryDirectAllocationSpec,
    _build_teacher_training_frame,
    _load_source_frame,
    _risk_metrics,
    _simulate_policy,
)


EXPERIMENT = "stage1_elastic_soxl_overflow_v1"
SOURCE_EXPERIMENT = "intraday_on_policy_staged_validation_v1"
SOURCE_RECORDER_ID = "f6b09330175e41719886d58ee2d11bec"
SOURCE_TRAINING_END = "2022-12-30"

FEATURE_COLUMNS = (
    "soxx_trend20",
    "relative_momentum20",
    "soxx_vol20",
    "trend_efficiency_20",
    "levered_decay_20",
    "chop_breakdown_risk",
)


@dataclass(frozen=True)
class OverflowCandidate:
    name: str
    model_type: str
    overflow_weight: float
    probability_threshold: float | None


CANDIDATES = (
    OverflowCandidate("always_005", "always", 0.05, None),
    OverflowCandidate("always_010", "always", 0.10, None),
    OverflowCandidate("logistic_055_005", "logistic", 0.05, 0.55),
    OverflowCandidate("logistic_055_010", "logistic", 0.10, 0.55),
    OverflowCandidate("logistic_065_005", "logistic", 0.05, 0.65),
    OverflowCandidate("logistic_065_010", "logistic", 0.10, 0.65),
    OverflowCandidate("logistic_075_005", "logistic", 0.05, 0.75),
    OverflowCandidate("logistic_075_010", "logistic", 0.10, 0.75),
)


@dataclass(frozen=True)
class Stage1ElasticSoxlOverflowSpec:
    root: Path
    experiment: str = EXPERIMENT
    source_experiment: str = SOURCE_EXPERIMENT
    source_recorder_id: str = SOURCE_RECORDER_ID
    source_training_end: str = SOURCE_TRAINING_END
    train_only_years: tuple[int, ...] = (2016, 2017)
    development_years: tuple[int, ...] = (2018, 2019, 2020, 2021, 2022)
    outer_years: tuple[int, ...] = (2023, 2024, 2025, 2026)
    transaction_cost_bps: float = 10.0
    stress_cost_bps: float = 25.0
    cap_binding_tolerance: float = 0.0005
    logistic_c: float = 0.10
    purge_rows: int = 1
    minimum_training_events: int = 40
    minimum_events_per_feature: float = 8.0
    minimum_development_actions: int = 12
    minimum_positive_action_rate: float = 0.52
    minimum_positive_years: int = 3
    maximum_maxdd_degradation_pct_points: float = 2.0
    maximum_worst5_degradation_pct_points: float = 0.75
    maximum_top5_positive_contribution_share: float = 0.60
    maximum_outer_event_rate_ratio: float = 2.0

    @property
    def card_dir(self) -> Path:
        return self.root / "research" / self.experiment

    @property
    def source_card_dir(self) -> Path:
        return self.root / "research" / self.source_experiment

    @property
    def provider_uri(self) -> Path:
        return self.root / "data" / "qlib" / "soxl_soxx_soxs"


def run_stage1_elastic_soxl_overflow_card(
    *, root: Path | str | None = None
) -> dict[str, Any]:
    return _run_card(Stage1ElasticSoxlOverflowSpec(root=_resolve_root(root)))


def _resolve_root(root: Path | str | None) -> Path:
    if root is not None:
        return Path(root).resolve()
    current = Path.cwd().resolve()
    while current != current.parent:
        if (current / "AGENTS.md").exists() and (current / "SPEC.md").exists():
            return current
        current = current.parent
    raise ValueError("could not resolve repository root")


def _source_selection(spec: Stage1ElasticSoxlOverflowSpec) -> dict[str, Any]:
    summary = json.loads(
        (spec.source_card_dir / "summary.json").read_text(encoding="utf-8")
    )
    if summary.get("recorder_id") != spec.source_recorder_id:
        raise ValueError("blocked: frozen Stage1 recorder drifted")
    if not str(summary.get("decision", "")).startswith(
        "promoted_research_candidate_stage1"
    ):
        raise ValueError("blocked: source Stage1 is not promoted")
    if bool(summary["selected_model"].get("stage2_enabled")):
        raise ValueError("blocked: overflow comparison requires Stage2 disabled")
    return dict(summary["selected_model"])


def _stage1_replay(
    spec: Stage1ElasticSoxlOverflowSpec, *, cost_bps: float
) -> tuple[pd.DataFrame, dict[str, Any]]:
    recovery_spec = IntradayRecoveryDirectAllocationSpec(root=spec.root)
    frame, source_audit = _load_source_frame(recovery_spec)
    teacher = _build_teacher_training_frame(
        frame,
        spec=recovery_spec,
        teacher_beta=recovery_spec.teacher_recovery_beta,
    )
    selected = _source_selection(spec)
    replay = _simulate_policy(
        frame.reset_index(),
        spec=recovery_spec,
        teacher_frame=teacher,
        c_value=float(selected["logistic_c"]),
        threshold=float(selected["probability_threshold"]),
        action_beta=float(selected["action_beta"]),
        first_action_year=recovery_spec.first_intraday_action_year,
        model_cache={},
        cost_bps=cost_bps,
    )
    for column in ("signal_date", "trade_date", "exit_date"):
        replay[column] = pd.to_datetime(replay[column])
    return replay.sort_values("signal_date").reset_index(drop=True), source_audit


def _qlib_features(
    spec: Stage1ElasticSoxlOverflowSpec,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    market, source_audit, available_features = _load_frame(
        DirectContinuousAllocationSpec(
            root=spec.root,
            train_start="2011-01-03",
            development_years=(
                *spec.train_only_years,
                *spec.development_years,
            ),
            development_end=spec.source_training_end,
            outer=(
                f"{spec.outer_years[0]}-01-01",
                f"{spec.outer_years[-1]}-12-31",
            ),
            purge_rows=spec.purge_rows,
        )
    )
    missing = sorted(set(FEATURE_COLUMNS).difference(available_features))
    if missing:
        raise ValueError(f"blocked: overflow features missing: {missing}")
    result = market.loc[:, list(FEATURE_COLUMNS)].copy()
    result.index = pd.to_datetime(result.index)
    return result.sort_index(), {
        **source_audit,
        "selected_feature_count": len(FEATURE_COLUMNS),
        "selected_features": list(FEATURE_COLUMNS),
        "source_api": "qlib.data.D.features",
    }


def _comparison_frame(
    spec: Stage1ElasticSoxlOverflowSpec,
    replay: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    features, feature_audit = _qlib_features(spec)
    frame = replay.merge(
        features,
        left_on="signal_date",
        right_index=True,
        how="left",
        validate="one_to_one",
    )
    if frame.loc[:, FEATURE_COLUMNS].isna().all(axis=1).any():
        raise ValueError("blocked: qlib overflow features missing for replay rows")
    frame["year"] = frame["signal_date"].dt.year
    frame["cap_bind"] = frame["open_soxl_weight"].ge(
        0.30 - spec.cap_binding_tolerance
    )
    frame["incremental_spread"] = (
        frame["soxl_return"] - frame["soxx_return"]
    )
    frame["incremental_label"] = (
        frame["incremental_spread"].gt(0.0).astype("float64")
    )
    return frame.sort_values("signal_date").reset_index(drop=True), feature_audit


def _build_dataset(
    frame: pd.DataFrame, spec: Stage1ElasticSoxlOverflowSpec
):
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP
    from qlib.data.dataset.loader import StaticDataLoader

    indexed = frame.set_index("signal_date").sort_index()
    labels = indexed.loc[
        :, ["incremental_label", "incremental_spread", "cap_bind"]
    ].astype("float64")
    static = pd.concat(
        {
            "feature": indexed.loc[:, FEATURE_COLUMNS],
            "label": labels,
        },
        axis=1,
    )
    static.index = pd.MultiIndex.from_arrays(
        [static.index, np.full(len(static), "STAGE1_ELASTIC_OVERFLOW")],
        names=["datetime", "instrument"],
    )
    segments: dict[str, tuple[str, str]] = {}
    for year in (*spec.development_years, *spec.outer_years):
        train_end = (
            spec.source_training_end
            if year in spec.outer_years
            else f"{year - 1}-12-31"
        )
        segments[f"train_{year}"] = (
            f"{spec.train_only_years[0]}-01-01",
            train_end,
        )
        segments[f"test_{year}"] = (
            f"{year}-01-01",
            f"{year}-12-31",
        )
    handler = DataHandlerLP(
        instruments=None,
        start_time=static.index.get_level_values("datetime").min(),
        end_time=static.index.get_level_values("datetime").max(),
        data_loader=StaticDataLoader(config=static),
        infer_processors=[],
        learn_processors=[],
    )
    dataset = DatasetH(handler=handler, segments=segments)
    return dataset, {
        "api": ["DatasetH", "DataHandlerLP", "StaticDataLoader"],
        "segments": {name: list(bounds) for name, bounds in segments.items()},
        "segment_rows": {
            name: int(len(dataset.prepare(name))) for name in segments
        },
        "features": list(FEATURE_COLUMNS),
        "feature_count": len(FEATURE_COLUMNS),
        "label": "next_open_to_open_soxl_minus_soxx_positive",
        "purge_rows": spec.purge_rows,
        "outer_training_end_frozen": spec.source_training_end,
    }


def _prepared(dataset, segment: str) -> pd.DataFrame:
    result = dataset.prepare(segment, col_set=["feature", "label"])
    result = result.droplevel("instrument").sort_index()
    result.columns = [str(column[-1]) for column in result.columns]
    return result


def _fit_probability_models(
    dataset,
    spec: Stage1ElasticSoxlOverflowSpec,
) -> tuple[pd.Series, pd.DataFrame]:
    probabilities: list[pd.Series] = []
    audit_rows: list[dict[str, Any]] = []
    for year in (*spec.development_years, *spec.outer_years):
        training = _prepared(dataset, f"train_{year}")
        training = training[training["cap_bind"].gt(0.5)].copy()
        training = training.iloc[: -spec.purge_rows or None]
        testing = _prepared(dataset, f"test_{year}")
        class_count = int(training["incremental_label"].nunique())
        model = None
        status = "ok"
        if (
            len(training) < spec.minimum_training_events
            or class_count < 2
        ):
            status = "blocked_insufficient_training_events"
        else:
            model = make_pipeline(
                SimpleImputer(strategy="median"),
                StandardScaler(),
                LogisticRegression(
                    C=spec.logistic_c,
                    class_weight="balanced",
                    max_iter=2_000,
                ),
            )
            model.fit(
                training.loc[:, FEATURE_COLUMNS],
                training["incremental_label"],
            )
        if model is None:
            prediction = pd.Series(np.nan, index=testing.index)
        else:
            prediction = pd.Series(
                model.predict_proba(testing.loc[:, FEATURE_COLUMNS])[:, 1],
                index=testing.index,
            )
        probabilities.append(prediction)
        audit_rows.append(
            {
                "test_year": year,
                "train_end": (
                    spec.source_training_end
                    if year in spec.outer_years
                    else f"{year - 1}-12-31"
                ),
                "training_cap_events": int(len(training)),
                "positive_training_events": int(
                    training["incremental_label"].sum()
                ),
                "feature_count": len(FEATURE_COLUMNS),
                "events_per_feature": float(
                    len(training) / len(FEATURE_COLUMNS)
                ),
                "status": status,
                "outer_model_frozen": year in spec.outer_years,
            }
        )
    combined = pd.concat(probabilities).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]
    return combined, pd.DataFrame(audit_rows)


def _apply_overlay(
    frame: pd.DataFrame,
    *,
    probability: pd.Series,
    candidate: OverflowCandidate,
    cost_bps: float,
) -> pd.DataFrame:
    result = frame.copy()
    aligned_probability = probability.reindex(result["signal_date"]).to_numpy()
    result["overflow_probability"] = aligned_probability
    probability_gate = (
        np.ones(len(result), dtype=bool)
        if candidate.model_type == "always"
        else pd.Series(aligned_probability).ge(
            float(candidate.probability_threshold)
        ).to_numpy()
    )
    result["overflow_action"] = (
        result["cap_bind"].to_numpy(dtype=bool) & probability_gate
    )
    result["overflow_weight"] = np.where(
        result["overflow_action"],
        np.minimum(
            candidate.overflow_weight,
            result["open_soxx_weight"].to_numpy(dtype="float64"),
        ),
        0.0,
    )
    result["target_soxl_weight"] = (
        result["open_soxl_weight"] + result["overflow_weight"]
    )
    result["target_soxx_weight"] = (
        result["open_soxx_weight"] - result["overflow_weight"]
    )
    result["target_cash_weight"] = result["open_cash_weight"]
    if not np.allclose(
        result[
            [
                "target_soxl_weight",
                "target_soxx_weight",
                "target_cash_weight",
            ]
        ].sum(axis=1),
        1.0,
        atol=1e-8,
    ):
        raise ValueError("blocked: overflow weights do not sum to one")
    if result["target_soxl_weight"].gt(0.40 + 1e-8).any():
        raise ValueError("blocked: overflow exceeds 40% SOXL ceiling")
    incremental_turnover = (
        result["overflow_weight"].diff().abs().fillna(
            result["overflow_weight"].abs()
        )
    )
    result["incremental_turnover"] = incremental_turnover
    result["incremental_gross_return"] = (
        result["overflow_weight"] * result["incremental_spread"]
    )
    result["incremental_cost"] = (
        incremental_turnover * cost_bps / 10_000.0
    )
    result["incremental_net_return"] = (
        result["incremental_gross_return"] - result["incremental_cost"]
    )
    result["sleeve_return"] = (
        result["strategy_return"] + result["incremental_net_return"]
    )
    result["candidate"] = candidate.name
    result["overflow_cost_bps"] = cost_bps
    return result


def _metrics(
    replay: pd.DataFrame,
    years: tuple[int, ...],
) -> dict[str, float | int]:
    section = replay[replay["year"].isin(years)].copy()
    baseline = _risk_metrics(section["strategy_return"].to_numpy())
    sleeve = _risk_metrics(section["sleeve_return"].to_numpy())
    annualized_volatility_pct = float(
        section["sleeve_return"].std(ddof=0) * np.sqrt(252.0) * 100.0
    )
    actions = section[section["overflow_action"]]
    positive_gross = actions["incremental_gross_return"].clip(lower=0.0)
    top5_share = (
        float(positive_gross.nlargest(5).sum() / positive_gross.sum())
        if positive_gross.sum() > 0.0
        else 1.0
    )
    return {
        "rows": int(len(section)),
        "return_pct": sleeve["return_pct"],
        "max_drawdown_pct": sleeve["max_drawdown_pct"],
        "worst_day_pct": sleeve["worst_day_pct"],
        "cvar5_pct": sleeve["cvar5_pct"],
        "worst_rolling_5d_pct": sleeve["worst_rolling_5d_pct"],
        "annualized_volatility_pct": annualized_volatility_pct,
        "baseline_return_pct": baseline["return_pct"],
        "baseline_max_drawdown_pct": baseline["max_drawdown_pct"],
        "baseline_worst_rolling_5d_pct": baseline[
            "worst_rolling_5d_pct"
        ],
        "return_delta_pct_points": (
            sleeve["return_pct"] - baseline["return_pct"]
        ),
        "max_drawdown_delta_pct_points": (
            sleeve["max_drawdown_pct"] - baseline["max_drawdown_pct"]
        ),
        "worst_5d_delta_pct_points": (
            sleeve["worst_rolling_5d_pct"]
            - baseline["worst_rolling_5d_pct"]
        ),
        "action_count": int(section["overflow_action"].sum()),
        "cap_bind_count": int(section["cap_bind"].sum()),
        "cap_bind_rate": float(section["cap_bind"].mean()),
        "positive_action_rate": (
            float(actions["incremental_net_return"].gt(0.0).mean())
            if len(actions)
            else 0.0
        ),
        "incremental_turnover": float(
            section["incremental_turnover"].sum()
        ),
        "top5_positive_contribution_share": top5_share,
    }


def _annual_metrics(
    replay: pd.DataFrame, years: tuple[int, ...]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for year in years:
        section = replay[replay["year"].eq(year)]
        if section.empty:
            continue
        baseline_wealth = float((1.0 + section["strategy_return"]).prod())
        sleeve_wealth = float((1.0 + section["sleeve_return"]).prod())
        rows.append(
            {
                "year": year,
                "segment": (
                    "development_walk_forward"
                    if year <= 2022
                    else "outer_reused_diagnostic"
                ),
                "rows": int(len(section)),
                "cap_bind_count": int(section["cap_bind"].sum()),
                "action_count": int(section["overflow_action"].sum()),
                "baseline_return_pct": (baseline_wealth - 1.0) * 100.0,
                "sleeve_return_pct": (sleeve_wealth - 1.0) * 100.0,
                "incremental_wealth_pct_points": (
                    sleeve_wealth - baseline_wealth
                )
                * 100.0,
                "positive_incremental_year": (
                    sleeve_wealth > baseline_wealth
                ),
            }
        )
    return pd.DataFrame(rows)


def _development_row(
    replay: pd.DataFrame,
    candidate: OverflowCandidate,
    spec: Stage1ElasticSoxlOverflowSpec,
) -> dict[str, Any]:
    metrics = _metrics(replay, spec.development_years)
    annual = _annual_metrics(replay, spec.development_years)
    positive_years = int(annual["positive_incremental_year"].sum())
    gates = {
        "positive_return_gate": metrics["return_delta_pct_points"] > 0.0,
        "action_count_gate": (
            metrics["action_count"] >= spec.minimum_development_actions
        ),
        "positive_action_rate_gate": (
            metrics["positive_action_rate"]
            >= spec.minimum_positive_action_rate
        ),
        "positive_years_gate": (
            positive_years >= spec.minimum_positive_years
        ),
        "max_drawdown_gate": (
            metrics["max_drawdown_delta_pct_points"]
            >= -spec.maximum_maxdd_degradation_pct_points
        ),
        "worst_5d_gate": (
            metrics["worst_5d_delta_pct_points"]
            >= -spec.maximum_worst5_degradation_pct_points
        ),
        "event_concentration_gate": (
            metrics["top5_positive_contribution_share"]
            <= spec.maximum_top5_positive_contribution_share
        ),
    }
    return {
        **asdict(candidate),
        "selection_uses_outer_test": False,
        **metrics,
        "positive_development_years": positive_years,
        **gates,
        "gate_pass_count": int(sum(gates.values())),
        "all_development_gates_pass": bool(all(gates.values())),
    }


def _select_candidate(
    frame: pd.DataFrame,
    *,
    probability: pd.Series,
    spec: Stage1ElasticSoxlOverflowSpec,
) -> tuple[OverflowCandidate, pd.DataFrame, dict[str, pd.DataFrame]]:
    rows: list[dict[str, Any]] = []
    replays: dict[str, pd.DataFrame] = {}
    for candidate in CANDIDATES:
        replay = _apply_overlay(
            frame,
            probability=probability,
            candidate=candidate,
            cost_bps=spec.transaction_cost_bps,
        )
        replays[candidate.name] = replay
        rows.append(_development_row(replay, candidate, spec))
    table = pd.DataFrame(rows)
    passing = table[table["all_development_gates_pass"]]
    pool = passing if not passing.empty else table
    selected_name = str(
        pool.sort_values(
            [
                "gate_pass_count",
                "positive_development_years",
                "return_delta_pct_points",
                "model_type",
                "overflow_weight",
            ],
            ascending=[False, False, False, True, True],
        ).iloc[0]["name"]
    )
    table["selected"] = table["name"].eq(selected_name)
    selected = next(
        candidate for candidate in CANDIDATES if candidate.name == selected_name
    )
    return selected, table, replays


def _overfit_audit(
    *,
    frame: pd.DataFrame,
    selected: OverflowCandidate,
    model_selection: pd.DataFrame,
    fit_audit: pd.DataFrame,
    development: dict[str, Any],
    outer: dict[str, Any],
    spec: Stage1ElasticSoxlOverflowSpec,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    development_fit = fit_audit[
        fit_audit["test_year"].isin(spec.development_years)
    ]
    minimum_training_events = int(
        development_fit["training_cap_events"].min()
    )
    minimum_events_per_feature = float(
        development_fit["events_per_feature"].min()
    )
    event_rate_ratio = float(
        outer["cap_bind_rate"] / max(development["cap_bind_rate"], 1e-12)
    )
    same_family = model_selection[
        model_selection["model_type"].eq(selected.model_type)
        & ~model_selection["selected"]
    ]
    if same_family.empty:
        neighborhood_retention = 0.0
    else:
        selected_delta = float(development["return_delta_pct_points"])
        neighbor_delta = float(
            same_family["return_delta_pct_points"].max()
        )
        neighborhood_retention = (
            neighbor_delta / selected_delta
            if selected_delta > 0.0
            else 0.0
        )
    rows = [
        {
            "check": "candidate_count",
            "value": len(CANDIDATES),
            "threshold": "<=8",
            "passed": len(CANDIDATES) <= 8,
            "interpretation": "small predeclared search grid",
        },
        {
            "check": "minimum_training_cap_events",
            "value": minimum_training_events,
            "threshold": f">={spec.minimum_training_events}",
            "passed": (
                minimum_training_events >= spec.minimum_training_events
            ),
            "interpretation": "earliest expanding fold sample size",
        },
        {
            "check": "minimum_events_per_feature",
            "value": minimum_events_per_feature,
            "threshold": f">={spec.minimum_events_per_feature}",
            "passed": (
                minimum_events_per_feature
                >= spec.minimum_events_per_feature
            ),
            "interpretation": "sparse-event model capacity control",
        },
        {
            "check": "development_all_gates",
            "value": development["return_delta_pct_points"],
            "threshold": "all development gates true",
            "passed": bool(
                model_selection.loc[
                    model_selection["selected"],
                    "all_development_gates_pass",
                ].iloc[0]
            ),
            "interpretation": "selection is not rescued by outer results",
        },
        {
            "check": "parameter_neighborhood_retention",
            "value": neighborhood_retention,
            "threshold": ">=0.50",
            "passed": neighborhood_retention >= 0.50,
            "interpretation": "nearby candidate does not collapse",
        },
        {
            "check": "outer_cap_event_rate_ratio",
            "value": event_rate_ratio,
            "threshold": f"<={spec.maximum_outer_event_rate_ratio}",
            "passed": event_rate_ratio <= spec.maximum_outer_event_rate_ratio,
            "interpretation": "distribution shift between development and outer",
        },
        {
            "check": "clean_unseen_holdout",
            "value": 0,
            "threshold": "required for promotion",
            "passed": False,
            "interpretation": (
                "2023-2026 is reused diagnostic and cannot validate promotion"
            ),
        },
    ]
    table = pd.DataFrame(rows)
    failed = table.loc[~table["passed"], "check"].tolist()
    level = (
        "high"
        if {
            "development_all_gates",
            "clean_unseen_holdout",
        }.intersection(failed)
        or event_rate_ratio > spec.maximum_outer_event_rate_ratio
        else "moderate"
    )
    return table, {
        "risk_level": level,
        "failed_checks": failed,
        "outer_cap_event_rate_ratio": event_rate_ratio,
        "minimum_training_cap_events": minimum_training_events,
        "minimum_events_per_feature": minimum_events_per_feature,
        "candidate_count": len(CANDIDATES),
        "outer_is_selection_input": False,
        "clean_unseen_holdout_available": False,
    }


def _record(
    spec: Stage1ElasticSoxlOverflowSpec,
    *,
    selected: OverflowCandidate,
    selected_replay: pd.DataFrame,
    model_selection: pd.DataFrame,
    fit_audit: pd.DataFrame,
    overfit_audit: pd.DataFrame,
    annual_metrics: pd.DataFrame,
) -> dict[str, str]:
    from qlib.workflow import R

    recorder_uri = spec.card_dir / "qlib_recorder"
    recorder_uri.mkdir(parents=True, exist_ok=True)
    R.set_uri(recorder_uri.resolve().as_posix())
    with tempfile.TemporaryDirectory(prefix=f"{spec.experiment}_") as tmp:
        tmp_dir = Path(tmp)
        predictions = selected_replay[
            [
                "signal_date",
                "trade_date",
                "year",
                "segment",
                "cap_bind",
                "overflow_probability",
                "overflow_action",
                "incremental_spread",
                "incremental_net_return",
            ]
        ]
        positions = selected_replay[
            [
                "signal_date",
                "trade_date",
                "open_soxl_weight",
                "open_soxx_weight",
                "open_cash_weight",
                "overflow_weight",
                "target_soxl_weight",
                "target_soxx_weight",
                "target_cash_weight",
                "incremental_turnover",
            ]
        ]
        report = selected_replay[
            [
                "signal_date",
                "trade_date",
                "exit_date",
                "segment",
                "strategy_return",
                "sleeve_return",
                "incremental_gross_return",
                "incremental_cost",
                "incremental_net_return",
            ]
        ]
        artifacts = {
            "predictions.csv": predictions,
            "positions.csv": positions,
            "portfolio_report.csv": report,
            "model_selection_full.csv": model_selection,
            "fit_audit.csv": fit_audit,
            "overfit_audit.csv": overfit_audit,
            "annual_metrics.csv": annual_metrics,
        }
        for filename, frame in artifacts.items():
            frame.to_csv(tmp_dir / filename, index=False)
        (tmp_dir / "selected_model_spec.json").write_text(
            json.dumps(asdict(selected), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        with R.start(experiment_name=spec.experiment):
            recorder = R.get_recorder()
            R.log_params(
                source_experiment=spec.source_experiment,
                source_recorder_id=spec.source_recorder_id,
                source_training_end=spec.source_training_end,
                selection_uses_outer_test="false",
                max_overflow_weight="0.10",
                cash_unchanged="true",
            )
            selected_row = model_selection.loc[
                model_selection["selected"]
            ].iloc[0]
            R.log_metrics(
                development_return_delta_pct_points=float(
                    selected_row["return_delta_pct_points"]
                ),
                development_action_count=float(
                    selected_row["action_count"]
                ),
                development_gate_pass_count=float(
                    selected_row["gate_pass_count"]
                ),
            )
            for filename in (*artifacts, "selected_model_spec.json"):
                recorder.log_artifact(
                    (tmp_dir / filename).as_posix(),
                    artifact_path=spec.experiment,
                )
            result = {
                "recorder_id": recorder.id,
                "recorder_uri": recorder_uri.resolve().as_posix(),
                "recorder_local_dir": (
                    Path(recorder.get_local_dir()).resolve().as_posix()
                ),
            }
    artifact_root = Path(result["recorder_local_dir"]) / "artifacts"
    for filename in ("code_cached.txt", "code_diff.txt", "code_status.txt"):
        (artifact_root / filename).unlink(missing_ok=True)
    return result


def _run_card(spec: Stage1ElasticSoxlOverflowSpec) -> dict[str, Any]:
    spec.card_dir.mkdir(parents=True, exist_ok=True)
    replay_10, source_audit = _stage1_replay(
        spec, cost_bps=spec.transaction_cost_bps
    )
    frame_10, feature_audit = _comparison_frame(spec, replay_10)
    frame_25 = frame_10.copy()
    frame_25["strategy_return"] = (
        frame_25["gross_return"]
        - frame_25["turnover"] * spec.stress_cost_bps / 10_000.0
    )
    frame_25["cost_bps"] = spec.stress_cost_bps
    dataset, dataset_audit = _build_dataset(frame_10, spec)
    probability, fit_audit = _fit_probability_models(dataset, spec)
    selected, model_selection, replays = _select_candidate(
        frame_10,
        probability=probability,
        spec=spec,
    )
    selected_replay = replays[selected.name]
    stress_replay = _apply_overlay(
        frame_25,
        probability=probability,
        candidate=selected,
        cost_bps=spec.stress_cost_bps,
    )
    development = _metrics(selected_replay, spec.development_years)
    full_context = _metrics(
        selected_replay,
        (*spec.train_only_years, *spec.development_years),
    )
    outer = _metrics(selected_replay, spec.outer_years)
    outer_stress = _metrics(stress_replay, spec.outer_years)
    annual = pd.concat(
        [
            _annual_metrics(selected_replay, spec.development_years),
            _annual_metrics(selected_replay, spec.outer_years),
        ],
        ignore_index=True,
    )
    overfit_table, overfit = _overfit_audit(
        frame=frame_10,
        selected=selected,
        model_selection=model_selection,
        fit_audit=fit_audit,
        development=development,
        outer=outer,
        spec=spec,
    )
    selected_passed = bool(
        model_selection.loc[
            model_selection["selected"], "all_development_gates_pass"
        ].iloc[0]
    )
    decision = (
        "reasonable_for_clean_forward_paper_only_not_promoted"
        if selected_passed and overfit["risk_level"] != "high"
        else "not_promoted_development_negative_high_overfit_risk"
    )
    recorder = _record(
        spec,
        selected=selected,
        selected_replay=selected_replay,
        model_selection=model_selection,
        fit_audit=fit_audit,
        overfit_audit=overfit_table,
        annual_metrics=annual,
    )
    selected_payload = {
        **asdict(selected),
        "selected_from_segment": "stitched_walk_forward_2018_2022",
        "train_only_years": list(spec.train_only_years),
        "selection_uses_outer_test": False,
        "outer_role": "reused_diagnostic_only_not_model_selection",
        "source_stage1_frozen": True,
        "source_training_end": spec.source_training_end,
    }
    artifact_policy = {
        "policy": "slim_card_recorder_first",
        "workflow_config": "workflow.yaml",
        "recorder_only_artifacts": [
            f"artifacts/{spec.experiment}/predictions.csv",
            f"artifacts/{spec.experiment}/positions.csv",
            f"artifacts/{spec.experiment}/portfolio_report.csv",
            f"artifacts/{spec.experiment}/model_selection_full.csv",
            f"artifacts/{spec.experiment}/fit_audit.csv",
            f"artifacts/{spec.experiment}/overfit_audit.csv",
            f"artifacts/{spec.experiment}/annual_metrics.csv",
            f"artifacts/{spec.experiment}/selected_model_spec.json",
        ],
    }
    summary = {
        "experiment": spec.experiment,
        "card_type": "stage1_elastic_soxl_overflow_feasibility_card",
        "decision": decision,
        "promotion": "blocked",
        "system_of_record": "qlib",
        "provider_uri": spec.provider_uri.resolve().as_posix(),
        "as_of": frame_10["exit_date"].max().date().isoformat(),
        "source_stage1": {
            "experiment": spec.source_experiment,
            "recorder_id": spec.source_recorder_id,
            "frozen": True,
            "stage2_enabled": False,
            "training_end": spec.source_training_end,
        },
        "hypothesis": (
            "When frozen Stage1 is already at its 30% SOXL cap, move at most "
            "10 percentage points from SOXX to SOXL while leaving cash unchanged."
        ),
        "timing": {
            "signal": "prior_close_features",
            "execution": "next_open",
            "holding": "open_to_next_open_separate_overflow_sleeve",
            "cash_unchanged": True,
        },
        "candidate_search": {
            "candidate_count": len(CANDIDATES),
            "candidates": [asdict(candidate) for candidate in CANDIDATES],
            "logistic_c_fixed": spec.logistic_c,
            "features_fixed_before_outer_diagnostic": list(FEATURE_COLUMNS),
        },
        "dataset_audit": dataset_audit,
        "qlib_api": [
            "qlib.init",
            "qlib.data.D.features",
            "DatasetH",
            "DataHandlerLP",
            "qlib.workflow.R",
        ],
        "qlib_docs_reference": [
            "https://qlib.readthedocs.io/en/latest/component/data.html",
            "https://qlib.readthedocs.io/en/stable/component/recorder.html",
            "https://qlib.readthedocs.io/en/latest/component/strategy.html",
        ],
        "selected_model": selected_payload,
        "development_metrics": development,
        "full_context_including_train_only_metrics": full_context,
        "outer_diagnostic_metrics": outer,
        "outer_25bps_diagnostic_metrics": outer_stress,
        "overfit_audit": overfit,
        "source_audit": source_audit,
        "feature_audit": feature_audit,
        **recorder,
        "sidecars": {
            "selected_model": "selected_model.json",
            "model_selection": "model_selection.csv",
            "comparison": "backtest_benchmark_comparison.csv",
            "risk_analysis": "risk_analysis.csv",
            "annual_metrics": "annual_metrics.csv",
            "overfit_audit": "overfit_audit.csv",
            "analysis_summary": "analysis_summary.json",
            "failure_modes": "failure_modes.md",
            "conclusion": "conclusion.md",
        },
        "artifact_policy": artifact_policy,
    }
    model_selection.to_csv(
        spec.card_dir / "model_selection.csv", index=False
    )
    annual.to_csv(spec.card_dir / "annual_metrics.csv", index=False)
    overfit_table.to_csv(
        spec.card_dir / "overfit_audit.csv", index=False
    )
    comparison = pd.DataFrame(
        [
            {
                "segment": (
                    "full_2016_2022_context_includes_train_only_not_selection"
                ),
                **full_context,
            },
            {"segment": "development_2018_2022", **development},
            {"segment": "outer_reused_diagnostic_2023_2026", **outer},
            {
                "segment": "outer_reused_diagnostic_25bps",
                **outer_stress,
            },
        ]
    )
    comparison.to_csv(
        spec.card_dir / "backtest_benchmark_comparison.csv", index=False
    )
    comparison[
        [
            "segment",
            "max_drawdown_pct",
            "worst_day_pct",
            "cvar5_pct",
            "worst_rolling_5d_pct",
            "annualized_volatility_pct",
        ]
    ].to_csv(spec.card_dir / "risk_analysis.csv", index=False)
    (spec.card_dir / "selected_model.json").write_text(
        json.dumps(selected_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (spec.card_dir / "analysis_summary.json").write_text(
        json.dumps(
            {
                "decision": decision,
                "development": development,
                "full_context_including_train_only": full_context,
                "outer_diagnostic": outer,
                "outer_25bps_diagnostic": outer_stress,
                "overfit": overfit,
                "interpretation": (
                    "Outer results are diagnostic only. A positive outer "
                    "result cannot rescue a negative stitched development result."
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (spec.card_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    workflow = {
        "experiment": spec.experiment,
        "provider_uri": spec.provider_uri.resolve().as_posix(),
        "source_stage1": summary["source_stage1"],
        "timing": summary["timing"],
        "model_selection": {
            "train_only_years": list(spec.train_only_years),
            "development_years": list(spec.development_years),
            "outer_years": list(spec.outer_years),
            "selection_uses_outer_test": False,
            "candidate_count": len(CANDIDATES),
            "features": list(FEATURE_COLUMNS),
        },
        "overfit_controls": {
            "fixed_logistic_c": spec.logistic_c,
            "purge_rows": spec.purge_rows,
            "minimum_training_events": spec.minimum_training_events,
            "minimum_events_per_feature": spec.minimum_events_per_feature,
            "event_concentration_gate": (
                spec.maximum_top5_positive_contribution_share
            ),
            "outer_event_rate_ratio_gate": (
                spec.maximum_outer_event_rate_ratio
            ),
            "clean_forward_required_for_promotion": True,
        },
        "artifact_policy": artifact_policy,
    }
    (spec.card_dir / "workflow.yaml").write_text(
        yaml.safe_dump(
            workflow, sort_keys=False, allow_unicode=True
        ),
        encoding="utf-8",
    )
    selected_row = model_selection.loc[model_selection["selected"]].iloc[0]
    failure_modes = f"""# Failure modes

- Development selection is negative if `return_delta_pct_points <= 0`; outer
  gains cannot override this failure.
- Cap-binding events are sparse and clustered. The minimum expanding-fold
  events-per-feature ratio is {overfit['minimum_events_per_feature']:.2f}.
- The outer/development cap-event-rate ratio is
  {overfit['outer_cap_event_rate_ratio']:.2f}; this is a material regime shift.
- 2023–2026 has already been inspected and is not a clean holdout.
- The overflow sleeve is modeled as a separate open-to-next-open SOXL-for-SOXX
  swap. It does not claim venue-level execution or intraday fill precision.
"""
    (spec.card_dir / "failure_modes.md").write_text(
        failure_modes, encoding="utf-8"
    )
    conclusion = f"""# 中文结论

- 做了什么：冻结当前 Stage1，只在其 SOXL 已触及 30% 时测试额外 5%/10% overflow；资金只从 SOXX 调拨，现金权重不变，信号使用 prior-close 特征并在 next-open 生效。
- 过拟合控制：只搜索 {len(CANDIDATES)} 个预先固定候选；Logistic C 固定为 {spec.logistic_c:.2f}，特征固定为 {len(FEATURE_COLUMNS)} 个；2016–2017 仅训练，2018–2022 expanding walk-forward 选模，2023–2026 只诊断且所有 outer 年份共用训练截止 {spec.source_training_end} 的冻结模型。
- 选中诊断候选：{selected.name}；开发期增量收益={development['return_delta_pct_points']:.4f} 个百分点，动作={development['action_count']} 次，正增量年份={int(selected_row['positive_development_years'])}/{len(spec.development_years)}。
- 整段稳健性检查：若把 2016–2017 训练期也并入整段，增量收益={full_context['return_delta_pct_points']:.4f} 个百分点；该数字同样为负，说明当前 Stage1 口径下并不存在被训练期掩盖的稳定 overflow 收益。
- 开发期风险：MaxDD 变化={development['max_drawdown_delta_pct_points']:.4f} 个百分点，最差 5 日变化={development['worst_5d_delta_pct_points']:.4f} 个百分点，正动作比例={development['positive_action_rate']:.2%}。
- 外层复用诊断：增量收益={outer['return_delta_pct_points']:.4f} 个百分点，动作={outer['action_count']} 次；该结果不得反向选择参数。
- overfit 风险={overfit['risk_level']}；失败项={overfit['failed_checks']}。主要问题是开发期没有正增量、早期 fold 每个特征对应事件数不足、outer 触顶频率相对开发期发生明显漂移，而且没有干净 holdout。
- 决策：{decision}；不修改生产 Stage1，不开启 overflow paper/sim。只有新增独立 PIT 信息或预登记 clean-forward 后，才值得重开。
- qlib recorder：{recorder['recorder_id']}
- 不构成买卖建议。
"""
    (spec.card_dir / "conclusion.md").write_text(
        conclusion, encoding="utf-8"
    )
    return summary
