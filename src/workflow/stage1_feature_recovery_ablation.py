from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nbformat
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data import QlibProviderConfig, load_qlib_feature_frame
from src.workflow.hierarchical_four_asset_policy import _benchmark_efficiency
from src.workflow.intraday_recovery_direct_allocation import (
    FEATURE_COLUMNS,
    IntradayRecoveryDirectAllocationSpec,
    _advance_state,
    _annual_metrics,
    _base_cap,
    _build_teacher_training_frame,
    _development_stability,
    _feature_row,
    _load_source_frame,
    _passes_risk,
    _risk_metrics,
    apply_fixed_decision_cost_stress,
    project_legal_weights,
)


EXPERIMENT = "stage1_feature_recovery_ablation_v2"
CARD_TYPE = "stage1_feature_recovery_ablation_card"

ABSOLUTE_RISK_FEATURES = (
    "vix",
    "underlying_ret_5",
    "underlying_realized_vol_20",
    "underlying_drawdown_60",
    "relative_momentum_20",
    "relative_momentum_60",
    "volatility_ratio_20",
    "intraday_risk_score",
)
NORMALIZED_RISK_FEATURES = (
    "realized_vol_20_z60",
    "drawdown_60_pctile60",
    "trend_efficiency_20_pctile60",
    "tracking_error_5_pctile60",
)
BREADTH_FEATURES = (
    "soxx_const_breadth_ret5_pos_lag1_z60",
    "soxx_const_downside_ret5_lt_m5_lag1_z60",
    "soxx_const_dispersion_ret5_lag1_z60",
)
EXTRA_FEATURES = (
    *ABSOLUTE_RISK_FEATURES,
    *NORMALIZED_RISK_FEATURES,
    *BREADTH_FEATURES,
)
FEATURE_SETS = {
    "baseline_stage1": tuple(FEATURE_COLUMNS),
    "absolute_risk": tuple(FEATURE_COLUMNS) + ABSOLUTE_RISK_FEATURES,
    "absolute_plus_normalized": (
        tuple(FEATURE_COLUMNS)
        + ABSOLUTE_RISK_FEATURES
        + NORMALIZED_RISK_FEATURES
    ),
    "absolute_plus_breadth": (
        tuple(FEATURE_COLUMNS) + ABSOLUTE_RISK_FEATURES + BREADTH_FEATURES
    ),
}
RECORDER_ONLY_ARTIFACTS = (
    f"artifacts/{EXPERIMENT}/positions.csv",
    f"artifacts/{EXPERIMENT}/strategy_returns.csv",
    f"artifacts/{EXPERIMENT}/baseline_strategy_returns.csv",
    f"artifacts/{EXPERIMENT}/recovery_training_frame.csv",
    f"artifacts/{EXPERIMENT}/model_selection_full.csv",
    f"artifacts/{EXPERIMENT}/dataset_audit.json",
)


@dataclass(frozen=True)
class Stage1FeatureRecoveryAblationSpec(
    IntradayRecoveryDirectAllocationSpec
):
    experiment: str = EXPERIMENT
    feature_set_names: tuple[str, ...] = (
        "baseline_stage1",
        "absolute_risk",
        "absolute_plus_normalized",
        "absolute_plus_breadth",
    )
    logistic_c_values: tuple[float, ...] = (0.01, 0.03, 0.10)
    probability_thresholds: tuple[float, ...] = (
        0.50,
        0.55,
        0.60,
        0.65,
        0.6625,
        0.675,
        0.6875,
        0.70,
        0.7125,
        0.725,
        0.7375,
        0.75,
    )
    action_beta_values: tuple[float, ...] = (0.60, 0.80)
    minimum_development_wealth_improvement: float = 0.01
    minimum_neighbor_count: int = 5
    minimum_neighbor_wealth_ratio: float = 0.95
    baseline_feature_set: str = "baseline_stage1"
    baseline_logistic_c: float = 0.03
    baseline_probability_threshold: float = 0.55
    baseline_action_beta: float = 0.60

    @property
    def normalized_provider_uri(self) -> Path:
        return (
            self.root
            / "data"
            / "qlib"
            / "dynamic_soxx_soxl_cash_allocation_wf_v1"
        )


def run_stage1_feature_recovery_ablation_card(
    *, root: Path | str | None = None
) -> dict[str, Any]:
    return _run_card(
        Stage1FeatureRecoveryAblationSpec(root=_resolve_root(root))
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


def _load_enhanced_source_frame(
    spec: Stage1FeatureRecoveryAblationSpec,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    # Resolve the source through the already-validated Stage1 card.  The
    # generic loader reads its caller card's workflow.yaml to find the pinned
    # source recorder; using the new card before it has a workflow would
    # silently fall back to the latest (and methodologically different)
    # source recorder.
    frame, source_audit = _load_source_frame(
        IntradayRecoveryDirectAllocationSpec(root=spec.root)
    )
    fields = [f"${feature}" for feature in EXTRA_FEATURES]
    qlib_features = load_qlib_feature_frame(
        QlibProviderConfig(provider_uri=spec.normalized_provider_uri),
        instruments=["SOXL"],
        fields=fields,
        start=frame["signal_date"].min().date().isoformat(),
        end=frame["signal_date"].max().date().isoformat(),
    ).reset_index()
    qlib_features = qlib_features.drop(columns=["instrument"]).rename(
        columns={
            "datetime": "signal_date",
            **{
                f"${feature}": feature
                for feature in EXTRA_FEATURES
            },
        }
    )
    qlib_features["signal_date"] = pd.to_datetime(
        qlib_features["signal_date"]
    ).dt.normalize()
    frame = frame.merge(
        qlib_features,
        on="signal_date",
        how="left",
        validate="many_to_one",
    )
    manifest_path = spec.normalized_provider_uri / "qlib_store_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    coverage = {
        feature: float(frame[feature].notna().mean())
        for feature in EXTRA_FEATURES
    }
    if min(coverage.values()) < 0.95:
        raise ValueError(
            "blocked: enhanced Stage1 feature coverage below 95%"
        )
    audit = {
        **source_audit,
        "enhanced_feature_provider_uri": (
            spec.normalized_provider_uri.resolve().as_posix()
        ),
        "enhanced_feature_provider_as_of": manifest.get("as_of"),
        "enhanced_feature_timing": (
            "daily features known on signal_date; SOXX constituent "
            "features are explicitly lagged one trading day"
        ),
        "enhanced_feature_count": len(EXTRA_FEATURES),
        "enhanced_feature_coverage": coverage,
        "enhanced_feature_source_api": [
            "qlib.init",
            "qlib.data.D.features",
        ],
    }
    return frame, audit


def _build_recovery_dataset(
    teacher: pd.DataFrame,
) -> tuple[Any, dict[str, Any]]:
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP
    from qlib.data.dataset.loader import StaticDataLoader

    all_features = list(
        dict.fromkeys(
            feature
            for feature_set in FEATURE_SETS.values()
            for feature in feature_set
        )
    )
    labels = ["teacher_action_label", "teacher_action_advantage"]
    static = pd.concat(
        {
            "feature": teacher.loc[:, all_features],
            "label": teacher.loc[:, labels],
        },
        axis=1,
    )
    static.index = pd.MultiIndex.from_arrays(
        [
            pd.to_datetime(teacher["signal_date"]).dt.normalize(),
            np.full(len(teacher), "SOXL_RECOVERY"),
        ],
        names=["datetime", "instrument"],
    )
    years = sorted(int(year) for year in teacher["year"].unique())
    segments = {
        f"train_before_{year}": (
            "2019-01-01",
            f"{year - 1}-12-31",
        )
        for year in years
        if year >= 2020
    }
    segments["all"] = (
        static.index.get_level_values("datetime").min().date().isoformat(),
        static.index.get_level_values("datetime").max().date().isoformat(),
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
        "rows": int(len(dataset.prepare("all"))),
        "feature_count": len(all_features),
        "label_count": len(labels),
        "segments": {
            name: list(bounds) for name, bounds in segments.items()
        },
        "segment_rows": {
            name: int(len(dataset.prepare(name))) for name in segments
        },
        "selection_uses_outer_test": False,
    }
    return dataset, audit


def _prepare_training_segment(dataset: Any, *, year: int) -> pd.DataFrame:
    segment = f"train_before_{year}"
    if segment not in dataset.segments:
        return pd.DataFrame()
    frame = dataset.prepare(segment, col_set=["feature", "label"])
    frame.columns = [str(column[-1]) for column in frame.columns]
    return frame.reset_index().sort_values("datetime")


def _fit_recovery_model(
    dataset: Any,
    *,
    year: int,
    c_value: float,
    feature_columns: tuple[str, ...],
) -> Any | None:
    train = _prepare_training_segment(dataset, year=year)
    if (
        len(train) < 20
        or train["teacher_action_label"].nunique() < 2
    ):
        return None
    model = make_pipeline(
        SimpleImputer(),
        StandardScaler(),
        LogisticRegression(
            C=c_value,
            class_weight="balanced",
            max_iter=2_000,
            random_state=42,
        ),
    )
    model.fit(
        train.loc[:, list(feature_columns)],
        train["teacher_action_label"],
    )
    return model


def _enhanced_feature_row(
    row: Any,
    *,
    drawdown: float,
    cap: float,
    raw_beta: float,
    state: int,
    history: list[float],
) -> dict[str, float]:
    result = _feature_row(
        row,
        drawdown=drawdown,
        cap=cap,
        raw_beta=raw_beta,
        state=state,
        history=history,
    )
    result.update(
        {
            feature: float(getattr(row, feature))
            if pd.notna(getattr(row, feature))
            else np.nan
            for feature in EXTRA_FEATURES
        }
    )
    return result


def simulate_feature_recovery_policy(
    frame: pd.DataFrame,
    *,
    spec: Stage1FeatureRecoveryAblationSpec,
    dataset: Any,
    feature_set_name: str,
    c_value: float,
    threshold: float,
    action_beta: float,
    first_action_year: int,
    model_cache: dict[tuple[str, int, float], Any],
    cost_bps: float,
) -> pd.DataFrame:
    if feature_set_name not in FEATURE_SETS:
        raise ValueError(f"unknown feature set: {feature_set_name}")
    feature_columns = FEATURE_SETS[feature_set_name]
    models: dict[int, Any] = {}
    for year in sorted(frame["signal_date"].dt.year.unique()):
        key = (feature_set_name, int(year), float(c_value))
        if key not in model_cache:
            model_cache[key] = _fit_recovery_model(
                dataset,
                year=int(year),
                c_value=c_value,
                feature_columns=feature_columns,
            )
        models[int(year)] = model_cache[key]

    equity = peak = 1.0
    previous = np.zeros(3)
    state = clock = 0
    history: list[float] = []
    rows: list[dict[str, Any]] = []
    for row in frame.sort_values("signal_date").itertuples():
        drawdown = max(0.0, 1.0 - equity / peak)
        state, clock, q = _advance_state(state, clock, drawdown, spec)
        cap, raw_beta = _base_cap(
            row,
            q=q,
            drawdown=drawdown,
            history=history,
            spec=spec,
        )
        open_weights = project_legal_weights(
            row.soxl_weight,
            row.soxx_weight,
            cap,
            minimum_cash_weight=spec.minimum_cash_weight,
            maximum_soxl_weight=spec.maximum_soxl_weight,
        )
        target_beta = min(raw_beta, action_beta)
        model = models.get(row.signal_date.year)
        eligible = (
            row.signal_date.year >= first_action_year
            and model is not None
            and drawdown >= spec.recovery_eligibility_drawdown
            and target_beta > cap + 1e-12
            and pd.notna(row.early_ret_soxx)
        )
        probability = np.nan
        recovery = False
        if eligible:
            feature = pd.DataFrame(
                [
                    _enhanced_feature_row(
                        row,
                        drawdown=drawdown,
                        cap=cap,
                        raw_beta=raw_beta,
                        state=state,
                        history=history,
                    )
                ]
            )
            probability = float(
                model.predict_proba(
                    feature.loc[:, list(feature_columns)]
                )[0, 1]
            )
            recovery = probability >= threshold

        if recovery:
            close_weights = project_legal_weights(
                row.soxl_weight,
                row.soxx_weight,
                target_beta,
                minimum_cash_weight=spec.minimum_cash_weight,
                maximum_soxl_weight=spec.maximum_soxl_weight,
            )
            early = np.array(
                [row.early_ret_soxl, row.early_ret_soxx, 0.0]
            )
            value_at_cut = 1.0 + open_weights @ early
            drifted = open_weights * (1.0 + early) / value_at_cut
            post = np.array(
                [
                    (1.0 + row.soxl_return)
                    / (1.0 + row.early_ret_soxl)
                    - 1.0,
                    (1.0 + row.soxx_return)
                    / (1.0 + row.early_ret_soxx)
                    - 1.0,
                    0.0,
                ]
            )
            gross_return = (
                value_at_cut * (1.0 + close_weights @ post) - 1.0
            )
            intraday_turnover = float(
                np.abs(close_weights[:2] - drifted[:2]).sum()
            )
        else:
            close_weights = open_weights
            gross_return = (
                open_weights[0] * row.soxl_return
                + open_weights[1] * row.soxx_return
            )
            intraday_turnover = 0.0
        open_turnover = float(
            np.abs(open_weights[:2] - previous[:2]).sum()
        )
        turnover = open_turnover + intraday_turnover
        strategy_return = (
            gross_return - turnover * cost_bps / 10_000.0
        )
        history.append(strategy_return)
        equity *= 1.0 + strategy_return
        peak = max(peak, equity)
        previous = close_weights
        rows.append(
            {
                "signal_date": row.signal_date,
                "trade_date": row.trade_date,
                "exit_date": row.exit_date,
                "segment": row.segment,
                "soxl_return": row.soxl_return,
                "soxx_return": row.soxx_return,
                "open_soxl_weight": open_weights[0],
                "open_soxx_weight": open_weights[1],
                "open_cash_weight": open_weights[2],
                "soxl_weight": close_weights[0],
                "soxx_weight": close_weights[1],
                "cash_weight": close_weights[2],
                "recovery_probability": probability,
                "recovery_action": recovery,
                "current_beta_cap": cap,
                "recovery_beta_target": (
                    target_beta if recovery else cap
                ),
                "gross_return": gross_return,
                "open_turnover": open_turnover,
                "intraday_turnover": intraday_turnover,
                "turnover": turnover,
                "strategy_return": strategy_return,
                "cost_bps": cost_bps,
                **{
                    feature: getattr(row, feature)
                    for feature in EXTRA_FEATURES
                },
            }
        )
    return pd.DataFrame(rows)


def annotate_candidate_neighborhood(
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    required = {
        "feature_set",
        "logistic_c",
        "probability_threshold",
        "action_beta",
        "terminal_wealth",
        "development_gate_pass",
    }
    missing = required.difference(candidates.columns)
    if missing:
        raise ValueError(
            f"candidate table is missing columns: {sorted(missing)}"
        )
    result = candidates.copy()
    result["neighbor_count"] = 0
    result["neighbor_min_wealth_ratio"] = np.nan
    result["neighbor_median_terminal_wealth"] = np.nan
    for index, row in result.iterrows():
        neighbors = result[
            result["feature_set"].eq(row["feature_set"])
            & result["logistic_c"].eq(row["logistic_c"])
            & result["action_beta"].eq(row["action_beta"])
            & result["probability_threshold"].between(
                row["probability_threshold"] - 0.0500001,
                row["probability_threshold"] + 0.0500001,
            )
            & result["development_gate_pass"].astype(bool)
        ]
        result.loc[index, "neighbor_count"] = len(neighbors)
        if len(neighbors):
            result.loc[index, "neighbor_min_wealth_ratio"] = (
                neighbors["terminal_wealth"].min()
                / row["terminal_wealth"]
            )
            result.loc[
                index, "neighbor_median_terminal_wealth"
            ] = neighbors["terminal_wealth"].median()
    return result


def select_feature_recovery_candidate(
    candidates: pd.DataFrame,
    *,
    spec: Stage1FeatureRecoveryAblationSpec,
) -> pd.Series:
    required = {
        "segment",
        "selection_uses_outer_test",
        "feature_set",
        "logistic_c",
        "probability_threshold",
        "action_beta",
        "development_gate_pass",
        "terminal_wealth",
        "total_turnover",
        "neighbor_count",
        "neighbor_min_wealth_ratio",
    }
    missing = required.difference(candidates.columns)
    if missing:
        raise ValueError(
            f"candidate table is missing columns: {sorted(missing)}"
        )
    development = candidates[
        candidates["segment"].eq("development_walk_forward")
        & ~candidates["selection_uses_outer_test"].astype(bool)
    ].copy()
    baseline = development[
        development["feature_set"].eq(spec.baseline_feature_set)
        & development["logistic_c"].eq(spec.baseline_logistic_c)
        & development["probability_threshold"].eq(
            spec.baseline_probability_threshold
        )
        & development["action_beta"].eq(spec.baseline_action_beta)
    ]
    if len(baseline) != 1:
        raise ValueError("baseline Stage1 candidate is missing or duplicated")
    baseline_row = baseline.iloc[0]
    eligible = development[
        development["development_gate_pass"].astype(bool)
        & development["neighbor_count"].ge(spec.minimum_neighbor_count)
        & development["neighbor_min_wealth_ratio"].ge(
            spec.minimum_neighbor_wealth_ratio
        )
        & development["terminal_wealth"].ge(
            float(baseline_row["terminal_wealth"])
            * (1.0 + spec.minimum_development_wealth_improvement)
        )
        & development["total_turnover"].le(
            float(baseline_row["total_turnover"])
        )
    ].copy()
    if eligible.empty:
        return baseline_row
    return eligible.sort_values(
        [
            "terminal_wealth",
            "relative_ratio_p25",
            "total_turnover",
        ],
        ascending=[False, False, True],
    ).iloc[0]


def _candidate_row(
    replay: pd.DataFrame,
    *,
    feature_set: str,
    c_value: float,
    threshold: float,
    action_beta: float,
) -> dict[str, Any]:
    metrics = _risk_metrics(replay["strategy_return"].to_numpy())
    stability = _development_stability(replay)
    gate = bool(
        _passes_risk(metrics)
        and stability["risk_window_pass_count"] == 7
        and stability["positive_relative_window_count"] >= 5
        and stability["relative_ratio_p25"] >= 0.95
    )
    return {
        "segment": "development_walk_forward",
        "selection_uses_outer_test": False,
        "feature_set": feature_set,
        "feature_count": len(FEATURE_SETS[feature_set]),
        "logistic_c": c_value,
        "probability_threshold": threshold,
        "action_beta": action_beta,
        **metrics,
        **stability,
        "total_turnover": float(replay["turnover"].sum()),
        "open_turnover": float(replay["open_turnover"].sum()),
        "intraday_turnover": float(
            replay["intraday_turnover"].sum()
        ),
        "recovery_action_count": int(
            replay["recovery_action"].sum()
        ),
        "avg_cash_weight_pct": float(
            100.0 * replay["cash_weight"].mean()
        ),
        "development_gate_pass": gate,
    }


def _feature_coefficients(
    *,
    model_cache: dict[tuple[str, int, float], Any],
    feature_set: str,
    c_value: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    features = FEATURE_SETS[feature_set]
    for (cached_set, year, cached_c), model in model_cache.items():
        if (
            cached_set != feature_set
            or not np.isclose(cached_c, c_value)
            or model is None
        ):
            continue
        coefficients = model.named_steps[
            "logisticregression"
        ].coef_[0]
        for feature, coefficient in zip(features, coefficients):
            rows.append(
                {
                    "model_year": int(year),
                    "feature_set": feature_set,
                    "feature": feature,
                    "standardized_coefficient": float(coefficient),
                    "absolute_coefficient": abs(float(coefficient)),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["model_year", "absolute_coefficient"],
        ascending=[True, False],
    )


def _regime_comparison(
    selected: pd.DataFrame,
    baseline: pd.DataFrame,
) -> pd.DataFrame:
    merged = selected[
        [
            "signal_date",
            "strategy_return",
            "vix",
            "intraday_risk_score",
            "soxx_const_breadth_ret5_pos_lag1_z60",
        ]
    ].merge(
        baseline[["signal_date", "strategy_return"]].rename(
            columns={"strategy_return": "baseline_return"}
        ),
        on="signal_date",
        how="inner",
        validate="one_to_one",
    )
    masks = {
        "vix_below_20": merged["vix"] < 20.0,
        "vix_20_to_30": merged["vix"].between(20.0, 30.0),
        "vix_above_30": merged["vix"] > 30.0,
        "intraday_risk_high": merged["intraday_risk_score"] >= 0.50,
        "intraday_risk_normal": merged["intraday_risk_score"] < 0.50,
        "breadth_weak": (
            merged["soxx_const_breadth_ret5_pos_lag1_z60"] <= -0.50
        ),
        "breadth_not_weak": (
            merged["soxx_const_breadth_ret5_pos_lag1_z60"] > -0.50
        ),
    }
    rows: list[dict[str, Any]] = []
    for regime, mask in masks.items():
        section = merged.loc[mask]
        if section.empty:
            continue
        selected_metrics = _risk_metrics(
            section["strategy_return"].to_numpy()
        )
        baseline_metrics = _risk_metrics(
            section["baseline_return"].to_numpy()
        )
        rows.append(
            {
                "regime": regime,
                "rows": int(len(section)),
                "selected_return_pct": selected_metrics["return_pct"],
                "baseline_return_pct": baseline_metrics["return_pct"],
                "selected_minus_baseline_return_pct_points": (
                    selected_metrics["return_pct"]
                    - baseline_metrics["return_pct"]
                ),
                "selected_max_drawdown_pct": selected_metrics[
                    "max_drawdown_pct"
                ],
                "baseline_max_drawdown_pct": baseline_metrics[
                    "max_drawdown_pct"
                ],
                "selection_uses_outer_test": False,
            }
        )
    return pd.DataFrame(rows)


def _record_artifacts(
    spec: Stage1FeatureRecoveryAblationSpec,
    *,
    selected_params: dict[str, Any],
    dev_metrics: dict[str, float],
    outer_metrics: dict[str, float],
    stress_metrics: dict[str, float],
    model_selection: pd.DataFrame,
    selected_outer: pd.DataFrame,
    baseline_outer: pd.DataFrame,
    teacher: pd.DataFrame,
    dataset_audit: dict[str, Any],
    source_audit: dict[str, Any],
) -> dict[str, str]:
    from qlib.workflow import R

    recorder_uri = spec.card_dir / "qlib_recorder"
    recorder_uri.mkdir(parents=True, exist_ok=True)
    R.set_uri(recorder_uri.resolve().as_posix())
    with tempfile.TemporaryDirectory(
        prefix=f"{spec.experiment}_"
    ) as temporary:
        temporary_dir = Path(temporary)
        positions = selected_outer[
            [
                "signal_date",
                "trade_date",
                "open_soxl_weight",
                "open_soxx_weight",
                "open_cash_weight",
                "soxl_weight",
                "soxx_weight",
                "cash_weight",
                "recovery_probability",
                "recovery_action",
            ]
        ]
        frames = {
            "positions.csv": positions,
            "strategy_returns.csv": selected_outer,
            "baseline_strategy_returns.csv": baseline_outer,
            "recovery_training_frame.csv": teacher,
            "model_selection_full.csv": model_selection,
        }
        for filename, frame in frames.items():
            frame.to_csv(temporary_dir / filename, index=False)
        (temporary_dir / "dataset_audit.json").write_text(
            json.dumps(dataset_audit, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        with R.start(experiment_name=spec.experiment):
            recorder = R.get_recorder()
            R.log_params(
                source_experiment=spec.source_experiment,
                source_recorder_id=source_audit["source_recorder_id"],
                feature_provider_uri=source_audit[
                    "enhanced_feature_provider_uri"
                ],
                selected_feature_set=selected_params["feature_set"],
                logistic_c=str(selected_params["logistic_c"]),
                probability_threshold=str(
                    selected_params["probability_threshold"]
                ),
                action_beta=str(selected_params["action_beta"]),
                selection_uses_outer_test="false",
                outer_role="diagnostic_only",
            )
            R.log_metrics(
                development_return_pct=dev_metrics["return_pct"],
                development_max_drawdown_pct=dev_metrics[
                    "max_drawdown_pct"
                ],
                outer_return_pct=outer_metrics["return_pct"],
                outer_max_drawdown_pct=outer_metrics[
                    "max_drawdown_pct"
                ],
                stress_25bps_return_pct=stress_metrics["return_pct"],
                stress_25bps_max_drawdown_pct=stress_metrics[
                    "max_drawdown_pct"
                ],
            )
            for filename in (*frames, "dataset_audit.json"):
                recorder.log_artifact(
                    (temporary_dir / filename).as_posix(),
                    artifact_path=spec.experiment,
                )
            result = {
                "recorder_id": recorder.id,
                "recorder_uri": recorder_uri.resolve().as_posix(),
                "recorder_local_dir": Path(
                    recorder.get_local_dir()
                ).resolve().as_posix(),
            }
    recorder_artifacts = Path(result["recorder_local_dir"]) / "artifacts"
    for filename in ("code_cached.txt", "code_diff.txt", "code_status.txt"):
        (recorder_artifacts / filename).unlink(missing_ok=True)
    return result


def _write_notebook(
    spec: Stage1FeatureRecoveryAblationSpec,
    *,
    conclusion: str,
) -> None:
    notebook = nbformat.v4.new_notebook()
    notebook["metadata"]["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook["cells"] = [
        nbformat.v4.new_markdown_cell(
            "# Stage1 feature recovery ablation v2\n\n"
            "假设：只给原 Stage1 recovery head 恢复绝对风险与滞后行业宽度，"
            "能否在不改合法仓位、控制器和 outer 边界的前提下提高超额并降低换手。"
        ),
        nbformat.v4.new_code_cell(
            "from pathlib import Path\n"
            "import sys\n"
            "root = Path.cwd().resolve()\n"
            "while root != root.parent and not "
            "((root / 'AGENTS.md').exists() and (root / 'SPEC.md').exists()):\n"
            "    root = root.parent\n"
            "sys.path.insert(0, str(root))\n"
            "import qlib\n"
            "from qlib.config import REG_US\n"
            "from qlib.data import D\n"
            "from qlib.data.dataset import DatasetH\n"
            "from qlib.workflow import R\n"
            "provider_uri = root / 'data' / 'qlib' / "
            "'dynamic_soxx_soxl_cash_allocation_wf_v1'\n"
            "qlib.init(provider_uri=str(provider_uri), region=REG_US, kernels=1)\n"
            "calendar = D.calendar(freq='day')\n"
            "from src.workflow.stage1_feature_recovery_ablation import "
            "run_stage1_feature_recovery_ablation_card\n"
            "summary = run_stage1_feature_recovery_ablation_card(root=root)\n"
            "{'calendar_end': str(calendar[-1]), "
            "'recorder_id': summary['recorder_id'], "
            "'decision': summary['decision']}"
        ),
        nbformat.v4.new_code_cell(
            "import pandas as pd\n"
            "card = root / 'research' / "
            "'stage1_feature_recovery_ablation_v2'\n"
            "pd.read_csv(card / 'feature_ablation.csv')"
        ),
        nbformat.v4.new_code_cell(
            "pd.read_csv(card / 'risk_analysis.csv')"
        ),
        nbformat.v4.new_markdown_cell(conclusion),
    ]
    nbformat.write(
        notebook,
        spec.card_dir / f"{spec.experiment}.ipynb",
    )


def _write_sidecars(
    spec: Stage1FeatureRecoveryAblationSpec,
    *,
    summary: dict[str, Any],
    selected: pd.Series,
    model_selection: pd.DataFrame,
    feature_ablation: pd.DataFrame,
    neighbor_stability: pd.DataFrame,
    annual_metrics: pd.DataFrame,
    risk_analysis: pd.DataFrame,
    benchmark_comparison: pd.DataFrame,
    regime_metrics: pd.DataFrame,
    feature_coefficients: pd.DataFrame,
) -> None:
    card = spec.card_dir
    selected_is_challenger = (
        summary["selected_model"]["feature_set"]
        != summary["baseline_model"]["feature_set"]
    )
    if selected_is_challenger:
        feature_conclusion = (
            "绝对风险单独和“绝对风险+标准化状态”均未形成可选增益；"
            "只有绝对 VIX/波动/回撤/相对动量与滞后一日 SOXX 宽度共同"
            "进入 recovery head 后，在开发期形成通过五点阈值邻域的候选。"
            "这仍是自适应开发期搜索证据，不等同于新的样本外证明。"
        )
    else:
        feature_conclusion = (
            "新增特征没有同时通过开发期收益、风险、换手和五点阈值邻域门，"
            "因此保留原 Stage1；不能把单个参数点的改善解释为稳定增量。"
        )
    if summary["outer_gate_summary"]["research_gate_pass"]:
        promotion_conclusion = (
            "通过研究门，仅登记为 paper challenger；2023-2026 已被反复"
            "查看，不是干净 holdout，因此仍不替换 production Stage1。"
        )
        next_action = (
            "冻结候选参数，与 baseline 做前向模拟盘影子对照，等待新的压力"
            "阶段后再决定是否替换生产模型。"
        )
    elif selected_is_challenger:
        promotion_conclusion = (
            "开发期候选被冻结 outer 泛化门否决，不进入 paper challenger，"
            "production Stage1 保持不变。"
        )
        next_action = (
            "停止继续调概率阈值；下一轮改成带交易成本和五日尾部惩罚的"
            "连续 advantage 目标，并用独立 tail-veto head 约束恢复动作。"
        )
    else:
        promotion_conclusion = (
            "没有开发期合格增量，保留 production Stage1，也不建立新的 "
            "paper challenger。"
        )
        next_action = (
            "下一轮优先改教师目标和交叉验证结构，不再扩大同一二元分类器"
            "的特征集合。"
        )
    selected_payload = {
        "selection_uses_outer_test": False,
        "selected_from_segment": "development_walk_forward",
        "feature_set": str(selected["feature_set"]),
        "feature_columns": list(
            FEATURE_SETS[str(selected["feature_set"])]
        ),
        "logistic_c": float(selected["logistic_c"]),
        "probability_threshold": float(
            selected["probability_threshold"]
        ),
        "action_beta": float(selected["action_beta"]),
        "development_metrics": {
            key: float(selected[key])
            for key in (
                "return_pct",
                "max_drawdown_pct",
                "worst_day_pct",
                "cvar5_pct",
                "worst_rolling_5d_pct",
                "relative_ratio_p25",
                "total_turnover",
                "avg_cash_weight_pct",
            )
        },
    }
    (card / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (card / "selected_model.json").write_text(
        json.dumps(selected_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    model_selection.sort_values(
        ["selection_eligible", "terminal_wealth"],
        ascending=[False, False],
    ).head(60).to_csv(card / "model_selection.csv", index=False)
    feature_ablation.to_csv(card / "feature_ablation.csv", index=False)
    neighbor_stability.to_csv(
        card / "neighbor_stability.csv", index=False
    )
    annual_metrics.to_csv(card / "annual_metrics.csv", index=False)
    risk_analysis.to_csv(card / "risk_analysis.csv", index=False)
    benchmark_comparison.to_csv(
        card / "backtest_benchmark_comparison.csv", index=False
    )
    regime_metrics.to_csv(card / "regime_metrics.csv", index=False)
    feature_coefficients.to_csv(
        card / "feature_coefficients.csv", index=False
    )
    workflow = f"""experiment: {spec.experiment}
source_experiment: {spec.source_experiment}
source_recorder_id: {summary['source_audit']['source_recorder_id']}
normalized_provider_uri: {spec.normalized_provider_uri.resolve()}
intraday_provider_uri: {spec.intraday_provider_uri.resolve()}
assets: [SOXL, SOXX, cash]
allocation_output: direct_legal_three_asset_weights
selection_segment: development_walk_forward_2016_2022
outer_role: frozen_diagnostic_only
selection_uses_outer_test: false
selected_feature_set: {summary['selected_model']['feature_set']}
transaction_cost_bps: {spec.transaction_cost_bps}
stress_cost_bps: {spec.stress_cost_bps}
artifact_policy:
  policy: slim_card_recorder_first
  workflow_config: workflow.yaml
  recorder_only_artifacts:
    - artifacts/{spec.experiment}/positions.csv
    - artifacts/{spec.experiment}/strategy_returns.csv
    - artifacts/{spec.experiment}/baseline_strategy_returns.csv
    - artifacts/{spec.experiment}/recovery_training_frame.csv
    - artifacts/{spec.experiment}/model_selection_full.csv
    - artifacts/{spec.experiment}/dataset_audit.json
"""
    (card / "workflow.yaml").write_text(workflow, encoding="utf-8")
    conclusion = f"""# 中文结论

- 做了什么：保持原 Stage1 合法 SOXL/SOXX/cash 输出、回撤状态机、第一小时执行时点和教师标签不变，只对 recovery head 做四组特征消融；训练通过 qlib DatasetH，证据写入 qlib Recorder。
- 开发期选择：`{summary['selected_model']['feature_set']}`，Logistic C={summary['selected_model']['logistic_c']:.2f}，阈值={summary['selected_model']['probability_threshold']:.2f}，动作 beta={summary['selected_model']['action_beta']:.1f}。
- 开发期效果：新候选 return={summary['development_metrics_10bps']['return_pct']:.4f}%，原 Stage1={summary['baseline_development_metrics_10bps']['return_pct']:.4f}%，财富提升={summary['development_effect']['terminal_wealth_improvement_pct']:.4f}%；总换手变化={summary['development_effect']['turnover_change_pct']:.4f}%。
- 特征结论：{feature_conclusion}
- outer 10bp 冻结诊断：新候选 return={summary['outer_diagnostic_metrics_10bps']['return_pct']:.4f}%，原 Stage1={summary['baseline_outer_metrics_10bps']['return_pct']:.4f}%，MaxDD={summary['outer_diagnostic_metrics_10bps']['max_drawdown_pct']:.4f}%，worst 5d={summary['outer_diagnostic_metrics_10bps']['worst_rolling_5d_pct']:.4f}%。
- 25bp 固定决策压力：return={summary['fixed_decision_stress_metrics_25bps']['return_pct']:.4f}%，MaxDD={summary['fixed_decision_stress_metrics_25bps']['max_drawdown_pct']:.4f}%，worst 5d={summary['fixed_decision_stress_metrics_25bps']['worst_rolling_5d_pct']:.4f}%。
- 稳健性：选中阈值邻域数量={summary['selection_gates']['selected_neighbor_count']}，邻域最差财富比例={summary['selection_gates']['selected_neighbor_min_wealth_ratio']:.4f}；outer 不参与任何候选、阈值或 beta 选择。
- 是否晋升：`{summary['decision']}`。{promotion_conclusion}
- 下一步：{next_action}
- 不构成买卖建议。
"""
    (card / "conclusion.md").write_text(conclusion, encoding="utf-8")
    selected_annual = annual_metrics[
        annual_metrics["variant"].eq("selected_feature_recovery")
    ]
    baseline_annual = annual_metrics[
        annual_metrics["variant"].eq("baseline_stage1")
    ]
    annual_action_comparison = selected_annual[
        ["year", "recovery_action_count"]
    ].merge(
        baseline_annual[["year", "recovery_action_count"]],
        on="year",
        suffixes=("_selected", "_baseline"),
        validate="one_to_one",
    )
    annual_action_table = "\n".join(
        [
            "| year | selected actions | baseline actions |",
            "| ---: | ---: | ---: |",
            *[
                f"| {int(row.year)} | "
                f"{int(row.recovery_action_count_selected)} | "
                f"{int(row.recovery_action_count_baseline)} |"
                for row in annual_action_comparison.itertuples(index=False)
            ],
        ]
    )
    failure_modes = f"""# Failure modes

- 开发期与 outer 分离：开发期财富提升 {summary['development_effect']['terminal_wealth_improvement_pct']:.4f}%，但冻结 outer 相对原 Stage1 变化 {summary['outer_effect']['terminal_wealth_improvement_pct']:.4f}%。
- 年度相对门：候选仅 {summary['outer_gate_summary']['annual_relative_pass_count']}/4 年跑赢 SOXX，要求至少 3/4。
- 动作漂移：增强分类器在 outer 的固定概率阈值产生明显年份动作稀疏差异；这说明分类概率尺度和标签关系都不能视为跨年稳定。
- 分位阈值诊断：只用历年训练分数校准可增加动作，但 2025 新增动作显著伤害收益，因此问题不只是概率压缩，不能靠调阈值修复。
- 结论：`{summary['decision']}`；outer 只做冻结否决，不反向选择候选。

## Outer recovery actions

{annual_action_table}

## Next objective

1. 预测扣除 10bp 成本后的连续 counterfactual advantage，不再只预测单日正负标签。
2. 独立预测动作后五日最差收益或尾部违约概率，作为 fail-closed veto。
3. 在开发期内部做嵌套 walk-forward 校准；保持 SOXL/SOXX/cash 合法投影、回撤控制器和 outer 边界不变。
"""
    (card / "failure_modes.md").write_text(
        failure_modes,
        encoding="utf-8",
    )
    _write_notebook(spec, conclusion=conclusion)


def _run_card(
    spec: Stage1FeatureRecoveryAblationSpec,
) -> dict[str, Any]:
    spec.card_dir.mkdir(parents=True, exist_ok=True)
    frame, source_audit = _load_enhanced_source_frame(spec)
    teacher = _build_teacher_training_frame(
        frame,
        spec=spec,
        teacher_beta=spec.teacher_recovery_beta,
    )
    teacher = teacher.merge(
        frame[["signal_date", *EXTRA_FEATURES]].drop_duplicates(
            "signal_date"
        ),
        on="signal_date",
        how="left",
        validate="many_to_one",
    )
    dataset, dataset_audit = _build_recovery_dataset(teacher)
    development = frame[
        frame["segment"].eq("development_walk_forward")
    ].copy()
    outer = frame[frame["segment"].eq("outer_diagnostic")].copy()

    model_cache: dict[tuple[str, int, float], Any] = {}
    baseline_development = simulate_feature_recovery_policy(
        development,
        spec=spec,
        dataset=dataset,
        feature_set_name=spec.baseline_feature_set,
        c_value=spec.baseline_logistic_c,
        threshold=spec.baseline_probability_threshold,
        action_beta=spec.baseline_action_beta,
        first_action_year=spec.first_intraday_action_year,
        model_cache=model_cache,
        cost_bps=spec.transaction_cost_bps,
    )
    baseline_dev_metrics = _risk_metrics(
        baseline_development["strategy_return"]
    )
    reference_summary = json.loads(
        (
            spec.root
            / "research"
            / "intraday_recovery_direct_allocation_v1"
            / "summary.json"
        ).read_text(encoding="utf-8")
    )
    reference_dev_metrics = reference_summary[
        "development_metrics_10bps"
    ]
    if not np.isclose(
        baseline_dev_metrics["terminal_wealth"],
        reference_dev_metrics["terminal_wealth"],
        rtol=0.0,
        atol=1e-8,
    ):
        raise ValueError(
            "blocked: feature ablation baseline does not reproduce "
            "the pinned Stage1 development path"
        )
    candidate_rows: list[dict[str, Any]] = []
    for feature_set in spec.feature_set_names:
        for c_value in spec.logistic_c_values:
            for threshold in spec.probability_thresholds:
                for action_beta in spec.action_beta_values:
                    replay = simulate_feature_recovery_policy(
                        development,
                        spec=spec,
                        dataset=dataset,
                        feature_set_name=feature_set,
                        c_value=c_value,
                        threshold=threshold,
                        action_beta=action_beta,
                        first_action_year=spec.first_intraday_action_year,
                        model_cache=model_cache,
                        cost_bps=spec.transaction_cost_bps,
                    )
                    candidate_rows.append(
                        _candidate_row(
                            replay,
                            feature_set=feature_set,
                            c_value=c_value,
                            threshold=threshold,
                            action_beta=action_beta,
                        )
                    )
    model_selection = annotate_candidate_neighborhood(
        pd.DataFrame(candidate_rows)
    )
    selected = select_feature_recovery_candidate(
        model_selection,
        spec=spec,
    )
    baseline_mask = (
        model_selection["feature_set"].eq(spec.baseline_feature_set)
        & model_selection["logistic_c"].eq(spec.baseline_logistic_c)
        & model_selection["probability_threshold"].eq(
            spec.baseline_probability_threshold
        )
        & model_selection["action_beta"].eq(spec.baseline_action_beta)
    )
    baseline_row = model_selection.loc[baseline_mask].iloc[0]
    model_selection["selection_eligible"] = (
        model_selection["development_gate_pass"].astype(bool)
        & model_selection["neighbor_count"].ge(spec.minimum_neighbor_count)
        & model_selection["neighbor_min_wealth_ratio"].ge(
            spec.minimum_neighbor_wealth_ratio
        )
        & model_selection["terminal_wealth"].ge(
            float(baseline_row["terminal_wealth"])
            * (1.0 + spec.minimum_development_wealth_improvement)
        )
        & model_selection["total_turnover"].le(
            float(baseline_row["total_turnover"])
        )
    )
    model_selection["selected"] = (
        model_selection.index == selected.name
    )
    selected_params = {
        "feature_set": str(selected["feature_set"]),
        "feature_columns": list(
            FEATURE_SETS[str(selected["feature_set"])]
        ),
        "logistic_c": float(selected["logistic_c"]),
        "probability_threshold": float(
            selected["probability_threshold"]
        ),
        "action_beta": float(selected["action_beta"]),
    }

    selected_development = simulate_feature_recovery_policy(
        development,
        spec=spec,
        dataset=dataset,
        feature_set_name=selected_params["feature_set"],
        c_value=selected_params["logistic_c"],
        threshold=selected_params["probability_threshold"],
        action_beta=selected_params["action_beta"],
        first_action_year=spec.first_intraday_action_year,
        model_cache=model_cache,
        cost_bps=spec.transaction_cost_bps,
    )
    selected_outer = simulate_feature_recovery_policy(
        outer,
        spec=spec,
        dataset=dataset,
        feature_set_name=selected_params["feature_set"],
        c_value=selected_params["logistic_c"],
        threshold=selected_params["probability_threshold"],
        action_beta=selected_params["action_beta"],
        first_action_year=min(spec.outer_years),
        model_cache=model_cache,
        cost_bps=spec.transaction_cost_bps,
    )
    baseline_outer = simulate_feature_recovery_policy(
        outer,
        spec=spec,
        dataset=dataset,
        feature_set_name=spec.baseline_feature_set,
        c_value=spec.baseline_logistic_c,
        threshold=spec.baseline_probability_threshold,
        action_beta=spec.baseline_action_beta,
        first_action_year=min(spec.outer_years),
        model_cache=model_cache,
        cost_bps=spec.transaction_cost_bps,
    )
    stress_outer = selected_outer.copy()
    stress_outer["strategy_return"] = apply_fixed_decision_cost_stress(
        stress_outer["gross_return"],
        stress_outer["turnover"],
        cost_bps=spec.stress_cost_bps,
    )
    stress_outer["cost_bps"] = spec.stress_cost_bps

    dev_metrics = _risk_metrics(selected_development["strategy_return"])
    outer_metrics = _risk_metrics(selected_outer["strategy_return"])
    baseline_outer_metrics = _risk_metrics(
        baseline_outer["strategy_return"]
    )
    stress_metrics = _risk_metrics(stress_outer["strategy_return"])
    annual_selected = _annual_metrics(selected_outer).assign(
        variant="selected_feature_recovery"
    )
    annual_baseline = _annual_metrics(baseline_outer).assign(
        variant="baseline_stage1"
    )
    annual_metrics = pd.concat(
        [annual_selected, annual_baseline],
        ignore_index=True,
    )
    annual_relative_pass_count = int(
        annual_selected["terminal_wealth_ratio_vs_soxx"].gt(1.0).sum()
    )
    annual_risk_pass_count = int(
        annual_selected.apply(
            lambda row: _passes_risk(row.to_dict()),
            axis=1,
        ).sum()
    )
    benchmark_efficiency = _benchmark_efficiency(
        selected_outer.assign(soxs_weight=0.0),
        cost_bps=int(spec.transaction_cost_bps),
    )
    soxx_wealth = float(
        np.prod(1.0 + selected_outer["soxx_return"].to_numpy())
    )
    terminal_wealth_ratio_vs_soxx = (
        outer_metrics["terminal_wealth"] / soxx_wealth
    )
    selected_is_challenger = (
        selected_params["feature_set"] != spec.baseline_feature_set
    )
    research_gate = bool(
        selected_is_challenger
        and bool(selected["development_gate_pass"])
        and int(selected["neighbor_count"]) >= spec.minimum_neighbor_count
        and float(selected["neighbor_min_wealth_ratio"])
        >= spec.minimum_neighbor_wealth_ratio
        and _passes_risk(outer_metrics)
        and _passes_risk(stress_metrics)
        and annual_relative_pass_count >= 3
        and annual_risk_pass_count >= 3
        and terminal_wealth_ratio_vs_soxx > 1.0
        and benchmark_efficiency["return_maxdd"]
        > benchmark_efficiency["matched_beta_return_maxdd"]
    )
    decision = (
        "paper_challenger_feature_recovery_not_production_promoted"
        if research_gate
        else (
            "retain_stage1_baseline_no_development_feature_gain"
            if not selected_is_challenger
            else "rejected_feature_recovery_outer_generalization_failed"
        )
    )

    feature_ablation = (
        model_selection[
            model_selection["development_gate_pass"].astype(bool)
        ]
        .sort_values(
            ["feature_set", "terminal_wealth"],
            ascending=[True, False],
        )
        .groupby("feature_set", as_index=False)
        .head(1)
        .sort_values("terminal_wealth", ascending=False)
    )
    selected_neighborhood = model_selection[
        model_selection["feature_set"].eq(selected_params["feature_set"])
        & model_selection["logistic_c"].eq(
            selected_params["logistic_c"]
        )
        & model_selection["action_beta"].eq(
            selected_params["action_beta"]
        )
        & model_selection["probability_threshold"].between(
            selected_params["probability_threshold"] - 0.0500001,
            selected_params["probability_threshold"] + 0.0500001,
        )
    ].sort_values("probability_threshold")
    risk_analysis = pd.DataFrame(
        [
            {
                "variant": "baseline_stage1",
                "segment": "development_walk_forward",
                "cost_bps": spec.transaction_cost_bps,
                **baseline_dev_metrics,
                "total_turnover": float(
                    baseline_development["turnover"].sum()
                ),
                "avg_cash_weight_pct": float(
                    100.0 * baseline_development["cash_weight"].mean()
                ),
            },
            {
                "variant": "selected_feature_recovery",
                "segment": "development_walk_forward",
                "cost_bps": spec.transaction_cost_bps,
                **dev_metrics,
                "total_turnover": float(
                    selected_development["turnover"].sum()
                ),
                "avg_cash_weight_pct": float(
                    100.0 * selected_development["cash_weight"].mean()
                ),
            },
            {
                "variant": "baseline_stage1",
                "segment": "outer_diagnostic_reused",
                "cost_bps": spec.transaction_cost_bps,
                **baseline_outer_metrics,
                "total_turnover": float(
                    baseline_outer["turnover"].sum()
                ),
                "avg_cash_weight_pct": float(
                    100.0 * baseline_outer["cash_weight"].mean()
                ),
            },
            {
                "variant": "selected_feature_recovery",
                "segment": "outer_diagnostic_reused",
                "cost_bps": spec.transaction_cost_bps,
                **outer_metrics,
                "total_turnover": float(
                    selected_outer["turnover"].sum()
                ),
                "avg_cash_weight_pct": float(
                    100.0 * selected_outer["cash_weight"].mean()
                ),
            },
            {
                "variant": "selected_feature_recovery_fixed_decisions",
                "segment": "outer_diagnostic_reused",
                "cost_bps": spec.stress_cost_bps,
                **stress_metrics,
                "total_turnover": float(
                    stress_outer["turnover"].sum()
                ),
                "avg_cash_weight_pct": float(
                    100.0 * stress_outer["cash_weight"].mean()
                ),
            },
        ]
    )
    benchmark_comparison = pd.DataFrame(
        [
            {
                "model": "selected_feature_recovery",
                "terminal_wealth_ratio_vs_soxx": (
                    terminal_wealth_ratio_vs_soxx
                ),
                **benchmark_efficiency,
            }
        ]
    )
    regime_metrics = _regime_comparison(
        selected_outer,
        baseline_outer,
    )
    coefficients = _feature_coefficients(
        model_cache=model_cache,
        feature_set=selected_params["feature_set"],
        c_value=selected_params["logistic_c"],
    )
    recorder = _record_artifacts(
        spec,
        selected_params=selected_params,
        dev_metrics=dev_metrics,
        outer_metrics=outer_metrics,
        stress_metrics=stress_metrics,
        model_selection=model_selection,
        selected_outer=selected_outer,
        baseline_outer=baseline_outer,
        teacher=teacher,
        dataset_audit=dataset_audit,
        source_audit=source_audit,
    )
    summary = {
        "experiment": spec.experiment,
        "card_type": CARD_TYPE,
        "decision": decision,
        "production_change": "none",
        "paper_challenger_only": bool(research_gate),
        "clean_unseen_holdout_available": False,
        "selection_uses_outer_test": False,
        "outer_role": "frozen_diagnostic_only_not_model_selection",
        "assets": ["SOXL", "SOXX", "cash"],
        "system_of_record": "qlib",
        "provider_uri": (
            spec.normalized_provider_uri.resolve().as_posix()
        ),
        "intraday_provider_uri": (
            spec.intraday_provider_uri.resolve().as_posix()
        ),
        "as_of": source_audit["enhanced_feature_provider_as_of"],
        "method": {
            "controlled_change": (
                "feature-set ablation on the Stage1 recovery classifier only"
            ),
            "unchanged_components": [
                "frozen source allocation model",
                "legal SOXL/SOXX/cash projection",
                "drawdown controller",
                "10:30 recovery execution",
                "teacher counterfactual label",
            ],
            "daily_feature_timing": (
                "signal-date data; constituent breadth lagged one day"
            ),
            "walk_forward": (
                "annual expanding fit; current year excluded from training"
            ),
            "threshold_grid_design": (
                "coarse 0.50-0.75 grid plus a development-only 0.0125 "
                "sensitivity grid over 0.65-0.75 after the initial "
                "feature ablation; treated as adaptive search, not clean "
                "holdout evidence"
            ),
        },
        "feature_sets": {
            name: list(FEATURE_SETS[name])
            for name in spec.feature_set_names
        },
        "selected_model": selected_params,
        "baseline_model": {
            "feature_set": spec.baseline_feature_set,
            "logistic_c": spec.baseline_logistic_c,
            "probability_threshold": (
                spec.baseline_probability_threshold
            ),
            "action_beta": spec.baseline_action_beta,
        },
        "development_metrics_10bps": dev_metrics,
        "baseline_development_metrics_10bps": baseline_dev_metrics,
        "outer_diagnostic_metrics_10bps": outer_metrics,
        "baseline_outer_metrics_10bps": baseline_outer_metrics,
        "fixed_decision_stress_metrics_25bps": stress_metrics,
        "development_effect": {
            "terminal_wealth_improvement_pct": float(
                100.0
                * (
                    dev_metrics["terminal_wealth"]
                    / baseline_dev_metrics["terminal_wealth"]
                    - 1.0
                )
            ),
            "return_improvement_pct_points": float(
                dev_metrics["return_pct"]
                - baseline_dev_metrics["return_pct"]
            ),
            "turnover_change_pct": float(
                100.0
                * (
                    selected_development["turnover"].sum()
                    / baseline_development["turnover"].sum()
                    - 1.0
                )
            ),
        },
        "outer_effect": {
            "terminal_wealth_improvement_pct": float(
                100.0
                * (
                    outer_metrics["terminal_wealth"]
                    / baseline_outer_metrics["terminal_wealth"]
                    - 1.0
                )
            ),
            "return_improvement_pct_points": float(
                outer_metrics["return_pct"]
                - baseline_outer_metrics["return_pct"]
            ),
            "turnover_change_pct": float(
                100.0
                * (
                    selected_outer["turnover"].sum()
                    / baseline_outer["turnover"].sum()
                    - 1.0
                )
            ),
        },
        "failure_analysis": {
            "development_pass_outer_fail": bool(
                selected_is_challenger and not research_gate
            ),
            "outer_return_gap_vs_baseline_pct_points": float(
                outer_metrics["return_pct"]
                - baseline_outer_metrics["return_pct"]
            ),
            "annual_relative_pass_count": annual_relative_pass_count,
            "selected_recovery_actions_by_year": {
                str(int(year)): int(count)
                for year, count in annual_selected[
                    ["year", "recovery_action_count"]
                ].itertuples(index=False, name=None)
            },
            "baseline_recovery_actions_by_year": {
                str(int(year)): int(count)
                for year, count in annual_baseline[
                    ["year", "recovery_action_count"]
                ].itertuples(index=False, name=None)
            },
            "diagnosis": (
                "enhanced probabilities and label relationship drift "
                "across years; training-score quantile calibration was "
                "diagnosed separately and added harmful 2025 actions"
            ),
        },
        "selection_gates": {
            "development_gate_pass": bool(
                selected["development_gate_pass"]
            ),
            "minimum_development_wealth_improvement": (
                spec.minimum_development_wealth_improvement
            ),
            "selected_neighbor_count": int(
                selected["neighbor_count"]
            ),
            "minimum_neighbor_count": spec.minimum_neighbor_count,
            "selected_neighbor_min_wealth_ratio": float(
                selected["neighbor_min_wealth_ratio"]
            ),
            "minimum_neighbor_wealth_ratio": (
                spec.minimum_neighbor_wealth_ratio
            ),
            "outer_used_for_selection": False,
        },
        "outer_gate_summary": {
            "annual_relative_pass_count": annual_relative_pass_count,
            "annual_risk_pass_count": annual_risk_pass_count,
            "required_year_count": 3,
            "research_gate_pass": research_gate,
        },
        "benchmark_comparison_10bps": {
            "terminal_wealth_ratio_vs_soxx": (
                terminal_wealth_ratio_vs_soxx
            ),
            **benchmark_efficiency,
        },
        "risk_gates": {
            "max_drawdown_floor_pct": -30.0,
            "worst_day_floor_pct": -12.0,
            "cvar5_floor_pct": -8.0,
            "worst_rolling_5d_floor_pct": -12.0,
        },
        "dataset_audit": dataset_audit,
        "source_audit": source_audit,
        "qlib_docs": {
            "data": (
                "https://qlib.readthedocs.io/en/latest/component/data.html"
            ),
            "workflow": (
                "https://qlib.readthedocs.io/en/latest/component/workflow.html"
            ),
            "strategy": (
                "https://qlib.readthedocs.io/en/latest/component/strategy.html"
            ),
        },
        "artifact_policy": {
            "policy": "slim_card_recorder_first",
            "workflow_config": "workflow.yaml",
            "recorder_only_artifacts": list(
                RECORDER_ONLY_ARTIFACTS
            ),
        },
        **recorder,
    }
    _write_sidecars(
        spec,
        summary=summary,
        selected=selected,
        model_selection=model_selection,
        feature_ablation=feature_ablation,
        neighbor_stability=selected_neighborhood,
        annual_metrics=annual_metrics,
        risk_analysis=risk_analysis,
        benchmark_comparison=benchmark_comparison,
        regime_metrics=regime_metrics,
        feature_coefficients=coefficients,
    )
    return summary
