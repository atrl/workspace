from __future__ import annotations

import json
import pickle
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import nbformat
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.factor.high_chop_ml import build_high_chop_ml_frame
from src.workflow.cash_floor_benchmark_relative_wf import _result_metrics
from src.workflow.cash_floor_relative_feature_reintegration import (
    CashFloorRelativeFeatureSpec,
    _load_reintegrated_frame,
)
from src.workflow.soxl_objective_feasibility_audit import _json_ready, _simulate
from src.workflow.trainable_regime_policy_nested_wf import (
    _remove_qlib_code_snapshot_artifacts,
)


EXPERIMENT = "hierarchical_soxx_excess_walk_forward_v1"
DEVELOPMENT_YEARS = tuple(range(2016, 2023))
OUTER_START = "2023-01-03"
TARGETS = ("soxl_relative_alpha_5d", "crash_hazard_5d", "cash_action_5d")
MODEL_FAMILIES = ("linear", "extra_trees")
FEATURE_BUNDLES = ("current", "reintegrated", "full")
RECORDER_ONLY_ARTIFACTS = (
    f"artifacts/{EXPERIMENT}/models/",
    f"artifacts/{EXPERIMENT}/predictions.csv",
    f"artifacts/{EXPERIMENT}/positions.csv",
    f"artifacts/{EXPERIMENT}/strategy_returns.csv",
)

ALPHA_FEATURES = (
    "soxl_ret3",
    "soxl_ret5",
    "soxl_ret10",
    "soxl_ret20",
    "soxl_ret60",
    "soxx_ret3",
    "soxx_ret5",
    "soxx_ret10",
    "soxx_ret20",
    "soxx_ret60",
    "relative_momentum20",
    "soxx_trend20",
    "soxx_trend60",
    "trend_efficiency_20",
    "levered_decay_20",
    "soxx_lead_10",
    "range_pos_252",
    "cross_breadth_pos_20",
    "cross_breadth_pos_60",
    "semis_vs_cross_20",
    "semis_vs_cross_60",
    "tech_breadth_20",
    "tech_breadth_60",
)

HAZARD_FEATURES = (
    "soxx_vol20",
    "soxx_drawdown60",
    "dist_high_252",
    "vix",
    "vix_chg5",
    "vix_z_60",
    "rv20",
    "rv60",
    "rv20_expansion",
    "tracking_error_5_z20",
    "tracking_error_5_pctile60",
    "panic_reversal_5",
    "chop_breakdown_risk",
    "high_chop_pressure",
    "cross_vol_breadth_high",
    "cross_dispersion_20",
    "cross_breadth_pos_20",
    "cross_breadth_pos_60",
    "tech_breadth_20",
    "tech_breadth_60",
)

CASH_FEATURES = tuple(
    dict.fromkeys(
        [
            *HAZARD_FEATURES,
            "soxx_ret5",
            "soxx_ret20",
            "soxx_ret60",
            "soxx_trend20",
            "soxx_trend60",
            "range_pos_252",
            "trend_efficiency_20",
            "relative_momentum20",
        ]
    )
)


@dataclass(frozen=True)
class PolicyProfile:
    name: str
    alpha_threshold: float
    maximum_soxl_share: float
    hazard_threshold: float
    cash_probability_threshold: float
    hazard_soxl_cap_fraction: float
    defense_cash: float
    cash_hold_days: int = 1


POLICY_PROFILES = (
    PolicyProfile("balanced_full_defense", 0.001, 0.40, 0.45, 0.55, 0.00, 0.55),
    PolicyProfile("balanced_full_defense_low_cash", 0.001, 0.40, 0.45, 0.55, 0.00, 0.25),
    PolicyProfile("balanced_half_defense", 0.001, 0.40, 0.45, 0.55, 0.50, 0.55),
    PolicyProfile("balanced_half_defense_hold3", 0.001, 0.40, 0.45, 0.55, 0.50, 0.55, 3),
    PolicyProfile("conservative_full_defense", 0.001, 0.30, 0.45, 0.55, 0.00, 0.55),
    PolicyProfile("conservative_full_defense_hold3", 0.001, 0.30, 0.45, 0.55, 0.00, 0.55, 3),
    PolicyProfile("conservative_rare_cash", 0.001, 0.30, 0.60, 0.60, 0.00, 0.55),
    PolicyProfile("low_leverage", 0.002, 0.20, 0.60, 0.55, 0.00, 0.55),
    PolicyProfile("tail_cash_only", 0.001, 0.20, 0.75, 0.50, 1.00, 0.55),
    PolicyProfile("aggressive_half_defense", 0.001, 0.50, 0.35, 0.50, 0.50, 0.55),
)


@dataclass(frozen=True)
class HierarchicalSoxxExcessSpec:
    root: Path
    experiment: str = EXPERIMENT
    train_start: str = "2011-01-03"
    development_years: tuple[int, ...] = DEVELOPMENT_YEARS
    development_end: str = "2022-12-30"
    outer_start: str = OUTER_START
    purge_rows: int = 20
    minimum_cash: float = 0.10
    cash_cost_hurdle: float = 0.001
    costs_bps: tuple[int, ...] = (10, 25)
    seed: int = 20260723

    @property
    def card_dir(self) -> Path:
        return self.root / "research" / self.experiment

    @property
    def daily_provider_uri(self) -> Path:
        return self.root / "data" / "qlib" / "soxl"


def run_hierarchical_soxx_excess_walk_forward_card(
    *, root: Path | str | None = None
) -> dict[str, Any]:
    repo_root = (
        Path(root).resolve() if root is not None else Path(__file__).resolve().parents[2]
    )
    return _run_card(HierarchicalSoxxExcessSpec(root=repo_root))


def _forward_log_return(returns: pd.Series, horizon: int = 5) -> pd.Series:
    return sum(
        np.log1p(returns.shift(-offset).clip(lower=-0.99))
        for offset in range(horizon)
    )


def _attach_targets(
    frame: pd.DataFrame, spec: HierarchicalSoxxExcessSpec
) -> pd.DataFrame:
    result = frame.copy()
    soxl = pd.to_numeric(result["soxl_forward_open_return"], errors="coerce")
    soxx = pd.to_numeric(result["soxx_forward_open_return"], errors="coerce")
    soxx_forward = pd.concat([soxx.shift(-offset) for offset in range(5)], axis=1)
    soxl_forward = pd.concat([soxl.shift(-offset) for offset in range(5)], axis=1)
    soxx_path = np.exp(np.log1p(soxx_forward.clip(lower=-0.99)).cumsum(axis=1)) - 1.0
    result["soxl_relative_alpha_5d"] = _forward_log_return(
        soxl
    ) - _forward_log_return(soxx)
    result["crash_hazard_5d"] = (
        soxx_path.min(axis=1).le(-0.06)
        | soxx_forward.min(axis=1).le(-0.04)
        | soxl_forward.min(axis=1).le(-0.12)
    ).astype(float)
    defensive_log = _forward_log_return(0.45 * soxx)
    normal_log = _forward_log_return(0.90 * soxx)
    cash_benefit = defensive_log - normal_log
    result["cash_relative_benefit_5d"] = cash_benefit
    result["cash_action_5d"] = cash_benefit.gt(spec.cash_cost_hurdle).astype(float)
    result.loc[cash_benefit.isna(), "cash_action_5d"] = np.nan
    return result


def _load_frame(
    spec: HierarchicalSoxxExcessSpec,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, dict[str, list[str]]], pd.DataFrame]:
    base_spec = CashFloorRelativeFeatureSpec(root=spec.root)
    frame, data_audit, source_sets, source_manifest = _load_reintegrated_frame(
        base_spec
    )
    frame = _attach_targets(frame, spec)
    frame = _attach_explicit_prior(frame, spec)
    current = list(source_sets["current10"])
    full = list(source_sets["full_reintegrated"])
    bundles = {
        "current": {target: current for target in TARGETS},
        "reintegrated": {
            "soxl_relative_alpha_5d": [column for column in ALPHA_FEATURES if column in frame],
            "crash_hazard_5d": [column for column in HAZARD_FEATURES if column in frame],
            "cash_action_5d": [column for column in CASH_FEATURES if column in frame],
        },
        "full": {target: full for target in TARGETS},
    }
    manifest_rows: list[dict[str, Any]] = []
    family_lookup = source_manifest.set_index("feature")["family"].to_dict()
    for bundle, by_head in bundles.items():
        for target, columns in by_head.items():
            for column in columns:
                manifest_rows.append(
                    {
                        "feature_bundle": bundle,
                        "head": target,
                        "feature": column,
                        "family": family_lookup.get(column, "derived_qlib_pit"),
                        "point_in_time": True,
                        "outer_selection": False,
                    }
                )
    data_audit = {
        **data_audit,
        "target_timing": "signal T close; target uses T+1 open through T+6 open",
        "target_horizon_days": 5,
        "cash_cost_hurdle": spec.cash_cost_hurdle,
        "base_constrained_ref_role": "explicit_benchmark_only",
        "base_constrained_ref_in_model_features": False,
    }
    return frame, data_audit, bundles, pd.DataFrame(manifest_rows)


def _attach_explicit_prior(
    frame: pd.DataFrame, spec: HierarchicalSoxxExcessSpec
) -> pd.DataFrame:
    import qlib
    from qlib.data import D

    qlib.init(
        provider_uri=spec.daily_provider_uri.resolve().as_posix(),
        region="us",
        kernels=1,
        skip_if_reg=False,
    )
    raw = D.features(
        ["SOXL", "SOXX"],
        ["$open", "$high", "$low", "$close", "$volume", "$vix"],
        start_time="2010-03-11",
        end_time=frame.index.max(),
        freq="day",
    )
    prior = build_high_chop_ml_frame(
        raw,
        start=frame.index.min().date().isoformat(),
        end=frame.index.max().date().isoformat(),
        include_incomplete_latest=True,
    ).set_index("datetime")
    result = frame.join(
        prior.loc[:, ["base_constrained_ref"]],
        how="left",
    )
    return result


def _build_dataset(
    frame: pd.DataFrame,
    bundles: dict[str, dict[str, list[str]]],
    spec: HierarchicalSoxxExcessSpec,
):
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP
    from qlib.data.dataset.loader import StaticDataLoader

    feature_columns = list(
        dict.fromkeys(
            column
            for by_head in bundles.values()
            for columns in by_head.values()
            for column in columns
        )
    )
    static = pd.concat(
        {"feature": frame.loc[:, feature_columns], "label": frame.loc[:, TARGETS]},
        axis=1,
    )
    static.index = pd.MultiIndex.from_arrays(
        [frame.index, np.full(len(frame), "SOXL_SOXX_CASH")],
        names=["datetime", "instrument"],
    )
    outer_end = frame.index.max().date().isoformat()
    segments: dict[str, tuple[str, str]] = {
        "development": (
            f"{spec.development_years[0]}-01-01",
            spec.development_end,
        ),
        "outer": (spec.outer_start, outer_end),
    }
    for year in spec.development_years:
        segments[f"fold_{year}"] = (f"{year}-01-01", f"{year}-12-31")
    handler = DataHandlerLP(
        instruments=None,
        start_time=frame.index.min(),
        end_time=frame.index.max(),
        data_loader=StaticDataLoader(config=static),
    )
    dataset = DatasetH(handler=handler, segments=segments)
    return dataset, {
        "api": ["DatasetH", "DataHandlerLP", "StaticDataLoader"],
        "segments": {name: list(bounds) for name, bounds in segments.items()},
        "segment_rows": {name: int(len(dataset.prepare(name))) for name in segments},
        "purge_rows": spec.purge_rows,
        "selection_uses_outer_test": False,
    }


def _prepare(dataset, segment: str) -> pd.DataFrame:
    result = dataset.prepare(segment, col_set=["feature", "label"])
    result = result.droplevel("instrument").sort_index()
    result.columns = [str(column[-1]) for column in result.columns]
    return result


def _model(family: str, target: str, seed: int):
    classification = target in {"crash_hazard_5d", "cash_action_5d"}
    if family == "linear":
        estimator = (
            LogisticRegression(
                C=0.25,
                class_weight="balanced",
                max_iter=2_000,
                random_state=seed,
            )
            if classification
            else Ridge(alpha=20.0)
        )
        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", estimator),
            ]
        )
    estimator = (
        ExtraTreesClassifier(
            n_estimators=180,
            max_depth=6,
            min_samples_leaf=15,
            max_features=0.70,
            class_weight="balanced",
            random_state=seed,
            n_jobs=1,
        )
        if classification
        else ExtraTreesRegressor(
            n_estimators=180,
            max_depth=6,
            min_samples_leaf=15,
            max_features=0.70,
            random_state=seed,
            n_jobs=1,
        )
    )
    return Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("model", estimator)]
    )


def _prediction(model, data: pd.DataFrame, target: str) -> np.ndarray:
    if target in {"crash_hazard_5d", "cash_action_5d"}:
        return model.predict_proba(data)[:, 1]
    return model.predict(data)


def _walk_forward_predictions(
    dataset,
    frame: pd.DataFrame,
    bundles: dict[str, dict[str, list[str]]],
    spec: HierarchicalSoxxExcessSpec,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    outputs: list[pd.DataFrame] = []
    models: dict[str, Any] = {}
    for bundle in FEATURE_BUNDLES:
        for family in MODEL_FAMILIES:
            for year in spec.development_years:
                test = _prepare(dataset, f"fold_{year}")
                test = test.loc[test.index.year == year].dropna(subset=list(TARGETS))
                start = frame.index.searchsorted(pd.Timestamp(f"{year}-01-01"))
                train_end = frame.index[start - spec.purge_rows - 1]
                train = frame.loc[spec.train_start:train_end].dropna(
                    subset=list(TARGETS)
                )
                output = pd.DataFrame(index=test.index)
                output["segment"] = "development_walk_forward"
                output["fold_year"] = year
                output["feature_bundle"] = bundle
                output["model_family"] = family
                output["train_end"] = train_end
                for target in TARGETS:
                    features = bundles[bundle][target]
                    head_train = (
                        train.loc[train["crash_hazard_5d"].eq(1.0)]
                        if target == "cash_action_5d"
                        else train
                    )
                    fitted = _model(family, target, spec.seed)
                    fitted.fit(head_train.loc[:, features], head_train[target])
                    output[f"{target}_actual"] = test[target]
                    output[f"{target}_prediction"] = _prediction(
                        fitted, test.loc[:, features], target
                    )
                    models[f"{bundle}__{family}__fold_{year}__{target}"] = fitted
                outputs.append(output)
    return pd.concat(outputs).sort_index(), models


def _outer_predictions(
    dataset,
    frame: pd.DataFrame,
    bundles: dict[str, dict[str, list[str]]],
    selected: dict[str, Any],
    spec: HierarchicalSoxxExcessSpec,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    outer = _prepare(dataset, "outer").dropna(subset=list(TARGETS))
    start = frame.index.searchsorted(pd.Timestamp(spec.outer_start))
    train_end = frame.index[start - spec.purge_rows - 1]
    train = frame.loc[spec.train_start:train_end].dropna(subset=list(TARGETS))
    output = pd.DataFrame(index=outer.index)
    output["segment"] = "outer_diagnostic"
    output["fold_year"] = output.index.year
    output["feature_bundle"] = selected["feature_bundle"]
    output["model_family"] = selected["model_family"]
    output["train_end"] = train_end
    models: dict[str, Any] = {}
    for target in TARGETS:
        features = bundles[str(selected["feature_bundle"])][target]
        head_train = (
            train.loc[train["crash_hazard_5d"].eq(1.0)]
            if target == "cash_action_5d"
            else train
        )
        fitted = _model(str(selected["model_family"]), target, spec.seed)
        fitted.fit(head_train.loc[:, features], head_train[target])
        output[f"{target}_actual"] = outer[target]
        output[f"{target}_prediction"] = _prediction(
            fitted, outer.loc[:, features], target
        )
        models[f"outer_refit__{target}"] = fitted
    return output, models


def _held_action(raw: np.ndarray, hold_days: int) -> np.ndarray:
    action = np.asarray(raw, dtype=bool).copy()
    if hold_days <= 1:
        return action
    held = action.copy()
    for offset in range(1, hold_days):
        held[offset:] |= action[:-offset]
    return held


def _policy_weights(
    predictions: pd.DataFrame,
    profile: PolicyProfile,
    spec: HierarchicalSoxxExcessSpec,
) -> tuple[np.ndarray, pd.DataFrame]:
    alpha = predictions["soxl_relative_alpha_5d_prediction"].to_numpy(dtype=float)
    hazard_probability = predictions["crash_hazard_5d_prediction"].to_numpy(
        dtype=float
    )
    cash_probability = predictions["cash_action_5d_prediction"].to_numpy(
        dtype=float
    )
    hazard = hazard_probability >= profile.hazard_threshold
    cash_entry = hazard & (
        cash_probability >= profile.cash_probability_threshold
    )
    cash_action = _held_action(cash_entry, profile.cash_hold_days)
    soxl_share = np.where(
        alpha > profile.alpha_threshold,
        profile.maximum_soxl_share,
        0.0,
    )
    hazard_cap = profile.maximum_soxl_share * profile.hazard_soxl_cap_fraction
    soxl_share[hazard] = np.minimum(soxl_share[hazard], hazard_cap)
    cash = np.where(cash_action, profile.defense_cash, spec.minimum_cash)
    risky = 1.0 - cash
    weights = np.column_stack(
        [
            risky * soxl_share,
            risky * (1.0 - soxl_share),
            np.zeros(len(predictions)),
            cash,
        ]
    )
    actions = pd.DataFrame(
        {
            "hazard_active": hazard,
            "cash_entry": cash_entry,
            "cash_action": cash_action,
            "alpha_active": alpha > profile.alpha_threshold,
        },
        index=predictions.index,
    )
    return weights, actions


def _drawdown_duration(returns: pd.Series) -> dict[str, int]:
    wealth = (1.0 + returns.astype(float)).cumprod()
    peak = wealth.cummax()
    underwater = wealth.lt(peak - 1e-12).to_numpy()
    longest = current = 0
    for value in underwater:
        current = current + 1 if value else 0
        longest = max(longest, current)
    drawdown = wealth / peak - 1.0
    trough = int(np.argmin(drawdown.to_numpy()))
    peak_value = float(peak.iloc[trough])
    later = wealth.iloc[trough + 1 :]
    recovered = np.flatnonzero(later.to_numpy() >= peak_value - 1e-12)
    recovery = int(recovered[0] + 1) if len(recovered) else int(len(later))
    return {
        "longest_underwater_days": int(longest),
        "max_drawdown_recovery_days": recovery,
    }


def _evaluate(
    frame: pd.DataFrame,
    predictions: pd.DataFrame,
    profile: PolicyProfile,
    cost_bps: int,
    spec: HierarchicalSoxxExcessSpec,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    market = frame.loc[predictions.index]
    weights, actions = _policy_weights(predictions, profile, spec)
    strategy = _simulate(
        market,
        weights,
        cost_bps=cost_bps,
        name=profile.name,
    )
    benchmark = _simulate(
        market,
        np.tile([0.0, 1.0, 0.0, 0.0], (len(market), 1)),
        cost_bps=cost_bps,
        name="soxx_buyhold",
    )
    metrics = _result_metrics(strategy, benchmark)
    strategy_duration = _drawdown_duration(strategy["strategy_return"])
    benchmark_duration = _drawdown_duration(benchmark["strategy_return"])
    downside = pd.to_numeric(
        market["soxx_trend60"], errors="coerce"
    ).lt(0.0).to_numpy()
    metrics.update(
        {
            **strategy_duration,
            "soxx_max_drawdown_pct": _result_metrics(benchmark, benchmark)[
                "max_drawdown_pct"
            ],
            "soxx_longest_underwater_days": benchmark_duration[
                "longest_underwater_days"
            ],
            "soxx_max_drawdown_recovery_days": benchmark_duration[
                "max_drawdown_recovery_days"
            ],
            "cash_entry_days": int(actions["cash_entry"].sum()),
            "cash_action_days": int(actions["cash_action"].sum()),
            "hazard_days": int(actions["hazard_active"].sum()),
            "downside_avg_cash_weight_pct": 100.0
            * float(strategy.loc[downside, "cash_weight"].mean()),
            "downside_min_cash_weight_pct": 100.0
            * float(strategy.loc[downside, "cash_weight"].min()),
            "turnover": float(strategy["turnover"].sum()),
        }
    )
    detail = strategy.join(actions.reset_index(drop=True))
    return metrics, detail, benchmark


def _annual_metrics(
    result: pd.DataFrame, benchmark: pd.DataFrame, metadata: dict[str, Any]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    years = pd.to_datetime(result["signal_date"]).dt.year
    for year in sorted(years.unique()):
        mask = years.eq(year)
        metrics = _result_metrics(
            result.loc[mask].reset_index(drop=True),
            benchmark.loc[mask].reset_index(drop=True),
        )
        rows.append({"fold_year": int(year), **metadata, **metrics})
    return rows


def _development_candidates(
    frame: pd.DataFrame,
    predictions: pd.DataFrame,
    spec: HierarchicalSoxxExcessSpec,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    annual: list[dict[str, Any]] = []
    for bundle in FEATURE_BUNDLES:
        for family in MODEL_FAMILIES:
            selected_predictions = predictions.loc[
                predictions["feature_bundle"].eq(bundle)
                & predictions["model_family"].eq(family)
            ].sort_index()
            for profile in POLICY_PROFILES:
                for cost_bps in spec.costs_bps:
                    metrics, result, benchmark = _evaluate(
                        frame,
                        selected_predictions,
                        profile,
                        cost_bps,
                        spec,
                    )
                    metadata = {
                        "candidate": f"{bundle}__{family}__{profile.name}",
                        "feature_bundle": bundle,
                        "model_family": family,
                        "policy_profile": profile.name,
                        "cost_bps": cost_bps,
                        "selection_uses_outer_test": False,
                    }
                    rows.append({**metadata, **metrics})
                    annual.extend(_annual_metrics(result, benchmark, metadata))
    return pd.DataFrame(rows), pd.DataFrame(annual)


def _select_candidate(
    comparison: pd.DataFrame,
    annual: pd.DataFrame,
    head_audit: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame]:
    ten = comparison.loc[comparison["cost_bps"].eq(10)].copy()
    stress = comparison.loc[
        comparison["cost_bps"].eq(25),
        ["candidate", "terminal_wealth_ratio_vs_soxx"],
    ].rename(columns={"terminal_wealth_ratio_vs_soxx": "ratio_25bps"})
    years = annual.loc[annual["cost_bps"].eq(10)].groupby("candidate").agg(
        positive_relative_years=(
            "terminal_wealth_ratio_vs_soxx",
            lambda values: int((values > 1.0).sum()),
        ),
        median_annual_ratio=("terminal_wealth_ratio_vs_soxx", "median"),
    )
    selection = ten.merge(stress, on="candidate").merge(years, on="candidate")
    selection = selection.rename(
        columns={"terminal_wealth_ratio_vs_soxx": "ratio_10bps"}
    )
    head_quality = _head_quality_summary(head_audit)
    selection = selection.merge(
        head_quality,
        on=["feature_bundle", "model_family"],
        how="left",
    )
    selection["return_gate"] = (
        selection["ratio_10bps"].ge(1.20)
        & selection["ratio_25bps"].gt(1.00)
        & selection["positive_relative_years"].ge(4)
        & selection["median_annual_ratio"].gt(1.0)
    )
    selection["drawdown_gate"] = selection["max_drawdown_pct"].gt(
        selection["soxx_max_drawdown_pct"]
    )
    selection["recovery_gate"] = (
        selection["longest_underwater_days"].lt(
            selection["soxx_longest_underwater_days"]
        )
        & selection["max_drawdown_recovery_days"].le(
            selection["soxx_max_drawdown_recovery_days"]
        )
    )
    selection["cash_gate"] = (
        selection["min_cash_weight_pct"].ge(10.0 - 1e-9)
        & selection["downside_min_cash_weight_pct"].ge(10.0 - 1e-9)
        & selection["downside_avg_cash_weight_pct"].gt(10.0)
        & selection["cash_entry_days"].gt(0)
    )
    selection["all_gate"] = (
        selection["return_gate"]
        & selection["drawdown_gate"]
        & selection["recovery_gate"]
        & selection["cash_gate"]
        & selection["head_quality_gate"]
    )
    selection["gate_count"] = selection[
        [
            "return_gate",
            "drawdown_gate",
            "recovery_gate",
            "cash_gate",
            "head_quality_gate",
        ]
    ].sum(axis=1)
    selection["drawdown_cushion_pct"] = (
        selection["max_drawdown_pct"] - selection["soxx_max_drawdown_pct"]
    )
    selection["selection_score"] = (
        2.0 * np.log(selection["ratio_10bps"].clip(lower=1e-9))
        + np.log(selection["ratio_25bps"].clip(lower=1e-9))
        + 0.04 * selection["drawdown_cushion_pct"]
        + 0.002
        * (
            selection["soxx_longest_underwater_days"]
            - selection["longest_underwater_days"]
        )
        - 0.0005 * selection["turnover"]
    )
    eligible = selection.loc[selection["all_gate"]]
    policy = "all_gate_then_robustness_score"
    if eligible.empty:
        quality = selection.loc[selection["head_quality_gate"]]
        if not quality.empty:
            eligible = quality.loc[
                quality["gate_count"].eq(quality["gate_count"].max())
            ]
            policy = "diagnostic_fallback_head_quality_then_maximum_gate_count"
        else:
            eligible = selection.loc[
                selection["gate_count"].eq(selection["gate_count"].max())
            ]
            policy = "diagnostic_fallback_no_head_quality_maximum_gate_count"
    selected = eligible.sort_values(
        ["selection_score", "ratio_25bps"],
        ascending=False,
    ).iloc[0]
    selection["selected"] = selection["candidate"].eq(selected["candidate"])
    return _json_ready(
        {
            **selected.to_dict(),
            "selection_policy": policy,
            "selection_uses_outer_test": False,
        }
    ), selection


def _selected_profile(selected: dict[str, Any]) -> PolicyProfile:
    return next(
        profile
        for profile in POLICY_PROFILES
        if profile.name == selected["policy_profile"]
    )


def _outer_evaluation(
    frame: pd.DataFrame,
    predictions: pd.DataFrame,
    selected: dict[str, Any],
    spec: HierarchicalSoxxExcessSpec,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    annual: list[dict[str, Any]] = []
    stored: list[pd.DataFrame] = []
    profile = _selected_profile(selected)
    for cost_bps in spec.costs_bps:
        metrics, result, benchmark = _evaluate(
            frame, predictions, profile, cost_bps, spec
        )
        metadata = {
            "segment": "outer_diagnostic",
            "candidate": selected["candidate"],
            "feature_bundle": selected["feature_bundle"],
            "model_family": selected["model_family"],
            "policy_profile": selected["policy_profile"],
            "cost_bps": cost_bps,
            "selection_uses_outer_test": False,
        }
        rows.append({**metadata, **metrics})
        annual.extend(_annual_metrics(result, benchmark, metadata))
        stored.append(
            result.assign(segment="outer_diagnostic", cost_bps=cost_bps)
        )
    return pd.DataFrame(rows), pd.DataFrame(annual), pd.concat(stored)


def _head_oos_audit(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (bundle, family), group in predictions.groupby(
        ["feature_bundle", "model_family"]
    ):
        for target in TARGETS:
            evaluation = (
                group.loc[group["crash_hazard_5d_actual"].eq(1.0)]
                if target == "cash_action_5d"
                else group
            )
            actual = evaluation[f"{target}_actual"].astype(float)
            prediction = evaluation[f"{target}_prediction"].astype(float)
            if target == "soxl_relative_alpha_5d":
                pooled = spearmanr(prediction, actual, nan_policy="omit").statistic
                folds = [
                    spearmanr(
                        fold[f"{target}_prediction"],
                        fold[f"{target}_actual"],
                        nan_policy="omit",
                    ).statistic
                    for _, fold in evaluation.groupby("fold_year")
                ]
                metric_name = "spearman_ic"
                threshold = 0.0
            else:
                pooled = (
                    roc_auc_score(actual, prediction)
                    if actual.nunique() == 2
                    else np.nan
                )
                folds = [
                    roc_auc_score(
                        fold[f"{target}_actual"],
                        fold[f"{target}_prediction"],
                    )
                    if fold[f"{target}_actual"].nunique() == 2
                    else np.nan
                    for _, fold in evaluation.groupby("fold_year")
                ]
                metric_name = "roc_auc"
                threshold = 0.5
            rows.append(
                {
                    "feature_bundle": bundle,
                    "model_family": family,
                    "head": target,
                    "evaluation_scope": (
                        "actual_crash_hazard_rows"
                        if target == "cash_action_5d"
                        else "all_rows"
                    ),
                    "metric_name": metric_name,
                    "pooled_metric": pooled,
                    "median_fold_metric": float(np.nanmedian(folds)),
                    "positive_fold_count": int(
                        (np.asarray(folds, dtype=float) > threshold).sum()
                    ),
                    "fold_count": len(folds),
                    "selection_uses_outer_test": False,
                }
            )
    return pd.DataFrame(rows)


def _head_quality_summary(head_audit: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (bundle, family), group in head_audit.groupby(
        ["feature_bundle", "model_family"]
    ):
        by_head = group.set_index("head")
        alpha = by_head.loc["soxl_relative_alpha_5d"]
        hazard = by_head.loc["crash_hazard_5d"]
        cash = by_head.loc["cash_action_5d"]
        alpha_pass = bool(
            alpha["pooled_metric"] > 0.0
            and alpha["positive_fold_count"] >= 5
        )
        hazard_pass = bool(
            hazard["pooled_metric"] >= 0.60
            and hazard["positive_fold_count"] >= 5
        )
        cash_pass = bool(
            cash["pooled_metric"] >= 0.55
            and cash["positive_fold_count"] >= 4
        )
        rows.append(
            {
                "feature_bundle": bundle,
                "model_family": family,
                "alpha_head_gate": alpha_pass,
                "hazard_head_gate": hazard_pass,
                "cash_head_gate": cash_pass,
                "head_quality_gate": alpha_pass and hazard_pass and cash_pass,
            }
        )
    return pd.DataFrame(rows)


def _outer_head_transfer_audit(
    dataset,
    frame: pd.DataFrame,
    bundles: dict[str, dict[str, list[str]]],
    spec: HierarchicalSoxxExcessSpec,
) -> pd.DataFrame:
    outer = _prepare(dataset, "outer").dropna(subset=list(TARGETS))
    start = frame.index.searchsorted(pd.Timestamp(spec.outer_start))
    train_end = frame.index[start - spec.purge_rows - 1]
    train = frame.loc[spec.train_start:train_end].dropna(subset=list(TARGETS))
    rows: list[dict[str, Any]] = []
    for bundle in FEATURE_BUNDLES:
        for family in MODEL_FAMILIES:
            for target in TARGETS:
                features = bundles[bundle][target]
                head_train = (
                    train.loc[train["crash_hazard_5d"].eq(1.0)]
                    if target == "cash_action_5d"
                    else train
                )
                evaluation = (
                    outer.loc[outer["crash_hazard_5d"].eq(1.0)]
                    if target == "cash_action_5d"
                    else outer
                )
                fitted = _model(family, target, spec.seed)
                fitted.fit(head_train.loc[:, features], head_train[target])
                prediction = _prediction(
                    fitted, evaluation.loc[:, features], target
                )
                actual = evaluation[target].to_numpy(dtype=float)
                if target == "soxl_relative_alpha_5d":
                    metric_name = "spearman_ic"
                    metric = spearmanr(
                        prediction, actual, nan_policy="omit"
                    ).statistic
                else:
                    metric_name = "roc_auc"
                    metric = (
                        roc_auc_score(actual, prediction)
                        if len(np.unique(actual)) == 2
                        else np.nan
                    )
                rows.append(
                    {
                        "segment": "outer_diagnostic",
                        "feature_bundle": bundle,
                        "model_family": family,
                        "head": target,
                        "evaluation_scope": (
                            "actual_crash_hazard_rows"
                            if target == "cash_action_5d"
                            else "all_rows"
                        ),
                        "metric_name": metric_name,
                        "metric": metric,
                        "rows": len(evaluation),
                        "train_end": train_end.date().isoformat(),
                        "selection_uses_outer_test": False,
                    }
                )
    return pd.DataFrame(rows)


def _regime_metrics(
    frame: pd.DataFrame, returns: pd.DataFrame
) -> pd.DataFrame:
    selected = returns.loc[returns["cost_bps"].eq(10)].copy()
    market = frame.loc[pd.to_datetime(selected["signal_date"])].copy()
    regimes = {
        "all": np.ones(len(selected), dtype=bool),
        "downtrend": pd.to_numeric(
            market["soxx_trend60"], errors="coerce"
        ).lt(0.0).to_numpy(),
        "high_vol": pd.to_numeric(market["soxx_vol20"], errors="coerce")
        .ge(pd.to_numeric(market["soxx_vol20"], errors="coerce").median())
        .to_numpy(),
        "positive_soxx": pd.to_numeric(
            market["soxx_forward_open_return"], errors="coerce"
        ).gt(0.0).to_numpy(),
    }
    rows: list[dict[str, Any]] = []
    for regime, mask in regimes.items():
        sliced = selected.loc[mask]
        wealth = float((1.0 + sliced["strategy_return"]).prod())
        soxx_wealth = float((1.0 + sliced["soxx_return"]).prod())
        positive = sliced["soxx_return"].gt(0.0)
        capture = (
            float(
                sliced.loc[positive, "strategy_return"].sum()
                / sliced.loc[positive, "soxx_return"].sum()
            )
            if positive.any()
            else np.nan
        )
        rows.append(
            {
                "regime": regime,
                "rows": int(mask.sum()),
                "return_pct": 100.0 * (wealth - 1.0),
                "soxx_return_pct": 100.0 * (soxx_wealth - 1.0),
                "terminal_wealth_ratio_vs_soxx": wealth / soxx_wealth,
                "avg_cash_weight_pct": 100.0 * float(sliced["cash_weight"].mean()),
                "avg_soxl_weight_pct": 100.0 * float(sliced["soxl_weight"].mean()),
                "positive_soxx_capture": capture,
            }
        )
    return pd.DataFrame(rows)


def _explicit_prior_comparison(
    frame: pd.DataFrame,
    predictions: pd.DataFrame,
    selected: dict[str, Any],
    spec: HierarchicalSoxxExcessSpec,
) -> pd.DataFrame:
    market = frame.loc[predictions.index]
    profile = _selected_profile(selected)
    selected_weights, _ = _policy_weights(predictions, profile, spec)
    prior_position = pd.to_numeric(
        market["base_constrained_ref"], errors="coerce"
    ).fillna(0.0).clip(0.0, 1.0).to_numpy()
    candidates = {
        "selected_hierarchical_policy": selected_weights,
        "soxx_buyhold": np.tile([0.0, 1.0, 0.0, 0.0], (len(market), 1)),
        "base_constrained_ref": np.column_stack(
            [
                prior_position,
                np.zeros(len(market)),
                np.zeros(len(market)),
                1.0 - prior_position,
            ]
        ),
    }
    benchmark = _simulate(
        market,
        candidates["soxx_buyhold"],
        cost_bps=10,
        name="soxx_buyhold",
    )
    rows: list[dict[str, Any]] = []
    for name, weights in candidates.items():
        result = _simulate(market, weights, cost_bps=10, name=name)
        rows.append(
            {
                "strategy": name,
                "role": (
                    "explicit_prior_benchmark_only"
                    if name == "base_constrained_ref"
                    else "primary_comparison"
                ),
                **_result_metrics(result, benchmark),
                **_drawdown_duration(result["strategy_return"]),
            }
        )
    return pd.DataFrame(rows)


def _action_attribution(strategy_returns: pd.DataFrame) -> pd.DataFrame:
    selected = strategy_returns.loc[strategy_returns["cost_bps"].eq(10)].copy()
    selected["year"] = pd.to_datetime(selected["signal_date"]).dt.year
    rows: list[dict[str, Any]] = []
    for (segment, year), group in selected.groupby(["segment", "year"]):
        strategy_wealth = float((1.0 + group["strategy_return"]).prod())
        soxx_wealth = float((1.0 + group["soxx_return"]).prod())
        rows.append(
            {
                "segment": segment,
                "year": int(year),
                "rows": len(group),
                "strategy_return_pct": 100.0 * (strategy_wealth - 1.0),
                "soxx_return_pct": 100.0 * (soxx_wealth - 1.0),
                "terminal_wealth_ratio_vs_soxx": strategy_wealth / soxx_wealth,
                "avg_soxl_weight_pct": 100.0 * float(group["soxl_weight"].mean()),
                "avg_cash_weight_pct": 100.0 * float(group["cash_weight"].mean()),
                "hazard_days": int(group["hazard_active"].sum()),
                "cash_entry_days": int(group["cash_entry"].sum()),
                "cash_action_days": int(group["cash_action"].sum()),
                "turnover": float(group["turnover"].sum()),
                "selection_uses_outer_test": False,
            }
        )
    return pd.DataFrame(rows)


def _policy_ablation(
    frame: pd.DataFrame,
    predictions: pd.DataFrame,
    selected: dict[str, Any],
    segment: str,
    spec: HierarchicalSoxxExcessSpec,
) -> pd.DataFrame:
    profile = _selected_profile(selected)
    ablations = (
        ("full_hierarchical_policy", profile),
        (
            "alpha_only_with_cash_floor",
            PolicyProfile(
                name="alpha_only_with_cash_floor",
                alpha_threshold=profile.alpha_threshold,
                maximum_soxl_share=profile.maximum_soxl_share,
                hazard_threshold=2.0,
                cash_probability_threshold=2.0,
                hazard_soxl_cap_fraction=1.0,
                defense_cash=spec.minimum_cash,
            ),
        ),
        (
            "hazard_rotation_without_extra_cash",
            PolicyProfile(
                name="hazard_rotation_without_extra_cash",
                alpha_threshold=profile.alpha_threshold,
                maximum_soxl_share=profile.maximum_soxl_share,
                hazard_threshold=profile.hazard_threshold,
                cash_probability_threshold=2.0,
                hazard_soxl_cap_fraction=profile.hazard_soxl_cap_fraction,
                defense_cash=spec.minimum_cash,
            ),
        ),
    )
    rows: list[dict[str, Any]] = []
    for name, candidate in ablations:
        metrics, _, _ = _evaluate(
            frame, predictions, candidate, 10, spec
        )
        rows.append(
            {
                "segment": segment,
                "ablation": name,
                "selection_uses_outer_test": False,
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def _run_card(spec: HierarchicalSoxxExcessSpec) -> dict[str, Any]:
    card = spec.card_dir
    card.mkdir(parents=True, exist_ok=True)
    frame, data_audit, bundles, feature_manifest = _load_frame(spec)
    dataset, dataset_audit = _build_dataset(frame, bundles, spec)
    dev_predictions, models = _walk_forward_predictions(
        dataset, frame, bundles, spec
    )
    comparison, annual = _development_candidates(
        frame, dev_predictions, spec
    )
    head_audit = _head_oos_audit(dev_predictions)
    selected, model_selection = _select_candidate(
        comparison, annual, head_audit
    )
    outer_predictions, outer_models = _outer_predictions(
        dataset, frame, bundles, selected, spec
    )
    models.update(outer_models)
    outer_comparison, outer_annual, outer_returns = _outer_evaluation(
        frame, outer_predictions, selected, spec
    )
    selected_dev_predictions = dev_predictions.loc[
        dev_predictions["feature_bundle"].eq(selected["feature_bundle"])
        & dev_predictions["model_family"].eq(selected["model_family"])
    ].sort_index()
    selected_profile = _selected_profile(selected)
    development_returns: list[pd.DataFrame] = []
    for cost_bps in spec.costs_bps:
        _, result, _ = _evaluate(
            frame, selected_dev_predictions, selected_profile, cost_bps, spec
        )
        development_returns.append(
            result.assign(segment="development_walk_forward", cost_bps=cost_bps)
        )
    strategy_returns = pd.concat(
        [*development_returns, outer_returns], ignore_index=True
    )
    predictions = pd.concat(
        [dev_predictions, outer_predictions], axis=0
    ).sort_index()
    outer_head_transfer = _outer_head_transfer_audit(
        dataset, frame, bundles, spec
    )
    regime_metrics = _regime_metrics(frame, strategy_returns)
    action_attribution = _action_attribution(strategy_returns)
    policy_ablation = pd.concat(
        [
            _policy_ablation(
                frame,
                selected_dev_predictions,
                selected,
                "development_walk_forward",
                spec,
            ),
            _policy_ablation(
                frame,
                outer_predictions,
                selected,
                "outer_diagnostic",
                spec,
            ),
        ],
        ignore_index=True,
    )
    prior_comparison = _explicit_prior_comparison(
        frame, outer_predictions, selected, spec
    )
    outer10 = outer_comparison.loc[
        outer_comparison["cost_bps"].eq(10)
    ].iloc[0]
    outer_gate = bool(
        outer10["terminal_wealth_ratio_vs_soxx"] >= 1.20
        and outer10["max_drawdown_pct"] > outer10["soxx_max_drawdown_pct"]
        and outer10["longest_underwater_days"]
        < outer10["soxx_longest_underwater_days"]
        and outer10["max_drawdown_recovery_days"]
        <= outer10["soxx_max_drawdown_recovery_days"]
        and outer10["downside_avg_cash_weight_pct"] > 10.0
    )
    promotion_decision = (
        "promoted_hierarchical_soxx_excess_development_gate_outer_diagnostic_pass"
        if bool(selected["all_gate"])
        and outer_gate
        else "needs_iteration_hierarchical_soxx_excess_transfer"
    )

    from qlib.workflow import R

    recorder_uri = card / "qlib_recorder"
    recorder_uri.mkdir(parents=True, exist_ok=True)
    R.set_uri(recorder_uri.resolve().as_posix())
    with tempfile.TemporaryDirectory(prefix=f"{spec.experiment}_") as temporary:
        temp = Path(temporary)
        model_dir = temp / "models"
        model_dir.mkdir()
        selected_prefix = (
            f"{selected['feature_bundle']}__{selected['model_family']}__"
        )
        retained = {
            name: model
            for name, model in models.items()
            if name.startswith(selected_prefix) or name.startswith("outer_refit__")
        }
        for name, model in retained.items():
            with (model_dir / f"{name}.pkl").open("wb") as handle:
                pickle.dump(model, handle)
        predictions.reset_index(names="signal_date").to_csv(
            temp / "predictions.csv", index=False
        )
        strategy_returns.to_csv(temp / "strategy_returns.csv", index=False)
        strategy_returns.loc[
            :,
            [
                "signal_date",
                "trade_date",
                "exit_date",
                "soxl_weight",
                "soxx_weight",
                "cash_weight",
                "turnover",
                "segment",
                "cost_bps",
            ],
        ].to_csv(temp / "positions.csv", index=False)
        with R.start(experiment_name=spec.experiment):
            recorder = R.get_recorder()
            R.log_params(
                benchmark="SOXX",
                assets="SOXL,SOXX,cash",
                development_years="2016:2022",
                outer="2023:2026-07-02",
                targets=",".join(TARGETS),
                selection_uses_outer_test="false",
                base_constrained_ref_role="explicit_benchmark_only",
            )
            R.log_metrics(
                selected_ratio_10bps=float(selected["ratio_10bps"]),
                selected_ratio_25bps=float(selected["ratio_25bps"]),
                selected_max_drawdown_pct=float(selected["max_drawdown_pct"]),
                selected_soxx_max_drawdown_pct=float(
                    selected["soxx_max_drawdown_pct"]
                ),
                selected_longest_underwater_days=float(
                    selected["longest_underwater_days"]
                ),
            )
            for artifact in [model_dir, *temp.glob("*.csv")]:
                recorder.log_artifact(
                    artifact.as_posix(), artifact_path=spec.experiment
                )
            recorder_id = recorder.id
            recorder_local_dir = Path(recorder.get_local_dir()).resolve()
            summary = _summary(
                spec=spec,
                data_audit=data_audit,
                dataset_audit=dataset_audit,
                bundles=bundles,
                selected=selected,
                comparison=comparison,
                outer_comparison=outer_comparison,
                head_audit=head_audit,
                outer_head_transfer=outer_head_transfer,
                prior_comparison=prior_comparison,
                promotion_decision=promotion_decision,
                recorder_id=recorder_id,
                recorder_uri=recorder_uri,
                recorder_local_dir=recorder_local_dir,
                retained_model_count=len(retained),
            )
            _write_card(
                card=card,
                summary=summary,
                feature_manifest=feature_manifest,
                model_selection=model_selection,
                comparison=comparison,
                annual=annual,
                outer_comparison=outer_comparison,
                outer_annual=outer_annual,
                head_audit=head_audit,
                outer_head_transfer=outer_head_transfer,
                regime_metrics=regime_metrics,
                action_attribution=action_attribution,
                policy_ablation=policy_ablation,
                prior_comparison=prior_comparison,
            )
            for filename in summary["sidecars"].values():
                path = card / filename
                if path.exists():
                    recorder.log_artifact(
                        path.as_posix(), artifact_path=spec.experiment
                    )
            _remove_qlib_code_snapshot_artifacts(recorder_local_dir)
    return _json_ready(summary)


def _summary(
    *,
    spec: HierarchicalSoxxExcessSpec,
    data_audit: dict[str, Any],
    dataset_audit: dict[str, Any],
    bundles: dict[str, dict[str, list[str]]],
    selected: dict[str, Any],
    comparison: pd.DataFrame,
    outer_comparison: pd.DataFrame,
    head_audit: pd.DataFrame,
    outer_head_transfer: pd.DataFrame,
    prior_comparison: pd.DataFrame,
    promotion_decision: str,
    recorder_id: str,
    recorder_uri: Path,
    recorder_local_dir: Path,
    retained_model_count: int,
) -> dict[str, Any]:
    sidecars = {
        "selected_model": "selected_model.json",
        "model_selection": "model_selection.csv",
        "feature_manifest": "feature_manifest.csv",
        "head_oos_audit": "head_oos_audit.csv",
        "outer_head_transfer_diagnostic": "outer_head_transfer_diagnostic.csv",
        "backtest_benchmark_comparison": "backtest_benchmark_comparison.csv",
        "backtest_walk_forward_metrics": "backtest_walk_forward_metrics.csv",
        "risk_analysis": "risk_analysis.csv",
        "regime_metrics": "regime_metrics.csv",
        "action_attribution": "action_attribution.csv",
        "policy_ablation": "policy_ablation.csv",
        "explicit_prior_comparison": "explicit_prior_comparison.csv",
        "analysis_summary": "analysis_summary.json",
        "failure_modes": "failure_modes.md",
    }
    return _json_ready(
        {
            "experiment": spec.experiment,
            "card_type": "hierarchical_soxx_excess_walk_forward_research_card",
            "primary_notebook": f"{spec.experiment}.ipynb",
            "promotion_decision": promotion_decision,
            "selection_uses_outer_test": False,
            "outer_test_use": "diagnostic_only_previously_observed",
            "data_audit": data_audit,
            "qlib_dataset": dataset_audit,
            "qlib_docs": [
                "https://qlib.readthedocs.io/en/latest/component/data.html",
                "https://qlib.readthedocs.io/en/latest/component/workflow.html",
                "https://qlib.readthedocs.io/en/latest/component/recorder.html",
                "https://qlib.readthedocs.io/en/latest/component/strategy.html",
            ],
            "architecture": {
                "alpha_head": "predict next-5d SOXL minus SOXX log return; controls SOXL share inside risky sleeve",
                "hazard_head": "predict next-5d drawdown, jump, or daily loss breach; only authorizes defense and caps SOXL",
                "cash_action_head": "predict whether defensive cash beats normal cash by more than 10bp-derived hurdle; cash increases only under hazard AND cash gate",
                "base_constrained_ref": "explicit benchmark only; excluded from all model feature bundles",
            },
            "targets": {
                "soxl_relative_alpha_5d": "sum log1p(next five SOXL open-to-open returns) minus SOXX",
                "crash_hazard_5d": "SOXX path <= -6pct OR SOXX day <= -4pct OR SOXL day <= -12pct",
                "cash_action_5d": "45pct SOXX plus 55pct cash beats 90pct SOXX plus 10pct cash by > 0.10pct over five days",
            },
            "feature_bundles": {
                bundle: {
                    head: len(columns) for head, columns in by_head.items()
                }
                for bundle, by_head in bundles.items()
            },
            "policy_profiles": [asdict(profile) for profile in POLICY_PROFILES],
            "development_gates": {
                "terminal_wealth_ratio_vs_soxx_10bps": ">=1.20",
                "terminal_wealth_ratio_vs_soxx_25bps": "stress_test_above_1.00",
                "max_drawdown": "shallower_than_soxx",
                "recovery": "longest_underwater_shorter_and_maxdd_recovery_no_longer_than_soxx",
                "annual": "at_least_4_of_7_positive_relative_years_and_median_ratio_above_1",
                "cash": "minimum_10pct_downside_average_above_10pct_and_at_least_one_valid_entry",
            },
            "selected_model": selected,
            "development_selected": comparison.loc[
                comparison["candidate"].eq(selected["candidate"])
            ].to_dict(orient="records"),
            "outer_selected_diagnostic": outer_comparison.to_dict(
                orient="records"
            ),
            "head_oos_audit": head_audit.to_dict(orient="records"),
            "outer_head_transfer_diagnostic": outer_head_transfer.to_dict(
                orient="records"
            ),
            "explicit_prior_comparison": prior_comparison.to_dict(
                orient="records"
            ),
            "recorder_id": recorder_id,
            "recorder_uri": recorder_uri.resolve().as_posix(),
            "recorder_local_dir": recorder_local_dir.as_posix(),
            "sidecars": sidecars,
            "artifact_policy": {
                "policy": "slim_card_recorder_first",
                "workflow_config": "workflow.yaml",
                "recorder_only_artifacts": list(RECORDER_ONLY_ARTIFACTS),
                "retained_model_scope": "selected_development_folds_and_outer_refit_only",
                "retained_model_count": retained_model_count,
            },
        }
    )


def _write_card(
    *,
    card: Path,
    summary: dict[str, Any],
    feature_manifest: pd.DataFrame,
    model_selection: pd.DataFrame,
    comparison: pd.DataFrame,
    annual: pd.DataFrame,
    outer_comparison: pd.DataFrame,
    outer_annual: pd.DataFrame,
    head_audit: pd.DataFrame,
    outer_head_transfer: pd.DataFrame,
    regime_metrics: pd.DataFrame,
    action_attribution: pd.DataFrame,
    policy_ablation: pd.DataFrame,
    prior_comparison: pd.DataFrame,
) -> None:
    selected = summary["selected_model"]
    selected_name = selected["candidate"]
    feature_manifest.to_csv(card / "feature_manifest.csv", index=False)
    model_selection.to_csv(card / "model_selection.csv", index=False)
    head_audit.to_csv(card / "head_oos_audit.csv", index=False)
    outer_head_transfer.to_csv(
        card / "outer_head_transfer_diagnostic.csv", index=False
    )
    regime_metrics.to_csv(card / "regime_metrics.csv", index=False)
    action_attribution.to_csv(card / "action_attribution.csv", index=False)
    policy_ablation.to_csv(card / "policy_ablation.csv", index=False)
    prior_comparison.to_csv(card / "explicit_prior_comparison.csv", index=False)
    development = comparison.loc[
        comparison["candidate"].eq(selected_name)
    ].assign(segment="development_walk_forward")
    backtest = pd.concat([development, outer_comparison], ignore_index=True)
    backtest.to_csv(card / "backtest_benchmark_comparison.csv", index=False)
    development_annual = annual.loc[
        annual["candidate"].eq(selected_name)
    ].assign(segment="development_walk_forward")
    pd.concat([development_annual, outer_annual], ignore_index=True).to_csv(
        card / "backtest_walk_forward_metrics.csv", index=False
    )
    backtest.loc[
        :,
        [
            "segment",
            "cost_bps",
            "terminal_wealth_ratio_vs_soxx",
            "return_pct",
            "soxx_return_pct",
            "max_drawdown_pct",
            "soxx_max_drawdown_pct",
            "longest_underwater_days",
            "soxx_longest_underwater_days",
            "max_drawdown_recovery_days",
            "soxx_max_drawdown_recovery_days",
            "worst_day_pct",
            "cvar5_pct",
            "worst_rolling_5d_pct",
            "avg_cash_weight_pct",
            "downside_avg_cash_weight_pct",
        ],
    ].to_csv(card / "risk_analysis.csv", index=False)
    (card / "selected_model.json").write_text(
        json.dumps(selected, indent=2) + "\n", encoding="utf-8"
    )
    analysis = {
        "promotion_decision": summary["promotion_decision"],
        "selected_model": selected,
        "development_selected": summary["development_selected"],
        "outer_selected_diagnostic": summary["outer_selected_diagnostic"],
        "head_oos_audit": summary["head_oos_audit"],
        "outer_head_transfer_diagnostic": summary[
            "outer_head_transfer_diagnostic"
        ],
        "primary_finding": (
            "The same legacy feature families retain value when each head maps to one action; "
            "the remaining question is whether the development-selected policy transfers to 2023+."
        ),
    }
    (card / "analysis_summary.json").write_text(
        json.dumps(analysis, indent=2) + "\n", encoding="utf-8"
    )
    (card / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    outer10 = outer_comparison.loc[outer_comparison["cost_bps"].eq(10)].iloc[0]
    failure = (
        "# Failure modes\n\n"
        f"- promotion_decision: {summary['promotion_decision']}\n"
        f"- development gate passed: {bool(selected['all_gate'])}\n"
        f"- outer 10bp wealth ratio vs SOXX: {outer10['terminal_wealth_ratio_vs_soxx']:.4f}\n"
        f"- outer strategy/SOXX MaxDD: {outer10['max_drawdown_pct']:.2f}% / {outer10['soxx_max_drawdown_pct']:.2f}%\n"
        "- outer years are diagnostic only and never participate in model, feature-bundle, or policy-profile selection.\n"
        "- cash_action head is trained and audited only inside realized crash-hazard rows because it distinguishes sustained decline from rebound after defense is authorized.\n"
        "- base_constrained_ref is reconstructed from qlib daily inputs for an explicit benchmark only; it is not a hidden feature or allocation prior.\n"
    )
    (card / "failure_modes.md").write_text(failure, encoding="utf-8")
    conclusion = (
        "# 分层 SOXL/SOXX/cash walk-forward 结论\n\n"
        f"- 跑了什么：2016–2022 年度 expanding walk-forward，三类特征包、两类模型、三个独立预测头；2023–{summary['data_audit']['common_feature_end']} 仅作 outer 诊断。\n"
        f"- 开发期：10bp 终值比 SOXX={selected['ratio_10bps']:.4f}，25bp={selected['ratio_25bps']:.4f}；MaxDD={selected['max_drawdown_pct']:.2f}% vs SOXX {selected['soxx_max_drawdown_pct']:.2f}%。\n"
        f"- Outer：10bp 终值比 SOXX={outer10['terminal_wealth_ratio_vs_soxx']:.4f}；MaxDD={outer10['max_drawdown_pct']:.2f}% vs SOXX {outer10['soxx_max_drawdown_pct']:.2f}%。\n"
        f"- Regime/窗口：开发期最长水下 {selected['longest_underwater_days']} 天 vs SOXX {selected['soxx_longest_underwater_days']} 天；downside 平均现金 {selected['downside_avg_cash_weight_pct']:.2f}%。\n"
        f"- 晋升决定：{summary['promotion_decision']}。\n"
        "- 风险：策略超额依赖收益头对少数上行窗口的识别；现金动作样本稀疏，outer 若未转移则不能因历史开发期通过而晋升。\n"
        "- 下一步：若未通过 outer，按 head_oos_audit、outer_head_transfer_diagnostic、action_attribution 和 policy_ablation 区分收益头失效、hazard 误杀或 cash gate 召回不足，不再扩大无约束参数网格。\n"
    )
    (card / "conclusion.md").write_text(conclusion, encoding="utf-8")
    workflow = (
        f"experiment: {EXPERIMENT}\n"
        "system_of_record: qlib\n"
        "benchmark: SOXX\n"
        "assets: [SOXL, SOXX, cash]\n"
        "development: [2016, 2022]\n"
        "outer_test: [2023, 2026-07-02]\n"
        "selection_uses_outer_test: false\n"
        "heads:\n"
        "  alpha: soxl_relative_alpha_5d\n"
        "  hazard: crash_hazard_5d\n"
        "  cash_action: cash_action_5d\n"
        "cash_rule: hazard_high AND predicted_cash_benefit_exceeds_cost\n"
        "base_constrained_ref_role: explicit_benchmark_only\n"
        "artifact_policy: slim_card_recorder_first\n"
    )
    (card / "workflow.yaml").write_text(workflow, encoding="utf-8")
    _write_notebook(card, summary)


def _write_notebook(card: Path, summary: dict[str, Any]) -> None:
    notebook = nbformat.v4.new_notebook()
    notebook["metadata"]["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook["cells"] = [
        nbformat.v4.new_markdown_cell(
            "# Hierarchical SOXX excess walk-forward\n\n"
            "收益头只决定 SOXL/SOXX，风险头只授权防守，现金头只在 hazard 与净现金收益同时满足时加 cash。"
        ),
        nbformat.v4.new_code_cell(
            "from pathlib import Path\n"
            "import sys\n"
            "repo_root = Path.cwd().parents[1]\n"
            "if str(repo_root) not in sys.path:\n"
            "    sys.path.insert(0, str(repo_root))\n"
            "from src.workflow.hierarchical_soxx_excess_walk_forward import "
            "run_hierarchical_soxx_excess_walk_forward_card\n"
            "summary = run_hierarchical_soxx_excess_walk_forward_card(root=repo_root)\n"
            "summary['promotion_decision']"
        ),
        nbformat.v4.new_code_cell(
            "import pandas as pd\n"
            "card = Path.cwd()\n"
            "pd.read_csv(card / 'backtest_benchmark_comparison.csv')"
        ),
        nbformat.v4.new_code_cell(
            "pd.read_csv(card / 'head_oos_audit.csv')"
        ),
        nbformat.v4.new_code_cell(
            "pd.read_csv(card / 'regime_metrics.csv')"
        ),
        nbformat.v4.new_markdown_cell(
            (card / "conclusion.md").read_text(encoding="utf-8")
        ),
    ]
    path = card / f"{summary['experiment']}.ipynb"
    nbformat.write(notebook, path)


if __name__ == "__main__":
    print(
        json.dumps(
            run_hierarchical_soxx_excess_walk_forward_card(),
            indent=2,
        )
    )
