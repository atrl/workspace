from __future__ import annotations

import json
import pickle
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.factor.high_chop_ml import DEFAULT_HIGH_CHOP_FEATURES, build_high_chop_ml_frame
from src.workflow.cash_floor_benchmark_relative_wf import _result_metrics
from src.workflow.soxl_objective_feasibility_audit import _json_ready, _simulate
from src.workflow.soxs_defensive_hurdle import (
    ALL_FEATURES,
    SoxsDefensiveHurdleSpec,
    _load_feature_frame,
)
from src.workflow.trainable_regime_policy_nested_wf import (
    _remove_qlib_code_snapshot_artifacts,
)


EXPERIMENT = "cash_floor_relative_target_feature_reintegration_v1"
DEVELOPMENT_FOLDS = tuple(range(2016, 2023))
TARGETS = ("cash_protection_5d", "future_vol_5d")
MODEL_FAMILIES = ("ridge", "extra_trees")
VOLATILITY_WEIGHTS = (0.25, 0.50, 0.75)
ROUTERS = ("none", "trend_vol_cap")
RECORDER_ONLY_ARTIFACTS = (
    f"artifacts/{EXPERIMENT}/models/",
    f"artifacts/{EXPERIMENT}/predictions.csv",
    f"artifacts/{EXPERIMENT}/positions.csv",
    f"artifacts/{EXPERIMENT}/strategy_returns.csv",
)

PROTECTION_FEATURES = (
    "soxx_trend60",
    "soxx_drawdown60",
    "dist_high_252",
    "soxx_ret60",
    "soxl_ret60",
    "range_pos_252",
    "relative_momentum20",
    "soxl_ret20",
    "soxx_ret20",
    "drawdown_60_z60",
    "drawdown_60_pctile60",
    "ret_60_z60",
    "ret_60_pctile60",
    "tech_breadth_20",
    "tech_breadth_60",
    "cross_breadth_pos_20",
    "semis_vs_cross_20",
    "semis_vs_cross_60",
)

RISK_FEATURES = (
    "tracking_error_5_z20",
    "tracking_error_5_pctile20",
    "tracking_error_5_pctile60",
    "vix_z_60",
    "vix",
    "rv20",
    "rv60",
    "trend_efficiency_20",
    "panic_reversal_5",
    "chop_breakdown_risk",
    "inverse_tracking_gap20",
    "cross_breadth_pos_60",
    "tech_breadth_60",
    "semis_vs_cross_20",
)


@dataclass(frozen=True)
class CashFloorRelativeFeatureSpec:
    root: Path
    experiment: str = EXPERIMENT
    train_start: str = "2011-01-03"
    development_folds: tuple[int, ...] = DEVELOPMENT_FOLDS
    development_end: str = "2022-12-30"
    outer: tuple[str, str] = ("2023-01-03", "2026-12-31")
    purge_rows: int = 20
    minimum_cash: float = 0.10
    maximum_cash: float = 0.60
    risk_on_maximum_cash: float = 0.35
    risky_sleeve_soxl_share: float = 0.30
    calibration_lower_quantile: float = 0.50
    calibration_upper_quantile: float = 0.90
    smoothing_span: int = 3
    rebalance_threshold: float = 0.05
    costs_bps: tuple[int, ...] = (10, 25)

    @property
    def card_dir(self) -> Path:
        return self.root / "research" / self.experiment

    @property
    def execution_provider_uri(self) -> Path:
        return self.root / "data" / "qlib" / "soxl_soxx_soxs"

    @property
    def daily_provider_uri(self) -> Path:
        return self.root / "data" / "qlib" / "soxl"

    @property
    def normalized_provider_uri(self) -> Path:
        return self.root / "data" / "qlib" / "dynamic_soxx_soxl_cash_allocation_wf_v1"


def run_cash_floor_relative_feature_reintegration_card(
    *, root: Path | str | None = None
) -> dict[str, Any]:
    repo_root = Path(root).resolve() if root is not None else Path(__file__).resolve().parents[2]
    return _run_card(CashFloorRelativeFeatureSpec(root=repo_root))


def _load_reintegrated_frame(
    spec: CashFloorRelativeFeatureSpec,
    *,
    include_incomplete_latest: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, list[str]], pd.DataFrame]:
    import qlib
    from qlib.data import D

    market, execution_audit = _load_feature_frame(
        SoxsDefensiveHurdleSpec(root=spec.root),
        include_incomplete_latest=include_incomplete_latest,
    )
    qlib.init(
        provider_uri=spec.daily_provider_uri.resolve().as_posix(),
        region="us",
        kernels=1,
        skip_if_reg=False,
    )
    daily_raw = D.features(
        ["SOXL", "SOXX"],
        ["$open", "$high", "$low", "$close", "$volume", "$vix"],
        start_time="2010-03-11",
        end_time=market.index.max(),
        freq="day",
    )
    high_chop = build_high_chop_ml_frame(
        daily_raw,
        start=spec.train_start,
        end=market.index.max().date().isoformat(),
        include_incomplete_latest=True,
    ).set_index("datetime")
    high_chop_cols = [
        column
        for column in DEFAULT_HIGH_CHOP_FEATURES
        if column in high_chop.columns and not column.startswith("base_")
    ]

    normalized_manifest = json.loads(
        (spec.normalized_provider_uri / "qlib_store_manifest.json").read_text(encoding="utf-8")
    )
    normalized_cols = [
        str(field).removeprefix("$")
        for field in normalized_manifest["fields"]
        if str(field).startswith("$")
        and ("_z" in str(field) or "_pctile" in str(field))
        and not str(field).startswith("$soxx_const")
    ]
    qlib.init(
        provider_uri=spec.normalized_provider_uri.resolve().as_posix(),
        region="us",
        kernels=1,
        skip_if_reg=False,
    )
    normalized_fields = [f"${column}" for column in normalized_cols]
    panel = D.features(
        ["SOXL", "SPXL", "TECL", "TQQQ", "UPRO"],
        [*normalized_fields, "$underlying_close"],
        start_time=spec.train_start,
        end_time=normalized_manifest["calendar_end"],
        freq="day",
    ).reset_index()
    panel = panel.rename(columns={f"${column}": column for column in normalized_cols})
    panel["instrument"] = panel["instrument"].astype(str).str.upper()
    panel["datetime"] = pd.to_datetime(panel["datetime"]).dt.normalize()
    soxl_normalized = panel.loc[panel["instrument"].eq("SOXL")].set_index("datetime")
    normalized_core_cols = [
        column
        for column in normalized_cols
        if not column.startswith(("volume_z_20_", "vix_z_60_"))
    ]

    underlying = panel.pivot(
        index="datetime", columns="instrument", values="$underlying_close"
    ).sort_index()
    cross = pd.DataFrame(index=underlying.index)
    for window in (5, 20, 60):
        returns = underlying.pct_change(window)
        cross[f"cross_breadth_pos_{window}"] = returns.gt(0.0).mean(axis=1)
        cross[f"cross_dispersion_{window}"] = returns.std(axis=1)
        cross[f"semis_vs_cross_{window}"] = returns["SOXL"] - returns.drop(
            columns="SOXL"
        ).mean(axis=1)
        cross[f"tech_breadth_{window}"] = returns.loc[
            :, ["SOXL", "TECL", "TQQQ"]
        ].gt(0.0).mean(axis=1)
    cross_vol = underlying.pct_change().rolling(20).std() * np.sqrt(252.0)
    cross["cross_vol_breadth_high"] = cross_vol.gt(
        cross_vol.rolling(252, min_periods=126).median()
    ).mean(axis=1)
    cross["semis_vol_vs_cross"] = cross_vol["SOXL"] - cross_vol.drop(
        columns="SOXL"
    ).mean(axis=1)

    execution_columns = [
        "soxl_forward_open_return",
        "soxx_forward_open_return",
        "soxs_forward_open_return",
        "trade_date",
        "exit_date",
    ]
    frame = market.loc[:, [*ALL_FEATURES, *execution_columns]].join(
        high_chop.loc[:, high_chop_cols], how="left", rsuffix="_high_chop"
    )
    frame = frame.join(
        soxl_normalized.loc[:, normalized_core_cols],
        how="left",
        rsuffix="_normalized",
    )
    frame = frame.join(cross, how="left")
    frame = _attach_relative_targets(frame, market)

    current = list(ALL_FEATURES)
    compact_protection = [column for column in PROTECTION_FEATURES if column in frame]
    compact_risk = [column for column in RISK_FEATURES if column in frame]
    compact_shared = list(dict.fromkeys([*compact_protection, *compact_risk]))
    full = list(
        dict.fromkeys(
            [
                *current,
                *high_chop_cols,
                *normalized_core_cols,
                *cross.columns.tolist(),
            ]
        )
    )
    feature_sets = {
        "current10": current,
        "compact_reintegrated": compact_shared,
        "full_reintegrated": [column for column in full if column in frame],
    }
    family_rows: list[dict[str, Any]] = []
    for column in current:
        family_rows.append(_feature_manifest_row(column, "current_soxl_soxx_soxs", True))
    for column in high_chop_cols:
        family_rows.append(_feature_manifest_row(column, "risk_hardened_high_chop", True))
    for column in normalized_core_cols:
        family_rows.append(_feature_manifest_row(column, "normalized_regime", True))
    for column in cross.columns:
        family_rows.append(_feature_manifest_row(column, "cross_asset_diffusion", True))
    feature_manifest = pd.DataFrame(family_rows).drop_duplicates("feature")

    common_end = min(
        market.index.max(),
        pd.Timestamp(normalized_manifest["calendar_end"]),
        high_chop.index.max(),
    )
    frame = frame.loc[:common_end].copy()
    data_audit = {
        "execution_provider_uri": spec.execution_provider_uri.resolve().as_posix(),
        "daily_provider_uri": spec.daily_provider_uri.resolve().as_posix(),
        "normalized_provider_uri": spec.normalized_provider_uri.resolve().as_posix(),
        "execution_provider": execution_audit,
        "calendar_start": frame.index.min().date().isoformat(),
        "calendar_end": frame.index.max().date().isoformat(),
        "common_feature_end": common_end.date().isoformat(),
        "source_api": "qlib.data.D.features",
        "include_incomplete_latest": bool(include_incomplete_latest),
        "excluded_sources": {
            "soxx_constituents": "excluded_from_model_non_PIT_single_snapshot_survivorship_bias",
            "macro_credit": "excluded_from_2016_2022_walk_forward_coverage_starts_2019",
            "hardware_cycle": "excluded_from_2016_2022_walk_forward_coverage_starts_2019",
            "intraday": "excluded_from_2016_2022_walk_forward_coverage_starts_2019",
        },
    }
    head_features = {
        "compact_reintegrated__cash_protection_5d": compact_protection,
        "compact_reintegrated__future_vol_5d": compact_risk,
    }
    return frame, data_audit, feature_sets | head_features, feature_manifest


def _feature_manifest_row(feature: str, family: str, pit: bool) -> dict[str, Any]:
    return {
        "feature": feature,
        "family": family,
        "point_in_time": pit,
        "role": "nested_development_candidate",
        "outer_selection": False,
    }


def _attach_relative_targets(frame: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    """Legacy reference-portfolio labels, not pure SOXL alpha or cash probability.

    The alpha reference is daily-reset 27% SOXL / 63% SOXX / 10% cash.
    Keep this frozen definition distinct from later 15%-minimum-cash policies.
    """
    result = frame.copy()
    minimum_cash_return = (
        0.27 * market["soxl_forward_open_return"]
        + 0.63 * market["soxx_forward_open_return"]
    )
    maximum_cash_return = (
        0.12 * market["soxl_forward_open_return"]
        + 0.28 * market["soxx_forward_open_return"]
    )
    soxx_return = market["soxx_forward_open_return"]
    minimum_log = sum(
        np.log1p(minimum_cash_return.shift(-offset).clip(lower=-0.99))
        for offset in range(5)
    )
    maximum_log = sum(
        np.log1p(maximum_cash_return.shift(-offset).clip(lower=-0.99))
        for offset in range(5)
    )
    benchmark_log = sum(
        np.log1p(soxx_return.shift(-offset).clip(lower=-0.99))
        for offset in range(5)
    )
    forward_soxx = pd.concat(
        [soxx_return.shift(-offset) for offset in range(5)], axis=1
    )
    result["relative_alpha_5d"] = minimum_log - benchmark_log
    result["cash_protection_5d"] = maximum_log - minimum_log
    result["future_vol_5d"] = forward_soxx.std(axis=1, ddof=1)
    return result


def _build_dataset(
    frame: pd.DataFrame,
    feature_columns: list[str],
    spec: CashFloorRelativeFeatureSpec,
):
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP
    from qlib.data.dataset.loader import StaticDataLoader

    static = pd.concat(
        {
            "feature": frame.loc[:, feature_columns],
            "label": frame.loc[:, ["relative_alpha_5d", *TARGETS]],
        },
        axis=1,
    )
    static.index = pd.MultiIndex.from_arrays(
        [frame.index, np.full(len(frame), "SOXL_SOXX_CASH")],
        names=["datetime", "instrument"],
    )
    outer_end = min(spec.outer[1], frame.index.max().date().isoformat())
    segments: dict[str, tuple[str, str]] = {
        "development": (f"{spec.development_folds[0]}-01-01", spec.development_end),
        "outer": (spec.outer[0], outer_end),
    }
    for year in spec.development_folds:
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


def _model(family: str):
    if family == "ridge":
        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", Ridge(alpha=20.0)),
            ]
        )
    if family == "extra_trees":
        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                (
                    "model",
                    ExtraTreesRegressor(
                        n_estimators=300,
                        max_depth=5,
                        min_samples_leaf=20,
                        max_features=0.70,
                        random_state=20260723,
                        n_jobs=1,
                    ),
                ),
            ]
        )
    raise ValueError(f"unknown model family: {family}")


def _head_columns(
    feature_set: str,
    target: str,
    feature_sets: dict[str, list[str]],
) -> list[str]:
    key = f"{feature_set}__{target}"
    return list(feature_sets.get(key, feature_sets[feature_set]))


def _fit_predict_head(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    features: list[str],
    target: str,
    family: str,
    spec: CashFloorRelativeFeatureSpec,
):
    usable = train.dropna(subset=[target])
    model = _model(family)
    model.fit(usable.loc[:, features], usable[target])
    train_prediction = model.predict(usable.loc[:, features])
    lower, upper = np.quantile(
        train_prediction,
        [spec.calibration_lower_quantile, spec.calibration_upper_quantile],
    )
    if upper <= lower:
        raise ValueError(f"blocked: invalid training-only calibration for {family}/{target}")
    prediction = model.predict(test.loc[:, features])
    score = np.clip((prediction - lower) / (upper - lower), 0.0, 1.0)
    return model, prediction, score, {
        "lower": float(lower),
        "upper": float(upper),
        "train_rows": int(len(usable)),
    }


def _walk_forward_predictions(
    dataset,
    frame: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    spec: CashFloorRelativeFeatureSpec,
):
    prediction_rows: list[pd.DataFrame] = []
    calibrations: list[dict[str, Any]] = []
    models: dict[str, Any] = {}
    selectable_sets = ("current10", "compact_reintegrated", "full_reintegrated")
    for feature_set in selectable_sets:
        for family in MODEL_FAMILIES:
            for year in spec.development_folds:
                fold = _prepare(dataset, f"fold_{year}")
                fold = fold.loc[fold.index.year == year].dropna(subset=list(TARGETS))
                start_position = frame.index.searchsorted(pd.Timestamp(f"{year}-01-01"))
                train_end_position = start_position - spec.purge_rows - 1
                train_end = frame.index[train_end_position]
                train = frame.loc[spec.train_start:train_end].dropna(subset=list(TARGETS))
                output = pd.DataFrame(index=fold.index)
                output["segment"] = "development_walk_forward"
                output["fold_year"] = year
                output["feature_set"] = feature_set
                output["model_family"] = family
                output["train_end"] = train_end
                for target in TARGETS:
                    columns = _head_columns(feature_set, target, feature_sets)
                    model, prediction, score, calibration = _fit_predict_head(
                        train,
                        fold,
                        features=columns,
                        target=target,
                        family=family,
                        spec=spec,
                    )
                    output[f"{target}_actual"] = fold[target]
                    output[f"{target}_prediction"] = prediction
                    output[f"{target}_score"] = score
                    model_key = f"{feature_set}__{family}__fold_{year}__{target}"
                    models[model_key] = model
                    calibrations.append(
                        {
                            "segment": "development_walk_forward",
                            "fold_year": year,
                            "feature_set": feature_set,
                            "model_family": family,
                            "target": target,
                            "feature_count": len(columns),
                            "train_end": train_end.date().isoformat(),
                            **calibration,
                            "selection_uses_outer_test": False,
                        }
                    )
                prediction_rows.append(output)
    return pd.concat(prediction_rows).sort_index(), pd.DataFrame(calibrations), models


def _outer_predictions(
    dataset,
    frame: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    *,
    selected: dict[str, Any],
    spec: CashFloorRelativeFeatureSpec,
):
    outer = _prepare(dataset, "outer").dropna(subset=list(TARGETS))
    outer_start = frame.index.searchsorted(pd.Timestamp(spec.outer[0]))
    train_end = frame.index[outer_start - spec.purge_rows - 1]
    train = frame.loc[spec.train_start:train_end].dropna(subset=list(TARGETS))
    output = pd.DataFrame(index=outer.index)
    output["segment"] = "outer_diagnostic"
    output["fold_year"] = output.index.year
    output["feature_set"] = selected["feature_set"]
    output["model_family"] = selected["model_family"]
    output["train_end"] = train_end
    models: dict[str, Any] = {}
    calibrations: list[dict[str, Any]] = []
    for target in TARGETS:
        columns = _head_columns(selected["feature_set"], target, feature_sets)
        model, prediction, score, calibration = _fit_predict_head(
            train,
            outer,
            features=columns,
            target=target,
            family=selected["model_family"],
            spec=spec,
        )
        output[f"{target}_actual"] = outer[target]
        output[f"{target}_prediction"] = prediction
        output[f"{target}_score"] = score
        models[f"outer_refit__{target}"] = model
        calibrations.append(
            {
                "segment": "outer_diagnostic",
                "fold_year": "all",
                "feature_set": selected["feature_set"],
                "model_family": selected["model_family"],
                "target": target,
                "feature_count": len(columns),
                "train_end": train_end.date().isoformat(),
                **calibration,
                "selection_uses_outer_test": False,
            }
        )
    return output, pd.DataFrame(calibrations), models


def _smoothed_cash(
    prediction: pd.DataFrame,
    market: pd.DataFrame,
    *,
    volatility_weight: float,
    router: str,
    spec: CashFloorRelativeFeatureSpec,
) -> np.ndarray:
    protection = prediction["cash_protection_5d_score"].to_numpy(dtype=float)
    volatility = prediction["future_vol_5d_score"].to_numpy(dtype=float)
    score = (1.0 - volatility_weight) * protection + volatility_weight * volatility
    maximum = np.full(len(score), spec.maximum_cash)
    if router == "trend_vol_cap":
        annualized_vol = market["soxx_vol20"].to_numpy(dtype=float) * np.sqrt(252.0)
        risk_on = market["soxx_trend60"].to_numpy(dtype=float) > 0.0
        risk_on &= annualized_vol < 0.35
        maximum[risk_on] = spec.risk_on_maximum_cash
    elif router != "none":
        raise ValueError(f"unknown router: {router}")
    raw_cash = spec.minimum_cash + (maximum - spec.minimum_cash) * score
    smoothed = pd.Series(raw_cash).ewm(span=spec.smoothing_span, adjust=False).mean().to_numpy()
    cash = np.empty(len(smoothed), dtype=float)
    previous = spec.minimum_cash
    for index, value in enumerate(smoothed):
        if abs(value - previous) >= spec.rebalance_threshold:
            previous = float(value)
        cash[index] = previous
    return np.clip(cash, spec.minimum_cash, spec.maximum_cash)


def _weights(cash: np.ndarray, spec: CashFloorRelativeFeatureSpec) -> np.ndarray:
    risky = 1.0 - np.asarray(cash, dtype=float)
    return np.column_stack(
        [
            risky * spec.risky_sleeve_soxl_share,
            risky * (1.0 - spec.risky_sleeve_soxl_share),
            np.zeros(len(risky)),
            cash,
        ]
    )


def _candidate_metrics(
    market: pd.DataFrame,
    prediction: pd.DataFrame,
    *,
    feature_set: str,
    model_family: str,
    volatility_weight: float,
    router: str,
    cost_bps: int,
    spec: CashFloorRelativeFeatureSpec,
) -> tuple[dict[str, Any], pd.DataFrame, list[dict[str, Any]]]:
    cash = _smoothed_cash(
        prediction,
        market,
        volatility_weight=volatility_weight,
        router=router,
        spec=spec,
    )
    name = f"{feature_set}__{model_family}__volw_{volatility_weight:.2f}__{router}"
    result = _simulate(market, _weights(cash, spec), cost_bps=cost_bps, name=name)
    benchmark = _simulate(
        market,
        np.tile([0.0, 1.0, 0.0, 0.0], (len(market), 1)),
        cost_bps=cost_bps,
        name="soxx_buyhold",
    )
    metrics = {
        "candidate": name,
        "feature_set": feature_set,
        "model_family": model_family,
        "volatility_weight": volatility_weight,
        "router": router,
        "cost_bps": cost_bps,
        **_result_metrics(result, benchmark),
    }
    fold_rows: list[dict[str, Any]] = []
    years = pd.to_datetime(result["signal_date"]).dt.year
    for year in sorted(years.unique()):
        mask = years.eq(year)
        fold_metrics = _result_metrics(
            result.loc[mask].reset_index(drop=True),
            benchmark.loc[mask].reset_index(drop=True),
        )
        fold_rows.append(
            {
                "candidate": name,
                "feature_set": feature_set,
                "model_family": model_family,
                "volatility_weight": volatility_weight,
                "router": router,
                "cost_bps": cost_bps,
                "fold_year": int(year),
                **fold_metrics,
            }
        )
    return metrics, result, fold_rows


def _development_candidates(
    frame: pd.DataFrame,
    predictions: pd.DataFrame,
    spec: CashFloorRelativeFeatureSpec,
):
    rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    returns: list[pd.DataFrame] = []
    for feature_set in ("current10", "compact_reintegrated", "full_reintegrated"):
        for family in MODEL_FAMILIES:
            selected_predictions = predictions.loc[
                predictions["feature_set"].eq(feature_set)
                & predictions["model_family"].eq(family)
            ].sort_index()
            market = frame.loc[selected_predictions.index]
            for volatility_weight in VOLATILITY_WEIGHTS:
                for router in ROUTERS:
                    for cost_bps in spec.costs_bps:
                        metrics, result, annual = _candidate_metrics(
                            market,
                            selected_predictions,
                            feature_set=feature_set,
                            model_family=family,
                            volatility_weight=volatility_weight,
                            router=router,
                            cost_bps=cost_bps,
                            spec=spec,
                        )
                        rows.append(metrics)
                        fold_rows.extend(annual)
                        returns.append(
                            result.assign(
                                segment="development_walk_forward",
                                cost_bps=cost_bps,
                            )
                        )
    return pd.DataFrame(rows), pd.DataFrame(fold_rows), pd.concat(returns, ignore_index=True)


def _select_candidate(
    comparison: pd.DataFrame,
    folds: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame]:
    ten = comparison.loc[comparison["cost_bps"].eq(10)].copy()
    stress = comparison.loc[comparison["cost_bps"].eq(25), ["candidate", "terminal_wealth_ratio_vs_soxx"]]
    stress = stress.rename(columns={"terminal_wealth_ratio_vs_soxx": "ratio_25bps"})
    annual = folds.loc[folds["cost_bps"].eq(10)].groupby("candidate").agg(
        positive_years=("terminal_wealth_ratio_vs_soxx", lambda values: int((values > 1.0).sum())),
        median_year_ratio=("terminal_wealth_ratio_vs_soxx", "median"),
    )
    selection = ten.merge(stress, on="candidate").merge(annual, on="candidate")
    selection = selection.rename(columns={"terminal_wealth_ratio_vs_soxx": "ratio_10bps"})
    selection["relative_gate"] = (
        selection["ratio_10bps"].gt(1.0)
        & selection["ratio_25bps"].gt(1.0)
        & selection["positive_years"].ge(4)
        & selection["median_year_ratio"].gt(1.0)
        & selection["avg_cash_weight_pct"].ge(15.0)
    )
    selection["risk_gate"] = (
        selection["max_drawdown_pct"].ge(-20.0)
        & selection["worst_day_pct"].ge(-12.0)
        & selection["cvar5_pct"].ge(-8.0)
        & selection["worst_rolling_5d_pct"].ge(-12.0)
    )
    selection["all_gate"] = selection["relative_gate"] & selection["risk_gate"]
    selection["risk_violation_score"] = (
        (-20.0 - selection["max_drawdown_pct"]).clip(lower=0.0) / 20.0
        + (-12.0 - selection["worst_day_pct"]).clip(lower=0.0) / 12.0
        + (-8.0 - selection["cvar5_pct"]).clip(lower=0.0) / 8.0
        + (-12.0 - selection["worst_rolling_5d_pct"]).clip(lower=0.0) / 12.0
    )
    selection["selection_score"] = (
        np.log(selection["ratio_10bps"].clip(lower=1e-9))
        + 0.5 * np.log(selection["ratio_25bps"].clip(lower=1e-9))
        - selection["risk_violation_score"]
    )
    eligible = selection.loc[selection["all_gate"]]
    selection_policy = "all_gate_then_relative_wealth"
    if eligible.empty:
        eligible = selection.loc[selection["relative_gate"]]
        selection_policy = "diagnostic_fallback_relative_gate_minimum_risk_violation"
    if eligible.empty:
        eligible = selection
        selection_policy = "diagnostic_fallback_no_relative_gate"
    selected = eligible.sort_values(
        ["risk_violation_score", "selection_score", "ratio_25bps"],
        ascending=[True, False, False],
    ).iloc[0]
    selection["selected"] = selection["candidate"].eq(selected["candidate"])
    payload = {
        **selected.to_dict(),
        "selection_policy": selection_policy,
        "selection_uses_outer_test": False,
    }
    return _json_ready(payload), selection


def _feature_oos_audit(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (feature_set, family), group in predictions.groupby(["feature_set", "model_family"]):
        for target in TARGETS:
            actual = group[f"{target}_actual"]
            prediction = group[f"{target}_prediction"]
            pooled = spearmanr(prediction, actual, nan_policy="omit").statistic
            fold_ic = [
                spearmanr(year_group[f"{target}_prediction"], year_group[f"{target}_actual"], nan_policy="omit").statistic
                for _, year_group in group.groupby("fold_year")
            ]
            lower = actual.loc[prediction.le(prediction.quantile(0.25))].mean()
            upper = actual.loc[prediction.ge(prediction.quantile(0.75))].mean()
            rows.append(
                {
                    "feature_set": feature_set,
                    "model_family": family,
                    "target": target,
                    "pooled_spearman": pooled,
                    "median_fold_spearman": float(np.nanmedian(fold_ic)),
                    "positive_fold_count": int((np.asarray(fold_ic) > 0.0).sum()),
                    "fold_count": len(fold_ic),
                    "prediction_q4_minus_q1_actual": float(upper - lower),
                    "selection_uses_outer_test": False,
                }
            )
    return pd.DataFrame(rows)


def _outer_evaluation(
    frame: pd.DataFrame,
    prediction: pd.DataFrame,
    selected: dict[str, Any],
    spec: CashFloorRelativeFeatureSpec,
):
    rows: list[dict[str, Any]] = []
    folds: list[dict[str, Any]] = []
    returns: list[pd.DataFrame] = []
    market = frame.loc[prediction.index]
    for cost_bps in spec.costs_bps:
        metrics, result, annual = _candidate_metrics(
            market,
            prediction,
            feature_set=str(selected["feature_set"]),
            model_family=str(selected["model_family"]),
            volatility_weight=float(selected["volatility_weight"]),
            router=str(selected["router"]),
            cost_bps=cost_bps,
            spec=spec,
        )
        metrics["segment"] = "outer_diagnostic"
        rows.append(metrics)
        for item in annual:
            item["segment"] = "outer_diagnostic"
        folds.extend(annual)
        returns.append(result.assign(segment="outer_diagnostic", cost_bps=cost_bps))
    return pd.DataFrame(rows), pd.DataFrame(folds), pd.concat(returns, ignore_index=True)


def _run_card(spec: CashFloorRelativeFeatureSpec) -> dict[str, Any]:
    card = spec.card_dir
    card.mkdir(parents=True, exist_ok=True)
    frame, data_audit, feature_sets, feature_manifest = _load_reintegrated_frame(spec)
    all_features = list(
        dict.fromkeys(
            column
            for key, values in feature_sets.items()
            if "__" not in key
            for column in values
        )
    )
    dataset, dataset_audit = _build_dataset(frame, all_features, spec)
    predictions, calibrations, models = _walk_forward_predictions(
        dataset, frame, feature_sets, spec
    )
    comparison, folds, development_returns = _development_candidates(
        frame, predictions, spec
    )
    selected, model_selection = _select_candidate(comparison, folds)
    outer_predictions, outer_calibrations, outer_models = _outer_predictions(
        dataset,
        frame,
        feature_sets,
        selected=selected,
        spec=spec,
    )
    outer_comparison, outer_folds, outer_returns = _outer_evaluation(
        frame, outer_predictions, selected, spec
    )
    feature_audit = _feature_oos_audit(predictions)
    calibrations = pd.concat([calibrations, outer_calibrations], ignore_index=True)
    predictions = pd.concat([predictions, outer_predictions]).sort_index()
    models.update(outer_models)
    development_returns = development_returns.loc[
        development_returns["strategy"].eq(selected["candidate"])
    ].copy()
    strategy_returns = pd.concat(
        [development_returns, outer_returns], ignore_index=True
    )
    positions = strategy_returns.loc[
        :,
        [
            "signal_date",
            "trade_date",
            "exit_date",
            "strategy",
            "soxl_weight",
            "soxx_weight",
            "soxs_weight",
            "cash_weight",
            "turnover",
            "segment",
            "cost_bps",
        ],
    ].copy()

    from qlib.workflow import R

    recorder_uri = card / "qlib_recorder"
    recorder_uri.mkdir(parents=True, exist_ok=True)
    R.set_uri(recorder_uri.resolve().as_posix())
    with tempfile.TemporaryDirectory(prefix=f"{spec.experiment}_") as temporary:
        temp = Path(temporary)
        model_dir = temp / "models"
        model_dir.mkdir()
        selected_model_prefix = (
            f"{selected['feature_set']}__{selected['model_family']}__"
        )
        retained_models = {
            name: model
            for name, model in models.items()
            if name.startswith(selected_model_prefix)
            or name.startswith("outer_refit__")
        }
        for name, model in retained_models.items():
            with (model_dir / f"{name}.pkl").open("wb") as handle:
                pickle.dump(model, handle)
        predictions.reset_index(names="signal_date").to_csv(
            temp / "predictions.csv", index=False
        )
        positions.to_csv(temp / "positions.csv", index=False)
        strategy_returns.to_csv(temp / "strategy_returns.csv", index=False)
        with R.start(experiment_name=spec.experiment):
            recorder = R.get_recorder()
            R.log_params(
                train_start=spec.train_start,
                development_folds=",".join(map(str, spec.development_folds)),
                outer=f"{spec.outer[0]}:{data_audit['common_feature_end']}",
                feature_sets="current10,compact_reintegrated,full_reintegrated",
                targets="cash_protection_5d,future_vol_5d",
                selection_uses_outer_test="false",
            )
            R.log_metrics(
                selected_ratio_10bps=float(selected["ratio_10bps"]),
                selected_ratio_25bps=float(selected["ratio_25bps"]),
                selected_maxdd_10bps=float(selected["max_drawdown_pct"]),
                selected_avg_cash_pct=float(selected["avg_cash_weight_pct"]),
            )
            for artifact in [model_dir, *temp.glob("*.csv")]:
                recorder.log_artifact(artifact.as_posix(), artifact_path=spec.experiment)
            recorder_id = recorder.id
            recorder_local_dir = Path(recorder.get_local_dir()).resolve()
            all_gate = bool(selected["all_gate"])
            promotion_decision = (
                "promote_cash_floor_relative_feature_reintegration"
                if all_gate
                else "needs_iteration_tail_risk_after_feature_reintegration"
            )
            summary = {
                "experiment": spec.experiment,
                "card_type": "cash_floor_relative_target_feature_reintegration_research_card",
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
                ],
                "feature_reintegration": {
                    "feature_sets": {
                        key: values for key, values in feature_sets.items() if "__" not in key
                    },
                    "compact_head_features": {
                        "cash_protection_5d": feature_sets[
                            "compact_reintegrated__cash_protection_5d"
                        ],
                        "future_vol_5d": feature_sets[
                            "compact_reintegrated__future_vol_5d"
                        ],
                    },
                    "excluded_non_pit_or_short_history_sources": data_audit[
                        "excluded_sources"
                    ],
                },
                "objective": {
                    "cash_protection_5d": "log_wealth(max_cash_60pct) - log_wealth(min_cash_10pct), fixed risky sleeve, next five open-to-open returns",
                    "future_vol_5d": "std(next five SOXX open-to-open returns)",
                    "relative_alpha_5d": "log_wealth(min_cash_10pct) - log_wealth(SOXX), audit_only",
                    "aggregation": "weighted_mean_not_pointwise_max",
                },
                "selected_model": selected,
                "portfolio_contract": {
                    "tradable_assets": ["SOXL", "SOXX", "cash"],
                    "soxs_feature_only": True,
                    "minimum_cash": spec.minimum_cash,
                    "maximum_cash": spec.maximum_cash,
                    "risky_sleeve_soxl_share": spec.risky_sleeve_soxl_share,
                    "risky_sleeve_soxx_share": 1.0 - spec.risky_sleeve_soxl_share,
                    "smoothing_span": spec.smoothing_span,
                    "rebalance_threshold": spec.rebalance_threshold,
                    "timing": "signal T close; target T+1 open to T+2 open",
                    "costs_bps": list(spec.costs_bps),
                },
                "development_selected": _json_ready(
                    comparison.loc[
                        comparison["candidate"].eq(selected["candidate"])
                    ].to_dict(orient="records")
                ),
                "outer_selected_diagnostic": _json_ready(
                    outer_comparison.to_dict(orient="records")
                ),
                "recorder_id": recorder_id,
                "recorder_uri": recorder_uri.resolve().as_posix(),
                "recorder_local_dir": recorder_local_dir.as_posix(),
                "sidecars": {
                    "selected_model": "selected_model.json",
                    "model_selection": "model_selection.csv",
                    "feature_manifest": "feature_manifest.csv",
                    "feature_oos_audit": "feature_oos_audit.csv",
                    "calibration_summary": "calibration_summary.csv",
                    "backtest_benchmark_comparison": "backtest_benchmark_comparison.csv",
                    "backtest_walk_forward_metrics": "backtest_walk_forward_metrics.csv",
                    "risk_analysis": "risk_analysis.csv",
                    "analysis_summary": "analysis_summary.json",
                    "failure_modes": "failure_modes.md",
                },
                "artifact_policy": {
                    "policy": "slim_card_recorder_first",
                    "workflow_config": "workflow.yaml",
                    "recorder_only_artifacts": list(RECORDER_ONLY_ARTIFACTS),
                    "retained_model_scope": "selected_development_folds_and_outer_refit_only",
                    "retained_model_count": len(retained_models),
                    "retained_portfolio_scope": "selected_candidate_only",
                },
            }
            _write_card(
                card,
                _json_ready(summary),
                feature_manifest,
                feature_audit,
                calibrations,
                model_selection,
                comparison,
                folds,
                outer_comparison,
                outer_folds,
            )
            for filename in summary["sidecars"].values():
                path = card / filename
                if path.exists():
                    recorder.log_artifact(path.as_posix(), artifact_path=spec.experiment)
            _remove_qlib_code_snapshot_artifacts(recorder_local_dir)
    return _json_ready(summary)


def _write_card(
    card: Path,
    summary: dict[str, Any],
    feature_manifest: pd.DataFrame,
    feature_audit: pd.DataFrame,
    calibrations: pd.DataFrame,
    model_selection: pd.DataFrame,
    comparison: pd.DataFrame,
    folds: pd.DataFrame,
    outer_comparison: pd.DataFrame,
    outer_folds: pd.DataFrame,
) -> None:
    feature_manifest.to_csv(card / "feature_manifest.csv", index=False)
    feature_audit.to_csv(card / "feature_oos_audit.csv", index=False)
    calibrations.to_csv(card / "calibration_summary.csv", index=False)
    model_selection.to_csv(card / "model_selection.csv", index=False)
    selected_name = summary["selected_model"]["candidate"]
    selected_development = comparison.loc[comparison["candidate"].eq(selected_name)].assign(
        segment="development_walk_forward"
    )
    benchmark_comparison = pd.concat(
        [selected_development, outer_comparison], ignore_index=True
    )
    benchmark_comparison.to_csv(card / "backtest_benchmark_comparison.csv", index=False)
    selected_folds = folds.loc[folds["candidate"].eq(selected_name)].assign(
        segment="development_walk_forward"
    )
    pd.concat([selected_folds, outer_folds], ignore_index=True).to_csv(
        card / "backtest_walk_forward_metrics.csv", index=False
    )
    benchmark_comparison.loc[
        :,
        [
            "segment",
            "candidate",
            "cost_bps",
            "return_pct",
            "max_drawdown_pct",
            "worst_day_pct",
            "cvar5_pct",
            "worst_rolling_5d_pct",
            "avg_cash_weight_pct",
        ],
    ].to_csv(card / "risk_analysis.csv", index=False)
    (card / "selected_model.json").write_text(
        json.dumps(summary["selected_model"], indent=2) + "\n", encoding="utf-8"
    )
    analysis = {
        "promotion_decision": summary["promotion_decision"],
        "selected_model": summary["selected_model"],
        "development_selected": summary["development_selected"],
        "outer_selected_diagnostic": summary["outer_selected_diagnostic"],
        "feature_oos_audit": _json_ready(feature_audit.to_dict(orient="records")),
    }
    (card / "analysis_summary.json").write_text(
        json.dumps(analysis, indent=2) + "\n", encoding="utf-8"
    )
    (card / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    dev10 = next(row for row in summary["development_selected"] if row["cost_bps"] == 10)
    dev25 = next(row for row in summary["development_selected"] if row["cost_bps"] == 25)
    outer10 = next(row for row in summary["outer_selected_diagnostic"] if row["cost_bps"] == 10)
    compact_ridge = next(
        row
        for row in analysis["feature_oos_audit"]
        if row["feature_set"] == "compact_reintegrated"
        and row["model_family"] == "ridge"
        and row["target"] == "cash_protection_5d"
    )
    current_ridge = next(
        row
        for row in analysis["feature_oos_audit"]
        if row["feature_set"] == "current10"
        and row["model_family"] == "ridge"
        and row["target"] == "cash_protection_5d"
    )
    conclusion = f"""# 中文结论

## 跑了什么

把历史特征按新目标重新整合，而不是复制旧模型 importance。主候选包括当前 10 特征、精简 reintegrated 特征和全量 reintegrated 特征；模型比较 Ridge 与 ExtraTrees。现金头改为未来五日 `60%现金配置相对10%现金配置` 的直接对数财富优势，波动头只作为加权风险限制器，不再逐日取两个头的最大值。2016–2022 expanding walk-forward，20 行 purge；2023–{summary['data_audit']['common_feature_end']} outer 仅诊断。

## 开发期结果

选中 `{summary['selected_model']['candidate']}`，选择政策 `{summary['selected_model']['selection_policy']}`。10bp 收益 {dev10['return_pct']:.2f}%，相对 SOXX 财富 {dev10['terminal_wealth_ratio_vs_soxx']:.4f}，MaxDD {dev10['max_drawdown_pct']:.2f}%，最差单日 {dev10['worst_day_pct']:.2f}%，CVaR5 {dev10['cvar5_pct']:.2f}%，最差五日 {dev10['worst_rolling_5d_pct']:.2f}%，平均现金 {dev10['avg_cash_weight_pct']:.2f}%。25bp 相对财富 {dev25['terminal_wealth_ratio_vs_soxx']:.4f}；正相对收益年份 {summary['selected_model']['positive_years']}/7，年度中位财富比 {summary['selected_model']['median_year_ratio']:.4f}。

精简 reintegrated 特征把 Ridge 现金保护头的 pooled Spearman 从 {current_ridge['pooled_spearman']:.4f} 提高到 {compact_ridge['pooled_spearman']:.4f}，但没有转化成更好的组合尾部。最终选择仍是 `{summary['selected_model']['feature_set']}`，说明旧特征在新标签下只有预测增量，没有稳定的决策增量。选中方案通过了开发期相对财富门，但严格尾部风险门={summary['selected_model']['risk_gate']}。全量特征只作为开发期对照，不因特征更多而优先。

## Outer 诊断

10bp 收益 {outer10['return_pct']:.2f}%，相对 SOXX 财富 {outer10['terminal_wealth_ratio_vs_soxx']:.4f}，MaxDD {outer10['max_drawdown_pct']:.2f}%。outer 不参与特征、模型、波动权重或 router 选择。

## 决定、风险和下一步

`{summary['promotion_decision']}`。趋势、距高点、VIX/实现波动、杠杆衰减、tracking error 和跨资产扩散可进入新目标候选，但本轮没有证明它们应取代当前 10 特征。它们仍不能可靠预测足以把 MaxDD 压到 -20% 的尾部时点。SOXX 成分股历史因非 PIT，宏观/硬件/盘中特征因 2019 后才覆盖，均未进入主模型。下一步不应再扩大日频价格特征数量，而应增加真正提前于尾部的 PIT 风险数据或接受风险门与相对收益目标之间的不可行边界。

不构成买卖建议。
"""
    (card / "conclusion.md").write_text(conclusion, encoding="utf-8")
    (card / "failure_modes.md").write_text(
        """# Failure modes

- More features can reduce walk-forward target IC through collinearity and regime drift.
- Current SOXX constituent history uses a current snapshot and is excluded from the promoted feature set.
- Macro, hardware-cycle, and intraday sources start in 2019 and cannot support the full 2016-2022 development contract.
- Five-day volatility is predictive but not directionally equivalent to cash outperformance.
- Direct cash-protection labels improve benchmark alignment but do not guarantee a -20% path drawdown.
- 2023 onward is previously observed and remains diagnostic only.
""",
        encoding="utf-8",
    )
    workflow = f"""experiment: {summary['experiment']}
card_type: {summary['card_type']}
provider_uri:
  execution: {summary['data_audit']['execution_provider_uri']}
  daily: {summary['data_audit']['daily_provider_uri']}
  normalized_cross_asset: {summary['data_audit']['normalized_provider_uri']}
recorder_uri: {summary['recorder_uri']}
recorder_id: {summary['recorder_id']}
qlib:
  market: all
  region: us
  data_api: qlib.data.D.features
  dataset: [DatasetH, DataHandlerLP, StaticDataLoader]
  recorder: qlib.workflow.R
walk_forward:
  train_start: 2011-01-03
  development_folds: [2016, 2017, 2018, 2019, 2020, 2021, 2022]
  purge_rows: 20
  outer: 2023-{summary['data_audit']['common_feature_end']}_diagnostic_only
  selection_uses_outer_test: false
objective:
  targets: [cash_protection_5d, future_vol_5d]
  aggregation: weighted_mean
feature_sets: [current10, compact_reintegrated, full_reintegrated]
model_families: [ridge, extra_trees]
portfolio:
  cash_bounds: [0.10, 0.60]
  risky_sleeve: {{SOXL: 0.30, SOXX: 0.70}}
  smoothing_span: 3
  rebalance_threshold: 0.05
  costs_bps: [10, 25]
promotion:
  decision: {summary['promotion_decision']}
artifact_policy:
  policy: slim_card_recorder_first
  workflow_config: workflow.yaml
  machine_entrypoint: summary.json
  recorder_only_artifacts:
    - artifacts/{summary['experiment']}/models/
    - artifacts/{summary['experiment']}/predictions.csv
    - artifacts/{summary['experiment']}/positions.csv
    - artifacts/{summary['experiment']}/strategy_returns.csv
"""
    (card / "workflow.yaml").write_text(workflow, encoding="utf-8")


if __name__ == "__main__":
    print(json.dumps(run_cash_floor_relative_feature_reintegration_card(), indent=2))
