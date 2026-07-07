from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline

from src.data import QlibProviderConfig, init_qlib_provider, load_qlib_feature_frame
from src.workflow.dynamic_soxx_soxl_cash_allocation_wf import (
    CALIBRATION_ASSET,
    ETF_RETURN_COL,
    UNDERLYING_RETURN_COL,
)
from src.workflow.dynamic_soxx_soxl_cash_allocation_wf_v2 import _return_drawdown
from src.workflow.multi_3x_etf_normalized_nested_wf import _json_ready, _resolve_root
from src.workflow.trainable_regime_policy_nested_wf import _remove_qlib_code_snapshot_artifacts


EXPERIMENT = "dynamic_soxx_soxl_cash_allocation_wf_v6"
CARD_TYPE = "dynamic_soxx_soxl_cash_same_day_15m_xgb_threshold_calibration_card"
SOURCE_EXPERIMENT = "dynamic_soxx_soxl_cash_allocation_wf_v1"
SOXX_EXCESS_FLOOR_PCT = 20.0
MAX_DRAWDOWN_FLOOR_PCT = -10.0
FORBIDDEN_FEATURE_COLUMNS = (
    ETF_RETURN_COL,
    UNDERLYING_RETURN_COL,
    "relative_alpha_5d_label",
    "underlying_positive_5d_label",
    "tail_risk_5d_label",
    "relative_excess_open_to_open_5d",
)


@dataclass(frozen=True)
class DynamicSoxxSoxlCashAllocationWfV6Spec:
    root: Path
    experiment: str = EXPERIMENT
    train_start: str = "2011-01-03"
    intraday_start: str = "2018-12-31 09:45:00"
    prediction_years: tuple[int, ...] = tuple(range(2019, 2027))
    test_years: tuple[int, ...] = (2023, 2024, 2025, 2026)
    validation_window_years: int = 4
    cost_per_turnover: float = 0.0003

    @property
    def card_dir(self) -> Path:
        return self.root / "research" / self.experiment

    @property
    def source_card_dir(self) -> Path:
        return self.root / "research" / SOURCE_EXPERIMENT

    @property
    def daily_provider_uri(self) -> Path:
        return self.root / "data" / "qlib" / SOURCE_EXPERIMENT

    @property
    def intraday_15m_provider_uri(self) -> Path:
        return self.root / "data" / "qlib" / "soxl_intraday_15min"


def run_dynamic_soxx_soxl_cash_allocation_wf_v6_card(*, root: Path | str | None = None) -> dict[str, Any]:
    spec = DynamicSoxxSoxlCashAllocationWfV6Spec(root=_resolve_root(root))
    return _run_card(spec)


def _run_card(spec: DynamicSoxxSoxlCashAllocationWfV6Spec) -> dict[str, Any]:
    card = spec.card_dir
    card.mkdir(parents=True, exist_ok=True)
    source_summary = json.loads((spec.source_card_dir / "summary.json").read_text(encoding="utf-8"))
    common_end = str(source_summary["common_data_end"])
    source_feature_cols = _source_causal_feature_cols(source_summary)
    frame, feature_cols, feature_audit = _load_model_frame(
        spec,
        common_end=common_end,
        source_feature_cols=source_feature_cols,
    )
    predictions = _rolling_xgb_predictions(frame, feature_cols=feature_cols, spec=spec)
    benchmark = _benchmark_metrics(predictions.loc[predictions["year"].isin(spec.test_years)].copy())
    param_grid = _candidate_param_grid()
    capacity = _capacity_grid(predictions, param_grid=param_grid, spec=spec, benchmark=benchmark)
    capacity_summary = _capacity_summary(capacity)
    nested = _nested_walk_forward(predictions, param_grid=param_grid, spec=spec)
    nested_summary = _nested_summary(nested, predictions=predictions, spec=spec, benchmark=benchmark)
    promotion_decision = (
        "promoted_same_day_15m_dynamic_soxx_soxl_cash_goal_passed"
        if nested_summary["required_policy_satisfied"]
        else "not_promoted_same_day_15m_capacity_and_nested_calibration_failed"
    )

    sidecars = {
        "capacity_grid": "capacity_grid.csv",
        "nested_walk_forward": "nested_walk_forward.csv",
        "risk_analysis": "risk_analysis.csv",
        "selected_model": "selected_model.json",
        "analysis_summary": "analysis_summary.json",
        "failure_modes": "failure_modes.md",
    }

    import qlib
    from qlib.workflow import R

    _ = qlib
    recorder_uri = card / "qlib_recorder"
    recorder_uri.mkdir(parents=True, exist_ok=True)
    R.set_uri(recorder_uri.resolve().as_posix())
    with tempfile.TemporaryDirectory(prefix=f"{spec.experiment}_") as tmp:
        tmp_dir = Path(tmp)
        capacity_path = tmp_dir / "capacity_grid.csv"
        nested_path = tmp_dir / "nested_walk_forward.csv"
        capacity.to_csv(capacity_path, index=False)
        nested.to_csv(nested_path, index=False)
        with R.start(experiment_name=spec.experiment):
            recorder = R.get_recorder()
            R.log_params(
                daily_provider_uri=spec.daily_provider_uri.resolve().as_posix(),
                intraday_15m_provider_uri=spec.intraday_15m_provider_uri.resolve().as_posix(),
                benchmark="SOXX",
                benchmark_basis="open_to_open",
                allocation_assets="SOXX,SOXL,cash",
                model_family="xgb",
                same_day_15m_timing="session_close_known_before_next_open_signal",
                nested_validation_window_years=str(spec.validation_window_years),
                objective="SOXX plus 20 percentage points with max drawdown above -10%",
            )
            R.log_metrics(
                soxx_return_pct=float(benchmark["return_pct"]),
                soxx_max_drawdown_pct=float(benchmark["max_drawdown_pct"]),
                capacity_passes_target_count=float(capacity_summary["passes_target_count"]),
                capacity_best_excess_candidate_return_pct=float(capacity_summary["best_excess_candidate_return_pct"]),
                capacity_best_excess_candidate_max_drawdown_pct=float(
                    capacity_summary["best_excess_candidate_max_drawdown_pct"]
                ),
                capacity_best_drawdown_safe_return_pct=float(capacity_summary["best_drawdown_safe_return_pct"]),
                nested_stitched_return_pct=float(nested_summary["stitched_return_pct"]),
                nested_stitched_max_drawdown_pct=float(nested_summary["stitched_max_drawdown_pct"]),
                nested_excess_vs_soxx_pct=float(nested_summary["stitched_excess_vs_soxx_pct"]),
            )
            recorder.log_artifact(capacity_path.as_posix(), artifact_path=spec.experiment)
            recorder.log_artifact(nested_path.as_posix(), artifact_path=spec.experiment)
            summary = _summary_payload(
                spec=spec,
                common_end=common_end,
                recorder_id=recorder.id,
                recorder_uri=recorder_uri,
                recorder_local_dir=Path(recorder.get_local_dir()).resolve(),
                sidecars=sidecars,
                source_summary=source_summary,
                feature_audit=feature_audit,
                benchmark=benchmark,
                capacity_summary=capacity_summary,
                nested_summary=nested_summary,
                predictions=predictions,
                promotion_decision=promotion_decision,
            )
            _write_sidecars(card, sidecars=sidecars, summary=summary, capacity=capacity, nested=nested)
            for rel_path in sidecars.values():
                path = card / rel_path
                if path.exists() and path.suffix != ".md":
                    recorder.log_artifact(path.as_posix(), artifact_path=spec.experiment)
            _remove_qlib_code_snapshot_artifacts(Path(recorder.get_local_dir()).resolve())
    _write_summary_files(card, summary)
    return summary


def _source_causal_feature_cols(source_summary: dict[str, Any]) -> list[str]:
    feature_sets = {
        str(item["feature_set"]): list(item["feature_cols"]) for item in source_summary["feature_sets"]["available"]
    }
    candidates = feature_sets.get("base_soxx_diffusion") or feature_sets.get("base") or []
    forbidden = {item.lower() for item in FORBIDDEN_FEATURE_COLUMNS}
    return [
        column
        for column in candidates
        if column.lower() not in forbidden
        and "label" not in column.lower()
        and "next" not in column.lower()
        and "future" not in column.lower()
        and "open_to_open" not in column.lower()
    ]


def _load_model_frame(
    spec: DynamicSoxxSoxlCashAllocationWfV6Spec,
    *,
    common_end: str,
    source_feature_cols: list[str],
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    daily = _load_daily_frame(spec, common_end=common_end, source_feature_cols=source_feature_cols)
    intraday = _same_day_15m_features(spec, common_end=common_end)
    frame = daily.merge(intraday, on="datetime", how="left")
    frame, custom_features = _attach_close_known_features(frame)
    same_day_15m_cols = [column for column in frame.columns if column.startswith("i15_same_")]
    feature_cols = list(dict.fromkeys([*source_feature_cols, *custom_features, *same_day_15m_cols]))
    feature_cols = [column for column in feature_cols if column in frame.columns]
    forbidden_used = sorted(set(feature_cols).intersection(FORBIDDEN_FEATURE_COLUMNS))
    if forbidden_used:
        raise ValueError(f"same-day 15m frame includes forbidden future-label columns: {forbidden_used}")

    daily_manifest = json.loads((spec.daily_provider_uri / "qlib_store_manifest.json").read_text(encoding="utf-8"))
    intraday_manifest = json.loads((spec.intraday_15m_provider_uri / "qlib_store_manifest.json").read_text(encoding="utf-8"))
    audit = {
        "daily_provider_uri": spec.daily_provider_uri.resolve().as_posix(),
        "intraday_15m_provider_uri": spec.intraday_15m_provider_uri.resolve().as_posix(),
        "daily_as_of": daily_manifest["as_of"],
        "intraday_as_of": intraday_manifest["as_of"],
        "common_end": common_end,
        "row_count": int(len(frame)),
        "feature_count": int(len(feature_cols)),
        "source_feature_count": int(len(source_feature_cols)),
        "custom_feature_count": int(len(custom_features)),
        "same_day_15m_feature_count": int(len(same_day_15m_cols)),
        "same_day_15m_timing": "session_close_known_before_next_open_signal",
        "same_day_15m_limitations": [
            "not valid for pre-open signal generation",
            "valid only for close-after-session signal used at the next open",
        ],
        "forbidden_feature_columns": list(FORBIDDEN_FEATURE_COLUMNS),
        "forbidden_feature_columns_used": forbidden_used,
        "causal_feature_columns": feature_cols,
    }
    return frame, feature_cols, audit


def _load_daily_frame(
    spec: DynamicSoxxSoxlCashAllocationWfV6Spec,
    *,
    common_end: str,
    source_feature_cols: list[str],
) -> pd.DataFrame:
    init_qlib_provider(QlibProviderConfig(provider_uri=spec.daily_provider_uri, market="all", region="us", kernels=1))
    from qlib.data import D

    fields = [
        "$open",
        "$close",
        "$volume",
        "$underlying_close",
        "$vix",
        f"${ETF_RETURN_COL}",
        f"${UNDERLYING_RETURN_COL}",
        *[f"${column}" for column in source_feature_cols],
    ]
    qframe = D.features([CALIBRATION_ASSET], fields, start_time=spec.train_start, end_time=common_end, freq="day")
    out = qframe.reset_index().rename(columns={field: field.removeprefix("$") for field in fields})
    out["datetime"] = pd.to_datetime(out["datetime"]).dt.normalize()
    out["instrument"] = out["instrument"].astype(str).str.upper()
    for column in ["open", "close", "volume", "underlying_close", "vix", ETF_RETURN_COL, UNDERLYING_RETURN_COL, *source_feature_cols]:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return out.dropna(subset=[ETF_RETURN_COL, UNDERLYING_RETURN_COL]).sort_values("datetime").reset_index(drop=True)


def _same_day_15m_features(spec: DynamicSoxxSoxlCashAllocationWfV6Spec, *, common_end: str) -> pd.DataFrame:
    end = (pd.Timestamp(common_end).normalize() + pd.Timedelta(hours=16)).strftime("%Y-%m-%d %H:%M:%S")
    qframe = load_qlib_feature_frame(
        QlibProviderConfig(
            provider_uri=spec.intraday_15m_provider_uri,
            market="all",
            region="us",
            freq="15min",
            kernels=1,
        ),
        instruments=["SOXL", "SOXX"],
        fields=["$open", "$high", "$low", "$close", "$volume", "$bar_count"],
        start=spec.intraday_start,
        end=end,
    )
    panel = qframe.reset_index().copy()
    panel["datetime"] = pd.to_datetime(panel["datetime"])
    panel["date"] = panel["datetime"].dt.normalize()
    panel["instrument"] = panel["instrument"].astype(str).str.upper()
    for column in ["$open", "$high", "$low", "$close", "$volume", "$bar_count"]:
        panel[column] = pd.to_numeric(panel[column], errors="coerce")

    rows = []
    for (instrument, date), group in panel.groupby(["instrument", "date"], sort=True):
        rows.append({"datetime": date, "instrument": instrument, **_aggregate_intraday_session(group)})
    daily = pd.DataFrame(rows)
    wide = daily.pivot(index="datetime", columns="instrument")
    wide.columns = [f"i15_same_{instrument.lower()}_{name}" for name, instrument in wide.columns]
    wide = wide.reset_index()
    wide["i15_same_soxl_minus_3x_soxx_ret"] = wide["i15_same_soxl_ret"] - 3.0 * wide["i15_same_soxx_ret"]
    wide["i15_same_rvol_ratio"] = wide["i15_same_soxl_rvol"] / (wide["i15_same_soxx_rvol"] + 1e-12)
    wide["i15_same_range_ratio"] = wide["i15_same_soxl_range"] / (wide["i15_same_soxx_range"] + 1e-12)
    return wide.replace([np.inf, -np.inf], np.nan)


def _aggregate_intraday_session(group: pd.DataFrame) -> dict[str, float]:
    group = group.sort_values("datetime")
    open_values = group["$open"].dropna()
    close_values = group["$close"].dropna()
    if open_values.empty or close_values.empty:
        open_ = close = np.nan
    else:
        open_ = float(open_values.iloc[0])
        close = float(close_values.iloc[-1])
    high = float(group["$high"].max())
    low = float(group["$low"].min())
    bar_returns = close_values.pct_change(fill_method=None).dropna()
    late_ret = np.nan
    if len(close_values) >= 2 and close_values.iloc[max(0, len(close_values) // 2)] != 0:
        late_ret = float(close / close_values.iloc[max(0, len(close_values) // 2)] - 1.0)
    expected_bar_count = float(group["$bar_count"].max())
    observed_bar_count = float(group["$close"].notna().sum())
    return {
        "ret": _safe_ratio(close, open_) - 1.0,
        "range": _safe_ratio(high, low) - 1.0,
        "rvol": float(bar_returns.std(ddof=0)) if len(bar_returns) else np.nan,
        "late": late_ret,
        "loc": float((close - low) / (high - low)) if np.isfinite(high) and high > low else np.nan,
        "volume": float(group["$volume"].sum()),
        "coverage": float(observed_bar_count / expected_bar_count)
        if np.isfinite(expected_bar_count) and expected_bar_count > 0
        else np.nan,
    }


def _attach_close_known_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = frame.copy()
    close = pd.to_numeric(out["close"], errors="coerce")
    open_ = pd.to_numeric(out["open"], errors="coerce")
    underlying = pd.to_numeric(out["underlying_close"], errors="coerce")
    vix = pd.to_numeric(out["vix"], errors="coerce")
    features = ["intraday_ret", "gap_ret"]
    out["intraday_ret"] = close / open_ - 1.0
    out["gap_ret"] = open_ / close.shift(1) - 1.0
    for window in [1, 2, 3, 5, 10, 20, 40, 60, 120, 200]:
        out[f"soxl_close_ret_{window}"] = close.pct_change(window)
        out[f"soxx_close_ret_{window}"] = underlying.pct_change(window)
        features.extend([f"soxl_close_ret_{window}", f"soxx_close_ret_{window}"])
    for window in [5, 20, 60, 120, 200]:
        min_periods = max(5, window // 3)
        out[f"soxl_ma_ratio_{window}"] = close / close.rolling(window, min_periods=min_periods).mean() - 1.0
        out[f"soxx_ma_ratio_{window}"] = (
            underlying / underlying.rolling(window, min_periods=min_periods).mean() - 1.0
        )
        out[f"soxl_drawdown_{window}"] = close / close.rolling(window, min_periods=min_periods).max() - 1.0
        out[f"soxx_drawdown_{window}"] = (
            underlying / underlying.rolling(window, min_periods=min_periods).max() - 1.0
        )
        out[f"soxx_realized_vol_{window}"] = underlying.pct_change().rolling(window, min_periods=min_periods).std()
        features.extend(
            [
                f"soxl_ma_ratio_{window}",
                f"soxx_ma_ratio_{window}",
                f"soxl_drawdown_{window}",
                f"soxx_drawdown_{window}",
                f"soxx_realized_vol_{window}",
            ]
        )
    for window in [5, 20, 60]:
        min_periods = max(5, window // 3)
        out[f"vix_z{window}"] = (
            (vix - vix.rolling(window, min_periods=min_periods).mean())
            / vix.rolling(window, min_periods=min_periods).std(ddof=0).replace(0.0, np.nan)
        )
        features.append(f"vix_z{window}")
    return out.replace([np.inf, -np.inf], np.nan), features


def _rolling_xgb_predictions(
    frame: pd.DataFrame,
    *,
    feature_cols: list[str],
    spec: DynamicSoxxSoxlCashAllocationWfV6Spec,
) -> pd.DataFrame:
    parts = []
    for year in spec.prediction_years:
        train = frame.loc[frame["datetime"].dt.year.lt(int(year))].copy()
        test = frame.loc[frame["datetime"].dt.year.eq(int(year))].copy().sort_values("datetime")
        if test.empty:
            continue
        train_x = train[feature_cols].astype("float64")
        test_x = test[feature_cols].astype("float64")
        soxl_model = _xgb_regressor(seed=3102)
        soxx_model = _xgb_regressor(seed=3102)
        up_model = _xgb_classifier(seed=3104)
        tail_model = _xgb_classifier(seed=3104)
        soxl_model.fit(train_x, train[ETF_RETURN_COL])
        soxx_model.fit(train_x, train[UNDERLYING_RETURN_COL])
        up_model.fit(train_x, train[ETF_RETURN_COL].gt(0.0).astype(int))
        tail_cut = float(train[UNDERLYING_RETURN_COL].quantile(0.08))
        tail_model.fit(train_x, train[UNDERLYING_RETURN_COL].le(tail_cut).astype(int))
        segment = test[["datetime", ETF_RETURN_COL, UNDERLYING_RETURN_COL]].copy()
        segment["year"] = int(year)
        segment["p_soxl"] = soxl_model.predict(test_x)
        segment["p_soxx"] = soxx_model.predict(test_x)
        segment["pup"] = _positive_proba(up_model, test_x)
        segment["ptail"] = _positive_proba(tail_model, test_x)
        parts.append(segment)
    return pd.concat(parts, ignore_index=True)


def _xgb_regressor(*, seed: int):
    from xgboost import XGBRegressor

    return make_pipeline(
        SimpleImputer(strategy="median"),
        XGBRegressor(
            n_estimators=100,
            max_depth=2,
            learning_rate=0.035,
            subsample=0.82,
            colsample_bytree=0.72,
            reg_alpha=0.25,
            reg_lambda=1.5,
            objective="reg:squarederror",
            random_state=seed,
            n_jobs=1,
        ),
    )


def _xgb_classifier(*, seed: int):
    from xgboost import XGBClassifier

    return make_pipeline(
        SimpleImputer(strategy="median"),
        XGBClassifier(
            n_estimators=90,
            max_depth=2,
            learning_rate=0.035,
            subsample=0.82,
            colsample_bytree=0.72,
            reg_alpha=0.25,
            reg_lambda=1.6,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=seed,
            n_jobs=1,
        ),
    )


def _positive_proba(model: Any, features: pd.DataFrame) -> np.ndarray:
    raw = model.predict_proba(features)
    classifier = model.named_steps[list(model.named_steps)[-1]]
    classes = list(classifier.classes_)
    if 1 not in classes:
        return np.zeros(len(features), dtype="float64")
    return raw[:, classes.index(1)]


def _candidate_param_grid() -> list[dict[str, float]]:
    rows = []
    for qret in [0.44, 0.45, 0.46, 0.47]:
        for qtail in [0.92, 0.93, 0.95, 0.96]:
            for pupthr in [0.575, 0.58, 0.585]:
                for fallback in [0.24, 0.25, 0.26, 0.28]:
                    for soxl_w in [0.90, 0.925, 0.95]:
                        rows.append(
                            {
                                "qret": qret,
                                "qtail": qtail,
                                "pupthr": pupthr,
                                "fallback": fallback,
                                "soxl_w": soxl_w,
                            }
                        )
    return rows


def _capacity_grid(
    predictions: pd.DataFrame,
    *,
    param_grid: list[dict[str, float]],
    spec: DynamicSoxxSoxlCashAllocationWfV6Spec,
    benchmark: dict[str, Any],
) -> pd.DataFrame:
    test = predictions.loc[predictions["year"].isin(spec.test_years)].copy()
    rows = []
    for params in param_grid:
        result = _simulate_policy(test, params=params, thresholds=None, cost_per_turnover=spec.cost_per_turnover)
        rows.append(
            {
                **params,
                **_policy_metric_payload(result, benchmark_return=float(benchmark["return_pct"])),
                "selection_uses_outer_test": True,
                "notes": "outer_test_distribution_capacity_diagnostic_not_promotable",
            }
        )
    return pd.DataFrame(rows)


def _nested_walk_forward(
    predictions: pd.DataFrame,
    *,
    param_grid: list[dict[str, float]],
    spec: DynamicSoxxSoxlCashAllocationWfV6Spec,
) -> pd.DataFrame:
    rows = []
    for outer_year in spec.test_years:
        valid = predictions.loc[
            predictions["year"].ge(int(outer_year) - int(spec.validation_window_years))
            & predictions["year"].lt(int(outer_year))
        ].copy()
        test = predictions.loc[predictions["year"].eq(int(outer_year))].copy()
        scored = []
        for params in param_grid:
            valid_result = _simulate_policy(valid, params=params, thresholds=None, cost_per_turnover=spec.cost_per_turnover)
            # Penalize drawdown misses, but retain enough return pressure to expose calibration drift.
            score = valid_result["return_pct"]
            if valid_result["max_drawdown_pct"] < MAX_DRAWDOWN_FLOOR_PCT:
                score -= 25.0 * abs(valid_result["max_drawdown_pct"] - MAX_DRAWDOWN_FLOOR_PCT)
            scored.append((score, valid_result, params))
        _, valid_result, selected = sorted(scored, key=lambda item: item[0], reverse=True)[0]
        test_result = _simulate_policy(
            test,
            params=selected,
            thresholds=valid_result["thresholds"],
            cost_per_turnover=spec.cost_per_turnover,
        )
        rows.append(
            {
                "outer_year": int(outer_year),
                **selected,
                "validation_return_pct": float(valid_result["return_pct"]),
                "validation_max_drawdown_pct": float(valid_result["max_drawdown_pct"]),
                "test_return_pct": float(test_result["return_pct"]),
                "test_max_drawdown_pct": float(test_result["max_drawdown_pct"]),
                "test_avg_soxx_weight": float(test_result["avg_soxx_weight"]),
                "test_avg_soxl_weight": float(test_result["avg_soxl_weight"]),
                "test_active_share": float(test_result["active_share"]),
                "test_mid_share": float(test_result["mid_share"]),
                "ret_threshold": float(valid_result["thresholds"][0]),
                "tail_threshold": float(valid_result["thresholds"][1]),
                "selection_uses_outer_test": False,
                "notes": "nested_threshold_calibration_uses_prior_validation_window",
            }
        )
    return pd.DataFrame(rows)


def _simulate_policy(
    frame: pd.DataFrame,
    *,
    params: dict[str, float],
    thresholds: tuple[float, float] | None,
    cost_per_turnover: float,
) -> dict[str, Any]:
    psoxl = pd.to_numeric(frame["p_soxl"], errors="coerce").to_numpy(dtype="float64")
    psoxx = pd.to_numeric(frame["p_soxx"], errors="coerce").to_numpy(dtype="float64")
    pup = pd.to_numeric(frame["pup"], errors="coerce").to_numpy(dtype="float64")
    ptail = pd.to_numeric(frame["ptail"], errors="coerce").to_numpy(dtype="float64")
    if thresholds is None:
        ret_threshold = float(np.nanquantile(psoxl, float(params["qret"])))
        tail_threshold = float(np.nanquantile(ptail, float(params["qtail"])))
    else:
        ret_threshold, tail_threshold = thresholds
    active = (psoxl >= ret_threshold) & (pup >= float(params["pupthr"])) & (ptail < tail_threshold)
    mid = (psoxx > 0.0) & ~active & (ptail < tail_threshold)
    soxx_weight = np.zeros(len(frame), dtype="float64")
    soxl_weight = np.zeros(len(frame), dtype="float64")
    soxx_weight[mid] = float(params["fallback"])
    soxl_weight[active] = float(params["soxl_w"])
    returns = _strategy_returns(
        frame,
        soxx_weight=soxx_weight,
        soxl_weight=soxl_weight,
        cost_per_turnover=cost_per_turnover,
    )
    ret, maxdd = _return_drawdown(returns)
    return {
        "return_pct": ret,
        "max_drawdown_pct": maxdd,
        "returns": returns,
        "thresholds": (ret_threshold, tail_threshold),
        "avg_soxx_weight": float(np.mean(soxx_weight)),
        "avg_soxl_weight": float(np.mean(soxl_weight)),
        "active_share": float(np.mean(active)),
        "mid_share": float(np.mean(mid)),
    }


def _strategy_returns(
    frame: pd.DataFrame,
    *,
    soxx_weight: np.ndarray,
    soxl_weight: np.ndarray,
    cost_per_turnover: float,
) -> np.ndarray:
    soxx_returns = pd.to_numeric(frame[UNDERLYING_RETURN_COL], errors="coerce").fillna(0.0).to_numpy(dtype="float64")
    soxl_returns = pd.to_numeric(frame[ETF_RETURN_COL], errors="coerce").fillna(0.0).to_numpy(dtype="float64")
    total = soxx_weight + soxl_weight
    over = total > 1.0
    soxx_weight = soxx_weight.copy()
    soxl_weight = soxl_weight.copy()
    soxx_weight[over] = soxx_weight[over] / total[over]
    soxl_weight[over] = soxl_weight[over] / total[over]
    turnover = np.abs(soxx_weight - np.r_[0.0, soxx_weight[:-1]]) + np.abs(soxl_weight - np.r_[0.0, soxl_weight[:-1]])
    return soxx_weight * soxx_returns + soxl_weight * soxl_returns - turnover * float(cost_per_turnover)


def _benchmark_metrics(test_frame: pd.DataFrame) -> dict[str, Any]:
    returns = pd.to_numeric(test_frame[UNDERLYING_RETURN_COL], errors="coerce").fillna(0.0).to_numpy(dtype="float64")
    ret, maxdd = _return_drawdown(returns)
    return {
        "asset": "SOXX",
        "return_pct": ret,
        "max_drawdown_pct": maxdd,
        "start": pd.Timestamp(test_frame["datetime"].min()).date().isoformat(),
        "end": pd.Timestamp(test_frame["datetime"].max()).date().isoformat(),
        "rows": int(len(test_frame)),
    }


def _policy_metric_payload(result: dict[str, Any], *, benchmark_return: float) -> dict[str, Any]:
    excess = float(result["return_pct"]) - float(benchmark_return)
    return {
        "return_pct": float(result["return_pct"]),
        "max_drawdown_pct": float(result["max_drawdown_pct"]),
        "excess_vs_soxx_pct": excess,
        "passes_excess_gate": bool(excess >= SOXX_EXCESS_FLOOR_PCT),
        "passes_drawdown_gate": bool(float(result["max_drawdown_pct"]) >= MAX_DRAWDOWN_FLOOR_PCT),
        "passes_target": bool(excess >= SOXX_EXCESS_FLOOR_PCT and float(result["max_drawdown_pct"]) >= MAX_DRAWDOWN_FLOOR_PCT),
        "avg_soxx_weight": float(result["avg_soxx_weight"]),
        "avg_soxl_weight": float(result["avg_soxl_weight"]),
        "active_share": float(result["active_share"]),
        "mid_share": float(result["mid_share"]),
    }


def _capacity_summary(capacity: pd.DataFrame) -> dict[str, Any]:
    passes = capacity.loc[capacity["passes_target"]].copy()
    best_pass = (
        passes.sort_values(["max_drawdown_pct", "return_pct"], ascending=[False, False]).iloc[0].to_dict()
        if not passes.empty
        else {}
    )
    best_dd_safe = capacity.loc[capacity["max_drawdown_pct"].ge(MAX_DRAWDOWN_FLOOR_PCT)].sort_values(
        "return_pct",
        ascending=False,
    ).iloc[0].to_dict()
    best_excess = capacity.loc[capacity["excess_vs_soxx_pct"].ge(SOXX_EXCESS_FLOOR_PCT)].sort_values(
        "max_drawdown_pct",
        ascending=False,
    ).iloc[0].to_dict()
    return {
        "model": "xgb",
        "uses_same_day_15m_features": True,
        "selection_role": "same_day_15m_xgb_threshold_capacity_diagnostic_not_promotable",
        "selection_uses_outer_test": True,
        "row_count": int(len(capacity)),
        "passes_target_count": int(capacity["passes_target"].sum()),
        "best_pass_return_pct": _maybe_float(best_pass, "return_pct"),
        "best_pass_max_drawdown_pct": _maybe_float(best_pass, "max_drawdown_pct"),
        "best_pass_excess_vs_soxx_pct": _maybe_float(best_pass, "excess_vs_soxx_pct"),
        "best_pass_params": _json_ready(
            {key: best_pass[key] for key in ["qret", "qtail", "pupthr", "fallback", "soxl_w"] if key in best_pass}
        ),
        "best_excess_candidate_return_pct": float(best_excess["return_pct"]),
        "best_excess_candidate_max_drawdown_pct": float(best_excess["max_drawdown_pct"]),
        "best_excess_candidate_excess_vs_soxx_pct": float(best_excess["excess_vs_soxx_pct"]),
        "best_excess_candidate_params": _json_ready(
            {key: best_excess[key] for key in ["qret", "qtail", "pupthr", "fallback", "soxl_w"]}
        ),
        "best_drawdown_safe_return_pct": float(best_dd_safe["return_pct"]),
        "best_drawdown_safe_max_drawdown_pct": float(best_dd_safe["max_drawdown_pct"]),
        "best_drawdown_safe_excess_vs_soxx_pct": float(best_dd_safe["excess_vs_soxx_pct"]),
    }


def _nested_summary(
    nested: pd.DataFrame,
    *,
    predictions: pd.DataFrame,
    spec: DynamicSoxxSoxlCashAllocationWfV6Spec,
    benchmark: dict[str, Any],
) -> dict[str, Any]:
    returns = []
    for row in nested.to_dict("records"):
        test = predictions.loc[predictions["year"].eq(int(row["outer_year"]))].copy()
        params = {key: float(row[key]) for key in ["qret", "qtail", "pupthr", "fallback", "soxl_w"]}
        result = _simulate_policy(
            test,
            params=params,
            thresholds=(float(row["ret_threshold"]), float(row["tail_threshold"])),
            cost_per_turnover=spec.cost_per_turnover,
        )
        returns.append(result["returns"])
    stitched_returns = np.concatenate(returns) if returns else np.asarray([], dtype="float64")
    ret, maxdd = _return_drawdown(stitched_returns)
    excess = ret - float(benchmark["return_pct"])
    required = bool(excess >= SOXX_EXCESS_FLOOR_PCT and maxdd >= MAX_DRAWDOWN_FLOOR_PCT)
    return {
        "selection_uses_outer_test": False,
        "validation_window_years": int(spec.validation_window_years),
        "stitched_return_pct": ret,
        "stitched_max_drawdown_pct": maxdd,
        "stitched_excess_vs_soxx_pct": excess,
        "required_policy_satisfied": required,
        "annual_rows": int(len(nested)),
        "blocking_reasons": []
        if required
        else ["same_day_15m_capacity_near_miss_does_not_survive_nested_threshold_calibration"],
    }


def _summary_payload(
    *,
    spec: DynamicSoxxSoxlCashAllocationWfV6Spec,
    common_end: str,
    recorder_id: str,
    recorder_uri: Path,
    recorder_local_dir: Path,
    sidecars: dict[str, str],
    source_summary: dict[str, Any],
    feature_audit: dict[str, Any],
    benchmark: dict[str, Any],
    capacity_summary: dict[str, Any],
    nested_summary: dict[str, Any],
    predictions: pd.DataFrame,
    promotion_decision: str,
) -> dict[str, Any]:
    return _json_ready(
        {
            "experiment": spec.experiment,
            "card_type": CARD_TYPE,
            "promotion_decision": promotion_decision,
            "common_data_end": common_end,
            "source_experiment": SOURCE_EXPERIMENT,
            "provider_uri": spec.daily_provider_uri.resolve().as_posix(),
            "intraday_15m_provider_uri": spec.intraday_15m_provider_uri.resolve().as_posix(),
            "benchmark_asset": "SOXX",
            "benchmark_basis": "open_to_open",
            "allocation_assets": ["SOXX", "SOXL", "cash"],
            "execution_timing": "same_day_15m_close_t_signal_next_open_to_following_open",
            "goal": {
                "excess_vs_soxx_floor_pct": SOXX_EXCESS_FLOOR_PCT,
                "max_drawdown_floor_pct": MAX_DRAWDOWN_FLOOR_PCT,
            },
            "benchmark": benchmark,
            "feature_audit": feature_audit,
            "same_day_15m_capacity": capacity_summary,
            "nested_calibration": nested_summary,
            "model_diagnostics": {
                "prediction_years": [int(year) for year in sorted(predictions["year"].unique())],
                "test_years": list(spec.test_years),
                "soxl_spearman_ic": float(pd.Series(predictions["p_soxl"]).corr(predictions[ETF_RETURN_COL], method="spearman")),
                "soxl_up_auc": _safe_auc(predictions[ETF_RETURN_COL].gt(0).astype(int), predictions["pup"]),
            },
            "source_v1": {
                "promotion_decision": source_summary.get("promotion_decision"),
                "recorder_id": source_summary.get("recorder_id"),
                "common_data_end": source_summary.get("common_data_end"),
            },
            "qlib_docs": {
                "data": "https://qlib.readthedocs.io/en/latest/component/data.html",
                "recorder": "https://qlib.readthedocs.io/en/latest/component/recorder.html",
                "workflow": "https://qlib.readthedocs.io/en/latest/component/workflow.html",
            },
            "qlib_native_api": [
                "qlib.init",
                "qlib.data.D.features",
                "qlib.workflow.R.start",
                "qlib.workflow.R.log_params",
                "qlib.workflow.R.log_metrics",
                "qlib.workflow.Recorder.log_artifact",
                "XGBRegressor",
                "XGBClassifier",
            ],
            "sidecars": sidecars,
            "artifact_policy": {
                "policy": "slim_card_recorder_first",
                "workflow_config": "workflow.yaml",
                "recorder_only_artifacts": [
                    f"artifacts/{spec.experiment}/capacity_grid.csv",
                    f"artifacts/{spec.experiment}/nested_walk_forward.csv",
                ],
            },
            "workflow_config": "workflow.yaml",
            "recorder_id": recorder_id,
            "recorder_uri": recorder_uri.resolve().as_posix(),
            "recorder_local_dir": recorder_local_dir.as_posix(),
            "recorder_artifacts": [
                f"artifacts/{spec.experiment}/capacity_grid.csv",
                f"artifacts/{spec.experiment}/nested_walk_forward.csv",
                f"artifacts/{spec.experiment}/analysis_summary.json",
                "meta.yaml",
            ],
        }
    )


def _write_sidecars(
    card: Path,
    *,
    sidecars: dict[str, str],
    summary: dict[str, Any],
    capacity: pd.DataFrame,
    nested: pd.DataFrame,
) -> None:
    capacity.to_csv(card / sidecars["capacity_grid"], index=False)
    nested.to_csv(card / sidecars["nested_walk_forward"], index=False)
    pd.DataFrame(
        [
            {
                "check": "same_day_15m_capacity",
                "status": "pass" if summary["same_day_15m_capacity"]["passes_target_count"] > 0 else "near_miss",
                "details": json.dumps(summary["same_day_15m_capacity"], ensure_ascii=False, sort_keys=True),
            },
            {
                "check": "nested_calibration",
                "status": "pass" if summary["nested_calibration"]["required_policy_satisfied"] else "fail",
                "details": json.dumps(summary["nested_calibration"], ensure_ascii=False, sort_keys=True),
            },
            {
                "check": "feature_audit",
                "status": "ok",
                "details": json.dumps(summary["feature_audit"], ensure_ascii=False, sort_keys=True),
            },
        ]
    ).to_csv(card / sidecars["risk_analysis"], index=False)
    selected = {
        "selected_model": None,
        "promotion_decision": summary["promotion_decision"],
        "reason": "same-day 15m XGB threshold capacity is a near miss and nested threshold calibration does not survive",
        "capacity_best_excess_candidate_params": summary["same_day_15m_capacity"]["best_excess_candidate_params"],
        "nested_calibration": summary["nested_calibration"],
    }
    (card / sidecars["selected_model"]).write_text(json.dumps(_json_ready(selected), ensure_ascii=False, indent=2), encoding="utf-8")
    analysis = {
        "source": "qlib_recorder_same_day_15m_xgb_threshold_calibration",
        "recorder_id": summary["recorder_id"],
        "same_day_15m_capacity": summary["same_day_15m_capacity"],
        "nested_calibration": summary["nested_calibration"],
        "model_diagnostics": summary["model_diagnostics"],
        "promotion_decision": summary["promotion_decision"],
    }
    (card / sidecars["analysis_summary"]).write_text(json.dumps(_json_ready(analysis), ensure_ascii=False, indent=2), encoding="utf-8")
    (card / sidecars["failure_modes"]).write_text(_failure_modes_text(summary), encoding="utf-8")


def _write_summary_files(card: Path, summary: dict[str, Any]) -> None:
    (card / "summary.json").write_text(json.dumps(_json_ready(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    (card / "conclusion.md").write_text(_conclusion_text(summary), encoding="utf-8")
    (card / "workflow.yaml").write_text(_workflow_yaml(summary), encoding="utf-8")


def _failure_modes_text(summary: dict[str, Any]) -> str:
    cap = summary["same_day_15m_capacity"]
    nested = summary["nested_calibration"]
    return (
        "# Failure Modes\n\n"
        "- same-day 15m XGB is the closest non-leaking structure tested so far, but the formal replay is still a capacity near miss.\n"
        "- The best excess-return candidate clears SOXX+20 but misses the -10% drawdown gate.\n"
        "- The best drawdown-safe candidate stays within the -10% drawdown gate but misses SOXX+20.\n"
        f"- Nested calibration falls to {_fmt_pct(nested['stitched_return_pct'])}% return and "
        f"{_fmt_pct(nested['stitched_max_drawdown_pct'])}% max drawdown.\n"
        f"- Best excess candidate return is {_fmt_pct(cap['best_excess_candidate_return_pct'])}% with "
        f"{_fmt_pct(cap['best_excess_candidate_max_drawdown_pct'])}% max drawdown.\n"
    )


def _conclusion_text(summary: dict[str, Any]) -> str:
    cap = summary["same_day_15m_capacity"]
    nested = summary["nested_calibration"]
    return (
        "# 中文结论\n\n"
        "- 做了什么：新增 v6 same-day 15m + XGB 阈值校准实验；same-day 15m 指收盘后已经完整可知的当日 15 分钟聚合，用于 next-open 信号，不用于盘前信号。\n"
        f"- 容量结果：capacity pass count={cap['passes_target_count']}；收益达标侧 best return={_fmt_pct(cap['best_excess_candidate_return_pct'])}%，"
        f"maxDD={_fmt_pct(cap['best_excess_candidate_max_drawdown_pct'])}%；回撤安全侧 best return={_fmt_pct(cap['best_drawdown_safe_return_pct'])}%，"
        f"maxDD={_fmt_pct(cap['best_drawdown_safe_max_drawdown_pct'])}%。\n"
        f"- nested 结果：rolling validation window={nested['validation_window_years']} 年，stitched return={_fmt_pct(nested['stitched_return_pct'])}%，"
        f"maxDD={_fmt_pct(nested['stitched_max_drawdown_pct'])}%，excess={_fmt_pct(nested['stitched_excess_vs_soxx_pct'])}%。\n"
        "- 为什么不晋升：正式 replay 下 capacity 已经是 near-miss；nested 阈值迁移后收益和回撤更不能同时过门槛。\n"
        f"- 是否晋升：不晋升；decision={summary['promotion_decision']}。\n"
        "- 下一步：优化方向应集中在在线阈值校准、score 分布漂移修正和按年度/波动 regime 的校准器，而不是继续扩大同一组 XGB 特征。\n"
        f"- qlib recorder：{summary['recorder_id']}\n"
        "- 不构成买卖建议。\n"
    )


def _workflow_yaml(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"experiment: {summary['experiment']}",
            f"card_type: {summary['card_type']}",
            f"provider_uri: {summary['provider_uri']}",
            f"intraday_15m_provider_uri: {summary['intraday_15m_provider_uri']}",
            f"recorder_uri: {summary['recorder_uri']}",
            "benchmark: SOXX",
            "benchmark_basis: open_to_open",
            "allocation_assets: [SOXX, SOXL, cash]",
            "model_selection:",
            "  source: same_day_15m_xgb_threshold_calibration",
            "  model: xgb",
            "  capacity_selection_uses_outer_test: true",
            "  nested_selection_uses_outer_test: false",
            "  promotion_allowed: false",
            "artifact_policy:",
            "  policy: slim_card_recorder_first",
            "  workflow_config: workflow.yaml",
            f"promotion_decision: {summary['promotion_decision']}",
            "",
        ]
    )


def _safe_auc(y_true: pd.Series, score: pd.Series) -> float:
    try:
        return float(roc_auc_score(y_true.astype(int), score.astype("float64")))
    except ValueError:
        return float("nan")


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or abs(float(denominator)) <= 1e-12:
        return np.nan
    return float(numerator) / float(denominator)


def _maybe_float(payload: dict[str, Any], key: str) -> float:
    if not payload or key not in payload:
        return float("nan")
    return float(payload[key])


def _fmt_pct(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "nan"
    if not np.isfinite(numeric):
        return "nan"
    return f"{numeric:.4f}"
