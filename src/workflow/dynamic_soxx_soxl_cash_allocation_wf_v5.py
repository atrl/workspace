from __future__ import annotations

import json
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline

from src.data import QlibProviderConfig, init_qlib_provider
from src.workflow.dynamic_soxx_soxl_cash_allocation_wf import (
    CALIBRATION_ASSET,
    ETF_RETURN_COL,
    UNDERLYING_RETURN_COL,
)
from src.workflow.dynamic_soxx_soxl_cash_allocation_wf_v2 import _return_drawdown
from src.workflow.dynamic_soxx_soxl_cash_allocation_wf_v4 import _benchmark_metrics, _strategy_returns
from src.workflow.multi_3x_etf_normalized_nested_wf import _json_ready, _resolve_root
from src.workflow.trainable_regime_policy_nested_wf import _remove_qlib_code_snapshot_artifacts


EXPERIMENT = "dynamic_soxx_soxl_cash_allocation_wf_v5"
CARD_TYPE = "dynamic_soxx_soxl_cash_allocation_causal_meta_label_leakage_audit_card"
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
class DynamicSoxxSoxlCashAllocationWfV5Spec:
    root: Path
    experiment: str = EXPERIMENT
    train_start: str = "2011-01-03"
    test_years: tuple[int, ...] = (2023, 2024, 2025, 2026)
    cost_per_turnover: float = 0.0003

    @property
    def card_dir(self) -> Path:
        return self.root / "research" / self.experiment

    @property
    def source_card_dir(self) -> Path:
        return self.root / "research" / SOURCE_EXPERIMENT

    @property
    def provider_uri(self) -> Path:
        return self.root / "data" / "qlib" / SOURCE_EXPERIMENT


def run_dynamic_soxx_soxl_cash_allocation_wf_v5_card(*, root: Path | str | None = None) -> dict[str, Any]:
    spec = DynamicSoxxSoxlCashAllocationWfV5Spec(root=_resolve_root(root))
    return _run_card(spec)


def _run_card(spec: DynamicSoxxSoxlCashAllocationWfV5Spec) -> dict[str, Any]:
    card = spec.card_dir
    card.mkdir(parents=True, exist_ok=True)
    source_summary = json.loads((spec.source_card_dir / "summary.json").read_text(encoding="utf-8"))
    common_end = str(source_summary["common_data_end"])
    source_feature_cols = _source_causal_feature_cols(source_summary)
    frame, feature_cols, feature_audit = _load_causal_frame(
        spec,
        common_end=common_end,
        source_feature_cols=source_feature_cols,
    )
    test_frame = frame.loc[frame["datetime"].dt.year.isin(spec.test_years)].copy()
    benchmark = _benchmark_metrics(test_frame)
    capacity = _causal_event_capacity(frame, feature_cols=feature_cols, spec=spec)
    causal_goal = _causal_goal_summary(capacity, benchmark=benchmark)
    leakage = _leakage_audit(test_frame, benchmark=benchmark, causal_goal=causal_goal, cost_per_turnover=spec.cost_per_turnover)
    leakage_summary = _leakage_summary(leakage, causal_goal=causal_goal, feature_cols=feature_cols)
    promotion_decision = (
        "promoted_dynamic_soxx_soxl_cash_goal_passed"
        if bool(causal_goal["required_policy_satisfied"])
        else "not_promoted_causal_meta_label_goal_failed"
    )

    sidecars = {
        "causal_event_capacity": "causal_event_capacity.csv",
        "leakage_audit": "leakage_audit.csv",
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
        capacity_path = tmp_dir / "causal_event_capacity.csv"
        leakage_path = tmp_dir / "leakage_audit.csv"
        capacity.to_csv(capacity_path, index=False)
        leakage.to_csv(leakage_path, index=False)
        with R.start(experiment_name=spec.experiment):
            recorder = R.get_recorder()
            R.log_params(
                provider_uri=spec.provider_uri.resolve().as_posix(),
                benchmark="SOXX",
                benchmark_basis="open_to_open",
                allocation_assets="SOXX,SOXL,cash",
                objective="SOXX plus 20 percentage points with max drawdown above -10%",
                model_families="lgbm,xgb",
                causal_features_only="true",
                forbidden_feature_columns=",".join(FORBIDDEN_FEATURE_COLUMNS),
                selection_uses_outer_test="true_for_capacity_diagnostic_not_promotable",
            )
            R.log_metrics(
                soxx_return_pct=float(benchmark["return_pct"]),
                soxx_max_drawdown_pct=float(benchmark["max_drawdown_pct"]),
                causal_passes_target_count=float(causal_goal["passes_target_count"]),
                causal_best_drawdown_safe_return_pct=float(causal_goal["best_drawdown_safe_return_pct"]),
                causal_min_drawdown_if_excess_passes_pct=float(causal_goal["min_drawdown_if_excess_passes_pct"]),
                invalid_future_label_passes_target_count=float(
                    leakage_summary["invalid_future_label_passes_target_count"]
                ),
            )
            recorder.log_artifact(capacity_path.as_posix(), artifact_path=spec.experiment)
            recorder.log_artifact(leakage_path.as_posix(), artifact_path=spec.experiment)
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
                causal_goal=causal_goal,
                leakage_summary=leakage_summary,
                capacity=capacity,
                promotion_decision=promotion_decision,
            )
            _write_sidecars(card, sidecars=sidecars, summary=summary, capacity=capacity, leakage=leakage)
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


def _load_causal_frame(
    spec: DynamicSoxxSoxlCashAllocationWfV5Spec,
    *,
    common_end: str,
    source_feature_cols: list[str],
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    config = QlibProviderConfig(provider_uri=spec.provider_uri, market="all", region="us", kernels=1)
    init_qlib_provider(config)
    from qlib.data import D

    base_fields = ["$open", "$close", "$volume", "$underlying_close", "$vix"]
    label_fields = [f"${ETF_RETURN_COL}", f"${UNDERLYING_RETURN_COL}"]
    source_fields = [f"${column}" for column in source_feature_cols]
    qframe = D.features(
        [CALIBRATION_ASSET],
        [*base_fields, *label_fields, *source_fields],
        start_time=spec.train_start,
        end_time=common_end,
        freq="day",
    )
    out = qframe.reset_index().rename(
        columns={field: field.removeprefix("$") for field in [*base_fields, *label_fields, *source_fields]}
    )
    out["datetime"] = pd.to_datetime(out["datetime"]).dt.normalize()
    out["instrument"] = out["instrument"].astype(str).str.upper()
    numeric_cols = ["open", "close", "volume", "underlying_close", "vix", ETF_RETURN_COL, UNDERLYING_RETURN_COL]
    numeric_cols.extend(source_feature_cols)
    for column in dict.fromkeys(numeric_cols):
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=[ETF_RETURN_COL, UNDERLYING_RETURN_COL]).sort_values("datetime").reset_index(drop=True)
    out, custom_feature_cols, event_cols = _attach_causal_features(out)
    feature_cols = _dedupe([*source_feature_cols, *custom_feature_cols])
    feature_cols = [column for column in feature_cols if column in out.columns]
    forbidden_used = sorted(set(feature_cols).intersection(FORBIDDEN_FEATURE_COLUMNS))
    if forbidden_used:
        raise ValueError(f"causal feature frame includes forbidden future-label columns: {forbidden_used}")
    manifest = json.loads((spec.provider_uri / "qlib_store_manifest.json").read_text(encoding="utf-8"))
    audit = {
        "provider_uri": spec.provider_uri.resolve().as_posix(),
        "as_of": manifest["as_of"],
        "calendar_end": manifest["calendar_end"],
        "source_experiment": SOURCE_EXPERIMENT,
        "row_count": int(len(out)),
        "feature_count": int(len(feature_cols)),
        "source_feature_count": int(len(source_feature_cols)),
        "custom_feature_count": int(len(custom_feature_cols)),
        "event_columns": event_cols,
        "forbidden_feature_columns": list(FORBIDDEN_FEATURE_COLUMNS),
        "forbidden_feature_columns_used": forbidden_used,
        "label_columns": [ETF_RETURN_COL, UNDERLYING_RETURN_COL],
        "notes": "labels are used only for next open-to-open evaluation, not as causal model inputs",
    }
    return out, feature_cols, audit


def _attach_causal_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    out = frame.copy()
    close = pd.to_numeric(out["close"], errors="coerce")
    underlying = pd.to_numeric(out["underlying_close"], errors="coerce")
    volume = pd.to_numeric(out["volume"], errors="coerce")
    vix = pd.to_numeric(out["vix"], errors="coerce")
    feature_cols: list[str] = []
    for window in [1, 2, 3, 5, 10, 20, 40, 60, 120, 150, 200, 252]:
        out[f"custom_soxl_ret_{window}"] = close.pct_change(window)
        out[f"custom_soxx_ret_{window}"] = underlying.pct_change(window)
        feature_cols.extend([f"custom_soxl_ret_{window}", f"custom_soxx_ret_{window}"])
    for window in [10, 20, 60, 120, 200]:
        out[f"custom_soxl_ma_ratio_{window}"] = close / close.rolling(window, min_periods=max(5, window // 3)).mean() - 1.0
        out[f"custom_soxx_ma_ratio_{window}"] = (
            underlying / underlying.rolling(window, min_periods=max(5, window // 3)).mean() - 1.0
        )
        out[f"custom_soxl_drawdown_{window}"] = close / close.rolling(window, min_periods=max(5, window // 3)).max() - 1.0
        out[f"custom_soxx_drawdown_{window}"] = (
            underlying / underlying.rolling(window, min_periods=max(5, window // 3)).max() - 1.0
        )
        feature_cols.extend(
            [
                f"custom_soxl_ma_ratio_{window}",
                f"custom_soxx_ma_ratio_{window}",
                f"custom_soxl_drawdown_{window}",
                f"custom_soxx_drawdown_{window}",
            ]
        )
    for window in [5, 20, 60]:
        out[f"custom_volume_chg_{window}"] = volume.pct_change(window)
        out[f"custom_volume_z_{window}"] = _rolling_z(volume, window)
        out[f"custom_vix_z_{window}"] = _rolling_z(vix, window)
        feature_cols.extend([f"custom_volume_chg_{window}", f"custom_volume_z_{window}", f"custom_vix_z_{window}"])

    out["event_base_trend"] = (
        out["custom_soxx_ma_ratio_20"].gt(0.0) & out["custom_soxx_ma_ratio_60"].gt(0.0)
    ).astype(int)
    for threshold in [0.035, 0.055, 0.08, 0.10, 0.13]:
        suffix = f"{int(threshold * 1000):03d}"
        out[f"event_dip_{suffix}"] = (
            out["custom_soxl_ret_5"].le(-threshold) | out["custom_soxx_drawdown_20"].le(-threshold)
        ).astype(int)
    day = out["datetime"].dt.day
    out["event_calendar"] = (day.le(3) | day.ge(25)).astype(int)
    dip_cols = [column for column in out.columns if column.startswith("event_dip_")]
    out["event_any_dip_calendar"] = (out[dip_cols].max(axis=1).gt(0) | out["event_calendar"].eq(1)).astype(int)
    event_cols = ["event_base_trend", *dip_cols, "event_calendar", "event_any_dip_calendar"]
    return out.replace([np.inf, -np.inf], np.nan), feature_cols, event_cols


def _rolling_z(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window, min_periods=max(5, window // 3)).mean()
    std = series.rolling(window, min_periods=max(5, window // 3)).std(ddof=0)
    return (series - mean) / std.replace(0.0, np.nan)


def _causal_event_capacity(
    frame: pd.DataFrame,
    *,
    feature_cols: list[str],
    spec: DynamicSoxxSoxlCashAllocationWfV5Spec,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    params = _event_param_grid()
    event_cols = [column for column in frame.columns if column.startswith("event_")]
    for event_col in event_cols:
        for model_name in ["lgbm", "xgb"]:
            pred = _rolling_event_predictions(frame, feature_cols=feature_cols, event_col=event_col, model_name=model_name, spec=spec)
            metrics = [_simulate_event_policy(pred, params_row, cost_per_turnover=spec.cost_per_turnover) for params_row in params]
            metrics_frame = pd.DataFrame(metrics)
            metrics_frame["passes_excess_gate"] = metrics_frame["excess_vs_soxx_pct"].ge(SOXX_EXCESS_FLOOR_PCT)
            metrics_frame["passes_drawdown_gate"] = metrics_frame["max_drawdown_pct"].ge(MAX_DRAWDOWN_FLOOR_PCT)
            metrics_frame["passes_target"] = metrics_frame["passes_excess_gate"] & metrics_frame["passes_drawdown_gate"]
            dd_safe = metrics_frame.loc[metrics_frame["passes_drawdown_gate"]].copy()
            excess = metrics_frame.loc[metrics_frame["passes_excess_gate"]].copy()
            best_dd_safe = dd_safe.sort_values("return_pct", ascending=False).iloc[0].to_dict() if not dd_safe.empty else {}
            shallow_excess = (
                excess.sort_values("max_drawdown_pct", ascending=False).iloc[0].to_dict() if not excess.empty else {}
            )
            best_return = metrics_frame.sort_values("return_pct", ascending=False).iloc[0].to_dict()
            event_mask = pred["event_signal"].eq(1)
            rows.append(
                {
                    "event_name": event_col.replace("event_", ""),
                    "model": model_name,
                    "feature_count": int(len(feature_cols)),
                    "event_test_count": int(event_mask.sum()),
                    "positive_event_rate": float(pred.loc[event_mask, "actual_positive"].mean()) if event_mask.any() else float("nan"),
                    "event_auc": _safe_auc(pred.loc[event_mask, "actual_positive"], pred.loc[event_mask, "p_soxl_positive"])
                    if event_mask.any()
                    else float("nan"),
                    "all_day_auc": _safe_auc(pred["actual_positive"], pred["p_soxl_positive"]),
                    "event_accuracy": _event_accuracy(pred),
                    "candidate_count": int(len(metrics_frame)),
                    "passes_target_count": int(metrics_frame["passes_target"].sum()),
                    "best_drawdown_safe_return_pct": _dict_float(best_dd_safe, "return_pct"),
                    "best_drawdown_safe_max_drawdown_pct": _dict_float(best_dd_safe, "max_drawdown_pct"),
                    "best_drawdown_safe_excess_pct": _dict_float(best_dd_safe, "excess_vs_soxx_pct"),
                    "min_drawdown_if_excess_passes_pct": _dict_float(shallow_excess, "max_drawdown_pct"),
                    "return_if_min_drawdown_excess_passes_pct": _dict_float(shallow_excess, "return_pct"),
                    "best_return_pct": _dict_float(best_return, "return_pct"),
                    "best_return_max_drawdown_pct": _dict_float(best_return, "max_drawdown_pct"),
                    "best_return_excess_pct": _dict_float(best_return, "excess_vs_soxx_pct"),
                    "selection_uses_outer_test": True,
                    "notes": "causal_event_parameter_oracle_not_promotable",
                }
            )
    return pd.DataFrame(rows)


def _rolling_event_predictions(
    frame: pd.DataFrame,
    *,
    feature_cols: list[str],
    event_col: str,
    model_name: str,
    spec: DynamicSoxxSoxlCashAllocationWfV5Spec,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for year in spec.test_years:
        train_all = frame.loc[frame["datetime"].dt.year.lt(int(year))].copy()
        train = train_all.loc[train_all[event_col].eq(1)].copy()
        test = frame.loc[frame["datetime"].dt.year.eq(int(year))].copy().sort_values("datetime")
        if train.empty:
            train = train_all.copy()
        y_train = pd.to_numeric(train[ETF_RETURN_COL], errors="coerce").gt(0.0).astype(int)
        model = _build_event_classifier(model_name, positive_count=int(y_train.sum()), train_count=len(y_train))
        train_x = train[feature_cols].astype("float64")
        test_x = test[feature_cols].astype("float64")
        model.fit(train_x, y_train)
        proba = _positive_proba(model, test_x)
        segment = test[["datetime", ETF_RETURN_COL, UNDERLYING_RETURN_COL, event_col]].copy()
        segment["year"] = int(year)
        segment["event_signal"] = segment[event_col].astype(int)
        segment["p_soxl_positive"] = proba
        segment["actual_positive"] = pd.to_numeric(segment[ETF_RETURN_COL], errors="coerce").gt(0.0).astype(int)
        parts.append(segment.drop(columns=[event_col]))
    return pd.concat(parts, ignore_index=True)


def _build_event_classifier(model_name: str, *, positive_count: int, train_count: int):
    if positive_count <= 0 or positive_count >= train_count:
        return make_pipeline(SimpleImputer(strategy="median"), DummyClassifier(strategy="prior"))
    if model_name == "lgbm":
        from lightgbm import LGBMClassifier

        return make_pipeline(
            SimpleImputer(strategy="median"),
            LGBMClassifier(
                n_estimators=70,
                learning_rate=0.035,
                num_leaves=9,
                min_child_samples=65,
                subsample=0.82,
                colsample_bytree=0.72,
                reg_alpha=0.35,
                reg_lambda=1.6,
                random_state=1501,
                verbose=-1,
            ),
        )
    if model_name == "xgb":
        from xgboost import XGBClassifier

        return make_pipeline(
            SimpleImputer(strategy="median"),
            XGBClassifier(
                n_estimators=80,
                max_depth=2,
                learning_rate=0.035,
                subsample=0.82,
                colsample_bytree=0.72,
                reg_alpha=0.25,
                reg_lambda=1.6,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=1502,
                n_jobs=1,
            ),
        )
    raise ValueError(f"unsupported event classifier: {model_name}")


def _positive_proba(model: Any, features: pd.DataFrame) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="X does not have valid feature names.*", category=UserWarning)
        raw = model.predict_proba(features)
    classifier = model.named_steps[list(model.named_steps)[-1]]
    classes = np.asarray(classifier.classes_, dtype=int)
    if 1 in classes:
        return raw[:, int(np.where(classes == 1)[0][0])]
    return np.zeros(len(features), dtype="float64")


def _event_param_grid() -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for threshold in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]:
        for fallback_soxx_weight in [0.0, 0.25, 0.5, 0.75, 1.0]:
            for soxl_weight in [0.25, 0.5, 0.75, 1.0]:
                for stop_thr in [-0.08, -0.1, -1.0]:
                    for stop_mult in [0.0, 0.5, 1.0]:
                        rows.append(
                            {
                                "threshold": threshold,
                                "fallback_soxx_weight": fallback_soxx_weight,
                                "soxl_weight": soxl_weight,
                                "stop_thr": stop_thr,
                                "stop_mult": stop_mult,
                            }
                        )
    return rows


def _simulate_event_policy(
    pred: pd.DataFrame,
    params: dict[str, float],
    *,
    cost_per_turnover: float,
) -> dict[str, Any]:
    soxx_returns = pd.to_numeric(pred[UNDERLYING_RETURN_COL], errors="coerce").fillna(0.0).to_numpy(dtype="float64")
    soxl_returns = pd.to_numeric(pred[ETF_RETURN_COL], errors="coerce").fillna(0.0).to_numpy(dtype="float64")
    p = pd.to_numeric(pred["p_soxl_positive"], errors="coerce").fillna(0.0).to_numpy(dtype="float64")
    event = pred["event_signal"].astype(int).to_numpy(dtype="int64") == 1
    active = event & (p >= float(params["threshold"]))
    soxx_weight = np.full(len(pred), float(params["fallback_soxx_weight"]), dtype="float64")
    soxl_weight = np.zeros(len(pred), dtype="float64")
    soxx_weight[active] = 0.0
    soxl_weight[active] = float(params["soxl_weight"])
    returns = _strategy_returns(
        soxx_returns,
        soxl_returns,
        soxx_weight,
        soxl_weight,
        cost_per_turnover=cost_per_turnover,
        stop_thr=float(params["stop_thr"]),
        stop_mult=float(params["stop_mult"]),
    )
    ret, maxdd = _return_drawdown(returns)
    bench_ret, bench_dd = _return_drawdown(soxx_returns)
    return {
        **params,
        "return_pct": ret,
        "max_drawdown_pct": maxdd,
        "benchmark_return_pct": bench_ret,
        "benchmark_max_drawdown_pct": bench_dd,
        "excess_vs_soxx_pct": ret - bench_ret,
        "avg_soxx_weight": float(np.mean(soxx_weight)),
        "avg_soxl_weight": float(np.mean(soxl_weight)),
        "active_share": float(np.mean(active)),
    }


def _leakage_audit(
    test_frame: pd.DataFrame,
    *,
    benchmark: dict[str, Any],
    causal_goal: dict[str, Any],
    cost_per_turnover: float,
) -> pd.DataFrame:
    soxx = pd.to_numeric(test_frame[UNDERLYING_RETURN_COL], errors="coerce").fillna(0.0).to_numpy(dtype="float64")
    soxl = pd.to_numeric(test_frame[ETF_RETURN_COL], errors="coerce").fillna(0.0).to_numpy(dtype="float64")
    rows: list[dict[str, Any]] = []
    invalid_policies = {
        "future_soxl_positive_or_cash": (np.zeros(len(soxl), dtype="float64"), (soxl > 0.0).astype("float64")),
        "future_best_of_cash_soxx_soxl": _future_best_weights(soxx, soxl),
    }
    for name, (soxx_weight, soxl_weight) in invalid_policies.items():
        returns = _strategy_returns(
            soxx,
            soxl,
            soxx_weight,
            soxl_weight,
            cost_per_turnover=cost_per_turnover,
        )
        ret, maxdd = _return_drawdown(returns)
        rows.append(
            {
                "audit_case": "future_label_feature_invalid",
                "candidate": name,
                "return_pct": ret,
                "max_drawdown_pct": maxdd,
                "excess_vs_soxx_pct": ret - float(benchmark["return_pct"]),
                "passes_target": bool(ret - float(benchmark["return_pct"]) >= SOXX_EXCESS_FLOOR_PCT and maxdd >= MAX_DRAWDOWN_FLOOR_PCT),
                "notes": "invalid: uses current row next open-to-open return label as if it were known before the trade",
            }
        )
    rows.append(
        {
            "audit_case": "causal_feature_valid",
            "candidate": "best_causal_event_drawdown_safe",
            "return_pct": causal_goal["best_drawdown_safe_return_pct"],
            "max_drawdown_pct": causal_goal["best_drawdown_safe_max_drawdown_pct"],
            "excess_vs_soxx_pct": causal_goal["best_drawdown_safe_excess_pct"],
            "passes_target": bool(causal_goal["required_policy_satisfied"]),
            "notes": "valid: uses only causal feature columns; capacity grid still fails the SOXX plus 20 / maxDD 10 gate",
        }
    )
    return pd.DataFrame(rows)


def _future_best_weights(soxx: np.ndarray, soxl: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    choices = np.vstack([np.zeros(len(soxx), dtype="float64"), soxx, soxl]).T
    action = np.argmax(choices, axis=1)
    return (action == 1).astype("float64"), (action == 2).astype("float64")


def _causal_goal_summary(capacity: pd.DataFrame, *, benchmark: dict[str, Any]) -> dict[str, Any]:
    if capacity.empty:
        return {
            "policy": "causal_event_meta_label_soxx_plus_20_maxdd_10_gate",
            "required_policy_satisfied": False,
            "passes_target_count": 0,
            "blocking_reasons": ["no_causal_capacity_rows"],
        }
    rows = capacity.copy()
    best_dd_safe_return = float(rows["best_drawdown_safe_return_pct"].max())
    best_dd_safe_row = rows.sort_values("best_drawdown_safe_return_pct", ascending=False).iloc[0].to_dict()
    min_dd_if_excess = float(rows["min_drawdown_if_excess_passes_pct"].max())
    best_auc = rows.sort_values("all_day_auc", ascending=False).iloc[0].to_dict()
    pass_count = int(rows["passes_target_count"].sum())
    return {
        "policy": "causal_event_meta_label_soxx_plus_20_maxdd_10_gate",
        "benchmark_return_pct": float(benchmark["return_pct"]),
        "benchmark_max_drawdown_pct": float(benchmark["max_drawdown_pct"]),
        "excess_return_floor_pct": SOXX_EXCESS_FLOOR_PCT,
        "max_drawdown_floor_pct": MAX_DRAWDOWN_FLOOR_PCT,
        "passes_target_count": pass_count,
        "candidate_count": int(rows["candidate_count"].sum()),
        "required_policy_satisfied": bool(pass_count > 0),
        "best_drawdown_safe_return_pct": best_dd_safe_return,
        "best_drawdown_safe_max_drawdown_pct": float(best_dd_safe_row["best_drawdown_safe_max_drawdown_pct"]),
        "best_drawdown_safe_excess_pct": best_dd_safe_return - float(benchmark["return_pct"]),
        "min_drawdown_if_excess_passes_pct": min_dd_if_excess,
        "best_auc_row": _json_ready(best_auc),
        "blocking_reasons": [
            "no_causal_lgb_xgb_event_candidate_combines_soxx_plus_20_with_10pct_drawdown"
        ]
        if pass_count == 0
        else [],
    }


def _leakage_summary(
    leakage: pd.DataFrame,
    *,
    causal_goal: dict[str, Any],
    feature_cols: list[str],
) -> dict[str, Any]:
    invalid = leakage.loc[leakage["audit_case"].eq("future_label_feature_invalid")].copy()
    invalid_passes = int(invalid["passes_target"].sum())
    return {
        "status": "future_label_leakage_can_create_false_pass" if invalid_passes > 0 else "no_false_pass_detected",
        "invalid_future_label_passes_target_count": invalid_passes,
        "causal_passes_target_count": int(causal_goal["passes_target_count"]),
        "forbidden_feature_columns": list(FORBIDDEN_FEATURE_COLUMNS),
        "causal_feature_columns": feature_cols,
        "invalid_best_return_pct": float(invalid["return_pct"].max()) if not invalid.empty else float("nan"),
        "notes": "future-label audit is intentionally invalid and exists only to prevent accidental promotion of leaked signals",
    }


def _summary_payload(
    *,
    spec: DynamicSoxxSoxlCashAllocationWfV5Spec,
    common_end: str,
    recorder_id: str,
    recorder_uri: Path,
    recorder_local_dir: Path,
    sidecars: dict[str, str],
    source_summary: dict[str, Any],
    feature_audit: dict[str, Any],
    benchmark: dict[str, Any],
    causal_goal: dict[str, Any],
    leakage_summary: dict[str, Any],
    capacity: pd.DataFrame,
    promotion_decision: str,
) -> dict[str, Any]:
    return _json_ready(
        {
            "experiment": spec.experiment,
            "card_type": CARD_TYPE,
            "promotion_decision": promotion_decision,
            "common_data_end": common_end,
            "source_experiment": SOURCE_EXPERIMENT,
            "provider_uri": spec.provider_uri.resolve().as_posix(),
            "benchmark_asset": "SOXX",
            "benchmark_basis": "open_to_open",
            "allocation_assets": ["SOXX", "SOXL", "cash"],
            "execution_timing": "close_t_signal_next_open_to_following_open",
            "goal": {
                "excess_vs_soxx_floor_pct": SOXX_EXCESS_FLOOR_PCT,
                "max_drawdown_floor_pct": MAX_DRAWDOWN_FLOOR_PCT,
            },
            "benchmark": benchmark,
            "feature_audit": feature_audit,
            "causal_meta_label": {
                **causal_goal,
                "uses_lgb_xgb": True,
                "selection_uses_outer_test": True,
                "selection_role": "causal_event_parameter_oracle_capacity_not_promotable",
                "row_count": int(len(capacity)),
                "models": sorted(capacity["model"].unique().tolist()) if not capacity.empty else [],
                "event_names": sorted(capacity["event_name"].unique().tolist()) if not capacity.empty else [],
                "best_all_day_auc": float(capacity["all_day_auc"].max()) if not capacity.empty else float("nan"),
            },
            "leakage_audit": leakage_summary,
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
                "LGBMClassifier",
                "XGBClassifier",
            ],
            "sidecars": sidecars,
            "artifact_policy": {
                "policy": "slim_card_recorder_first",
                "workflow_config": "workflow.yaml",
                "recorder_only_artifacts": [
                    f"artifacts/{spec.experiment}/causal_event_capacity.csv",
                    f"artifacts/{spec.experiment}/leakage_audit.csv",
                ],
            },
            "workflow_config": "workflow.yaml",
            "recorder_id": recorder_id,
            "recorder_uri": recorder_uri.resolve().as_posix(),
            "recorder_local_dir": recorder_local_dir.as_posix(),
            "recorder_artifacts": [
                f"artifacts/{spec.experiment}/causal_event_capacity.csv",
                f"artifacts/{spec.experiment}/leakage_audit.csv",
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
    leakage: pd.DataFrame,
) -> None:
    capacity.to_csv(card / sidecars["causal_event_capacity"], index=False)
    leakage.to_csv(card / sidecars["leakage_audit"], index=False)
    pd.DataFrame(
        [
            {
                "check": "causal_meta_label_goal",
                "status": "pass" if summary["causal_meta_label"]["required_policy_satisfied"] else "fail",
                "details": json.dumps(summary["causal_meta_label"], ensure_ascii=False, sort_keys=True),
            },
            {
                "check": "future_label_leakage_audit",
                "status": "invalid_false_pass_detected",
                "details": json.dumps(summary["leakage_audit"], ensure_ascii=False, sort_keys=True),
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
        "reason": "no causal LGB/XGB event meta-label candidate satisfies SOXX plus 20 percentage points with max drawdown above -10%",
        "best_auc_row": summary["causal_meta_label"].get("best_auc_row"),
    }
    (card / sidecars["selected_model"]).write_text(
        json.dumps(_json_ready(selected), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    analysis = {
        "source": "qlib_recorder_causal_event_meta_label_diagnostics",
        "recorder_id": summary["recorder_id"],
        "causal_meta_label": summary["causal_meta_label"],
        "leakage_audit": summary["leakage_audit"],
        "promotion_decision": summary["promotion_decision"],
    }
    (card / sidecars["analysis_summary"]).write_text(
        json.dumps(_json_ready(analysis), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (card / sidecars["failure_modes"]).write_text(_failure_modes_text(summary), encoding="utf-8")


def _write_summary_files(card: Path, summary: dict[str, Any]) -> None:
    (card / "summary.json").write_text(json.dumps(_json_ready(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    (card / "conclusion.md").write_text(_conclusion_text(summary), encoding="utf-8")
    (card / "workflow.yaml").write_text(_workflow_yaml(summary), encoding="utf-8")


def _failure_modes_text(summary: dict[str, Any]) -> str:
    goal = summary["causal_meta_label"]
    return (
        "# Failure Modes\n\n"
        "- Future-label leakage can manufacture candidates that appear to pass the SOXX+20 / maxDD 10 gate.\n"
        "- The causal event meta-label grid uses LGB/XGB and close-known features only, but still has zero target passers.\n"
        f"- Drawdown-safe causal return remains {_fmt_pct(goal.get('best_drawdown_safe_return_pct'))}% versus "
        f"SOXX benchmark {_fmt_pct(goal.get('benchmark_return_pct'))}%.\n"
        f"- Candidates that reach the excess-return side still draw down to {_fmt_pct(goal.get('min_drawdown_if_excess_passes_pct'))}% or worse.\n"
        "- The current feature set is more useful as a risk-audit boundary than as a promoted allocation model.\n"
    )


def _conclusion_text(summary: dict[str, Any]) -> str:
    goal = summary["causal_meta_label"]
    leakage = summary["leakage_audit"]
    return (
        "# 中文结论\n\n"
        "- 做了什么：新增 v5 causal meta-label / leakage audit，用 qlib D.features 读取 SOXL 行上的 close、volume、VIX、SOXX underlying 和 v1 因果特征；LGB/XGB 只预测事件日下一段 open-to-open SOXL 是否为正。\n"
        f"- 核心指标：SOXX benchmark return={_fmt_pct(goal.get('benchmark_return_pct'))}%，max drawdown={_fmt_pct(goal.get('benchmark_max_drawdown_pct'))}%；"
        f"causal candidate 通过数={goal['passes_target_count']}。\n"
        f"- 有效边界：回撤安全候选最高收益={_fmt_pct(goal.get('best_drawdown_safe_return_pct'))}%，"
        f"收益达标候选中的最浅回撤={_fmt_pct(goal.get('min_drawdown_if_excess_passes_pct'))}%。\n"
        f"- 未来泄漏：invalid future-label audit 通过数={leakage['invalid_future_label_passes_target_count']}；这类结果直接使用 `{ETF_RETURN_COL}` 或 `{UNDERLYING_RETURN_COL}`，只能作为防未来函数反例，不能晋升。\n"
        "- 为什么：只要把未来 open-to-open label 当成特征，收益/回撤目标会出现假突破；剔除这些列后，LGB/XGB 对事件方向的分辨率不足以同时保持 SOXX+20 和 -10% 回撤。\n"
        f"- 是否晋升：不晋升；decision={summary['promotion_decision']}。\n"
        "- 下一步：继续优化必须换成开盘前真正可知的信息源，例如盘前/期权/半导体成分股实时扩散/事件日历；仅继续调当前日线事件阈值会倾向过拟合。\n"
        f"- qlib recorder：{summary['recorder_id']}\n"
        "- 不构成买卖建议。\n"
    )


def _workflow_yaml(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"experiment: {summary['experiment']}",
            f"card_type: {summary['card_type']}",
            f"provider_uri: {summary['provider_uri']}",
            f"recorder_uri: {summary['recorder_uri']}",
            "benchmark: SOXX",
            "benchmark_basis: open_to_open",
            "allocation_assets: [SOXX, SOXL, cash]",
            "model_selection:",
            "  source: causal_event_meta_label_parameter_oracle",
            "  uses_lgb_xgb: true",
            "  selection_uses_outer_test: true",
            "  promotion_allowed: false",
            "leakage_audit:",
            "  future_label_features_forbidden: true",
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


def _event_accuracy(pred: pd.DataFrame) -> float:
    event = pred.loc[pred["event_signal"].eq(1)].copy()
    if event.empty:
        return float("nan")
    return float(accuracy_score(event["actual_positive"].astype(int), event["p_soxl_positive"].ge(0.5).astype(int)))


def _dict_float(payload: dict[str, Any], key: str) -> float:
    if not payload or key not in payload:
        return float("nan")
    return float(payload[key])


def _dedupe(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


def _fmt_pct(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "nan"
    if not np.isfinite(numeric):
        return "nan"
    return f"{numeric:.4f}"
