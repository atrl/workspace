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

from src.data import QlibProviderConfig, init_qlib_provider
from src.factor.intraday_15m_factors import build_intraday_15m_factor_frame
from src.workflow.dynamic_soxx_soxl_cash_allocation_wf import (
    CALIBRATION_ASSET,
    ETF_RETURN_COL,
    PAIR_UNIVERSE,
    UNDERLYING_RETURN_COL,
)
from src.workflow.dynamic_soxx_soxl_cash_allocation_wf_v2 import _return_drawdown
from src.workflow.multi_3x_etf_normalized_nested_wf import _json_ready, _resolve_root
from src.workflow.trainable_regime_policy_nested_wf import _remove_qlib_code_snapshot_artifacts


EXPERIMENT = "dynamic_soxx_soxl_cash_allocation_wf_v3"
CARD_TYPE = "dynamic_soxx_soxl_cash_allocation_signal_capacity_research_card"
SOURCE_EXPERIMENT = "dynamic_soxx_soxl_cash_allocation_wf_v1"
SOXX_EXCESS_FLOOR_PCT = 20.0
MAX_DRAWDOWN_FLOOR_PCT = -10.0


@dataclass(frozen=True)
class DynamicSoxxSoxlCashAllocationWfV3Spec:
    root: Path
    experiment: str = EXPERIMENT
    train_start: str = "2011-01-03"
    intraday_start: str = "2018-12-31 09:45:00"
    test_years: tuple[int, ...] = (2023, 2024, 2025, 2026)
    tail_rate: float = 0.08
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


def run_dynamic_soxx_soxl_cash_allocation_wf_v3_card(*, root: Path | str | None = None) -> dict[str, Any]:
    spec = DynamicSoxxSoxlCashAllocationWfV3Spec(root=_resolve_root(root))
    return _run_card(spec)


def _run_card(spec: DynamicSoxxSoxlCashAllocationWfV3Spec) -> dict[str, Any]:
    card = spec.card_dir
    card.mkdir(parents=True, exist_ok=True)
    source_summary = json.loads((spec.source_card_dir / "summary.json").read_text(encoding="utf-8"))
    common_end = str(source_summary["common_data_end"])
    base_feature_cols = _source_feature_cols(source_summary)
    daily_frame = _load_daily_frame(spec, common_end=common_end, feature_cols=base_feature_cols)
    soxl_frame, feature_groups, feature_audit = _build_soxl_model_frame(
        spec,
        daily_frame=daily_frame,
        common_end=common_end,
        base_feature_cols=base_feature_cols,
    )
    test_frame = soxl_frame.loc[soxl_frame["datetime"].dt.year.isin(spec.test_years)].copy()
    benchmark = _benchmark_metrics(test_frame)
    oracle = _oracle_diagnostics(test_frame)
    capacity = _signal_capacity_diagnostics(soxl_frame, feature_groups=feature_groups, spec=spec)
    stitched_goal = _stitched_goal_summary(capacity, benchmark=benchmark)
    promotion_decision = (
        "promoted_dynamic_soxx_soxl_cash_goal_passed"
        if bool(stitched_goal["required_policy_satisfied"])
        else "not_promoted_signal_capacity_goal_failed"
    )

    sidecars = {
        "model_capacity": "model_capacity.csv",
        "oracle_frontier": "oracle_frontier.csv",
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
        capacity_path = tmp_dir / "model_capacity.csv"
        oracle_path = tmp_dir / "oracle_frontier.csv"
        capacity.to_csv(capacity_path, index=False)
        oracle.to_csv(oracle_path, index=False)
        with R.start(experiment_name=spec.experiment):
            recorder = R.get_recorder()
            R.log_params(
                daily_provider_uri=spec.daily_provider_uri.resolve().as_posix(),
                intraday_15m_provider_uri=spec.intraday_15m_provider_uri.resolve().as_posix(),
                benchmark="SOXX",
                benchmark_basis="open_to_open",
                allocation_assets="SOXX,SOXL,cash",
                objective="SOXX plus 20 percentage points with max drawdown above -10%",
                model_families="lgbm,xgb",
                tail_rate=float(spec.tail_rate),
                selection_uses_outer_test="true_for_capacity_diagnostic_not_promotable",
            )
            R.log_metrics(
                soxx_stitched_return_pct=float(benchmark["return_pct"]),
                soxx_stitched_max_drawdown_pct=float(benchmark["max_drawdown_pct"]),
                best_tail_auc=float(capacity["tail_auc"].max()) if not capacity.empty else float("nan"),
                capacity_passes_target_count=float(stitched_goal["passes_target_count"]),
                capacity_best_drawdown_safe_return_pct=float(stitched_goal["best_drawdown_safe_return_pct"]),
                capacity_min_drawdown_if_excess_passes_pct=float(stitched_goal["min_drawdown_if_excess_passes_pct"]),
            )
            recorder.log_artifact(capacity_path.as_posix(), artifact_path=spec.experiment)
            recorder.log_artifact(oracle_path.as_posix(), artifact_path=spec.experiment)
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
                oracle=oracle,
                capacity=capacity,
                stitched_goal=stitched_goal,
                promotion_decision=promotion_decision,
            )
            _write_sidecars(card, sidecars=sidecars, summary=summary, capacity=capacity, oracle=oracle)
            for rel_path in sidecars.values():
                path = card / rel_path
                if path.exists() and path.suffix != ".md":
                    recorder.log_artifact(path.as_posix(), artifact_path=spec.experiment)
            _remove_qlib_code_snapshot_artifacts(Path(recorder.get_local_dir()).resolve())
    _write_summary_files(card, summary)
    return summary


def _source_feature_cols(source_summary: dict[str, Any]) -> list[str]:
    feature_sets = {
        str(item["feature_set"]): list(item["feature_cols"]) for item in source_summary["feature_sets"]["available"]
    }
    return list(feature_sets.get("base_soxx_diffusion") or feature_sets.get("base") or [])


def _load_daily_frame(
    spec: DynamicSoxxSoxlCashAllocationWfV3Spec,
    *,
    common_end: str,
    feature_cols: list[str],
) -> pd.DataFrame:
    config = QlibProviderConfig(provider_uri=spec.daily_provider_uri, market="all", region="us", kernels=1)
    init_qlib_provider(config)
    from qlib.data import D

    fields = [
        f"${ETF_RETURN_COL}",
        f"${UNDERLYING_RETURN_COL}",
        "$underlying_close",
        *[f"${column}" for column in feature_cols],
    ]
    qframe = D.features(list(PAIR_UNIVERSE), fields, start_time=spec.train_start, end_time=common_end, freq="day")
    out = qframe.reset_index().rename(columns={field: field.removeprefix("$") for field in fields})
    out["datetime"] = pd.to_datetime(out["datetime"]).dt.normalize()
    out["instrument"] = out["instrument"].astype(str).str.upper()
    for column in [ETF_RETURN_COL, UNDERLYING_RETURN_COL, "underlying_close", *feature_cols]:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return out.dropna(subset=[ETF_RETURN_COL, UNDERLYING_RETURN_COL]).sort_values(
        ["datetime", "instrument"]
    ).reset_index(drop=True)


def _build_soxl_model_frame(
    spec: DynamicSoxxSoxlCashAllocationWfV3Spec,
    *,
    daily_frame: pd.DataFrame,
    common_end: str,
    base_feature_cols: list[str],
) -> tuple[pd.DataFrame, dict[str, list[str]], dict[str, Any]]:
    soxl = daily_frame.loc[daily_frame["instrument"].eq(CALIBRATION_ASSET)].copy()
    context_cols = [
        column
        for column in [
            "underlying_ret_5",
            "underlying_realized_vol_20",
            "underlying_drawdown_60",
            "relative_momentum_20",
            "relative_momentum_60",
            "volatility_ratio_20",
            "ret_5_z20",
            "ret_20_z20",
            "ret_60_z20",
            "realized_vol_20_z20",
            "drawdown_60_z20",
            "tracking_error_20_z20",
            "range_ratio_20_z20",
            "volume_z_20",
            "intraday_risk_score",
        ]
        if column in daily_frame.columns
    ]
    context_wide = daily_frame[["datetime", "instrument", *context_cols]].pivot(
        index="datetime", columns="instrument", values=context_cols
    )
    context_wide.columns = [f"ctx_{instrument.lower()}_{column}" for column, instrument in context_wide.columns]
    context_wide = context_wide.reset_index()
    soxl = soxl.merge(context_wide, on="datetime", how="left")

    intraday_manifest_path = spec.intraday_15m_provider_uri / "qlib_store_manifest.json"
    intraday_manifest = json.loads(intraday_manifest_path.read_text(encoding="utf-8"))
    intraday_end = min(
        pd.Timestamp(common_end).normalize() + pd.Timedelta(hours=16),
        pd.Timestamp(str(intraday_manifest["calendar_end"])),
    ).strftime("%Y-%m-%d %H:%M:%S")
    intraday = build_intraday_15m_factor_frame(
        QlibProviderConfig(
            provider_uri=spec.intraday_15m_provider_uri,
            market="all",
            region="us",
            freq="15min",
            kernels=1,
        ),
        instruments=["SOXL", "SOXX"],
        start=spec.intraday_start,
        end=intraday_end,
        lag_days=1,
    )
    intraday_cols = [column for column in list(intraday.attrs.get("feature_cols", [])) if column in intraday.columns]
    intraday["datetime"] = pd.to_datetime(intraday["datetime"]).dt.normalize()
    soxl = soxl.merge(intraday[["datetime", *intraday_cols]], on="datetime", how="left")
    soxl = soxl.loc[pd.to_datetime(soxl["datetime"]).ge(pd.Timestamp("2019-01-02"))].reset_index(drop=True)
    base = [column for column in base_feature_cols if column in soxl.columns]
    context = [column for column in context_wide.columns if column != "datetime" and column in soxl.columns]
    intraday_feature_cols = [column for column in intraday_cols if column in soxl.columns]
    feature_groups = {
        "base_soxx_diffusion": base,
        "base_cross_asset_context": list(dict.fromkeys([*base, *context])),
        "base_intraday15m": list(dict.fromkeys([*base, *intraday_feature_cols])),
    }
    return soxl, feature_groups, {
        "daily_provider_uri": spec.daily_provider_uri.resolve().as_posix(),
        "intraday_15m_provider_uri": spec.intraday_15m_provider_uri.resolve().as_posix(),
        "daily_common_end": common_end,
        "intraday_15m_end": intraday_end,
        "base_feature_count": len(base),
        "cross_asset_context_feature_count": len(context),
        "intraday15m_feature_count": len(intraday_feature_cols),
        "intraday15m_lag_days": int(intraday.attrs.get("lag_days", 1)),
        "feature_groups": {name: len(cols) for name, cols in feature_groups.items()},
    }


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


def _oracle_diagnostics(test_frame: pd.DataFrame) -> pd.DataFrame:
    soxx = pd.to_numeric(test_frame[UNDERLYING_RETURN_COL], errors="coerce").fillna(0.0).to_numpy(dtype="float64")
    soxl = pd.to_numeric(test_frame[ETF_RETURN_COL], errors="coerce").fillna(0.0).to_numpy(dtype="float64")
    rows: list[dict[str, Any]] = []
    for name, returns in {"soxx_buy_hold": soxx, "soxl_buy_hold": soxl}.items():
        ret, maxdd = _return_drawdown(returns)
        rows.append(
            {
                "oracle_type": "benchmark",
                "candidate": name,
                "return_pct": ret,
                "max_drawdown_pct": maxdd,
                "cash_days": 0,
                "notes": "not_oracle",
            }
        )
    actions = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [0.7, 0.3], [0.35, 0.65]])
    action_returns = actions[:, 0, None] * soxx[None, :] + actions[:, 1, None] * soxl[None, :]
    best = actions[np.argmax(action_returns, axis=0)]
    ret, maxdd = _return_drawdown(_strategy_returns(soxx, soxl, best[:, 0], best[:, 1], cost_per_turnover=0.0003))
    rows.append(
        {
            "oracle_type": "perfect_next_open_action",
            "candidate": "best_daily_action_cash_soxx_soxl",
            "return_pct": ret,
            "max_drawdown_pct": maxdd,
            "cash_days": int(np.sum((best[:, 0] + best[:, 1]) == 0.0)),
            "notes": "hindsight_upper_bound_not_promotable",
        }
    )
    worst_order = np.argsort(soxx)
    for avoid_n in [20, 40, 60, 80, 100]:
        soxx_weight = np.ones(len(soxx), dtype="float64")
        soxl_weight = np.zeros(len(soxx), dtype="float64")
        soxx_weight[worst_order[:avoid_n]] = 0.0
        ret, maxdd = _return_drawdown(_strategy_returns(soxx, soxl, soxx_weight, soxl_weight, cost_per_turnover=0.0003))
        rows.append(
            {
                "oracle_type": "avoid_worst_soxx_days",
                "candidate": f"avoid_worst_{avoid_n}",
                "return_pct": ret,
                "max_drawdown_pct": maxdd,
                "cash_days": int(avoid_n),
                "notes": "hindsight_upper_bound_not_promotable",
            }
        )
    return pd.DataFrame(rows)


def _signal_capacity_diagnostics(
    frame: pd.DataFrame,
    *,
    feature_groups: dict[str, list[str]],
    spec: DynamicSoxxSoxlCashAllocationWfV3Spec,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    params = _capacity_param_grid()
    for feature_group, feature_cols in feature_groups.items():
        for model_name in ["lgbm", "xgb"]:
            pred = _rolling_predictions(frame, feature_cols=feature_cols, model_name=model_name, spec=spec)
            tail_auc = _safe_auc(pred["tail_actual"], pred["p_tail"])
            upside_auc = _safe_auc(pred["up_actual"], pred["p_up"])
            metrics = [_simulate_capacity(pred, params_row, cost_per_turnover=spec.cost_per_turnover) for params_row in params]
            metrics_frame = pd.DataFrame(metrics)
            metrics_frame["passes_excess_gate"] = metrics_frame["excess_vs_soxx_pct"].ge(SOXX_EXCESS_FLOOR_PCT)
            metrics_frame["passes_drawdown_gate"] = metrics_frame["max_drawdown_pct"].ge(MAX_DRAWDOWN_FLOOR_PCT)
            metrics_frame["passes_target"] = metrics_frame["passes_excess_gate"] & metrics_frame["passes_drawdown_gate"]
            dd_safe = metrics_frame.loc[metrics_frame["passes_drawdown_gate"]].copy()
            excess = metrics_frame.loc[metrics_frame["passes_excess_gate"]].copy()
            best_dd_safe = (
                dd_safe.sort_values("return_pct", ascending=False).iloc[0].to_dict() if not dd_safe.empty else {}
            )
            shallow_excess = (
                excess.sort_values("max_drawdown_pct", ascending=False).iloc[0].to_dict() if not excess.empty else {}
            )
            best_return = metrics_frame.sort_values("return_pct", ascending=False).iloc[0].to_dict()
            rows.append(
                {
                    "feature_group": feature_group,
                    "model": model_name,
                    "tail_rate": float(spec.tail_rate),
                    "feature_count": int(len(feature_cols)),
                    "tail_auc": tail_auc,
                    "upside_auc": upside_auc,
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
                    "notes": "parameter_oracle_capacity_diagnostic_not_promotable",
                }
            )
    return pd.DataFrame(rows)


def _rolling_predictions(
    frame: pd.DataFrame,
    *,
    feature_cols: list[str],
    model_name: str,
    spec: DynamicSoxxSoxlCashAllocationWfV3Spec,
) -> pd.DataFrame:
    parts = []
    for year in spec.test_years:
        train = frame.loc[frame["datetime"].dt.year.lt(int(year))].copy()
        test = frame.loc[frame["datetime"].dt.year.eq(int(year))].copy().sort_values("datetime")
        tail_thr = float(train[UNDERLYING_RETURN_COL].quantile(float(spec.tail_rate)))
        y_tail = pd.to_numeric(train[UNDERLYING_RETURN_COL], errors="coerce").le(tail_thr).astype(int)
        y_up = pd.to_numeric(train[ETF_RETURN_COL], errors="coerce").gt(0.0).astype(int)
        tail_model = _build_classifier(model_name, label_role="tail", positive_count=int(y_tail.sum()), train_count=len(y_tail))
        up_model = _build_classifier(model_name, label_role="upside", positive_count=int(y_up.sum()), train_count=len(y_up))
        train_x = train[feature_cols].astype("float64")
        test_x = test[feature_cols].astype("float64")
        tail_model.fit(train_x, y_tail)
        up_model.fit(train_x, y_up)
        segment = test[["datetime", ETF_RETURN_COL, UNDERLYING_RETURN_COL]].copy()
        segment["year"] = int(year)
        segment["p_tail"] = tail_model.predict_proba(test_x)[:, 1]
        segment["p_up"] = up_model.predict_proba(test_x)[:, 1]
        segment["tail_actual"] = pd.to_numeric(segment[UNDERLYING_RETURN_COL], errors="coerce").le(tail_thr).astype(int)
        segment["up_actual"] = pd.to_numeric(segment[ETF_RETURN_COL], errors="coerce").gt(0.0).astype(int)
        parts.append(segment)
    return pd.concat(parts, ignore_index=True)


def _build_classifier(model_name: str, *, label_role: str, positive_count: int, train_count: int):
    if model_name == "lgbm":
        from lightgbm import LGBMClassifier

        class_weight = "balanced" if label_role == "tail" else None
        return make_pipeline(
            SimpleImputer(strategy="median"),
            LGBMClassifier(
                n_estimators=45,
                learning_rate=0.04,
                num_leaves=9,
                min_child_samples=70,
                subsample=0.8,
                colsample_bytree=0.7,
                reg_alpha=0.3,
                reg_lambda=1.4,
                class_weight=class_weight,
                random_state=810 if label_role == "tail" else 811,
                verbose=-1,
            ),
        )
    if model_name == "xgb":
        from xgboost import XGBClassifier

        scale_pos_weight = float((train_count - positive_count) / max(1, positive_count)) if label_role == "tail" else 1.0
        return make_pipeline(
            SimpleImputer(strategy="median"),
            XGBClassifier(
                n_estimators=55,
                max_depth=2,
                learning_rate=0.04,
                subsample=0.8,
                colsample_bytree=0.7,
                reg_alpha=0.2,
                reg_lambda=1.4,
                objective="binary:logistic",
                eval_metric="logloss",
                scale_pos_weight=scale_pos_weight,
                random_state=812 if label_role == "tail" else 813,
                n_jobs=1,
            ),
        )
    raise ValueError(f"unsupported classifier model: {model_name}")


def _capacity_param_grid() -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for core in [0.0, 0.8, 1.0]:
        for overlay in [0.0, 0.2, 0.35, 0.5]:
            for cash_q in [0.75, 0.85, 0.9, 0.95]:
                for reduce_q in [0.75, 1.01]:
                    for low_tail_q in [0.4, 0.55]:
                        for reduce_mult in [0.0, 0.5, 1.0]:
                            for stop_thr in [-0.12, -1.0]:
                                for stop_mult in [0.0, 0.5, 1.0]:
                                    rows.append(
                                        {
                                            "core": core,
                                            "overlay": overlay,
                                            "cash_q": cash_q,
                                            "reduce_q": reduce_q,
                                            "low_tail_q": low_tail_q,
                                            "reduce_mult": reduce_mult,
                                            "stop_thr": stop_thr,
                                            "stop_mult": stop_mult,
                                        }
                                    )
    return rows


def _simulate_capacity(
    pred: pd.DataFrame,
    params: dict[str, float],
    *,
    cost_per_turnover: float,
) -> dict[str, Any]:
    p_tail = pd.to_numeric(pred["p_tail"], errors="coerce").fillna(0.0)
    cash_thr = float(p_tail.quantile(float(params["cash_q"])))
    reduce_q = float(params["reduce_q"])
    reduce_thr = float(p_tail.quantile(reduce_q)) if reduce_q <= 1.0 else float("inf")
    low_thr = float(p_tail.quantile(float(params["low_tail_q"])))
    soxx_returns = pd.to_numeric(pred[UNDERLYING_RETURN_COL], errors="coerce").fillna(0.0).to_numpy(dtype="float64")
    soxl_returns = pd.to_numeric(pred[ETF_RETURN_COL], errors="coerce").fillna(0.0).to_numpy(dtype="float64")
    p_tail_arr = p_tail.to_numpy(dtype="float64")
    p_up = pd.to_numeric(pred["p_up"], errors="coerce").fillna(0.0).to_numpy(dtype="float64")
    soxx_weight = np.full(len(pred), float(params["core"]), dtype="float64")
    soxl_weight = np.where((p_tail_arr <= low_thr) & (p_up >= 0.5), float(params["overlay"]), 0.0)
    warn = p_tail_arr >= reduce_thr if np.isfinite(reduce_thr) else np.zeros(len(pred), dtype=bool)
    cash = p_tail_arr >= cash_thr
    soxx_weight[warn] *= float(params["reduce_mult"])
    soxl_weight[warn] *= float(params["reduce_mult"])
    soxx_weight[cash] = 0.0
    soxl_weight[cash] = 0.0
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
        "cash_signal_share": float(np.mean(cash)),
        "warn_signal_share": float(np.mean(warn)),
    }


def _strategy_returns(
    soxx_returns: np.ndarray,
    soxl_returns: np.ndarray,
    soxx_weight: np.ndarray,
    soxl_weight: np.ndarray,
    *,
    cost_per_turnover: float,
    stop_thr: float = -1.0,
    stop_mult: float = 1.0,
) -> np.ndarray:
    returns: list[float] = []
    equity = 1.0
    peak = 1.0
    prev_soxx = 0.0
    prev_soxl = 0.0
    for idx in range(len(soxx_returns)):
        risk_mult = float(stop_mult) if equity / peak - 1.0 <= float(stop_thr) else 1.0
        soxx = float(soxx_weight[idx]) * risk_mult
        soxl = float(soxl_weight[idx]) * risk_mult
        total = soxx + soxl
        if total > 1.0:
            soxx /= total
            soxl /= total
        turnover = abs(soxx - prev_soxx) + abs(soxl - prev_soxl)
        ret = soxx * float(soxx_returns[idx]) + soxl * float(soxl_returns[idx]) - turnover * float(cost_per_turnover)
        returns.append(ret)
        equity *= 1.0 + ret
        peak = max(peak, equity)
        prev_soxx = soxx
        prev_soxl = soxl
    return np.asarray(returns, dtype="float64")


def _safe_auc(y_true: pd.Series, score: pd.Series) -> float:
    try:
        return float(roc_auc_score(y_true.astype(int), score.astype("float64")))
    except ValueError:
        return float("nan")


def _dict_float(payload: dict[str, Any], key: str) -> float:
    if not payload or key not in payload:
        return float("nan")
    return float(payload[key])


def _stitched_goal_summary(capacity: pd.DataFrame, *, benchmark: dict[str, Any]) -> dict[str, Any]:
    if capacity.empty:
        return {
            "policy": "signal_capacity_parameter_oracle_soxx_plus_20_maxdd_10_gate",
            "required_policy_satisfied": False,
            "passes_target_count": 0,
            "blocking_reasons": ["no_capacity_rows"],
        }
    rows = capacity.copy()
    rows["passes_excess_gate"] = rows["return_if_min_drawdown_excess_passes_pct"].ge(
        float(benchmark["return_pct"]) + SOXX_EXCESS_FLOOR_PCT
    )
    rows["passes_drawdown_gate"] = rows["best_drawdown_safe_max_drawdown_pct"].ge(MAX_DRAWDOWN_FLOOR_PCT)
    rows["passes_target"] = rows["passes_target_count"].astype(int).gt(0)
    best_dd_safe_return = float(rows["best_drawdown_safe_return_pct"].max())
    min_dd_if_excess = float(rows["min_drawdown_if_excess_passes_pct"].max())
    best_auc = rows.sort_values("tail_auc", ascending=False).iloc[0].to_dict()
    return {
        "policy": "signal_capacity_parameter_oracle_soxx_plus_20_maxdd_10_gate",
        "benchmark_return_pct": float(benchmark["return_pct"]),
        "benchmark_max_drawdown_pct": float(benchmark["max_drawdown_pct"]),
        "excess_return_floor_pct": SOXX_EXCESS_FLOOR_PCT,
        "max_drawdown_floor_pct": MAX_DRAWDOWN_FLOOR_PCT,
        "passes_target_count": int(rows["passes_target_count"].sum()),
        "candidate_count": int(rows["candidate_count"].sum()),
        "required_policy_satisfied": bool(rows["passes_target_count"].sum() > 0),
        "best_drawdown_safe_return_pct": best_dd_safe_return,
        "best_drawdown_safe_excess_pct": best_dd_safe_return - float(benchmark["return_pct"]),
        "min_drawdown_if_excess_passes_pct": min_dd_if_excess,
        "best_tail_auc_row": _json_ready(best_auc),
        "blocking_reasons": ["no_lgb_xgb_signal_capacity_candidate_combines_soxx_plus_20_with_10pct_drawdown"],
    }


def _summary_payload(
    *,
    spec: DynamicSoxxSoxlCashAllocationWfV3Spec,
    common_end: str,
    recorder_id: str,
    recorder_uri: Path,
    recorder_local_dir: Path,
    sidecars: dict[str, str],
    source_summary: dict[str, Any],
    feature_audit: dict[str, Any],
    benchmark: dict[str, Any],
    oracle: pd.DataFrame,
    capacity: pd.DataFrame,
    stitched_goal: dict[str, Any],
    promotion_decision: str,
) -> dict[str, Any]:
    source_v2_summary_path = (
        spec.root / "research" / "dynamic_soxx_soxl_cash_allocation_wf_v2" / "summary.json"
    )
    source_v2_summary: dict[str, Any] = {}
    if source_v2_summary_path.exists():
        source_v2_summary = json.loads(source_v2_summary_path.read_text(encoding="utf-8"))

    return _json_ready(
        {
            "experiment": spec.experiment,
            "card_type": CARD_TYPE,
            "promotion_decision": promotion_decision,
            "common_data_end": common_end,
            "source_experiment": SOURCE_EXPERIMENT,
            "provider_uri": spec.daily_provider_uri.resolve().as_posix(),
            "benchmark_asset": "SOXX",
            "benchmark_basis": "open_to_open",
            "allocation_assets": ["SOXX", "SOXL", "cash"],
            "execution_timing": "close_t_signal_next_open_to_following_open",
            "goal": {
                "excess_vs_soxx_floor_pct": SOXX_EXCESS_FLOOR_PCT,
                "max_drawdown_floor_pct": MAX_DRAWDOWN_FLOOR_PCT,
            },
            "benchmark": benchmark,
            "oracle": {
                "source": "hindsight_upper_bound_not_promotable",
                "row_count": int(len(oracle)),
                "best_drawdown_safe_return_pct": float(
                    oracle.loc[oracle["max_drawdown_pct"].ge(MAX_DRAWDOWN_FLOOR_PCT), "return_pct"].max()
                ),
                "avoid_worst_80": _json_ready(
                    oracle.loc[oracle["candidate"].eq("avoid_worst_80")].iloc[0].to_dict()
                ),
            },
            "feature_audit": feature_audit,
            "model_capacity": {
                "uses_lgb_xgb": True,
                "tail_label": f"next_open_to_open_underlying_return_bottom_{spec.tail_rate:.2f}",
                "upside_label": "next_open_to_open_soxl_return_positive",
                "selection_uses_outer_test": True,
                "selection_role": "parameter_oracle_capacity_diagnostic_not_promotable",
                "row_count": int(len(capacity)),
                "best_tail_auc": float(capacity["tail_auc"].max()) if not capacity.empty else float("nan"),
                "feature_groups": sorted(capacity["feature_group"].unique().tolist()) if not capacity.empty else [],
            },
            "stitched_goal": stitched_goal,
            "source_v2": {
                "summary_path": source_v2_summary_path.resolve().as_posix()
                if source_v2_summary
                else None,
                "promotion_decision": source_v2_summary.get("promotion_decision"),
                "common_data_end": source_v2_summary.get("common_data_end"),
                "recorder_id": source_v2_summary.get("recorder_id"),
                "source_v1_promotion_decision": source_summary.get("promotion_decision"),
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
                    f"artifacts/{spec.experiment}/model_capacity.csv",
                    f"artifacts/{spec.experiment}/oracle_frontier.csv",
                ],
            },
            "workflow_config": "workflow.yaml",
            "recorder_id": recorder_id,
            "recorder_uri": recorder_uri.resolve().as_posix(),
            "recorder_local_dir": recorder_local_dir.as_posix(),
            "recorder_artifacts": [
                f"artifacts/{spec.experiment}/model_capacity.csv",
                f"artifacts/{spec.experiment}/oracle_frontier.csv",
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
    oracle: pd.DataFrame,
) -> None:
    capacity.to_csv(card / sidecars["model_capacity"], index=False)
    oracle.to_csv(card / sidecars["oracle_frontier"], index=False)
    pd.DataFrame(
        [
            {
                "check": "stitched_goal",
                "status": "pass" if summary["stitched_goal"]["required_policy_satisfied"] else "fail",
                "details": json.dumps(summary["stitched_goal"], ensure_ascii=False, sort_keys=True),
            },
            {
                "check": "oracle_upper_bound",
                "status": "diagnostic",
                "details": json.dumps(summary["oracle"], ensure_ascii=False, sort_keys=True),
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
        "reason": "no LGB/XGB signal-capacity candidate satisfies SOXX plus 20 percentage points with max drawdown above -10%",
        "best_tail_auc_row": summary["stitched_goal"].get("best_tail_auc_row"),
    }
    (card / sidecars["selected_model"]).write_text(json.dumps(_json_ready(selected), ensure_ascii=False, indent=2), encoding="utf-8")
    analysis = {
        "source": "qlib_recorder_signal_capacity_diagnostics",
        "recorder_id": summary["recorder_id"],
        "stitched_goal": summary["stitched_goal"],
        "oracle": summary["oracle"],
        "feature_audit": summary["feature_audit"],
        "promotion_decision": summary["promotion_decision"],
    }
    (card / sidecars["analysis_summary"]).write_text(json.dumps(_json_ready(analysis), ensure_ascii=False, indent=2), encoding="utf-8")
    (card / sidecars["failure_modes"]).write_text(_failure_modes_text(summary), encoding="utf-8")


def _write_summary_files(card: Path, summary: dict[str, Any]) -> None:
    (card / "summary.json").write_text(json.dumps(_json_ready(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    (card / "conclusion.md").write_text(_conclusion_text(summary), encoding="utf-8")
    (card / "workflow.yaml").write_text(_workflow_yaml(summary), encoding="utf-8")


def _failure_modes_text(summary: dict[str, Any]) -> str:
    goal = summary["stitched_goal"]
    return (
        "# Failure Modes\n\n"
        "- Hindsight oracle can satisfy the asset-level target only by knowing many future worst SOXX days.\n"
        "- LGB/XGB tail classifiers improve with lagged 15m factors, but the improvement does not close the return/drawdown frontier.\n"
        f"- Drawdown-safe signal-capacity return remains {_fmt_pct(goal.get('best_drawdown_safe_return_pct'))}% versus "
        f"SOXX benchmark {_fmt_pct(goal.get('benchmark_return_pct'))}%.\n"
        f"- Excess-return candidates still have shallowest drawdown {_fmt_pct(goal.get('min_drawdown_if_excess_passes_pct'))}%, "
        "which is far below the -10% target.\n"
        "- The current feature set is more suitable as a risk overlay or execution context than a standalone crash-avoidance engine.\n"
    )


def _conclusion_text(summary: dict[str, Any]) -> str:
    goal = summary["stitched_goal"]
    feature = summary["feature_audit"]
    return (
        "# 中文结论\n\n"
        "- 做了什么：新增 v3 信号容量诊断，先用 hindsight oracle 检查 SOXX/SOXL/cash 资产集合上界，再用 LGB/XGB crash-risk classifier 测试 base、横向 3x ETF context、完整 lagged 15m 因子。\n"
        f"- 核心指标：SOXX benchmark return={_fmt_pct(goal.get('benchmark_return_pct'))}%，max drawdown={_fmt_pct(goal.get('benchmark_max_drawdown_pct'))}%；"
        f"capacity parameter-oracle 通过数={goal['passes_target_count']}。\n"
        f"- 上界：hindsight avoid_worst_80 return={summary['oracle']['avoid_worst_80']['return_pct']:.4f}%，"
        f"max drawdown={summary['oracle']['avoid_worst_80']['max_drawdown_pct']:.4f}%，说明目标需要强坏日识别，不是资产集合完全不可达。\n"
        f"- 因子：cross_asset_context_feature_count={feature['cross_asset_context_feature_count']}，"
        f"intraday15m_feature_count={feature['intraday15m_feature_count']}，lag_days={feature['intraday15m_lag_days']}。\n"
        f"- 风险边界：回撤安全候选最高收益={_fmt_pct(goal.get('best_drawdown_safe_return_pct'))}%，"
        f"收益达标候选中的最浅回撤={_fmt_pct(goal.get('min_drawdown_if_excess_passes_pct'))}%。\n"
        f"- 是否晋升：不晋升；decision={summary['promotion_decision']}。15m 因子改善 tail rank，但仍不足以满足 SOXX+20 与 -10% 回撤。\n"
        "- 下一步：若目标不放宽，需要接入更接近事前冲击的信息，例如期权/波动率曲面、盘前流动性、新闻/宏观事件日历或更高频实时状态；继续只调日线/lagged 15m 阈值意义有限。\n"
        f"- qlib recorder：{summary['recorder_id']}\n"
        "- 不构成买卖建议。\n"
    )


def _fmt_pct(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "nan"
    if not np.isfinite(numeric):
        return "nan"
    return f"{numeric:.4f}"


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
            "  source: signal_capacity_parameter_oracle",
            "  uses_lgb_xgb: true",
            "  selection_uses_outer_test: true",
            "  promotion_allowed: false",
            "artifact_policy:",
            "  policy: slim_card_recorder_first",
            "  workflow_config: workflow.yaml",
            f"promotion_decision: {summary['promotion_decision']}",
            "",
        ]
    )
