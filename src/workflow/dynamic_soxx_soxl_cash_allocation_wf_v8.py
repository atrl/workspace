from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

from src.data import QlibProviderConfig, load_qlib_feature_frame
from src.workflow.dynamic_soxx_soxl_cash_allocation_wf import ETF_RETURN_COL, UNDERLYING_RETURN_COL
from src.workflow.dynamic_soxx_soxl_cash_allocation_wf_v2 import _return_drawdown
from src.workflow.dynamic_soxx_soxl_cash_allocation_wf_v6 import (
    DynamicSoxxSoxlCashAllocationWfV6Spec,
    _benchmark_metrics,
    _load_model_frame,
    _positive_proba,
    _source_causal_feature_cols,
)
from src.workflow.multi_3x_etf_normalized_nested_wf import _json_ready, _resolve_root
from src.workflow.trainable_regime_policy_nested_wf import _remove_qlib_code_snapshot_artifacts


EXPERIMENT = "dynamic_soxx_soxl_cash_allocation_wf_v8"
CARD_TYPE = "dynamic_soxx_soxl_cash_intraday_cppi_ml_risk_budget_frontier_card"
SOURCE_EXPERIMENT = "dynamic_soxx_soxl_cash_allocation_wf_v1"
SOXX_EXCESS_FLOOR_PCT = 20.0
MAX_DRAWDOWN_FLOOR_PCT = -10.0
MODEL_FAMILIES = ("xgb", "lgbm", "avg_xgb_lgb")


@dataclass(frozen=True)
class DynamicSoxxSoxlCashAllocationWfV8Spec:
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
    def v6_spec(self) -> DynamicSoxxSoxlCashAllocationWfV6Spec:
        return DynamicSoxxSoxlCashAllocationWfV6Spec(root=self.root)

    @property
    def daily_provider_uri(self) -> Path:
        return self.v6_spec.daily_provider_uri

    @property
    def intraday_15m_provider_uri(self) -> Path:
        return self.v6_spec.intraday_15m_provider_uri


def run_dynamic_soxx_soxl_cash_allocation_wf_v8_card(*, root: Path | str | None = None) -> dict[str, Any]:
    spec = DynamicSoxxSoxlCashAllocationWfV8Spec(root=_resolve_root(root))
    return _run_card(spec)


def _run_card(spec: DynamicSoxxSoxlCashAllocationWfV8Spec) -> dict[str, Any]:
    card = spec.card_dir
    card.mkdir(parents=True, exist_ok=True)
    source_summary = json.loads((spec.source_card_dir / "summary.json").read_text(encoding="utf-8"))
    common_end = str(source_summary["common_data_end"])
    source_feature_cols = _source_causal_feature_cols(source_summary)
    frame, feature_cols, feature_audit = _load_model_frame(
        spec.v6_spec,
        common_end=common_end,
        source_feature_cols=source_feature_cols,
    )
    intraday_low, intraday_audit = _load_next_session_intraday_low(spec, frame=frame, common_end=common_end)
    test_frame = frame.loc[frame["datetime"].dt.year.isin(spec.test_years)].copy()
    benchmark = _benchmark_metrics(test_frame)
    capacity = _capacity_frontier(frame, intraday_low=intraday_low, spec=spec, benchmark=benchmark)
    capacity_summary = _capacity_summary(capacity)
    predictions_by_model = _build_predictions_by_model(frame, feature_cols=feature_cols, spec=spec)
    ml_grid, model_comparison, ml_summary = _ml_risk_budget_selector(
        predictions_by_model,
        frame=frame,
        intraday_low=intraday_low,
        spec=spec,
        benchmark=benchmark,
    )
    promotion_decision = (
        "promoted_intraday_cppi_ml_risk_budget_goal_passed"
        if ml_summary["required_policy_satisfied"]
        else "not_promoted_intraday_cppi_ml_risk_budget_goal_failed"
    )
    sidecars = {
        "analysis_summary": "analysis_summary.json",
        "failure_modes": "failure_modes.md",
        "frontier_summary": "frontier_summary.csv",
        "model_comparison": "model_comparison.csv",
        "risk_analysis": "risk_analysis.csv",
        "selected_model": "selected_model.json",
    }

    import qlib
    from qlib.workflow import R

    _ = qlib
    recorder_uri = card / "qlib_recorder"
    recorder_uri.mkdir(parents=True, exist_ok=True)
    R.set_uri(recorder_uri.resolve().as_posix())
    with tempfile.TemporaryDirectory(prefix=f"{spec.experiment}_") as tmp:
        tmp_dir = Path(tmp)
        capacity_path = tmp_dir / "capacity_frontier.csv"
        ml_grid_path = tmp_dir / "ml_risk_budget_grid.csv"
        capacity.to_csv(capacity_path, index=False)
        ml_grid.to_csv(ml_grid_path, index=False)
        with R.start(experiment_name=spec.experiment):
            recorder = R.get_recorder()
            R.log_params(
                daily_provider_uri=spec.daily_provider_uri.resolve().as_posix(),
                intraday_15m_provider_uri=spec.intraday_15m_provider_uri.resolve().as_posix(),
                benchmark="SOXX",
                benchmark_basis="open_to_open",
                allocation_assets="SOXX,SOXL,cash",
                frontier_policy="trend_vix_active_signal_plus_cppi_intraday_stop",
                model_families="xgb,lgbm,avg_xgb_lgb",
                objective="SOXX plus 20 percentage points with max drawdown above -10%",
            )
            R.log_metrics(
                soxx_return_pct=float(benchmark["return_pct"]),
                soxx_max_drawdown_pct=float(benchmark["max_drawdown_pct"]),
                capacity_passes_target_count=float(capacity_summary["passes_target_count"]),
                capacity_best_drawdown_safe_return_pct=float(capacity_summary["best_drawdown_safe_return_pct"]),
                ml_stitched_return_pct=float(ml_summary["stitched_return_pct"]),
                ml_stitched_max_drawdown_pct=float(ml_summary["stitched_max_drawdown_pct"]),
                ml_stitched_excess_vs_soxx_pct=float(ml_summary["stitched_excess_vs_soxx_pct"]),
            )
            recorder.log_artifact(capacity_path.as_posix(), artifact_path=spec.experiment)
            recorder.log_artifact(ml_grid_path.as_posix(), artifact_path=spec.experiment)
            summary = _summary_payload(
                spec=spec,
                common_end=common_end,
                recorder_id=recorder.id,
                recorder_uri=recorder_uri,
                recorder_local_dir=Path(recorder.get_local_dir()).resolve(),
                source_summary=source_summary,
                feature_audit=feature_audit,
                intraday_audit=intraday_audit,
                benchmark=benchmark,
                capacity_summary=capacity_summary,
                ml_summary=ml_summary,
                sidecars=sidecars,
                promotion_decision=promotion_decision,
                model_comparison=model_comparison,
            )
            _write_sidecars(
                card,
                sidecars=sidecars,
                summary=summary,
                capacity=capacity,
                model_comparison=model_comparison,
            )
            for rel_path in sidecars.values():
                path = card / rel_path
                if path.exists() and path.suffix != ".md":
                    recorder.log_artifact(path.as_posix(), artifact_path=spec.experiment)
            _remove_qlib_code_snapshot_artifacts(Path(recorder.get_local_dir()).resolve())
    _write_summary_files(card, summary)
    return summary


def _load_next_session_intraday_low(
    spec: DynamicSoxxSoxlCashAllocationWfV8Spec,
    *,
    frame: pd.DataFrame,
    common_end: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
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
        fields=["$open", "$low"],
        start=spec.intraday_start,
        end=end,
    )
    panel = qframe.reset_index().copy()
    panel["datetime"] = pd.to_datetime(panel["datetime"])
    panel["date"] = panel["datetime"].dt.normalize()
    panel["instrument"] = panel["instrument"].astype(str).str.upper()
    for column in ["$open", "$low"]:
        panel[column] = pd.to_numeric(panel[column], errors="coerce")

    rows: list[dict[str, Any]] = []
    for (instrument, date), group in panel.groupby(["instrument", "date"], sort=True):
        group = group.sort_values("datetime")
        open_values = group["$open"].dropna()
        open_ = float(open_values.iloc[0]) if not open_values.empty else np.nan
        low = float(group["$low"].min())
        rows.append(
            {
                "date": pd.Timestamp(date).normalize(),
                "instrument": str(instrument),
                "intraday_low_return_from_open": low / open_ - 1.0 if np.isfinite(open_) and open_ != 0 else np.nan,
            }
        )
    daily = pd.DataFrame(rows)
    wide = daily.pivot(index="date", columns="instrument", values="intraday_low_return_from_open").reset_index()
    wide = wide.rename(columns={"SOXL": "soxl_next_intraday_low_return", "SOXX": "soxx_next_intraday_low_return"})
    dates = pd.to_datetime(frame["datetime"]).dt.normalize()
    aligned = pd.DataFrame({"datetime": dates, "trade_date": dates.shift(-1)}).merge(
        wide,
        left_on="trade_date",
        right_on="date",
        how="left",
    )
    out = aligned.loc[:, ["datetime", "trade_date", "soxl_next_intraday_low_return", "soxx_next_intraday_low_return"]]
    test_mask = dates.dt.year.isin(spec.test_years).to_numpy()
    audit = {
        "intraday_15m_provider_uri": spec.intraday_15m_provider_uri.resolve().as_posix(),
        "row_count": int(len(panel)),
        "daily_session_count": int(wide["date"].nunique()),
        "test_soxl_low_coverage": float(out.loc[test_mask, "soxl_next_intraday_low_return"].notna().mean()),
        "test_soxx_low_coverage": float(out.loc[test_mask, "soxx_next_intraday_low_return"].notna().mean()),
        "timing": "daily_close_t_signal_next_session_15m_intraday_stop",
    }
    return out, audit


def _capacity_frontier(
    frame: pd.DataFrame,
    *,
    intraday_low: pd.DataFrame,
    spec: DynamicSoxxSoxlCashAllocationWfV8Spec,
    benchmark: dict[str, Any],
) -> pd.DataFrame:
    work = _attach_rule_features(frame)
    test_mask = work["datetime"].dt.year.isin(spec.test_years).to_numpy()
    low_soxl, low_soxx = _intraday_low_arrays(intraday_low)
    rows: list[dict[str, Any]] = []
    for active_name, active in _active_rule_masks(work).items():
        for mid_name, mid_raw in _mid_rule_masks(work).items():
            mid = mid_raw & ~active
            for base in [0.0, 0.2, 0.4, 0.8, 1.0]:
                for active_soxl in [0.5, 0.7, 1.0]:
                    for active_soxx in [0.0, 0.2, 0.4]:
                        if active_soxx + active_soxl > 1.0:
                            continue
                        for floor_dd in [0.20, 0.25]:
                            for multiplier in [2.0, 3.0, 5.0]:
                                for day_stop in [0.0, 0.02, 0.03]:
                                    result = _simulate_cppi_intraday_stop(
                                        work,
                                        active=active,
                                        mid=mid,
                                        low_soxl=low_soxl,
                                        low_soxx=low_soxx,
                                        mask=test_mask,
                                        params={
                                            "base": base,
                                            "active_soxx": active_soxx,
                                            "active_soxl": active_soxl,
                                            "floor_dd": floor_dd,
                                            "multiplier": multiplier,
                                            "day_stop": day_stop,
                                        },
                                        cost_per_turnover=spec.cost_per_turnover,
                                    )
                                    rows.append(
                                        {
                                            "active_rule": active_name,
                                            "mid_rule": mid_name,
                                            "base": base,
                                            "active_soxx": active_soxx,
                                            "active_soxl": active_soxl,
                                            "floor_dd": floor_dd,
                                            "multiplier": multiplier,
                                            "day_stop": day_stop,
                                            "uses_15m_intraday_stop": bool(day_stop > 0.0),
                                            **_metric_payload(result, benchmark_return=float(benchmark["return_pct"])),
                                            "selection_uses_outer_test": True,
                                            "notes": "outer_test_capacity_frontier_not_promotable",
                                        }
                                    )
    return pd.DataFrame(rows).sort_values(["passes_target", "max_drawdown_pct", "return_pct"], ascending=[False, False, False])


def _build_predictions_by_model(
    frame: pd.DataFrame,
    *,
    feature_cols: list[str],
    spec: DynamicSoxxSoxlCashAllocationWfV8Spec,
) -> dict[str, pd.DataFrame]:
    xgb = _rolling_predictions("xgb", frame, feature_cols=feature_cols, spec=spec)
    lgbm = _rolling_predictions("lgbm", frame, feature_cols=feature_cols, spec=spec)
    avg = xgb[["datetime", "year", ETF_RETURN_COL, UNDERLYING_RETURN_COL]].copy()
    avg["pup"] = (xgb["pup"].to_numpy(dtype="float64") + lgbm["pup"].to_numpy(dtype="float64")) / 2.0
    avg["ptail"] = (xgb["ptail"].to_numpy(dtype="float64") + lgbm["ptail"].to_numpy(dtype="float64")) / 2.0
    return {"xgb": xgb, "lgbm": lgbm, "avg_xgb_lgb": avg}


def _rolling_predictions(
    model_name: str,
    frame: pd.DataFrame,
    *,
    feature_cols: list[str],
    spec: DynamicSoxxSoxlCashAllocationWfV8Spec,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    data = frame.copy().sort_values("datetime")
    for year in spec.prediction_years:
        train = data.loc[data["datetime"].dt.year.lt(int(year))].copy()
        test = data.loc[data["datetime"].dt.year.eq(int(year))].copy()
        if len(train) < 500 or test.empty:
            continue
        train_x = train[feature_cols].astype("float64")
        test_x = test[feature_cols].astype("float64")
        up_model = _classifier(model_name, seed=8100 + int(year))
        tail_model = _classifier(model_name, seed=8200 + int(year))
        up_model.fit(train_x, train[ETF_RETURN_COL].gt(train[UNDERLYING_RETURN_COL]).astype(int))
        tail_cut = float(train[ETF_RETURN_COL].quantile(0.08))
        tail_model.fit(train_x, train[ETF_RETURN_COL].le(tail_cut).astype(int))
        segment = test[["datetime", ETF_RETURN_COL, UNDERLYING_RETURN_COL]].copy()
        segment["year"] = int(year)
        segment["pup"] = _positive_proba(up_model, test_x)
        segment["ptail"] = _positive_proba(tail_model, test_x)
        parts.append(segment)
    return pd.concat(parts, ignore_index=True).sort_values("datetime").reset_index(drop=True)


def _classifier(model_name: str, *, seed: int):
    if model_name == "xgb":
        from xgboost import XGBClassifier

        return make_pipeline(
            SimpleImputer(strategy="median"),
            XGBClassifier(
                n_estimators=70,
                max_depth=2,
                learning_rate=0.04,
                subsample=0.82,
                colsample_bytree=0.75,
                reg_alpha=0.15,
                reg_lambda=1.4,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=seed,
                n_jobs=1,
            ),
        )
    if model_name == "lgbm":
        from lightgbm import LGBMClassifier

        return make_pipeline(
            SimpleImputer(strategy="median"),
            LGBMClassifier(
                n_estimators=80,
                max_depth=2,
                num_leaves=7,
                learning_rate=0.04,
                subsample=0.82,
                colsample_bytree=0.75,
                reg_alpha=0.15,
                reg_lambda=1.4,
                random_state=seed,
                n_jobs=1,
                verbose=-1,
            ),
        )
    raise ValueError(f"unsupported model family: {model_name}")


def _ml_risk_budget_selector(
    predictions_by_model: dict[str, pd.DataFrame],
    *,
    frame: pd.DataFrame,
    intraday_low: pd.DataFrame,
    spec: DynamicSoxxSoxlCashAllocationWfV8Spec,
    benchmark: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    low_lookup = intraday_low.set_index("datetime")
    grid = _ml_param_grid()
    all_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    best_summary: dict[str, Any] | None = None
    best_score = -np.inf
    for model_name, pred in predictions_by_model.items():
        aligned_low = low_lookup.reindex(pd.to_datetime(pred["datetime"]).dt.normalize())
        low_soxl = aligned_low["soxl_next_intraday_low_return"].to_numpy(dtype="float64")
        low_soxx = aligned_low["soxx_next_intraday_low_return"].to_numpy(dtype="float64")
        model_rows = _nested_model_rows(
            model_name,
            pred,
            low_soxl=low_soxl,
            low_soxx=low_soxx,
            grid=grid,
            spec=spec,
            benchmark=benchmark,
        )
        all_rows.extend(model_rows)
        stitched_return = float(model_rows[0]["stitched_return_pct"]) if model_rows else float("nan")
        stitched_dd = float(model_rows[0]["stitched_max_drawdown_pct"]) if model_rows else float("nan")
        stitched_excess = float(model_rows[0]["stitched_excess_vs_soxx_pct"]) if model_rows else float("nan")
        required = bool(stitched_excess >= SOXX_EXCESS_FLOOR_PCT and stitched_dd >= MAX_DRAWDOWN_FLOOR_PCT)
        comparison_rows.append(
            {
                "model": model_name,
                "selection_uses_outer_test": False,
                "stitched_return_pct": stitched_return,
                "stitched_max_drawdown_pct": stitched_dd,
                "stitched_excess_vs_soxx_pct": stitched_excess,
                "required_policy_satisfied": required,
            }
        )
        score = stitched_return - 100.0 * max(0.0, MAX_DRAWDOWN_FLOOR_PCT - stitched_dd)
        if np.isfinite(score) and score > best_score:
            best_score = score
            best_summary = comparison_rows[-1]
    ml_grid = pd.DataFrame(all_rows)
    model_comparison = pd.DataFrame(comparison_rows)
    selected = best_summary or {
        "model": "",
        "stitched_return_pct": float("nan"),
        "stitched_max_drawdown_pct": float("nan"),
        "stitched_excess_vs_soxx_pct": float("nan"),
        "required_policy_satisfied": False,
    }
    required = bool(selected["required_policy_satisfied"])
    summary = {
        "selection_uses_outer_test": False,
        "selection_policy": "rolling_validation_selects_lgb_xgb_risk_budget_cppi_params",
        "model_families": list(MODEL_FAMILIES),
        "selected_model": str(selected["model"]),
        "validation_window_years": int(spec.validation_window_years),
        "stitched_return_pct": float(selected["stitched_return_pct"]),
        "stitched_max_drawdown_pct": float(selected["stitched_max_drawdown_pct"]),
        "stitched_excess_vs_soxx_pct": float(selected["stitched_excess_vs_soxx_pct"]),
        "required_policy_satisfied": required,
        "blocking_reasons": [] if required else ["risk_budget_ml_selector_failed_goal"],
    }
    return ml_grid, model_comparison, summary


def _nested_model_rows(
    model_name: str,
    pred: pd.DataFrame,
    *,
    low_soxl: np.ndarray,
    low_soxx: np.ndarray,
    grid: list[dict[str, float]],
    spec: DynamicSoxxSoxlCashAllocationWfV8Spec,
    benchmark: dict[str, Any],
) -> list[dict[str, Any]]:
    years = pred["year"].to_numpy(dtype="int64")
    candidate_returns: list[np.ndarray] = []
    for params in grid:
        active = (
            pd.to_numeric(pred["pup"], errors="coerce").fillna(0.0).to_numpy(dtype="float64") >= float(params["pupthr"])
        ) & (
            pd.to_numeric(pred["ptail"], errors="coerce").fillna(1.0).to_numpy(dtype="float64") <= float(params["ptailthr"])
        )
        mid = pd.to_numeric(pred["pup"], errors="coerce").fillna(0.0).to_numpy(dtype="float64") >= float(params["midthr"])
        returns, _, _ = _simulate_cppi_intraday_stop_arrays(
            pd.to_numeric(pred[UNDERLYING_RETURN_COL], errors="coerce").fillna(0.0).to_numpy(dtype="float64"),
            pd.to_numeric(pred[ETF_RETURN_COL], errors="coerce").fillna(0.0).to_numpy(dtype="float64"),
            np.nan_to_num(low_soxx, nan=0.0),
            np.nan_to_num(low_soxl, nan=0.0),
            active,
            mid & ~active,
            base=float(params["base"]),
            active_soxx=float(params["active_soxx"]),
            active_soxl=float(params["active_soxl"]),
            floor_dd=float(params["floor_dd"]),
            multiplier=float(params["multiplier"]),
            day_stop=float(params["day_stop"]),
            cost_per_turnover=spec.cost_per_turnover,
        )
        candidate_returns.append(returns)
    stitched: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    for outer_year in spec.test_years:
        valid_mask = (years >= int(outer_year) - int(spec.validation_window_years)) & (years < int(outer_year))
        test_mask = years == int(outer_year)
        best: tuple[float, dict[str, float], np.ndarray, float, float] | None = None
        for params, returns in zip(grid, candidate_returns, strict=True):
            valid_return, valid_dd = _return_drawdown(returns[valid_mask])
            score = valid_return - 100.0 * max(0.0, MAX_DRAWDOWN_FLOOR_PCT - valid_dd)
            if best is None or score > best[0]:
                best = (score, params, returns, valid_return, valid_dd)
        if best is None:
            continue
        _, selected, returns, valid_return, valid_dd = best
        test_return, test_dd = _return_drawdown(returns[test_mask])
        annual_benchmark, _ = _return_drawdown(
            pd.to_numeric(pred.loc[test_mask, UNDERLYING_RETURN_COL], errors="coerce").fillna(0.0).to_numpy(dtype="float64")
        )
        stitched.append(returns[test_mask])
        rows.append(
            {
                "model": model_name,
                "outer_year": int(outer_year),
                **selected,
                "valid_return_pct": float(valid_return),
                "valid_max_drawdown_pct": float(valid_dd),
                "test_return_pct": float(test_return),
                "test_max_drawdown_pct": float(test_dd),
                "test_soxx_return_pct": float(annual_benchmark),
                "test_excess_vs_annual_soxx_pct": float(test_return - annual_benchmark),
                "selection_uses_outer_test": False,
            }
        )
    stitched_returns = np.concatenate(stitched) if stitched else np.asarray([], dtype="float64")
    stitched_return, stitched_dd = _return_drawdown(stitched_returns)
    stitched_excess = stitched_return - float(benchmark["return_pct"])
    required = bool(stitched_excess >= SOXX_EXCESS_FLOOR_PCT and stitched_dd >= MAX_DRAWDOWN_FLOOR_PCT)
    for row in rows:
        row["stitched_return_pct"] = float(stitched_return)
        row["stitched_max_drawdown_pct"] = float(stitched_dd)
        row["stitched_excess_vs_soxx_pct"] = float(stitched_excess)
        row["required_policy_satisfied"] = required
    return rows


def _ml_param_grid() -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for pupthr in [0.52, 0.56, 0.60]:
        for ptailthr in [0.45, 0.60, 0.75]:
            for midthr in [0.50, 0.56]:
                for base in [0.0, 0.2, 0.4]:
                    for active_soxl in [0.5, 0.7]:
                        rows.append(
                            {
                                "pupthr": pupthr,
                                "ptailthr": ptailthr,
                                "midthr": midthr,
                                "base": base,
                                "active_soxx": 0.2,
                                "active_soxl": active_soxl,
                                "floor_dd": 0.25,
                                "multiplier": 3.0,
                                "day_stop": 0.02,
                            }
                        )
    return rows


def _attach_rule_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy().sort_values("datetime").reset_index(drop=True)
    soxx = pd.to_numeric(out["underlying_close"], errors="coerce")
    vix = pd.to_numeric(out["vix"], errors="coerce")
    out["soxx_ret_20"] = soxx / soxx.shift(20) - 1.0
    out["soxx_ret_60"] = soxx / soxx.shift(60) - 1.0
    out["soxx_gap_60"] = soxx / soxx.rolling(60, min_periods=30).mean() - 1.0
    out["soxx_gap_200"] = soxx / soxx.rolling(200, min_periods=100).mean() - 1.0
    out["vix_z60"] = (vix - vix.rolling(60, min_periods=20).mean()) / vix.rolling(60, min_periods=20).std(ddof=0)
    return out.replace([np.inf, -np.inf], np.nan)


def _active_rule_masks(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "soxx_ret60_gt_0389_vixz60_lt_018": (
            pd.to_numeric(frame["soxx_ret_60"], errors="coerce").fillna(-9.0).to_numpy(dtype="float64") > 0.0389
        )
        & (pd.to_numeric(frame["vix_z60"], errors="coerce").fillna(9.0).to_numpy(dtype="float64") < 0.1803),
        "soxx_ret60_gt_04_vixz60_lt_0": (
            pd.to_numeric(frame["soxx_ret_60"], errors="coerce").fillna(-9.0).to_numpy(dtype="float64") > 0.04
        )
        & (pd.to_numeric(frame["vix_z60"], errors="coerce").fillna(9.0).to_numpy(dtype="float64") < 0.0),
        "soxx_gap60_gt_0_vixz60_lt_0": (
            pd.to_numeric(frame["soxx_gap_60"], errors="coerce").fillna(-9.0).to_numpy(dtype="float64") > 0.0
        )
        & (pd.to_numeric(frame["vix_z60"], errors="coerce").fillna(9.0).to_numpy(dtype="float64") < 0.0),
    }


def _mid_rule_masks(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "cash_only": np.zeros(len(frame), dtype=bool),
        "soxx_ret20_gt_0": pd.to_numeric(frame["soxx_ret_20"], errors="coerce").fillna(-9.0).to_numpy(dtype="float64") > 0.0,
        "soxx_gap200_gt_0": pd.to_numeric(frame["soxx_gap_200"], errors="coerce").fillna(-9.0).to_numpy(dtype="float64") > 0.0,
        "soxx_gap60_gt_0": pd.to_numeric(frame["soxx_gap_60"], errors="coerce").fillna(-9.0).to_numpy(dtype="float64") > 0.0,
    }


def _intraday_low_arrays(intraday_low: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    return (
        pd.to_numeric(intraday_low["soxl_next_intraday_low_return"], errors="coerce").fillna(0.0).to_numpy(dtype="float64"),
        pd.to_numeric(intraday_low["soxx_next_intraday_low_return"], errors="coerce").fillna(0.0).to_numpy(dtype="float64"),
    )


def _simulate_cppi_intraday_stop(
    frame: pd.DataFrame,
    *,
    active: np.ndarray,
    mid: np.ndarray,
    low_soxl: np.ndarray,
    low_soxx: np.ndarray,
    mask: np.ndarray,
    params: dict[str, float],
    cost_per_turnover: float,
) -> dict[str, Any]:
    returns, soxx_weight, soxl_weight = _simulate_cppi_intraday_stop_arrays(
        pd.to_numeric(frame[UNDERLYING_RETURN_COL], errors="coerce").fillna(0.0).to_numpy(dtype="float64"),
        pd.to_numeric(frame[ETF_RETURN_COL], errors="coerce").fillna(0.0).to_numpy(dtype="float64"),
        low_soxx,
        low_soxl,
        active,
        mid,
        base=float(params["base"]),
        active_soxx=float(params["active_soxx"]),
        active_soxl=float(params["active_soxl"]),
        floor_dd=float(params["floor_dd"]),
        multiplier=float(params["multiplier"]),
        day_stop=float(params["day_stop"]),
        cost_per_turnover=cost_per_turnover,
    )
    ret, maxdd = _return_drawdown(returns[mask])
    return {
        "return_pct": ret,
        "max_drawdown_pct": maxdd,
        "avg_soxx_weight": float(np.mean(soxx_weight[mask])),
        "avg_soxl_weight": float(np.mean(soxl_weight[mask])),
        "active_share": float(np.mean(active[mask])),
        "mid_share": float(np.mean(mid[mask])),
    }


def _simulate_cppi_intraday_stop_arrays(
    soxx_returns: np.ndarray,
    soxl_returns: np.ndarray,
    low_soxx: np.ndarray,
    low_soxl: np.ndarray,
    active: np.ndarray,
    mid: np.ndarray,
    *,
    base: float,
    active_soxx: float,
    active_soxl: float,
    floor_dd: float,
    multiplier: float,
    day_stop: float,
    cost_per_turnover: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    returns = np.zeros(len(soxx_returns), dtype="float64")
    soxx_weight = np.zeros(len(soxx_returns), dtype="float64")
    soxl_weight = np.zeros(len(soxx_returns), dtype="float64")
    prev_soxx = 0.0
    prev_soxl = 0.0
    equity = 1.0
    peak = 1.0
    for idx in range(len(soxx_returns)):
        floor_value = peak * (1.0 - float(floor_dd))
        cushion = max(0.0, (equity - floor_value) / max(equity, 1e-12))
        scale = min(1.0, float(multiplier) * cushion)
        if active[idx]:
            target_soxx = float(active_soxx)
            target_soxl = float(active_soxl)
        elif mid[idx]:
            target_soxx = float(base)
            target_soxl = 0.0
        else:
            target_soxx = 0.0
            target_soxl = 0.0
        current_soxx = target_soxx * scale
        current_soxl = target_soxl * scale
        turnover = abs(current_soxx - prev_soxx) + abs(current_soxl - prev_soxl)
        intraday_low_return = current_soxx * low_soxx[idx] + current_soxl * low_soxl[idx]
        if day_stop > 0.0 and intraday_low_return <= -float(day_stop) and (current_soxx + current_soxl) > 0.0:
            daily_return = -float(day_stop) - turnover * float(cost_per_turnover)
        else:
            daily_return = current_soxx * soxx_returns[idx] + current_soxl * soxl_returns[idx] - turnover * float(cost_per_turnover)
        returns[idx] = daily_return
        soxx_weight[idx] = current_soxx
        soxl_weight[idx] = current_soxl
        equity *= 1.0 + daily_return
        peak = max(peak, equity)
        prev_soxx = current_soxx
        prev_soxl = current_soxl
    return returns, soxx_weight, soxl_weight


def _metric_payload(result: dict[str, Any], *, benchmark_return: float) -> dict[str, Any]:
    excess = float(result["return_pct"]) - float(benchmark_return)
    return {
        "return_pct": float(result["return_pct"]),
        "max_drawdown_pct": float(result["max_drawdown_pct"]),
        "excess_vs_soxx_pct": excess,
        "passes_target": bool(excess >= SOXX_EXCESS_FLOOR_PCT and float(result["max_drawdown_pct"]) >= MAX_DRAWDOWN_FLOOR_PCT),
        "avg_soxx_weight": float(result["avg_soxx_weight"]),
        "avg_soxl_weight": float(result["avg_soxl_weight"]),
        "active_share": float(result["active_share"]),
        "mid_share": float(result["mid_share"]),
    }


def _capacity_summary(capacity: pd.DataFrame) -> dict[str, Any]:
    dd_safe = capacity.loc[capacity["max_drawdown_pct"].ge(MAX_DRAWDOWN_FLOOR_PCT)].copy()
    passes = capacity.loc[capacity["passes_target"]].copy()
    best_dd = dd_safe.sort_values("return_pct", ascending=False).head(1)
    best_return = capacity.sort_values("return_pct", ascending=False).head(1)
    return {
        "selection_uses_outer_test": True,
        "frontier_policy": "trend_vix_active_signal_plus_cppi_intraday_stop",
        "uses_15m_intraday_stop": True,
        "row_count": int(len(capacity)),
        "passes_target_count": int(capacity["passes_target"].sum()),
        "best_drawdown_safe_return_pct": _frame_float(best_dd, "return_pct"),
        "best_drawdown_safe_max_drawdown_pct": _frame_float(best_dd, "max_drawdown_pct"),
        "best_return_pct": _frame_float(best_return, "return_pct"),
        "best_return_max_drawdown_pct": _frame_float(best_return, "max_drawdown_pct"),
        "best_drawdown_safe_params": _row_params(best_dd),
    }


def _summary_payload(
    *,
    spec: DynamicSoxxSoxlCashAllocationWfV8Spec,
    common_end: str,
    recorder_id: str,
    recorder_uri: Path,
    recorder_local_dir: Path,
    source_summary: dict[str, Any],
    feature_audit: dict[str, Any],
    intraday_audit: dict[str, Any],
    benchmark: dict[str, Any],
    capacity_summary: dict[str, Any],
    ml_summary: dict[str, Any],
    sidecars: dict[str, str],
    promotion_decision: str,
    model_comparison: pd.DataFrame,
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
            "execution_timing": "close_t_signal_next_open_with_15m_intraday_stop",
            "goal": {
                "excess_vs_soxx_floor_pct": SOXX_EXCESS_FLOOR_PCT,
                "max_drawdown_floor_pct": MAX_DRAWDOWN_FLOOR_PCT,
            },
            "benchmark": benchmark,
            "feature_audit": feature_audit,
            "intraday_audit": intraday_audit,
            "intraday_cppi_capacity": capacity_summary,
            "ml_risk_budget_selector": ml_summary,
            "model_comparison_preview": model_comparison.to_dict("records"),
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
                "XGBClassifier",
                "LGBMClassifier",
            ],
            "sidecars": sidecars,
            "artifact_policy": {
                "policy": "slim_card_recorder_first",
                "workflow_config": "workflow.yaml",
                "recorder_only_artifacts": [
                    f"artifacts/{spec.experiment}/capacity_frontier.csv",
                    f"artifacts/{spec.experiment}/ml_risk_budget_grid.csv",
                ],
            },
            "workflow_config": "workflow.yaml",
            "recorder_id": recorder_id,
            "recorder_uri": recorder_uri.resolve().as_posix(),
            "recorder_local_dir": recorder_local_dir.resolve().as_posix(),
        }
    )


def _write_sidecars(
    card: Path,
    *,
    sidecars: dict[str, str],
    summary: dict[str, Any],
    capacity: pd.DataFrame,
    model_comparison: pd.DataFrame,
) -> None:
    frontier = pd.concat(
        [
            capacity.loc[capacity["max_drawdown_pct"].ge(MAX_DRAWDOWN_FLOOR_PCT)].sort_values("return_pct", ascending=False).head(20),
            capacity.sort_values("return_pct", ascending=False).head(20),
        ],
        ignore_index=True,
    ).drop_duplicates()
    frontier.to_csv(card / sidecars["frontier_summary"], index=False)
    model_comparison.to_csv(card / sidecars["model_comparison"], index=False)
    analysis = {
        "source": "qlib_recorder_intraday_cppi_ml_risk_budget_frontier",
        "recorder_id": summary["recorder_id"],
        "intraday_cppi_capacity": summary["intraday_cppi_capacity"],
        "ml_risk_budget_selector": summary["ml_risk_budget_selector"],
        "promotion_decision": summary["promotion_decision"],
    }
    (card / sidecars["analysis_summary"]).write_text(json.dumps(_json_ready(analysis), ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(
        [
            {
                "check": "intraday_cppi_capacity",
                "status": "fail",
                "details": json.dumps(summary["intraday_cppi_capacity"], ensure_ascii=False, sort_keys=True),
            },
            {
                "check": "ml_risk_budget_selector",
                "status": "fail" if not summary["ml_risk_budget_selector"]["required_policy_satisfied"] else "pass",
                "details": json.dumps(summary["ml_risk_budget_selector"], ensure_ascii=False, sort_keys=True),
            },
        ]
    ).to_csv(card / sidecars["risk_analysis"], index=False)
    selected = {
        "selected_model": None,
        "promotion_decision": summary["promotion_decision"],
        "reason": "15m CPPI and ML risk-budget selector did not satisfy SOXX+20/maxDD<10 under validation-only selection",
        "capacity": summary["intraday_cppi_capacity"],
        "ml_risk_budget_selector": summary["ml_risk_budget_selector"],
    }
    (card / sidecars["selected_model"]).write_text(json.dumps(_json_ready(selected), ensure_ascii=False, indent=2), encoding="utf-8")
    (card / sidecars["failure_modes"]).write_text(_failure_modes_text(summary), encoding="utf-8")


def _write_summary_files(card: Path, summary: dict[str, Any]) -> None:
    (card / "summary.json").write_text(json.dumps(_json_ready(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    (card / "conclusion.md").write_text(_conclusion_text(summary), encoding="utf-8")
    (card / "workflow.yaml").write_text(_workflow_yaml(summary), encoding="utf-8")


def _failure_modes_text(summary: dict[str, Any]) -> str:
    cap = summary["intraday_cppi_capacity"]
    ml = summary["ml_risk_budget_selector"]
    return (
        "# Failure Modes\n\n"
        "- 15m intraday stop and CPPI reduce drawdown by cutting SOXL exposure, but drawdown-safe return remains below SOXX+20.\n"
        "- LGB/XGB risk-budget selector uses validation-only parameter selection and does not satisfy the promotion gate.\n"
        f"- Best drawdown-safe frontier return is {_fmt_pct(cap['best_drawdown_safe_return_pct'])}% with "
        f"{_fmt_pct(cap['best_drawdown_safe_max_drawdown_pct'])}% max drawdown.\n"
        f"- ML stitched return is {_fmt_pct(ml['stitched_return_pct'])}% with "
        f"{_fmt_pct(ml['stitched_max_drawdown_pct'])}% max drawdown.\n"
    )


def _conclusion_text(summary: dict[str, Any]) -> str:
    cap = summary["intraday_cppi_capacity"]
    ml = summary["ml_risk_budget_selector"]
    return (
        "# 中文结论\n\n"
        "- 做了什么：新增 v8 15m intraday stop + CPPI 风险预算边界，并加入 LGB/XGB/平均模型的 risk-budget selector。\n"
        f"- 容量结果：frontier rows={cap['row_count']}，passes={cap['passes_target_count']}，"
        f"best drawdown-safe return={_fmt_pct(cap['best_drawdown_safe_return_pct'])}%，"
        f"maxDD={_fmt_pct(cap['best_drawdown_safe_max_drawdown_pct'])}%。\n"
        f"- ML 结果：selected={ml['selected_model']}，stitched return={_fmt_pct(ml['stitched_return_pct'])}%，"
        f"maxDD={_fmt_pct(ml['stitched_max_drawdown_pct'])}%，excess={_fmt_pct(ml['stitched_excess_vs_soxx_pct'])}%。\n"
        "- 结论：15m intraday stop 与 CPPI 能降低回撤，但一旦把回撤压到 10% 内，SOXL 平均风险预算过低，无法达到 SOXX+20；验证期选择的 ML 风险预算也未通过，不晋升。\n"
        f"- 是否晋升：不晋升；decision={summary['promotion_decision']}。\n"
        "- 下一步：如果继续冲击该目标，需要引入更细的 execution alpha 或不同目标约束；当前日线信号加 15m 风控不足以证明实盘能力。\n"
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
            "  models: [xgb, lgbm, avg_xgb_lgb]",
            "  selection_uses_outer_test: false",
            "frontier:",
            "  policy: trend_vix_active_signal_plus_cppi_intraday_stop",
            "  capacity_selection_uses_outer_test: true",
            "artifact_policy:",
            "  policy: slim_card_recorder_first",
            "  workflow_config: workflow.yaml",
            f"promotion_decision: {summary['promotion_decision']}",
            "",
        ]
    )


def _frame_float(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame:
        return float("nan")
    return float(frame[column].iloc[0])


def _row_params(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {}
    row = frame.iloc[0].to_dict()
    return _json_ready(
        {
            key: row[key]
            for key in [
                "active_rule",
                "mid_rule",
                "base",
                "active_soxx",
                "active_soxl",
                "floor_dd",
                "multiplier",
                "day_stop",
            ]
            if key in row
        }
    )


def _fmt_pct(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "nan"
    if not np.isfinite(numeric):
        return "nan"
    return f"{numeric:.4f}"
