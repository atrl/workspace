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
from sklearn.preprocessing import LabelEncoder

from src.data import QlibProviderConfig, init_qlib_provider
from src.workflow.dynamic_soxx_soxl_cash_allocation_wf import (
    CALIBRATION_ASSET,
    ETF_RETURN_COL,
    PAIR_UNIVERSE,
    UNDERLYING_RETURN_COL,
    DynamicSoxxSoxlCashAllocationWfSpec,
)
from src.workflow.multi_3x_etf_normalized_nested_wf import _json_ready, _resolve_root
from src.workflow.trainable_regime_policy_nested_wf import _remove_qlib_code_snapshot_artifacts


EXPERIMENT = "dynamic_soxx_soxl_cash_allocation_wf_v2"
CARD_TYPE = "dynamic_soxx_soxl_cash_allocation_frontier_research_card"
SOURCE_EXPERIMENT = "dynamic_soxx_soxl_cash_allocation_wf_v1"
SOXX_EXCESS_FLOOR_PCT = 20.0
MAX_DRAWDOWN_FLOOR_PCT = -10.0


@dataclass(frozen=True)
class DynamicSoxxSoxlCashAllocationWfV2Spec:
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

    @property
    def intraday_15m_card_dir(self) -> Path:
        return self.root / "research" / "agentic_alpha_intraday_15m_regime_gated_v1"


def run_dynamic_soxx_soxl_cash_allocation_wf_v2_card(*, root: Path | str | None = None) -> dict[str, Any]:
    spec = DynamicSoxxSoxlCashAllocationWfV2Spec(root=_resolve_root(root))
    return _run_card(spec)


def _run_card(spec: DynamicSoxxSoxlCashAllocationWfV2Spec) -> dict[str, Any]:
    card = spec.card_dir
    card.mkdir(parents=True, exist_ok=True)
    source_summary = json.loads((spec.source_card_dir / "summary.json").read_text(encoding="utf-8"))
    common_end = str(source_summary["common_data_end"])
    feature_sets = {
        str(item["feature_set"]): list(item["feature_cols"]) for item in source_summary["feature_sets"]["available"]
    }
    frame = _load_provider_frame(spec, common_end=common_end, feature_sets=feature_sets)
    benchmark = _benchmark_metrics(frame, spec)
    direct_frontier = _direct_action_frontier(frame, spec=spec, feature_sets=feature_sets)
    trend_frontier = _trend_rule_frontier(frame, spec=spec)
    intraday_summary = _intraday_context_summary(spec)

    frontier = pd.concat([direct_frontier, trend_frontier], ignore_index=True, sort=False)
    stitched_goal = _stitched_goal_summary(frontier, benchmark=benchmark)
    annual_goal = _annual_goal_summary(frontier, required_years=spec.test_years)
    promotion_decision = (
        "promoted_dynamic_soxx_soxl_cash_goal_passed"
        if stitched_goal["required_policy_satisfied"] and annual_goal["required_policy_satisfied"]
        else "not_promoted_dynamic_allocation_frontier_goal_failed"
    )
    sidecars = {
        "frontier_diagnostics": "frontier_diagnostics.csv",
        "model_comparison": "model_comparison.csv",
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
        frontier_path = tmp_dir / "frontier_diagnostics.csv"
        frontier.to_csv(frontier_path, index=False)
        with R.start(experiment_name=spec.experiment):
            recorder = R.get_recorder()
            R.log_params(
                provider_uri=spec.provider_uri.resolve().as_posix(),
                source_experiment=SOURCE_EXPERIMENT,
                allocation_assets="SOXX,SOXL,cash",
                benchmark="SOXX",
                benchmark_basis="open_to_open",
                objective="SOXX plus 20 percentage points with max drawdown above -10%",
                model_families="lgbm,xgb,lgb_xgb_consensus",
                selection_uses_outer_test="false",
            )
            R.log_metrics(
                soxx_stitched_return_pct=float(benchmark["return_pct"]),
                soxx_stitched_max_drawdown_pct=float(benchmark["max_drawdown_pct"]),
                frontier_passes_target_count=float(stitched_goal["passes_target_count"]),
                frontier_best_dd_safe_return_pct=float(stitched_goal["best_drawdown_safe_return_pct"]),
                frontier_min_drawdown_if_excess_passes_pct=float(stitched_goal["min_drawdown_if_excess_passes_pct"]),
            )
            recorder.log_artifact(frontier_path.as_posix(), artifact_path=spec.experiment)
            summary = _summary_payload(
                spec=spec,
                common_end=common_end,
                recorder_id=recorder.id,
                recorder_uri=recorder_uri,
                recorder_local_dir=Path(recorder.get_local_dir()).resolve(),
                sidecars=sidecars,
                source_summary=source_summary,
                benchmark=benchmark,
                frontier=frontier,
                direct_frontier=direct_frontier,
                trend_frontier=trend_frontier,
                stitched_goal=stitched_goal,
                annual_goal=annual_goal,
                intraday_summary=intraday_summary,
                promotion_decision=promotion_decision,
            )
            _write_sidecars(card, sidecars=sidecars, summary=summary, frontier=frontier)
            for rel_path in sidecars.values():
                path = card / rel_path
                if path.exists() and path.suffix != ".md":
                    recorder.log_artifact(path.as_posix(), artifact_path=spec.experiment)
            _remove_qlib_code_snapshot_artifacts(Path(recorder.get_local_dir()).resolve())
    _write_summary_files(card, summary)
    return summary


def _load_provider_frame(
    spec: DynamicSoxxSoxlCashAllocationWfV2Spec,
    *,
    common_end: str,
    feature_sets: dict[str, list[str]],
) -> pd.DataFrame:
    config = QlibProviderConfig(provider_uri=spec.provider_uri, market="all", region="us", kernels=1)
    init_qlib_provider(config)
    from qlib.data import D

    feature_cols = list(dict.fromkeys(column for cols in feature_sets.values() for column in cols))
    fields = [
        f"${ETF_RETURN_COL}",
        f"${UNDERLYING_RETURN_COL}",
        "$close",
        "$underlying_close",
        *[f"${column}" for column in feature_cols],
    ]
    qframe = D.features(list(PAIR_UNIVERSE), fields, start_time=spec.train_start, end_time=common_end, freq="day")
    out = qframe.reset_index().rename(columns={field: field.removeprefix("$") for field in fields})
    out["datetime"] = pd.to_datetime(out["datetime"]).dt.normalize()
    out["instrument"] = out["instrument"].astype(str).str.upper()
    for column in [ETF_RETURN_COL, UNDERLYING_RETURN_COL, "close", "underlying_close", *feature_cols]:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return out.dropna(subset=[ETF_RETURN_COL, UNDERLYING_RETURN_COL]).sort_values(
        ["datetime", "instrument"]
    ).reset_index(drop=True)


def _benchmark_metrics(frame: pd.DataFrame, spec: DynamicSoxxSoxlCashAllocationWfV2Spec) -> dict[str, float]:
    test = _calibration_test_frame(frame, spec)
    returns = pd.to_numeric(test[UNDERLYING_RETURN_COL], errors="coerce").fillna(0.0).to_numpy(dtype="float64")
    ret, maxdd = _return_drawdown(returns)
    return {
        "asset": "SOXX",
        "return_pct": ret,
        "max_drawdown_pct": maxdd,
        "start": pd.Timestamp(test["datetime"].min()).date().isoformat(),
        "end": pd.Timestamp(test["datetime"].max()).date().isoformat(),
        "rows": int(len(test)),
    }


def _calibration_test_frame(frame: pd.DataFrame, spec: DynamicSoxxSoxlCashAllocationWfV2Spec) -> pd.DataFrame:
    return frame.loc[
        frame["instrument"].eq(CALIBRATION_ASSET) & frame["datetime"].dt.year.isin([int(year) for year in spec.test_years])
    ].copy()


def _direct_action_frontier(
    frame: pd.DataFrame,
    *,
    spec: DynamicSoxxSoxlCashAllocationWfV2Spec,
    feature_sets: dict[str, list[str]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    feature_names = [name for name in ("base", "base_soxx_diffusion") if name in feature_sets]
    actions = _action_grid()
    label_col = "direct_action_label"
    data = frame.copy()
    data[label_col] = _direct_action_labels(data, actions=actions, downside_penalty=0.0)
    policy_variants = [
        {"variant": "aggressive_no_stop", "gamma": 3.0, "scale": 1.0, "stop_thr": -1.0, "stop_mult": 1.0},
        {"variant": "medium_no_stop", "gamma": 2.0, "scale": 1.0, "stop_thr": -1.0, "stop_mult": 1.0},
        {"variant": "dd_safe_fast_stop", "gamma": 3.0, "scale": 0.80, "stop_thr": -0.04, "stop_mult": 0.10},
        {"variant": "dd_safe_early_stop", "gamma": 3.0, "scale": 0.75, "stop_thr": -0.02, "stop_mult": 0.20},
        {"variant": "low_risk_scale", "gamma": 1.0, "scale": 0.65, "stop_thr": -0.03, "stop_mult": 0.35},
    ]
    for feature_set in feature_names:
        feature_cols = feature_sets[feature_set]
        for model_name in ("lgbm", "xgb"):
            fold_predictions: dict[int, tuple[pd.DataFrame, pd.Series, pd.Series]] = {}
            annual_cache: dict[tuple[int, str], dict[str, Any]] = {}
            for year in spec.test_years:
                train = data.loc[data["datetime"].dt.year.lt(int(year))].dropna(subset=[label_col]).copy()
                test = data.loc[
                    data["instrument"].eq(CALIBRATION_ASSET) & data["datetime"].dt.year.eq(int(year))
                ].copy()
                if len(train) < 300 or len(test) < 40:
                    continue
                encoder, y_encoded = _encode_action_labels(train[label_col])
                estimator = _build_direct_action_classifier(model_name, class_count=len(encoder.classes_))
                estimator.fit(train[feature_cols].astype("float64"), y_encoded)
                weights_by_gamma = {
                    float(variant["gamma"]): _predict_action_weights(
                        estimator,
                        encoder,
                        test[feature_cols].astype("float64"),
                        actions=actions,
                        gamma=float(variant["gamma"]),
                    )
                    for variant in policy_variants
                }
                for gamma, weights in weights_by_gamma.items():
                    fold_predictions[(int(year), str(gamma))] = (test, weights[0], weights[1])
            for variant in policy_variants:
                parts: list[np.ndarray] = []
                annual_rows: list[dict[str, Any]] = []
                avg_soxx: list[float] = []
                avg_soxl: list[float] = []
                turnovers: list[float] = []
                for year in spec.test_years:
                    key = (int(year), str(float(variant["gamma"])))
                    if key not in fold_predictions:
                        continue
                    test, soxx_weight, soxl_weight = fold_predictions[key]
                    path = _simulate_path(
                        test,
                        soxx_weight.to_numpy(dtype="float64"),
                        soxl_weight.to_numpy(dtype="float64"),
                        scale=float(variant["scale"]),
                        stop_thr=float(variant["stop_thr"]),
                        stop_mult=float(variant["stop_mult"]),
                        cost_per_turnover=spec.cost_per_turnover,
                    )
                    annual = _annual_row(
                        year=int(year),
                        returns=path["returns"],
                        soxx_weight=path["soxx_weight"],
                        soxl_weight=path["soxl_weight"],
                        turnover=path["turnover"],
                        benchmark_returns=test[UNDERLYING_RETURN_COL].to_numpy(dtype="float64"),
                    )
                    annual_cache[(int(year), str(variant["variant"]))] = annual
                    annual_rows.append(annual)
                    parts.append(path["returns"])
                    avg_soxx.append(float(np.mean(path["soxx_weight"])))
                    avg_soxl.append(float(np.mean(path["soxl_weight"])))
                    turnovers.append(float(np.sum(path["turnover"])))
                if not parts:
                    continue
                returns = np.concatenate(parts)
                ret, maxdd = _return_drawdown(returns)
                row = _frontier_row(
                    branch="direct_action_lgb_xgb",
                    candidate=f"{model_name}_{feature_set}_{variant['variant']}",
                    model=model_name,
                    feature_set=feature_set,
                    return_pct=ret,
                    max_drawdown_pct=maxdd,
                    avg_soxx_weight=float(np.mean(avg_soxx)),
                    avg_soxl_weight=float(np.mean(avg_soxl)),
                    turnover=float(np.sum(turnovers)),
                    selection_uses_outer_test=False,
                    notes="direct_action_label_learned_on_prior_years_only",
                )
                row.update(
                    {
                        "variant": str(variant["variant"]),
                        "gamma": float(variant["gamma"]),
                        "scale": float(variant["scale"]),
                        "stop_thr": float(variant["stop_thr"]),
                        "stop_mult": float(variant["stop_mult"]),
                        "annual_pass_years": int(
                            sum(
                                annual["excess_vs_soxx_pct"] >= SOXX_EXCESS_FLOOR_PCT
                                and annual["max_drawdown_pct"] >= MAX_DRAWDOWN_FLOOR_PCT
                                for annual in annual_rows
                            )
                        ),
                        "min_annual_excess_pct": float(min(annual["excess_vs_soxx_pct"] for annual in annual_rows)),
                        "worst_annual_drawdown_pct": float(min(annual["max_drawdown_pct"] for annual in annual_rows)),
                    }
                )
                rows.append(row)
    return pd.DataFrame(rows)


def _trend_rule_frontier(frame: pd.DataFrame, *, spec: DynamicSoxxSoxlCashAllocationWfV2Spec) -> pd.DataFrame:
    test = _calibration_test_frame(frame, spec).sort_values("datetime").reset_index(drop=True)
    work = test.copy()
    underlying_close = pd.to_numeric(work["underlying_close"], errors="coerce")
    for window in (50, 100, 150, 200):
        work[f"ma{window}"] = underlying_close / underlying_close.rolling(window, min_periods=max(5, window // 3)).mean() - 1.0
    for window in (20, 60, 120):
        work[f"mom{window}"] = underlying_close / underlying_close.shift(window) - 1.0
    etf_returns = work[ETF_RETURN_COL].to_numpy(dtype="float64")
    underlying_returns = work[UNDERLYING_RETURN_COL].to_numpy(dtype="float64")
    base_rows: list[dict[str, Any]] = []
    for trend_col in ("ma100", "ma150", "ma200"):
        trend_values = work[trend_col].to_numpy(dtype="float64")
        for trend_thr in (0.0, 0.03, 0.06):
            for mom_col in ("mom60", "mom120"):
                mom_values = work[mom_col].to_numpy(dtype="float64")
                for mom_thr in (-0.03, 0.0, 0.03):
                    for dd_thr in (-0.25, -0.18):
                        for vol_thr in (0.045, 0.060, 999.0):
                            risk = (
                                (trend_values > trend_thr)
                                & (mom_values > mom_thr)
                                & (work["underlying_drawdown_60"].to_numpy(dtype="float64") > dd_thr)
                                & (work["underlying_realized_vol_20"].to_numpy(dtype="float64") < vol_thr)
                            )
                            strong = (
                                risk
                                & (mom_values > mom_thr + 0.04)
                                & (trend_values > trend_thr + 0.02)
                                & (work["vix_z_60"].to_numpy(dtype="float64") < 1.0)
                            )
                            for core in (0.6, 0.8):
                                for overlay in (0.4, 0.6, 0.8):
                                    soxx = np.where(risk, core, 0.0).astype("float64")
                                    soxl = np.where(strong, overlay, 0.0).astype("float64")
                                    soxx, soxl = _normalize_weights(soxx, soxl)
                                    path = _simulate_path(
                                        work,
                                        soxx,
                                        soxl,
                                        scale=1.0,
                                        stop_thr=-1.0,
                                        stop_mult=1.0,
                                        cost_per_turnover=spec.cost_per_turnover,
                                    )
                                    ret, maxdd = _return_drawdown(path["returns"])
                                    base_rows.append(
                                        {
                                            "candidate": (
                                                f"{trend_col}_{trend_thr}_{mom_col}_{mom_thr}_"
                                                f"{dd_thr}_{vol_thr}_{core}_{overlay}"
                                            ),
                                            "trend_col": trend_col,
                                            "trend_thr": trend_thr,
                                            "mom_col": mom_col,
                                            "mom_thr": mom_thr,
                                            "dd_thr": dd_thr,
                                            "vol_thr": vol_thr,
                                            "core": core,
                                            "overlay": overlay,
                                            "base_return_pct": ret,
                                            "base_max_drawdown_pct": maxdd,
                                            "base_soxx_weight": soxx,
                                            "base_soxl_weight": soxl,
                                        }
                                    )
    base_frame = pd.DataFrame([{k: v for k, v in row.items() if not k.endswith("_weight")} for row in base_rows])
    benchmark = _benchmark_metrics(frame, spec)
    top_return = base_frame.sort_values("base_return_pct", ascending=False).head(20)["candidate"].tolist()
    top_shallow_excess = (
        base_frame.loc[base_frame["base_return_pct"].ge(float(benchmark["return_pct"]) + SOXX_EXCESS_FLOOR_PCT)]
        .sort_values("base_max_drawdown_pct", ascending=False)
        .head(20)["candidate"]
        .tolist()
    )
    selected = []
    selected_ids = set(top_return + top_shallow_excess)
    for row in base_rows:
        if row["candidate"] in selected_ids:
            selected.append(row)
    variants = [
        {"variant": "no_stop", "scale": 1.0, "stop_thr": -1.0, "stop_mult": 1.0},
        {"variant": "fast_stop", "scale": 0.65, "stop_thr": -0.03, "stop_mult": 0.35},
        {"variant": "early_stop", "scale": 0.75, "stop_thr": -0.02, "stop_mult": 0.20},
        {"variant": "deep_stop", "scale": 0.75, "stop_thr": -0.05, "stop_mult": 0.10},
    ]
    rows: list[dict[str, Any]] = []
    for base in selected:
        for variant in variants:
            path = _simulate_path(
                work,
                base["base_soxx_weight"],
                base["base_soxl_weight"],
                scale=float(variant["scale"]),
                stop_thr=float(variant["stop_thr"]),
                stop_mult=float(variant["stop_mult"]),
                cost_per_turnover=spec.cost_per_turnover,
            )
            ret, maxdd = _return_drawdown(path["returns"])
            row = _frontier_row(
                branch="trend_momentum_oracle_frontier",
                candidate=f"{base['candidate']}_{variant['variant']}",
                model="rule_oracle_parameter_probe",
                feature_set="daily_trend_momentum_vol",
                return_pct=ret,
                max_drawdown_pct=maxdd,
                avg_soxx_weight=float(np.mean(path["soxx_weight"])),
                avg_soxl_weight=float(np.mean(path["soxl_weight"])),
                turnover=float(np.sum(path["turnover"])),
                selection_uses_outer_test=True,
                notes="oracle_frontier_uses_2023_2026_to_bound_feasibility_not_promotable",
            )
            row.update({k: base[k] for k in ("trend_col", "trend_thr", "mom_col", "mom_thr", "dd_thr", "vol_thr", "core", "overlay")})
            row.update(variant)
            rows.append(row)
    return pd.DataFrame(rows)


def _action_grid() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"action_id": 0, "name": "cash", "weight_soxx": 0.0, "weight_soxl": 0.0},
            {"action_id": 1, "name": "half_soxx", "weight_soxx": 0.5, "weight_soxl": 0.0},
            {"action_id": 2, "name": "full_soxx", "weight_soxx": 1.0, "weight_soxl": 0.0},
            {"action_id": 3, "name": "core_plus_small_soxl", "weight_soxx": 0.70, "weight_soxl": 0.15},
            {"action_id": 4, "name": "core_plus_mid_soxl", "weight_soxx": 0.55, "weight_soxl": 0.30},
            {"action_id": 5, "name": "barbell_soxl", "weight_soxx": 0.35, "weight_soxl": 0.50},
            {"action_id": 6, "name": "soxl_heavy", "weight_soxx": 0.0, "weight_soxl": 0.75},
        ]
    )


def _direct_action_labels(frame: pd.DataFrame, *, actions: pd.DataFrame, downside_penalty: float) -> np.ndarray:
    etf_return = pd.to_numeric(frame[ETF_RETURN_COL], errors="coerce").fillna(0.0).to_numpy(dtype="float64")
    underlying_return = pd.to_numeric(frame[UNDERLYING_RETURN_COL], errors="coerce").fillna(0.0).to_numpy(dtype="float64")
    utilities = []
    for action in actions.itertuples(index=False):
        action_return = float(action.weight_soxx) * underlying_return + float(action.weight_soxl) * etf_return
        utilities.append(action_return - float(downside_penalty) * np.square(np.minimum(action_return + 0.015, 0.0)))
    utility_matrix = np.vstack(utilities).T
    best_idx = np.nanargmax(utility_matrix, axis=1)
    return actions["action_id"].to_numpy(dtype=int)[best_idx]


def _encode_action_labels(labels: pd.Series | np.ndarray) -> tuple[LabelEncoder, np.ndarray]:
    encoder = LabelEncoder()
    encoded = encoder.fit_transform(np.asarray(labels, dtype=int))
    return encoder, encoded.astype(int)


def _build_direct_action_classifier(model: str, *, class_count: int):
    if model == "lgbm":
        from lightgbm import LGBMClassifier

        return make_pipeline(
            SimpleImputer(strategy="median"),
            LGBMClassifier(
                n_estimators=60,
                learning_rate=0.035,
                num_leaves=15,
                min_child_samples=35,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.15,
                reg_lambda=0.8,
                random_state=75,
                verbose=-1,
            ),
        )
    if model == "xgb":
        from xgboost import XGBClassifier

        return make_pipeline(
            SimpleImputer(strategy="median"),
            XGBClassifier(
                n_estimators=70,
                max_depth=3,
                learning_rate=0.035,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.2,
                objective="multi:softprob",
                eval_metric="mlogloss",
                num_class=int(class_count),
                random_state=76,
                n_jobs=1,
            ),
        )
    raise ValueError(f"unsupported direct action model: {model}")


def _predict_action_weights(
    estimator: Any,
    encoder: LabelEncoder,
    features: pd.DataFrame,
    *,
    actions: pd.DataFrame,
    gamma: float,
) -> tuple[pd.Series, pd.Series]:
    probabilities = np.asarray(estimator.predict_proba(features), dtype="float64")
    if gamma != 1.0:
        probabilities = np.power(np.clip(probabilities, 1e-9, 1.0), float(gamma))
        probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    action_ids = encoder.inverse_transform(np.arange(len(encoder.classes_)))
    soxx_weight = np.zeros(len(features), dtype="float64")
    soxl_weight = np.zeros(len(features), dtype="float64")
    for idx, action_id in enumerate(action_ids):
        action = actions.loc[actions["action_id"].eq(int(action_id))].iloc[0]
        soxx_weight += probabilities[:, idx] * float(action["weight_soxx"])
        soxl_weight += probabilities[:, idx] * float(action["weight_soxl"])
    return pd.Series(soxx_weight, index=features.index), pd.Series(soxl_weight, index=features.index)


def _simulate_path(
    frame: pd.DataFrame,
    soxx_weight: np.ndarray,
    soxl_weight: np.ndarray,
    *,
    scale: float,
    stop_thr: float,
    stop_mult: float,
    cost_per_turnover: float,
) -> dict[str, np.ndarray]:
    underlying_return = pd.to_numeric(frame[UNDERLYING_RETURN_COL], errors="coerce").fillna(0.0).to_numpy(dtype="float64")
    etf_return = pd.to_numeric(frame[ETF_RETURN_COL], errors="coerce").fillna(0.0).to_numpy(dtype="float64")
    returns: list[float] = []
    soxx_path: list[float] = []
    soxl_path: list[float] = []
    turnover_path: list[float] = []
    equity = 1.0
    peak = 1.0
    prev_soxx = 0.0
    prev_soxl = 0.0
    for idx in range(len(frame)):
        risk_mult = float(stop_mult) if equity / peak - 1.0 <= float(stop_thr) else 1.0
        soxx = float(soxx_weight[idx]) * float(scale) * risk_mult
        soxl = float(soxl_weight[idx]) * float(scale) * risk_mult
        total = soxx + soxl
        if total > 1.0:
            soxx /= total
            soxl /= total
        turnover = abs(soxx - prev_soxx) + abs(soxl - prev_soxl)
        ret = soxx * float(underlying_return[idx]) + soxl * float(etf_return[idx]) - turnover * float(cost_per_turnover)
        returns.append(ret)
        soxx_path.append(soxx)
        soxl_path.append(soxl)
        turnover_path.append(turnover)
        equity *= 1.0 + ret
        peak = max(peak, equity)
        prev_soxx = soxx
        prev_soxl = soxl
    return {
        "returns": np.asarray(returns, dtype="float64"),
        "soxx_weight": np.asarray(soxx_path, dtype="float64"),
        "soxl_weight": np.asarray(soxl_path, dtype="float64"),
        "turnover": np.asarray(turnover_path, dtype="float64"),
    }


def _normalize_weights(soxx_weight: np.ndarray, soxl_weight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    soxx = np.asarray(soxx_weight, dtype="float64").copy()
    soxl = np.asarray(soxl_weight, dtype="float64").copy()
    total = soxx + soxl
    over = total > 1.0
    soxx[over] = soxx[over] / total[over]
    soxl[over] = soxl[over] / total[over]
    return soxx, soxl


def _return_drawdown(returns: np.ndarray) -> tuple[float, float]:
    clean = np.nan_to_num(np.asarray(returns, dtype="float64"), nan=0.0, posinf=0.0, neginf=0.0)
    if clean.size == 0:
        return 0.0, 0.0
    equity = np.cumprod(1.0 + clean)
    peak = np.maximum.accumulate(equity)
    drawdown = equity / peak - 1.0
    return float((equity[-1] - 1.0) * 100.0), float(drawdown.min() * 100.0)


def _annual_row(
    *,
    year: int,
    returns: np.ndarray,
    soxx_weight: np.ndarray,
    soxl_weight: np.ndarray,
    turnover: np.ndarray,
    benchmark_returns: np.ndarray,
) -> dict[str, Any]:
    ret, maxdd = _return_drawdown(returns)
    bench_ret, bench_dd = _return_drawdown(benchmark_returns)
    return {
        "year": int(year),
        "return_pct": ret,
        "max_drawdown_pct": maxdd,
        "benchmark_return_pct": bench_ret,
        "benchmark_max_drawdown_pct": bench_dd,
        "excess_vs_soxx_pct": ret - bench_ret,
        "avg_soxx_weight": float(np.mean(soxx_weight)) if len(soxx_weight) else 0.0,
        "avg_soxl_weight": float(np.mean(soxl_weight)) if len(soxl_weight) else 0.0,
        "turnover": float(np.sum(turnover)) if len(turnover) else 0.0,
    }


def _frontier_row(
    *,
    branch: str,
    candidate: str,
    model: str,
    feature_set: str,
    return_pct: float,
    max_drawdown_pct: float,
    avg_soxx_weight: float,
    avg_soxl_weight: float,
    turnover: float,
    selection_uses_outer_test: bool,
    notes: str,
) -> dict[str, Any]:
    return {
        "branch": branch,
        "candidate": candidate,
        "model": model,
        "feature_set": feature_set,
        "return_pct": float(return_pct),
        "max_drawdown_pct": float(max_drawdown_pct),
        "avg_soxx_weight": float(avg_soxx_weight),
        "avg_soxl_weight": float(avg_soxl_weight),
        "avg_cash_weight": float(max(0.0, 1.0 - avg_soxx_weight - avg_soxl_weight)),
        "turnover": float(turnover),
        "selection_uses_outer_test": bool(selection_uses_outer_test),
        "notes": notes,
    }


def _stitched_goal_summary(frontier: pd.DataFrame, *, benchmark: dict[str, float]) -> dict[str, Any]:
    rows = frontier.copy()
    if rows.empty:
        return {
            "policy": "stitched_2023_2026_soxx_plus_20_maxdd_10_gate",
            "required_policy_satisfied": False,
            "passes_target_count": 0,
            "blocking_reasons": ["no_frontier_rows"],
        }
    rows["excess_vs_soxx_pct"] = pd.to_numeric(rows["return_pct"], errors="coerce") - float(benchmark["return_pct"])
    rows["passes_excess_gate"] = rows["excess_vs_soxx_pct"].ge(SOXX_EXCESS_FLOOR_PCT)
    rows["passes_drawdown_gate"] = pd.to_numeric(rows["max_drawdown_pct"], errors="coerce").ge(MAX_DRAWDOWN_FLOOR_PCT)
    rows["passes_target"] = rows["passes_excess_gate"] & rows["passes_drawdown_gate"]
    dd_safe = rows.loc[rows["passes_drawdown_gate"]].copy()
    excess_safe = rows.loc[rows["passes_excess_gate"]].copy()
    best_dd_safe_return = float(dd_safe["return_pct"].max()) if not dd_safe.empty else float("nan")
    min_dd_if_excess = float(excess_safe["max_drawdown_pct"].max()) if not excess_safe.empty else float("nan")
    best_return = rows.sort_values("return_pct", ascending=False).iloc[0].to_dict()
    blocking = []
    if not bool(rows["passes_excess_gate"].any()):
        blocking.append("no_candidate_reaches_soxx_plus_20")
    if not bool(rows["passes_target"].any()):
        blocking.append("no_candidate_combines_soxx_plus_20_with_10pct_drawdown")
    return {
        "policy": "stitched_2023_2026_soxx_plus_20_maxdd_10_gate",
        "benchmark_return_pct": float(benchmark["return_pct"]),
        "benchmark_max_drawdown_pct": float(benchmark["max_drawdown_pct"]),
        "excess_return_floor_pct": SOXX_EXCESS_FLOOR_PCT,
        "max_drawdown_floor_pct": MAX_DRAWDOWN_FLOOR_PCT,
        "passes_target_count": int(rows["passes_target"].sum()),
        "candidate_count": int(len(rows)),
        "required_policy_satisfied": bool(rows["passes_target"].any()),
        "best_drawdown_safe_return_pct": best_dd_safe_return,
        "best_drawdown_safe_excess_pct": best_dd_safe_return - float(benchmark["return_pct"]) if np.isfinite(best_dd_safe_return) else float("nan"),
        "min_drawdown_if_excess_passes_pct": min_dd_if_excess,
        "best_return_candidate": _json_ready(best_return),
        "blocking_reasons": blocking or ["ok"],
    }


def _annual_goal_summary(frontier: pd.DataFrame, *, required_years: tuple[int, ...]) -> dict[str, Any]:
    return {
        "policy": "annual_gate_reported_by_v1_and_direct_action_rows",
        "required_test_years": [int(year) for year in required_years],
        "return_gate": "each_test_year_return_pct_ge_soxx_buyhold_return_pct_plus_20pp",
        "max_drawdown_floor_pct": MAX_DRAWDOWN_FLOOR_PCT,
        "required_policy_satisfied": False,
        "blocking_reasons": ["annual_gate_not_satisfied_by_frontier_or_v1"],
    }


def _intraday_context_summary(spec: DynamicSoxxSoxlCashAllocationWfV2Spec) -> dict[str, Any]:
    summary_path = spec.intraday_15m_card_dir / "summary.json"
    if not summary_path.exists():
        return {"status": "missing", "role": "not_used"}
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    selected = payload.get("selected_model", {})
    return {
        "status": "ok",
        "source_card": str(spec.intraday_15m_card_dir.relative_to(spec.root)),
        "role": "candidate_intraday_execution_sleeve_not_core_allocation_alpha",
        "promotion_decision": payload.get("promotion_decision"),
        "selected_model": selected.get("model"),
        "test_return_pct": selected.get("test_net_return_pct"),
        "test_max_drawdown_pct": selected.get("test_max_drawdown_pct"),
        "test_ic": selected.get("test_ic"),
        "limitation": "intraday returns are small and replay/slippage proof is missing; not enough to close daily allocation target gap",
    }


def _summary_payload(
    *,
    spec: DynamicSoxxSoxlCashAllocationWfV2Spec,
    common_end: str,
    recorder_id: str,
    recorder_uri: Path,
    recorder_local_dir: Path,
    sidecars: dict[str, str],
    source_summary: dict[str, Any],
    benchmark: dict[str, float],
    frontier: pd.DataFrame,
    direct_frontier: pd.DataFrame,
    trend_frontier: pd.DataFrame,
    stitched_goal: dict[str, Any],
    annual_goal: dict[str, Any],
    intraday_summary: dict[str, Any],
    promotion_decision: str,
) -> dict[str, Any]:
    recorder_artifacts = [
        f"artifacts/{spec.experiment}/frontier_diagnostics.csv",
        f"artifacts/{spec.experiment}/analysis_summary.json",
        f"artifacts/{spec.experiment}/frontier_diagnostics.csv",
        "meta.yaml",
    ]
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
            "stitched_goal": stitched_goal,
            "annual_goal": annual_goal,
            "frontier": {
                "candidate_count": int(len(frontier)),
                "direct_action_candidate_count": int(len(direct_frontier)),
                "trend_oracle_candidate_count": int(len(trend_frontier)),
                "uses_lgb_xgb": True,
                "contains_soxx_constituent_feature_probe": "base_soxx_diffusion" in set(direct_frontier.get("feature_set", [])),
                "selection_uses_outer_test": bool(frontier.get("selection_uses_outer_test", pd.Series([False])).any()),
            },
            "model_selection": {
                "source": "diagnostic_frontier",
                "selection_uses_outer_test": False,
                "model_families": ["lgbm_classifier", "xgb_classifier", "lgb_xgb_consensus_context"],
                "direct_action_label": "argmax_next_open_to_open_action_utility_trained_only_on_prior_years",
                "manual_template_selection": False,
                "promotion_gate": "stitched_and_annual_goal_must_pass",
            },
            "intraday_execution_context": intraday_summary,
            "source_v1": {
                "promotion_decision": source_summary.get("promotion_decision"),
                "annual_goal_satisfied": source_summary.get("annual_goal", {}).get("required_policy_satisfied"),
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
                "recorder_only_artifacts": [f"artifacts/{spec.experiment}/frontier_diagnostics.csv"],
            },
            "workflow_config": "workflow.yaml",
            "recorder_id": recorder_id,
            "recorder_uri": recorder_uri.resolve().as_posix(),
            "recorder_local_dir": recorder_local_dir.as_posix(),
            "recorder_artifacts": recorder_artifacts,
        }
    )


def _write_sidecars(card: Path, *, sidecars: dict[str, str], summary: dict[str, Any], frontier: pd.DataFrame) -> None:
    frontier.to_csv(card / sidecars["frontier_diagnostics"], index=False)
    model_comparison = (
        frontier.groupby(["branch", "model", "feature_set"], dropna=False)
        .agg(
            candidate_count=("candidate", "size"),
            best_return_pct=("return_pct", "max"),
            best_max_drawdown_pct=("max_drawdown_pct", "max"),
            avg_return_pct=("return_pct", "mean"),
        )
        .reset_index()
    )
    model_comparison.to_csv(card / sidecars["model_comparison"], index=False)
    pd.DataFrame(
        [
            {
                "check": "stitched_goal",
                "status": "pass" if summary["stitched_goal"]["required_policy_satisfied"] else "fail",
                "details": json.dumps(summary["stitched_goal"], ensure_ascii=False, sort_keys=True),
            },
            {
                "check": "annual_goal",
                "status": "pass" if summary["annual_goal"]["required_policy_satisfied"] else "fail",
                "details": json.dumps(summary["annual_goal"], ensure_ascii=False, sort_keys=True),
            },
            {
                "check": "intraday_execution_context",
                "status": summary["intraday_execution_context"]["status"],
                "details": json.dumps(summary["intraday_execution_context"], ensure_ascii=False, sort_keys=True),
            },
        ]
    ).to_csv(card / sidecars["risk_analysis"], index=False)
    selected = {
        "selected_model": None,
        "promotion_decision": summary["promotion_decision"],
        "reason": "no frontier candidate satisfies SOXX plus 20 percentage points with max drawdown above -10%",
        "best_return_candidate": summary["stitched_goal"].get("best_return_candidate"),
    }
    (card / sidecars["selected_model"]).write_text(json.dumps(_json_ready(selected), ensure_ascii=False, indent=2), encoding="utf-8")
    analysis = {
        "source": "qlib_recorder_frontier_diagnostics",
        "recorder_id": summary["recorder_id"],
        "stitched_goal": summary["stitched_goal"],
        "annual_goal": summary["annual_goal"],
        "promotion_decision": summary["promotion_decision"],
    }
    (card / sidecars["analysis_summary"]).write_text(json.dumps(_json_ready(analysis), ensure_ascii=False, indent=2), encoding="utf-8")
    (card / sidecars["failure_modes"]).write_text(_failure_modes_text(summary), encoding="utf-8")


def _write_summary_files(card: Path, summary: dict[str, Any]) -> None:
    (card / "summary.json").write_text(json.dumps(_json_ready(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    (card / "conclusion.md").write_text(_conclusion_text(summary), encoding="utf-8")
    (card / "workflow.yaml").write_text(_workflow_yaml(summary), encoding="utf-8")


def _failure_modes_text(summary: dict[str, Any]) -> str:
    return (
        "# Failure Modes\n\n"
        "- Direct-action LGB/XGB can raise SOXL participation, but drawdown expands before the state stop can react.\n"
        "- Trend/momentum oracle rules can exceed SOXX only with drawdown far below the -10% target.\n"
        "- Drawdown-safe variants become mostly cash or tiny SOXL sleeves and under-participate in SOXX bull windows.\n"
        "- 15min intraday model remains an execution sleeve candidate, not enough core return to solve the allocation target.\n"
        f"- Current promotion decision: {summary['promotion_decision']}.\n"
    )


def _conclusion_text(summary: dict[str, Any]) -> str:
    goal = summary["stitched_goal"]
    intraday = summary["intraday_execution_context"]
    return (
        "# 中文结论\n\n"
        "- 做了什么：重跑 SOXX/SOXL/cash 动态配置 v2 边界诊断，包含 direct-action LGB/XGB、base_soxx_diffusion 成分股特征探测，以及趋势/动量规则上限探测。\n"
        f"- 核心指标：SOXX benchmark stitched return={goal['benchmark_return_pct']:.4f}%，max drawdown={goal['benchmark_max_drawdown_pct']:.4f}%；"
        f"满足 SOXX+20 且回撤不低于 -10% 的候选数={goal['passes_target_count']}。\n"
        f"- 风险边界：回撤安全候选的最高收益={goal['best_drawdown_safe_return_pct']:.4f}%，"
        f"收益达标候选中的最浅回撤={goal['min_drawdown_if_excess_passes_pct']:.4f}%。\n"
        f"- 日内数据：15min execution sleeve 状态={intraday.get('status')}，test_return={intraday.get('test_return_pct')}，"
        f"test_dd={intraday.get('test_max_drawdown_pct')}，role={intraday.get('role')}。\n"
        f"- 是否晋升：不晋升；decision={summary['promotion_decision']}。当前证据说明目标缺口来自收益/回撤不可同时满足，而不是单一参数未调。\n"
        "- 下一步：若继续追求该目标，需要新增可提前识别半导体拥挤去杠杆的外部/实时风险数据，或放宽回撤目标；继续调日线仓位模板意义有限。\n"
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
            "  source: diagnostic_frontier",
            "  selection_uses_outer_test: false",
            "artifact_policy:",
            "  policy: slim_card_recorder_first",
            "  workflow_config: workflow.yaml",
            f"promotion_decision: {summary['promotion_decision']}",
            "",
        ]
    )
