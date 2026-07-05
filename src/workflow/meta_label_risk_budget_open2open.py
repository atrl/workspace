from __future__ import annotations

import json
import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier

from src.backtest.ml_position import _build_model, _max_drawdown_pct, _positive_probability, _tail_risk_metrics
from src.data import QlibProviderConfig
from src.factor import build_intraday_15m_factor_frame
from src.workflow.cognitive_alpha_goal_codex_exec import (
    BASE_CONSTRAINED_REF_COL,
    EXECUTION_TIMING,
    LABEL_OPEN_TO_OPEN_5D_COL,
    OPEN_TO_OPEN_RETURN_COL,
    QLIB_NATIVE_API,
    _attach_constituent_intraday_combo_features,
    _attach_intraday_15m_model_features,
    _attach_open_to_open_execution_context,
    _dynamic_open_to_open_benchmarks,
    _ensure_intraday_15m_qlib_provider,
    _intraday_15m_end_time,
)
from src.workflow.risk_hardened_open2open import (
    _baseline_risk_fallback_payload,
    _build_frame_bundle as _build_risk_hardened_frame_bundle,
    _feature_groups as _risk_hardened_feature_groups,
    _float_or_none,
    _load_previous_model_return_floor,
    _previous_model_return_floor_payload,
    _run_execution_stress,
)


EXPERIMENT = "meta_label_risk_budget_open2open_v1"
CARD_TYPE = "meta_label_risk_budget_open2open_research_card"
OPPORTUNITY_SIDE_COL = "meta_opportunity_side"
TAKE_TRADE_LABEL_COL = "take_trade_prob_label"
TAIL_LOSS_LABEL_COL = "tail_loss_prob_label"
INTRADAY_EXECUTION_CAP_COL = "intraday_execution_cap"


@dataclass(frozen=True)
class MetaLabelRiskBudgetOpen2OpenSpec:
    root: Path
    experiment: str = EXPERIMENT
    previous_model_return_floor_card: str = "cognitive_alpha_goal_codex_exec_v1"
    train: tuple[str, str] = ("2019-01-02", "2024-12-31")
    validation: tuple[str, str] = ("2025-01-02", "2025-12-31")
    test_start: str = "2026-01-02"
    opportunity_threshold: float = 0.01
    take_label_return_threshold: float = 0.0
    tail_loss_return_threshold: float = -0.05
    validation_max_drawdown_floor: float = -20.0
    validation_max_daily_loss_floor: float = -12.0
    validation_cvar_5pct_floor: float = -8.0
    validation_worst_5_daily_loss_floor: float = -6.0
    max_validation_underwater_day_ratio: float = 0.95
    test_max_drawdown_floor: float = -20.0
    test_max_daily_loss_floor: float = -12.0
    test_cvar_5pct_floor: float = -8.0
    test_worst_5_daily_loss_floor: float = -6.0
    max_test_underwater_day_ratio: float = 0.95
    max_validation_total_turnover: float = 20.0
    max_test_total_turnover: float = 20.0
    max_extreme_loss_day_count: int = 7
    extreme_loss_threshold: float = -0.05
    min_avg_abs_position: float = 0.01
    max_zero_position_day_ratio: float = 0.98
    cost_per_turnover: float = 0.0003

    @property
    def card_dir(self) -> Path:
        return self.root / "research" / self.experiment

    @property
    def provider_uri(self) -> Path:
        return self.root / "data" / "qlib" / "soxl"

    @property
    def soxx_constituent_provider_uri(self) -> Path:
        return self.root / "data" / "qlib" / "soxx_constituents"

    @property
    def intraday_15m_provider_uri(self) -> Path:
        return self.root / "data" / "qlib" / "soxl_intraday_15min"


@dataclass(frozen=True)
class _MetaFrameBundle:
    frame: pd.DataFrame
    manifest: dict[str, Any]
    provider_meta: dict[str, Any]
    as_of: str
    test: tuple[str, str]
    feature_cols: list[str]
    feature_groups: dict[str, list[str]]
    intraday_15m_factor_summary: dict[str, Any]
    constituent_factor_summary: dict[str, Any]


def run_meta_label_risk_budget_open2open_card(*, root: Path | str | None = None) -> dict[str, Any]:
    spec = MetaLabelRiskBudgetOpen2OpenSpec(root=_resolve_root(root))
    return _run_card(spec)


def evaluate_meta_label_risk_budget_candidates(
    frame: pd.DataFrame,
    *,
    feature_cols: list[str],
    train_window: tuple[str, str],
    validation_window: tuple[str, str],
    test_window: tuple[str, str],
    return_col: str = OPEN_TO_OPEN_RETURN_COL,
    label_col: str = LABEL_OPEN_TO_OPEN_5D_COL,
    base_position_col: str = BASE_CONSTRAINED_REF_COL,
    realized_vol_col: str = "realized_vol_20",
    datetime_col: str = "datetime",
    candidate_specs: list[dict[str, Any]] | None = None,
    cost_per_turnover: float = 0.0003,
    opportunity_threshold: float = 0.01,
    take_label_return_threshold: float = 0.0,
    tail_loss_return_threshold: float = -0.05,
    validation_gates: dict[str, Any] | None = None,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train LGB/XGB meta labels and evaluate risk-budget position candidates."""

    if frame.empty:
        raise ValueError("meta-label risk-budget frame is empty")
    if not feature_cols:
        raise ValueError("at least one meta-label feature is required")
    required = {datetime_col, return_col, label_col, base_position_col, realized_vol_col, *feature_cols}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"meta-label risk-budget frame missing columns: {missing}")

    data = _prepare_meta_frame(
        frame,
        datetime_col=datetime_col,
        return_col=return_col,
        label_col=label_col,
        base_position_col=base_position_col,
        opportunity_threshold=opportunity_threshold,
        take_label_return_threshold=take_label_return_threshold,
        tail_loss_return_threshold=tail_loss_return_threshold,
    )
    candidate_specs = candidate_specs or _candidate_grid()
    gates = validation_gates or {}

    train = _segment(data, train_window).dropna(subset=[label_col, return_col]).reset_index(drop=True)
    validation = _segment(data, validation_window).reset_index(drop=True)
    test = _segment(data, test_window).reset_index(drop=True)
    if train.empty or validation.empty or test.empty:
        raise ValueError("train, validation, and test windows must all contain rows")

    model_payload = _fit_meta_label_models(
        train,
        feature_cols,
        take_label_col=TAKE_TRADE_LABEL_COL,
        tail_label_col=TAIL_LOSS_LABEL_COL,
        opportunity_col=OPPORTUNITY_SIDE_COL,
        random_state=random_state,
    )
    metric_rows: list[dict[str, Any]] = []
    position_frames: list[pd.DataFrame] = []
    for segment_name, segment_frame in [("validation", validation), ("test", test)]:
        probabilities = _predict_meta_probabilities(model_payload, segment_frame, feature_cols)
        for model_key, probability_pair in probabilities.items():
            for candidate_spec in candidate_specs:
                candidate = _candidate_name(model_key, candidate_spec)
                metrics, positions = _simulate_meta_label_candidate(
                    segment_frame,
                    take_probability=probability_pair["take_probability"],
                    tail_probability=probability_pair["tail_probability"],
                    candidate=candidate,
                    segment=segment_name,
                    model_key=model_key,
                    position_spec=candidate_spec,
                    return_col=return_col,
                    base_position_col=base_position_col,
                    realized_vol_col=realized_vol_col,
                    cost_per_turnover=cost_per_turnover,
                    extreme_loss_threshold=float(gates.get("extreme_loss_threshold", -0.05)),
                )
                metrics.update(
                    {
                        "model": model_key,
                        "model_task": "dual_binary_meta_label",
                        "take_label_col": TAKE_TRADE_LABEL_COL,
                        "tail_label_col": TAIL_LOSS_LABEL_COL,
                        "return_col": return_col,
                        "model_target_col": label_col,
                        "position_architecture": "opportunity_side_times_edge_times_risk_budgets",
                        "selection_disabled": bool(model_key != "lgb_xgb_consensus"),
                        "selection_segment": "validation",
                        "report_segment": "test",
                        "train_start": pd.Timestamp(train["datetime"].min()).date().isoformat(),
                        "train_end": pd.Timestamp(train["datetime"].max()).date().isoformat(),
                        "validation_start": pd.Timestamp(validation["datetime"].min()).date().isoformat(),
                        "validation_end": pd.Timestamp(validation["datetime"].max()).date().isoformat(),
                        "test_start": pd.Timestamp(test["datetime"].min()).date().isoformat(),
                        "test_end": pd.Timestamp(test["datetime"].max()).date().isoformat(),
                        "params": json.dumps(candidate_spec, sort_keys=True),
                    }
                )
                metric_rows.append(metrics)
                position_frames.append(positions)

    metrics = pd.DataFrame(metric_rows)
    positions_frame = pd.concat(position_frames, ignore_index=True) if position_frames else pd.DataFrame()
    if metrics.empty or positions_frame.empty:
        raise ValueError("meta-label risk-budget evaluation produced no candidates")

    selected_candidate, selection_pool_reason, eligible_candidates = _select_candidate(metrics, gates)
    metrics["selection_eligible"] = metrics["candidate"].astype(str).isin(eligible_candidates).map(bool).astype(object)
    metrics["selection_gate"] = json.dumps({**gates, "selection_pool_reason": selection_pool_reason}, sort_keys=True)
    metrics["selected"] = metrics["candidate"].astype(str).eq(selected_candidate).map(bool).astype(object)
    positions_frame["selected"] = positions_frame["candidate"].astype(str).eq(selected_candidate)
    metrics = metrics.sort_values(
        ["selected", "segment", "selection_objective_score", "return_pct"],
        ascending=[False, True, False, False],
    ).reset_index(drop=True)
    label_diagnostics = _label_diagnostics(data, train_window, validation_window, test_window)
    feature_importance = _feature_importance_frame(model_payload)
    return metrics, positions_frame, label_diagnostics, feature_importance


def _resolve_root(root: Path | str | None) -> Path:
    if root is not None:
        return Path(root).resolve()
    current = Path.cwd().resolve()
    while current != current.parent:
        if (current / "AGENTS.md").exists() and (current / "SPEC.md").exists():
            return current
        current = current.parent
    raise ValueError("could not resolve repository root")


def _run_card(spec: MetaLabelRiskBudgetOpen2OpenSpec) -> dict[str, Any]:
    card = spec.card_dir
    card.mkdir(parents=True, exist_ok=True)
    bundle = _build_meta_frame_bundle(spec)
    benchmarks = _dynamic_open_to_open_benchmarks(bundle.frame, test=bundle.test)
    if not benchmarks:
        raise ValueError("meta-label risk-budget card requires dynamic open-to-open benchmarks")

    gates = _risk_gates_for_spec(spec)
    metrics, positions, label_diagnostics, feature_importance = evaluate_meta_label_risk_budget_candidates(
        bundle.frame,
        feature_cols=bundle.feature_cols,
        train_window=spec.train,
        validation_window=spec.validation,
        test_window=bundle.test,
        cost_per_turnover=spec.cost_per_turnover,
        opportunity_threshold=spec.opportunity_threshold,
        take_label_return_threshold=spec.take_label_return_threshold,
        tail_loss_return_threshold=spec.tail_loss_return_threshold,
        validation_gates=gates,
    )
    selected = _selected_payload(metrics)
    comparison = _benchmark_comparison(selected["test"], benchmarks=benchmarks)
    comparison_summary = _comparison_summary(selected["test"], comparison)
    walk_forward_metrics, walk_forward_positions = _run_walk_forward(spec, bundle)
    execution_stress, execution_summary = _run_execution_stress(bundle.frame, positions, root=spec.root)
    previous_model_floor = _load_previous_model_return_floor(spec)
    risk_analysis = _risk_analysis_payload(
        spec,
        selected,
        comparison_summary,
        walk_forward_metrics,
        execution_summary,
        previous_model_floor,
    )
    latest_signal = _latest_signal(positions, selected)
    _attach_open_to_open_execution_context({"selected_candidate": selected}, latest_signal)

    sidecars = {
        "selected_model": "selected_model.json",
        "model_selection": "model_selection.csv",
        "backtest_benchmark_comparison": "backtest_benchmark_comparison.csv",
        "backtest_walk_forward_metrics": "backtest_walk_forward_metrics.csv",
        "label_diagnostics": "label_diagnostics.csv",
        "feature_importance": "feature_importance.csv",
        "risk_analysis": "risk_analysis.csv",
        "execution_stress": "execution_stress.csv",
        "execution_stress_summary": "execution_stress_summary.json",
        "latest_signal": "latest_signal.json",
        "position_summary": "position_summary.json",
        "feature_sets": "feature_sets.csv",
        "failure_modes": "failure_modes.md",
    }

    import qlib
    from qlib.workflow import R

    _ = qlib
    recorder_uri = card / "qlib_recorder"
    recorder_uri.mkdir(parents=True, exist_ok=True)
    R.set_uri(recorder_uri.as_posix())

    with tempfile.TemporaryDirectory(prefix="meta_label_risk_budget_open2open_") as tmp:
        tmp_dir = Path(tmp)
        positions_path = tmp_dir / "positions.csv"
        walk_forward_positions_path = tmp_dir / "walk_forward_positions.csv"
        positions.to_csv(positions_path, index=False)
        walk_forward_positions.to_csv(walk_forward_positions_path, index=False)
        with R.start(experiment_name=spec.experiment):
            recorder = R.get_recorder()
            R.log_params(
                provider_uri=bundle.provider_meta["provider_uri"],
                market=bundle.manifest["market"],
                region=bundle.manifest["region"],
                instrument="SOXL",
                train_start=spec.train[0],
                train_end=spec.train[1],
                validation_start=spec.validation[0],
                validation_end=spec.validation[1],
                test_start=bundle.test[0],
                test_end=bundle.test[1],
                return_col=OPEN_TO_OPEN_RETURN_COL,
                label_col=LABEL_OPEN_TO_OPEN_5D_COL,
                take_label_col=TAKE_TRADE_LABEL_COL,
                tail_label_col=TAIL_LOSS_LABEL_COL,
                model_families="lgbm_classifier,xgb_classifier,lgb_xgb_consensus",
                architecture="base_opportunity_plus_meta_label_risk_budget",
                intraday_15m_provider_uri=bundle.intraday_15m_factor_summary["provider_uri"],
            )
            _write_sidecars(
                card,
                sidecars=sidecars,
                selected=selected,
                metrics=metrics,
                comparison=comparison,
                walk_forward_metrics=walk_forward_metrics,
                label_diagnostics=label_diagnostics,
                feature_importance=feature_importance,
                risk_analysis=risk_analysis,
                execution_stress=execution_stress,
                execution_summary=execution_summary,
                latest_signal=latest_signal,
                feature_groups=bundle.feature_groups,
            )
            recorder.log_artifact(positions_path.as_posix(), artifact_path="meta_label_risk_budget_open2open")
            recorder.log_artifact(
                walk_forward_positions_path.as_posix(),
                artifact_path="meta_label_risk_budget_open2open",
            )
            for rel_path in sidecars.values():
                path = card / rel_path
                if path.exists() and path.name != "failure_modes.md":
                    recorder.log_artifact(path.as_posix(), artifact_path=spec.experiment)
            recorder_local_dir = Path(recorder.get_local_dir()).resolve()
            recorder_id = recorder.id

    summary = _summary_payload(
        spec=spec,
        bundle=bundle,
        sidecars=sidecars,
        selected=selected,
        metrics=metrics,
        comparison=comparison,
        comparison_summary=comparison_summary,
        walk_forward_metrics=walk_forward_metrics,
        label_diagnostics=label_diagnostics,
        feature_importance=feature_importance,
        risk_analysis=risk_analysis,
        execution_summary=execution_summary,
        latest_signal=latest_signal,
        previous_model_floor=previous_model_floor,
        recorder_uri=recorder_uri,
        recorder_id=recorder_id,
        recorder_local_dir=recorder_local_dir,
    )
    _write_summary_files(card, summary)
    return summary


def _build_meta_frame_bundle(spec: MetaLabelRiskBudgetOpen2OpenSpec) -> _MetaFrameBundle:
    risk_bundle = _build_risk_hardened_frame_bundle(spec)
    frame = risk_bundle.frame.copy()
    intraday_manifest = _ensure_intraday_15m_qlib_provider(spec.root)
    intraday_end = _intraday_15m_end_time(as_of=risk_bundle.as_of, manifest=intraday_manifest)
    intraday_factor_frame = build_intraday_15m_factor_frame(
        QlibProviderConfig(
            provider_uri=spec.intraday_15m_provider_uri,
            region=str(intraday_manifest["region"]),
            market=str(intraday_manifest["market"]),
            freq="15min",
            kernels=1,
        ),
        instruments=list(intraday_manifest["instruments"]),
        start=f"{spec.train[0]} 09:45:00",
        end=intraday_end,
        lag_days=1,
    )
    frame, intraday_feature_cols = _attach_intraday_15m_model_features(frame, intraday_factor_frame)
    frame, combo_feature_cols = _attach_constituent_intraday_combo_features(frame)
    frame = _prepare_meta_frame(
        frame,
        datetime_col="datetime",
        return_col=OPEN_TO_OPEN_RETURN_COL,
        label_col=LABEL_OPEN_TO_OPEN_5D_COL,
        base_position_col=BASE_CONSTRAINED_REF_COL,
        opportunity_threshold=spec.opportunity_threshold,
        take_label_return_threshold=spec.take_label_return_threshold,
        tail_loss_return_threshold=spec.tail_loss_return_threshold,
    )
    base_groups = _risk_hardened_feature_groups(replace(risk_bundle, frame=frame))
    intraday = _existing_unique([*intraday_feature_cols, *combo_feature_cols], frame)
    feature_groups = {
        **base_groups,
        "intraday_15m": intraday,
        "base_cogalpha_soxx_intraday_15m": _existing_unique([*base_groups["base_cogalpha_soxx"], *intraday], frame),
    }
    intraday_summary = {
        "source": intraday_factor_frame.attrs.get("source", "qlib.data.D.features"),
        "provider_uri": intraday_factor_frame.attrs.get("provider_uri", spec.intraday_15m_provider_uri.as_posix()),
        "qlib_store_manifest": (spec.intraday_15m_provider_uri / "qlib_store_manifest.json").as_posix(),
        "calendar_end": intraday_manifest.get("calendar_end"),
        "end": intraday_end,
        "feature_cols": intraday_feature_cols,
        "combo_feature_cols": combo_feature_cols,
        "feature_count": len(intraday_feature_cols) + len(combo_feature_cols),
        "freq": intraday_factor_frame.attrs.get("freq", "15min"),
        "lag_days": intraday_factor_frame.attrs.get("lag_days", 1),
        "role": "execution_overlay_and_risk_budget_features_not_primary_alpha",
        "limitations": list(intraday_factor_frame.attrs.get("limitations", [])),
    }
    return _MetaFrameBundle(
        frame=frame,
        manifest=risk_bundle.manifest,
        provider_meta=risk_bundle.provider_meta,
        as_of=risk_bundle.as_of,
        test=risk_bundle.test,
        feature_cols=feature_groups["base_cogalpha_soxx_intraday_15m"],
        feature_groups=feature_groups,
        intraday_15m_factor_summary=intraday_summary,
        constituent_factor_summary=risk_bundle.constituent_factor_summary,
    )


def _prepare_meta_frame(
    frame: pd.DataFrame,
    *,
    datetime_col: str,
    return_col: str,
    label_col: str,
    base_position_col: str,
    opportunity_threshold: float,
    take_label_return_threshold: float,
    tail_loss_return_threshold: float,
) -> pd.DataFrame:
    out = frame.copy()
    out["datetime"] = pd.to_datetime(out[datetime_col]).dt.normalize()
    out = out.sort_values("datetime").reset_index(drop=True)
    for column in [return_col, label_col, base_position_col, "realized_vol_20"]:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    base = out[base_position_col].fillna(0.0).clip(lower=0.0, upper=1.0)
    out[OPPORTUNITY_SIDE_COL] = base.ge(float(opportunity_threshold)).astype("float64")
    out[TAKE_TRADE_LABEL_COL] = (
        out[label_col].gt(float(take_label_return_threshold)) & out[OPPORTUNITY_SIDE_COL].gt(0.0)
    ).astype("float64")
    out[TAIL_LOSS_LABEL_COL] = out[return_col].le(float(tail_loss_return_threshold)).astype("float64")
    return out


def _segment(data: pd.DataFrame, window: tuple[str, str]) -> pd.DataFrame:
    start, end = pd.Timestamp(window[0]), pd.Timestamp(window[1])
    return data.loc[(data["datetime"] >= start) & (data["datetime"] <= end)].copy()


def _fit_meta_label_models(
    train: pd.DataFrame,
    feature_cols: list[str],
    *,
    take_label_col: str,
    tail_label_col: str,
    opportunity_col: str,
    random_state: int,
) -> dict[str, Any]:
    active_train = train.loc[train[opportunity_col].astype(float).gt(0.0)].copy()
    if len(active_train) < 100 or active_train[take_label_col].nunique(dropna=True) < 2:
        active_train = train.copy()
    payload: dict[str, Any] = {
        "models": {},
        "feature_cols": list(feature_cols),
        "importance_rows": [],
        "training_rows": {
            "take_trade_rows": int(len(active_train)),
            "tail_loss_rows": int(len(train)),
            "take_trade_positive_rate": float(active_train[take_label_col].mean()) if len(active_train) else None,
            "tail_loss_positive_rate": float(train[tail_label_col].mean()) if len(train) else None,
        },
    }
    for index, model_name in enumerate(["lgbm_classifier", "xgb_classifier"]):
        take_model = _fit_binary_probability_model(
            active_train,
            feature_cols,
            label_col=take_label_col,
            model_name=model_name,
            random_state=random_state + index,
        )
        tail_model = _fit_binary_probability_model(
            train,
            feature_cols,
            label_col=tail_label_col,
            model_name=model_name,
            random_state=random_state + 10 + index,
        )
        payload["models"][model_name] = {"take": take_model, "tail": tail_model}
        payload["importance_rows"].extend(
            _model_feature_importance(take_model, feature_cols, model_name=model_name, label_role="take_trade_prob")
        )
        payload["importance_rows"].extend(
            _model_feature_importance(tail_model, feature_cols, model_name=model_name, label_role="tail_loss_prob")
        )
    return payload


def _fit_binary_probability_model(
    train: pd.DataFrame,
    feature_cols: list[str],
    *,
    label_col: str,
    model_name: str,
    random_state: int,
):
    y = train[label_col].fillna(0.0).astype(int)
    x = train[feature_cols].astype("float64")
    if y.nunique() < 2:
        model = DummyClassifier(strategy="constant", constant=int(y.iloc[0] if len(y) else 0))
    else:
        model = _build_model(model_name, random_state=random_state)
    model.fit(x, y)
    return model


def _predict_meta_probabilities(
    model_payload: dict[str, Any],
    segment: pd.DataFrame,
    feature_cols: list[str],
) -> dict[str, dict[str, pd.Series]]:
    x = segment[feature_cols].astype("float64")
    out: dict[str, dict[str, pd.Series]] = {}
    for model_name, models in model_payload["models"].items():
        out[model_name] = {
            "take_probability": _positive_probability(models["take"], x),
            "tail_probability": _positive_probability(models["tail"], x),
        }
    take_stack = pd.concat(
        [payload["take_probability"].rename(model) for model, payload in out.items()],
        axis=1,
    )
    tail_stack = pd.concat(
        [payload["tail_probability"].rename(model) for model, payload in out.items()],
        axis=1,
    )
    out["lgb_xgb_consensus"] = {
        "take_probability": take_stack.min(axis=1).clip(0.0, 1.0).rename("take_probability"),
        "tail_probability": tail_stack.max(axis=1).clip(0.0, 1.0).rename("tail_probability"),
    }
    return out


def _candidate_grid() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for max_position in [0.55, 0.65, 0.75]:
        for take_threshold in [0.50, 0.55, 0.60]:
            for target_daily_vol in [0.035, 0.045]:
                for tail_soft_threshold, tail_hard_threshold, tail_soft_cap, tail_hard_cap in [
                    (0.20, 0.45, 0.60, 0.20),
                    (0.28, 0.50, 0.50, 0.10),
                ]:
                    for rebalance_threshold in [0.05, 0.10]:
                        rows.append(
                            {
                                "max_position": max_position,
                                "take_threshold": take_threshold,
                                "target_daily_vol": target_daily_vol,
                                "min_vol_scale": 0.0,
                                "max_vol_scale": 1.0,
                                "tail_soft_threshold": tail_soft_threshold,
                                "tail_hard_threshold": tail_hard_threshold,
                                "tail_soft_cap": tail_soft_cap,
                                "tail_hard_cap": tail_hard_cap,
                                "drawdown_soft_floor": -0.08,
                                "drawdown_hard_floor": -0.12,
                                "drawdown_soft_cap": 0.55,
                                "drawdown_hard_cap": 0.25,
                                "intraday_vol_ratio_threshold": 3.25,
                                "intraday_range_ratio_threshold": 3.35,
                                "intraday_late_ret_floor": -0.03,
                                "intraday_dislocation_floor": -0.006,
                                "intraday_soft_cap": 0.65,
                                "intraday_hard_cap": 0.35,
                                "rebalance_threshold": rebalance_threshold,
                                "max_daily_change": 0.35,
                            }
                        )
    return rows


def _candidate_name(model_key: str, spec: dict[str, Any]) -> str:
    return (
        f"{model_key}_meta_label_risk_budget"
        f"_p{float(spec['take_threshold']):g}"
        f"_mp{float(spec['max_position']):g}"
        f"_tv{float(spec['target_daily_vol']):g}"
        f"_ts{float(spec['tail_soft_threshold']):g}"
        f"_th{float(spec['tail_hard_threshold']):g}"
        f"_rb{float(spec['rebalance_threshold']):g}"
    )


def _simulate_meta_label_candidate(
    data: pd.DataFrame,
    *,
    take_probability: pd.Series,
    tail_probability: pd.Series,
    candidate: str,
    segment: str,
    model_key: str,
    position_spec: dict[str, Any],
    return_col: str,
    base_position_col: str,
    realized_vol_col: str,
    cost_per_turnover: float,
    extreme_loss_threshold: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    components = _risk_budget_components(
        data,
        take_probability=take_probability,
        tail_probability=tail_probability,
        position_spec=position_spec,
        base_position_col=base_position_col,
        realized_vol_col=realized_vol_col,
    )
    forward_return = pd.to_numeric(data[return_col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    raw_target = components["raw_target"].astype("float64").clip(lower=0.0, upper=float(position_spec["max_position"]))
    target_positions: list[float] = []
    turnovers: list[float] = []
    strategy_returns: list[float] = []
    drawdown_caps: list[float] = []
    equity = 1.0
    peak = 1.0
    last_position = 0.0
    rebalance_threshold = float(position_spec.get("rebalance_threshold", 0.0))
    max_daily_change = float(position_spec.get("max_daily_change", 1.0))
    max_position = float(position_spec.get("max_position", 1.0))
    for desired, next_return in zip(raw_target.to_numpy(dtype="float64"), forward_return.to_numpy(dtype="float64")):
        drawdown = equity / max(peak, 1e-12) - 1.0
        if drawdown <= float(position_spec["drawdown_hard_floor"]):
            drawdown_cap = float(position_spec["drawdown_hard_cap"])
        elif drawdown <= float(position_spec["drawdown_soft_floor"]):
            drawdown_cap = float(position_spec["drawdown_soft_cap"])
        else:
            drawdown_cap = 1.0
        desired = min(float(desired), max_position * drawdown_cap)
        if abs(desired - last_position) < rebalance_threshold:
            target = last_position
        else:
            change = float(np.clip(desired - last_position, -max_daily_change, max_daily_change))
            target = float(np.clip(last_position + change, 0.0, max_position))
        turnover = abs(target - last_position)
        strategy_return = target * float(next_return) - turnover * float(cost_per_turnover)
        equity *= 1.0 + strategy_return
        peak = max(peak, equity)
        last_position = target
        drawdown_caps.append(drawdown_cap)
        target_positions.append(target)
        turnovers.append(turnover)
        strategy_returns.append(strategy_return)

    target_position = pd.Series(target_positions, index=data.index, name="target_position")
    turnover = pd.Series(turnovers, index=data.index, name="turnover")
    returns = pd.Series(strategy_returns, index=data.index, name="strategy_return")
    baseline_target = pd.to_numeric(data[base_position_col], errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0)
    baseline_turnover = baseline_target.diff().abs().fillna(baseline_target.abs())
    baseline_returns = baseline_target * forward_return - baseline_turnover * float(cost_per_turnover)
    metrics = _strategy_metrics_from_returns(
        candidate=candidate,
        segment=segment,
        strategy_returns=returns,
        target_position=target_position,
        turnover=turnover,
        baseline_returns=baseline_returns,
        extreme_loss_threshold=extreme_loss_threshold,
    )
    metrics.update(
        {
            "position_mode": "meta_label_risk_budget",
            "take_threshold": float(position_spec["take_threshold"]),
            "tail_soft_threshold": float(position_spec["tail_soft_threshold"]),
            "tail_hard_threshold": float(position_spec["tail_hard_threshold"]),
            "target_daily_vol": float(position_spec["target_daily_vol"]),
            "risk_budget_max_position": float(position_spec["max_position"]),
            "avg_take_probability": float(take_probability.mean()),
            "avg_tail_probability": float(tail_probability.mean()),
            "avg_edge_score": float(components["edge_score"].mean()),
            "avg_tail_cap": float(components["tail_cap"].mean()),
            "avg_volatility_budget": float(components["volatility_budget"].mean()),
            "avg_intraday_execution_cap": float(components["intraday_execution_cap"].mean()),
            "min_drawdown_budget": float(min(drawdown_caps) if drawdown_caps else 1.0),
        }
    )
    positions = pd.DataFrame(
        {
            "datetime": pd.to_datetime(data["datetime"]).dt.normalize().to_numpy(),
            "candidate": candidate,
            "segment": segment,
            "model": model_key,
            "take_probability": take_probability.reindex(data.index).to_numpy(dtype="float64"),
            "tail_probability": tail_probability.reindex(data.index).to_numpy(dtype="float64"),
            "edge_score": components["edge_score"].to_numpy(dtype="float64"),
            "volatility_budget": components["volatility_budget"].to_numpy(dtype="float64"),
            "tail_cap": components["tail_cap"].to_numpy(dtype="float64"),
            "intraday_execution_cap": components["intraday_execution_cap"].to_numpy(dtype="float64"),
            "drawdown_budget": np.asarray(drawdown_caps, dtype="float64"),
            "raw_target": raw_target.to_numpy(dtype="float64"),
            "target_position": target_position.to_numpy(dtype="float64"),
            "turnover": turnover.to_numpy(dtype="float64"),
            "strategy_return": returns.to_numpy(dtype="float64"),
        }
    )
    return metrics, positions


def _risk_budget_components(
    data: pd.DataFrame,
    *,
    take_probability: pd.Series,
    tail_probability: pd.Series,
    position_spec: dict[str, Any],
    base_position_col: str,
    realized_vol_col: str,
) -> dict[str, pd.Series]:
    take = take_probability.reindex(data.index).astype("float64").fillna(0.0).clip(0.0, 1.0)
    tail = tail_probability.reindex(data.index).astype("float64").fillna(1.0).clip(0.0, 1.0)
    base = pd.to_numeric(data[base_position_col], errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0)
    opportunity = base.gt(0.0).astype("float64")
    threshold = float(position_spec["take_threshold"])
    edge_score = ((take - threshold) / max(1.0 - threshold, 1e-12)).clip(0.0, 1.0)
    realized_vol = pd.to_numeric(data[realized_vol_col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    volatility_budget = (
        float(position_spec["target_daily_vol"]) / realized_vol.replace(0.0, np.nan)
    ).clip(
        lower=float(position_spec.get("min_vol_scale", 0.0)),
        upper=float(position_spec.get("max_vol_scale", 1.0)),
    )
    volatility_budget = volatility_budget.fillna(float(position_spec.get("max_vol_scale", 1.0)))
    tail_cap = pd.Series(1.0, index=data.index, dtype="float64")
    tail_cap = tail_cap.where(tail < float(position_spec["tail_soft_threshold"]), float(position_spec["tail_soft_cap"]))
    tail_cap = tail_cap.where(tail < float(position_spec["tail_hard_threshold"]), float(position_spec["tail_hard_cap"]))
    intraday_cap = _intraday_execution_cap(data, position_spec)
    raw_target = (
        opportunity
        * edge_score
        * volatility_budget
        * tail_cap
        * intraday_cap
        * float(position_spec["max_position"])
    )
    raw_target = raw_target.where(take >= threshold, 0.0)
    return {
        "edge_score": edge_score,
        "volatility_budget": volatility_budget,
        "tail_cap": tail_cap,
        "intraday_execution_cap": intraday_cap,
        "raw_target": raw_target.rename("raw_target"),
    }


def _intraday_execution_cap(data: pd.DataFrame, position_spec: dict[str, Any]) -> pd.Series:
    cap = pd.Series(1.0, index=data.index, dtype="float64")
    risk_count = pd.Series(0, index=data.index, dtype="int64")
    checks = [
        (
            "intraday15m_realized_vol_ratio_soxl_soxx_lag1",
            "above",
            float(position_spec["intraday_vol_ratio_threshold"]),
        ),
        (
            "intraday15m_range_ratio_soxl_soxx_lag1",
            "above",
            float(position_spec["intraday_range_ratio_threshold"]),
        ),
        (
            "intraday15m_soxl_late_ret_lag1",
            "below",
            float(position_spec["intraday_late_ret_floor"]),
        ),
        (
            "intraday15m_soxl_minus_3x_soxx_ret_lag1",
            "below",
            float(position_spec["intraday_dislocation_floor"]),
        ),
    ]
    observed = 0
    for column, direction, threshold in checks:
        if column not in data.columns:
            continue
        observed += 1
        values = pd.to_numeric(data[column], errors="coerce")
        if direction == "above":
            risk_count += values.gt(threshold).fillna(False).astype("int64")
        else:
            risk_count += values.lt(threshold).fillna(False).astype("int64")
    if observed == 0:
        return cap
    cap = cap.where(risk_count <= 0, float(position_spec["intraday_soft_cap"]))
    cap = cap.where(risk_count < 2, float(position_spec["intraday_hard_cap"]))
    for coverage_col in ["intraday15m_soxl_bar_coverage_lag1", "intraday15m_soxx_bar_coverage_lag1"]:
        if coverage_col in data.columns:
            coverage = pd.to_numeric(data[coverage_col], errors="coerce")
            cap = cap.where(coverage.ge(0.99).fillna(True), np.minimum(cap, 0.50))
    return cap.rename(INTRADAY_EXECUTION_CAP_COL)


def _strategy_metrics_from_returns(
    *,
    candidate: str,
    segment: str,
    strategy_returns: pd.Series,
    target_position: pd.Series,
    turnover: pd.Series,
    baseline_returns: pd.Series,
    extreme_loss_threshold: float,
) -> dict[str, Any]:
    returns = strategy_returns.astype("float64").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    baseline = baseline_returns.astype("float64").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return_pct = float(((1.0 + returns).prod() - 1.0) * 100.0)
    max_drawdown_pct = _max_drawdown_pct(returns)
    max_daily_loss_pct = float(returns.min() * 100.0) if not returns.empty else 0.0
    tail_metrics = _tail_risk_metrics(returns)
    baseline_return_pct = float(((1.0 + baseline).prod() - 1.0) * 100.0)
    baseline_max_drawdown_pct = _max_drawdown_pct(baseline)
    baseline_max_daily_loss_pct = float(baseline.min() * 100.0) if not baseline.empty else 0.0
    baseline_tail = _tail_risk_metrics(baseline)
    excess_return_pct = return_pct - baseline_return_pct
    trade_count = int(turnover.gt(1e-9).sum())
    zero_ratio = float(target_position.abs().le(1e-9).mean()) if len(target_position) else 0.0
    monthly = returns.groupby(pd.RangeIndex(len(returns)) // 21).apply(lambda values: (1.0 + values).prod() - 1.0)
    positive_month_ratio = float(monthly.gt(0.0).mean()) if len(monthly) else 0.0
    return_to_drawdown = return_pct / max(abs(max_drawdown_pct), 1.0)
    excess_return_to_drawdown = excess_return_pct / max(abs(max_drawdown_pct), 1.0)
    total_turnover = float(turnover.sum())
    extreme_loss_day_count = int(returns.le(float(extreme_loss_threshold)).sum())
    max_drawdown_vs_baseline_gap_pct = max_drawdown_pct - baseline_max_drawdown_pct
    max_daily_loss_vs_baseline_gap_pct = max_daily_loss_pct - baseline_max_daily_loss_pct
    cvar_gap = float(tail_metrics["cvar_5pct_pct"] - baseline_tail["cvar_5pct_pct"])
    worst5_gap = float(tail_metrics["worst_5_daily_loss_pct"] - baseline_tail["worst_5_daily_loss_pct"])
    baseline_risk_improvement_pass = bool(
        max_drawdown_vs_baseline_gap_pct > 0.0
        and max_daily_loss_vs_baseline_gap_pct > 0.0
        and cvar_gap > 0.0
        and worst5_gap > 0.0
    )
    objective = (
        return_to_drawdown * 8.0
        + excess_return_to_drawdown * 6.0
        + return_pct * 0.015
        + max(0.0, max_drawdown_vs_baseline_gap_pct) * 0.20
        + max(0.0, max_daily_loss_vs_baseline_gap_pct) * 0.35
        + max(0.0, cvar_gap) * 0.20
        - total_turnover * 1.75
        - extreme_loss_day_count * 3.0
    )
    return {
        "candidate": candidate,
        "segment": segment,
        "return_pct": return_pct,
        "max_drawdown_pct": max_drawdown_pct,
        "max_daily_loss_pct": max_daily_loss_pct,
        "return_to_drawdown": float(return_to_drawdown),
        "selection_objective_score": float(objective),
        "baseline_return_pct": baseline_return_pct,
        "baseline_max_drawdown_pct": baseline_max_drawdown_pct,
        "baseline_max_daily_loss_pct": baseline_max_daily_loss_pct,
        "baseline_worst_5_daily_loss_pct": float(baseline_tail["worst_5_daily_loss_pct"]),
        "baseline_cvar_5pct_pct": float(baseline_tail["cvar_5pct_pct"]),
        "baseline_underwater_day_ratio": float(baseline_tail["underwater_day_ratio"]),
        "excess_return_pct": float(excess_return_pct),
        "excess_return_to_drawdown": float(excess_return_to_drawdown),
        "max_drawdown_vs_baseline_gap_pct": float(max_drawdown_vs_baseline_gap_pct),
        "max_daily_loss_vs_baseline_gap_pct": float(max_daily_loss_vs_baseline_gap_pct),
        "cvar_5pct_vs_baseline_gap_pct": float(cvar_gap),
        "worst_5_daily_loss_vs_baseline_gap_pct": float(worst5_gap),
        "beats_baseline_return": bool(excess_return_pct > 0.0),
        "baseline_risk_improvement_pass": baseline_risk_improvement_pass,
        **tail_metrics,
        "avg_abs_position": float(target_position.abs().mean()) if len(target_position) else 0.0,
        "max_position": float(target_position.max()) if len(target_position) else 0.0,
        "min_position": float(target_position.min()) if len(target_position) else 0.0,
        "zero_position_day_ratio": zero_ratio,
        "total_turnover": total_turnover,
        "trade_count": trade_count,
        "positive_month_ratio": positive_month_ratio,
        "extreme_loss_threshold": float(extreme_loss_threshold),
        "extreme_loss_day_count": extreme_loss_day_count,
        "cost_per_turnover": 0.0003,
    }


def _select_candidate(metrics: pd.DataFrame, gates: dict[str, Any]) -> tuple[str, str, set[str]]:
    validation = metrics.loc[metrics["segment"].eq("validation")].copy()
    selectable = validation.loc[~validation["selection_disabled"].astype(bool)].copy()
    gate_mask = pd.Series(True, index=selectable.index, dtype=bool)
    gate_mask &= selectable["return_pct"].astype(float).gt(float(gates.get("min_validation_return_pct", 0.0)))
    gate_mask &= selectable["avg_abs_position"].astype(float).ge(float(gates.get("min_avg_abs_position", 0.0)))
    gate_mask &= selectable["zero_position_day_ratio"].astype(float).le(float(gates.get("max_zero_position_day_ratio", 1.0)))
    gate_mask &= selectable["max_drawdown_pct"].astype(float).ge(float(gates.get("validation_max_drawdown_floor", -100.0)))
    gate_mask &= selectable["max_daily_loss_pct"].astype(float).ge(float(gates.get("validation_max_daily_loss_floor", -100.0)))
    gate_mask &= selectable["cvar_5pct_pct"].astype(float).ge(float(gates.get("validation_cvar_5pct_floor", -100.0)))
    gate_mask &= selectable["worst_5_daily_loss_pct"].astype(float).ge(
        float(gates.get("validation_worst_5_daily_loss_floor", -100.0))
    )
    gate_mask &= selectable["underwater_day_ratio"].astype(float).le(
        float(gates.get("max_validation_underwater_day_ratio", 1.0))
    )
    gate_mask &= selectable["total_turnover"].astype(float).le(float(gates.get("max_validation_total_turnover", 1e12)))
    gate_mask &= selectable["extreme_loss_day_count"].astype(int).le(
        int(gates.get("max_extreme_loss_day_count", 10**9))
    )
    eligible = selectable.loc[gate_mask].copy()
    if not eligible.empty:
        pool = eligible
        reason = "eligible_validation_risk_first"
    elif not selectable.empty:
        pool = selectable
        reason = "selectable_validation_fallback_failed_risk_gates"
    else:
        pool = validation
        reason = "all_validation_fallback"
    selected = pool.sort_values(["selection_objective_score", "return_pct"], ascending=[False, False]).iloc[0]
    return str(selected["candidate"]), reason, set(eligible["candidate"].astype(str))


def _risk_gates_for_spec(spec: MetaLabelRiskBudgetOpen2OpenSpec) -> dict[str, Any]:
    return {
        "min_validation_return_pct": 0.0,
        "min_avg_abs_position": spec.min_avg_abs_position,
        "max_zero_position_day_ratio": spec.max_zero_position_day_ratio,
        "validation_max_drawdown_floor": spec.validation_max_drawdown_floor,
        "validation_max_daily_loss_floor": spec.validation_max_daily_loss_floor,
        "validation_cvar_5pct_floor": spec.validation_cvar_5pct_floor,
        "validation_worst_5_daily_loss_floor": spec.validation_worst_5_daily_loss_floor,
        "max_validation_underwater_day_ratio": spec.max_validation_underwater_day_ratio,
        "max_validation_total_turnover": spec.max_validation_total_turnover,
        "test_max_drawdown_floor": spec.test_max_drawdown_floor,
        "test_max_daily_loss_floor": spec.test_max_daily_loss_floor,
        "test_cvar_5pct_floor": spec.test_cvar_5pct_floor,
        "test_worst_5_daily_loss_floor": spec.test_worst_5_daily_loss_floor,
        "max_test_underwater_day_ratio": spec.max_test_underwater_day_ratio,
        "max_test_total_turnover": spec.max_test_total_turnover,
        "max_extreme_loss_day_count": spec.max_extreme_loss_day_count,
        "extreme_loss_threshold": spec.extreme_loss_threshold,
    }


def _segment_passes(row: dict[str, Any], gates: dict[str, Any], *, prefix: str) -> bool:
    return bool(
        float(row.get("avg_abs_position", 0.0)) >= float(gates.get("min_avg_abs_position", 0.0))
        and float(row.get("zero_position_day_ratio", 0.0)) <= float(gates.get("max_zero_position_day_ratio", 1.0))
        and float(row.get("max_drawdown_pct", 0.0)) >= float(gates.get(f"{prefix}_max_drawdown_floor", -100.0))
        and float(row.get("max_daily_loss_pct", 0.0)) >= float(gates.get(f"{prefix}_max_daily_loss_floor", -100.0))
        and float(row.get("cvar_5pct_pct", 0.0)) >= float(gates.get(f"{prefix}_cvar_5pct_floor", -100.0))
        and float(row.get("worst_5_daily_loss_pct", 0.0)) >= float(gates.get(f"{prefix}_worst_5_daily_loss_floor", -100.0))
        and float(row.get("underwater_day_ratio", 0.0)) <= float(gates.get(f"max_{prefix}_underwater_day_ratio", 1.0))
        and float(row.get("total_turnover", 0.0)) <= float(gates.get(f"max_{prefix}_total_turnover", 1e12))
        and int(row.get("extreme_loss_day_count", 0) or 0) <= int(gates.get("max_extreme_loss_day_count", 10**9))
    )


def _selected_payload(metrics: pd.DataFrame) -> dict[str, Any]:
    rows = metrics.loc[metrics["selected"].astype(bool)].copy()
    if rows.empty:
        raise ValueError("meta-label risk-budget model did not select a candidate")
    payload: dict[str, Any] = {
        "selected_by": "validation_risk_first_objective",
        "candidate": str(rows.iloc[0]["candidate"]),
        "model": str(rows.iloc[0]["model"]),
    }
    for segment in ["validation", "test"]:
        segment_rows = rows.loc[rows["segment"].eq(segment)]
        if not segment_rows.empty:
            payload[segment] = _jsonable(segment_rows.iloc[0].to_dict())
    return payload


def _benchmark_comparison(test_row: dict[str, Any], *, benchmarks: list[dict[str, Any]]) -> pd.DataFrame:
    rows = [
        {
            "benchmark": "meta_label_risk_budget_selected",
            "return_pct": float(test_row["return_pct"]),
            "max_drawdown_pct": float(test_row["max_drawdown_pct"]),
            "max_daily_loss_pct": float(test_row["max_daily_loss_pct"]),
            "trade_count": int(test_row.get("trade_count", 0) or 0),
            "source": "meta_label_risk_budget_open2open",
        }
    ]
    for benchmark in benchmarks:
        rows.append(
            {
                "benchmark": str(benchmark["name"]),
                "return_pct": float(benchmark.get("return_pct", 0.0)),
                "max_drawdown_pct": float(benchmark.get("max_drawdown_pct", 0.0)),
                "max_daily_loss_pct": float(benchmark.get("max_daily_loss_pct", 0.0) or 0.0),
                "trade_count": int(benchmark.get("trade_count", 0) or 0),
                "source": str(benchmark.get("source", benchmark["name"])),
            }
        )
    selected_return = rows[0]["return_pct"]
    selected_drawdown = rows[0]["max_drawdown_pct"]
    selected_daily_loss = rows[0]["max_daily_loss_pct"]
    for row in rows:
        row["selected_return_gap_pct"] = float(selected_return - row["return_pct"])
        row["selected_max_drawdown_gap_pct"] = float(selected_drawdown - row["max_drawdown_pct"])
        row["selected_max_daily_loss_gap_pct"] = float(selected_daily_loss - row["max_daily_loss_pct"])
        row["selected_beats_return"] = bool(selected_return > row["return_pct"])
    return pd.DataFrame(rows)


def _comparison_summary(test_row: dict[str, Any], comparison: pd.DataFrame) -> dict[str, Any]:
    summary = {
        "selected_return_pct": float(test_row["return_pct"]),
        "selected_max_drawdown_pct": float(test_row["max_drawdown_pct"]),
        "selected_max_daily_loss_pct": float(test_row["max_daily_loss_pct"]),
    }
    for _, row in comparison.loc[comparison["benchmark"] != "meta_label_risk_budget_selected"].iterrows():
        name = str(row["benchmark"])
        summary[f"beats_{name}"] = bool(row["selected_beats_return"])
        summary[f"selected_vs_{name}_return_gap_pct"] = float(row["selected_return_gap_pct"])
        summary[f"selected_vs_{name}_max_drawdown_gap_pct"] = float(row["selected_max_drawdown_gap_pct"])
        summary[f"selected_vs_{name}_max_daily_loss_gap_pct"] = float(row["selected_max_daily_loss_gap_pct"])
    summary.setdefault("beats_base_constrained_ref", False)
    summary.setdefault("selected_vs_base_constrained_ref_return_gap_pct", None)
    return summary


def _run_walk_forward(
    spec: MetaLabelRiskBudgetOpen2OpenSpec,
    bundle: _MetaFrameBundle,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    folds = [
        {
            "fold": "wf_2023",
            "train": ("2019-01-02", "2021-12-31"),
            "validation": ("2022-01-03", "2022-12-30"),
            "test": ("2023-01-03", "2023-12-29"),
        },
        {
            "fold": "wf_2024",
            "train": ("2019-01-02", "2022-12-30"),
            "validation": ("2023-01-03", "2023-12-29"),
            "test": ("2024-01-02", "2024-12-31"),
        },
        {
            "fold": "wf_2025",
            "train": ("2019-01-02", "2023-12-29"),
            "validation": ("2024-01-02", "2024-12-31"),
            "test": ("2025-01-02", "2025-12-31"),
        },
        {
            "fold": "wf_2026_ytd",
            "train": ("2019-01-02", "2024-12-31"),
            "validation": ("2025-01-02", "2025-12-31"),
            "test": bundle.test,
        },
    ]
    rows: list[dict[str, Any]] = []
    position_frames: list[pd.DataFrame] = []
    gates = _risk_gates_for_spec(spec)
    for index, fold in enumerate(folds, start=1):
        metrics, positions, _, _ = evaluate_meta_label_risk_budget_candidates(
            bundle.frame,
            feature_cols=bundle.feature_cols,
            train_window=fold["train"],
            validation_window=fold["validation"],
            test_window=fold["test"],
            cost_per_turnover=spec.cost_per_turnover,
            opportunity_threshold=spec.opportunity_threshold,
            take_label_return_threshold=spec.take_label_return_threshold,
            tail_loss_return_threshold=spec.tail_loss_return_threshold,
            validation_gates=gates,
            random_state=42 + index,
        )
        selected = _selected_payload(metrics)
        validation_row = selected.get("validation", {})
        test_row = selected.get("test", {})
        validation_pass = _segment_passes(validation_row, gates, prefix="validation")
        test_pass = _segment_passes(test_row, gates, prefix="test")
        rows.append(
            {
                "fold": fold["fold"],
                "segment": "test",
                "train_start": fold["train"][0],
                "train_end": fold["train"][1],
                "validation_start": fold["validation"][0],
                "validation_end": fold["validation"][1],
                "test_start": fold["test"][0],
                "test_end": fold["test"][1],
                "selected_candidate": selected["candidate"],
                "selected_model": selected["model"],
                "validation_return_pct": float(validation_row.get("return_pct", 0.0)),
                "validation_max_drawdown_pct": float(validation_row.get("max_drawdown_pct", 0.0)),
                "validation_max_daily_loss_pct": float(validation_row.get("max_daily_loss_pct", 0.0)),
                "validation_cvar_5pct_pct": float(validation_row.get("cvar_5pct_pct", 0.0)),
                "validation_worst_5_daily_loss_pct": float(validation_row.get("worst_5_daily_loss_pct", 0.0)),
                "validation_underwater_day_ratio": float(validation_row.get("underwater_day_ratio", 0.0)),
                "validation_total_turnover": float(validation_row.get("total_turnover", 0.0)),
                "validation_risk_gate_pass": bool(validation_pass),
                "test_return_pct": float(test_row.get("return_pct", 0.0)),
                "test_max_drawdown_pct": float(test_row.get("max_drawdown_pct", 0.0)),
                "test_max_daily_loss_pct": float(test_row.get("max_daily_loss_pct", 0.0)),
                "test_cvar_5pct_pct": float(test_row.get("cvar_5pct_pct", 0.0)),
                "test_worst_5_daily_loss_pct": float(test_row.get("worst_5_daily_loss_pct", 0.0)),
                "test_underwater_day_ratio": float(test_row.get("underwater_day_ratio", 0.0)),
                "test_total_turnover": float(test_row.get("total_turnover", 0.0)),
                "test_avg_abs_position": float(test_row.get("avg_abs_position", 0.0)),
                "test_extreme_loss_day_count": int(test_row.get("extreme_loss_day_count", 0) or 0),
                "test_risk_gate_pass": bool(test_pass),
                "risk_gate_pass": bool(validation_pass and test_pass),
            }
        )
        selected_positions = positions.loc[positions["selected"].astype(bool)].copy()
        selected_positions.insert(0, "walk_forward_fold", fold["fold"])
        position_frames.append(selected_positions)
    return (
        pd.DataFrame(rows),
        pd.concat(position_frames, ignore_index=True) if position_frames else pd.DataFrame(),
    )


def _walk_forward_summary(metrics: pd.DataFrame) -> dict[str, Any]:
    if metrics.empty:
        return {"fold_count": 0, "all_folds_pass_risk_gates": None}
    returns = metrics["test_return_pct"].astype(float)
    return {
        "fold_count": int(len(metrics)),
        "positive_test_fold_count": int(returns.gt(0.0).sum()),
        "positive_test_fold_ratio": float(returns.gt(0.0).mean()),
        "mean_test_return_pct": float(returns.mean()),
        "median_test_return_pct": float(returns.median()),
        "min_test_return_pct": float(returns.min()),
        "worst_test_max_drawdown_pct": float(metrics["test_max_drawdown_pct"].astype(float).min()),
        "worst_test_max_daily_loss_pct": float(metrics["test_max_daily_loss_pct"].astype(float).min()),
        "worst_test_cvar_5pct_pct": float(metrics["test_cvar_5pct_pct"].astype(float).min()),
        "worst_test_worst_5_daily_loss_pct": float(metrics["test_worst_5_daily_loss_pct"].astype(float).min()),
        "max_test_total_turnover": float(metrics["test_total_turnover"].astype(float).max()),
        "all_folds_pass_risk_gates": bool(metrics["risk_gate_pass"].astype(bool).all()),
    }


def _risk_analysis_payload(
    spec: MetaLabelRiskBudgetOpen2OpenSpec,
    selected: dict[str, Any],
    comparison_summary: dict[str, Any],
    walk_forward_metrics: pd.DataFrame,
    execution_summary: dict[str, Any],
    previous_model_floor: dict[str, Any],
) -> pd.DataFrame:
    gates = _risk_gates_for_spec(spec)
    validation_pass = _segment_passes(selected.get("validation", {}), gates, prefix="validation")
    test_pass = _segment_passes(selected.get("test", {}), gates, prefix="test")
    walk_forward = _walk_forward_summary(walk_forward_metrics)
    baseline_fallback = _baseline_risk_fallback_payload(selected.get("test", {}))
    previous_floor = _previous_model_return_floor_payload(selected.get("test", {}), previous_model_floor)
    return pd.DataFrame(
        [
            {
                "check": "architecture_replaces_probability_boost",
                "status": "pass",
                "value": selected["test"].get("position_architecture"),
                "detail": "base_constrained_ref is only opportunity_side; final target multiplies edge, volatility, tail, drawdown, and intraday caps",
            },
            {
                "check": "lgb_xgb_dual_meta_label_selection",
                "status": "pass" if selected["model"] == "lgb_xgb_consensus" else "fail",
                "value": selected["model"],
                "detail": "selectable model must use LGB/XGB consensus for take_trade and tail_loss probabilities",
            },
            {
                "check": "selected_validation_risk_gates",
                "status": "pass" if validation_pass else "fail",
                "value": validation_pass,
                "detail": json.dumps(selected.get("validation", {}), sort_keys=True),
            },
            {
                "check": "selected_test_risk_gates",
                "status": "pass" if test_pass else "fail",
                "value": test_pass,
                "detail": json.dumps(selected.get("test", {}), sort_keys=True),
            },
            {
                "check": "walk_forward_all_years",
                "status": "pass" if walk_forward.get("all_folds_pass_risk_gates") else "fail",
                "value": walk_forward.get("all_folds_pass_risk_gates"),
                "detail": json.dumps(walk_forward, sort_keys=True),
            },
            {
                "check": "base_relative_risk_fallback",
                "status": "pass" if baseline_fallback["pass"] else "fail",
                "value": baseline_fallback["selected_vs_base_return_gap_pct"],
                "detail": json.dumps(baseline_fallback, sort_keys=True),
            },
            {
                "check": "previous_model_return_floor",
                "status": "pass" if previous_floor["pass"] else "fail",
                "value": previous_floor["selected_vs_floor_return_gap_pct"],
                "detail": json.dumps(previous_floor, sort_keys=True),
            },
            {
                "check": "intraday_execution_stress",
                "status": execution_summary.get("status", "needs_data"),
                "value": execution_summary.get("coverage_days"),
                "detail": json.dumps(execution_summary.get("scenarios", []), sort_keys=True),
            },
            {
                "check": "direct_base_comparison",
                "status": "pass" if comparison_summary.get("beats_base_constrained_ref") else "fail",
                "value": comparison_summary.get("selected_vs_base_constrained_ref_return_gap_pct"),
                "detail": json.dumps(comparison_summary, sort_keys=True),
            },
        ]
    )


def _promotion_decision(
    selected: dict[str, Any],
    comparison_summary: dict[str, Any],
    walk_forward_summary: dict[str, Any],
    execution_summary: dict[str, Any],
    risk_analysis: pd.DataFrame,
) -> str:
    failed_checks = set(risk_analysis.loc[risk_analysis["status"].eq("fail"), "check"].astype(str))
    if selected.get("model") != "lgb_xgb_consensus":
        return "not_promoted_no_lgb_xgb_consensus"
    if "selected_validation_risk_gates" in failed_checks or "selected_test_risk_gates" in failed_checks:
        return "not_promoted_failed_selected_tail_risk_gates"
    if not bool(walk_forward_summary.get("all_folds_pass_risk_gates", False)):
        return "not_promoted_failed_walk_forward_risk_gates"
    if not bool(comparison_summary.get("beats_base_constrained_ref", False)):
        return "not_promoted_under_base_constrained_ref"
    if execution_summary.get("status") != "ok":
        return "not_promoted_needs_intraday_execution_data"
    return "candidate_for_validated_research_meta_label_risk_budget"


def _latest_signal(positions: pd.DataFrame, selected: dict[str, Any]) -> dict[str, Any]:
    rows = positions.loc[positions["selected"].astype(bool) & positions["segment"].eq("test")].copy()
    if rows.empty:
        return {"status": "needs_data"}
    row = rows.sort_values("datetime").iloc[-1]
    target = float(row["target_position"])
    return {
        "status": "ok",
        "as_of": pd.Timestamp(row["datetime"]).date().isoformat(),
        "candidate": selected["candidate"],
        "model": selected["model"],
        "target_position": target,
        "target_percent": target * 100.0,
        "action": "score_next_open_target_position",
        "take_probability": float(row["take_probability"]),
        "tail_probability": float(row["tail_probability"]),
        "edge_score": float(row["edge_score"]),
        "volatility_budget": float(row["volatility_budget"]),
        "tail_cap": float(row["tail_cap"]),
        "intraday_execution_cap": float(row["intraday_execution_cap"]),
        "drawdown_budget": float(row["drawdown_budget"]),
    }


def _label_diagnostics(
    data: pd.DataFrame,
    train_window: tuple[str, str],
    validation_window: tuple[str, str],
    test_window: tuple[str, str],
) -> pd.DataFrame:
    rows = []
    for name, window in [("train", train_window), ("validation", validation_window), ("test", test_window)]:
        segment = _segment(data, window)
        rows.append(
            {
                "segment": name,
                "start": window[0],
                "end": window[1],
                "row_count": int(len(segment)),
                "opportunity_rate": float(segment[OPPORTUNITY_SIDE_COL].mean()) if len(segment) else None,
                "take_trade_positive_rate": float(segment[TAKE_TRADE_LABEL_COL].mean()) if len(segment) else None,
                "tail_loss_positive_rate": float(segment[TAIL_LOSS_LABEL_COL].mean()) if len(segment) else None,
                "avg_open_to_open_return_pct": float(segment[OPEN_TO_OPEN_RETURN_COL].mean() * 100.0)
                if len(segment)
                else None,
                "avg_label_open_to_open_5d_pct": float(segment[LABEL_OPEN_TO_OPEN_5D_COL].mean() * 100.0)
                if len(segment)
                else None,
            }
        )
    return pd.DataFrame(rows)


def _model_feature_importance(fitted_model, feature_cols: list[str], *, model_name: str, label_role: str) -> list[dict[str, Any]]:
    estimator = getattr(fitted_model, "steps", [(None, fitted_model)])[-1][1]
    values = getattr(estimator, "feature_importances_", None)
    if values is None:
        return []
    total = float(np.sum(np.abs(values))) or 1.0
    rows = []
    for feature, value in zip(feature_cols, values):
        rows.append(
            {
                "model": model_name,
                "label_role": label_role,
                "feature": feature,
                "importance": float(value),
                "importance_share": float(abs(value) / total),
            }
        )
    return rows


def _feature_importance_frame(model_payload: dict[str, Any]) -> pd.DataFrame:
    frame = pd.DataFrame(model_payload.get("importance_rows", []))
    if frame.empty:
        return pd.DataFrame(columns=["model", "label_role", "feature", "importance", "importance_share"])
    return frame.sort_values(["label_role", "importance_share"], ascending=[True, False]).reset_index(drop=True)


def _write_sidecars(
    card: Path,
    *,
    sidecars: dict[str, str],
    selected: dict[str, Any],
    metrics: pd.DataFrame,
    comparison: pd.DataFrame,
    walk_forward_metrics: pd.DataFrame,
    label_diagnostics: pd.DataFrame,
    feature_importance: pd.DataFrame,
    risk_analysis: pd.DataFrame,
    execution_stress: pd.DataFrame,
    execution_summary: dict[str, Any],
    latest_signal: dict[str, Any],
    feature_groups: dict[str, list[str]],
) -> None:
    (card / sidecars["selected_model"]).write_text(json.dumps(selected, indent=2, sort_keys=True), encoding="utf-8")
    metrics.to_csv(card / sidecars["model_selection"], index=False)
    comparison.to_csv(card / sidecars["backtest_benchmark_comparison"], index=False)
    walk_forward_metrics.to_csv(card / sidecars["backtest_walk_forward_metrics"], index=False)
    label_diagnostics.to_csv(card / sidecars["label_diagnostics"], index=False)
    feature_importance.to_csv(card / sidecars["feature_importance"], index=False)
    risk_analysis.to_csv(card / sidecars["risk_analysis"], index=False)
    execution_stress.to_csv(card / sidecars["execution_stress"], index=False)
    (card / sidecars["execution_stress_summary"]).write_text(
        json.dumps(execution_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (card / sidecars["latest_signal"]).write_text(
        json.dumps(latest_signal, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (card / sidecars["position_summary"]).write_text(
        json.dumps({"selected_candidate": selected}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {"feature_group": name, "feature_count": len(columns), "features": json.dumps(columns)}
            for name, columns in feature_groups.items()
        ]
    ).to_csv(card / sidecars["feature_sets"], index=False)


def _summary_payload(
    *,
    spec: MetaLabelRiskBudgetOpen2OpenSpec,
    bundle: _MetaFrameBundle,
    sidecars: dict[str, str],
    selected: dict[str, Any],
    metrics: pd.DataFrame,
    comparison: pd.DataFrame,
    comparison_summary: dict[str, Any],
    walk_forward_metrics: pd.DataFrame,
    label_diagnostics: pd.DataFrame,
    feature_importance: pd.DataFrame,
    risk_analysis: pd.DataFrame,
    execution_summary: dict[str, Any],
    latest_signal: dict[str, Any],
    previous_model_floor: dict[str, Any],
    recorder_uri: Path,
    recorder_id: str,
    recorder_local_dir: Path,
) -> dict[str, Any]:
    walk_forward = _walk_forward_summary(walk_forward_metrics)
    previous_floor = _previous_model_return_floor_payload(selected.get("test", {}), previous_model_floor)
    baseline_fallback = _baseline_risk_fallback_payload(selected.get("test", {}))
    promotion_decision = _promotion_decision(
        selected,
        comparison_summary,
        walk_forward,
        execution_summary,
        risk_analysis,
    )
    return {
        "experiment": spec.experiment,
        "card_type": CARD_TYPE,
        "status": "complete",
        "promotion_decision": promotion_decision,
        "as_of": bundle.as_of,
        "provider_uri": bundle.provider_meta["provider_uri"],
        "recorder_uri": recorder_uri.as_posix(),
        "recorder_id": recorder_id,
        "recorder_local_dir": recorder_local_dir.as_posix(),
        "market": bundle.manifest["market"],
        "region": bundle.manifest["region"],
        "segments": {"train": list(spec.train), "validation": list(spec.validation), "test": list(bundle.test)},
        "benchmark_basis": "open_to_open",
        "execution_timing": EXECUTION_TIMING,
        "return_col": OPEN_TO_OPEN_RETURN_COL,
        "model_target_col": LABEL_OPEN_TO_OPEN_5D_COL,
        "take_label_col": TAKE_TRADE_LABEL_COL,
        "tail_label_col": TAIL_LOSS_LABEL_COL,
        "qlib_docs": {
            "data": "https://qlib.readthedocs.io/en/latest/component/data.html",
            "workflow": "https://qlib.readthedocs.io/en/latest/component/workflow.html",
            "recorder": "https://qlib.readthedocs.io/en/latest/component/recorder.html",
        },
        "qlib_native_api": QLIB_NATIVE_API,
        "external_method_basis": {
            "meta_labeling": "base signal defines opportunity; ML estimates whether to take it and whether a tail loss is likely",
            "volatility_targeting": "position is risk-budgeted by realized volatility before return maximization",
            "execution_overlay": "15m lagged features cap execution risk; 1m cache is used for open/VWAP stress",
        },
        "model_selection": {
            "source": "qlib_D_features_Alpha158_intraday15m_soxx_constituents",
            "model_families": ["lgbm_classifier", "xgb_classifier", "lgb_xgb_consensus"],
            "candidate_count": int(metrics["candidate"].nunique()),
            "selection_metric": "risk_first_meta_label_risk_budget_objective",
            "selection_column": "selection_objective_score",
            "selected_model": selected["model"],
            "selected_candidate": selected["candidate"],
            "individual_models": "diagnostic_only_selection_disabled",
        },
        "position_model": {
            "workflow": "meta_label_risk_budget_open2open",
            "architecture": "base_opportunity_plus_ml_meta_labels_plus_risk_budget_caps",
            "formula": "opportunity_side * calibrated_edge_score * volatility_budget * tail_risk_cap * drawdown_budget * intraday_execution_cap",
            "base_constrained_ref_role": "opportunity_side_only_not_final_position",
            "selected_candidate": selected,
            "comparison": comparison_summary,
            "risk_gates": _risk_gates_for_spec(spec),
            "walk_forward": walk_forward,
        },
        "latest_signal": latest_signal,
        "meta_label_risk_budget_goal": {
            "target": "replace base_constrained_ref plus probability boost with risk-first meta-label sizing",
            "selected_test_return_pct": selected["test"]["return_pct"],
            "selected_test_max_drawdown_pct": selected["test"]["max_drawdown_pct"],
            "selected_test_max_daily_loss_pct": selected["test"]["max_daily_loss_pct"],
            "selected_test_cvar_5pct_pct": selected["test"].get("cvar_5pct_pct"),
            "selected_test_worst_5_daily_loss_pct": selected["test"].get("worst_5_daily_loss_pct"),
            "base_constrained_ref_test_return_pct": selected["test"].get("baseline_return_pct"),
            "selected_vs_base_constrained_ref_return_gap_pct": comparison_summary.get(
                "selected_vs_base_constrained_ref_return_gap_pct"
            ),
            "baseline_risk_fallback_pass": baseline_fallback["pass"],
            "baseline_risk_fallback": baseline_fallback,
            "previous_model_return_floor_pass": previous_floor["pass"],
            "previous_model_return_floor": previous_floor,
            "walk_forward_gate_pass": walk_forward.get("all_folds_pass_risk_gates"),
            "intraday_execution_status": execution_summary.get("status"),
            "strict_checks": risk_analysis.to_dict(orient="records"),
        },
        "labels": label_diagnostics.to_dict(orient="records"),
        "feature_sets": {
            name: {"feature_count": len(columns), "features": columns}
            for name, columns in bundle.feature_groups.items()
        },
        "intraday_15m_factors": bundle.intraday_15m_factor_summary,
        "constituent_factors": bundle.constituent_factor_summary,
        "feature_importance": {
            "method": "model_native_gain_importance",
            "row_count": int(len(feature_importance)),
            "top_features": feature_importance.head(20).to_dict(orient="records"),
        },
        "benchmark_comparison": comparison.to_dict(orient="records"),
        "walk_forward_metrics": walk_forward_metrics.to_dict(orient="records"),
        "execution_stress": execution_summary,
        "risk_analysis": risk_analysis.to_dict(orient="records"),
        "sidecars": sidecars,
        "artifact_policy": {
            "policy": "slim_card_recorder_first",
            "workflow_config": "workflow.yaml",
            "machine_entrypoint": "summary.json",
            "recorder_only_artifacts": {
                "positions": "artifacts/meta_label_risk_budget_open2open/positions.csv",
                "walk_forward_positions": "artifacts/meta_label_risk_budget_open2open/walk_forward_positions.csv",
            },
        },
        "implementation_notes": {
            "lgb_xgb_required": True,
            "lgb_xgb_consensus_selection": selected["model"] == "lgb_xgb_consensus",
            "old_probability_boost_replaced": True,
            "intraday_15m_role": "lagged_risk_budget_and_execution_overlay",
            "candidate_count": int(metrics["candidate"].nunique()),
            "risk_analysis_pass_count": int(risk_analysis["status"].eq("pass").sum()),
            "risk_analysis_fail_count": int(risk_analysis["status"].eq("fail").sum()),
        },
    }


def _write_summary_files(card: Path, summary: dict[str, Any]) -> None:
    (card / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    (card / "workflow.yaml").write_text(_workflow_yaml(summary), encoding="utf-8")
    (card / "failure_modes.md").write_text(_failure_modes(summary), encoding="utf-8")
    (card / "conclusion.md").write_text(_conclusion(summary), encoding="utf-8")
    _write_notebook(card, summary)


def _workflow_yaml(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"experiment: {summary['experiment']}",
            f"card_type: {summary['card_type']}",
            f"provider_uri: {summary['provider_uri']}",
            f"recorder_uri: {summary['recorder_uri']}",
            f"recorder_id: {summary['recorder_id']}",
            f"as_of: {summary['as_of']}",
            "qlib:",
            f"  market: {summary['market']}",
            f"  region: {summary['region']}",
            "architecture:",
            "  old_structure: base_constrained_ref + ML probability boost + risk gate",
            "  new_structure: base opportunity + ML meta-label + volatility/tail/drawdown/intraday risk budget",
            f"  formula: {summary['position_model']['formula']}",
            "model_selection:",
            f"  return_col: {summary['return_col']}",
            f"  label_col: {summary['model_target_col']}",
            f"  take_label_col: {summary['take_label_col']}",
            f"  tail_label_col: {summary['tail_label_col']}",
            "  models: [lgbm_classifier, xgb_classifier, lgb_xgb_consensus]",
            "  selectable_model: lgb_xgb_consensus",
            "risk_gates:",
            f"  selected_model: {summary['model_selection']['selected_model']}",
            f"  walk_forward_passes_all: {summary['position_model']['walk_forward'].get('all_folds_pass_risk_gates')}",
            f"  baseline_risk_fallback_pass: {summary['meta_label_risk_budget_goal']['baseline_risk_fallback_pass']}",
            f"  previous_model_return_floor_pass: {summary['meta_label_risk_budget_goal']['previous_model_return_floor_pass']}",
            "intraday_15m:",
            f"  provider_uri: {summary['intraday_15m_factors']['provider_uri']}",
            f"  feature_count: {summary['intraday_15m_factors']['feature_count']}",
            f"  role: {summary['intraday_15m_factors']['role']}",
            "artifact_policy:",
            "  policy: slim_card_recorder_first",
            "  workflow_config: workflow.yaml",
            "  machine_entrypoint: summary.json",
            "  recorder_only_artifacts:",
            "    positions: artifacts/meta_label_risk_budget_open2open/positions.csv",
            "    walk_forward_positions: artifacts/meta_label_risk_budget_open2open/walk_forward_positions.csv",
            "",
        ]
    )


def _failure_modes(summary: dict[str, Any]) -> str:
    goal = summary["meta_label_risk_budget_goal"]
    return "\n".join(
        [
            "# Failure Modes",
            "",
            f"- promotion_decision: {summary['promotion_decision']}",
            f"- walk_forward_gate_pass: {goal['walk_forward_gate_pass']}",
            f"- baseline_risk_fallback_pass: {goal['baseline_risk_fallback_pass']}",
            f"- previous_model_return_floor_pass: {goal['previous_model_return_floor_pass']}",
            f"- intraday_execution_status: {goal['intraday_execution_status']}",
            f"- selected_vs_base_constrained_ref_return_gap_pct: {goal['selected_vs_base_constrained_ref_return_gap_pct']}",
            "- primary_architecture_risk: base_constrained_ref 仍决定 opportunity 集合；如果 base 完全错过新 regime，meta-label 只能做 take/skip，不能凭空发现方向。",
            "- intraday_limit: 15m 特征只用 lag1，适合开盘前风控；盘中做 T 仍需要独立的实时执行模型和更细订单/成交验证。",
            "- constituent_limit: SOXX holdings 若不是完整 point-in-time，会残留 survivorship bias。",
            "- execution_gap: 1m stress 是开盘/VWAP 压力测试，不是真实订单簿成交保证。",
            "",
        ]
    )


def _conclusion(summary: dict[str, Any]) -> str:
    selected = summary["position_model"]["selected_candidate"]["test"]
    latest = summary["latest_signal"]
    walk_forward = summary["position_model"]["walk_forward"]
    fallback = summary["meta_label_risk_budget_goal"]["baseline_risk_fallback"]
    return (
        "# 结论\n\n"
        f"- what_ran: 执行 {summary['experiment']}，用 qlib `D.features`/Alpha158、SOXX 成分股、"
        "15m lagged intraday features、LGB/XGB dual meta-label，验证 base opportunity + risk budget 架构。\n"
        f"- key_metrics: selected_return={selected['return_pct']:.4f}%，maxDD={selected['max_drawdown_pct']:.4f}%，"
        f"max_daily_loss={selected['max_daily_loss_pct']:.4f}%，CVaR5={selected.get('cvar_5pct_pct', 0.0):.4f}%。\n"
        f"- base_relative: base_return={fallback['base_return_pct']:.4f}%，return_gap={fallback['selected_vs_base_return_gap_pct']:.4f}%，"
        f"risk_fallback_pass={fallback['pass']}。\n"
        f"- walk_forward: folds={walk_forward['fold_count']}，all_pass={walk_forward['all_folds_pass_risk_gates']}，"
        f"median_return={walk_forward['median_test_return_pct']:.4f}%，worst_daily_loss={walk_forward['worst_test_max_daily_loss_pct']:.4f}%。\n"
        f"- execution: intraday_status={summary['execution_stress']['status']}，coverage_days={summary['execution_stress']['coverage_days']}，"
        "15m 用作 lagged execution cap，1m cache 用作 open/VWAP stress。\n"
        f"- latest_signal: as_of={latest.get('as_of')}，action={latest.get('action')}，"
        f"target_position={latest.get('target_percent', 0.0):.4f}%，tail_prob={latest.get('tail_probability', 0.0):.4f}。\n"
        f"- promotion_decision: {summary['promotion_decision']}。\n"
        "- decision: 旧的 base_constrained_ref + ML probability boost + risk gate 不再作为主方案；本卡把它降级为 opportunity source，并用 tail/vol/drawdown/intraday 预算决定仓位。\n"
        "- risks: 仍需 PIT holdings、更长分钟/订单级执行样本和实盘 paper 验证；不构成买卖建议。\n"
        f"- next_action: 若未晋升，先修 walk-forward/tail gate 失败项，再考虑 regime specialist 或盘中独立执行模型。recorder={summary['recorder_id']}。\n"
    )


def _write_notebook(card: Path, summary: dict[str, Any]) -> None:
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "id": "meta-label-risk-budget-title",
                "metadata": {},
                "source": [
                    f"# {summary['experiment']}\n",
                    "\n",
                    "qlib-native SOXL research card replacing probability boost with LGB/XGB meta-labels, risk-budget sizing, and lagged 15m execution overlay.",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": 1,
                "id": "meta-label-risk-budget-run",
                "metadata": {},
                "outputs": [],
                "source": [
                    "from __future__ import annotations\n",
                    "import json, sys\n",
                    "from pathlib import Path\n",
                    "ROOT = Path.cwd().resolve()\n",
                    "while not (ROOT / 'AGENTS.md').exists() and ROOT != ROOT.parent:\n",
                    "    ROOT = ROOT.parent\n",
                    "if ROOT.as_posix() not in sys.path:\n",
                    "    sys.path.insert(0, ROOT.as_posix())\n",
                    "from src.workflow import run_meta_label_risk_budget_open2open_card\n",
                    "# qlib.init, D.features, DatasetH/DataHandlerLP via build_alpha158_legacy_dataset, R.start\n",
                    "# LGBMClassifier, XGBClassifier, take_trade_prob, tail_loss_prob, intraday15m execution cap\n",
                    "summary = run_meta_label_risk_budget_open2open_card(root=ROOT)\n",
                    "print(json.dumps({'recorder_id': summary['recorder_id'], 'promotion_decision': summary['promotion_decision'], 'as_of': summary['as_of']}, indent=2, sort_keys=True))\n",
                ],
            },
            {
                "cell_type": "markdown",
                "id": "meta-label-risk-budget-conclusion",
                "metadata": {},
                "source": _conclusion(summary).splitlines(keepends=True),
            },
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (card / f"{summary['experiment']}.ipynb").write_text(json.dumps(notebook, indent=1), encoding="utf-8")


def _existing_unique(columns: list[str], frame: pd.DataFrame) -> list[str]:
    return [column for column in dict.fromkeys(columns) if column in frame.columns]


def _jsonable(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in row.items():
        if value is None:
            out[key] = None
            continue
        try:
            if pd.isna(value):
                out[key] = None
                continue
        except (TypeError, ValueError):
            pass
        if isinstance(value, (bool, str, int, float)):
            out[key] = value
        elif isinstance(value, (np.bool_, np.integer, np.floating)):
            out[key] = value.item()
        else:
            out[key] = str(value)
    return out
