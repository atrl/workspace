from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data import QlibProviderConfig, init_qlib_provider, load_qlib_feature_frame
from src.workflow.cognitive_alpha_intraday_overlay import (
    DAILY_EXECUTION_TIMING,
    DAILY_SOURCE_CARD,
    OPEN_BUY_DUMP_THRESHOLD,
    PREDICTION_AVAILABLE_GATE_COL,
    QLIB_DOCS,
    QLIB_NATIVE_API,
    REQUIRED_ANNUAL_WALK_FORWARD_TEST_YEARS,
    REQUIRED_WALK_FORWARD_START,
    SAME_DAY_CLOSE_EXIT_POLICY,
    _json_ready_dict,
    _json_ready_records,
    _load_daily_open_prices,
    _load_daily_selected_positions,
    _read_json,
    _recorder_files,
    _remove_transient_code_snapshots,
    _resolve_root,
    evaluate_observable_delayed_signal_candidates,
    select_validation_robust_delayed_policy,
    simulate_observable_delayed_signal_overlay,
    summarize_execution_risk_diagnostics,
)
from src.workflow.intraday_bar_model import QLIB_FIELDS, _models
from src.workflow.intraday_execution_label_model import (
    ADVERSE_EVENT_RISK_COL,
    LABEL_POLICY,
    MODEL_NAMES,
    build_execution_label_frame,
    _augment_selection as _augment_execution_model_selection,
    _fit_predict_once as _fit_execution_model_once,
)


EXPERIMENT = "cognitive_alpha_intraday_local_fallback_overlay_v1"
INTRADAY_SOURCE_CARD = "research/agentic_alpha_intraday_execution_label_v1"
OVERLAY_SIGNAL_TIMING = "first_30_or_60m_observed_then_15m_bar_close_to_next_bar_open"
REQUIRED_COST_BPS = (20.0, 50.0)
ADVERSE_EVENT_RISK_SCORE_COL = "adverse_event_risk_score"
ADVERSE_EVENT_RISK_SCORE_SOURCE_COL = "adverse_event_risk_score_source"
ADVERSE_EVENT_RISK_PROB_COLS = {
    "lgb": "risk_prob_lgb",
    "xgb": "risk_prob_xgb",
}


@dataclass(frozen=True)
class CognitiveAlphaIntradayLocalFallbackOverlaySpec:
    root: Path
    experiment: str = EXPERIMENT
    market: str = "all"
    region: str = "us"
    as_of: str = "2026-07-02"
    selection_cost_bps: float = 20.0
    required_cost_bps: tuple[float, ...] = REQUIRED_COST_BPS
    source_daily_card: str = DAILY_SOURCE_CARD
    source_intraday_card: str = INTRADAY_SOURCE_CARD

    @property
    def output_dir(self) -> Path:
        return self.root / "research" / self.experiment

    @property
    def recorder_uri(self) -> Path:
        return self.output_dir / "qlib_recorder"

    @property
    def daily_card_dir(self) -> Path:
        return self.root / self.source_daily_card

    @property
    def intraday_card_dir(self) -> Path:
        return self.root / self.source_intraday_card

    @property
    def daily_provider_uri(self) -> Path:
        return self.root / "data" / "qlib" / "soxl"

    @property
    def intraday_provider_uri(self) -> Path:
        return self.root / "data" / "qlib" / "soxl_intraday_15min"

    @property
    def intraday_5min_provider_uri(self) -> Path:
        return self.root / "data" / "qlib" / "soxl_intraday_5min"


def run_cognitive_alpha_intraday_local_fallback_overlay_card(
    *,
    root: Path | str | None = None,
) -> dict[str, Any]:
    from qlib.workflow import R

    spec = CognitiveAlphaIntradayLocalFallbackOverlaySpec(root=_resolve_root(root))
    spec.output_dir.mkdir(parents=True, exist_ok=True)
    spec.recorder_uri.mkdir(parents=True, exist_ok=True)

    daily_summary = _read_json(spec.daily_card_dir / "summary.json")
    intraday_summary = _read_json(spec.intraday_card_dir / "summary.json")
    workflow_yaml = _write_workflow_yaml(spec, daily_summary=daily_summary, intraday_summary=intraday_summary)

    daily_positions = _load_daily_selected_positions(daily_summary, prefer_walk_forward=True)
    as_of = str(daily_summary.get("as_of") or spec.as_of)
    daily_prices = _load_daily_open_prices(
        spec,  # type: ignore[arg-type]
        start=daily_positions["datetime"].min().date().isoformat(),
        end=as_of,
    )
    provider_meta, intraday_frame, feature_cols, label_audit = _load_intraday_label_frame(spec, as_of=as_of)
    fold_predictions, model_selection, feature_importance = _fit_local_intraday_walk_forward(
        intraday_frame,
        feature_cols=feature_cols,
        spec=spec,
        as_of=as_of,
    )
    (
        annual_walk_forward,
        selected_policy_cost_sensitivity,
        selected_trades,
        all_policy_selection,
        risk_analysis,
    ) = _evaluate_local_overlay_walk_forward(
        daily_positions=daily_positions,
        daily_prices=daily_prices,
        fold_predictions=fold_predictions,
        spec=spec,
        as_of=as_of,
    )
    selected_model = _selected_model_payload(annual_walk_forward)
    walk_forward_summary = _walk_forward_summary(annual_walk_forward)
    optimization_plan = _local_fallback_optimization_plan(
        selected_model=selected_model,
        walk_forward_summary=walk_forward_summary,
        local_intraday_manifest=_read_optional_json(spec.intraday_provider_uri / "qlib_store_manifest.json"),
        local_intraday_5min_manifest=_read_optional_json(spec.intraday_5min_provider_uri / "qlib_store_manifest.json"),
        external_context_summary=_read_optional_json(spec.root / "research" / "intraday_external_context_data_v1" / "summary.json"),
    )
    analysis_summary = _analysis_summary(
        spec,
        daily_summary=daily_summary,
        intraday_summary=intraday_summary,
        provider_meta=provider_meta,
        daily_positions=daily_positions,
        daily_prices=daily_prices,
        intraday_frame=intraday_frame,
        fold_predictions=fold_predictions,
        feature_cols=feature_cols,
        label_audit=label_audit,
        model_selection=model_selection,
        annual_walk_forward=annual_walk_forward,
        selected_policy_cost_sensitivity=selected_policy_cost_sensitivity,
        risk_analysis=risk_analysis,
        selected_model=selected_model,
        walk_forward_summary=walk_forward_summary,
        optimization_plan=optimization_plan,
    )
    sidecars = _write_sidecars(
        spec,
        annual_walk_forward=annual_walk_forward,
        model_selection=model_selection,
        selected_policy_cost_sensitivity=selected_policy_cost_sensitivity,
        risk_analysis=risk_analysis,
        feature_importance=feature_importance,
        selected_model=selected_model,
        analysis_summary=analysis_summary,
    )

    upload_dir = Path(tempfile.mkdtemp(prefix=f"{spec.experiment}_recorder_"))
    try:
        execution_dir = upload_dir / "local_fallback_overlay"
        execution_dir.mkdir(parents=True, exist_ok=True)
        fold_predictions.to_csv(execution_dir / "intraday_fold_predictions.csv", index=False)
        all_policy_selection.to_csv(execution_dir / "all_policy_selection.csv.gz", index=False, compression="gzip")
        selected_trades.to_csv(execution_dir / "selected_trades.csv", index=False)
        daily_positions.to_csv(execution_dir / "daily_walk_forward_positions.csv", index=False)

        R.set_uri(spec.recorder_uri.resolve().as_posix())
        with R.start(experiment_name=spec.experiment):
            recorder = R.get_recorder()
            R.log_params(
                experiment=spec.experiment,
                daily_source=spec.source_daily_card,
                intraday_source=spec.source_intraday_card,
                label_policy=LABEL_POLICY,
                selection_segment="validation",
                report_segment="test",
                selection_uses_test_segment=False,
                required_cost_bps="20.0,50.0",
                walk_forward_policy=walk_forward_summary["policy"],
            )
            R.log_metrics(
                feasible_fold_count=float(len(walk_forward_summary["feasible_test_years"])),
                required_policy_satisfied=float(1.0 if walk_forward_summary["required_policy_satisfied"] else 0.0),
                target_pass_fold_count=float(len(walk_forward_summary["target_pass_test_years"])),
                risk_reduction_fold_count=float(len(walk_forward_summary["risk_reduction_test_years"])),
            )
            for file_name in [
                "intraday_fold_predictions.csv",
                "all_policy_selection.csv.gz",
                "selected_trades.csv",
                "daily_walk_forward_positions.csv",
            ]:
                recorder.log_artifact((execution_dir / file_name).as_posix(), artifact_path="local_fallback_overlay")
            for rel_path in sidecars.values():
                recorder.log_artifact((spec.output_dir / rel_path).as_posix(), artifact_path=spec.experiment)
            recorder_id = recorder.id
            recorder_local_dir = Path(recorder.get_local_dir())
    finally:
        shutil.rmtree(upload_dir, ignore_errors=True)

    _remove_transient_code_snapshots(recorder_local_dir)
    recorder_artifacts = _recorder_files(recorder_local_dir)
    promotion_decision = _promotion_decision(walk_forward_summary)
    summary = {
        "experiment": spec.experiment,
        "card_type": "intraday_local_fallback_execution_overlay",
        "workflow_config": workflow_yaml.name,
        "promotion_decision": promotion_decision,
        "as_of": as_of,
        "market": spec.market,
        "region": spec.region,
        "source_daily_model": {
            "card": spec.source_daily_card,
            "recorder_id": daily_summary.get("recorder_id"),
            "execution_timing": daily_summary.get("position_model", {}).get("execution_timing", DAILY_EXECUTION_TIMING),
            "return_col": daily_summary.get("position_model", {}).get("return_col", "open_to_open_return"),
            "has_soxx_constituent_features": any(
                str(col).startswith("soxx_const_")
                for col in daily_summary.get("position_model", {}).get("feature_cols", [])
            ),
        },
        "source_intraday_model": {
            "card": spec.source_intraday_card,
            "recorder_id": intraday_summary.get("recorder_id"),
            "label_policy_name": LABEL_POLICY,
            "feature_set": intraday_summary.get("feature_set", {}),
            "external_context_required": False,
            "external_context": "not_used_local_soxl_soxx_only",
            "adverse_event_risk_head": {
                "target_col": ADVERSE_EVENT_RISK_COL,
                "score_col": ADVERSE_EVENT_RISK_SCORE_COL,
                "prob_cols": ADVERSE_EVENT_RISK_PROB_COLS,
                "models": ["lgb", "xgb", "lgb_xgb_consensus"],
                "score_semantics": "higher_score_means_higher_same_session_open_buy_dump_risk",
                "training_policy": "fit_on_train_only_predict_valid_and_test",
                "selection_uses_test_segment": False,
            },
        },
        "provider_uris": {
            "daily": spec.daily_provider_uri.resolve().as_posix(),
            "15min": spec.intraday_provider_uri.resolve().as_posix(),
            "5min_probe": spec.intraday_5min_provider_uri.resolve().as_posix(),
        },
        "execution_overlay": {
            "frequency": "15min",
            "decision_model_timing": "daily_close_t_signal",
            "daily_execution_timing": DAILY_EXECUTION_TIMING,
            "overlay_signal_timing": OVERLAY_SIGNAL_TIMING,
            "open_to_open_baseline": True,
            "intraday_overlays_hold_overnight": True,
            "intraday_flat_can_close_same_day": True,
            "default_exit_timing": "following_open_or_same_day_close",
            "variants": [
                "baseline_next_open",
                "delayed_signal_after_30m",
                "delayed_signal_after_60m",
                "delayed_signal_after_30m_same_day_close",
                "delayed_signal_after_60m_same_day_close",
            ],
        },
        "model_selection": {
            "models": MODEL_NAMES,
            "adverse_event_risk_head_models": ["lgb", "xgb", "lgb_xgb_consensus"],
            "selection_segment": "validation",
            "report_segment": "test",
            "selection_uses_test_segment": False,
            "required_cost_bps": [float(value) for value in spec.required_cost_bps],
            "return_retention_target_pct": [85.0, 95.0],
            "max_drawdown_floor_pct": -18.0,
            "policy_candidate_rows": int(len(all_policy_selection)),
            "selected_policy_cost_rows": int(len(selected_policy_cost_sensitivity)),
        },
        "walk_forward": walk_forward_summary,
        "selected_model": selected_model,
        "analysis_summary": analysis_summary,
        "optimization_plan": optimization_plan,
        "recorder_id": recorder_id,
        "recorder_uri": spec.recorder_uri.resolve().as_posix(),
        "recorder_local_dir": recorder_local_dir.as_posix(),
        "recorder_artifacts": recorder_artifacts,
        "sidecars": sidecars,
        "artifact_policy": _artifact_policy(spec.experiment, sidecars),
        "qlib_docs": QLIB_DOCS,
        "qlib_native_api": QLIB_NATIVE_API
        + ["LGBMClassifier", "XGBClassifier", "triple_barrier_1h_2h_execution_improvement"],
    }
    (spec.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_conclusion(spec, summary=summary, selected_model=selected_model, annual_walk_forward=annual_walk_forward)
    return summary


def _load_intraday_label_frame(
    spec: CognitiveAlphaIntradayLocalFallbackOverlaySpec,
    *,
    as_of: str,
) -> tuple[dict[str, Any], pd.DataFrame, list[str], dict[str, Any]]:
    config = QlibProviderConfig(
        provider_uri=spec.intraday_provider_uri,
        region=spec.region,
        market=spec.market,
        freq="15min",
        kernels=1,
    )
    provider_meta = init_qlib_provider(config)
    raw = load_qlib_feature_frame(
        config,
        instruments=["SOXL", "SOXX"],
        fields=QLIB_FIELDS,
        start=f"{REQUIRED_WALK_FORWARD_START} 09:45:00",
        end=f"{as_of} 16:00:00",
    )
    frame, feature_cols, label_audit = build_execution_label_frame(
        raw,
        external_raw=None,
        require_external_context=False,
        cost_bps=float(spec.selection_cost_bps),
    )
    return provider_meta, frame, feature_cols, label_audit


def _fit_local_intraday_walk_forward(
    frame: pd.DataFrame,
    *,
    feature_cols: list[str],
    spec: CognitiveAlphaIntradayLocalFallbackOverlaySpec,
    as_of: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prediction_frames: list[pd.DataFrame] = []
    model_rows: list[pd.DataFrame] = []
    importance_rows: list[dict[str, Any]] = []
    bar_prices = frame[["datetime", "date", "soxl_open", "soxl_close"]].rename(
        columns={"soxl_open": "open", "soxl_close": "close"}
    )
    for test_year in REQUIRED_ANNUAL_WALK_FORWARD_TEST_YEARS:
        fold = f"wf_{int(test_year)}"
        train_start = REQUIRED_WALK_FORWARD_START
        train_end = f"{int(test_year) - 2}-12-31"
        valid_start = f"{int(test_year) - 1}-01-01"
        valid_end = f"{int(test_year) - 1}-12-31"
        test_start = f"{int(test_year)}-01-01"
        test_end = as_of if int(test_year) == int(as_of[:4]) else f"{int(test_year)}-12-31"
        train = _date_slice(frame, train_start, train_end)
        valid = _date_slice(frame, valid_start, valid_end)
        test = _date_slice(frame, test_start, test_end)
        metrics, predictions, feature_importance = _fit_execution_model_once(
            {"train": train, "valid": valid, "test": test},
            feature_cols=feature_cols,
            cost_bps=float(spec.selection_cost_bps),
        )
        selection = _augment_execution_model_selection(pd.DataFrame(metrics))
        selection["fold"] = fold
        selection["test_year"] = int(test_year)
        selection["train_start"] = train_start
        selection["train_end"] = train_end
        selection["valid_start"] = valid_start
        selection["valid_end"] = valid_end
        selection["test_start"] = test_start
        selection["test_end"] = test_end
        selection["selection_uses_test_segment"] = False
        selection["selected"] = False
        if not selection.empty:
            selection.loc[selection.index[0], "selected"] = True
        model_rows.append(selection)

        selected_model = str(selection.iloc[0]["model"])
        selected_threshold = float(selection.iloc[0]["threshold"])
        fold_predictions = predictions.loc[predictions["segment"].isin(["valid", "test"])].copy()
        fold_predictions, risk_importance = _append_adverse_event_risk_scores(
            fold_predictions,
            segments={"train": train, "valid": valid, "test": test},
            feature_cols=feature_cols,
        )
        fold_predictions = fold_predictions.merge(bar_prices, on=["datetime", "date"], how="left")
        fold_predictions["walk_forward_fold"] = fold
        fold_predictions["test_year"] = int(test_year)
        fold_predictions["fold_segment"] = fold_predictions["segment"].replace({"valid": "validation"})
        fold_predictions["selected_intraday_model"] = selected_model
        fold_predictions["selected_intraday_threshold"] = selected_threshold
        fold_predictions[PREDICTION_AVAILABLE_GATE_COL] = fold_predictions[f"prob_{selected_model}"].notna()
        prediction_frames.append(fold_predictions)

        for row in feature_importance:
            importance_rows.append(
                {
                    "fold": fold,
                    "test_year": int(test_year),
                    **row,
                }
            )
        for row in risk_importance.to_dict("records"):
            importance_rows.append(
                {
                    "fold": fold,
                    "test_year": int(test_year),
                    **row,
                }
            )

    return (
        pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame(),
        pd.concat(model_rows, ignore_index=True) if model_rows else pd.DataFrame(),
        pd.DataFrame(importance_rows),
    )


def _append_adverse_event_risk_scores(
    predictions: pd.DataFrame,
    *,
    segments: dict[str, pd.DataFrame],
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = predictions.copy()
    if out.empty:
        return out, pd.DataFrame(columns=["model", "feature", "importance"])

    train = segments.get("train", pd.DataFrame()).copy()
    target_col = ADVERSE_EVENT_RISK_COL
    score_cols = list(ADVERSE_EVENT_RISK_PROB_COLS.values()) + [
        ADVERSE_EVENT_RISK_SCORE_COL,
        ADVERSE_EVENT_RISK_SCORE_SOURCE_COL,
    ]
    if target_col not in train.columns:
        for col in score_cols:
            out[col] = np.nan if col != ADVERSE_EVENT_RISK_SCORE_SOURCE_COL else "unavailable_missing_target"
        return out, pd.DataFrame(columns=["model", "feature", "importance"])

    train_target = pd.to_numeric(train[target_col], errors="coerce")
    train_fit = train.loc[train_target.notna(), feature_cols + [target_col]].dropna(subset=feature_cols + [target_col]).copy()
    if train_fit.empty or pd.to_numeric(train_fit[target_col], errors="coerce").nunique() < 2:
        for col in score_cols:
            out[col] = np.nan if col != ADVERSE_EVENT_RISK_SCORE_SOURCE_COL else "unavailable_insufficient_train_classes"
        return out, pd.DataFrame(columns=["model", "feature", "importance"])

    score_frames: list[pd.DataFrame] = []
    importance_rows: list[dict[str, Any]] = []
    fitted_models = _models()
    for model_name, prob_col in ADVERSE_EVENT_RISK_PROB_COLS.items():
        model = fitted_models[model_name]
        model.fit(train_fit[feature_cols], pd.to_numeric(train_fit[target_col], errors="coerce").astype(int))
        if hasattr(model, "feature_importances_"):
            top = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False).head(25)
            importance_rows.extend(
                {
                    "model": f"adverse_event_risk_{model_name}",
                    "feature": feature,
                    "importance": float(importance),
                }
                for feature, importance in top.items()
            )
        prediction_pieces: list[pd.DataFrame] = []
        for segment_name in ["valid", "test"]:
            segment = segments.get(segment_name, pd.DataFrame()).copy()
            if segment.empty:
                continue
            base = segment[["datetime", "date"]].copy()
            base["segment"] = segment_name
            feature_frame = segment[feature_cols].copy()
            valid_features = feature_frame.notna().all(axis=1)
            base[prob_col] = np.nan
            if valid_features.any():
                probabilities = np.asarray(model.predict_proba(feature_frame.loc[valid_features]))
                base.loc[valid_features, prob_col] = probabilities[:, 1]
            prediction_pieces.append(base)
        if prediction_pieces:
            score_frames.append(pd.concat(prediction_pieces, ignore_index=True))

    if not score_frames:
        for col in score_cols:
            out[col] = np.nan if col != ADVERSE_EVENT_RISK_SCORE_SOURCE_COL else "unavailable_no_prediction_segments"
        return out, pd.DataFrame(importance_rows)

    scores = score_frames[0]
    for frame in score_frames[1:]:
        prob_cols = [col for col in frame.columns if col in ADVERSE_EVENT_RISK_PROB_COLS.values()]
        scores = scores.merge(frame[["datetime", "date", "segment"] + prob_cols], on=["datetime", "date", "segment"], how="outer")
    prob_cols = [col for col in ADVERSE_EVENT_RISK_PROB_COLS.values() if col in scores.columns]
    scores[ADVERSE_EVENT_RISK_SCORE_COL] = scores[prob_cols].mean(axis=1)
    scores[ADVERSE_EVENT_RISK_SCORE_SOURCE_COL] = "lgb_xgb_adverse_event_consensus"
    out = out.merge(scores, on=["datetime", "date", "segment"], how="left")
    return out, pd.DataFrame(importance_rows)


def _evaluate_local_overlay_walk_forward(
    *,
    daily_positions: pd.DataFrame,
    daily_prices: pd.DataFrame,
    fold_predictions: pd.DataFrame,
    spec: CognitiveAlphaIntradayLocalFallbackOverlaySpec,
    as_of: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    annual_rows: list[dict[str, Any]] = []
    selected_cost_rows: list[pd.DataFrame] = []
    selected_trade_frames: list[pd.DataFrame] = []
    all_policy_frames: list[pd.DataFrame] = []
    risk_frames: list[pd.DataFrame] = []

    for test_year in REQUIRED_ANNUAL_WALK_FORWARD_TEST_YEARS:
        fold = f"wf_{int(test_year)}"
        train_start = REQUIRED_WALK_FORWARD_START
        train_end = f"{int(test_year) - 2}-12-31"
        valid_start = f"{int(test_year) - 1}-01-01"
        valid_end = f"{int(test_year) - 1}-12-31"
        test_start = f"{int(test_year)}-01-01"
        test_end = as_of if int(test_year) == int(as_of[:4]) else f"{int(test_year)}-12-31"
        fold_daily = daily_positions.loc[daily_positions["walk_forward_fold"].astype(str).eq(fold)].copy()
        fold_intraday = fold_predictions.loc[fold_predictions["walk_forward_fold"].astype(str).eq(fold)].copy()
        if fold_daily.empty or fold_intraday.empty:
            annual_rows.append(
                _infeasible_fold_row(
                    fold=fold,
                    test_year=int(test_year),
                    train_start=train_start,
                    train_end=train_end,
                    valid_start=valid_start,
                    valid_end=valid_end,
                    test_start=test_start,
                    test_end=test_end,
                    reason="missing_daily_or_intraday_fold_data",
                )
            )
            continue
        selected_intraday_model = str(fold_intraday["selected_intraday_model"].dropna().iloc[0])
        prob_col = f"prob_{selected_intraday_model}"
        policy_selection = evaluate_observable_delayed_signal_candidates(
            fold_daily,
            daily_prices=daily_prices,
            intraday=fold_intraday,
            cost_bps_values=tuple(float(value) for value in spec.required_cost_bps),
            prob_col=prob_col,
            gate_col=PREDICTION_AVAILABLE_GATE_COL,
        )
        if not policy_selection.empty:
            policy_selection["fold"] = fold
            policy_selection["test_year"] = int(test_year)
            policy_selection["selected_intraday_model"] = selected_intraday_model
            all_policy_frames.append(policy_selection)
        selected_policy = select_validation_robust_delayed_policy(
            policy_selection,
            required_cost_bps=tuple(float(value) for value in spec.required_cost_bps),
            selection_cost_bps=float(spec.selection_cost_bps),
        )
        selected_policy_rows = (
            policy_selection.loc[policy_selection["policy_id"].eq(selected_policy.get("policy_id"))].copy()
            if not policy_selection.empty and selected_policy.get("policy_id")
            else pd.DataFrame()
        )
        if not selected_policy_rows.empty:
            selected_policy_rows["fold"] = fold
            selected_policy_rows["test_year"] = int(test_year)
            selected_policy_rows["selected_intraday_model"] = selected_intraday_model
            selected_policy_rows["selection_uses_test_segment"] = False
            selected_cost_rows.append(selected_policy_rows)

        selected_trades = _simulate_selected_policy_trades(
            selected_policy,
            fold_daily=fold_daily,
            daily_prices=daily_prices,
            fold_intraday=fold_intraday,
            required_cost_bps=tuple(float(value) for value in spec.required_cost_bps),
            prob_col=prob_col,
        )
        if not selected_trades.empty:
            selected_trades["walk_forward_fold"] = fold
            selected_trades["test_year"] = int(test_year)
            selected_trade_frames.append(selected_trades)
            fold_risk = summarize_execution_risk_diagnostics(selected_trades)
            if not fold_risk.empty:
                fold_risk["fold"] = fold
                fold_risk["test_year"] = int(test_year)
                risk_frames.append(fold_risk)

        annual_rows.append(
            _annual_fold_row(
                selected_policy,
                selected_policy_rows=selected_policy_rows,
                selected_trades=selected_trades,
                risk_analysis=fold_risk if not selected_trades.empty and not fold_risk.empty else pd.DataFrame(),
                fold=fold,
                test_year=int(test_year),
                train_start=train_start,
                train_end=train_end,
                valid_start=valid_start,
                valid_end=valid_end,
                test_start=test_start,
                test_end=test_end,
                selected_intraday_model=selected_intraday_model,
                selected_intraday_threshold=float(fold_intraday["selected_intraday_threshold"].dropna().iloc[0]),
                intraday_prediction_rows=int(fold_intraday[prob_col].notna().sum()),
                policy_candidate_rows=int(len(policy_selection)),
            )
        )

    return (
        pd.DataFrame(annual_rows),
        pd.concat(selected_cost_rows, ignore_index=True) if selected_cost_rows else pd.DataFrame(),
        pd.concat(selected_trade_frames, ignore_index=True) if selected_trade_frames else pd.DataFrame(),
        pd.concat(all_policy_frames, ignore_index=True) if all_policy_frames else pd.DataFrame(),
        pd.concat(risk_frames, ignore_index=True) if risk_frames else pd.DataFrame(),
    )


def _simulate_selected_policy_trades(
    selected_policy: dict[str, Any],
    *,
    fold_daily: pd.DataFrame,
    daily_prices: pd.DataFrame,
    fold_intraday: pd.DataFrame,
    required_cost_bps: tuple[float, ...],
    prob_col: str,
) -> pd.DataFrame:
    if not selected_policy.get("policy_id"):
        return pd.DataFrame()
    risk_threshold = selected_policy.get("risk_threshold")
    risk_threshold = None if risk_threshold is None or pd.isna(risk_threshold) else float(risk_threshold)
    frames = []
    for cost_bps in required_cost_bps:
        frames.append(
            simulate_observable_delayed_signal_overlay(
                fold_daily,
                daily_prices=daily_prices,
                intraday=fold_intraday,
                observation_bars=int(selected_policy["observation_bars"]),
                condition_feature=str(selected_policy["condition_feature"]),
                condition_operator=str(selected_policy["condition_operator"]),
                condition_threshold=float(selected_policy["condition_threshold"]),
                signal_threshold=float(selected_policy["signal_threshold"]),
                risk_threshold=risk_threshold,
                exit_policy=str(selected_policy.get("exit_policy") or "hold_to_next_open"),
                cost_bps=float(cost_bps),
                prob_col=prob_col,
                gate_col=PREDICTION_AVAILABLE_GATE_COL,
            ).assign(cost_bps=float(cost_bps))
        )
    trades = pd.concat(frames, ignore_index=True)
    return _add_entry_day_strategy_returns(trades, intraday=fold_intraday)


def _add_entry_day_strategy_returns(trades: pd.DataFrame, *, intraday: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return trades
    result = trades.copy()
    bars = intraday.copy()
    bars["date"] = pd.to_datetime(bars["date"]).dt.date.astype(str)
    bars["datetime"] = pd.to_datetime(bars["datetime"])
    entry_day_close = (
        bars.sort_values("datetime")
        .groupby("date", as_index=True)
        .agg(entry_day_close_time=("datetime", "last"), entry_day_close_price=("close", "last"))
    )
    entry_dates = pd.to_datetime(result["entry_date"], errors="coerce").dt.date.astype(str)
    missing_entry_date = entry_dates.eq("NaT")
    fallback_dates = pd.to_datetime(result["entry_time"], errors="coerce").dt.date.astype(str)
    entry_dates = entry_dates.where(~missing_entry_date, fallback_dates)
    result["entry_day_close_time"] = entry_dates.map(entry_day_close["entry_day_close_time"])
    result["entry_day_close_price"] = pd.to_numeric(
        entry_dates.map(entry_day_close["entry_day_close_price"]),
        errors="coerce",
    )
    target = pd.to_numeric(result.get("target_position"), errors="coerce").fillna(0.0)
    entry_price = pd.to_numeric(result.get("entry_price"), errors="coerce")
    cost_bps = pd.to_numeric(result.get("cost_bps"), errors="coerce").fillna(0.0)
    result["entry_day_strategy_return"] = (
        target * (result["entry_day_close_price"] / entry_price - 1.0)
        - target.abs() * cost_bps / 10000.0
    )
    result.loc[entry_price.isna() | result["entry_day_close_price"].isna(), "entry_day_strategy_return"] = np.nan
    return result


def _annual_fold_row(
    selected_policy: dict[str, Any],
    *,
    selected_policy_rows: pd.DataFrame,
    selected_trades: pd.DataFrame,
    risk_analysis: pd.DataFrame,
    fold: str,
    test_year: int,
    train_start: str,
    train_end: str,
    valid_start: str,
    valid_end: str,
    test_start: str,
    test_end: str,
    selected_intraday_model: str,
    selected_intraday_threshold: float,
    intraday_prediction_rows: int,
    policy_candidate_rows: int,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "fold": fold,
        "test_year": int(test_year),
        "status": "ok" if selected_policy.get("policy_id") else "infeasible",
        "blocking_reason": selected_policy.get("blocking_reason") or ("ok" if selected_policy.get("policy_id") else "missing_policy"),
        "required_cost_bps": ",".join(str(float(value)) for value in REQUIRED_COST_BPS),
        "train_start": train_start,
        "train_end": train_end,
        "valid_start": valid_start,
        "valid_end": valid_end,
        "test_start": test_start,
        "test_end": test_end,
        "selected_intraday_model": selected_intraday_model,
        "selected_intraday_threshold": float(selected_intraday_threshold),
        "intraday_prediction_rows": int(intraday_prediction_rows),
        "policy_candidate_rows": int(policy_candidate_rows),
        "selected_policy_id": selected_policy.get("policy_id"),
        "selected_variant": selected_policy.get("variant"),
        "exit_policy": selected_policy.get("exit_policy"),
        "observation_minutes": selected_policy.get("observation_minutes"),
        "condition_feature": selected_policy.get("condition_feature"),
        "condition_operator": selected_policy.get("condition_operator"),
        "condition_threshold": selected_policy.get("condition_threshold"),
        "signal_threshold": selected_policy.get("signal_threshold"),
        "selection_uses_test_segment": False,
        "validation_passes_target": bool(selected_policy.get("validation_passes_target", False)),
        "test_passes_target": bool(selected_policy.get("passes_test_target", False)),
    }
    if selected_policy_rows.empty:
        return row

    required = selected_policy_rows.loc[selected_policy_rows["cost_bps"].isin(list(REQUIRED_COST_BPS))].copy()
    row["required_cost_validation_target_count"] = int(required["passes_validation_target"].astype(bool).sum())
    row["required_cost_test_target_count"] = int(required["passes_test_target"].astype(bool).sum())
    row["required_cost_validation_and_test_target_count"] = int(
        required["passes_validation_and_test_target"].astype(bool).sum()
    )
    for _, cost_row in required.sort_values("cost_bps").iterrows():
        cost_label = f"{float(cost_row['cost_bps']):.0f}bp"
        for column in [
            "validation_return_pct",
            "validation_return_retention_pct",
            "validation_max_drawdown_pct",
            "validation_skipped_periods",
            "test_return_pct",
            "test_return_retention_pct",
            "test_max_drawdown_pct",
            "test_skipped_periods",
            "passes_validation_target",
            "passes_test_target",
            "passes_validation_and_test_target",
        ]:
            if column in cost_row:
                row[f"{cost_label}_{column}"] = cost_row[column]

    if not risk_analysis.empty:
        selected_variant = str(selected_policy.get("variant"))
        for _, risk_row in risk_analysis.loc[
            risk_analysis["variant"].eq(selected_variant)
            & risk_analysis["cost_bps"].isin(list(REQUIRED_COST_BPS))
            & risk_analysis["segment"].eq("test")
        ].iterrows():
            cost_label = f"{float(risk_row['cost_bps']):.0f}bp"
            for column in [
                "open_buy_dump_period_reduction",
                "crash_period_reduction",
                "worst_entry_day_return_improvement_pct",
                "worst_strategy_return_improvement_pct",
            ]:
                row[f"{cost_label}_test_{column}"] = risk_row[column]
    row["risk_reduction_passes_target"] = bool(
        max(
            float(row.get("20bp_test_open_buy_dump_period_reduction") or 0.0),
            float(row.get("50bp_test_open_buy_dump_period_reduction") or 0.0),
            float(row.get("20bp_test_crash_period_reduction") or 0.0),
            float(row.get("50bp_test_crash_period_reduction") or 0.0),
        )
        > 0.0
    )
    row["test_passes_required_cost_target"] = bool(row.get("required_cost_test_target_count", 0) >= len(REQUIRED_COST_BPS))
    row["validation_passes_required_cost_target"] = bool(
        row.get("required_cost_validation_target_count", 0) >= len(REQUIRED_COST_BPS)
    )
    row["validation_and_test_passes_required_cost_target"] = bool(
        row.get("required_cost_validation_and_test_target_count", 0) >= len(REQUIRED_COST_BPS)
    )
    return row


def _infeasible_fold_row(
    *,
    fold: str,
    test_year: int,
    train_start: str,
    train_end: str,
    valid_start: str,
    valid_end: str,
    test_start: str,
    test_end: str,
    reason: str,
) -> dict[str, Any]:
    return {
        "fold": fold,
        "test_year": int(test_year),
        "status": "infeasible",
        "blocking_reason": reason,
        "required_cost_bps": ",".join(str(float(value)) for value in REQUIRED_COST_BPS),
        "train_start": train_start,
        "train_end": train_end,
        "valid_start": valid_start,
        "valid_end": valid_end,
        "test_start": test_start,
        "test_end": test_end,
        "selected_intraday_model": None,
        "selection_uses_test_segment": False,
        "validation_passes_target": False,
        "test_passes_target": False,
        "risk_reduction_passes_target": False,
    }


def _selected_model_payload(annual_walk_forward: pd.DataFrame) -> dict[str, Any]:
    if annual_walk_forward.empty:
        return {
            "policy_id": None,
            "selection_uses_test_segment": False,
            "validation_passes_target": False,
            "passes_test_target": False,
        }
    latest = annual_walk_forward.sort_values("test_year").iloc[-1].to_dict()
    payload = _json_ready_dict(latest)
    payload["selection_segment"] = "validation"
    payload["report_segment"] = "test"
    payload["selection_uses_test_segment"] = False
    payload["selected_by"] = "latest_fold_policy_shape_with_annual_walk_forward_summary"
    payload["validation_passes_target"] = bool(payload.get("validation_passes_required_cost_target", False))
    payload["passes_test_target"] = bool(payload.get("test_passes_required_cost_target", False))
    payload["variant"] = payload.get("selected_variant")
    payload["test_return_retention_pct"] = payload.get("20bp_test_return_retention_pct")
    payload["test_max_drawdown_pct"] = payload.get("20bp_test_max_drawdown_pct")
    return payload


def _walk_forward_summary(annual_walk_forward: pd.DataFrame) -> dict[str, Any]:
    required_years = [int(year) for year in REQUIRED_ANNUAL_WALK_FORWARD_TEST_YEARS]
    feasible_years = (
        sorted(int(value) for value in annual_walk_forward.loc[annual_walk_forward["status"].eq("ok"), "test_year"].tolist())
        if not annual_walk_forward.empty
        else []
    )
    target_pass_years = (
        sorted(
            int(value)
            for value in annual_walk_forward.loc[
                annual_walk_forward.get(
                    "validation_and_test_passes_required_cost_target",
                    pd.Series(False, index=annual_walk_forward.index),
                ).fillna(False).astype(bool),
                "test_year",
            ].tolist()
        )
        if not annual_walk_forward.empty
        else []
    )
    risk_years = (
        sorted(
            int(value)
            for value in annual_walk_forward.loc[
                annual_walk_forward.get("risk_reduction_passes_target", pd.Series(False, index=annual_walk_forward.index))
                .fillna(False)
                .astype(bool),
                "test_year",
            ].tolist()
        )
        if not annual_walk_forward.empty
        else []
    )
    selection_uses_test = bool(
        annual_walk_forward.get("selection_uses_test_segment", pd.Series(False, index=annual_walk_forward.index))
        .fillna(False)
        .astype(bool)
        .any()
    )
    blocking_reasons = (
        sorted(
            {
                str(value)
                for value in annual_walk_forward.get("blocking_reason", pd.Series(dtype=object)).dropna().tolist()
                if str(value) != "ok"
            }
        )
        if not annual_walk_forward.empty
        else ["missing_walk_forward"]
    )
    required_policy_satisfied = bool(
        not selection_uses_test
        and set(required_years).issubset(feasible_years)
        and set(required_years).issubset(target_pass_years)
        and set(required_years).issubset(risk_years)
    )
    if not set(required_years).issubset(target_pass_years):
        blocking_reasons.append("annual_target_gate_failed")
    if not set(required_years).issubset(risk_years):
        blocking_reasons.append("annual_risk_reduction_gate_failed")
    return {
        "policy": "annual_train_to_T_minus_2_valid_T_minus_1_test_T",
        "required_start": REQUIRED_WALK_FORWARD_START,
        "required_test_years": required_years,
        "feasible_test_years": feasible_years,
        "target_pass_test_years": target_pass_years,
        "risk_reduction_test_years": risk_years,
        "missing_required_test_years": [year for year in required_years if year not in set(feasible_years)],
        "rows": int(len(annual_walk_forward)),
        "selection_uses_test_segment": selection_uses_test,
        "required_policy_satisfied": required_policy_satisfied,
        "blocking_reasons": sorted(set(blocking_reasons)),
    }


def _analysis_summary(
    spec: CognitiveAlphaIntradayLocalFallbackOverlaySpec,
    *,
    daily_summary: dict[str, Any],
    intraday_summary: dict[str, Any],
    provider_meta: dict[str, Any],
    daily_positions: pd.DataFrame,
    daily_prices: pd.DataFrame,
    intraday_frame: pd.DataFrame,
    fold_predictions: pd.DataFrame,
    feature_cols: list[str],
    label_audit: dict[str, Any],
    model_selection: pd.DataFrame,
    annual_walk_forward: pd.DataFrame,
    selected_policy_cost_sensitivity: pd.DataFrame,
    risk_analysis: pd.DataFrame,
    selected_model: dict[str, Any],
    walk_forward_summary: dict[str, Any],
    optimization_plan: dict[str, Any],
) -> dict[str, Any]:
    return {
        "source": "local_soxl_soxx_15m_lgb_xgb_triple_barrier_execution_overlay",
        "as_of": spec.as_of,
        "selection_uses_test_segment": False,
        "why_intraday_did_not_beat_daily": [
            "intraday_label_targets_1h_2h_execution_quality_not_multi_day_open_to_open_trend",
            "annual_local_intraday_model_folds_are_unstable_before_overlay",
            "delayed_entries_reduce_bad_open_risk_but_can_skip_the_daily_model_trend_source",
            "cost_and_same_day_close_reduce_overnight_alpha_capture",
        ],
        "data_audit": {
            "daily_walk_forward_position_rows": int(len(daily_positions)),
            "daily_walk_forward_start": daily_positions["datetime"].min().date().isoformat(),
            "daily_walk_forward_end": daily_positions["datetime"].max().date().isoformat(),
            "daily_price_rows": int(len(daily_prices)),
            "intraday_label_rows": int(len(intraday_frame)),
            "intraday_prediction_rows": int(len(fold_predictions)),
            "adverse_event_risk_prediction_rows": int(
                fold_predictions.get(ADVERSE_EVENT_RISK_SCORE_COL, pd.Series(dtype=float)).notna().sum()
            ),
            "intraday_provider_uri": provider_meta.get("provider_uri"),
            "intraday_calendar_start": provider_meta.get("calendar_start"),
            "intraday_calendar_end": provider_meta.get("calendar_end"),
        },
        "label_policy": label_audit,
        "feature_set": {
            "feature_count": int(len(feature_cols)),
            "features": feature_cols,
            "local_soxl_soxx_only": True,
            "external_context_required": False,
        },
        "source_daily_model": {
            "recorder_id": daily_summary.get("recorder_id"),
            "execution_timing": daily_summary.get("position_model", {}).get("execution_timing"),
            "feature_count": daily_summary.get("position_model", {}).get("feature_count"),
            "has_soxx_constituent_features": any(
                str(col).startswith("soxx_const_")
                for col in daily_summary.get("position_model", {}).get("feature_cols", [])
            ),
        },
        "source_intraday_model": {
            "recorder_id": intraday_summary.get("recorder_id"),
            "label_policy": intraday_summary.get("label_policy"),
            "adverse_event_risk_head": {
                "target_col": ADVERSE_EVENT_RISK_COL,
                "score_col": ADVERSE_EVENT_RISK_SCORE_COL,
                "prob_cols": ADVERSE_EVENT_RISK_PROB_COLS,
                "score_semantics": "higher_score_means_higher_same_session_open_buy_dump_risk",
                "training_policy": "fit_on_train_only_predict_valid_and_test",
            },
        },
        "model_selection_rows": int(len(model_selection)),
        "annual_walk_forward_rows": int(len(annual_walk_forward)),
        "selected_policy_cost_rows": int(len(selected_policy_cost_sensitivity)),
        "risk_analysis_rows": int(len(risk_analysis)),
        "selected_model": selected_model,
        "walk_forward": walk_forward_summary,
        "optimization_plan": optimization_plan,
    }


def _local_fallback_optimization_plan(
    *,
    selected_model: dict[str, Any],
    walk_forward_summary: dict[str, Any],
    local_intraday_manifest: dict[str, Any] | None,
    local_intraday_5min_manifest: dict[str, Any] | None,
    external_context_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    required = set(walk_forward_summary.get("required_test_years", []))
    target_years = set(walk_forward_summary.get("target_pass_test_years", []))
    risk_years = set(walk_forward_summary.get("risk_reduction_test_years", []))
    local_intraday_manifest = local_intraday_manifest or {}
    local_intraday_5min_manifest = local_intraday_5min_manifest or {}
    external_context_summary = external_context_summary or {}
    if not required.issubset(set(walk_forward_summary.get("feasible_test_years", []))):
        status = "needs_data_or_daily_replay_coverage"
        primary_next_action = "repair_missing_annual_fold_coverage"
    elif not required.issubset(target_years):
        status = "needs_new_intraday_factor_or_label"
        primary_next_action = "retrain_with_5min_replay_and_external_lead_lag_before_policy_search"
    elif not required.issubset(risk_years):
        status = "needs_cost_robust_risk_sleeve"
        primary_next_action = "optimize_risk_sleeve_for_open_buy_dump_and_crash_reduction"
    else:
        status = "needs_execution_replay"
        primary_next_action = "run_bid_ask_latency_replay"
    return {
        "status": status,
        "primary_next_action": primary_next_action,
        "guardrails": [
            "selection_uses_validation_only",
            "test_segment_report_only",
            "retain_85_to_95_pct_return_band",
            "max_drawdown_above_minus_18_pct",
            "do_not_promote_without_20bp_50bp_annual_pass",
        ],
        "why_not_promoted": walk_forward_summary.get("blocking_reasons", []),
        "selected_overlay_status": {
            "variant": selected_model.get("variant"),
            "test_return_retention_pct": selected_model.get("test_return_retention_pct"),
            "test_max_drawdown_pct": selected_model.get("test_max_drawdown_pct"),
            "annual_walk_forward_required_policy_satisfied": bool(walk_forward_summary.get("required_policy_satisfied")),
            "target_pass_test_years": sorted(target_years),
            "risk_reduction_test_years": sorted(risk_years),
        },
        "local_intraday_history": {
            "status": local_intraday_manifest.get("status"),
            "freq": local_intraday_manifest.get("freq"),
            "calendar_start": local_intraday_manifest.get("calendar_start"),
            "calendar_end": local_intraday_manifest.get("calendar_end"),
            "instruments": local_intraday_manifest.get("instruments", []),
            "supports_5min_probe": bool(local_intraday_5min_manifest.get("status") == "ok"),
            "freq_5min": local_intraday_5min_manifest.get("freq"),
            "calendar_start_5min": local_intraday_5min_manifest.get("calendar_start"),
            "calendar_end_5min": local_intraday_5min_manifest.get("calendar_end"),
        },
        "external_context_gap": external_context_summary.get("coverage_gap_analysis", {}),
        "ranked_actions": [
            {
                "rank": 1,
                "action": "retrain_with_5min_replay_and_external_lead_lag_before_policy_search",
                "reason": "Local SOXL/SOXX 15m annual folds are feasible but fail return-retention and drawdown targets.",
            },
            {
                "rank": 2,
                "action": "separate_risk_sleeve_from_alpha_entry_gate",
                "reason": "The delayed entry gate cuts crashes in some years but skips too much daily trend return.",
            },
            {
                "rank": 3,
                "action": "run_bid_ask_latency_replay_after_annual_target_passes",
                "reason": "Execution replay is useful only after annual 20/50bp policy gates pass.",
            },
        ],
    }


def _write_sidecars(
    spec: CognitiveAlphaIntradayLocalFallbackOverlaySpec,
    *,
    annual_walk_forward: pd.DataFrame,
    model_selection: pd.DataFrame,
    selected_policy_cost_sensitivity: pd.DataFrame,
    risk_analysis: pd.DataFrame,
    feature_importance: pd.DataFrame,
    selected_model: dict[str, Any],
    analysis_summary: dict[str, Any],
) -> dict[str, str]:
    sidecars = {
        "annual_walk_forward": "annual_walk_forward.csv",
        "model_selection": "model_selection.csv",
        "selected_policy_cost_sensitivity": "selected_policy_cost_sensitivity.csv",
        "risk_analysis": "risk_analysis.csv",
        "feature_importance": "feature_importance.csv",
        "selected_model": "selected_model.json",
        "analysis_summary": "analysis_summary.json",
        "failure_modes": "failure_modes.md",
    }
    annual_walk_forward.to_csv(spec.output_dir / sidecars["annual_walk_forward"], index=False)
    model_selection.to_csv(spec.output_dir / sidecars["model_selection"], index=False)
    selected_policy_cost_sensitivity.to_csv(
        spec.output_dir / sidecars["selected_policy_cost_sensitivity"],
        index=False,
    )
    risk_analysis.to_csv(spec.output_dir / sidecars["risk_analysis"], index=False)
    feature_importance.to_csv(spec.output_dir / sidecars["feature_importance"], index=False)
    (spec.output_dir / sidecars["selected_model"]).write_text(
        json.dumps(selected_model, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (spec.output_dir / sidecars["analysis_summary"]).write_text(
        json.dumps(analysis_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    failure_modes = (
        "# Failure Modes\n\n"
        "- 本地 fallback 只用 SOXL/SOXX 15m 特征；没有 QQQ/SMH/成分股分钟级上下文，因此不能证明外部 lead-lag 因子。\n"
        "- 每个年度 fold 独立训练 LGB/XGB/consensus，valid 只用于选择 intraday 模型和 delayed policy，test 只报告。\n"
        "- adverse_event_risk_head 单独训练买入后同日 open-buy-dump 风险，只用 train 拟合并只给 valid/test 输出风险分数；仍不是晋升信号。\n"
        "- 日内模型的目标是 1h/2h triple-barrier execution improvement，不是日线 open-to-open 趋势收益；若它跳过入场，可能降低回撤但也会丢掉日线 alpha。\n"
        "- same_day_close 是可交易时序更纯的候选，但仍缺 bid/ask、收盘成交容量、延迟、盘前盘后和极端波动 replay。\n"
        "- 晋升必须同时满足 2023-2026 年度 20/50bp 的 85%-95% retention、max_dd>-18%、且降低 open-buy-dump 或 crash 暴露。\n"
    )
    (spec.output_dir / sidecars["failure_modes"]).write_text(failure_modes, encoding="utf-8")
    return sidecars


def _write_workflow_yaml(
    spec: CognitiveAlphaIntradayLocalFallbackOverlaySpec,
    *,
    daily_summary: dict[str, Any],
    intraday_summary: dict[str, Any],
) -> Path:
    path = spec.output_dir / "workflow.yaml"
    path.write_text(
        "\n".join(
            [
                f"experiment: {spec.experiment}",
                "qlib:",
                f"  recorder_uri: {spec.recorder_uri.resolve().as_posix()}",
                f"  daily_provider_uri: {spec.daily_provider_uri.resolve().as_posix()}",
                f"  intraday_provider_uri: {spec.intraday_provider_uri.resolve().as_posix()}",
                f"  intraday_5min_probe_provider_uri: {spec.intraday_5min_provider_uri.resolve().as_posix()}",
                "sources:",
                f"  daily_card: {spec.source_daily_card}",
                f"  daily_recorder_id: {daily_summary.get('recorder_id')}",
                f"  intraday_label_card: {spec.source_intraday_card}",
                f"  intraday_label_recorder_id: {intraday_summary.get('recorder_id')}",
                "label_policy:",
                f"  name: {LABEL_POLICY}",
                "  source: local_soxl_soxx_15m",
                "model_selection:",
                "  models: [lgb, xgb, lgb_xgb_consensus]",
                "  adverse_event_risk_head_models: [lgb, xgb, lgb_xgb_consensus]",
                f"  adverse_event_risk_target: {ADVERSE_EVENT_RISK_COL}",
                f"  adverse_event_risk_score_col: {ADVERSE_EVENT_RISK_SCORE_COL}",
                "  adverse_event_risk_training_policy: fit_on_train_only_predict_valid_and_test",
                "  selection_segment: validation",
                "  report_segment: test",
                "  selection_uses_test_segment: false",
                "  required_cost_bps: [20.0, 50.0]",
                "execution:",
                f"  daily_timing: {DAILY_EXECUTION_TIMING}",
                f"  overlay_signal_timing: {OVERLAY_SIGNAL_TIMING}",
                f"  same_day_close_candidate: {SAME_DAY_CLOSE_EXIT_POLICY}",
                "walk_forward:",
                "  policy: annual_train_to_T_minus_2_valid_T_minus_1_test_T",
                f"  required_start: {REQUIRED_WALK_FORWARD_START}",
                f"  required_test_years: {list(REQUIRED_ANNUAL_WALK_FORWARD_TEST_YEARS)}",
                "artifact_policy:",
                "  policy: slim_card_recorder_first",
                "  machine_entrypoint: summary.json",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def _artifact_policy(experiment: str, sidecars: dict[str, str]) -> dict[str, Any]:
    return {
        "policy": "slim_card_recorder_first",
        "primary_artifact": f"{experiment}.ipynb",
        "machine_entrypoint": "summary.json",
        "workflow_config": "workflow.yaml",
        "kept_sidecars": sorted(sidecars.values()),
        "recorder_only_artifacts": {
            "intraday_fold_predictions": "artifacts/local_fallback_overlay/intraday_fold_predictions.csv",
            "all_policy_selection": "artifacts/local_fallback_overlay/all_policy_selection.csv.gz",
            "selected_trades": "artifacts/local_fallback_overlay/selected_trades.csv",
            "daily_walk_forward_positions": "artifacts/local_fallback_overlay/daily_walk_forward_positions.csv",
        },
        "reproducible_from_recorder": True,
    }


def _promotion_decision(walk_forward_summary: dict[str, Any]) -> str:
    if walk_forward_summary.get("selection_uses_test_segment"):
        return "not_promoted_selection_leakage_detected"
    if not set(walk_forward_summary.get("required_test_years", [])).issubset(
        set(walk_forward_summary.get("feasible_test_years", []))
    ):
        return "not_promoted_local_intraday_walk_forward_incomplete"
    if not set(walk_forward_summary.get("required_test_years", [])).issubset(
        set(walk_forward_summary.get("target_pass_test_years", []))
    ):
        return "not_promoted_local_intraday_target_gate_failed"
    if not set(walk_forward_summary.get("required_test_years", [])).issubset(
        set(walk_forward_summary.get("risk_reduction_test_years", []))
    ):
        return "not_promoted_local_intraday_risk_reduction_unstable"
    return "not_promoted_intraday_replay_missing"


def _write_conclusion(
    spec: CognitiveAlphaIntradayLocalFallbackOverlaySpec,
    *,
    summary: dict[str, Any],
    selected_model: dict[str, Any],
    annual_walk_forward: pd.DataFrame,
) -> None:
    pass_count = len(summary["walk_forward"].get("target_pass_test_years", []))
    risk_count = len(summary["walk_forward"].get("risk_reduction_test_years", []))
    latest = annual_walk_forward.sort_values("test_year").iloc[-1].to_dict() if not annual_walk_forward.empty else {}
    text = (
        "# 中文结论\n\n"
        "- 做了什么：建立本地 SOXL/SOXX 15m fallback overlay。日线 LGB/XGB open-to-open 模型仍决定 target_position；"
        "本卡每个年度 fold 重新训练本地 15m LGB/XGB/consensus triple-barrier 模型，并额外训练同日 open-buy-dump adverse-event 风险头，"
        "再用 validation-only delayed policy 决定是否延迟入场或 same_day_close。\n"
        "- 为什么之前日内模型没有超过日线模型：日内 label 学的是 1h/2h 执行改善，不是多日 open-to-open 趋势；"
        "延迟入场能规避部分开盘后下杀，但也会错过 SOXL 日线模型主要依赖的隔夜和趋势段收益。\n"
        f"- 年度 walk-forward：2023-2026 都完成 train_to_T-2 / valid_T-1 / test_T；"
        f"20/50bp 同时过 85%-95% retention 与 max_dd>-18% 的年份数={pass_count}；"
        f"同时降低 open-buy-dump 或 crash 暴露的年份数={risk_count}。\n"
        f"- 最新 fold：{latest.get('fold')} 选 {latest.get('selected_intraday_model')} + {latest.get('selected_variant')}，"
        f"20bp test retention={float(latest.get('20bp_test_return_retention_pct') or 0.0):.2f}%，"
        f"test max_dd={float(latest.get('20bp_test_max_drawdown_pct') or 0.0):.2f}%，"
        f"open-buy-dump reduction={float(latest.get('20bp_test_open_buy_dump_period_reduction') or 0.0):.0f}。\n"
        f"- 是否晋升：{summary['promotion_decision']}。当前优化方向应该是保留日线模型作为主仓位，"
        "把日内层当作风险/执行 sleeve；要超过日线模型，需要更稳定的外部 lead-lag/成分股分钟上下文或 bid/ask replay 证据，而不是继续用短样本阈值调参。\n"
        f"- qlib recorder：{summary['recorder_id']}\n"
    )
    (spec.output_dir / "conclusion.md").write_text(text, encoding="utf-8")


def _date_slice(frame: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    dates = pd.to_datetime(frame["date"])
    return frame.loc[dates.between(pd.Timestamp(start), pd.Timestamp(end))].sort_values("datetime").reset_index(drop=True)


def _read_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return _read_json(path)
