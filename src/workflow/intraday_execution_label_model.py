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
from src.workflow.intraday_bar_model import (
    QLIB_FIELDS,
    _build_dataset_row_audit,
    _dedupe_preserve_order,
    _models,
    _recorder_files,
    _remove_transient_code_snapshots,
    _safe_spearman,
    _time_of_day_zscore,
    _wide_feature_frame,
)


EXPERIMENT = "agentic_alpha_intraday_execution_label_v1"
MODEL_NAMES = ["lgb", "xgb", "lgb_xgb_consensus"]
RETURN_COL = "target_return"
MODEL_TARGET_COL = "target_positive"
ENTRY_DAY_WORST_RETURN_COL = "entry_day_worst_return"
OPEN_BUY_DUMP_TARGET_COL = "open_buy_dump_target"
CRASH_EVENT_TARGET_COL = "crash_event_target"
ADVERSE_EVENT_RISK_COL = "adverse_event_risk_target"
LABEL_POLICY = "triple_barrier_1h_2h_execution_improvement"
ADVERSE_EVENT_POLICY = "same_session_entry_day_adverse_event_risk"
EXECUTION_TIMING = "bar_t_close_signal_next_15m_open_then_horizon_or_barrier_exit"
OPEN_BUY_DUMP_THRESHOLD = -0.03
CRASH_EVENT_THRESHOLD = -0.08
REQUESTED_EXTERNAL_INTRADAY_SYMBOLS = ["QQQ", "SMH", "NVDA", "AVGO", "AMD", "TSM", "ASML"]
REQUESTED_EXTERNAL_CONTEXT_SYMBOLS = ["QQQ", "SMH", "SOXX", "NVDA", "AVGO", "AMD", "TSM", "ASML"]
EXTERNAL_SEMIS_BREADTH_SYMBOLS = ["NVDA", "AVGO", "AMD", "TSM", "ASML"]
QLIB_DOCS = {
    "data": "https://qlib.readthedocs.io/en/latest/component/data.html",
    "model": "https://qlib.readthedocs.io/en/latest/component/model.html",
    "workflow": "https://qlib.readthedocs.io/en/latest/component/workflow.html",
    "highfreq": "https://qlib.readthedocs.io/en/latest/component/highfreq.html",
}
QLIB_NATIVE_API = [
    "qlib.init",
    "qlib.data.D.features",
    "DatasetH",
    "DataHandlerLP",
    "QlibDataLoader",
    "qlib.workflow.R.start",
    "qlib.workflow.R.log_params",
    "qlib.workflow.R.log_metrics",
    "qlib.workflow.Recorder.log_artifact",
    "LGBMClassifier",
    "XGBClassifier",
]


@dataclass(frozen=True)
class IntradayExecutionLabelModelSpec:
    experiment: str
    output_dir: Path
    recorder_uri: Path
    provider_uri: Path
    external_provider_uri: Path | None = None
    external_provider_card: str | None = None
    market: str = "all"
    region: str = "us"
    instruments: tuple[str, str] = ("SOXL", "SOXX")
    train: tuple[str, str] = ("2019-01-02", "2024-12-31")
    valid: tuple[str, str] = ("2025-01-02", "2025-12-31")
    test: tuple[str, str] = ("2026-01-02", "2026-07-02")
    as_of: str = "2026-07-02"
    cost_bps: float = 20.0
    cost_stress_bps: tuple[float, ...] = (10.0, 20.0, 50.0)
    profit_take: float = 0.018
    stop_loss: float = 0.014
    horizon_bars: int = 8
    purge_bars: int = 8
    require_external_context: bool = False
    feature_set_name: str = "execution_context_v1"
    required_walk_forward_start: str = "2019-01-02"


def run_intraday_execution_label_model_card(spec: IntradayExecutionLabelModelSpec) -> dict[str, Any]:
    from qlib.workflow import R

    spec.output_dir.mkdir(parents=True, exist_ok=True)
    spec.recorder_uri.mkdir(parents=True, exist_ok=True)
    workflow_yaml = _write_workflow_yaml(spec)

    config = QlibProviderConfig(provider_uri=spec.provider_uri, region=spec.region, market=spec.market, freq="15min")
    provider_meta = init_qlib_provider(config)
    qlib_segments = {
        "train": (f"{spec.train[0]} 09:45:00", f"{spec.train[1]} 16:00:00"),
        "valid": (f"{spec.valid[0]} 09:45:00", f"{spec.valid[1]} 16:00:00"),
        "test": (f"{spec.test[0]} 09:45:00", f"{spec.test[1]} 16:00:00"),
    }
    dataset_audit = _build_dataset_row_audit(config, segments=qlib_segments, instruments=list(spec.instruments))
    raw = load_qlib_feature_frame(
        config,
        instruments=list(spec.instruments),
        fields=QLIB_FIELDS,
        start=f"{spec.train[0]} 09:45:00",
        end=f"{spec.test[1]} 16:00:00",
    )
    external_raw = None
    external_provider_meta: dict[str, Any] | None = None
    external_context_data_status = "not_requested"
    if spec.external_provider_uri is not None:
        external_config = QlibProviderConfig(
            provider_uri=spec.external_provider_uri,
            region=spec.region,
            market=spec.market,
            freq="15min",
        )
        external_provider_meta = init_qlib_provider(external_config)
        external_start = max(spec.train[0], external_provider_meta["calendar_start"][:10])
        external_raw = load_qlib_feature_frame(
            external_config,
            instruments=REQUESTED_EXTERNAL_CONTEXT_SYMBOLS,
            fields=["$open", "$high", "$low", "$close", "$volume", "$vwap", "$bar_index", "$bar_count"],
            start=f"{external_start} 09:45:00",
            end=f"{spec.test[1]} 16:00:00",
        )
        external_context_data_status = (
            "ok"
            if external_provider_meta["calendar_start"][:10] <= spec.required_walk_forward_start
            else "ok_partial_insufficient_for_full_walk_forward"
        )
    frame, feature_cols, label_audit = build_execution_label_frame(
        raw,
        external_raw=external_raw,
        require_external_context=spec.require_external_context,
        profit_take=spec.profit_take,
        stop_loss=spec.stop_loss,
        horizon_bars=spec.horizon_bars,
        cost_bps=spec.cost_bps,
    )
    segments = _split_segments(frame, spec=spec)
    model_rows, predictions, feature_importance = _fit_predict_once(
        segments,
        feature_cols=feature_cols,
        cost_bps=spec.cost_bps,
    )
    model_selection = _augment_selection(pd.DataFrame(model_rows))
    selected = model_selection.iloc[0].to_dict()
    selected_model = _selected_model_payload(selected)
    cost_sensitivity = pd.DataFrame(
        _cost_sensitivity(predictions, model_selection=model_selection, cost_stress_bps=spec.cost_stress_bps)
    )
    walk_forward = pd.DataFrame(
        _walk_forward_metrics(
            frame,
            feature_cols=feature_cols,
            cost_bps=spec.cost_bps,
            cost_stress_bps=spec.cost_stress_bps,
            spec=spec,
        )
    )
    label_diagnostics = _label_diagnostics(frame, spec=spec)
    feature_importance_frame = _feature_importance_frame(feature_importance, feature_cols=feature_cols)
    analysis_summary = _analysis_summary(
        spec,
        provider_meta=provider_meta,
        external_provider_meta=external_provider_meta,
        external_context_data_status=external_context_data_status,
        dataset_audit=dataset_audit,
        frame=frame,
        feature_cols=feature_cols,
        label_audit=label_audit,
        model_selection=model_selection,
        selected_model=selected_model,
        walk_forward=walk_forward,
    )
    sidecars = _write_sidecars(
        spec,
        model_selection=model_selection,
        selected_model=selected_model,
        cost_sensitivity=cost_sensitivity,
        walk_forward=walk_forward,
        feature_importance=feature_importance_frame,
        label_diagnostics=label_diagnostics,
        analysis_summary=analysis_summary,
    )

    selected_prob_col = f"prob_{selected_model['model']}"
    selected_predictions = predictions.loc[
        predictions["segment"].eq("test"),
        ["datetime", "target_return", "target_positive", "execution_improvement_return", selected_prob_col],
    ].rename(columns={selected_prob_col: "score"})
    selected_labels = selected_predictions[["datetime", "target_return", "target_positive"]].copy()
    upload_dir = Path(tempfile.mkdtemp(prefix=f"{spec.experiment}_recorder_"))
    try:
        selected_predictions.to_pickle(upload_dir / "pred.pkl")
        selected_labels.to_pickle(upload_dir / "label.pkl")
        predictions.to_csv(upload_dir / "all_model_predictions.csv", index=False)
        R.set_uri(spec.recorder_uri.resolve().as_posix())
        with R.start(experiment_name=spec.experiment):
            recorder = R.get_recorder()
            R.log_params(
                experiment=spec.experiment,
                market=spec.market,
                region=spec.region,
                train_start=spec.train[0],
                train_end=spec.train[1],
                valid_start=spec.valid[0],
                valid_end=spec.valid[1],
                test_start=spec.test[0],
                test_end=spec.test[1],
                label_policy=LABEL_POLICY,
                horizon_bars=spec.horizon_bars,
                profit_take=spec.profit_take,
                stop_loss=spec.stop_loss,
                cost_bps=spec.cost_bps,
                selected_model=selected_model["model"],
                selected_threshold=selected_model["threshold"],
                selection_uses_test_segment=False,
                external_context=bool(spec.external_provider_uri is not None),
            )
            R.log_metrics(
                selected_valid_return_pct=selected_model["valid_net_return_pct"],
                selected_test_return_pct=selected_model["test_net_return_pct"],
                selected_test_max_drawdown_pct=selected_model["test_max_drawdown_pct"],
                selected_test_ic=selected_model["test_ic"],
            )
            recorder.log_artifact((upload_dir / "pred.pkl").as_posix())
            recorder.log_artifact((upload_dir / "label.pkl").as_posix())
            recorder.log_artifact((upload_dir / "all_model_predictions.csv").as_posix(), artifact_path="model_selection")
            for rel_path in sidecars.values():
                recorder.log_artifact((spec.output_dir / rel_path).as_posix(), artifact_path=spec.experiment)
            recorder_id = recorder.id
            recorder_local_dir = Path(recorder.get_local_dir())
    finally:
        shutil.rmtree(upload_dir, ignore_errors=True)

    _remove_transient_code_snapshots(recorder_local_dir)
    recorder_artifacts = _recorder_files(recorder_local_dir)
    promotion_decision = _promotion_decision(
        spec=spec,
        selected_model=selected_model,
        walk_forward=walk_forward,
        external_provider_meta=external_provider_meta,
    )
    summary = {
        "experiment": spec.experiment,
        "card_type": "intraday_execution_label_model_search",
        "workflow_config": workflow_yaml.name,
        "promotion_decision": promotion_decision,
        "as_of": spec.as_of,
        "market": spec.market,
        "region": spec.region,
        "provider_uris": _provider_uris(spec),
        "provider_audit": {
            "15min": {
                "provider_uri": provider_meta["provider_uri"],
                "calendar_start": provider_meta["calendar_start"],
                "calendar_end": provider_meta["calendar_end"],
                "qlib_feature_rows": int(len(raw)),
                "model_rows": int(len(frame)),
                "feature_count": int(len(feature_cols)),
                "manifest": (spec.provider_uri / "qlib_store_manifest.json").as_posix(),
            }
        }
        | _external_provider_audit_payload(spec, external_provider_meta=external_provider_meta, external_raw=external_raw),
        "segments": {"train": list(spec.train), "valid": list(spec.valid), "test": list(spec.test)},
        "label_policy": {
            "name": LABEL_POLICY,
            "entry_timing": "bar_t_close_signal_next_15m_open",
            "exit_timing": "first_profit_or_stop_hit_else_horizon_close",
            "horizon_bars": int(spec.horizon_bars),
            "horizon_minutes": int(spec.horizon_bars * 15),
            "profit_take": float(spec.profit_take),
            "stop_loss": float(spec.stop_loss),
            "return_col": RETURN_COL,
            "model_target_col": MODEL_TARGET_COL,
            "uses_same_bar_future": False,
            "same_session_only": True,
        },
        "execution_timing": EXECUTION_TIMING,
        "feature_set": {
            "name": spec.feature_set_name,
            "feature_count": int(len(feature_cols)),
            "lookback_only": True,
            "new_features": _execution_context_feature_names(feature_cols),
            "external_features": _external_context_feature_names(feature_cols),
        },
        "external_intraday_context": _external_context_summary(
            spec,
            external_provider_meta=external_provider_meta,
            external_context_data_status=external_context_data_status,
            frame=frame,
            feature_cols=feature_cols,
        ),
        "cost_bps": float(spec.cost_bps),
        "cost_stress_bps": [float(value) for value in sorted(set(spec.cost_stress_bps))],
        "selected_model": selected_model,
        "model_selection": {
            "source": "qlib_dataset_intraday_execution_label",
            "models": MODEL_NAMES,
            "selection_segment": "valid",
            "report_segment": "test",
            "selection_uses_test_segment": False,
            "candidate_count": int(len(model_selection)),
            "selected_model": selected_model["model"],
            "selection_metric": selected_model["selection_metric"],
        },
        "walk_forward": {
            "policy": "train_to_T_minus_2_valid_T_minus_1_test_T",
            "test_years": _walk_forward_years(walk_forward),
            "required_policy_satisfied": _required_walk_forward_satisfied(
                spec,
                external_provider_meta=external_provider_meta,
                walk_forward=walk_forward,
            ),
        },
        "recorder_id": recorder_id,
        "recorder_uri": spec.recorder_uri.resolve().as_posix(),
        "recorder_local_dir": recorder_local_dir.as_posix(),
        "recorder_artifacts": recorder_artifacts,
        "sidecars": sidecars,
        "artifact_policy": _artifact_policy(spec.experiment, sidecars),
        "qlib_docs": QLIB_DOCS,
        "qlib_native_api": QLIB_NATIVE_API,
    }
    (spec.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_conclusion(spec, summary=summary, selected_model=selected_model, walk_forward=walk_forward)
    return summary


def build_execution_label_frame(
    raw: pd.DataFrame,
    *,
    external_raw: pd.DataFrame | None = None,
    require_external_context: bool = False,
    profit_take: float = 0.018,
    stop_loss: float = 0.014,
    horizon_bars: int = 8,
    cost_bps: float = 20.0,
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    frame = _wide_feature_frame(raw).sort_values("datetime").reset_index(drop=True)
    frame["date"] = pd.to_datetime(frame["date"].astype(str)).dt.date
    for prefix in ["soxl", "soxx"]:
        frame[f"{prefix}_ret_bar"] = frame[f"{prefix}_close"] / frame[f"{prefix}_open"] - 1.0
        frame[f"{prefix}_ret_cc1"] = frame.groupby("date")[f"{prefix}_close"].pct_change()
        frame[f"{prefix}_range"] = frame[f"{prefix}_high"] / frame[f"{prefix}_low"] - 1.0
        frame[f"{prefix}_close_loc"] = (
            (frame[f"{prefix}_close"] - frame[f"{prefix}_low"])
            / (frame[f"{prefix}_high"] - frame[f"{prefix}_low"] + 1e-12)
        )
        typical = (frame[f"{prefix}_high"] + frame[f"{prefix}_low"] + frame[f"{prefix}_close"]) / 3.0
        pv = typical * frame[f"{prefix}_volume"]
        frame[f"{prefix}_cum_vwap"] = pv.groupby(frame["date"]).cumsum() / (
            frame.groupby("date")[f"{prefix}_volume"].cumsum() + 1e-12
        )
        frame[f"{prefix}_vwap_dist"] = frame[f"{prefix}_close"] / frame[f"{prefix}_cum_vwap"] - 1.0
        for window in [2, 3, 4, 8]:
            frame[f"{prefix}_mom_{window}"] = frame.groupby("date")[f"{prefix}_close"].pct_change(window)
            frame[f"{prefix}_rv_{window}"] = (
                frame.groupby("date")[f"{prefix}_ret_cc1"]
                .rolling(window, min_periods=max(2, window // 2))
                .std()
                .reset_index(level=0, drop=True)
            )
    frame["tod_frac"] = frame["soxl_bar_index"] / (frame["soxl_bar_count"] - 1.0).replace(0.0, np.nan)
    frame["tod_sin"] = np.sin(2.0 * np.pi * frame["tod_frac"])
    frame["tod_cos"] = np.cos(2.0 * np.pi * frame["tod_frac"])
    frame["soxl_minus_3x_soxx_ret"] = frame["soxl_ret_cc1"] - 3.0 * frame["soxx_ret_cc1"]
    frame["soxl_minus_3x_soxx_mom3"] = frame["soxl_mom_3"] - 3.0 * frame["soxx_mom_3"]
    frame["range_ratio_soxl_soxx"] = frame["soxl_range"] / (frame["soxx_range"] + 1e-12)
    frame, context_features = _add_execution_context_features(frame)
    external_context_features: list[str] = []
    external_context_audit: dict[str, Any] = {"external_context_requested": False}
    if external_raw is not None:
        external_context, external_context_features, external_context_audit = build_external_context_feature_frame(
            external_raw
        )
        frame, external_context_features, merge_audit = merge_external_context_features(frame, external_context)
        external_context_audit = external_context_audit | merge_audit
        if require_external_context:
            frame = frame.loc[frame["external_context_available"]].copy()
    elif require_external_context:
        raise ValueError("external context is required but external_raw is None")
    frame, label_audit = _add_execution_labels(
        frame,
        profit_take=profit_take,
        stop_loss=stop_loss,
        horizon_bars=horizon_bars,
        cost_bps=cost_bps,
    )
    frame, adverse_audit = add_entry_day_adverse_event_labels(
        frame,
        open_buy_dump_threshold=OPEN_BUY_DUMP_THRESHOLD,
        crash_threshold=CRASH_EVENT_THRESHOLD,
    )
    feature_cols = [
        col
        for col in frame.columns
        if any(
            col.startswith(prefix)
            for prefix in [
                "soxl_ret",
                "soxl_range",
                "soxl_close_loc",
                "soxl_vwap_dist",
                "soxl_mom",
                "soxl_rv",
                "soxx_ret",
                "soxx_range",
                "soxx_close_loc",
                "soxx_vwap_dist",
                "soxx_mom",
                "soxx_rv",
            ]
        )
    ]
    feature_cols.extend(
        [
            "tod_frac",
            "tod_sin",
            "tod_cos",
            "soxl_minus_3x_soxx_ret",
            "soxl_minus_3x_soxx_mom3",
            "range_ratio_soxl_soxx",
            *context_features,
            *external_context_features,
        ]
    )
    feature_cols = _dedupe_preserve_order(feature_cols)
    frame[feature_cols] = frame[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    frame = frame.dropna(subset=["entry_datetime", RETURN_COL, MODEL_TARGET_COL]).copy()
    frame["date"] = pd.to_datetime(frame["date"].astype(str)).dt.date.astype(str)
    label_audit["external_context"] = external_context_audit
    label_audit["adverse_event_policy"] = adverse_audit
    return frame.reset_index(drop=True), feature_cols, label_audit


def build_external_context_feature_frame(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    external = _wide_feature_frame(raw).sort_values("datetime").reset_index(drop=True)
    external["date"] = pd.to_datetime(external["date"].astype(str)).dt.date
    available_symbols = sorted(
        symbol.upper()
        for symbol in {column.split("_", 1)[0] for column in external.columns if column.endswith("_close")}
    )
    external = external.rename(
        columns={column: column.replace("soxx_", "external_soxx_", 1) for column in external.columns if column.startswith("soxx_")}
    )
    feature_cols: list[str] = []
    for symbol in REQUESTED_EXTERNAL_CONTEXT_SYMBOLS:
        prefix = _external_context_column_prefix(symbol)
        if f"{prefix}_close" not in external:
            continue
        external[f"{prefix}_ret_bar"] = external[f"{prefix}_close"] / external[f"{prefix}_open"] - 1.0
        external[f"{prefix}_ret_cc1"] = external.groupby("date")[f"{prefix}_close"].pct_change()
        external[f"{prefix}_mom_3"] = external.groupby("date")[f"{prefix}_close"].pct_change(3)
        if f"{prefix}_vwap" in external:
            external[f"{prefix}_vwap_dist"] = external[f"{prefix}_close"] / (external[f"{prefix}_vwap"] + 1e-12) - 1.0
        else:
            external[f"{prefix}_vwap_dist"] = 0.0
        feature_cols.extend(
            [
                f"{prefix}_ret_bar",
                f"{prefix}_ret_cc1",
                f"{prefix}_mom_3",
                f"{prefix}_vwap_dist",
            ]
        )
    for symbol in ["QQQ", "SMH", "SOXX"]:
        prefix = _external_context_column_prefix(symbol)
        if f"{prefix}_mom_3" in external:
            external[f"{prefix}_lead_lag_vs_soxl_3"] = external[f"{prefix}_mom_3"]
            feature_cols.append(f"{prefix}_lead_lag_vs_soxl_3")
    leader_mom_cols = [f"{symbol.lower()}_mom_3" for symbol in EXTERNAL_SEMIS_BREADTH_SYMBOLS if f"{symbol.lower()}_mom_3" in external]
    leader_ret_cols = [f"{symbol.lower()}_ret_cc1" for symbol in EXTERNAL_SEMIS_BREADTH_SYMBOLS if f"{symbol.lower()}_ret_cc1" in external]
    leader_vwap_cols = [
        f"{symbol.lower()}_vwap_dist" for symbol in EXTERNAL_SEMIS_BREADTH_SYMBOLS if f"{symbol.lower()}_vwap_dist" in external
    ]
    if leader_mom_cols:
        external["semis_breadth_mom3_pos"] = (external[leader_mom_cols] > 0.0).mean(axis=1)
        external["semis_equal_weight_mom3"] = external[leader_mom_cols].mean(axis=1)
        feature_cols.extend(["semis_breadth_mom3_pos", "semis_equal_weight_mom3"])
    if leader_ret_cols:
        external["semis_breadth_ret_pos_1"] = (external[leader_ret_cols] > 0.0).mean(axis=1)
        feature_cols.append("semis_breadth_ret_pos_1")
    if leader_vwap_cols:
        external["semis_vwap_breadth"] = (external[leader_vwap_cols] > 0.0).mean(axis=1)
        feature_cols.append("semis_vwap_breadth")
    required_close_cols = [
        f"{_external_context_column_prefix(symbol)}_close"
        for symbol in REQUESTED_EXTERNAL_CONTEXT_SYMBOLS
        if f"{_external_context_column_prefix(symbol)}_close" in external
    ]
    external["external_context_available"] = external[required_close_cols].notna().all(axis=1) if required_close_cols else False
    feature_cols = _dedupe_preserve_order([column for column in feature_cols if column in external.columns])
    external[feature_cols] = external[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return external[["datetime", "external_context_available", *feature_cols]].copy(), feature_cols, {
        "external_context_requested": True,
        "available_symbols": available_symbols,
        "missing_symbols": sorted(set(REQUESTED_EXTERNAL_CONTEXT_SYMBOLS) - set(available_symbols)),
        "external_context_rows": int(len(external)),
        "coverage_start": _format_datetime(external["datetime"].min()) if not external.empty else "",
        "coverage_end": _format_datetime(external["datetime"].max()) if not external.empty else "",
    }


def merge_external_context_features(
    frame: pd.DataFrame,
    external_context: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    feature_cols = [column for column in external_context.columns if column not in {"datetime", "external_context_available"}]
    merged = frame.merge(external_context, on="datetime", how="left")
    merged["external_context_available"] = merged["external_context_available"].fillna(False).astype(bool)
    for column in feature_cols:
        merged[column] = pd.to_numeric(merged[column], errors="coerce").fillna(0.0)
    if {"qqq_mom_3", "smh_mom_3", "external_soxx_mom_3", "soxl_mom_3"}.issubset(merged.columns):
        merged["qqq_smh_soxx_lead_lag_3"] = (
            merged[["qqq_mom_3", "smh_mom_3", "external_soxx_mom_3"]].mean(axis=1)
            - merged["soxl_mom_3"] / 3.0
        )
        feature_cols.append("qqq_smh_soxx_lead_lag_3")
    feature_cols = _dedupe_preserve_order([column for column in feature_cols if column in merged.columns])
    return merged, feature_cols, {
        "external_context_rows": int(len(external_context)),
        "merged_rows": int(len(merged)),
        "available_rows": int(merged["external_context_available"].sum()),
        "lookback_only": True,
    }


def _add_execution_context_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    frame = frame.copy()
    feature_cols: list[str] = []
    for prefix in ["soxl", "soxx"]:
        day_open = frame.groupby("date")[f"{prefix}_open"].transform("first")
        day_high = frame.groupby("date")[f"{prefix}_high"].cummax()
        day_low = frame.groupby("date")[f"{prefix}_low"].cummin()
        prior_close_by_day = frame.groupby("date")[f"{prefix}_close"].last().shift(1)
        frame[f"{prefix}_prior_day_close"] = frame["date"].map(prior_close_by_day)
        frame[f"{prefix}_open_gap_from_prior_close"] = day_open / (frame[f"{prefix}_prior_day_close"] + 1e-12) - 1.0
        frame[f"{prefix}_cum_ret_from_open"] = frame[f"{prefix}_close"] / day_open - 1.0
        frame[f"{prefix}_range_so_far"] = day_high / (day_low + 1e-12) - 1.0
        frame[f"{prefix}_dist_day_high"] = frame[f"{prefix}_close"] / (day_high + 1e-12) - 1.0
        frame[f"{prefix}_dist_day_low"] = frame[f"{prefix}_close"] / (day_low + 1e-12) - 1.0
        frame[f"{prefix}_realized_vol_so_far"] = (
            frame.groupby("date")[f"{prefix}_ret_cc1"]
            .expanding(min_periods=2)
            .std()
            .reset_index(level=0, drop=True)
        )
        tod_rv_baseline = frame.groupby(f"{prefix}_bar_index")[f"{prefix}_realized_vol_so_far"].transform(
            lambda item: item.shift(1).rolling(60, min_periods=20).mean()
        )
        frame[f"{prefix}_realized_vol_expansion"] = (
            frame[f"{prefix}_realized_vol_so_far"] / (tod_rv_baseline + 1e-12) - 1.0
        )
        frame[f"{prefix}_volume_tod_z60"] = _time_of_day_zscore(frame, f"{prefix}_volume", bar_col=f"{prefix}_bar_index")
        frame[f"{prefix}_ret_tod_z60"] = _time_of_day_zscore(frame, f"{prefix}_ret_bar", bar_col=f"{prefix}_bar_index")
        frame[f"{prefix}_first30_ret"] = frame[f"{prefix}_cum_ret_from_open"].where(frame[f"{prefix}_bar_index"] <= 1)
        frame[f"{prefix}_first30_ret"] = frame.groupby("date")[f"{prefix}_first30_ret"].ffill().fillna(0.0)
        frame[f"{prefix}_first60_trend_quality"] = (
            frame[f"{prefix}_cum_ret_from_open"].abs() / (frame[f"{prefix}_range_so_far"] + 1e-12)
        ).clip(0.0, 5.0)
        frame[f"{prefix}_volume_price_absorption"] = frame[f"{prefix}_volume_tod_z60"].fillna(0.0) - (
            frame[f"{prefix}_ret_tod_z60"].abs().fillna(0.0) * 0.5
        )
        feature_cols.extend(
            [
                f"{prefix}_open_gap_from_prior_close",
                f"{prefix}_cum_ret_from_open",
                f"{prefix}_range_so_far",
                f"{prefix}_dist_day_high",
                f"{prefix}_dist_day_low",
                f"{prefix}_realized_vol_expansion",
                f"{prefix}_volume_tod_z60",
                f"{prefix}_ret_tod_z60",
                f"{prefix}_first30_ret",
                f"{prefix}_first60_trend_quality",
                f"{prefix}_volume_price_absorption",
            ]
        )
    frame["soxx_lead_lag_confirm_3"] = frame["soxx_mom_3"] - frame["soxl_mom_3"] / 3.0
    frame["leveraged_replication_absorption"] = frame["soxl_volume_price_absorption"] - frame[
        "soxx_volume_price_absorption"
    ]
    frame["open_gap_reclaim_quality"] = np.where(
        frame["soxl_open_gap_from_prior_close"].abs() > 1e-12,
        frame["soxl_cum_ret_from_open"] / frame["soxl_open_gap_from_prior_close"].abs(),
        0.0,
    )
    feature_cols.extend(["soxx_lead_lag_confirm_3", "leveraged_replication_absorption", "open_gap_reclaim_quality"])
    return frame, feature_cols


def _add_execution_labels(
    frame: pd.DataFrame,
    *,
    profit_take: float,
    stop_loss: float,
    horizon_bars: int,
    cost_bps: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = frame.copy()
    frame["entry_datetime"] = frame.groupby("date")["datetime"].shift(-1)
    frame["entry_open"] = frame.groupby("date")["soxl_open"].shift(-1)
    outcomes: list[str | None] = []
    returns: list[float] = []
    improvements: list[float] = []
    exit_times: list[pd.Timestamp | pd.NaT] = []
    exit_prices: list[float] = []
    for _, row in frame.iterrows():
        date = row["date"]
        entry_dt = row["entry_datetime"]
        entry_open = row["entry_open"]
        if pd.isna(entry_dt) or pd.isna(entry_open):
            outcomes.append(None)
            returns.append(np.nan)
            improvements.append(np.nan)
            exit_times.append(pd.NaT)
            exit_prices.append(np.nan)
            continue
        future = frame.loc[(frame["date"] == date) & (frame["datetime"] >= entry_dt)].head(horizon_bars)
        if future.empty:
            outcomes.append(None)
            returns.append(np.nan)
            improvements.append(np.nan)
            exit_times.append(pd.NaT)
            exit_prices.append(np.nan)
            continue
        outcome = "timeout"
        exit_row = future.iloc[-1]
        for _, future_row in future.iterrows():
            hit_profit = float(future_row["soxl_high"]) / float(entry_open) - 1.0 >= profit_take
            hit_stop = float(future_row["soxl_low"]) / float(entry_open) - 1.0 <= -stop_loss
            if hit_profit and hit_stop:
                outcome = "stop" if float(future_row["soxl_close"]) < float(entry_open) else "profit"
                exit_row = future_row
                break
            if hit_profit:
                outcome = "profit"
                exit_row = future_row
                break
            if hit_stop:
                outcome = "stop"
                exit_row = future_row
                break
        exit_price = float(exit_row["soxl_close"])
        target_return = exit_price / float(entry_open) - 1.0
        open_return = exit_price / float(row["soxl_open"]) - 1.0
        improvement = target_return - open_return
        outcomes.append(outcome)
        returns.append(target_return)
        improvements.append(improvement)
        exit_times.append(pd.Timestamp(exit_row["datetime"]))
        exit_prices.append(exit_price)
    frame["triple_barrier_outcome"] = outcomes
    frame["exit_datetime"] = exit_times
    frame["exit_price"] = exit_prices
    frame[RETURN_COL] = returns
    frame["execution_improvement_return"] = improvements
    cost = cost_bps / 10000.0
    frame[MODEL_TARGET_COL] = (
        (frame["triple_barrier_outcome"].eq("profit"))
        | ((frame[RETURN_COL] > cost) & (frame["execution_improvement_return"] > -cost))
    ).astype(float)
    frame.loc[frame[RETURN_COL].isna(), MODEL_TARGET_COL] = np.nan
    return frame, {
        "label_policy": LABEL_POLICY,
        "entry_timing": "next_15m_open",
        "horizon_bars": int(horizon_bars),
        "profit_take": float(profit_take),
        "stop_loss": float(stop_loss),
        "cost_bps": float(cost_bps),
        "same_session_last_rows_dropped": int(frame[RETURN_COL].isna().sum()),
    }


def add_entry_day_adverse_event_labels(
    frame: pd.DataFrame,
    *,
    open_buy_dump_threshold: float = OPEN_BUY_DUMP_THRESHOLD,
    crash_threshold: float = CRASH_EVENT_THRESHOLD,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = frame.copy()
    worst_returns: list[float] = []
    open_buy_dump_targets: list[float] = []
    crash_targets: list[float] = []
    for _, row in frame.iterrows():
        date = row.get("date")
        entry_dt = row.get("entry_datetime")
        entry_open = row.get("entry_open")
        if pd.isna(entry_dt) or pd.isna(entry_open) or float(entry_open) <= 0.0:
            worst_returns.append(np.nan)
            open_buy_dump_targets.append(np.nan)
            crash_targets.append(np.nan)
            continue
        future = frame.loc[(frame["date"] == date) & (frame["datetime"] >= entry_dt)]
        if future.empty or future["soxl_low"].isna().all():
            worst_returns.append(np.nan)
            open_buy_dump_targets.append(np.nan)
            crash_targets.append(np.nan)
            continue
        worst_return = float(pd.to_numeric(future["soxl_low"], errors="coerce").min() / float(entry_open) - 1.0)
        worst_returns.append(worst_return)
        open_buy_dump_targets.append(float(worst_return <= float(open_buy_dump_threshold)))
        crash_targets.append(float(worst_return <= float(crash_threshold)))
    frame[ENTRY_DAY_WORST_RETURN_COL] = worst_returns
    frame[OPEN_BUY_DUMP_TARGET_COL] = open_buy_dump_targets
    frame[CRASH_EVENT_TARGET_COL] = crash_targets
    frame[ADVERSE_EVENT_RISK_COL] = frame[OPEN_BUY_DUMP_TARGET_COL]
    return frame, {
        "label_policy": ADVERSE_EVENT_POLICY,
        "entry_timing": "next_15m_open",
        "risk_horizon": "same_session_entry_to_close",
        "worst_return_col": ENTRY_DAY_WORST_RETURN_COL,
        "open_buy_dump_target_col": OPEN_BUY_DUMP_TARGET_COL,
        "crash_event_target_col": CRASH_EVENT_TARGET_COL,
        "model_target_col": ADVERSE_EVENT_RISK_COL,
        "open_buy_dump_threshold": float(open_buy_dump_threshold),
        "crash_threshold": float(crash_threshold),
        "uses_same_bar_future": False,
        "same_session_only": True,
        "missing_target_rows": int(frame[ADVERSE_EVENT_RISK_COL].isna().sum()),
    }


def _split_segments(frame: pd.DataFrame, *, spec: IntradayExecutionLabelModelSpec) -> dict[str, pd.DataFrame]:
    return {
        "train": _date_slice(frame, *spec.train).reset_index(drop=True),
        "valid": _date_slice(frame, *spec.valid).reset_index(drop=True),
        "test": _date_slice(frame, *spec.test).reset_index(drop=True),
    }


def _date_slice(frame: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    dates = pd.to_datetime(frame["date"])
    return frame.loc[dates.between(pd.Timestamp(start), pd.Timestamp(end))].sort_values("datetime").copy()


def _fit_predict_once(
    segments: dict[str, pd.DataFrame],
    *,
    feature_cols: list[str],
    cost_bps: float,
) -> tuple[list[dict[str, Any]], pd.DataFrame, list[dict[str, Any]]]:
    train = segments["train"].copy()
    valid = segments["valid"].copy()
    test = segments["test"].copy()
    feature_importance: list[dict[str, Any]] = []
    prediction_frames = []
    model_metrics: list[dict[str, Any]] = []
    for name, model in _models().items():
        model.fit(train[feature_cols], train[MODEL_TARGET_COL].astype(int))
        model_prediction_pieces = []
        for segment_name, segment in [("train", train), ("valid", valid), ("test", test)]:
            segment[f"prob_{name}"] = model.predict_proba(segment[feature_cols])[:, 1]
            model_prediction_pieces.append(
                segment[
                    [
                        "datetime",
                        "date",
                        RETURN_COL,
                        MODEL_TARGET_COL,
                        "execution_improvement_return",
                        "triple_barrier_outcome",
                        f"prob_{name}",
                    ]
                ].assign(segment=segment_name)
            )
        prediction_frames.append(pd.concat(model_prediction_pieces, ignore_index=True))
        if hasattr(model, "feature_importances_"):
            top = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False).head(25)
            feature_importance.extend(
                {"model": name, "feature": feature, "importance": float(importance)}
                for feature, importance in top.items()
            )
        threshold, valid_metrics = _choose_threshold(valid, prob_col=f"prob_{name}", cost_bps=cost_bps)
        test_metrics = _trade_metrics(test, prob_col=f"prob_{name}", threshold=threshold, cost_bps=cost_bps)
        model_metrics.append(
            {
                "model": name,
                "threshold": float(threshold),
                "valid_ic": _safe_spearman(valid[f"prob_{name}"], valid[RETURN_COL]),
                "test_ic": _safe_spearman(test[f"prob_{name}"], test[RETURN_COL]),
                **{f"valid_{key}": value for key, value in valid_metrics.items()},
                **{f"test_{key}": value for key, value in test_metrics.items()},
            }
        )
    merged = _merge_prediction_frames(prediction_frames)
    merged["prob_lgb_xgb_consensus"] = merged[["prob_lgb", "prob_xgb"]].mean(axis=1)
    threshold, valid_metrics = _choose_threshold(
        merged.loc[merged["segment"].eq("valid")].copy(),
        prob_col="prob_lgb_xgb_consensus",
        cost_bps=cost_bps,
    )
    test_metrics = _trade_metrics(
        merged.loc[merged["segment"].eq("test")].copy(),
        prob_col="prob_lgb_xgb_consensus",
        threshold=threshold,
        cost_bps=cost_bps,
    )
    model_metrics.append(
        {
            "model": "lgb_xgb_consensus",
            "threshold": float(threshold),
            "valid_ic": _safe_spearman(
                merged.loc[merged["segment"].eq("valid"), "prob_lgb_xgb_consensus"],
                merged.loc[merged["segment"].eq("valid"), RETURN_COL],
            ),
            "test_ic": _safe_spearman(
                merged.loc[merged["segment"].eq("test"), "prob_lgb_xgb_consensus"],
                merged.loc[merged["segment"].eq("test"), RETURN_COL],
            ),
            **{f"valid_{key}": value for key, value in valid_metrics.items()},
            **{f"test_{key}": value for key, value in test_metrics.items()},
        }
    )
    return model_metrics, merged, feature_importance


def _feature_importance_frame(feature_importance: list[dict[str, Any]], *, feature_cols: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame(feature_importance)
    recorded = set(frame["feature"]) if not frame.empty and "feature" in frame.columns else set()
    filler_rows = [
        {"model": "feature_set", "feature": feature, "importance": 0.0}
        for feature in _external_context_feature_names(feature_cols)
        if feature not in recorded
    ]
    if filler_rows:
        frame = pd.concat([frame, pd.DataFrame(filler_rows)], ignore_index=True)
    return frame


def _merge_prediction_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    base_cols = [
        "datetime",
        "date",
        "segment",
        RETURN_COL,
        MODEL_TARGET_COL,
        "execution_improvement_return",
        "triple_barrier_outcome",
    ]
    merged: pd.DataFrame | None = None
    for frame in frames:
        prob_cols = [col for col in frame.columns if col.startswith("prob_")]
        current = frame[base_cols + prob_cols].copy()
        if merged is None:
            merged = current
        else:
            merged = merged.merge(current[["datetime", "segment"] + prob_cols], on=["datetime", "segment"], how="outer")
    if merged is None:
        return pd.DataFrame(columns=base_cols)
    return merged.sort_values(["segment", "datetime"]).reset_index(drop=True)


def _choose_threshold(frame: pd.DataFrame, *, prob_col: str, cost_bps: float) -> tuple[float, dict[str, Any]]:
    if frame.empty:
        return 1.0, _trade_metrics(frame, prob_col=prob_col, threshold=1.0, cost_bps=cost_bps)
    candidates = sorted(
        set(
            [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
            + [float(frame[prob_col].quantile(q)) for q in [0.55, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]]
        )
    )
    min_trades = max(60, int(len(frame) * 0.01))
    scored: list[tuple[float, float, dict[str, Any]]] = []
    for threshold in candidates:
        metrics = _trade_metrics(frame, prob_col=prob_col, threshold=threshold, cost_bps=cost_bps)
        if metrics["trades"] >= min_trades:
            score = (
                metrics["net_return_pct"]
                + 4.0 * metrics["daily_sharpe"]
                + 0.5 * metrics["max_drawdown_pct"]
                + 0.08 * metrics["avg_execution_improvement_bps"]
                + 0.03 * metrics["profit_outcome_ratio"] * 100.0
            )
            scored.append((score, threshold, metrics))
    if not scored:
        threshold = float(frame[prob_col].quantile(0.8))
        return threshold, _trade_metrics(frame, prob_col=prob_col, threshold=threshold, cost_bps=cost_bps)
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[0][1], scored[0][2]


def _trade_metrics(frame: pd.DataFrame, *, prob_col: str, threshold: float, cost_bps: float) -> dict[str, Any]:
    if frame.empty or prob_col not in frame.columns:
        return _empty_metrics()
    trades = frame.loc[pd.to_numeric(frame[prob_col], errors="coerce") >= threshold].copy()
    if trades.empty:
        return _empty_metrics(coverage_days=int(pd.to_datetime(frame["date"]).nunique()) if "date" in frame else 0)
    trades["net_ret"] = pd.to_numeric(trades[RETURN_COL], errors="coerce").fillna(0.0) - cost_bps / 10000.0
    equity = (1.0 + trades["net_ret"]).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    daily = trades.groupby("date")["net_ret"].apply(lambda item: (1.0 + item).prod() - 1.0)
    sharpe = 0.0 if daily.std(ddof=0) == 0 else float(daily.mean() / daily.std(ddof=0) * np.sqrt(252.0))
    return {
        "trades": int(len(trades)),
        "trade_days": int(daily.shape[0]),
        "coverage_days": int(pd.to_datetime(frame["date"]).nunique()),
        "net_return_pct": float((equity.iloc[-1] - 1.0) * 100.0),
        "gross_return_pct": float(((1.0 + trades[RETURN_COL]).cumprod().iloc[-1] - 1.0) * 100.0),
        "max_drawdown_pct": float(drawdown.min() * 100.0),
        "daily_sharpe": sharpe,
        "win_rate": float((trades["net_ret"] > 0.0).mean()),
        "avg_trade_bps": float(trades["net_ret"].mean() * 10000.0),
        "worst_trade_bps": float(trades["net_ret"].min() * 10000.0),
        "avg_execution_improvement_bps": float(trades["execution_improvement_return"].mean() * 10000.0),
        "profit_outcome_ratio": float(trades["triple_barrier_outcome"].eq("profit").mean()),
        "stop_outcome_ratio": float(trades["triple_barrier_outcome"].eq("stop").mean()),
    }


def _empty_metrics(*, coverage_days: int = 0) -> dict[str, Any]:
    return {
        "trades": 0,
        "trade_days": 0,
        "coverage_days": int(coverage_days),
        "net_return_pct": 0.0,
        "gross_return_pct": 0.0,
        "max_drawdown_pct": 0.0,
        "daily_sharpe": 0.0,
        "win_rate": np.nan,
        "avg_trade_bps": np.nan,
        "worst_trade_bps": np.nan,
        "avg_execution_improvement_bps": np.nan,
        "profit_outcome_ratio": 0.0,
        "stop_outcome_ratio": 0.0,
    }


def _augment_selection(model_selection: pd.DataFrame) -> pd.DataFrame:
    frame = model_selection.copy()
    frame["selection_score"] = (
        frame["valid_net_return_pct"]
        + 4.0 * frame["valid_daily_sharpe"]
        + 0.5 * frame["valid_max_drawdown_pct"]
        + 0.08 * frame["valid_avg_execution_improvement_bps"].fillna(0.0)
        + 0.03 * frame["valid_profit_outcome_ratio"].fillna(0.0) * 100.0
    )
    frame["selection_eligible"] = (
        (frame["valid_trades"] >= 60)
        & (frame["valid_net_return_pct"] > 0.0)
        & (frame["valid_avg_trade_bps"] > 0.0)
        & (frame["valid_max_drawdown_pct"] > -25.0)
    )
    frame.loc[~frame["selection_eligible"], "selection_score"] -= 1000.0
    return frame.sort_values(["selection_score", "valid_net_return_pct"], ascending=False).reset_index(drop=True)


def _selected_model_payload(selected: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "valid_net_return_pct",
        "valid_max_drawdown_pct",
        "valid_avg_execution_improvement_bps",
        "valid_trades",
        "test_net_return_pct",
        "test_max_drawdown_pct",
        "test_avg_execution_improvement_bps",
        "test_trades",
        "test_ic",
        "selection_score",
    ]
    payload = {
        "model": str(selected["model"]),
        "threshold": float(selected["threshold"]),
        "selection_segment": "valid",
        "report_segment": "test",
        "selection_uses_test_segment": False,
        "selection_metric": (
            "valid_net_return + 4*valid_sharpe + 0.5*valid_drawdown "
            "+ 0.08*avg_execution_improvement_bps + 0.03*profit_outcome_ratio"
        ),
        "selection_eligible": bool(selected["selection_eligible"]),
    }
    for key in keys:
        payload[key] = float(selected.get(key, 0.0))
    return payload


def _cost_sensitivity(
    predictions: pd.DataFrame,
    *,
    model_selection: pd.DataFrame,
    cost_stress_bps: tuple[float, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _, selected in model_selection.iterrows():
        model = str(selected["model"])
        prob_col = f"prob_{model}"
        for cost_bps in sorted(set(float(value) for value in cost_stress_bps)):
            row = {"model": model, "threshold": float(selected["threshold"]), "cost_bps": cost_bps}
            for segment in ["valid", "test"]:
                metrics = _trade_metrics(
                    predictions.loc[predictions["segment"].eq(segment)].copy(),
                    prob_col=prob_col,
                    threshold=float(selected["threshold"]),
                    cost_bps=cost_bps,
                )
                row.update({f"{segment}_{key}": value for key, value in metrics.items()})
            rows.append(row)
    return rows


def _walk_forward_metrics(
    frame: pd.DataFrame,
    *,
    feature_cols: list[str],
    cost_bps: float,
    cost_stress_bps: tuple[float, ...],
    spec: IntradayExecutionLabelModelSpec,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for test_year in [2023, 2024, 2025, 2026]:
        train = _date_slice(frame, "2019-01-02", f"{test_year - 2}-12-31")
        valid = _date_slice(frame, f"{test_year - 1}-01-01", f"{test_year - 1}-12-31")
        test_end = f"{test_year}-12-31" if test_year < 2026 else spec.test[1]
        test = _date_slice(frame, f"{test_year}-01-01", test_end)
        if train.empty or valid.empty or test.empty:
            continue
        model_rows, predictions, _ = _fit_predict_once(
            {"train": train, "valid": valid, "test": test},
            feature_cols=feature_cols,
            cost_bps=cost_bps,
        )
        selection = _augment_selection(pd.DataFrame(model_rows))
        selected = selection.iloc[0].to_dict()
        model = str(selected["model"])
        prob_col = f"prob_{model}"
        stress_rows = {}
        for stress in sorted(set(float(value) for value in cost_stress_bps)):
            metrics = _trade_metrics(
                predictions.loc[predictions["segment"].eq("test")].copy(),
                prob_col=prob_col,
                threshold=float(selected["threshold"]),
                cost_bps=stress,
            )
            stress_rows[f"test_{int(stress)}bp_net_return_pct"] = metrics["net_return_pct"]
        rows.append(
            {
                "fold": f"wf_{test_year}",
                "test_year": int(test_year),
                "train_start": "2019-01-02",
                "train_end": f"{test_year - 2}-12-31",
                "valid_start": f"{test_year - 1}-01-01",
                "valid_end": f"{test_year - 1}-12-31",
                "test_start": f"{test_year}-01-01",
                "test_end": test_end,
                "selected_model": model,
                "threshold": float(selected["threshold"]),
                "selection_uses_test_segment": False,
                "valid_net_return_pct": float(selected["valid_net_return_pct"]),
                "valid_max_drawdown_pct": float(selected["valid_max_drawdown_pct"]),
                "test_net_return_pct": float(selected["test_net_return_pct"]),
                "test_max_drawdown_pct": float(selected["test_max_drawdown_pct"]),
                "test_avg_execution_improvement_bps": float(selected["test_avg_execution_improvement_bps"]),
                **stress_rows,
            }
        )
    return rows


def _label_diagnostics(frame: pd.DataFrame, *, spec: IntradayExecutionLabelModelSpec) -> pd.DataFrame:
    rows = []
    segmented = frame.assign(segment="train")
    segmented.loc[pd.to_datetime(segmented["date"]).between(pd.Timestamp(spec.valid[0]), pd.Timestamp(spec.valid[1])), "segment"] = "valid"
    segmented.loc[pd.to_datetime(segmented["date"]).between(pd.Timestamp(spec.test[0]), pd.Timestamp(spec.test[1])), "segment"] = "test"
    for (segment, outcome), group in segmented.groupby(["segment", "triple_barrier_outcome"], sort=True):
        rows.append(
            {
                "segment": segment,
                "triple_barrier_outcome": outcome,
                "rows": int(len(group)),
                "positive_ratio": float(group[MODEL_TARGET_COL].mean()),
                "avg_target_return_bps": float(group[RETURN_COL].mean() * 10000.0),
                "avg_execution_improvement_bps": float(group["execution_improvement_return"].mean() * 10000.0),
            }
        )
    return pd.DataFrame(rows)


def _analysis_summary(
    spec: IntradayExecutionLabelModelSpec,
    *,
    provider_meta: dict[str, Any],
    external_provider_meta: dict[str, Any] | None = None,
    external_context_data_status: str = "not_requested",
    dataset_audit: dict[str, Any],
    frame: pd.DataFrame,
    feature_cols: list[str],
    label_audit: dict[str, Any],
    model_selection: pd.DataFrame,
    selected_model: dict[str, Any],
    walk_forward: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "source": "qlib_dataset_intraday_execution_label_model",
        "as_of": spec.as_of,
        "provider_meta": provider_meta,
        "dataset_audit": dataset_audit,
        "label_audit": label_audit,
        "feature_count": int(len(feature_cols)),
        "model_rows": int(len(frame)),
        "selected_model": selected_model,
        "model_selection_rows": int(len(model_selection)),
        "walk_forward_rows": int(len(walk_forward)),
        "selection_uses_test_segment": False,
        "external_intraday_context": _external_context_summary(
            spec,
            external_provider_meta=external_provider_meta,
            external_context_data_status=external_context_data_status,
            frame=frame,
            feature_cols=feature_cols,
        ),
    }


def _write_workflow_yaml(spec: IntradayExecutionLabelModelSpec) -> Path:
    path = spec.output_dir / "workflow.yaml"
    path.write_text(
        "\n".join(
            [
                f"experiment: {spec.experiment}",
                "qlib:",
                f"  recorder_uri: {spec.recorder_uri.resolve().as_posix()}",
                f"  provider_uri: {spec.provider_uri.resolve().as_posix()}",
                *(
                    [f"  external_provider_uri: {spec.external_provider_uri.resolve().as_posix()}"]
                    if spec.external_provider_uri is not None
                    else []
                ),
                f"  market: {spec.market}",
                f"  region: {spec.region}",
                "segments:",
                f"  train: {spec.train[0]} to {spec.train[1]}",
                f"  valid: {spec.valid[0]} to {spec.valid[1]}",
                f"  test: {spec.test[0]} to {spec.test[1]}",
                "label_policy:",
                f"  name: {LABEL_POLICY}",
                f"  horizon_bars: {spec.horizon_bars}",
                f"  profit_take: {spec.profit_take}",
                f"  stop_loss: {spec.stop_loss}",
                "model_selection:",
                "  models: [lgb, xgb, lgb_xgb_consensus]",
                "  threshold_segment: valid",
                "  selection_uses_test_segment: false",
                f"  cost_bps: {spec.cost_bps}",
                f"  cost_stress_bps: {list(spec.cost_stress_bps)}",
                "artifact_policy:",
                "  policy: slim_card_recorder_first",
                "  machine_entrypoint: summary.json",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def _write_sidecars(
    spec: IntradayExecutionLabelModelSpec,
    *,
    model_selection: pd.DataFrame,
    selected_model: dict[str, Any],
    cost_sensitivity: pd.DataFrame,
    walk_forward: pd.DataFrame,
    feature_importance: pd.DataFrame,
    label_diagnostics: pd.DataFrame,
    analysis_summary: dict[str, Any],
) -> dict[str, str]:
    sidecars = {
        "model_selection": "model_selection.csv",
        "selected_model": "selected_model.json",
        "cost_sensitivity": "cost_sensitivity.csv",
        "backtest_walk_forward_metrics": "backtest_walk_forward_metrics.csv",
        "feature_importance": "feature_importance.csv",
        "label_diagnostics": "label_diagnostics.csv",
        "analysis_summary": "analysis_summary.json",
        "failure_modes": "failure_modes.md",
    }
    model_selection.to_csv(spec.output_dir / sidecars["model_selection"], index=False)
    cost_sensitivity.to_csv(spec.output_dir / sidecars["cost_sensitivity"], index=False)
    walk_forward.to_csv(spec.output_dir / sidecars["backtest_walk_forward_metrics"], index=False)
    feature_importance.to_csv(spec.output_dir / sidecars["feature_importance"], index=False)
    label_diagnostics.to_csv(spec.output_dir / sidecars["label_diagnostics"], index=False)
    (spec.output_dir / sidecars["selected_model"]).write_text(
        json.dumps(selected_model, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (spec.output_dir / sidecars["analysis_summary"]).write_text(
        json.dumps(analysis_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    external_context_line = (
        "- 外部上下文已接入 QQQ/SMH/SOXX 与 NVDA/AVGO/AMD/TSM/ASML 的 15min qlib store，但历史从 2025 才开始，不能满足 2019 起步的完整 walk-forward。\n"
        if spec.external_provider_uri is not None
        else "- 当前 15min qlib provider 只有 SOXL/SOXX；QQQ/SMH 和 NVDA/AVGO/AMD/TSM/ASML 分钟级上下文尚未进入 qlib store。\n"
    )
    failure_modes = (
        "# Failure Modes\n\n"
        "- Label 已从下一根 15m 涨跌升级到 triple-barrier + execution-improvement，但仍是 bar 数据回放，不是 bid/ask 或队列成交。\n"
        f"{external_context_line}"
        "- Walk-forward 只允许 valid 选阈值，test 只报告；如果某一年失败，不能用 2026 test 表现反向调参。\n"
        "- 该模型仍然只适合作为日级仓位模型的执行层候选，不应替代日级 open-to-open 趋势仓位模型。\n"
    )
    (spec.output_dir / sidecars["failure_modes"]).write_text(failure_modes, encoding="utf-8")
    return sidecars


def _artifact_policy(experiment: str, sidecars: dict[str, str]) -> dict[str, Any]:
    return {
        "policy": "slim_card_recorder_first",
        "primary_artifact": f"{experiment}.ipynb",
        "machine_entrypoint": "summary.json",
        "workflow_config": "workflow.yaml",
        "kept_sidecars": sorted(sidecars.values()),
        "recorder_only_artifacts": {
            "predictions": "artifacts/pred.pkl",
            "labels": "artifacts/label.pkl",
            "all_model_predictions": "artifacts/model_selection/all_model_predictions.csv",
        },
        "reproducible_from_recorder": True,
    }


def _promotion_decision(
    *,
    spec: IntradayExecutionLabelModelSpec,
    selected_model: dict[str, Any],
    walk_forward: pd.DataFrame,
    external_provider_meta: dict[str, Any] | None = None,
) -> str:
    if not _required_walk_forward_satisfied(spec, external_provider_meta=external_provider_meta, walk_forward=walk_forward):
        return "not_promoted_external_history_insufficient_for_required_walk_forward"
    if float(selected_model.get("test_net_return_pct", 0.0)) <= 0.0:
        return "not_promoted_test_return_failed"
    if walk_forward.empty or (walk_forward["test_net_return_pct"] > 0.0).mean() < 0.75:
        return "not_promoted_walk_forward_not_proven"
    return "not_promoted_intraday_replay_missing"


def _write_conclusion(
    spec: IntradayExecutionLabelModelSpec,
    *,
    summary: dict[str, Any],
    selected_model: dict[str, Any],
    walk_forward: pd.DataFrame,
) -> None:
    wf_positive = float((walk_forward["test_net_return_pct"] > 0.0).mean()) if not walk_forward.empty else 0.0
    external_sentence = (
        "并接入外部 15m 上下文：QQQ/SMH/SOXX lead-lag，NVDA/AVGO/AMD/TSM/ASML 半导体成分股 breadth/equal-weight。"
        if spec.external_provider_uri is not None
        else ""
    )
    risk_sentence = (
        "- 风险：外部 15m 上下文只有 2025-01-02 之后的数据，完整 train-to-T-2 / valid-T-1 / test-T 的 2019 起步 walk-forward 不满足；"
        "真实执行还缺 bid/ask replay、排队成交、延迟和极端波动成交验证。\n"
        if spec.external_provider_uri is not None
        else "- 风险：当前分钟级外部上下文只有 SOXL/SOXX，QQQ/SMH 与 NVDA/AVGO/AMD/TSM/ASML 还没有分钟级 qlib store；"
        "真实执行还缺 bid/ask replay、排队成交、延迟和极端波动成交验证。\n"
    )
    text = (
        "# 中文结论\n\n"
        "- 做了什么：把 15m 日内模型的 label 从下一根 bar 涨跌升级为 triple-barrier + execution-improvement，"
        "并加入 open gap、VWAP 距离、realized vol 扩张、前 30/60 分钟趋势质量、量价吸收/派发、SOXX lead-lag 等同日上下文因子。"
        f"{external_sentence}\n"
        f"- 核心指标：validation 选择 {selected_model['model']}，valid return_pct={selected_model['valid_net_return_pct']:.4f}、"
        f"test return_pct={selected_model['test_net_return_pct']:.4f}、test max_drawdown_pct={selected_model['test_max_drawdown_pct']:.4f}。\n"
        f"- walk-forward：按 train 到 T-2、valid T-1、test T 记录 2023-2026，正收益比例={wf_positive:.2f}。\n"
        f"- 是否晋升：不晋升；decision={summary['promotion_decision']}。它是执行层候选，不替代日级 open-to-open 仓位模型。\n"
        f"{risk_sentence}"
        f"- qlib recorder：{summary['recorder_id']}\n"
    )
    (spec.output_dir / "conclusion.md").write_text(text, encoding="utf-8")


def _execution_context_feature_names(feature_cols: list[str]) -> list[str]:
    prefixes = [
        "soxl_open_gap_from_prior_close",
        "soxl_realized_vol_expansion",
        "soxl_first30_ret",
        "soxl_first60_trend_quality",
        "soxl_volume_price_absorption",
        "soxx_lead_lag_confirm_3",
        "leveraged_replication_absorption",
        "open_gap_reclaim_quality",
    ]
    return [feature for feature in prefixes if feature in feature_cols]


def _provider_uris(spec: IntradayExecutionLabelModelSpec) -> dict[str, str]:
    provider_uris = {"15min": spec.provider_uri.resolve().as_posix()}
    if spec.external_provider_uri is not None:
        provider_uris["external_15min"] = spec.external_provider_uri.resolve().as_posix()
    return provider_uris


def _external_provider_audit_payload(
    spec: IntradayExecutionLabelModelSpec,
    *,
    external_provider_meta: dict[str, Any] | None,
    external_raw: pd.DataFrame | None,
) -> dict[str, Any]:
    if spec.external_provider_uri is None:
        return {}
    payload = {
        "external_15min": {
            "provider_uri": spec.external_provider_uri.resolve().as_posix(),
            "provider_card": spec.external_provider_card,
            "requested_symbols": REQUESTED_EXTERNAL_CONTEXT_SYMBOLS,
            "qlib_feature_rows": int(len(external_raw)) if external_raw is not None else 0,
            "manifest": (spec.external_provider_uri / "qlib_store_manifest.json").as_posix(),
        }
    }
    if external_provider_meta is not None:
        payload["external_15min"].update(
            {
                "calendar_start": external_provider_meta["calendar_start"],
                "calendar_end": external_provider_meta["calendar_end"],
            }
        )
    return payload


def _external_context_summary(
    spec: IntradayExecutionLabelModelSpec,
    *,
    external_provider_meta: dict[str, Any] | None,
    external_context_data_status: str,
    frame: pd.DataFrame,
    feature_cols: list[str],
) -> dict[str, Any]:
    if spec.external_provider_uri is None:
        return {
            "available_symbols": list(spec.instruments),
            "missing_requested_symbols": REQUESTED_EXTERNAL_INTRADAY_SYMBOLS,
            "reason": "current 15min qlib provider contains SOXL and SOXX only",
        }

    label_external = {}
    if "external_context" in frame.attrs:
        label_external = frame.attrs["external_context"]
    external_label_audit = {}
    if "external_context_available" in frame.columns:
        external_label_audit["available_rows"] = int(frame["external_context_available"].sum())

    available_symbols = REQUESTED_EXTERNAL_CONTEXT_SYMBOLS
    missing_symbols: list[str] = []
    if external_provider_meta is not None:
        manifest_path = spec.external_provider_uri / "qlib_store_manifest.json"
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                available_symbols = sorted(str(symbol).upper() for symbol in manifest.get("instruments", available_symbols))
                missing_symbols = sorted(set(REQUESTED_EXTERNAL_CONTEXT_SYMBOLS) - set(available_symbols))
            except json.JSONDecodeError:
                missing_symbols = []

    return {
        "provider_uri": spec.external_provider_uri.resolve().as_posix(),
        "provider_card": spec.external_provider_card,
        "requested_symbols": REQUESTED_EXTERNAL_CONTEXT_SYMBOLS,
        "available_symbols": available_symbols,
        "missing_requested_symbols": missing_symbols,
        "data_status": external_context_data_status,
        "coverage_start": external_provider_meta["calendar_start"] if external_provider_meta is not None else "",
        "coverage_end": external_provider_meta["calendar_end"] if external_provider_meta is not None else "",
        "required_walk_forward_start": spec.required_walk_forward_start,
        "required_walk_forward_satisfied": bool(
            external_provider_meta is not None
            and external_provider_meta["calendar_start"][:10] <= spec.required_walk_forward_start
        ),
        "feature_count": len(_external_context_feature_names(feature_cols)),
        "features": _external_context_feature_names(feature_cols),
        **label_external,
        **external_label_audit,
    }


def _external_context_feature_names(feature_cols: list[str]) -> list[str]:
    prefixes = (
        "qqq_",
        "smh_",
        "external_soxx_",
        "nvda_",
        "avgo_",
        "amd_",
        "tsm_",
        "asml_",
        "semis_",
        "qqq_smh_soxx_lead_lag_",
    )
    return [feature for feature in feature_cols if feature.startswith(prefixes)]


def _external_context_column_prefix(symbol: str) -> str:
    normalized = symbol.lower()
    return "external_soxx" if normalized == "soxx" else normalized


def _walk_forward_years(walk_forward: pd.DataFrame) -> list[int]:
    if walk_forward.empty or "test_year" not in walk_forward.columns:
        return []
    return sorted(int(value) for value in walk_forward["test_year"].dropna().unique())


def _required_walk_forward_satisfied(
    spec: IntradayExecutionLabelModelSpec,
    *,
    external_provider_meta: dict[str, Any] | None,
    walk_forward: pd.DataFrame,
) -> bool:
    years = set(_walk_forward_years(walk_forward))
    has_required_folds = years >= {2023, 2024, 2025, 2026}
    if spec.external_provider_uri is None:
        return has_required_folds
    return bool(
        external_provider_meta is not None
        and external_provider_meta["calendar_start"][:10] <= spec.required_walk_forward_start
        and has_required_folds
    )


def _format_datetime(value: Any) -> str:
    if pd.isna(value):
        return ""
    return pd.Timestamp(value).strftime("%Y-%m-%d %H:%M:%S")
