from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.model import (
    PanelSequenceSpec,
    RegimeMoESpec,
    RiskAdjustedGateSpec,
    SegmentWindow,
    TailBudgetSpec,
    aggregate_walk_forward,
    build_risk_adjusted_labels,
    compute_risk_metrics,
    evaluate_promotion,
    run_panel_sequence_models,
    run_regime_moe_models,
    run_tail_budget_models,
)


QLIB_DOCS = {
    "data": "https://qlib.readthedocs.io/en/latest/component/data.html",
    "dataset": "https://qlib.readthedocs.io/en/latest/component/data.html#data-handler",
    "recorder": "https://qlib.readthedocs.io/en/latest/component/recorder.html",
}

RECORDER_ONLY_ARTIFACTS = {
    "predictions": "artifacts/pred.pkl",
    "labels": "artifacts/label.pkl",
    "all_model_predictions": "artifacts/model_selection/all_model_predictions.csv",
    "portfolio_report": "artifacts/portfolio_analysis/report_normal_1day.pkl",
    "positions": "artifacts/portfolio_analysis/positions_normal_1day.pkl",
    "risk_analysis": "artifacts/portfolio_analysis/port_analysis_1day.pkl",
    "legacy_report": "artifacts/evidence/legacy_report.pkl",
}

FRONTIER_RECORDER_ONLY_ARTIFACTS = {
    "canonical_dataset_audit": "artifacts/frontier/canonical_dataset_audit.pkl",
    "route_summaries": "artifacts/frontier/route_summaries.pkl",
}

REQUIRED_PROMOTION_CHECKS = frozenset(
    {
        "data_quality_valid",
        "label_leakage_free",
        "test_time_selection_free",
        "labels_complete",
        "test_candidate_metrics_valid",
        "test_legacy_metrics_valid",
        "walk_forward_candidate_metrics_valid",
        "walk_forward_legacy_metrics_valid",
        "walk_forward_candidate_structure_valid",
        "walk_forward_legacy_structure_valid",
        "test_extreme_loss_threshold_comparable",
        "walk_forward_extreme_loss_threshold_comparable",
        "test_observation_count_matched",
        "walk_forward_fold_counts_matched",
        "candidate_walk_forward_fold_count",
        "legacy_walk_forward_fold_count",
        "walk_forward_observation_count_matched",
        "walk_forward_positive_folds",
        "nested_walk_forward",
        *{
            f"{scope}_{metric}"
            for scope in ("test", "walk_forward")
            for metric in (
                "max_drawdown",
                "participation",
                "zero_position_day_ratio",
                "positive_return_capture",
                "return_to_drawdown",
                "cvar_5pct",
                "worst_five_day_mean",
                "extreme_loss_day_count",
                "total_turnover",
            )
        },
    }
)

ROUTES = ("tail_budget", "regime_moe", "panel_sequence")
CANONICAL_RECORDER_URI = "git-common-parent:.qlib-recorders/soxl-risk-adjusted-frontier-v1"


def resolve_recorder_uri(locator: str | Path = CANONICAL_RECORDER_URI) -> Path:
    value = str(locator)
    prefix = "git-common-parent:"
    if not value.startswith(prefix):
        return Path(value).resolve()
    common = subprocess.check_output(
        ["git", "rev-parse", "--git-common-dir"], text=True
    ).strip()
    common_path = Path(common)
    if not common_path.is_absolute():
        common_path = (Path.cwd() / common_path).resolve()
    main_repo = common_path.parent
    return (main_repo.parent / value.removeprefix(prefix)).resolve()


@dataclass(frozen=True)
class RiskAdjustedRouteCardSpec:
    experiment: str
    route: str
    output_dir: Path
    provider_uri: Path = Path("data/qlib/soxl")
    recorder_uri: Path = Path(CANONICAL_RECORDER_URI)
    market: str = "all"
    region: str = "us"
    data_start: str = "2014-01-02"
    as_of: str = "2026-07-02"
    train: tuple[str, str] = ("2014-01-02", "2021-12-31")
    valid: tuple[str, str] = ("2022-01-03", "2023-12-29")
    test: tuple[str, str] = ("2024-01-02", "2026-07-02")
    cost_rate: float = 0.0004

    def __post_init__(self) -> None:
        if self.route not in ROUTES:
            raise ValueError(f"unknown risk-adjusted route: {self.route}")
        if self.experiment != f"soxl_{self.route}_v1" and not (
            self.route == "panel_sequence" and self.experiment == "soxl_panel_sequence_v1"
        ):
            raise ValueError("experiment name must match the route card")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        item = float(value)
        return item if np.isfinite(item) else None
    if value is None or isinstance(value, (str, int)):
        return value
    return str(value)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def _recorder_files(recorder_dir: Path) -> list[str]:
    return sorted(
        path.relative_to(recorder_dir).as_posix()
        for path in recorder_dir.rglob("*")
        if path.is_file()
        and path.name not in {"code_cached.txt", "code_diff.txt", "code_status.txt"}
    )


def _remove_transient_recorder_files(recorder_dir: Path) -> None:
    for name in ("code_cached.txt", "code_diff.txt", "code_status.txt"):
        path = recorder_dir / "artifacts" / name
        if path.exists():
            path.unlink()


def build_card_summary(
    *,
    experiment: str,
    route: str,
    provider_uri: Path,
    recorder_uri: Path,
    recorder_id: str,
    recorder_local_dir: Path,
    data_range: tuple[str, str],
    segments: Mapping[str, tuple[str, str]],
    route_payload: Mapping[str, Any],
    sidecars: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "experiment": experiment,
        "route": route,
        "provider_uri": provider_uri.as_posix(),
        "recorder_uri": recorder_uri.as_posix(),
        "recorder_id": recorder_id,
        "market": "all",
        "region": "us",
        "as_of": data_range[1],
        "data_range": list(data_range),
        "segments": {name: list(window) for name, window in segments.items()},
        "label_contract": {
            "future_return_5d": "five compounded next-open daily returns",
            "future_downside_5d": "negative part of future_return_5d",
            "future_crash_5d": "train tenth percentile or daily next-open return <= -8%",
            "horizon": 5,
            "purge_rows": 6,
            "decision_time": "after close_t",
            "entry_time": "next session open",
        },
        "selection_frozen_before_test": bool(route_payload["selection_frozen_before_test"]),
        "test_labels_used_for_selection": bool(route_payload["test_labels_used_for_selection"]),
        **_jsonable(route_payload),
        "qlib_native_api": [
            "qlib.init",
            "qlib.data.D.features",
            "DatasetH",
            "DataHandlerLP",
            "qlib.workflow.R.start",
            "qlib.workflow.Recorder.save_objects",
        ],
        "qlib_docs": QLIB_DOCS,
        "sidecars": dict(sidecars),
        "artifact_policy": {
            "policy": "slim_card_recorder_first",
            "primary_artifact": f"{experiment}.ipynb",
            "workflow_config": "workflow.yaml",
            "machine_entrypoint": "summary.json",
            "kept_sidecars": sorted(sidecars.values()),
            "recorder_only_artifacts": dict(RECORDER_ONLY_ARTIFACTS),
        },
    }


def build_calendar_alignment_evidence(
    expected_index: pd.DatetimeIndex,
    features: pd.DataFrame,
    labels: pd.DataFrame,
) -> dict[str, Any]:
    expected = pd.DatetimeIndex(expected_index)
    feature_index = pd.DatetimeIndex(features.index)
    label_index = pd.DatetimeIndex(labels.index)
    calendar_complete = feature_index.equals(expected) and label_index.equals(expected)
    feature_invalid = ~np.isfinite(features.to_numpy(dtype="float64")).all(axis=1)
    label_invalid = ~np.isfinite(labels.to_numpy(dtype="float64")).all(axis=1)
    return {
        "expected_rows": int(len(expected)),
        "calendar_rows_complete": bool(calendar_complete),
        "data_quality_valid": bool(calendar_complete and not feature_invalid.any()),
        "labels_complete": bool(calendar_complete and not label_invalid.any()),
        "internal_feature_missing_dates": [
            timestamp.date().isoformat() for timestamp in feature_index[feature_invalid]
        ],
        "internal_label_missing_dates": [
            timestamp.date().isoformat() for timestamp in label_index[label_invalid]
        ],
    }


def _canonical_dataset(spec: RiskAdjustedRouteCardSpec):
    import qlib
    from qlib.data import D
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP

    provider = spec.provider_uri.resolve()
    if not (provider / "qlib_store_manifest.json").exists():
        raise FileNotFoundError("blocked_missing_required_source: canonical qlib manifest")
    manifest = json.loads((provider / "qlib_store_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("as_of") != spec.as_of:
        raise ValueError(
            f"blocked_stale_data: expected canonical as_of {spec.as_of}, got {manifest.get('as_of')}"
        )
    qlib.init(provider_uri=provider.as_posix(), region=spec.region, kernels=1)
    raw = D.features(
        ["SOXL", "SOXX"],
        ["$open", "$high", "$low", "$close", "$volume", "$vix"],
        start_time=spec.data_start,
        end_time=spec.as_of,
        freq="day",
    )
    if raw.empty:
        raise ValueError("blocked_missing_required_source: empty qlib feature response")
    soxl = raw.xs("SOXL", level="instrument").sort_index()
    soxx = raw.xs("SOXX", level="instrument").sort_index()
    aligned = soxl.index.intersection(soxx.index)
    soxl = soxl.loc[aligned]
    soxx = soxx.loc[aligned]
    if pd.Timestamp(aligned.max()).date().isoformat() != spec.as_of:
        raise ValueError("blocked_stale_data: qlib D.features max date does not match manifest")

    soxl_return = soxl["$close"].pct_change()
    soxx_return = soxx["$close"].pct_change()
    vix_level = soxl["$vix"]
    vix_change = vix_level.pct_change()
    rolling_high = soxl["$close"].rolling(60, min_periods=20).max()
    rolling_low = soxl["$close"].rolling(14, min_periods=5).min()
    rolling_span = soxl["$close"].rolling(14, min_periods=5).max() - rolling_low
    raw_features = pd.DataFrame(
        {
            "soxl_return": soxl_return,
            "soxx_return": soxx_return,
            "vix_level": vix_level,
            "vix_change": vix_change,
            "lagged_return": soxl_return,
            "trend": soxl["$close"].pct_change(20),
            "soxl_trend": soxl["$close"].pct_change(20),
            "volatility": soxl_return.rolling(20, min_periods=10).std() * np.sqrt(252.0),
            "soxl_volatility": soxl_return.rolling(20, min_periods=10).std() * np.sqrt(252.0),
            "breadth": 0.5 * np.sign(soxl_return) + 0.5 * np.sign(soxx_return),
            "cross_asset_momentum": soxx["$close"].pct_change(20),
            "oversold": -(soxl["$close"] - rolling_low) / rolling_span.replace(0.0, np.nan),
            "residual": soxl_return - 3.0 * soxx_return,
            "range": (soxl["$high"] - soxl["$low"]) / soxl["$close"],
            "volatility_gap": soxl_return.rolling(20, min_periods=10).std() * np.sqrt(252.0)
            - vix_level / 100.0,
            "soxl_drawdown_60": soxl["$close"] / rolling_high - 1.0,
            "soxl_volume_ratio_20": soxl["$volume"] / soxl["$volume"].rolling(20, min_periods=10).mean(),
            "soxx_trend_20": soxx["$close"].pct_change(20),
            "vix_percentile_252": vix_level.rolling(252, min_periods=60).rank(pct=True),
        },
        index=aligned,
    ).shift(1)

    next_open_return = soxl["$open"].shift(-2) / soxl["$open"].shift(-1) - 1.0
    windows = (
        SegmentWindow("train", *spec.train),
        SegmentWindow("valid", *spec.valid),
        SegmentWindow("test", *spec.test),
    )
    labels = build_risk_adjusted_labels(next_open_return, segments=windows)
    label_frame = labels.labels.drop(columns=["segment", "future_min_daily_return_5d"])
    calendar = pd.DatetimeIndex(
        D.calendar(start_time=spec.data_start, end_time=spec.as_of, freq="day")
    )
    train_feature_window = raw_features.loc[spec.train[0] : spec.train[1]]
    finite_train = np.isfinite(train_feature_window.to_numpy(dtype="float64")).all(axis=1)
    if not finite_train.any():
        raise ValueError("blocked_missing_required_source: no complete train feature row")
    feature_warmup_end = pd.Timestamp(train_feature_window.index[np.flatnonzero(finite_train)[0]])
    expected_by_segment: dict[str, pd.DatetimeIndex] = {}
    for name, (start, end) in {"train": spec.train, "valid": spec.valid, "test": spec.test}.items():
        segment_calendar = calendar[(calendar >= pd.Timestamp(start)) & (calendar <= pd.Timestamp(end))]
        eligible = segment_calendar[: max(len(segment_calendar) - 6, 0)]
        if name == "train":
            eligible = eligible[eligible >= feature_warmup_end]
        if eligible.empty:
            raise ValueError(f"blocked_missing_required_source: empty {name} calendar segment")
        expected_by_segment[name] = eligible
    expected = pd.DatetimeIndex(
        np.concatenate([values.to_numpy() for values in expected_by_segment.values()])
    )
    feature_table = raw_features.reindex(expected).replace([np.inf, -np.inf], np.nan)
    aligned_labels = label_frame.reindex(expected).replace([np.inf, -np.inf], np.nan)
    table = feature_table.join(aligned_labels)
    alignment = build_calendar_alignment_evidence(expected, feature_table, aligned_labels)
    index = pd.MultiIndex.from_arrays(
        [["SOXL"] * len(table), table.index], names=["instrument", "datetime"]
    )
    frame = pd.DataFrame(
        table.to_numpy(dtype="float64"),
        index=index,
        columns=pd.MultiIndex.from_tuples(
            [("feature", name) for name in raw_features.columns]
            + [
                ("label", "future_return_5d"),
                ("label", "future_downside_5d"),
                ("label", "future_crash_5d"),
            ]
        ),
    )
    effective_segments: dict[str, tuple[str, str]] = {}
    segment_alignment: dict[str, Any] = {}
    for name, dates in expected_by_segment.items():
        effective_segments[name] = (
            pd.Timestamp(dates.min()).date().isoformat(),
            pd.Timestamp(dates.max()).date().isoformat(),
        )
        segment_alignment[name] = build_calendar_alignment_evidence(
            dates,
            feature_table.reindex(dates),
            aligned_labels.reindex(dates),
        )
    dataset = DatasetH(handler=DataHandlerLP.from_df(frame), segments=effective_segments)
    lineage = {
        "source": "qlib.data.D.features",
        "provider_uri": spec.provider_uri.as_posix(),
        "manifest_hash": manifest.get("hash"),
        "manifest_as_of": manifest.get("as_of"),
        "calendar_start": manifest.get("calendar_start"),
        "calendar_end": manifest.get("calendar_end"),
        "instruments": ["SOXL", "SOXX"],
        "fields": ["$open", "$high", "$low", "$close", "$volume", "$vix"],
        "feature_rows": int(len(frame)),
        "feature_count": int(len(raw_features.columns)),
        "feature_warmup_end": feature_warmup_end.date().isoformat(),
        "calendar_alignment": alignment,
        "segment_alignment": segment_alignment,
    }
    return dataset, next_open_return, labels, effective_segments, lineage


def _run_route(route: str, dataset, lineage: Mapping[str, Any]):
    if route == "tail_budget":
        return run_tail_budget_models(
            dataset,
            spec=TailBudgetSpec(
                lightgbm_rounds=16,
                catboost_iterations=20,
                densemble_epochs=8,
                random_state=20260717,
            ),
        )
    if route == "regime_moe":
        return run_regime_moe_models(
            dataset,
            spec=RegimeMoESpec(catboost_iterations=20, random_state=20260717),
        )
    return run_panel_sequence_models(
        dataset,
        spec=PanelSequenceSpec(
            hidden_width=8,
            temporal_layers=1,
            dropout=0.10,
            max_epochs=4,
            early_stopping_patience=2,
            batch_size=64,
            random_state=20260717,
        ),
        pit_panel=None,
        qlib_lineage=lineage,
    )


def _series_datetime(series: pd.Series) -> pd.Series:
    result = series.copy()
    if isinstance(result.index, pd.MultiIndex):
        result.index = pd.DatetimeIndex(result.index.get_level_values("datetime"))
    else:
        result.index = pd.DatetimeIndex(result.index)
    return result.sort_index()


class _FrozenPositionSignalModel:
    """Qlib SignalRecord adapter for a validation-frozen continuous position path."""

    def __init__(self, positions: pd.Series) -> None:
        self.positions = positions.astype("float64").rename("score")

    def predict(self, dataset) -> pd.Series:
        del dataset
        return self.positions.copy()


def _legacy_paths(start: str, end: str, dates: pd.DatetimeIndex):
    from soxl_backtest.reference import replay_soxlbacktest_reference

    legacy = replay_soxlbacktest_reference(start=start, end=end)
    daily = legacy.daily.copy()
    daily.index = pd.to_datetime(daily.pop("date"))
    legacy_return = daily["capital"].pct_change().fillna(0.0).reindex(dates).fillna(0.0)
    legacy_position = daily["position"].reindex(dates).ffill().fillna(0.0)
    return legacy, legacy_return, legacy_position


def _metrics_dict(metrics) -> dict[str, Any]:
    payload = asdict(metrics)
    payload.pop("daily_returns", None)
    payload.pop("positions", None)
    payload.pop("daily_turnover", None)
    payload["return_to_drawdown"] = metrics.return_to_drawdown
    return _jsonable(payload)


def _compact_selection_evidence(evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Keep the card entry surface small; full evidence is recorder-only."""

    keys = (
        "route_status",
        "selection_segment",
        "threshold_fit_segment",
        "preprocessing_fit_segment",
        "calibration_fit_segment",
        "test_labels_used_for_selection",
        "test_accessed_after_freeze",
        "selected_validation_utility",
        "selected_window",
        "selected_family",
        "label_horizon",
        "purge_rows",
        "random_state",
        "regime_occupancy",
        "usable_regime_observations",
        "min_regime_observations",
        "test_prediction_failure",
        "data_quality_failures_by_segment",
    )
    return _jsonable({key: evidence[key] for key in keys if key in evidence})


def _derive_selection_gates(
    route: str, selection_evidence: Mapping[str, Any], lineage: Mapping[str, Any]
) -> dict[str, bool]:
    quality = selection_evidence.get(
        "data_quality_failures_by_segment",
        selection_evidence.get("data_quality_evidence", {}),
    )
    data_quality_valid = bool(quality) and all(
        item.get("invalid_count") == 0 for item in quality.values()
    ) and all(
        item.get("data_quality_valid") is True
        for item in lineage.get("segment_alignment", {}).values()
    )
    frozen = isinstance(selection_evidence.get("frozen_policy"), Mapping)
    common = bool(
        selection_evidence.get("selection_segment") == "valid"
        and selection_evidence.get("test_labels_used_for_selection") is False
        and selection_evidence.get("test_accessed_after_freeze") is True
        and frozen
    )
    if route == "tail_budget":
        route_leakage = (
            selection_evidence.get("preprocessing_fit_segment") == "train"
            and selection_evidence.get("calibration_fit_segment") == "train"
        )
    elif route == "regime_moe":
        route_leakage = selection_evidence.get("threshold_fit_segment") == "train"
    else:
        trace = tuple(selection_evidence.get("event_trace", ()))
        route_leakage = (
            selection_evidence.get("early_stopping_segment") == "valid"
            and trace == ("candidate_selection_frozen", "test_dk_i_accessed")
        )
    labels_complete = all(
        item.get("labels_complete") is True
        for item in lineage.get("segment_alignment", {}).values()
    )
    return {
        "data_quality_valid": bool(data_quality_valid),
        "label_leakage_free": bool(common and route_leakage and labels_complete),
        "test_time_selection_free": bool(common),
        "labels_complete": bool(labels_complete),
        "frozen_policy": bool(frozen),
    }


def _route_metrics_from_report(
    *,
    route: str,
    report: pd.DataFrame,
    legacy_report: pd.DataFrame,
    selection_evidence: Mapping[str, Any],
    lineage: Mapping[str, Any],
):
    report = report.sort_index()
    required_report_columns = {"return", "cost", "turnover", "value", "account"}
    if not required_report_columns.issubset(report.columns):
        raise ValueError("qlib portfolio report is missing required native columns")
    legacy_report = legacy_report.sort_index()
    if not required_report_columns.issubset(legacy_report.columns):
        raise ValueError("qlib legacy portfolio report is missing required native columns")
    if not legacy_report.index.equals(report.index):
        raise ValueError("candidate and legacy qlib portfolio reports must use one matched calendar")
    candidate_return = (report["return"].astype("float64") - report["cost"].astype("float64")).fillna(0.0)
    position = (report["value"].astype("float64") / report["account"].replace(0.0, np.nan)).fillna(0.0)
    position = position.clip(0.0, 1.0)
    legacy, _, _ = _legacy_paths(
        position.index.min().date().isoformat(),
        position.index.max().date().isoformat(),
        position.index,
    )
    legacy_return = (
        legacy_report["return"].astype("float64")
        - legacy_report["cost"].astype("float64")
    ).fillna(0.0)
    legacy_position = (
        legacy_report["value"].astype("float64")
        / legacy_report["account"].replace(0.0, np.nan)
    ).fillna(0.0).clip(0.0, 1.0)
    candidate_turnover = report["turnover"].astype("float64").fillna(0.0)
    legacy_turnover = legacy_report["turnover"].astype("float64").fillna(0.0)
    candidate_test = compute_risk_metrics(
        candidate_return, position, daily_turnover=candidate_turnover
    )
    legacy_test = compute_risk_metrics(
        legacy_return, legacy_position, daily_turnover=legacy_turnover
    )
    chunks = np.array_split(np.arange(len(position)), 3)
    candidate_folds = tuple(
        compute_risk_metrics(
            candidate_return.iloc[chunk],
            position.iloc[chunk],
            daily_turnover=candidate_turnover.iloc[chunk],
        )
        for chunk in chunks
    )
    legacy_folds = tuple(
        compute_risk_metrics(
            legacy_return.iloc[chunk],
            legacy_position.iloc[chunk],
            daily_turnover=legacy_turnover.iloc[chunk],
        )
        for chunk in chunks
    )
    candidate_wf = aggregate_walk_forward(candidate_folds)
    legacy_wf = aggregate_walk_forward(legacy_folds)
    gates = _derive_selection_gates(route, selection_evidence, lineage)
    evaluation = evaluate_promotion(
        candidate_test,
        legacy_test,
        candidate_wf,
        legacy_wf,
        spec=RiskAdjustedGateSpec(),
        data_quality_valid=gates["data_quality_valid"],
        label_leakage_free=gates["label_leakage_free"],
        test_time_selection_free=gates["test_time_selection_free"],
        labels_complete=gates["labels_complete"],
    )
    checks = dict(evaluation.checks)
    checks["nested_walk_forward"] = False
    failed = tuple(dict.fromkeys((*evaluation.failed_checks, "nested_walk_forward")))
    promotion = {
        "promoted": False,
        "decision": "not_promoted_failed_risk_adjusted_gates",
        "checks": checks,
        "failed_checks": list(failed),
    }
    project_report = report.copy()
    project_report["candidate_return_after_cost"] = candidate_return
    project_report["actual_position_weight"] = position
    project_report["legacy_return"] = legacy_return
    project_report["legacy_position"] = legacy_position
    project_report["legacy_turnover"] = legacy_turnover
    risk = pd.DataFrame(
        [
            {"scope": "test", "side": "candidate", **_metrics_dict(candidate_test)},
            {"scope": "test", "side": "legacy", **_metrics_dict(legacy_test)},
            {"scope": "walk_forward", "side": "candidate", **_metrics_dict(candidate_wf.metrics)},
            {"scope": "walk_forward", "side": "legacy", **_metrics_dict(legacy_wf.metrics)},
        ]
    )
    wf_rows = []
    for idx, (candidate, legacy_fold) in enumerate(zip(candidate_folds, legacy_folds), start=1):
        wf_rows.extend(
            [
                {"fold": idx, "side": "candidate", **_metrics_dict(candidate)},
                {"fold": idx, "side": "legacy", **_metrics_dict(legacy_fold)},
            ]
        )
    return {
        "candidate_test": candidate_test,
        "legacy_test": legacy_test,
        "candidate_wf": candidate_wf,
        "legacy_wf": legacy_wf,
        "promotion": promotion,
        "report": project_report,
        "risk": risk,
        "walk_forward": pd.DataFrame(wf_rows),
        "legacy": legacy,
        "selection_gates": gates,
    }


def _matched_legacy_qlib_report(
    spec: RiskAdjustedRouteCardSpec, candidate_positions: pd.Series
) -> pd.DataFrame:
    """Replay legacy target weights through the same qlib strategy and exchange contract."""

    from qlib.backtest import backtest
    from qlib.data import D
    from soxl_backtest.reference import replay_soxlbacktest_reference

    positions = _series_datetime(candidate_positions)
    calendar = pd.DatetimeIndex(
        D.calendar(
            start_time=positions.index.min(),
            end_time=spec.as_of,
            freq="day",
        )
    )
    later = calendar[calendar > positions.index.max()]
    if later.empty:
        raise ValueError("legacy qlib replay requires the next execution session")
    end = pd.Timestamp(later.min())
    legacy = replay_soxlbacktest_reference(
        start=positions.index.min().date().isoformat(), end=end.date().isoformat()
    )
    daily = legacy.daily.copy()
    daily.index = pd.to_datetime(daily.pop("date"))
    weights = daily["position"].reindex(positions.index).ffill().fillna(0.0).rename("score")
    weights.index = pd.MultiIndex.from_arrays(
        [weights.index, ["SOXL"] * len(weights)], names=["datetime", "instrument"]
    )
    portfolio, _ = backtest(
        start_time=positions.index.min(),
        end_time=end,
        strategy={
            "class": "ContinuousSignalWeightStrategy",
            "module_path": "src.backtest.strategy",
            "kwargs": {
                "signal": weights,
                "instrument": "SOXL",
                "min_weight": 0.0,
                "max_weight": 1.0,
                "risk_degree": 1.0,
            },
        },
        executor={
            "class": "SimulatorExecutor",
            "module_path": "qlib.backtest.executor",
            "kwargs": {"time_per_step": "day", "generate_portfolio_metrics": True},
        },
        benchmark="SOXL",
        account=100000.0,
        exchange_kwargs={
            "freq": "day",
            "limit_threshold": None,
            "deal_price": "open",
            "open_cost": spec.cost_rate / 2.0,
            "close_cost": spec.cost_rate / 2.0,
            "min_cost": 0.0,
        },
    )
    report, _ = portfolio["1day"]
    return report


def _record_route(
    spec: RiskAdjustedRouteCardSpec,
    result,
    dataset,
):
    from qlib.workflow import R
    from qlib.workflow.record_temp import PortAnaRecord, SignalRecord

    recorder_uri = resolve_recorder_uri(spec.recorder_uri)
    recorder_uri.mkdir(parents=True, exist_ok=True)
    R.set_uri(recorder_uri.as_posix())
    legacy_report = _matched_legacy_qlib_report(
        spec, result.positions_by_segment["test"]
    )
    all_predictions = pd.concat(
        {name: values for name, values in result.predictions_by_segment.items()}, names=["segment"]
    )
    with tempfile.TemporaryDirectory(prefix="soxl-risk-adjusted-recorder-") as temp:
        temp_dir = Path(temp)
        all_path = temp_dir / "all_model_predictions.csv"
        all_predictions.to_csv(all_path)
        with R.start(experiment_name=spec.experiment):
            recorder = R.get_recorder()
            R.log_params(
                route=spec.route,
                provider_uri=spec.provider_uri.as_posix(),
                as_of=spec.as_of,
                label_horizon=5,
                purge_rows=6,
                selection_segment="valid",
                test_labels_used_for_selection=False,
                selected_candidate=result.selected_candidate,
                cost_rate=spec.cost_rate,
                deal_price="open",
                execution_timing="signal_after_close_t_to_next_open",
                backtest_strategy="ContinuousSignalWeightStrategy",
            )
            recorder.log_artifact(all_path.as_posix(), artifact_path="model_selection")
            model = _FrozenPositionSignalModel(result.positions_by_segment["test"])
            SignalRecord(model, dataset, recorder).generate()
            PortAnaRecord(
                recorder,
                {
                    "strategy": {
                        "class": "ContinuousSignalWeightStrategy",
                        "module_path": "src.backtest.strategy",
                        "kwargs": {
                            "signal": "<PRED>",
                            "instrument": "SOXL",
                            "min_weight": 0.0,
                            "max_weight": 1.0,
                            "risk_degree": 1.0,
                        },
                    },
                    "backtest": {
                        "start_time": None,
                        "end_time": None,
                        "account": 100000.0,
                        "benchmark": "SOXL",
                        "exchange_kwargs": {
                            "freq": "day",
                            "limit_threshold": None,
                            "deal_price": "open",
                            "open_cost": spec.cost_rate / 2.0,
                            "close_cost": spec.cost_rate / 2.0,
                            "min_cost": 0.0,
                        },
                    },
                },
                risk_analysis_freq="day",
            ).generate()
            R.save_objects(artifact_path="model", **{"fitted_policy.pkl": result.fitted_policy})
            R.save_objects(artifact_path="evidence", **{"selection_evidence.pkl": result.selection_evidence})
            R.save_objects(artifact_path="evidence", **{"legacy_report.pkl": legacy_report})
            recorder_id = recorder.id
            recorder_local_dir = Path(recorder.get_local_dir())
    _remove_transient_recorder_files(recorder_local_dir)
    persisted = R.get_recorder(experiment_name=spec.experiment, recorder_id=recorder_id)
    if persisted.status != "FINISHED":
        raise RuntimeError(f"recorder write did not finish: {recorder_id} ({persisted.status})")
    for artifact in (
        "pred.pkl",
        "label.pkl",
        "portfolio_analysis/report_normal_1day.pkl",
        "portfolio_analysis/positions_normal_1day.pkl",
        "portfolio_analysis/port_analysis_1day.pkl",
        "evidence/selection_evidence.pkl",
        "evidence/legacy_report.pkl",
    ):
        try:
            persisted.load_object(artifact)
        except Exception as exc:
            raise RuntimeError(f"recorder write is incomplete: {artifact}") from exc
    return recorder_id, recorder_local_dir


def _workflow_text(spec: RiskAdjustedRouteCardSpec, segments: Mapping[str, tuple[str, str]]) -> str:
    lines = [
        f"experiment: {spec.experiment}",
        f"route: {spec.route}",
        "qlib:",
        f"  provider_uri: {spec.provider_uri.as_posix()}",
        f"  recorder_uri: {spec.recorder_uri.as_posix()}",
        f"  market: {spec.market}",
        f"  region: {spec.region}",
        "segments:",
    ]
    for name, window in segments.items():
        lines.extend([f"  {name}:", f"    start: {window[0]}", f"    end: {window[1]}"])
    lines.extend(
        [
            "model_selection:",
            "  segment: valid",
            "  test_reselection: false",
            "execution:",
            "  timing: signal_after_close_t_to_next_open",
            "  deal_price: open",
            f"  cost_rate: {spec.cost_rate}",
            "artifact_policy:",
            "  policy: slim_card_recorder_first",
            "  machine_entrypoint: summary.json",
            "  workflow_config: workflow.yaml",
            "",
        ]
    )
    return "\n".join(lines)


def _route_conclusion(summary: Mapping[str, Any]) -> str:
    metrics = summary["metrics"]["test"]["candidate"]
    legacy = summary["metrics"]["test"]["legacy"]
    blocked = summary.get("blocked_modes", {})
    return f"""# {summary['experiment']} 结论

## 做了什么

使用 canonical qlib store、DatasetH/DataHandlerLP 和 qlib Recorder 执行 `{summary['route']}` 路线；模型、预测、标签、仓位和组合证据均登记在 recorder `{summary['recorder_id']}`。

## 核心指标

- 候选测试收益：{metrics['total_return']:.4f}；最大回撤：{metrics['max_drawdown']:.4f}；平均绝对仓位：{metrics['avg_abs_position']:.4f}。
- 匹配 legacy 收益：{legacy['total_return']:.4f}；最大回撤：{legacy['max_drawdown']:.4f}。
- as-of 为 {summary['as_of']}，没有外推到更新日期。

## Regime / window 差异

路线状态为 `{summary['route_status']}`；选择候选为 `{summary['selected_candidate']}`。Regime 或窗口明细见 sidecars；序列横截面模式状态为 `{blocked.get('cross_sectional_pit_panel', 'not_applicable')}`。

## 晋升决定

`{summary['promotion']['decision']}`。失败门包括：{', '.join(summary['promotion']['failed_checks'])}。

## 风险

三折报告为冻结策略的时间切片诊断，尚未完成每折重新拟合的 nested walk-forward，所以 `nested_walk_forward` 明确失败；canonical 数据截至 {summary['as_of']}。本结果不构成买卖建议。

## 下一步

先补齐 nested expanding-window 重训；序列横截面路线还需经过验证的 PIT 成分股历史，再决定是否进入 validated card。
"""


def notebook_conclusion_markdown(summary: Mapping[str, Any]) -> str:
    """Render the notebook's final Chinese conclusion from its current summary."""

    if summary.get("route") != "frontier":
        return _route_conclusion(summary)
    comparison = summary["frontier"]
    return f"""# SOXL 风险调整模型前沿结论

## 做了什么

通过 qlib Recorder 对象读取三条冻结路线的预测、标签、组合报告、仓位、风险分析和选择证据；前沿 recorder 为 `{summary['recorder_id']}`，没有在 test 上重新选择或重训。

## 核心指标

三条路线均保留在 `model_selection.csv`；最终决定为 `{comparison['decision']}`。

## Regime / window 差异

Regime MoE、tail budget、20/60 日序列结果按同一窗口和执行成本比较；blocked 路线不会被省略。

## 晋升决定

`{comparison['decision']}`。

## 风险

canonical 数据截至 {summary['as_of']}；三折仍是冻结策略诊断，nested walk-forward 未完成。本结果不构成买卖建议。

## 下一步

完成 expanding-window nested 重训，并补齐 PIT 成分股历史后再尝试 validated promotion。
"""


def sync_notebook_conclusion(path: Path, summary: Mapping[str, Any]) -> None:
    """Replace or append the final markdown cell with summary-derived evidence."""

    import nbformat

    notebook = nbformat.read(path, as_version=4)
    conclusion = notebook_conclusion_markdown(summary)
    if notebook.cells and notebook.cells[-1].cell_type == "markdown":
        notebook.cells[-1].source = conclusion
    else:
        notebook.cells.append(nbformat.v4.new_markdown_cell(conclusion))
    nbformat.write(notebook, path)


def run_risk_adjusted_route_card(spec: RiskAdjustedRouteCardSpec) -> dict[str, Any]:
    spec.output_dir.mkdir(parents=True, exist_ok=True)
    dataset, next_open_return, labels, segments, lineage = _canonical_dataset(spec)
    result = _run_route(spec.route, dataset, lineage)
    del next_open_return
    recorder_id, recorder_local_dir = _record_route(spec, result, dataset)
    recorder_evidence = load_route_recorder_evidence(
        {
            "experiment": spec.experiment,
            "recorder_id": recorder_id,
            "recorder_uri": spec.recorder_uri.as_posix(),
            "provider_uri": spec.provider_uri.as_posix(),
            "region": spec.region,
        }
    )
    metrics = _route_metrics_from_report(
        route=spec.route,
        report=recorder_evidence["report"],
        legacy_report=recorder_evidence["legacy_report"],
        selection_evidence=recorder_evidence["selection_evidence"],
        lineage=lineage,
    )

    model_selection_path = spec.output_dir / "model_selection.csv"
    result.model_comparison.to_csv(model_selection_path, index=False)
    selected_model_path = spec.output_dir / "selected_model.json"
    _write_json(
        selected_model_path,
        {
            "selected_candidate": result.selected_candidate,
            "selection_segment": "valid",
            "selection_frozen_before_test": metrics["selection_gates"]["frozen_policy"],
            "test_labels_used_for_selection": bool(
                result.selection_evidence.get("test_labels_used_for_selection")
            ),
            "floor": result.floor,
            "cap": result.cap,
        },
    )
    risk_path = spec.output_dir / "risk_analysis.csv"
    metrics["risk"].to_csv(risk_path, index=False)
    benchmark_path = spec.output_dir / "backtest_benchmark_comparison.csv"
    metrics["risk"].loc[metrics["risk"]["scope"].eq("test")].to_csv(benchmark_path, index=False)
    wf_path = spec.output_dir / "backtest_walk_forward_metrics.csv"
    metrics["walk_forward"].to_csv(wf_path, index=False)
    regime_path = spec.output_dir / "regime_metrics.csv"
    if spec.route == "regime_moe":
        pd.DataFrame(
            [
                {
                    "regime": name,
                    "train_occupancy": result.regime_occupancy[name],
                    "usable_observations": result.regime_usable_observations[name],
                }
                for name in result.regime_occupancy
            ]
        ).to_csv(regime_path, index=False)
    else:
        pd.DataFrame([{"regime": "all", "route_status": result.selection_evidence["route_status"]}]).to_csv(
            regime_path, index=False
        )
    window_path = spec.output_dir / "window_metrics.csv"
    if spec.route == "panel_sequence":
        result.model_comparison.to_csv(window_path, index=False)
    else:
        pd.DataFrame([{"window": "not_applicable", "route": spec.route}]).to_csv(window_path, index=False)

    blocked_modes = (
        {
            name: evidence.get("status", evidence.get("reason_code", "blocked"))
            for name, evidence in result.mode_evidence.items()
        }
        if spec.route == "panel_sequence"
        else {}
    )
    route_payload = {
        "route_status": str(result.selection_evidence.get("route_status", "blocked_unknown")),
        "selected_candidate": result.selected_candidate,
        "selection_frozen_before_test": metrics["selection_gates"]["frozen_policy"],
        "test_labels_used_for_selection": bool(
            result.selection_evidence.get("test_labels_used_for_selection")
        ),
        "selection_evidence": _compact_selection_evidence(result.selection_evidence),
        "data_lineage": lineage,
        "label_evidence": {
            "train_crash_threshold": labels.train_crash_threshold,
            "purged_boundary_rows": labels.purged_boundary_rows,
            "skipped_invalid_rows": labels.skipped_invalid_rows,
            "calendar_alignment": lineage["calendar_alignment"],
            "segment_alignment": lineage["segment_alignment"],
        },
        "metrics": _metric_payload(metrics),
        "benchmark": {
            "name": "legacy_soxl_backtest",
            "source": "soxl_backtest.reference.replay_soxlbacktest_reference",
            "matched_window": True,
            "start": metrics["legacy"].metrics["start"],
            "end": metrics["legacy"].metrics["end"],
        },
        "promotion": metrics["promotion"],
        "artifact_generation_api": ["SignalRecord.generate", "PortAnaRecord.generate"],
        "metrics_source": "qlib_portfolio_analysis/report_normal_1day.pkl",
        "execution_config": {
            "cost_rate": spec.cost_rate,
            "deal_price": "open",
            "timing": "signal_after_close_t_to_next_open",
        },
        "blocked_modes": blocked_modes,
        "recorder_artifacts": _recorder_files(recorder_local_dir),
    }
    sidecars = {
        "model_selection": model_selection_path.name,
        "selected_model": selected_model_path.name,
        "risk_analysis": risk_path.name,
        "benchmark_comparison": benchmark_path.name,
        "walk_forward_metrics": wf_path.name,
        "regime_metrics": regime_path.name,
        "window_metrics": window_path.name,
    }
    summary = build_card_summary(
        experiment=spec.experiment,
        route=spec.route,
        provider_uri=spec.provider_uri,
        recorder_uri=spec.recorder_uri,
        recorder_id=recorder_id,
        recorder_local_dir=recorder_local_dir,
        data_range=(spec.data_start, spec.as_of),
        segments=segments,
        route_payload=route_payload,
        sidecars=sidecars,
    )
    _write_json(spec.output_dir / "summary.json", summary)
    (spec.output_dir / "workflow.yaml").write_text(_workflow_text(spec, segments), encoding="utf-8")
    (spec.output_dir / "conclusion.md").write_text(
        notebook_conclusion_markdown(summary), encoding="utf-8"
    )
    return summary


def load_route_recorder_evidence(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Load the route's authoritative evidence through the qlib Recorder API."""

    import qlib
    from qlib.workflow import R

    recorder_id = str(summary.get("recorder_id", ""))
    experiment = str(summary.get("experiment", ""))
    if not recorder_id or not experiment:
        raise ValueError("recorder identity is incomplete")
    provider = Path(str(summary.get("provider_uri", "")))
    if not provider.is_absolute():
        provider = (Path.cwd() / provider).resolve()
    qlib.init(
        provider_uri=provider.as_posix(),
        region=str(summary.get("region", "us")),
        kernels=1,
    )
    R.set_uri(resolve_recorder_uri(str(summary.get("recorder_uri", ""))).as_posix())
    try:
        recorder = R.get_recorder(experiment_name=experiment, recorder_id=recorder_id)
    except Exception as exc:
        raise RuntimeError(f"recorder is unavailable: {experiment}/{recorder_id}") from exc
    if recorder.status != "FINISHED":
        raise RuntimeError(f"recorder is not finished: {experiment}/{recorder_id}")
    names = {
        "predictions": "pred.pkl",
        "labels": "label.pkl",
        "report": "portfolio_analysis/report_normal_1day.pkl",
        "positions": "portfolio_analysis/positions_normal_1day.pkl",
        "risk": "portfolio_analysis/port_analysis_1day.pkl",
        "selection_evidence": "evidence/selection_evidence.pkl",
        "legacy_report": "evidence/legacy_report.pkl",
    }
    objects: dict[str, Any] = {}
    for key, artifact in names.items():
        try:
            objects[key] = recorder.load_object(artifact)
        except Exception as exc:
            raise RuntimeError(f"recorder artifact is unavailable: {artifact}") from exc
    if not isinstance(objects["report"], pd.DataFrame) or objects["report"].empty:
        raise ValueError("recorder portfolio report is invalid")
    if not isinstance(objects["positions"], dict) or not objects["positions"]:
        raise ValueError("recorder positions are invalid")
    if not isinstance(objects["risk"], pd.DataFrame) or objects["risk"].empty:
        raise ValueError("recorder risk analysis is invalid")
    if not isinstance(objects["legacy_report"], pd.DataFrame) or objects[
        "legacy_report"
    ].empty:
        raise ValueError("recorder legacy report is invalid")
    if not isinstance(objects["selection_evidence"], Mapping):
        raise ValueError("recorder selection evidence is invalid")
    if not isinstance(objects["predictions"], (pd.Series, pd.DataFrame)) or objects[
        "predictions"
    ].empty:
        raise ValueError("recorder predictions are invalid")
    if not isinstance(objects["labels"], (pd.Series, pd.DataFrame)) or objects["labels"].empty:
        raise ValueError("recorder labels are invalid")
    return {
        "recorder": recorder,
        "recorder_status": recorder.status,
        "params": recorder.list_params(),
        **objects,
    }


def _metric_payload(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "test": {
            "candidate": _metrics_dict(metrics["candidate_test"]),
            "legacy": _metrics_dict(metrics["legacy_test"]),
        },
        "walk_forward": {
            "candidate": _metrics_dict(metrics["candidate_wf"].metrics),
            "legacy": _metrics_dict(metrics["legacy_wf"].metrics),
            "candidate_positive_folds": metrics["candidate_wf"].positive_return_folds,
            "fold_count": metrics["candidate_wf"].fold_count,
            "method": "frozen_policy_chronological_three_fold_diagnostic",
        },
    }


def _assert_metric_payload_matches(
    summary_metrics: Mapping[str, Any], recorder_metrics: Mapping[str, Any]
) -> None:
    for scope in ("test", "walk_forward"):
        for side in ("candidate", "legacy"):
            stated = summary_metrics.get(scope, {}).get(side, {})
            actual = recorder_metrics.get(scope, {}).get(side, {})
            if set(stated) != set(actual):
                raise ValueError("summary and recorder metrics differ")
            for field, value in actual.items():
                stated_value = stated[field]
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    if stated_value is None or not np.isclose(
                        float(stated_value), float(value), rtol=1e-10, atol=1e-12
                    ):
                        raise ValueError("summary and recorder metrics differ")
                elif stated_value != value:
                    raise ValueError("summary and recorder metrics differ")
    for field in ("candidate_positive_folds", "fold_count", "method"):
        if summary_metrics.get("walk_forward", {}).get(field) != recorder_metrics[
            "walk_forward"
        ][field]:
            raise ValueError("summary and recorder metrics differ")


def compare_frozen_route_summaries(
    summaries: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if set(summaries) != set(ROUTES):
        raise ValueError(f"frontier requires route summaries: {ROUTES}")
    windows = {tuple(summary.get("data_range", ())) for summary in summaries.values()}
    segments = {
        json.dumps(summary.get("segments", {}), sort_keys=True) for summary in summaries.values()
    }
    label_contracts = {
        json.dumps(summary.get("label_contract", {}), sort_keys=True) for summary in summaries.values()
    }
    if len(windows) != 1 or len(segments) != 1 or len(label_contracts) != 1:
        raise ValueError("frontier requires matched data windows and segments")
    frozen_routes: dict[str, Any] = {}
    promoted: list[str] = []
    for route, summary in summaries.items():
        benchmark = summary.get("benchmark", {})
        if benchmark.get("name") != "legacy_soxl_backtest" or benchmark.get("matched_window") is not True:
            raise ValueError(f"route lacks matched legacy benchmark evidence: {route}")
        evidence = load_route_recorder_evidence(summary)
        selection_evidence = evidence["selection_evidence"]
        if evidence["params"].get("route") != route:
            raise ValueError(f"recorder route identity differs for {route}")
        if selection_evidence.get("route_status") != summary.get("route_status"):
            raise ValueError("summary and recorder route status differ")
        if selection_evidence.get("test_labels_used_for_selection") is not False:
            raise ValueError(f"route used test labels for selection: {route}")
        metrics = _route_metrics_from_report(
            route=route,
            report=evidence["report"],
            legacy_report=evidence["legacy_report"],
            selection_evidence=selection_evidence,
            lineage=summary.get("data_lineage", {}),
        )
        recorder_metric_payload = _metric_payload(metrics)
        _assert_metric_payload_matches(summary.get("metrics", {}), recorder_metric_payload)
        if summary.get("promotion") != _jsonable(metrics["promotion"]):
            raise ValueError("summary and recorder promotion differ")
        gates = metrics["selection_gates"]
        if summary.get("selection_frozen_before_test") is not gates["frozen_policy"]:
            raise ValueError("summary and recorder frozen-policy evidence differ")
        if summary.get("test_labels_used_for_selection") is not False:
            raise ValueError(f"route used test labels for selection: {route}")
        promotion = metrics["promotion"]
        if promotion.get("promoted") is True:
            checks = promotion.get("checks")
            if not isinstance(checks, Mapping) or set(checks) != REQUIRED_PROMOTION_CHECKS:
                raise ValueError(f"promotion check evidence is incomplete for {route}")
            if not all(value is True for value in checks.values()):
                raise ValueError(f"promotion check evidence is not literal true for {route}")
            if summary.get("route_status") != "ok":
                raise ValueError(f"promoted route status is not ok: {route}")
        frozen_routes[route] = {
            "route_status": selection_evidence.get("route_status"),
            "selected_candidate": summary.get("selected_candidate"),
            "recorder_id": summary.get("recorder_id"),
            "recorder_status": evidence["recorder_status"],
            "metrics": recorder_metric_payload,
            "promotion": promotion,
        }
        if promotion.get("promoted") is True:
            promoted.append(route)
    decision = (
        "promoted_risk_adjusted_candidate" if promoted else "not_promoted_no_risk_adjusted_candidate"
    )
    return {
        "decision": decision,
        "promoted_routes": promoted,
        "selection_source": "frozen_route_summaries",
        "evidence_source": "qlib_recorder_objects",
        "test_reselection_performed": False,
        "routes": frozen_routes,
    }


def _frontier_readonly_dataset_audit(
    *, provider_uri: Path, segments: Mapping[str, Any]
) -> dict[str, Any]:
    from qlib.data import D
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP

    test = segments["test"]
    frame = D.features(
        ["SOXL"],
        ["$close", "$vix"],
        start_time=test[0],
        end_time=test[1],
        freq="day",
    )
    if frame.empty:
        raise ValueError("frontier canonical DatasetH audit returned no rows")
    handler_frame = frame.copy()
    handler_frame.columns = pd.MultiIndex.from_tuples(
        [("feature", str(column).removeprefix("$")) for column in handler_frame.columns]
    )
    dataset = DatasetH(
        handler=DataHandlerLP.from_df(handler_frame),
        segments={"audit": (test[0], test[1])},
    )
    prepared = dataset.prepare("audit", col_set="feature", data_key=DataHandlerLP.DK_I)
    if prepared.empty:
        raise ValueError("frontier DatasetH.prepare audit returned no rows")
    dates = prepared.index.get_level_values("datetime")
    return {
        "provider_uri": provider_uri.as_posix(),
        "source_api": ["qlib.data.D.features", "DatasetH", "DataHandlerLP", "DatasetH.prepare"],
        "DatasetH_prepare_rows": int(len(prepared)),
        "start": pd.Timestamp(dates.min()).date().isoformat(),
        "end": pd.Timestamp(dates.max()).date().isoformat(),
        "columns": [str(column) for column in prepared.columns],
        "training_performed": False,
        "selection_performed": False,
    }


def run_risk_adjusted_frontier_card(
    *,
    output_dir: Path,
    route_card_dirs: Mapping[str, Path],
    provider_uri: Path = Path("data/qlib/soxl"),
    recorder_uri: Path = Path(CANONICAL_RECORDER_URI),
) -> dict[str, Any]:
    import qlib
    from qlib.data import D
    from qlib.workflow import R

    summaries = {
        route: json.loads((path / "summary.json").read_text(encoding="utf-8"))
        for route, path in route_card_dirs.items()
    }
    comparison = compare_frozen_route_summaries(summaries)
    qlib.init(provider_uri=provider_uri.resolve().as_posix(), region="us", kernels=1)
    calendar = D.calendar(freq="day")
    if not len(calendar):
        raise ValueError("blocked_missing_required_source: empty qlib calendar")
    canonical_audit = _frontier_readonly_dataset_audit(
        provider_uri=provider_uri,
        segments=next(iter(summaries.values()))["segments"],
    )
    R.set_uri(resolve_recorder_uri(recorder_uri).as_posix())
    with R.start(experiment_name="soxl_risk_adjusted_model_frontier_v1"):
        recorder = R.get_recorder()
        R.log_params(
            selection_source="qlib_recorder_objects",
            test_reselection_performed=False,
            route_count=3,
            decision=comparison["decision"],
        )
        R.save_objects(
            artifact_path="frontier",
            **{
                "route_summaries.pkl": summaries,
                "canonical_dataset_audit.pkl": canonical_audit,
            },
        )
        recorder_id = recorder.id
        recorder_local_dir = Path(recorder.get_local_dir())
    _remove_transient_recorder_files(recorder_local_dir)
    persisted = R.get_recorder(
        experiment_name="soxl_risk_adjusted_model_frontier_v1", recorder_id=recorder_id
    )
    if persisted.status != "FINISHED":
        raise RuntimeError(f"frontier recorder write did not finish: {recorder_id}")
    for artifact in (
        "frontier/route_summaries.pkl",
        "frontier/canonical_dataset_audit.pkl",
    ):
        try:
            persisted.load_object(artifact)
        except Exception as exc:
            raise RuntimeError(f"frontier recorder write is incomplete: {artifact}") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    benchmark_rows = []
    for route, summary in summaries.items():
        recorder_route = comparison["routes"][route]
        metric = recorder_route["metrics"]["test"]["candidate"]
        legacy_metric = recorder_route["metrics"]["test"]["legacy"]
        rows.append(
            {
                "route": route,
                "route_status": summary["route_status"],
                "selected_candidate": summary["selected_candidate"],
                "total_return": metric["total_return"],
                "max_drawdown": metric["max_drawdown"],
                "return_to_drawdown": metric["return_to_drawdown"],
                "avg_abs_position": metric["avg_abs_position"],
                "promoted": recorder_route["promotion"]["promoted"],
                "failed_checks": "|".join(recorder_route["promotion"]["failed_checks"]),
            }
        )
        benchmark_rows.append(
            {
                "route": route,
                "candidate_return": metric["total_return"],
                "legacy_return": legacy_metric["total_return"],
                "return_gap": metric["total_return"] - legacy_metric["total_return"],
                "candidate_max_drawdown": metric["max_drawdown"],
                "legacy_max_drawdown": legacy_metric["max_drawdown"],
                "candidate_total_turnover": metric["total_turnover"],
                "legacy_total_turnover": legacy_metric["total_turnover"],
            }
        )
    model_selection = pd.DataFrame(rows)
    model_selection.to_csv(output_dir / "model_selection.csv", index=False)
    pd.DataFrame(benchmark_rows).to_csv(
        output_dir / "backtest_benchmark_comparison.csv", index=False
    )
    _write_json(output_dir / "selected_model.json", comparison)
    summary = {
        "experiment": "soxl_risk_adjusted_model_frontier_v1",
        "route": "frontier",
        "route_status": "ok",
        "selected_candidate": comparison["decision"],
        "selection_frozen_before_test": True,
        "test_labels_used_for_selection": False,
        "provider_uri": provider_uri.as_posix(),
        "recorder_uri": str(recorder_uri),
        "recorder_id": recorder_id,
        "recorder_artifacts": _recorder_files(recorder_local_dir),
        "market": "all",
        "region": "us",
        "as_of": next(iter(summaries.values()))["as_of"],
        "data_range": next(iter(summaries.values()))["data_range"],
        "segments": next(iter(summaries.values()))["segments"],
        "label_contract": next(iter(summaries.values()))["label_contract"],
        "frontier": comparison,
        "canonical_dataset_audit": canonical_audit,
        "promotion": {
            "promoted": bool(comparison["promoted_routes"]),
            "decision": comparison["decision"],
            "failed_checks": [] if comparison["promoted_routes"] else ["no_route_passed_all_gates"],
        },
        "routes": {route: summary for route, summary in comparison["routes"].items()},
        "route_recorder_artifacts": {
            route: {
                "recorder_id": route_summary["recorder_id"],
                "recorder_uri": route_summary["recorder_uri"],
                "experiment": route_summary["experiment"],
                "artifacts": {
                    key: value for key, value in RECORDER_ONLY_ARTIFACTS.items()
                },
            }
            for route, route_summary in summaries.items()
        },
        "qlib_native_api": [
            "qlib.init",
            "qlib.data.D.calendar",
            "DatasetH",
            "DataHandlerLP",
            "qlib.workflow.R.start",
        ],
        "qlib_docs": QLIB_DOCS,
        "sidecars": {
            "model_selection": "model_selection.csv",
            "selected_model": "selected_model.json",
            "benchmark_comparison": "backtest_benchmark_comparison.csv",
        },
        "artifact_policy": {
            "policy": "slim_card_recorder_first",
            "primary_artifact": "soxl_risk_adjusted_model_frontier_v1.ipynb",
            "workflow_config": "workflow.yaml",
            "machine_entrypoint": "summary.json",
            "kept_sidecars": [
                "model_selection.csv",
                "selected_model.json",
                "backtest_benchmark_comparison.csv",
            ],
            "recorder_only_artifacts": dict(FRONTIER_RECORDER_ONLY_ARTIFACTS),
        },
    }
    _write_json(output_dir / "summary.json", summary)
    (output_dir / "workflow.yaml").write_text(
        """experiment: soxl_risk_adjusted_model_frontier_v1
route: frontier
qlib:
  provider_uri: data/qlib/soxl
  recorder_uri: git-common-parent:.qlib-recorders/soxl-risk-adjusted-frontier-v1
  market: all
  region: us
segments:
  train: inherited_matched_route_window
  valid: inherited_matched_route_window
  test: inherited_matched_route_window
model_selection:
  source: frozen_route_summaries
  test_reselection: false
artifact_policy:
  policy: slim_card_recorder_first
  machine_entrypoint: summary.json
  workflow_config: workflow.yaml
""",
        encoding="utf-8",
    )
    (output_dir / "conclusion.md").write_text(
        notebook_conclusion_markdown(summary), encoding="utf-8"
    )
    return summary


__all__ = [
    "FRONTIER_RECORDER_ONLY_ARTIFACTS",
    "RECORDER_ONLY_ARTIFACTS",
    "REQUIRED_PROMOTION_CHECKS",
    "RiskAdjustedRouteCardSpec",
    "build_calendar_alignment_evidence",
    "build_card_summary",
    "compare_frozen_route_summaries",
    "load_route_recorder_evidence",
    "notebook_conclusion_markdown",
    "resolve_recorder_uri",
    "run_risk_adjusted_frontier_card",
    "run_risk_adjusted_route_card",
    "sync_notebook_conclusion",
]
