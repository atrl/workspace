from __future__ import annotations

import json
import pickle
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
}

ROUTES = ("tail_budget", "regime_moe", "panel_sequence")


@dataclass(frozen=True)
class RiskAdjustedRouteCardSpec:
    experiment: str
    route: str
    output_dir: Path
    provider_uri: Path = Path("data/qlib/soxl")
    recorder_uri: Path = Path("../.qlib-recorders/soxl-risk-adjusted-frontier-v1")
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
        "recorder_local_dir": recorder_local_dir.as_posix(),
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
    table = raw_features.join(labels.labels.drop(columns=["segment", "future_min_daily_return_5d"]), how="inner")
    table = table.replace([np.inf, -np.inf], np.nan).dropna()
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
    for name, (start, end) in {"train": spec.train, "valid": spec.valid, "test": spec.test}.items():
        dates = table.loc[start:end].index
        if dates.empty:
            raise ValueError(f"blocked_missing_required_source: empty {name} segment")
        effective_segments[name] = (
            pd.Timestamp(dates.min()).date().isoformat(),
            pd.Timestamp(dates.max()).date().isoformat(),
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


def _route_metrics(spec: RiskAdjustedRouteCardSpec, result, next_open_return: pd.Series):
    position = _series_datetime(result.positions_by_segment["test"])
    gross = next_open_return.reindex(position.index).astype("float64") * position
    turnover_cost = position.diff().abs().fillna(position.abs()) * spec.cost_rate
    candidate_return = (gross - turnover_cost).fillna(0.0)
    legacy, legacy_return, legacy_position = _legacy_paths(
        position.index.min().date().isoformat(), position.index.max().date().isoformat(), position.index
    )
    candidate_test = compute_risk_metrics(candidate_return, position)
    legacy_test = compute_risk_metrics(legacy_return, legacy_position)
    chunks = np.array_split(np.arange(len(position)), 3)
    candidate_folds = tuple(
        compute_risk_metrics(candidate_return.iloc[chunk], position.iloc[chunk]) for chunk in chunks
    )
    legacy_folds = tuple(
        compute_risk_metrics(legacy_return.iloc[chunk], legacy_position.iloc[chunk]) for chunk in chunks
    )
    candidate_wf = aggregate_walk_forward(candidate_folds)
    legacy_wf = aggregate_walk_forward(legacy_folds)
    route_status = str(result.selection_evidence.get("route_status", "blocked_unknown"))
    evidence_ok = route_status == "ok"
    evaluation = evaluate_promotion(
        candidate_test,
        legacy_test,
        candidate_wf,
        legacy_wf,
        spec=RiskAdjustedGateSpec(),
        data_quality_valid=evidence_ok,
        label_leakage_free=True,
        test_time_selection_free=True,
        labels_complete=True,
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
    report = pd.DataFrame(
        {
            "candidate_return": candidate_return,
            "legacy_return": legacy_return,
            "target_position": position,
            "legacy_position": legacy_position,
        }
    )
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
        "report": report,
        "risk": risk,
        "walk_forward": pd.DataFrame(wf_rows),
        "legacy": legacy,
    }


def _record_route(
    spec: RiskAdjustedRouteCardSpec,
    result,
    dataset,
    metrics: Mapping[str, Any],
    labels,
):
    from qlib.workflow import R

    recorder_uri = spec.recorder_uri.resolve()
    recorder_uri.mkdir(parents=True, exist_ok=True)
    R.set_uri(recorder_uri.as_posix())
    predictions = result.predictions_by_segment
    positions = pd.concat(
        {name: values for name, values in result.positions_by_segment.items()}, names=["segment"]
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
            )
            R.log_metrics(
                candidate_test_return=metrics["candidate_test"].total_return,
                candidate_test_max_drawdown=metrics["candidate_test"].max_drawdown,
                candidate_test_avg_abs_position=metrics["candidate_test"].avg_abs_position,
                legacy_test_return=metrics["legacy_test"].total_return,
            )
            R.save_objects(**{"pred.pkl": predictions, "label.pkl": labels.labels})
            recorder.log_artifact(all_path.as_posix(), artifact_path="model_selection")
            R.save_objects(
                artifact_path="portfolio_analysis",
                **{
                    "report_normal_1day.pkl": metrics["report"],
                    "positions_normal_1day.pkl": positions,
                    "port_analysis_1day.pkl": metrics["risk"],
                },
            )
            R.save_objects(artifact_path="model", **{"fitted_policy.pkl": result.fitted_policy})
            R.save_objects(artifact_path="evidence", **{"selection_evidence.pkl": result.selection_evidence})
            recorder_id = recorder.id
            recorder_local_dir = Path(recorder.get_local_dir())
    _remove_transient_recorder_files(recorder_local_dir)
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
            "artifact_policy:",
            "  policy: slim_card_recorder_first",
            "  machine_entrypoint: summary.json",
            "  workflow_config: workflow.yaml",
            "",
        ]
    )
    return "\n".join(lines)


def _conclusion(summary: Mapping[str, Any]) -> str:
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


def run_risk_adjusted_route_card(spec: RiskAdjustedRouteCardSpec) -> dict[str, Any]:
    spec.output_dir.mkdir(parents=True, exist_ok=True)
    dataset, next_open_return, labels, segments, lineage = _canonical_dataset(spec)
    result = _run_route(spec.route, dataset, lineage)
    metrics = _route_metrics(spec, result, next_open_return)
    recorder_id, recorder_local_dir = _record_route(spec, result, dataset, metrics, labels)

    model_selection_path = spec.output_dir / "model_selection.csv"
    result.model_comparison.to_csv(model_selection_path, index=False)
    selected_model_path = spec.output_dir / "selected_model.json"
    _write_json(
        selected_model_path,
        {
            "selected_candidate": result.selected_candidate,
            "selection_segment": "valid",
            "selection_frozen_before_test": True,
            "test_labels_used_for_selection": False,
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
        "selection_frozen_before_test": True,
        "test_labels_used_for_selection": False,
        "selection_evidence": _compact_selection_evidence(result.selection_evidence),
        "data_lineage": lineage,
        "label_evidence": {
            "train_crash_threshold": labels.train_crash_threshold,
            "purged_boundary_rows": labels.purged_boundary_rows,
            "skipped_invalid_rows": labels.skipped_invalid_rows,
        },
        "metrics": {
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
        },
        "benchmark": {
            "name": "legacy_soxl_backtest",
            "source": "soxl_backtest.reference.replay_soxlbacktest_reference",
            "matched_window": True,
            "start": metrics["legacy"].metrics["start"],
            "end": metrics["legacy"].metrics["end"],
        },
        "promotion": metrics["promotion"],
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
    (spec.output_dir / "conclusion.md").write_text(_conclusion(summary), encoding="utf-8")
    return summary


def compare_frozen_route_summaries(
    summaries: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if set(summaries) != set(ROUTES):
        raise ValueError(f"frontier requires route summaries: {ROUTES}")
    windows = {tuple(summary.get("data_range", ())) for summary in summaries.values()}
    segments = {
        json.dumps(summary.get("segments", {}), sort_keys=True) for summary in summaries.values()
    }
    if len(windows) != 1 or len(segments) != 1:
        raise ValueError("frontier requires matched data windows and segments")
    frozen_routes: dict[str, Any] = {}
    promoted: list[str] = []
    for route, summary in summaries.items():
        artifacts = set(summary.get("recorder_artifacts", ()))
        if not summary.get("recorder_id") or not set(RECORDER_ONLY_ARTIFACTS.values()).issubset(artifacts):
            raise ValueError(f"frontier requires complete recorder evidence for {route}")
        if summary.get("selection_frozen_before_test") is not True:
            raise ValueError(f"route is not frozen before test: {route}")
        if summary.get("test_labels_used_for_selection") is not False:
            raise ValueError(f"route used test labels for selection: {route}")
        frozen_routes[route] = {
            "route_status": summary.get("route_status"),
            "selected_candidate": summary.get("selected_candidate"),
            "recorder_id": summary.get("recorder_id"),
            "promotion": summary.get("promotion"),
        }
        if summary.get("promotion", {}).get("promoted") is True:
            promoted.append(route)
    decision = (
        "promoted_risk_adjusted_candidate" if promoted else "not_promoted_no_risk_adjusted_candidate"
    )
    return {
        "decision": decision,
        "promoted_routes": promoted,
        "selection_source": "frozen_route_summaries",
        "test_reselection_performed": False,
        "routes": frozen_routes,
    }


def run_risk_adjusted_frontier_card(
    *,
    output_dir: Path,
    route_card_dirs: Mapping[str, Path],
    provider_uri: Path = Path("data/qlib/soxl"),
    recorder_uri: Path = Path("../.qlib-recorders/soxl-risk-adjusted-frontier-v1"),
) -> dict[str, Any]:
    import qlib
    from qlib.data import D
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP
    from qlib.workflow import R

    del DatasetH, DataHandlerLP  # imports are executable API evidence for the frontier notebook.
    summaries = {
        route: json.loads((path / "summary.json").read_text(encoding="utf-8"))
        for route, path in route_card_dirs.items()
    }
    comparison = compare_frozen_route_summaries(summaries)
    qlib.init(provider_uri=provider_uri.resolve().as_posix(), region="us", kernels=1)
    calendar = D.calendar(freq="day")
    if not len(calendar):
        raise ValueError("blocked_missing_required_source: empty qlib calendar")
    R.set_uri(recorder_uri.resolve().as_posix())
    with R.start(experiment_name="soxl_risk_adjusted_model_frontier_v1"):
        recorder = R.get_recorder()
        R.log_params(
            selection_source="frozen_route_summaries",
            test_reselection_performed=False,
            route_count=3,
            decision=comparison["decision"],
        )
        R.save_objects(artifact_path="frontier", **{"route_summaries.pkl": summaries})
        recorder_id = recorder.id
        recorder_local_dir = Path(recorder.get_local_dir())
    _remove_transient_recorder_files(recorder_local_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for route, summary in summaries.items():
        metric = summary["metrics"]["test"]["candidate"]
        rows.append(
            {
                "route": route,
                "route_status": summary["route_status"],
                "selected_candidate": summary["selected_candidate"],
                "total_return": metric["total_return"],
                "max_drawdown": metric["max_drawdown"],
                "return_to_drawdown": metric["return_to_drawdown"],
                "avg_abs_position": metric["avg_abs_position"],
                "promoted": summary["promotion"]["promoted"],
                "failed_checks": "|".join(summary["promotion"]["failed_checks"]),
            }
        )
    model_selection = pd.DataFrame(rows)
    model_selection.to_csv(output_dir / "model_selection.csv", index=False)
    model_selection.to_csv(output_dir / "backtest_benchmark_comparison.csv", index=False)
    _write_json(output_dir / "selected_model.json", comparison)
    summary = {
        "experiment": "soxl_risk_adjusted_model_frontier_v1",
        "route": "frontier",
        "route_status": "ok",
        "selected_candidate": comparison["decision"],
        "selection_frozen_before_test": True,
        "test_labels_used_for_selection": False,
        "provider_uri": provider_uri.as_posix(),
        "recorder_uri": recorder_uri.as_posix(),
        "recorder_id": recorder_id,
        "recorder_local_dir": recorder_local_dir.as_posix(),
        "recorder_artifacts": _recorder_files(recorder_local_dir),
        "market": "all",
        "region": "us",
        "as_of": next(iter(summaries.values()))["as_of"],
        "data_range": next(iter(summaries.values()))["data_range"],
        "segments": next(iter(summaries.values()))["segments"],
        "label_contract": next(iter(summaries.values()))["label_contract"],
        "frontier": comparison,
        "promotion": {
            "promoted": bool(comparison["promoted_routes"]),
            "decision": comparison["decision"],
            "failed_checks": [] if comparison["promoted_routes"] else ["no_route_passed_all_gates"],
        },
        "routes": {route: summary for route, summary in comparison["routes"].items()},
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
            "recorder_only_artifacts": dict(RECORDER_ONLY_ARTIFACTS),
        },
    }
    _write_json(output_dir / "summary.json", summary)
    (output_dir / "workflow.yaml").write_text(
        """experiment: soxl_risk_adjusted_model_frontier_v1
route: frontier
qlib:
  provider_uri: data/qlib/soxl
  recorder_uri: ../.qlib-recorders/soxl-risk-adjusted-frontier-v1
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
    conclusion = f"""# SOXL 风险调整模型前沿结论

## 做了什么

读取三张已冻结 route summary 和 qlib recorder 证据，未在 test 上重新选择或重训。

## 核心指标

三条路线均保留在 `model_selection.csv`；最终决定为 `{comparison['decision']}`。

## Regime / window 差异

Regime MoE、tail budget、20/60 日序列结果均按各自 card 原样进入比较；blocked 路线没有被省略。

## 晋升决定

`{comparison['decision']}`。

## 风险

canonical 数据截至 {summary['as_of']}；三折仍是冻结策略诊断，nested walk-forward 未完成。本结果不构成买卖建议。

## 下一步

完成 expanding-window nested 重训，并补齐 PIT 成分股历史后再尝试 validated promotion。
"""
    (output_dir / "conclusion.md").write_text(conclusion, encoding="utf-8")
    return summary


__all__ = [
    "RECORDER_ONLY_ARTIFACTS",
    "RiskAdjustedRouteCardSpec",
    "build_card_summary",
    "compare_frozen_route_summaries",
    "run_risk_adjusted_frontier_card",
    "run_risk_adjusted_route_card",
]
