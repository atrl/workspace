from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.analysis import build_analysis_report_from_qlib_recorder
from src.backtest import (
    build_backtest_extension_report_from_qlib_recorder,
    build_portfolio_report_from_qlib_recorder,
    calibrated_signal_threshold,
)
from src.data import QlibProviderConfig, build_alpha158_legacy_dataset, init_qlib_provider, load_qlib_feature_frame
from src.model import quantbench_supervised_model_specs, run_qlib_model_selection


QLIB_DOCS = {
    "data": "https://qlib.readthedocs.io/en/latest/component/data.html",
    "recorder": "https://qlib.readthedocs.io/en/latest/component/recorder.html",
    "report": "https://qlib.readthedocs.io/en/latest/component/report.html",
}


@dataclass(frozen=True)
class QlibNativeExperimentSpec:
    experiment: str
    provider_uri: Path
    recorder_uri: Path
    output_dir: Path
    market: str
    region: str
    instrument: str
    train: tuple[str, str]
    test: tuple[str, str]
    valid: tuple[str, str] | None = None
    model_selection_metric: str = "utility_score"
    fixed_model_name: str | None = None
    promoted_from: str | None = None


@dataclass(frozen=True)
class QlibNativeExperimentResult:
    recorder_id: str
    recorder_local_dir: Path
    summary: dict[str, Any]
    analysis_summary: dict[str, Any]


class LastReturnSignalModel:
    def fit(self, dataset) -> "LastReturnSignalModel":
        self.train_rows = int(len(dataset.prepare("train")))
        return self

    def predict(self, dataset) -> pd.Series:
        frame = dataset.prepare("test")
        return frame["$close"].pct_change().fillna(0.0).rename("score")


def _recorder_files(recorder_dir: Path) -> list[str]:
    return sorted(path.relative_to(recorder_dir).as_posix() for path in recorder_dir.rglob("*") if path.is_file())


def _remove_transient_code_snapshots(recorder_dir: Path) -> None:
    for name in ["code_cached.txt", "code_diff.txt", "code_status.txt"]:
        path = recorder_dir / "artifacts" / name
        if path.exists():
            path.unlink()


def _write_dataframe_sidecars(output_dir: Path, recorder_dir: Path) -> dict[str, str]:
    pred = pd.read_pickle(recorder_dir / "artifacts" / "pred.pkl")
    label = pd.read_pickle(recorder_dir / "artifacts" / "label.pkl")
    pred_path = output_dir / "predictions.csv"
    signal_path = output_dir / "signals.csv"
    pred.reset_index().to_csv(pred_path, index=False)
    signals = pred.join(label, how="left")
    signals.reset_index().to_csv(signal_path, index=False)
    return {"predictions": pred_path.name, "signals": signal_path.name}


def _target_position_strategy_summary(*, score_threshold: float) -> dict[str, Any]:
    return {
        "class": "SignalWeightStrategy",
        "module_path": "src.backtest.strategy",
        "signal_source": "<PRED>",
        "threshold_source": "train_prediction_median",
        "score_threshold": float(score_threshold),
        "min_weight": 0.0,
        "max_weight": 1.0,
        "risk_degree": 0.95,
    }


def _port_analysis_config(spec: QlibNativeExperimentSpec, *, score_threshold: float) -> dict[str, Any]:
    return {
        "strategy": {
            "class": "SignalWeightStrategy",
            "module_path": "src.backtest.strategy",
            "kwargs": {
                "signal": "<PRED>",
                "instrument": spec.instrument,
                "score_threshold": float(score_threshold),
                "min_weight": 0.0,
                "max_weight": 1.0,
                "risk_degree": 0.95,
            },
        },
        "backtest": {
            "start_time": spec.test[0],
            "end_time": spec.test[1],
            "account": 100000.0,
            "benchmark": spec.instrument,
            "exchange_kwargs": {
                "freq": "day",
                "limit_threshold": None,
                "deal_price": "close",
                "open_cost": 0.0002,
                "close_cost": 0.0002,
                "min_cost": 0.0,
            },
        },
    }


def _selection_segment(spec: QlibNativeExperimentSpec) -> str:
    return "valid" if spec.valid is not None else "test"


def _model_specs_for_experiment(spec: QlibNativeExperimentSpec):
    specs = quantbench_supervised_model_specs()
    if spec.fixed_model_name is None:
        return specs
    fixed = [model_spec for model_spec in specs if model_spec.name == spec.fixed_model_name]
    if not fixed:
        raise ValueError(f"unknown fixed qlib model spec: {spec.fixed_model_name}")
    return fixed


def _write_qlib_portfolio_sidecars(output_dir: Path, recorder_dir: Path) -> dict[str, str]:
    portfolio_dir = recorder_dir / "artifacts" / "portfolio_analysis"
    report_path = portfolio_dir / "report_normal_1day.pkl"
    positions_path = portfolio_dir / "positions_normal_1day.pkl"
    risk_path = portfolio_dir / "port_analysis_1day.pkl"
    missing = [path.as_posix() for path in [report_path, positions_path, risk_path] if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing qlib PortAnaRecord artifacts: {missing}")

    report = pd.read_pickle(report_path)
    positions = pd.read_pickle(positions_path)
    risk = pd.read_pickle(risk_path)
    report_csv = output_dir / "qlib_portfolio_report.csv"
    positions_csv = output_dir / "qlib_positions.csv"
    risk_csv = output_dir / "risk_analysis.csv"
    report.reset_index().to_csv(report_csv, index=False)
    if isinstance(positions, dict):
        pd.DataFrame(
            [
                {"datetime": str(date), "positions": json.dumps(value, sort_keys=True, default=str)}
                for date, value in positions.items()
            ]
        ).to_csv(positions_csv, index=False)
    else:
        positions.reset_index().to_csv(positions_csv, index=False)
    risk.reset_index().to_csv(risk_csv, index=False)
    return {
        "qlib_portfolio_report": report_csv.name,
        "qlib_positions": positions_csv.name,
        "risk_analysis": risk_csv.name,
    }


def _legacy_soxlbacktest_benchmark(spec: QlibNativeExperimentSpec, *, strategy_return_pct: float) -> dict[str, Any]:
    from soxl_backtest.reference import replay_soxlbacktest_reference

    legacy = replay_soxlbacktest_reference(start=spec.test[0], end=spec.test[1])
    legacy_return = float(legacy.metrics["rulebase_return_pct"])
    return {
        "benchmark_name": "legacy_soxl_backtest",
        "benchmark_source": "soxl_backtest.reference.replay_soxlbacktest_reference",
        "start": legacy.metrics["start"],
        "end": legacy.metrics["end"],
        "rows": int(legacy.metrics["rows"]),
        "strategy_return_pct": float(strategy_return_pct),
        "legacy_rulebase_return_pct": legacy_return,
        "legacy_rulebase_max_drawdown_pct": float(legacy.metrics["rulebase_max_drawdown_pct"]),
        "legacy_trade_count": int(legacy.metrics["trade_count"]),
        "excess_return_pct": round(float(strategy_return_pct - legacy_return), 10),
        "exceeds_legacy_rulebase": bool(strategy_return_pct > legacy_return),
    }


def run_qlib_native_experiment(spec: QlibNativeExperimentSpec) -> QlibNativeExperimentResult:
    from qlib.workflow import R
    from qlib.workflow.record_temp import PortAnaRecord, SigAnaRecord, SignalRecord

    config = QlibProviderConfig(provider_uri=spec.provider_uri, region=spec.region, market=spec.market)
    provider_meta = init_qlib_provider(config)
    dataset = build_alpha158_legacy_dataset(
        config,
        instruments=[spec.instrument],
        label="Ref($close, -1) / $close - 1",
        train=spec.train,
        valid=spec.valid,
        test=spec.test,
    )
    feature_frame = load_qlib_feature_frame(
        config,
        instruments=[spec.instrument],
        fields=["$close", "$vix", "$volume"],
        start=spec.train[0],
        end=spec.test[1],
    )
    spec.output_dir.mkdir(parents=True, exist_ok=True)
    spec.recorder_uri.mkdir(parents=True, exist_ok=True)
    model_selection = run_qlib_model_selection(
        dataset,
        specs=_model_specs_for_experiment(spec),
        output_dir=spec.output_dir,
        selection_metric=spec.model_selection_metric,
        selection_segment=_selection_segment(spec),
        report_segment="test",
    )
    model = model_selection.selected_model
    score_threshold = calibrated_signal_threshold(model, dataset=dataset, segment="train")
    target_position_strategy = _target_position_strategy_summary(score_threshold=score_threshold)
    R.set_uri(spec.recorder_uri.as_posix())

    with R.start(experiment_name=spec.experiment):
        recorder = R.get_recorder()
        R.log_params(
            provider_uri=provider_meta["provider_uri"],
            market=spec.market,
            region=spec.region,
            instrument=spec.instrument,
            train_start=spec.train[0],
            train_end=spec.train[1],
            valid_start=spec.valid[0] if spec.valid else "",
            valid_end=spec.valid[1] if spec.valid else "",
            test_start=spec.test[0],
            test_end=spec.test[1],
            selected_model=model_selection.selected_spec.name,
            selected_model_implementation=model.implementation,
            model_selection_metric=spec.model_selection_metric,
            model_selection_mode="fixed_model" if spec.fixed_model_name else "model_search",
            fixed_model_name=spec.fixed_model_name or "",
            promoted_from=spec.promoted_from or "",
            model_selection_segment=_selection_segment(spec),
            model_report_segment="test",
            feature_set_source=dataset.qlib_feature_set["source"],
            feature_count=dataset.qlib_feature_set["feature_count"],
            target_position_strategy=target_position_strategy["class"],
            signal_threshold=score_threshold,
            signal_threshold_source=target_position_strategy["threshold_source"],
            max_weight=target_position_strategy["max_weight"],
            risk_degree=target_position_strategy["risk_degree"],
        )
        for artifact in [
            spec.output_dir / model_selection.sidecars["model_selection"],
            spec.output_dir / model_selection.sidecars["all_model_predictions"],
            model_selection.selected_model_path,
        ]:
            recorder.log_artifact(artifact.as_posix(), artifact_path="model_selection")
        SignalRecord(model, dataset, recorder).generate()
        SigAnaRecord(recorder).generate()
        PortAnaRecord(recorder, _port_analysis_config(spec, score_threshold=score_threshold), risk_analysis_freq="day").generate()
        recorder_id = recorder.id
        recorder_local_dir = Path(recorder.get_local_dir())

    _remove_transient_code_snapshots(recorder_local_dir)
    sidecars = _write_dataframe_sidecars(spec.output_dir, recorder_local_dir)
    qlib_portfolio_sidecars = _write_qlib_portfolio_sidecars(spec.output_dir, recorder_local_dir)
    summary = {
        "experiment": spec.experiment,
        "provider_uri": provider_meta["provider_uri"],
        "market": spec.market,
        "region": spec.region,
        "as_of": spec.test[1],
        "data_range": [spec.train[0], spec.test[1]],
        "segments": {
            "train": list(spec.train),
            **({"valid": list(spec.valid)} if spec.valid else {}),
            "test": list(spec.test),
        },
        "instrument": spec.instrument,
        "experiment_id": recorder_local_dir.parent.name,
        "recorder_id": recorder_id,
        "recorder_uri": spec.recorder_uri.resolve().as_posix(),
        "recorder_local_dir": recorder_local_dir.as_posix(),
        "recorder_artifacts": _recorder_files(recorder_local_dir),
        "qlib_native_api": [
            "qlib.init",
            "qlib.data.D.features",
            "DatasetH",
            "DataHandlerLP",
            "Alpha158.get_feature_config",
            "qlib.workflow.R.start",
            "WeightStrategyBase",
            "SignalRecord",
            "SigAnaRecord",
            "PortAnaRecord",
        ],
        "qlib_record_classes_available": ["SignalRecord", "SigAnaRecord", "PortAnaRecord"],
        "qlib_docs": QLIB_DOCS,
        "model_selection": {
            "mode": "fixed_model" if spec.fixed_model_name else "model_search",
            "selected_model": model_selection.selected_spec.name,
            "selected_implementation": model.implementation,
            "selection_metric": spec.model_selection_metric,
            "selection_segment": _selection_segment(spec),
            "report_segment": "test",
            "selection_column": f"selection_{spec.model_selection_metric}",
            "report_column": f"test_{spec.model_selection_metric}",
            "candidate_count": int(len(model_selection.model_metrics)),
            "source": "qlib_dataset",
            **({"fixed_model_name": spec.fixed_model_name} if spec.fixed_model_name else {}),
            **({"promoted_from": spec.promoted_from} if spec.promoted_from else {}),
        },
        "feature_set": dataset.qlib_feature_set,
        "target_position_strategy": target_position_strategy,
        "sidecars": sidecars,
    }
    if spec.fixed_model_name:
        summary["production_fixed_parameters"] = {
            "fixed_model_name": spec.fixed_model_name,
            "model_selection_metric": spec.model_selection_metric,
            "selection_segment": _selection_segment(spec),
            "report_segment": "test",
            "promoted_from": spec.promoted_from,
        }
    portfolio = build_portfolio_report_from_qlib_recorder(summary, cost_bps=2.0, slippage_bps=1.0)
    portfolio_path = spec.output_dir / "portfolio_report.csv"
    portfolio.report.to_csv(portfolio_path, index=False)
    backtest_extension = build_backtest_extension_report_from_qlib_recorder(
        summary,
        portfolio.report,
        windows={"test": spec.test, "train_to_test": (spec.train[0], spec.test[1])},
        stress_windows={"test_tail": (spec.test[0], spec.test[1])},
        walk_forward={"train_size": 1, "test_size": 2},
        cost_bps=2.0,
        slippage_bps=1.0,
    )
    backtest_paths = backtest_extension.write_sidecars(spec.output_dir)
    strategy_return_pct = float(backtest_extension.benchmark_comparison.iloc[0]["strategy_return_pct"])
    summary["legacy_soxlbacktest_benchmark"] = _legacy_soxlbacktest_benchmark(
        spec,
        strategy_return_pct=strategy_return_pct,
    )
    report = build_analysis_report_from_qlib_recorder(
        summary,
        portfolio.report,
        windows={"test": spec.test, "train_to_test": (spec.train[0], spec.test[1])},
    )
    analysis_paths = report.write_sidecars(spec.output_dir)
    analysis_summary = report.summary | {
        "rows": int(len(feature_frame)),
        "qlib_record_classes": ["SignalRecord", "SigAnaRecord", "PortAnaRecord"],
        "qlib_record_classes_available": ["SignalRecord", "SigAnaRecord", "PortAnaRecord"],
    }
    (spec.output_dir / "analysis_summary.json").write_text(
        json.dumps(analysis_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    summary["sidecars"] = sidecars | qlib_portfolio_sidecars | {
        **model_selection.sidecars,
        "portfolio_report": portfolio_path.name,
        "backtest_window_metrics": backtest_paths["backtest_window_metrics"].name,
        "backtest_walk_forward_metrics": backtest_paths["backtest_walk_forward_metrics"].name,
        "backtest_benchmark_comparison": backtest_paths["backtest_benchmark_comparison"].name,
        "analysis_summary": analysis_paths["summary"].name,
        "regime_metrics": analysis_paths["regime_metrics"].name,
        "window_metrics": analysis_paths["window_metrics"].name,
        "model_comparison": analysis_paths["model_comparison"].name,
        "explainability": analysis_paths["explainability"].name,
        "failure_modes": analysis_paths["failure_modes"].name,
    }
    (spec.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    (spec.output_dir / "conclusion.md").write_text(
        "# 中文结论\n\n"
        "- 做了什么：通过 qlib.init、D.features、DatasetH/DataHandlerLP、R.start、SignalRecord、SigAnaRecord 和 PortAnaRecord 跑通 SOXL qlib-native reference，并在 qlib Dataset 上完成多模型选择。\n"
        f"- 模型逻辑：候选模型 {len(model_selection.model_metrics)} 个，按 {_selection_segment(spec)} 段的 {spec.model_selection_metric} 选择、test 段报告，选中 {model_selection.selected_spec.name} / {model.implementation}，结果见 model_selection.csv、all_model_predictions.csv 和 selected_model.json。\n"
        "- 核心指标：已生成 qlib recorder、pred.pkl、label.pkl、sig_analysis、portfolio_analysis、risk_analysis.csv、backtest_window_metrics.csv、backtest_walk_forward_metrics.csv、backtest_benchmark_comparison.csv 和 model_comparison.csv。\n"
        "- regime 差异：regime_metrics.csv 基于 qlib recorder 和 qlib Data API 的 VIX、趋势、回撤、波动、成交量和市场状态分层，短窗口仅用于验收。\n"
        "- 是否晋升：不晋升；这是 qlib 原生工作流验收卡。\n"
        "- 风险：样本窗口短，IC 可能为 NaN；不能作为交易信号。\n"
        f"- 下一步：在 production paper/sim notebook 复用 recorder id {recorder_id}。\n",
        encoding="utf-8",
    )
    return QlibNativeExperimentResult(
        recorder_id=recorder_id,
        recorder_local_dir=recorder_local_dir,
        summary=summary,
        analysis_summary=analysis_summary,
    )
