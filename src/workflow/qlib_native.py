from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.analysis import build_analysis_report_from_qlib_recorder
from src.backtest import build_portfolio_report_from_qlib_recorder
from src.data import QlibProviderConfig, build_qlib_dataset, init_qlib_provider, load_qlib_feature_frame


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


def _port_analysis_config(spec: QlibNativeExperimentSpec) -> dict[str, Any]:
    return {
        "strategy": {
            "class": "TopkDropoutStrategy",
            "module_path": "qlib.contrib.strategy",
            "kwargs": {"signal": "<PRED>", "topk": 1, "n_drop": 0},
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


def run_qlib_native_experiment(spec: QlibNativeExperimentSpec) -> QlibNativeExperimentResult:
    from qlib.workflow import R
    from qlib.workflow.record_temp import PortAnaRecord, SigAnaRecord, SignalRecord

    config = QlibProviderConfig(provider_uri=spec.provider_uri, region=spec.region, market=spec.market)
    provider_meta = init_qlib_provider(config)
    dataset = build_qlib_dataset(
        config,
        instruments=[spec.instrument],
        features=["$close", "$vix"],
        label="Ref($close, -1) / $close - 1",
        train=spec.train,
        test=spec.test,
    )
    feature_frame = load_qlib_feature_frame(
        config,
        instruments=[spec.instrument],
        fields=["$close", "$vix"],
        start=spec.train[0],
        end=spec.test[1],
    )
    model = LastReturnSignalModel().fit(dataset)
    spec.output_dir.mkdir(parents=True, exist_ok=True)
    spec.recorder_uri.mkdir(parents=True, exist_ok=True)
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
            test_start=spec.test[0],
            test_end=spec.test[1],
        )
        SignalRecord(model, dataset, recorder).generate()
        SigAnaRecord(recorder).generate()
        PortAnaRecord(recorder, _port_analysis_config(spec), risk_analysis_freq="day").generate()
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
        "instrument": spec.instrument,
        "recorder_id": recorder_id,
        "recorder_uri": spec.recorder_uri.resolve().as_posix(),
        "recorder_local_dir": recorder_local_dir.as_posix(),
        "recorder_artifacts": _recorder_files(recorder_local_dir),
        "qlib_native_api": [
            "qlib.init",
            "qlib.data.D.features",
            "DatasetH",
            "DataHandlerLP",
            "qlib.workflow.R.start",
            "SignalRecord",
            "SigAnaRecord",
            "PortAnaRecord",
        ],
        "qlib_record_classes_available": ["SignalRecord", "SigAnaRecord", "PortAnaRecord"],
        "qlib_docs": QLIB_DOCS,
        "sidecars": sidecars,
    }
    portfolio = build_portfolio_report_from_qlib_recorder(summary, cost_bps=2.0, slippage_bps=1.0)
    portfolio_path = spec.output_dir / "portfolio_report.csv"
    portfolio.report.to_csv(portfolio_path, index=False)
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
        "portfolio_report": portfolio_path.name,
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
        "- 做了什么：通过 qlib.init、D.features、DatasetH/DataHandlerLP、R.start、SignalRecord、SigAnaRecord 和 PortAnaRecord 跑通 SOXL qlib-native reference。\n"
        "- 核心指标：已生成 qlib recorder、pred.pkl、label.pkl、sig_analysis、portfolio_analysis、risk_analysis.csv、window_metrics.csv 和 model_comparison.csv。\n"
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
