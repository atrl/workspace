from __future__ import annotations

import json
import importlib
from pathlib import Path

import pytest
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP


ROOT = Path(__file__).resolve().parents[1]
PROVIDER_URI = ROOT / "data" / "qlib" / "soxl"


def _native_modules():
    data_module = importlib.import_module("src.data.qlib_native")
    workflow_module = importlib.import_module("src.workflow.qlib_native")
    return data_module, workflow_module


def test_qlib_data_api_and_dataset_are_native():
    data_module, _ = _native_modules()
    config = data_module.QlibProviderConfig(provider_uri=PROVIDER_URI, region="us", market="all")

    init_result = data_module.init_qlib_provider(config)
    frame = data_module.load_qlib_feature_frame(
        config,
        instruments=["SOXL"],
        fields=["$close", "$vix"],
        start="2026-01-02",
        end="2026-01-06",
    )
    dataset = data_module.build_qlib_dataset(
        config,
        instruments=["SOXL"],
        features=["$close", "$vix"],
        label="Ref($close, -1) / $close - 1",
        train=("2026-01-02", "2026-01-06"),
        test=("2026-01-07", "2026-01-09"),
    )
    train = dataset.prepare("train")

    assert init_result["provider_uri"] == PROVIDER_URI.resolve().as_posix()
    assert "qlib.init" in init_result["qlib_api"]
    assert frame.attrs["qlib_data_api"] == ["qlib.data.D.features"]
    assert frame.index.names == ["instrument", "datetime"]
    assert frame.loc[("SOXL", "2026-01-02"), "$close"] > 0
    assert isinstance(dataset, DatasetH)
    assert isinstance(dataset.handler, DataHandlerLP)
    assert {"$close", "$vix", "Ref($close, -1) / $close - 1"}.issubset(train.columns)


def test_missing_provider_fails_closed_without_pandas_fallback(tmp_path):
    data_module, _ = _native_modules()
    config = data_module.QlibProviderConfig(provider_uri=tmp_path / "missing-provider", region="us", market="all")

    with pytest.raises(data_module.QlibUnavailableError, match="blocked: qlib unavailable"):
        data_module.init_qlib_provider(config)


def test_qlib_workflow_recorder_and_record_outputs_are_durable(tmp_path):
    _, workflow_module = _native_modules()
    spec = workflow_module.QlibNativeExperimentSpec(
        experiment="qlib_native_reference_v1",
        provider_uri=PROVIDER_URI,
        recorder_uri=tmp_path / "qlib_recorder",
        output_dir=tmp_path / "card",
        market="all",
        region="us",
        instrument="SOXL",
        train=("2026-01-02", "2026-01-08"),
        test=("2026-01-09", "2026-01-15"),
    )

    result = workflow_module.run_qlib_native_experiment(spec)
    summary = json.loads((spec.output_dir / "summary.json").read_text(encoding="utf-8"))
    analysis = json.loads((spec.output_dir / "analysis_summary.json").read_text(encoding="utf-8"))

    assert summary["experiment"] == "qlib_native_reference_v1"
    assert summary["provider_uri"] == PROVIDER_URI.resolve().as_posix()
    assert summary["recorder_id"] == result.recorder_id
    assert summary["qlib_native_api"] == [
        "qlib.init",
        "qlib.data.D.features",
        "DatasetH",
        "DataHandlerLP",
        "Alpha158.get_feature_config",
        "qlib.workflow.R.start",
        "SignalRecord",
        "SigAnaRecord",
        "PortAnaRecord",
    ]
    assert "PortAnaRecord" in summary["qlib_record_classes_available"]
    assert "artifacts/code_status.txt" not in summary["recorder_artifacts"]
    assert "artifacts/code_diff.txt" not in summary["recorder_artifacts"]
    assert "artifacts/code_cached.txt" not in summary["recorder_artifacts"]
    assert "https://qlib.readthedocs.io/en/latest/component/data.html" == summary["qlib_docs"]["data"]
    assert (Path(summary["recorder_local_dir"]) / "artifacts" / "pred.pkl").exists()
    assert (Path(summary["recorder_local_dir"]) / "artifacts" / "label.pkl").exists()
    assert (Path(summary["recorder_local_dir"]) / "artifacts" / "sig_analysis" / "ic.pkl").exists()
    assert (Path(summary["recorder_local_dir"]) / "artifacts" / "portfolio_analysis" / "report_normal_1day.pkl").exists()
    assert (Path(summary["recorder_local_dir"]) / "artifacts" / "portfolio_analysis" / "port_analysis_1day.pkl").exists()
    assert analysis["source"] == "qlib_recorder"
    assert (spec.output_dir / "predictions.csv").exists()
    assert (spec.output_dir / "signals.csv").exists()
    assert (spec.output_dir / "qlib_portfolio_report.csv").exists()
    assert (spec.output_dir / "risk_analysis.csv").exists()
