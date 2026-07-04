from __future__ import annotations

import json
import importlib
from pathlib import Path

import pandas as pd
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


def test_qlib_store_builder_preserves_intraday_frequency_timestamps(tmp_path):
    data_module, _ = _native_modules()
    provider_uri = tmp_path / "qlib" / "soxl_intraday_5min"
    raw = pd.DataFrame(
        [
            {
                "instrument": "SOXL",
                "datetime": pd.Timestamp("2026-01-02 09:35", tz="America/New_York"),
                "open": 100.0,
                "close": 101.0,
            },
            {
                "instrument": "SOXL",
                "datetime": pd.Timestamp("2026-01-02 09:40", tz="America/New_York"),
                "open": 101.0,
                "close": 100.5,
            },
            {
                "instrument": "SOXL",
                "datetime": pd.Timestamp("2026-01-02 09:45", tz="America/New_York"),
                "open": 100.5,
                "close": 102.0,
            },
        ]
    )

    manifest = data_module.build_qlib_store_from_frame(
        raw,
        provider_uri=provider_uri,
        freq="5min",
        as_of="2026-01-02",
    )
    loaded = data_module.load_qlib_feature_frame(
        data_module.QlibProviderConfig(provider_uri=provider_uri, region="us", market="all", freq="5min"),
        instruments=["SOXL"],
        fields=["$open", "$close"],
        start="2026-01-02 09:35:00",
        end="2026-01-02 09:45:00",
    )

    assert manifest["freq"] == "5min"
    assert manifest["calendar_start"] == "2026-01-02 09:35:00"
    assert manifest["calendar_end"] == "2026-01-02 09:45:00"
    assert manifest["max_datetime"] == "2026-01-02 09:45:00"
    assert loaded.index.get_level_values("datetime").tolist() == [
        pd.Timestamp("2026-01-02 09:35:00"),
        pd.Timestamp("2026-01-02 09:40:00"),
        pd.Timestamp("2026-01-02 09:45:00"),
    ]
    assert loaded.loc[("SOXL", pd.Timestamp("2026-01-02 09:40:00")), "$close"] == pytest.approx(100.5)


def test_qlib_provider_uses_single_worker_for_reproducible_local_runs():
    from qlib.config import C

    data_module, _ = _native_modules()
    config = data_module.QlibProviderConfig(provider_uri=PROVIDER_URI, region="us", market="all")

    init_result = data_module.init_qlib_provider(config)

    assert init_result["kernels"] == 1
    assert C.kernels == 1


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
        "WeightStrategyBase",
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
    assert summary["workflow_config"] == "workflow.yaml"
    assert (spec.output_dir / "workflow.yaml").exists()
    assert summary["artifact_policy"]["policy"] == "slim_card_recorder_first"
    assert summary["artifact_policy"]["recorder_only_artifacts"]["predictions"] == "artifacts/pred.pkl"
    assert summary["artifact_policy"]["recorder_only_artifacts"]["portfolio_report"] == "artifacts/portfolio_analysis/report_normal_1day.pkl"
    assert "predictions" not in summary["sidecars"]
    assert "signals" not in summary["sidecars"]
    assert "qlib_portfolio_report" not in summary["sidecars"]
    assert not (spec.output_dir / "predictions.csv").exists()
    assert not (spec.output_dir / "signals.csv").exists()
    assert not (spec.output_dir / "qlib_portfolio_report.csv").exists()
    assert (spec.output_dir / "risk_analysis.csv").exists()
