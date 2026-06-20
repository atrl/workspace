from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.data import QlibProviderConfig, build_qlib_dataset
from src.workflow import QlibNativeExperimentSpec, run_qlib_native_experiment


ROOT = Path(__file__).resolve().parents[1]
PROVIDER_URI = ROOT / "data" / "qlib" / "soxl"


def _dataset():
    return build_qlib_dataset(
        QlibProviderConfig(provider_uri=PROVIDER_URI, region="us", market="all"),
        instruments=["SOXL"],
        features=["$close", "$vix"],
        label="Ref($close, -1) / $close - 1",
        train=("2026-01-02", "2026-01-08"),
        test=("2026-01-09", "2026-01-15"),
    )


def _workflow_result(tmp_path):
    return run_qlib_native_experiment(
        QlibNativeExperimentSpec(
            experiment="qlib_native_platform_contracts",
            provider_uri=PROVIDER_URI,
            recorder_uri=tmp_path / "qlib_recorder",
            output_dir=tmp_path / "card",
            market="all",
            region="us",
            instrument="SOXL",
            train=("2026-01-02", "2026-01-08"),
            test=("2026-01-09", "2026-01-15"),
        )
    )


def test_core_manifest_requires_qlib_provider_and_recorder():
    from src.core import ArtifactManifest, RunConfig

    manifest = ArtifactManifest(
        kind="analysis_summary",
        path="research/qlib_native_reference_v1/analysis_summary.json",
        source="qlib_recorder",
        provider_uri=PROVIDER_URI.as_posix(),
        recorder_id="abc123",
        date_range=("2026-01-02", "2026-01-15"),
        purpose="dashboard",
        metadata={"experiment": "qlib_native_reference_v1"},
    )
    config = RunConfig(
        name="qlib_native_reference_v1",
        surface="research",
        provider_uri=PROVIDER_URI.as_posix(),
        market="all",
        region="us",
        as_of="2026-01-15",
        output_dir="research/qlib_native_reference_v1",
        recorder_id="abc123",
        artifacts=["summary.json", "analysis_summary.json"],
    )

    assert manifest.to_dict()["source"] == "qlib_recorder"
    assert manifest.to_dict()["artifact_id"] == manifest.artifact_id
    assert config.to_manifest()["qlib_native"] is True
    assert config.to_manifest()["recorder_id"] == "abc123"

    with pytest.raises(ValueError, match="qlib"):
        ArtifactManifest(
            kind="bad",
            path="research/bad/summary.json",
            source="csv_only",
            provider_uri=PROVIDER_URI.as_posix(),
            recorder_id="abc123",
            date_range=("2026-01-02", "2026-01-15"),
            purpose="dashboard",
        )


def test_factor_model_and_objective_are_qlib_dataset_driven(tmp_path):
    from src.factor import qlib_factor_diagnostics
    from src.model import QlibMeanReturnModel, quantbench_supervised_model_specs
    from src.objective import promotion_scores

    dataset = _dataset()
    factor_report = qlib_factor_diagnostics(
        dataset,
        segment="test",
        factor_cols=["$close", "$vix"],
        label_col="Ref($close, -1) / $close - 1",
        quantiles=2,
    )
    sidecars = factor_report.write_sidecars(tmp_path)

    model = QlibMeanReturnModel().fit(dataset)
    predictions = model.predict(dataset)
    model_path = tmp_path / "qlib_mean_return_model.json"
    model.save(model_path)
    loaded = QlibMeanReturnModel.load(model_path)

    ranked = promotion_scores(
        pd.DataFrame(
            [
                {
                    "model": "qlib_mean_return",
                    "return_pct": 12.0,
                    "max_drawdown_pct": -3.0,
                    "turnover": 0.2,
                    "ic": 0.03,
                    "rank_ic": 0.04,
                    "recorder_id": "abc123",
                }
            ]
        ),
        min_score=5.0,
        max_drawdown_floor=-10.0,
    )

    assert factor_report.source == "qlib_dataset"
    assert factor_report.ic_table["factor"].tolist() == ["$close", "$vix"]
    assert {"factor_ic.csv", "factor_correlation.csv", "factor_quantiles.csv"} == {
        path.name for path in sidecars.values()
    }
    assert predictions.index.names == ["datetime", "instrument"]
    assert loaded.predict(dataset).tolist() == predictions.tolist()
    assert {spec.family for spec in quantbench_supervised_model_specs()} >= {"linear", "tree", "sequence"}
    assert ranked.loc[0, "promote"] is True
    assert ranked.loc[0, "source"] == "qlib_recorder_metrics"


def test_backtest_analysis_and_dashboard_read_qlib_recorder_artifacts(tmp_path):
    from src.analysis import build_analysis_report_from_qlib_recorder
    from src.backtest import build_portfolio_report_from_qlib_recorder
    from src.dashboard import load_dashboard_export, write_dashboard_export

    result = _workflow_result(tmp_path)
    portfolio = build_portfolio_report_from_qlib_recorder(result.summary, cost_bps=2.0, slippage_bps=1.0)
    report = build_analysis_report_from_qlib_recorder(
        result.summary,
        portfolio.report,
        windows={"smoke": ("2026-01-09", "2026-01-15")},
    )
    paths = report.write_sidecars(tmp_path / "analysis")
    export = write_dashboard_export(
        tmp_path / "dashboard_manifest.json",
        manifests=[portfolio.manifest],
        analysis_summary=json.loads(paths["summary"].read_text(encoding="utf-8")),
        generated_by="qlib-native-platform-contracts",
    )

    assert portfolio.source == "qlib_recorder"
    assert "pred.pkl" in portfolio.input_artifacts
    assert portfolio.report["nav"].iloc[-1] > 0
    assert report.summary["source"] == "qlib_recorder"
    assert report.summary["recorder_id"] == result.recorder_id
    assert export == load_dashboard_export(tmp_path / "dashboard_manifest.json")
    assert export["artifacts"][0]["source"] == "qlib_recorder"


def test_schedule_and_api_are_qlib_manifest_driven(tmp_path):
    from src.api import QlibDataAdapter, PaperBroker, RealBrokerAdapter, TargetPosition
    from src.schedule import RefreshPlan, run_refresh_plan

    config = QlibProviderConfig(provider_uri=PROVIDER_URI, region="us", market="all")
    adapter = QlibDataAdapter(config=config, instruments=["SOXL"], fields=["$close"])
    frame = adapter.load_frame(start="2026-01-09", end="2026-01-15")
    adapter_manifest = adapter.to_manifest(start="2026-01-09", end="2026-01-15")
    broker = PaperBroker(cash=10_000)
    intent = broker.create_order_intent(
        TargetPosition(
            symbol="SOXL",
            target_percent=0.25,
            price=float(frame["$close"].iloc[-1]),
            as_of="2026-01-15",
            recorder_id="abc123",
        )
    )

    events: list[str] = []

    def data_refresh():
        events.append("data")
        return [adapter_manifest]

    def dashboard_export(manifests):
        events.append("dashboard")
        return {"status": "ok", "artifacts": len(manifests)}

    summary = run_refresh_plan(
        RefreshPlan(name="qlib-data-dashboard-refresh", lock_path=tmp_path / "refresh.lock"),
        data_refresh=data_refresh,
        dashboard_export=dashboard_export,
    )

    assert frame.attrs["qlib_data_api"] == ["qlib.data.D.features"]
    assert adapter_manifest["source"] == "qlib.data.D.features"
    assert intent.to_dict()["execution_mode"] == "paper_sim"
    assert intent.to_dict()["real_broker_order"] is False
    assert intent.to_dict()["recorder_id"] == "abc123"
    assert events == ["data", "dashboard"]
    assert summary["status"] == "ok"

    with pytest.raises(RuntimeError, match="disabled in v1"):
        RealBrokerAdapter().submit_order(intent)
