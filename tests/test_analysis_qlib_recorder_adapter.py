from __future__ import annotations

import json

import pandas as pd

from src.analysis import build_qlib_recorder_analysis_report, load_qlib_recorder_analysis, write_analysis_report
from src.core import DatasetSnapshot


class FakeQlibRecorder:
    def __init__(self, objects: dict[str, object]):
        self.objects = objects

    def load_object(self, name: str):
        if name not in self.objects:
            raise FileNotFoundError(name)
        return self.objects[name]


def _qlib_report() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "return": [0.0, 0.02, -0.01, 0.03],
            "bench": [0.0, 0.01, -0.02, 0.02],
            "cost": [0.0, 0.001, 0.001, 0.002],
        },
        index=pd.date_range("2026-01-01", periods=4, freq="D"),
    )


def _qlib_positions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "datetime": pd.date_range("2026-01-01", periods=2, freq="D"),
            "instrument": ["SOXL", "SOXX"],
            "weight": [0.8, 0.2],
        }
    )


def test_load_qlib_recorder_analysis_normalizes_common_recorder_artifacts():
    recorder = FakeQlibRecorder(
        {
            "portfolio_analysis/report_normal_1day.pkl": _qlib_report(),
            "portfolio_analysis/risk_analysis_1day.pkl": {
                "annualized_return": 0.42,
                "information_ratio": 1.25,
                "max_drawdown": -0.08,
            },
            "portfolio_analysis/positions_normal_1day.pkl": _qlib_positions(),
        }
    )

    analysis = load_qlib_recorder_analysis(recorder, model_name="ridge_mse")

    assert analysis.summary["schema_version"] == "analysis.qlib_recorder.v1"
    assert analysis.summary["source_type"] == "recorder"
    assert analysis.summary["loaded_artifacts"]["qlib_portfolio_report"] == "portfolio_analysis/report_normal_1day.pkl"
    assert analysis.summary["missing_artifact_keys"] == []

    portfolio = analysis.tables["qlib_portfolio_report"]
    assert {"date", "model", "strategy_return", "benchmark_return", "excess_return", "nav"}.issubset(
        portfolio.columns
    )
    assert portfolio["model"].unique().tolist() == ["ridge_mse"]
    assert abs(float(portfolio.iloc[1]["strategy_return"]) - 0.02) < 1e-12
    assert abs(float(portfolio.iloc[1]["excess_return"]) - 0.01) < 1e-12
    assert float(portfolio.iloc[-1]["nav"]) > 1.0

    risk = analysis.tables["qlib_risk_analysis"]
    assert {"metric", "value", "model", "source_artifact"}.issubset(risk.columns)
    assert set(risk["metric"]) == {"annualized_return", "information_ratio", "max_drawdown"}
    assert "qlib_positions" in analysis.tables


def test_load_qlib_recorder_analysis_reads_local_artifact_directory_and_writes_report(tmp_path):
    artifact_dir = tmp_path / "qlib_recorder"
    artifact_dir.mkdir()
    _qlib_report().to_pickle(artifact_dir / "report_normal_1day.pkl")
    pd.DataFrame(
        {
            "metric": ["annualized_return", "max_drawdown"],
            "value": [0.31, -0.12],
        }
    ).to_csv(artifact_dir / "risk_analysis_1day.csv", index=False)

    analysis = load_qlib_recorder_analysis(artifact_dir, model_name="gcn_mse")
    report = build_qlib_recorder_analysis_report(
        artifact_id="qlib_recorder_adapter_v1",
        recorder_analysis=analysis,
        summary={"status": "qlib_recorder_loaded"},
    )
    dataset = DatasetSnapshot(
        name="qlib_unit",
        universe=("SOXL",),
        start="2026-01-01",
        end="2026-01-04",
        calendar_path="data/qlib/soxl/calendars/day.txt",
        data_paths=("qlib_recorder",),
    )

    result = write_analysis_report(report, tmp_path / "report", dataset=dataset)

    assert result["paths"]["qlib_portfolio_report"] == "qlib_portfolio_report.csv"
    assert result["paths"]["qlib_risk_analysis"] == "qlib_risk_analysis.csv"
    assert result["table_counts"]["qlib_portfolio_report"] == 4

    summary = json.loads((tmp_path / "report" / "analysis_summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((tmp_path / "report" / "analysis_manifest.json").read_text(encoding="utf-8"))
    assert summary["schema_version"] == "analysis.report.v1"
    assert summary["status"] == "qlib_recorder_loaded"
    assert summary["qlib_recorder_summary"]["source_type"] == "artifact_dir"
    assert "qlib_risk_analysis" in summary["analysis_layers"]
    assert manifest["metadata"]["source_type"] == "artifact_dir"
    assert manifest["paths"]["qlib_portfolio_report"] == "qlib_portfolio_report.csv"
