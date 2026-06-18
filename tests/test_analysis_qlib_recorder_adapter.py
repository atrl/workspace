from __future__ import annotations

import json

import pandas as pd

from src.analysis import (
    build_qlib_recorder_analysis_report,
    build_qlib_recorder_regime_window_analysis_report,
    load_qlib_recorder_analysis,
    write_analysis_report,
)
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


def _qlib_report_from_returns(returns: list[float], benchmark: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "return": returns,
            "bench": benchmark,
            "cost": [0.0, 0.001, 0.002, 0.001, 0.002, 0.001, 0.001, 0.001],
        },
        index=pd.date_range("2026-01-01", periods=8, freq="D"),
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


def test_build_qlib_recorder_regime_window_report_compares_models_from_recorder_artifacts(tmp_path):
    ridge = load_qlib_recorder_analysis(
        FakeQlibRecorder(
            {
                "portfolio_analysis/report_normal_1day.pkl": _qlib_report_from_returns(
                    [0.0, 0.02, 0.02, -0.01, -0.015, -0.01, 0.005, 0.01],
                    [0.0, 0.01, 0.01, -0.01, -0.02, -0.01, 0.0, 0.0],
                ),
                "portfolio_analysis/risk_analysis_1day.pkl": {
                    "annualized_return": 0.18,
                    "information_ratio": 0.9,
                    "max_drawdown": -0.04,
                },
            }
        ),
        model_name="ridge_mse",
    )
    gcn = load_qlib_recorder_analysis(
        FakeQlibRecorder(
            {
                "portfolio_analysis/report_normal_1day.pkl": _qlib_report_from_returns(
                    [0.0, 0.04, 0.04, -0.01, -0.06, -0.04, 0.01, 0.01],
                    [0.0, 0.01, 0.01, -0.01, -0.02, -0.01, 0.0, 0.0],
                ),
                "portfolio_analysis/risk_analysis_1day.pkl": {
                    "annualized_return": 0.22,
                    "information_ratio": 0.7,
                    "max_drawdown": -0.11,
                },
            }
        ),
        model_name="gcn_mse",
    )
    regime_frame = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=8, freq="D"),
            "regime": ["bull", "bull", "bull", "neutral", "stress", "stress", "stress", "downtrend"],
            "vix": [17, 18, 19, 24, 36, 34, 31, 23],
            "trend": [0.02, 0.03, 0.04, 0.0, -0.06, -0.05, -0.03, -0.02],
            "drawdown": [0.0, 0.0, 0.0, -0.01, -0.12, -0.15, -0.10, -0.04],
        }
    )

    report = build_qlib_recorder_regime_window_analysis_report(
        artifact_id="qlib_recorder_regime_window_v1",
        recorder_analyses={"ridge_mse": ridge, "gcn_mse": gcn},
        regime_frame=regime_frame,
        windows={"early": ("2026-01-01", "2026-01-04"), "stress_window": ("2026-01-05", "2026-01-07")},
        baseline_model="ridge_mse",
        summary={"note": "中文 review 入口：从 qlib recorder artifact 直接生成 regime/window 模型差异。"},
    )

    assert report.summary["status"] == "qlib_recorder_regime_window_analysis_completed"
    assert report.summary["source_model_count"] == 2
    assert report.summary["analysis_summary"]["baseline_model"] == "ridge_mse"
    assert report.summary["qlib_recorder_summary"]["gcn_mse"]["model"] == "gcn_mse"
    assert "model_review_briefing" in report.summary["analysis_layers"]
    assert {"qlib_portfolio_report", "qlib_risk_analysis", "model_review_briefing"}.issubset(report.tables)
    assert len(report.tables["qlib_portfolio_report"]) == 16

    segment_matrix = report.tables["segment_effect_matrix"]
    assert "window_regime" in set(segment_matrix["segment_type"])
    stress = segment_matrix.loc[segment_matrix["segment"] == "stress_window::stress"].iloc[0]
    assert stress["worst_relative_return_model"] == "gcn_mse"

    briefing = report.tables["model_review_briefing"]
    assert "weakness" in set(briefing["review_signal"])
    assert "strength" in set(briefing["review_signal"])
    assert "stress" in "\n".join(briefing["segment"].astype(str).tolist())

    dataset = DatasetSnapshot(
        name="qlib_recorder_unit",
        universe=("SOXL", "SOXX"),
        start="2026-01-01",
        end="2026-01-08",
        calendar_path="data/qlib/soxl/calendars/day.txt",
        data_paths=("qlib_recorder",),
    )
    result = write_analysis_report(report, tmp_path / "report", dataset=dataset)

    assert result["paths"]["qlib_portfolio_report"] == "qlib_portfolio_report.csv"
    assert result["paths"]["qlib_risk_analysis"] == "qlib_risk_analysis.csv"
    assert result["paths"]["model_review_briefing"] == "model_review_briefing.csv"
    assert result["table_counts"]["segment_effect_matrix"] >= 4
    summary = json.loads((tmp_path / "report" / "analysis_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "qlib_recorder_regime_window_analysis_completed"
    assert summary["analysis_table_counts"]["qlib_portfolio_report"] == 16
