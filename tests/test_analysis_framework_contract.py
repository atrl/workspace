from __future__ import annotations

import json

import pandas as pd

from src.analysis import AnalysisSpec, MarketRegimeConfig, build_model_regime_comparison_report, write_analysis_report
from src.core import DatasetSnapshot


def _daily_frame() -> pd.DataFrame:
    dates = list(pd.date_range("2026-01-01", periods=8, freq="D")) * 2
    regimes = ["bull", "bull", "bull", "neutral", "stress", "stress", "stress", "downtrend"] * 2
    return pd.DataFrame(
        {
            "date": dates,
            "model": ["ridge_mse"] * 8 + ["gcn_mse"] * 8,
            "regime": regimes,
            "nav": [
                1.00,
                1.02,
                1.04,
                1.03,
                1.01,
                0.99,
                1.00,
                1.01,
                1.00,
                1.04,
                1.08,
                1.07,
                1.02,
                0.96,
                0.97,
                0.98,
            ],
            "benchmark_nav": [1.0, 1.01, 1.02, 1.02, 1.00, 0.99, 1.00, 1.00] * 2,
            "position": [0.2, 0.3, 0.5, 0.1, 0.4, 0.2, 0.1, 0.2] * 2,
            "turnover": [0.0, 0.1, 0.2, 0.4, 0.3, 0.1, 0.1, 0.2] * 2,
        }
    )


def _market_panel() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=8, freq="D"),
            "close": [100.0, 104.0, 106.0, 101.0, 90.0, 95.0, 93.0, 98.0],
            "volume": [100, 105, 110, 220, 150, 125, 115, 120],
            "soxx_close": [50.0, 51.0, 51.5, 50.5, 48.0, 49.0, 48.4, 49.2],
            "vix": [17.0, 18.0, 19.0, 22.0, 36.0, 25.0, 24.0, 18.0],
            "trend": [0.04, 0.05, 0.04, 0.00, -0.12, 0.04, -0.04, 0.03],
            "drawdown": [0.0, 0.0, 0.0, -0.02, -0.16, -0.08, -0.10, -0.04],
            "realized_vol": [0.01, 0.015, 0.018, 0.02, 0.06, 0.05, 0.052, 0.018],
            "volume_z": [0.0, 0.2, 0.3, 2.4, 0.7, 0.4, 0.2, 0.1],
        }
    )


class FakeQlibRecorder:
    def __init__(self, objects: dict[str, object]):
        self.objects = objects

    def load_object(self, name: str):
        if name not in self.objects:
            raise FileNotFoundError(name)
        return self.objects[name]


def _qlib_report_from_returns(returns: list[float], benchmark: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "return": returns,
            "bench": benchmark,
            "cost": [0.0, 0.001, 0.002, 0.001, 0.002, 0.001, 0.001, 0.001],
        },
        index=pd.date_range("2026-01-01", periods=8, freq="D"),
    )


def test_analysis_spec_serializes_stable_contract_fields():
    spec = AnalysisSpec(
        artifact_id="analysis_framework_contract_v1",
        baseline_model="ridge_mse",
        windows={"early": ("2026-01-01", "2026-01-04")},
        source_type="daily_backtest",
        summary_note="中文 review：统一 analysis pipeline contract。",
    )

    payload = spec.to_dict()

    assert payload["schema_version"] == "analysis.spec.v1"
    assert payload["artifact_id"] == "analysis_framework_contract_v1"
    assert payload["baseline_model"] == "ridge_mse"
    assert payload["windows"] == {"early": ["2026-01-01", "2026-01-04"]}
    assert payload["source_type"] == "daily_backtest"
    assert payload["summary_note"].startswith("中文 review")
    assert AnalysisSpec.from_dict(payload) == spec


def test_analysis_spec_serializes_optional_market_regime_config():
    spec = AnalysisSpec(
        artifact_id="analysis_with_config",
        baseline_model="ridge_mse",
        source_type="market_panel",
        market_regime_config=MarketRegimeConfig(high_vix_threshold=28.0, trend_window=10),
    )

    payload = spec.to_dict()
    restored = AnalysisSpec.from_dict(payload)

    assert payload["market_regime_config"]["high_vix_threshold"] == 28.0
    assert payload["market_regime_config"]["trend_window"] == 10
    assert restored.market_regime_config == spec.market_regime_config


def test_pipeline_builds_daily_backtest_report_and_manifest_ready_export(tmp_path):
    spec = AnalysisSpec(
        artifact_id="unit_daily_pipeline",
        baseline_model="ridge_mse",
        windows={"early": ("2026-01-01", "2026-01-04"), "stress": ("2026-01-05", "2026-01-07")},
        source_type="daily_backtest",
        summary_note="中文 review：daily backtest 通过统一 pipeline 生成模型差异。",
    )

    report = build_model_regime_comparison_report(spec=spec, daily_backtest=_daily_frame())

    assert report.artifact_id == "unit_daily_pipeline"
    assert report.summary["status"] == "analysis_pipeline_completed"
    assert report.summary["analysis_pipeline_schema_version"] == "analysis.spec.v1"
    assert report.summary["source_type"] == "daily_backtest"
    assert report.summary["baseline_model"] == "ridge_mse"
    assert report.summary["summary_note"].startswith("中文 review")
    assert report.summary["analysis_summary"]["baseline_model"] == "ridge_mse"
    assert "analysis_overview" in report.summary
    assert {
        "model_comparison",
        "regime_metrics",
        "window_metrics",
        "window_regime_metrics",
        "model_review_briefing",
    }.issubset(report.tables)

    dataset = DatasetSnapshot(
        name="unit_daily_pipeline",
        universe=("SOXL", "SOXX"),
        start="2026-01-01",
        end="2026-01-08",
        calendar_path="data/qlib/soxl/calendars/day.txt",
        data_paths=("unit.csv",),
    )
    export = write_analysis_report(report, tmp_path, dataset=dataset)
    manifest = json.loads((tmp_path / "analysis_manifest.json").read_text(encoding="utf-8"))

    assert export["paths"]["model_review_briefing"] == "model_review_briefing.csv"
    assert manifest["metadata"]["source_type"] == "daily_backtest"
    assert manifest["metadata"]["analysis_pipeline_schema_version"] == "analysis.spec.v1"


def test_pipeline_market_panel_route_rebuilds_regime_context():
    daily = _daily_frame().assign(regime="legacy")
    spec = AnalysisSpec(
        artifact_id="unit_market_pipeline",
        baseline_model="ridge_mse",
        windows={"stress": ("2026-01-05", "2026-01-07")},
        source_type="market_panel",
        summary_note="中文 review：market panel 通过 pipeline 重算 regime。",
    )

    report = build_model_regime_comparison_report(spec=spec, daily_backtest=daily, market_frame=_market_panel())

    assert report.summary["status"] == "analysis_pipeline_completed"
    assert report.summary["source_type"] == "market_panel"
    assert report.summary["source_report_status"] == "market_regime_analysis_completed"
    assert "market_regimes" in report.tables
    assert "market_regime_daily_backtest" in report.tables
    assert "legacy" not in set(report.tables["market_regime_daily_backtest"]["regime"])
    assert report.tables["market_regime_daily_backtest"]["regime"].nunique() > 1


def test_pipeline_qlib_recorder_route_normalizes_and_compares_models():
    ridge = FakeQlibRecorder(
        {
            "portfolio_analysis/report_normal_1day.pkl": _qlib_report_from_returns(
                [0.0, 0.02, 0.02, -0.01, -0.015, -0.01, 0.005, 0.01],
                [0.0, 0.01, 0.01, -0.01, -0.02, -0.01, 0.0, 0.0],
            ),
            "portfolio_analysis/risk_analysis_1day.pkl": {"annualized_return": 0.18, "information_ratio": 0.9},
        }
    )
    gcn = FakeQlibRecorder(
        {
            "portfolio_analysis/report_normal_1day.pkl": _qlib_report_from_returns(
                [0.0, 0.04, 0.04, -0.01, -0.06, -0.04, 0.01, 0.01],
                [0.0, 0.01, 0.01, -0.01, -0.02, -0.01, 0.0, 0.0],
            ),
            "portfolio_analysis/risk_analysis_1day.pkl": {"annualized_return": 0.22, "information_ratio": 0.7},
        }
    )
    regime_frame = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=8, freq="D"),
            "regime": ["bull", "bull", "bull", "neutral", "stress", "stress", "stress", "downtrend"],
        }
    )
    spec = AnalysisSpec(
        artifact_id="unit_qlib_pipeline",
        baseline_model="ridge_mse",
        windows={"stress": ("2026-01-05", "2026-01-07")},
        source_type="qlib_recorder",
        summary_note="中文 review：qlib recorder 通过 pipeline 生成模型差异。",
    )

    report = build_model_regime_comparison_report(
        spec=spec,
        qlib_recorders={"ridge_mse": ridge, "gcn_mse": gcn},
        regime_frame=regime_frame,
    )

    assert report.summary["status"] == "analysis_pipeline_completed"
    assert report.summary["source_type"] == "qlib_recorder"
    assert report.summary["source_report_status"] == "qlib_recorder_regime_window_analysis_completed"
    assert {"qlib_portfolio_report", "qlib_risk_analysis", "segment_effect_matrix", "model_review_briefing"}.issubset(
        report.tables
    )
    assert len(report.tables["qlib_portfolio_report"]) == 16
