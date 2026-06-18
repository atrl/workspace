from __future__ import annotations

import json

import pandas as pd

from src.analysis import MarketRegimeConfig, build_market_regime_analysis_report, write_analysis_report
from src.core import DatasetSnapshot


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


def _daily_backtest() -> pd.DataFrame:
    dates = list(pd.date_range("2026-01-01", periods=8, freq="D")) * 2
    return pd.DataFrame(
        {
            "date": dates,
            "model": ["ridge_mse"] * 8 + ["gcn_mse"] * 8,
            "regime": ["legacy"] * 16,
            "nav": [
                1.00,
                1.02,
                1.04,
                1.01,
                0.99,
                1.00,
                1.01,
                1.03,
                1.00,
                1.05,
                1.08,
                1.04,
                0.94,
                0.97,
                0.96,
                1.02,
            ],
            "benchmark_nav": [1.0, 1.03, 1.04, 1.00, 0.91, 0.94, 0.92, 0.97] * 2,
            "position": [0.3, 0.4, 0.4, 0.2, 0.2, 0.3, 0.2, 0.3] * 2,
            "turnover": [0.0, 0.1, 0.0, 0.2, 0.0, 0.1, 0.1, 0.1] * 2,
        }
    )


def test_build_market_regime_analysis_report_reclassifies_daily_backtest_and_exports_regime_tables(tmp_path):
    report = build_market_regime_analysis_report(
        artifact_id="unit_market_regime_report",
        daily_backtest=_daily_backtest(),
        market_frame=_market_panel(),
        windows={"early": ("2026-01-01", "2026-01-04"), "stress": ("2026-01-05", "2026-01-07")},
        baseline_model="ridge_mse",
        config=MarketRegimeConfig(
            high_vix_threshold=30.0,
            stress_drawdown_threshold=-0.10,
            uptrend_threshold=0.03,
            downtrend_threshold=-0.03,
            high_volatility_threshold=0.04,
            volume_shock_z_threshold=2.0,
        ),
        summary={"note": "中文 review：从 market regime 重算模型差异。"},
    )

    assert report.summary["status"] == "market_regime_analysis_completed"
    assert report.summary["market_regime_schema_version"] == "analysis.market_regime.v1"
    assert report.summary["market_regime_count"] >= 3
    assert report.summary["source_daily_model_count"] == 2
    assert report.summary["analysis_summary"]["baseline_model"] == "ridge_mse"
    assert "market_regimes" in report.summary["analysis_layers"]
    assert "market_regime_daily_backtest" in report.summary["analysis_layers"]
    assert "market_regime_counts" in report.summary["analysis_layers"]

    market_regimes = report.tables["market_regimes"]
    assert {"date", "regime", "regime_tags", "regime_reason"}.issubset(market_regimes.columns)
    assert "stress" in set(market_regimes["regime"])

    counts = report.tables["market_regime_counts"]
    assert {"regime", "row_count", "row_share"}.issubset(counts.columns)
    assert abs(float(counts["row_share"].sum()) - 1.0) < 1e-12

    reclassified = report.tables["market_regime_daily_backtest"]
    assert reclassified["regime"].nunique() > 1
    assert "legacy" not in set(reclassified["regime"])
    assert {"market_regime", "regime_reason"}.issubset(reclassified.columns)

    dataset = DatasetSnapshot(
        name="unit_market_regime_report",
        universe=("SOXL", "SOXX", "VIX"),
        start="2026-01-01",
        end="2026-01-08",
        calendar_path="data/qlib/soxl/calendars/day.txt",
        data_paths=("unit_market.csv",),
    )
    result = write_analysis_report(report, tmp_path, dataset=dataset)

    assert result["paths"]["market_regimes"] == "market_regimes.csv"
    assert result["paths"]["market_regime_counts"] == "market_regime_counts.csv"
    assert result["paths"]["market_regime_daily_backtest"] == "market_regime_daily_backtest.csv"
    assert result["table_counts"]["market_regime_daily_backtest"] == 16

    summary = json.loads((tmp_path / "analysis_summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((tmp_path / "analysis_manifest.json").read_text(encoding="utf-8"))
    assert summary["analysis_table_counts"]["market_regimes"] == 8
    assert manifest["paths"]["market_regimes"] == "market_regimes.csv"
    assert manifest["metadata"]["source_type"] == "market_regime_analysis"
