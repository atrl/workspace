from __future__ import annotations

import pandas as pd
import pytest

import src.analysis as analysis_api
from src.core import DatasetSnapshot


def test_build_analysis_windows_returns_fixed_recent_historical_and_custom_windows():
    windows = analysis_api.build_analysis_windows(
        start="2020-01-01",
        end="2026-06-09",
        custom_windows={"covid_crash": ("2020-02-19", "2020-03-23")},
    )

    assert windows["ytd"] == ("2026-01-01", "2026-06-09")
    assert windows["trailing_1m"] == ("2026-05-09", "2026-06-09")
    assert windows["trailing_3m"] == ("2026-03-09", "2026-06-09")
    assert windows["trailing_6m"] == ("2025-12-09", "2026-06-09")
    assert windows["trailing_12m"] == ("2025-06-09", "2026-06-09")
    assert windows["year_2020"] == ("2020-01-01", "2020-12-31")
    assert windows["year_2022"] == ("2022-01-01", "2022-12-31")
    assert windows["period_2023_2024"] == ("2023-01-01", "2024-12-31")
    assert windows["covid_crash"] == ("2020-02-19", "2020-03-23")


def test_build_analysis_windows_clips_to_available_range_and_skips_non_overlapping_windows():
    windows = analysis_api.build_analysis_windows(
        start="2022-03-01",
        end="2026-06-09",
        custom_windows={"before_data": ("2020-01-01", "2020-12-31"), "custom_overlap": ("2022-01-01", "2022-04-15")},
    )

    assert "year_2020" not in windows
    assert "before_data" not in windows
    assert windows["year_2022"] == ("2022-03-01", "2022-12-31")
    assert windows["period_2023_2024"] == ("2023-01-01", "2024-12-31")
    assert windows["custom_overlap"] == ("2022-03-01", "2022-04-15")


def test_build_analysis_window_table_serializes_window_contract_for_sidecars():
    windows = {
        "ytd": ("2026-01-01", "2026-06-09"),
        "custom": ("2026-05-01", "2026-05-31"),
    }

    table = analysis_api.build_analysis_window_table(windows, source="unit_test")

    assert table.to_dict(orient="records") == [
        {
            "schema_version": "analysis.window_presets.v1",
            "window": "ytd",
            "start": "2026-01-01",
            "end": "2026-06-09",
            "source": "unit_test",
        },
        {
            "schema_version": "analysis.window_presets.v1",
            "window": "custom",
            "start": "2026-05-01",
            "end": "2026-05-31",
            "source": "unit_test",
        },
    ]


def test_build_analysis_windows_rejects_invalid_window_ranges():
    with pytest.raises(ValueError, match="analysis window end before start"):
        analysis_api.build_analysis_windows(start="2026-06-09", end="2026-01-01")

    with pytest.raises(ValueError, match="custom window 'bad' end before start"):
        analysis_api.build_analysis_windows(
            start="2026-01-01",
            end="2026-06-09",
            custom_windows={"bad": ("2026-05-01", "2026-04-01")},
        )


def test_regime_window_analysis_summary_records_window_definitions():
    dates = list(pd.date_range("2026-01-01", periods=8, freq="D")) * 2
    daily = pd.DataFrame(
        {
            "date": dates,
            "model": ["ridge_mse"] * 8 + ["gcn_mse"] * 8,
            "regime": ["bull", "bull", "bull", "neutral", "stress", "stress", "stress", "downtrend"] * 2,
            "nav": [1.0, 1.02, 1.04, 1.03, 1.01, 0.99, 1.0, 1.01, 1.0, 1.04, 1.08, 1.07, 1.02, 0.96, 0.97, 0.98],
            "benchmark_nav": [1.0, 1.01, 1.02, 1.02, 1.0, 0.99, 1.0, 1.0] * 2,
            "position": [0.2, 0.3, 0.5, 0.1, 0.4, 0.2, 0.1, 0.2] * 2,
            "turnover": [0.0, 0.1, 0.2, 0.4, 0.3, 0.1, 0.1, 0.2] * 2,
        }
    )
    windows = analysis_api.build_analysis_windows(
        start="2026-01-01",
        end="2026-01-08",
        presets=("ytd",),
        custom_windows={"stress_window": ("2026-01-05", "2026-01-07")},
    )

    analysis = analysis_api.analyze_model_effects_by_regime_and_window(daily, windows=windows, baseline_model="ridge_mse")

    assert analysis.summary["window_definitions"] == [
        {"window": "ytd", "start": "2026-01-01", "end": "2026-01-08"},
        {"window": "stress_window", "start": "2026-01-05", "end": "2026-01-07"},
    ]
    assert analysis.summary["window_names"] == ["ytd", "stress_window"]


def test_analysis_report_export_promotes_window_definitions_to_summary(tmp_path):
    dates = list(pd.date_range("2026-01-01", periods=8, freq="D")) * 2
    daily = pd.DataFrame(
        {
            "date": dates,
            "model": ["ridge_mse"] * 8 + ["gcn_mse"] * 8,
            "regime": ["bull", "bull", "bull", "neutral", "stress", "stress", "stress", "downtrend"] * 2,
            "nav": [1.0, 1.02, 1.04, 1.03, 1.01, 0.99, 1.0, 1.01, 1.0, 1.04, 1.08, 1.07, 1.02, 0.96, 0.97, 0.98],
            "benchmark_nav": [1.0, 1.01, 1.02, 1.02, 1.0, 0.99, 1.0, 1.0] * 2,
            "position": [0.2, 0.3, 0.5, 0.1, 0.4, 0.2, 0.1, 0.2] * 2,
            "turnover": [0.0, 0.1, 0.2, 0.4, 0.3, 0.1, 0.1, 0.2] * 2,
        }
    )
    windows = analysis_api.build_analysis_windows(
        start="2026-01-01",
        end="2026-01-08",
        presets=("ytd",),
        custom_windows={"stress_window": ("2026-01-05", "2026-01-07")},
    )
    effect_analysis = analysis_api.analyze_model_effects_by_regime_and_window(
        daily,
        windows=windows,
        baseline_model="ridge_mse",
    )
    report = analysis_api.build_regime_window_analysis_report(
        artifact_id="unit_window_report",
        effect_analysis=effect_analysis,
    )

    analysis_api.write_analysis_report(
        report,
        tmp_path,
        dataset=DatasetSnapshot(
            name="unit",
            universe=["SOXL"],
            start="2026-01-01",
            end="2026-01-08",
            calendar_path="data/qlib/soxl/calendars/day.txt",
            data_paths=[],
        ),
    )

    payload = pd.read_json(tmp_path / "analysis_summary.json", typ="series").to_dict()
    assert payload["window_names"] == ["ytd", "stress_window"]
    assert payload["window_definitions"] == [
        {"window": "ytd", "start": "2026-01-01", "end": "2026-01-08"},
        {"window": "stress_window", "start": "2026-01-05", "end": "2026-01-07"},
    ]
