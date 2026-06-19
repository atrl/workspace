from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.analysis import AnalysisReport, assign_regimes, compare_model_metrics, compute_window_metrics
from src.api import PaperBroker
from src.backtest import compare_to_benchmark, run_costed_backtest, signal_to_position, walk_forward_slices
from src.core import ArtifactManifest, SCHEMA_VERSION, stable_hash
from src.dashboard import export_manifest_payload
from src.data import canonical_snapshot_from_panel, load_qlib_calendar
from src.factor import factor_correlation_matrix, factor_ic, quantile_forward_returns
from src.model import MeanReturnModel, ModelRegistry, ModelSpec
from src.objective import classification_accuracy, information_coefficient, mean_squared_error, rank_ic, utility_score
from src.schedule import JobLock, retry


ROOT = Path(__file__).resolve().parents[1]


def test_core_manifest_has_stable_schema_hash_and_validation():
    left = {"b": 2, "a": [1, 2]}
    right = {"a": [1, 2], "b": 2}

    assert SCHEMA_VERSION == "1.0"
    assert stable_hash(left) == stable_hash(right)

    manifest = ArtifactManifest(
        kind="analysis_summary",
        path="research/demo/analysis_summary.json",
        source="qlib://soxl",
        date_range=("2026-01-01", "2026-06-09"),
        purpose="audit",
        metadata={"b": 2, "a": 1},
    )

    payload = manifest.to_dict()

    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["artifact_id"] == manifest.artifact_id
    assert payload["path"] == "research/demo/analysis_summary.json"
    assert payload["metadata"] == {"a": 1, "b": 2}


def test_data_snapshot_aligns_to_qlib_calendar_and_blocks_future_rows():
    calendar = load_qlib_calendar(ROOT / "data" / "qlib" / "soxl")
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-02", "2026-01-03", "2026-01-05", "2026-01-07"]),
            "close": [10.0, 11.0, 12.0, 13.0],
            "volume": [100, 200, 300, 400],
        }
    )

    snapshot = canonical_snapshot_from_panel(
        panel,
        calendar=calendar,
        instrument="SOXL",
        as_of="2026-01-06",
        source_path=ROOT / "data" / "processed" / "panels" / "soxl_soxx_vix_full.csv",
    )

    assert snapshot.instrument == "SOXL"
    assert snapshot.as_of == "2026-01-06"
    assert snapshot.data["date"].dt.date.astype(str).tolist() == ["2026-01-02", "2026-01-05"]
    assert snapshot.attrs["data_source"] == "qlib_calendar_plus_canonical_snapshot"
    assert snapshot.attrs["source_path"].endswith("soxl_soxx_vix_full.csv")


def test_factor_and_objective_metrics_are_deterministic():
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=4, freq="D"),
            "factor_a": [1.0, 2.0, 3.0, 4.0],
            "factor_b": [4.0, 3.0, 2.0, 1.0],
            "forward_return": [0.01, 0.02, 0.03, 0.04],
            "label": [0, 1, 1, 0],
            "prediction": [0, 1, 1, 1],
        }
    )

    assert factor_ic(frame, "factor_a", "forward_return") == 1.0
    assert information_coefficient(frame["factor_a"], frame["forward_return"]) == 1.0
    assert rank_ic(frame["factor_b"], frame["forward_return"]) == -1.0
    assert classification_accuracy(frame["label"], frame["prediction"]) == 0.75
    assert mean_squared_error([1.0, 2.0], [1.5, 1.5]) == 0.25
    assert utility_score(return_pct=20.0, max_drawdown_pct=-5.0, turnover=0.5) == 16.5

    corr = factor_correlation_matrix(frame, ["factor_a", "factor_b"])
    assert corr.loc["factor_a", "factor_b"] == -1.0

    quantiles = quantile_forward_returns(frame, "factor_a", "forward_return", quantiles=2)
    assert quantiles.to_dict("records") == [
        {"quantile": 1, "mean_forward_return": 0.015, "count": 2},
        {"quantile": 2, "mean_forward_return": 0.035, "count": 2},
    ]


def test_backtest_analysis_and_report_outputs_cover_windows_and_regimes(tmp_path):
    daily = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07"]),
            "close": [100.0, 110.0, 99.0, 108.9],
            "signal": [-1.0, 0.5, 2.0, 1.0],
            "benchmark_nav": [1.0, 1.05, 1.00, 1.08],
            "vix": [15.0, 18.0, 25.0, 35.0],
            "volume": [100, 110, 200, 500],
        }
    )

    positions = signal_to_position(daily["signal"], min_position=0.0, max_position=1.0)
    result = run_costed_backtest(daily, price_col="close", signal_col="signal", cost_bps=10, slippage_bps=5)
    comparison = compare_to_benchmark(result, benchmark_nav=daily["benchmark_nav"])
    regimes = assign_regimes(result.assign(vix=daily["vix"], volume=daily["volume"]), price_col="close")
    windows = compute_window_metrics(result, windows={"full": ("2026-01-02", "2026-01-07")})
    model_compare = compare_model_metrics(
        pd.DataFrame(
            [
                {"model": "stable", "return_pct": 10.0, "max_drawdown_pct": -2.0, "turnover": 0.2},
                {"model": "fast", "return_pct": 12.0, "max_drawdown_pct": -8.0, "turnover": 1.2},
            ]
        )
    )
    report = AnalysisReport(
        name="spec_rebuild_foundation_v1",
        summary={"winner": model_compare.iloc[0]["model"]},
        regime_metrics=regimes[["date", "vix_regime", "trend_regime"]],
        window_metrics=windows,
        model_comparison=model_compare,
        failure_modes=["sample-size-small"],
    )
    paths = report.write_sidecars(tmp_path)

    assert positions.tolist() == [0.0, 0.5, 1.0, 1.0]
    assert {"nav", "strategy_return", "turnover", "cost"}.issubset(result.columns)
    assert comparison["excess_return_pct"] != 0.0
    assert regimes["vix_regime"].tolist() == ["low_vix", "low_vix", "mid_vix", "high_vix"]
    assert windows.loc[0, "window"] == "full"
    assert model_compare["model"].tolist() == ["stable", "fast"]
    assert {path.name for path in paths.values()} == {
        "analysis_summary.json",
        "regime_metrics.csv",
        "window_metrics.csv",
        "model_comparison.csv",
        "failure_modes.md",
    }

    folds = walk_forward_slices(pd.date_range("2026-01-01", periods=8, freq="D"), train_size=3, test_size=2)
    assert folds[0] == {
        "train_start": "2026-01-01",
        "train_end": "2026-01-03",
        "test_start": "2026-01-04",
        "test_end": "2026-01-05",
    }


def test_model_dashboard_schedule_and_paper_api_contracts(tmp_path):
    registry = ModelRegistry()
    spec = ModelSpec(name="mean_return", family="tabular", objective="mse", features=["f1"], description="smoke")
    registry.register(spec)

    train = pd.DataFrame({"f1": [1.0, 2.0, 3.0], "target": [0.01, 0.03, 0.05]})
    model = MeanReturnModel().fit(train, target_col="target")
    prediction = model.predict(pd.DataFrame({"f1": [10.0, 20.0]}))

    manifest = ArtifactManifest(
        kind="model_registry",
        path="production/demo/summary.json",
        source="qlib://soxl",
        date_range=("2026-01-01", "2026-01-03"),
        purpose="dashboard-export",
        metadata={"model": spec.name},
    )
    dashboard_payload = export_manifest_payload([manifest])

    calls = {"count": 0}

    def flaky_job() -> str:
        calls["count"] += 1
        if calls["count"] < 3:
            raise RuntimeError("not yet")
        return "ok"

    broker = PaperBroker(cash=10_000)
    order = broker.set_target_position("SOXL", target_percent=0.25, price=50.0)

    with JobLock(tmp_path / "job.lock"):
        assert (tmp_path / "job.lock").exists()

    assert registry.get("mean_return") == spec
    assert prediction.tolist() == [0.03, 0.03]
    assert dashboard_payload["artifacts"][0]["kind"] == "model_registry"
    assert retry(flaky_job, attempts=3, delay_seconds=0.0) == "ok"
    assert order == {"symbol": "SOXL", "target_percent": 0.25, "quantity": 50}
    assert broker.positions["SOXL"] == 50
