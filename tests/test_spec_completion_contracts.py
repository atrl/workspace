from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.data import QlibProviderConfig, build_qlib_dataset
from src.workflow import QlibNativeExperimentSpec, run_qlib_native_experiment


ROOT = Path(__file__).resolve().parents[1]
PROVIDER_URI = ROOT / "data" / "qlib" / "soxl"


def _config() -> QlibProviderConfig:
    return QlibProviderConfig(provider_uri=PROVIDER_URI, region="us", market="all")


def _dataset():
    return build_qlib_dataset(
        _config(),
        instruments=["SOXL"],
        features=["$close", "$vix", "$volume"],
        label="Ref($close, -1) / $close - 1",
        train=("2026-01-02", "2026-01-08"),
        valid=("2026-01-09", "2026-01-09"),
        test=("2026-01-12", "2026-01-15"),
    )


def test_alpha158_legacy_dataset_stacks_qlib_alpha158_with_previous_soxl_features():
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP

    from src.data import build_alpha158_legacy_dataset

    dataset = build_alpha158_legacy_dataset(
        _config(),
        instruments=["SOXL"],
        label="Ref($close, -1) / $close - 1",
        train=("2026-01-02", "2026-01-08"),
        valid=("2026-01-09", "2026-01-09"),
        test=("2026-01-12", "2026-01-15"),
    )
    train = dataset.prepare("train")
    valid = dataset.prepare("valid")

    assert isinstance(dataset, DatasetH)
    assert isinstance(dataset.handler, DataHandlerLP)
    assert dataset.qlib_feature_set["source"] == "Alpha158.get_feature_config+legacy_soxl_features"
    assert dataset.qlib_feature_set["alpha158_count"] == 158
    assert dataset.qlib_feature_set["legacy_feature_names"] == [
        "VIX",
        "RSI6",
        "HV_20",
        "HV_30",
        "VOL_COEF",
        "PANIC_SCORE",
        "SOXL_RET_5",
        "VOLUME_RATIO_5",
    ]
    assert {"KMID", "ROC5", "MA20", "VIX", "RSI6", "PANIC_SCORE"}.issubset(train.columns)
    assert not valid.empty
    assert dataset.qlib_segments["valid"] == ("2026-01-09", "2026-01-09")
    assert "Alpha158.get_feature_config" in dataset.qlib_dataset_api


def _workflow(tmp_path):
    return run_qlib_native_experiment(
        QlibNativeExperimentSpec(
            experiment="spec_completion_contracts",
            provider_uri=PROVIDER_URI,
            recorder_uri=tmp_path / "qlib_recorder",
            output_dir=tmp_path / "card",
            market="all",
            region="us",
            instrument="SOXL",
            train=("2026-01-02", "2026-01-08"),
            valid=("2026-01-09", "2026-01-09"),
            test=("2026-01-12", "2026-01-15"),
            model_selection_metric="objective_score",
        )
    )


def test_data_layer_audits_qlib_store_and_writes_snapshot_manifest(tmp_path):
    from src.data import audit_qlib_provider, build_data_update_manifest, write_snapshot_manifest

    audit = audit_qlib_provider(
        _config(),
        instruments=["SOXL"],
        fields=["$close", "$vix", "$volume"],
        start="2026-01-02",
        end="2026-01-15",
    )
    manifest_path = write_snapshot_manifest(audit, tmp_path, as_of="2026-01-15")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    update_manifest = build_data_update_manifest(audit, as_of="2026-01-15")

    assert audit["status"] == "ok"
    assert audit["source_api"] == [
        "qlib.init",
        "qlib.data.D.calendar",
        "qlib.data.D.instruments",
        "qlib.data.D.features",
    ]
    assert audit["provider_uri"] == PROVIDER_URI.resolve().as_posix()
    assert audit["feature_rows"] > 0
    assert audit["max_datetime"] <= "2026-01-15"
    assert audit["missing_features"] == {}
    assert manifest["source"] == "qlib.data.D.features"
    assert manifest["snapshot_hash"] == audit["snapshot_hash"]
    assert manifest["as_of"] == "2026-01-15"
    assert update_manifest["operation"] == "qlib_data_update"
    assert update_manifest["source"] == "qlib.data.D.features"
    assert update_manifest["status"] == "ok"


def test_data_layer_builds_qlib_store_from_raw_csv_then_reads_back_with_qlib(tmp_path):
    from src.data import audit_qlib_provider, build_qlib_store_from_csv, QlibUnavailableError

    raw = tmp_path / "raw_soxl.csv"
    pd.DataFrame(
        {
            "instrument": ["SOXL", "SOXL", "SOXL"],
            "datetime": ["2026-01-02", "2026-01-05", "2026-01-06"],
            "close": [10.0, 11.0, 12.0],
            "vix": [18.0, 19.0, 17.5],
            "volume": [1000, 1200, 900],
        }
    ).to_csv(raw, index=False)

    provider_uri = tmp_path / "qlib_provider"
    manifest = build_qlib_store_from_csv(
        raw,
        provider_uri=provider_uri,
        as_of="2026-01-06",
        field_map={"close": "close", "vix": "vix", "volume": "volume"},
    )
    audit = audit_qlib_provider(
        QlibProviderConfig(provider_uri=provider_uri, region="us", market="all"),
        instruments=["SOXL"],
        fields=["$close", "$vix", "$volume"],
        start="2026-01-02",
        end="2026-01-06",
    )

    assert manifest["operation"] == "qlib_store_build"
    assert manifest["status"] == "ok"
    assert manifest["source"] == raw.resolve().as_posix()
    assert manifest["feature_file_count"] == 3
    assert manifest["source_api"] == [
        "qlib.init",
        "qlib.data.storage.file_storage.FileFeatureStorage",
        "qlib.data.D.calendar",
        "qlib.data.D.instruments",
        "qlib.data.D.features",
    ]
    assert (provider_uri / "qlib_store_manifest.json").exists()
    assert (provider_uri / "features" / "soxl" / "close.day.bin").exists()
    assert audit["status"] == "ok"
    assert audit["source_api"][-1] == "qlib.data.D.features"

    with pytest.raises(QlibUnavailableError, match="raw max_datetime .* exceeds as_of"):
        build_qlib_store_from_csv(raw, provider_uri=tmp_path / "blocked_provider", as_of="2026-01-05")


def test_objective_layer_exposes_classification_mse_ic_ranking_and_utility_metrics():
    from src.objective import classification_metrics, ic_metrics, mse_metrics, ranking_metrics, utility_score

    actual = pd.Series([0.02, -0.01, 0.03, -0.02], name="actual")
    predicted = pd.Series([0.01, -0.02, 0.04, 0.01], name="predicted")

    assert classification_metrics(actual, predicted)["accuracy"] == 0.75
    assert mse_metrics(actual, predicted)["mse"] == pytest.approx(0.0003)
    assert ic_metrics(actual, predicted)["ic"] > 0.5
    assert ranking_metrics(actual, predicted)["rank_ic"] > 0.0
    assert utility_score(return_pct=12.0, max_drawdown_pct=-3.0, turnover=0.4, ic=0.05, rank_ic=0.04) > 0.0


def test_factor_layer_combines_scores_from_qlib_dataset():
    from src.factor import combine_qlib_factor_scores, qlib_factor_diagnostics

    dataset = _dataset()
    scored = combine_qlib_factor_scores(
        dataset,
        segment="test",
        weights={"$close": 0.7, "$vix": -0.3},
        output_col="combo_score",
    )
    report = qlib_factor_diagnostics(
        scored,
        segment="test",
        factor_cols=["$close", "$vix"],
        score_col="combo_score",
        label_col="Ref($close, -1) / $close - 1",
        quantiles=2,
    )

    assert scored.attrs["source"] == "qlib_dataset"
    assert "combo_score" in scored.prepare("test").columns
    assert report.ic_table["factor"].tolist() == ["$close", "$vix", "combo_score"]


def test_model_registry_each_family_has_qlib_dataset_fit_predict_smoke():
    from src.model import quantbench_supervised_model_specs, run_model_family_smoke_matrix

    rows = run_model_family_smoke_matrix(_dataset(), specs=quantbench_supervised_model_specs())
    families = {row["family"] for row in rows}

    assert families >= {"linear", "tree", "sequence", "transformer"}
    assert {row["source"] for row in rows} == {"qlib_dataset"}
    assert all(row["prediction_rows"] > 0 for row in rows)


def test_model_selection_trains_distinct_qlib_models_and_persists_selected_model(tmp_path):
    from src.model import load_qlib_model, quantbench_supervised_model_specs, run_qlib_model_selection

    dataset = _dataset()
    result = run_qlib_model_selection(
        dataset,
        specs=quantbench_supervised_model_specs(),
        output_dir=tmp_path,
        selection_metric="utility_score",
        selection_segment="valid",
        report_segment="test",
    )

    rows = result.model_metrics
    selected = rows.loc[rows["selected"]]
    score_signatures = {
        tuple(round(float(value), 8) for value in scores["score"].head(4))
        for scores in result.predictions_by_model.values()
    }

    assert len(rows) == len(quantbench_supervised_model_specs())
    assert rows["source"].eq("qlib_dataset").all()
    assert rows["selection_segment"].eq("valid").all()
    assert rows["report_segment"].eq("test").all()
    assert {"selection_utility_score", "test_utility_score", "selection_objective_score", "test_objective_score"}.issubset(rows.columns)
    assert rows["implementation"].nunique() >= 4
    assert {"mse", "ic", "ranking", "utility"}.issubset(set(rows["objective"]))
    assert len(score_signatures) >= 3
    assert len(selected) == 1
    assert selected.iloc[0]["model"] == result.selected_spec.name
    assert result.selected_model_path.exists()
    assert json.loads(result.selected_model_path.read_text(encoding="utf-8"))["model"] == result.selected_spec.name
    loaded = load_qlib_model(result.selected_model_path)
    pd.testing.assert_series_equal(
        result.selected_model.predict(dataset).round(10),
        loaded.predict(dataset).round(10),
    )


def test_workflow_generates_native_portana_record_outputs(tmp_path):
    result = _workflow(tmp_path)
    artifacts = set(result.summary["recorder_artifacts"])

    assert "PortAnaRecord" in result.summary["qlib_native_api"]
    assert "WeightStrategyBase" in result.summary["qlib_native_api"]
    assert result.summary["target_position_strategy"] | {"score_threshold": 0.0} == {
        "class": "SignalWeightStrategy",
        "module_path": "src.backtest.strategy",
        "signal_source": "<PRED>",
        "threshold_source": "train_prediction_median",
        "score_threshold": 0.0,
        "min_weight": 0.0,
        "max_weight": 1.0,
        "risk_degree": 0.95,
    }
    assert isinstance(result.summary["target_position_strategy"]["score_threshold"], float)
    assert result.summary["feature_set"]["source"] == "Alpha158.get_feature_config+legacy_soxl_features"
    assert result.summary["model_selection"]["selection_segment"] == "valid"
    assert result.summary["model_selection"]["report_segment"] == "test"
    assert result.summary["model_selection"]["selection_metric"] == "objective_score"
    assert result.summary["model_selection"]["selection_column"] == "selection_objective_score"
    assert result.summary["legacy_soxlbacktest_benchmark"]["benchmark_name"] == "legacy_soxl_backtest"
    assert isinstance(result.summary["legacy_soxlbacktest_benchmark"]["exceeds_legacy_rulebase"], bool)
    assert "artifacts/portfolio_analysis/report_normal_1day.pkl" in artifacts
    assert "artifacts/portfolio_analysis/positions_normal_1day.pkl" in artifacts
    assert "artifacts/portfolio_analysis/port_analysis_1day.pkl" in artifacts
    assert result.summary["sidecars"]["qlib_portfolio_report"] == "qlib_portfolio_report.csv"
    assert result.summary["sidecars"]["risk_analysis"] == "risk_analysis.csv"
    assert (tmp_path / "card" / "qlib_portfolio_report.csv").exists()
    assert (tmp_path / "card" / "risk_analysis.csv").exists()


def test_backtest_layer_derives_windows_walk_forward_benchmark_and_costs_from_qlib_report(tmp_path):
    from src.backtest import build_backtest_extension_report_from_qlib_recorder, build_portfolio_report_from_qlib_recorder

    result = _workflow(tmp_path)
    portfolio = build_portfolio_report_from_qlib_recorder(result.summary, cost_bps=2.0, slippage_bps=1.0)
    extension = build_backtest_extension_report_from_qlib_recorder(
        result.summary,
        portfolio.report,
        windows={"acceptance": ("2026-01-09", "2026-01-15")},
        stress_windows={"mini_stress": ("2026-01-12", "2026-01-15")},
        walk_forward={"train_size": 1, "test_size": 2},
        cost_bps=2.0,
        slippage_bps=1.0,
    )

    assert extension.source == "qlib_portfolio_analysis"
    assert {"acceptance", "mini_stress"}.issubset(set(extension.window_metrics["window"]))
    assert any(str(window).startswith("walk_forward_") for window in extension.walk_forward_metrics["window"])
    assert {
        "strategy_return_pct",
        "benchmark_return_pct",
        "excess_return_pct",
        "max_drawdown_pct",
        "turnover",
        "cost",
        "slippage_cost",
    }.issubset(extension.window_metrics.columns)
    assert extension.benchmark_comparison.iloc[0]["benchmark_source"] == "qlib_portfolio_analysis"
    assert extension.benchmark_comparison.iloc[0]["recorder_id"] == result.summary["recorder_id"]
    assert "portfolio_analysis/report_normal_1day.pkl" in extension.manifest.metadata["input_artifacts"]
    assert "portfolio_analysis/port_analysis_1day.pkl" in extension.manifest.metadata["input_artifacts"]


def test_analysis_covers_required_regimes_windows_walk_forward_and_explainability(tmp_path):
    from src.analysis import build_analysis_report_from_qlib_recorder
    from src.backtest import build_portfolio_report_from_qlib_recorder

    result = _workflow(tmp_path)
    portfolio = build_portfolio_report_from_qlib_recorder(result.summary, cost_bps=2.0, slippage_bps=1.0)
    report = build_analysis_report_from_qlib_recorder(
        result.summary,
        portfolio.report,
        windows={"custom_test": ("2026-01-09", "2026-01-15")},
        walk_forward={"train_size": 2, "test_size": 2},
    )

    assert {
        "vix",
        "trend",
        "drawdown",
        "volatility",
        "volume",
        "market",
    }.issubset(set(report.regime_metrics["regime_type"]))
    assert {"custom_test", "walk_forward_1"}.issubset(set(report.window_metrics["window"]))
    assert {"feature", "importance", "method"}.issubset(report.explainability.columns)
    assert set(report.explainability["method"]) >= {"permutation", "window_sensitivity"}


def test_schedule_runs_data_model_dashboard_notify_with_retry(tmp_path):
    from src.schedule import RefreshPlan, run_refresh_plan

    events: list[str] = []
    calls = {"data": 0}
    manifest = {"source": "qlib.data.D.features", "provider_uri": PROVIDER_URI.as_posix(), "row_count": 5}

    def data_refresh():
        events.append("data")
        calls["data"] += 1
        if calls["data"] == 1:
            raise RuntimeError("transient qlib refresh error")
        return [manifest]

    def model_refresh(manifests):
        events.append("model")
        return [{"model": "qlib_mean_return", "source": "qlib_recorder_metrics", "manifests": len(manifests)}]

    def dashboard_export(manifests, model_results):
        events.append("dashboard")
        return {"source": "qlib_recorder", "artifacts": len(manifests), "models": len(model_results)}

    def notify(summary):
        events.append(f"notify:{summary['status']}")

    summary = run_refresh_plan(
        RefreshPlan(name="qlib-full-refresh", lock_path=tmp_path / "refresh.lock", attempts=2),
        data_refresh=data_refresh,
        model_refresh=model_refresh,
        dashboard_export=dashboard_export,
        notify=notify,
    )

    assert events == ["data", "data", "model", "dashboard", "notify:ok"]
    assert summary["status"] == "ok"
    assert summary["attempts"] == 2
    assert summary["model_results"] == 1
