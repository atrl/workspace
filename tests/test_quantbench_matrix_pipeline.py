from __future__ import annotations

import json

import pandas as pd

from src.backtest import backtest_prediction_frame
from src.data import build_supervised_dataset, load_soxl_support_frame
from src.model import QUANTBENCH_MODEL_NAMES
from src.model.matrix import run_quantbench_matrix_smoke
from src.model.training import fit_predict_model


def test_load_soxl_support_frame_builds_non_leaking_supervised_dataset():
    frame = load_soxl_support_frame(start="2026-01-01", end="2026-03-31")
    dataset = build_supervised_dataset(frame, horizon=1)

    assert {"soxl_return_1d", "soxx_return_1d", "vix_change_1d", "target_return"}.issubset(dataset.frame.columns)
    assert dataset.feature_columns
    assert dataset.frame["date"].min().date().isoformat() >= "2026-01-01"
    assert dataset.frame["date"].max().date().isoformat() < "2026-03-31"

    source = frame.set_index("date")["close"].pct_change().shift(-1)
    expected = source.loc[pd.Timestamp(dataset.frame.iloc[0]["date"])]
    assert abs(float(dataset.frame.iloc[0]["target_return"]) - float(expected)) < 1e-12


def test_fit_predict_model_uses_registry_spec_and_emits_date_aligned_predictions():
    frame = load_soxl_support_frame(start="2026-01-01", end="2026-04-30")
    dataset = build_supervised_dataset(frame, horizon=1)
    train = dataset.frame.loc[dataset.frame["date"] <= "2026-03-31"]
    test = dataset.frame.loc[dataset.frame["date"] > "2026-03-31"]

    predictions = fit_predict_model("ridge_mse", train=train, test=test, feature_columns=dataset.feature_columns)

    assert predictions["model"].unique().tolist() == ["ridge_mse"]
    assert predictions["date"].tolist() == test["date"].tolist()
    assert predictions["prediction"].notna().all()
    assert predictions["target_return"].notna().all()


def test_prediction_backtest_outputs_analysis_ready_path():
    frame = load_soxl_support_frame(start="2026-01-01", end="2026-04-30")
    dataset = build_supervised_dataset(frame, horizon=1)
    train = dataset.frame.loc[dataset.frame["date"] <= "2026-03-31"]
    test = dataset.frame.loc[dataset.frame["date"] > "2026-03-31"]
    predictions = fit_predict_model("ridge_mse", train=train, test=test, feature_columns=dataset.feature_columns)

    result = backtest_prediction_frame(predictions, price_frame=frame)

    assert {"date", "model", "nav", "benchmark_nav", "position", "turnover", "vix", "trend", "drawdown"}.issubset(result.columns)
    assert result["model"].unique().tolist() == ["ridge_mse"]
    assert result["nav"].iloc[0] == 1.0


def test_quantbench_matrix_smoke_runs_every_registered_model_and_writes_sidecars(tmp_path):
    outputs = run_quantbench_matrix_smoke(
        card_dir=tmp_path,
        start="2026-01-01",
        end="2026-06-09",
        train_end="2026-04-30",
    )

    assert set(outputs.predictions["model"].unique()) == set(QUANTBENCH_MODEL_NAMES)
    assert set(outputs.daily["model"].unique()) == set(QUANTBENCH_MODEL_NAMES)
    assert set(outputs.model_comparison["model"]) == set(QUANTBENCH_MODEL_NAMES)

    expected_files = {
        "analysis_summary.json",
        "analysis_manifest.json",
        "artifact_manifest.json",
        "predictions.csv",
        "daily_backtest.csv",
        "regime_metrics.csv",
        "window_metrics.csv",
        "window_regime_metrics.csv",
        "model_comparison.csv",
        "model_capabilities.csv",
        "objective_scores.csv",
        "promotion_scores.csv",
        "qlib_risk_metrics.csv",
        "model_effect_deltas.csv",
    }
    assert expected_files.issubset({path.name for path in tmp_path.iterdir()})

    summary = json.loads((tmp_path / "analysis_summary.json").read_text(encoding="utf-8"))
    analysis_manifest = json.loads((tmp_path / "analysis_manifest.json").read_text(encoding="utf-8"))
    assert summary["schema_version"] == "analysis.report.v1"
    assert summary["model_count"] == len(QUANTBENCH_MODEL_NAMES)
    assert summary["train_end"] == "2026-04-30"
    assert summary["status"] == "smoke_completed"
    assert summary["adapter_status"] == "paper_strict_adapter"
    assert set(summary["adapter_statuses"]) == {"paper_strict_adapter"}
    assert "smoke_adapter" not in summary["adapter_statuses"]
    assert "smoke_fallback_v1" not in summary["model_adapters"]
    assert {"ridge_mse_v1", "xgboost_mse_v1", "catboost_rank_v1"}.issubset(
        set(summary["model_adapters"])
    )
    assert "qlib_risk_metrics" in summary["analysis_layers"]
    assert "window_regime_metrics" in summary["analysis_layers"]
    assert summary["backtest_contract"]["schema_version"] == "backtest.config.v1"
    assert summary["backtest_contract"]["cost_model"]["slippage_bps"] > 0.0
    assert summary["walk_forward"]["schema_version"] == "backtest.walk_forward.v1"
    assert summary["walk_forward"]["fold_count"] >= 1
    assert {"test", "ytd", "trailing_1m", "trailing_3m", "trailing_6m", "trailing_12m"}.issubset(
        set(summary["analysis_summary"]["window_names"])
    )
    assert summary["objective_evaluation"]["schema_version"] == "objective.matrix_evaluation.v1"
    assert summary["objective_evaluation"]["model_count"] == len(QUANTBENCH_MODEL_NAMES)
    assert summary["promotion_summary"]["schema_version"] == "objective.promotion_summary.v1"
    assert summary["promotion_summary"]["model_count"] == len(QUANTBENCH_MODEL_NAMES)
    assert analysis_manifest["artifact_type"] == "analysis_report"
    assert analysis_manifest["paths"]["qlib_risk_metrics"] == "qlib_risk_metrics.csv"
    assert analysis_manifest["paths"]["backtest_config"] == "backtest_config.json"
    assert analysis_manifest["paths"]["walk_forward_splits"] == "walk_forward_splits.csv"
    assert analysis_manifest["paths"]["objective_scores"] == "objective_scores.csv"
    assert analysis_manifest["paths"]["promotion_scores"] == "promotion_scores.csv"
    assert len(analysis_manifest["manifest_hash"]) == 64
    daily = pd.read_csv(tmp_path / "daily_backtest.csv")
    folds = pd.read_csv(tmp_path / "walk_forward_splits.csv")
    objective_scores = pd.read_csv(tmp_path / "objective_scores.csv")
    promotion_scores = pd.read_csv(tmp_path / "promotion_scores.csv")
    assert {"gross_strategy_return", "cost_return", "net_strategy_return"}.issubset(daily.columns)
    assert daily["cost_return"].abs().sum() > 0.0
    assert {"fold", "train_start", "train_end", "test_start", "test_end", "schema_version"}.issubset(folds.columns)
    assert {"schema_version", "model", "mse", "classification_accuracy", "ic", "rank_ic"}.issubset(
        objective_scores.columns
    )
    assert {"schema_version", "model", "promotion_score", "promotion_decision"}.issubset(promotion_scores.columns)
