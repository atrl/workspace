from __future__ import annotations

import json

from src.data import build_supervised_dataset, load_soxl_support_frame
from src.factor import prepare_factor_feature_sets
from src.model.matrix import run_factor_feature_set_matrix


def test_prepare_factor_feature_sets_uses_train_only_ic_and_applies_composite_without_test_leakage():
    frame = load_soxl_support_frame(start="2026-01-01", end="2026-04-30")
    dataset = build_supervised_dataset(frame, horizon=1)
    train = dataset.frame.loc[dataset.frame["date"] <= "2026-03-31"].copy()
    test = dataset.frame.loc[dataset.frame["date"] > "2026-03-31"].copy()

    plan = prepare_factor_feature_sets(
        train=train,
        test=test,
        factor_columns=dataset.feature_columns,
        target_column=dataset.target_column,
        top_n=4,
    )

    assert [spec.name for spec in plan.feature_sets] == ["all_features", "top_ic_factors", "ic_weighted_composite"]
    assert len(plan.feature_sets[1].feature_columns) == 4
    assert plan.feature_sets[2].feature_columns == ("ic_weighted_composite",)
    assert "ic_weighted_composite" in plan.train.columns
    assert "ic_weighted_composite" in plan.test.columns
    expected_selected = (
        plan.diagnostics.ic.assign(abs_ic=plan.diagnostics.ic["ic_mean"].astype(float).abs())
        .sort_values(["abs_ic", "ic_mean", "factor"], ascending=[False, False, True])["factor"]
        .head(4)
        .tolist()
    )
    assert expected_selected == list(plan.feature_sets[1].feature_columns)
    assert plan.summary["selection_source"] == "train_only_factor_ic"
    assert plan.summary["top_n"] == 4


def test_run_factor_feature_set_matrix_compares_same_models_across_factor_sets(tmp_path):
    outputs = run_factor_feature_set_matrix(
        card_dir=tmp_path,
        start="2026-01-01",
        end="2026-06-09",
        train_end="2026-04-30",
        model_names=("ridge_mse", "random_forest_clf"),
        top_n_factors=3,
    )

    assert set(outputs.predictions["feature_set"]) == {"all_features", "top_ic_factors", "ic_weighted_composite"}
    assert set(outputs.predictions["base_model"]) == {"ridge_mse", "random_forest_clf"}
    assert outputs.predictions["model"].str.contains("__").all()
    assert {"model", "base_model", "feature_set"}.issubset(outputs.model_comparison.columns)

    expected_files = {
        "analysis_summary.json",
        "analysis_manifest.json",
        "artifact_manifest.json",
        "feature_set_specs.json",
        "factor_ic.csv",
        "predictions.csv",
        "daily_backtest.csv",
        "model_comparison.csv",
        "objective_scores.csv",
        "promotion_scores.csv",
        "regime_metrics.csv",
        "window_metrics.csv",
        "window_regime_metrics.csv",
        "qlib_risk_metrics.csv",
        "model_effect_deltas.csv",
    }
    assert expected_files.issubset({path.name for path in tmp_path.iterdir()})

    summary = json.loads((tmp_path / "analysis_summary.json").read_text(encoding="utf-8"))
    analysis_manifest = json.loads((tmp_path / "analysis_manifest.json").read_text(encoding="utf-8"))
    assert summary["schema_version"] == "analysis.report.v1"
    assert summary["status"] == "feature_set_matrix_completed"
    assert summary["base_model_count"] == 2
    assert summary["feature_set_count"] == 3
    assert summary["adapter_status"] == "paper_strict_adapter"
    assert summary["adapter_statuses"] == ["paper_strict_adapter"]
    assert {"ridge_mse_v1", "random_forest_direction_v1"}.issubset(set(summary["model_adapters"]))
    assert summary["selection_source"] == "train_only_factor_ic"
    assert "window_regime_metrics" in summary["analysis_layers"]
    assert {"test", "ytd", "trailing_1m", "trailing_3m", "trailing_6m", "trailing_12m"}.issubset(
        set(summary["analysis_summary"]["window_names"])
    )
    assert summary["backtest_contract"]["schema_version"] == "backtest.config.v1"
    assert summary["backtest_contract"]["cost_model"]["open_cost_bps"] > 0.0
    assert summary["walk_forward"]["fold_count"] >= 1
    assert summary["objective_evaluation"]["model_count"] == summary["variant_count"]
    assert summary["promotion_summary"]["model_count"] == summary["variant_count"]
    assert analysis_manifest["metadata"]["analysis_report_schema_version"] == "analysis.report.v1"
    assert analysis_manifest["paths"]["model_effect_deltas"] == "model_effect_deltas.csv"
    assert analysis_manifest["paths"]["backtest_config"] == "backtest_config.json"
    assert analysis_manifest["paths"]["walk_forward_splits"] == "walk_forward_splits.csv"
    assert analysis_manifest["paths"]["objective_scores"] == "objective_scores.csv"
    assert analysis_manifest["paths"]["promotion_scores"] == "promotion_scores.csv"
