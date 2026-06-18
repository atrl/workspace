from __future__ import annotations

import pandas as pd

from src.objective import (
    PromotionThresholds,
    build_forward_return_labels,
    build_promotion_scores,
    evaluate_prediction_objectives,
)


def _price_frame() -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=4, freq="D")
    return pd.DataFrame(
        {
            "date": list(dates) * 2,
            "instrument": ["SOXL"] * 4 + ["SOXX"] * 4,
            "close": [100.0, 102.0, 101.0, 104.0, 50.0, 51.0, 50.5, 51.5],
        }
    )


def test_build_forward_return_labels_supports_classification_mse_and_ranking_targets():
    labels = build_forward_return_labels(_price_frame(), horizons=(1, 2))

    assert {
        "date",
        "instrument",
        "close",
        "forward_return_1d",
        "classification_label_1d",
        "ranking_label_1d",
        "label_available_1d",
        "forward_return_2d",
        "classification_label_2d",
        "ranking_label_2d",
        "label_available_2d",
    }.issubset(labels.columns)

    soxl_day1 = labels.loc[(labels["instrument"] == "SOXL") & (labels["date"] == pd.Timestamp("2026-01-01"))].iloc[0]
    assert abs(float(soxl_day1["forward_return_1d"]) - 0.02) < 1e-12
    assert int(soxl_day1["classification_label_1d"]) == 1
    assert bool(soxl_day1["label_available_1d"])

    final_soxx = labels.loc[(labels["instrument"] == "SOXX") & (labels["date"] == pd.Timestamp("2026-01-04"))].iloc[0]
    assert pd.isna(final_soxx["forward_return_1d"])
    assert not bool(final_soxx["label_available_1d"])

    rank_day1 = labels.loc[labels["date"] == pd.Timestamp("2026-01-01")].sort_values("ranking_label_1d")
    assert rank_day1["instrument"].tolist() == ["SOXL", "SOXX"]
    assert set(rank_day1["ranking_label_1d"].dropna()) == {1.0, 2.0}


def test_evaluate_prediction_objectives_reports_mse_accuracy_ic_and_rank_ic():
    frame = pd.DataFrame(
        {
            "date": list(pd.date_range("2026-01-01", periods=3, freq="D")) * 2,
            "instrument": ["SOXL"] * 3 + ["SOXX"] * 3,
            "prediction": [0.03, -0.02, 0.01, 0.02, -0.01, -0.02],
            "target_return": [0.02, -0.01, 0.015, 0.01, -0.02, -0.01],
        }
    )

    scores = evaluate_prediction_objectives(frame, group_col="date")

    assert scores["schema_version"] == "objective.evaluation.v1"
    assert scores["row_count"] == 6
    assert 0.0 <= scores["mse"] < 0.001
    assert scores["classification_accuracy"] == 1.0
    assert scores["ic"] > 0.0
    assert scores["rank_ic"] > 0.0
    assert scores["rank_group_count"] == 3


def test_build_promotion_scores_applies_return_risk_utility_and_hard_gates():
    model_summary = pd.DataFrame(
        {
            "model": ["stable_mse", "fragile_graph"],
            "return_pct": [18.0, 25.0],
            "sharpe": [1.4, 1.2],
            "max_drawdown_pct": [8.0, 28.0],
            "turnover": [0.8, 3.5],
            "relative_return_pct": [6.0, 10.0],
        }
    )

    scores = build_promotion_scores(
        model_summary,
        thresholds=PromotionThresholds(max_drawdown_pct=15.0, max_turnover=2.0, min_relative_return_pct=0.0),
    )

    assert {
        "schema_version",
        "rank",
        "model",
        "promotion_score",
        "promotion_decision",
        "promotion_blockers",
    }.issubset(scores.columns)
    assert scores.iloc[0]["model"] == "stable_mse"
    stable = scores.loc[scores["model"] == "stable_mse"].iloc[0]
    fragile = scores.loc[scores["model"] == "fragile_graph"].iloc[0]
    assert stable["promotion_decision"] == "promote_candidate"
    assert stable["promotion_blockers"] == ""
    assert fragile["promotion_decision"] == "blocked"
    assert "max_drawdown_pct" in fragile["promotion_blockers"]
    assert "turnover" in fragile["promotion_blockers"]
