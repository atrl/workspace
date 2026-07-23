from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


EXPERIMENT = "cash_floor_relative_target_feature_reintegration_v1"


def _card() -> tuple[Path, dict]:
    root = Path(__file__).resolve().parents[1]
    card = root / "research" / EXPERIMENT
    return card, json.loads((card / "summary.json").read_text(encoding="utf-8"))


def test_feature_reintegration_card_is_qlib_native_outer_safe_and_slim() -> None:
    card, summary = _card()
    notebook = json.loads((card / f"{EXPERIMENT}.ipynb").read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

    assert summary["selection_uses_outer_test"] is False
    assert summary["outer_test_use"] == "diagnostic_only_previously_observed"
    assert summary["qlib_dataset"]["api"] == ["DatasetH", "DataHandlerLP", "StaticDataLoader"]
    assert summary["artifact_policy"]["policy"] == "slim_card_recorder_first"
    assert (
        summary["artifact_policy"]["retained_model_scope"]
        == "selected_development_folds_and_outer_refit_only"
    )
    assert summary["artifact_policy"]["retained_model_count"] == 16
    assert (
        summary["artifact_policy"]["retained_portfolio_scope"]
        == "selected_candidate_only"
    )
    assert summary["objective"]["aggregation"] == "weighted_mean_not_pointwise_max"
    assert summary["selected_model"]["volatility_weight"] < 1.0
    assert set(summary["feature_reintegration"]["feature_sets"]) == {
        "current10",
        "compact_reintegrated",
        "full_reintegrated",
    }
    assert "survivorship_bias" in summary["data_audit"]["excluded_sources"]["soxx_constituents"]
    for name in ("predictions.csv", "positions.csv", "strategy_returns.csv"):
        assert not (card / name).exists()
    recorder_models = (
        Path(summary["recorder_local_dir"]) / "artifacts" / EXPERIMENT / "models"
    )
    model_names = sorted(path.name for path in recorder_models.glob("*.pkl"))
    assert len(model_names) == 16
    assert all(
        name.startswith("current10__ridge__")
        or name.startswith("outer_refit__")
        for name in model_names
    )
    for token in ("qlib.init", "D.calendar", "DatasetH", "DataHandlerLP", "R.get_recorder"):
        assert token in source
    assert "最终中文结论" in source
    assert notebook["cells"][1]["execution_count"] is not None
    assert summary["recorder_id"] in json.dumps(notebook)


def test_reintegrated_oos_audit_records_predictive_but_not_decision_increment() -> None:
    card, summary = _card()
    audit = pd.read_csv(card / "feature_oos_audit.csv")
    cash_ridge = audit.query(
        "model_family == 'ridge' and target == 'cash_protection_5d'"
    ).set_index("feature_set")

    assert cash_ridge.loc["compact_reintegrated", "pooled_spearman"] > cash_ridge.loc[
        "current10", "pooled_spearman"
    ]
    assert cash_ridge.loc["compact_reintegrated", "positive_fold_count"] == 7
    assert summary["selected_model"]["feature_set"] == "current10"
    assert summary["selected_model"]["relative_gate"] is True
    assert summary["selected_model"]["risk_gate"] is False
    assert summary["promotion_decision"] == "needs_iteration_tail_risk_after_feature_reintegration"


def test_selected_feature_reintegration_positions_obey_contract_and_fail_honestly() -> None:
    card, summary = _card()
    artifact_root = Path(summary["recorder_local_dir"]) / "artifacts" / EXPERIMENT
    positions = pd.read_csv(artifact_root / "positions.csv")
    selected = str(summary["selected_model"]["candidate"])
    assert set(positions["strategy"]) == {selected}
    development = positions.loc[
        positions["segment"].eq("development_walk_forward")
        & positions["strategy"].eq(selected)
        & positions["cost_bps"].eq(10)
    ].copy()
    weights = development[
        ["soxl_weight", "soxx_weight", "soxs_weight", "cash_weight"]
    ]

    assert np.allclose(weights.sum(axis=1), 1.0)
    assert development["cash_weight"].min() >= 0.10 - 1e-12
    assert development["cash_weight"].max() <= 0.60 + 1e-12
    assert development["cash_weight"].mean() >= 0.15
    assert development["soxs_weight"].eq(0.0).all()
    risky = development["soxl_weight"] + development["soxx_weight"]
    assert np.allclose(development["soxl_weight"] / risky, 0.30)
    assert (pd.to_datetime(development["signal_date"]) < pd.to_datetime(development["trade_date"])).all()

    comparison = pd.read_csv(card / "backtest_benchmark_comparison.csv")
    development_metrics = comparison.query(
        "segment == 'development_walk_forward'"
    ).set_index("cost_bps")
    assert development_metrics.loc[10, "terminal_wealth_ratio_vs_soxx"] > 1.0
    assert development_metrics.loc[25, "terminal_wealth_ratio_vs_soxx"] > 1.0
    assert development_metrics.loc[10, "max_drawdown_pct"] < -20.0
    assert development_metrics.loc[10, "worst_rolling_5d_pct"] < -12.0
    assert development_metrics.loc[10, "terminal_wealth_ratio_vs_soxx"] == pytest.approx(
        summary["selected_model"]["ratio_10bps"]
    )
