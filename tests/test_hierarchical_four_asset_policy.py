from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.workflow.hierarchical_four_asset_policy import (
    VARIANTS,
    HierarchicalPolicyConfig,
    candidate_configs,
)


ROOT = Path(__file__).resolve().parents[1]
CARD = ROOT / "research" / "hierarchical_four_asset_policy_v1"


def test_candidate_grid_keeps_soxs_overlay_narrow() -> None:
    configs = candidate_configs()
    assert {config.variant for config in configs} == set(VARIANTS)
    assert all(isinstance(config, HierarchicalPolicyConfig) for config in configs)
    assert max(config.soxs_weight for config in configs) == 0.10
    assert max(config.max_soxs_days for config in configs) == 3
    assert all(config.cooldown_days == 2 for config in configs)
    assert all(config.strong_soxl_weight <= 0.40 for config in configs)


def test_committed_card_is_recorder_first_and_outer_safe() -> None:
    summary = json.loads((CARD / "summary.json").read_text(encoding="utf-8"))
    assert summary["selection_uses_outer_test"] is False
    assert summary["artifact_policy"]["policy"] == "slim_card_recorder_first"
    assert summary["constraints"]["soxl_soxs_mutually_exclusive"] is True
    assert summary["constraints"]["soxs_weights_tested"] == [0.05, 0.10]
    assert summary["constraints"]["soxs_max_days_tested"] == [1, 3]
    assert summary["constraints"]["return_maxdd_must_exceed_same_average_risk_soxl"] is True
    assert summary["constraints"]["return_maxdd_must_exceed_matched_beta_soxl_soxx_cash"] is True
    assert summary["selected_model"]["selection_uses_outer_test"] is False
    assert summary["outer_diagnostic"]["selection_uses_outer_test"] is False
    assert summary["outer_diagnostic"]["years"] == [2023, 2024, 2025, 2026]
    for forbidden in (
        "predictions.csv",
        "signals.csv",
        "positions.csv",
        "strategy_returns.csv",
        "portfolio_report.csv",
    ):
        assert not (CARD / forbidden).exists()

    recorder = Path(summary["recorder_local_dir"])
    comparison = pd.read_csv(CARD / "backtest_benchmark_comparison.csv")
    strategies = comparison.loc[comparison["variant"].ne("soxx_buyhold")]
    assert strategies["same_avg_risk_soxl_return_maxdd"].notna().all()
    assert strategies["matched_beta_return_maxdd"].notna().all()
    outer_years = pd.read_csv(CARD / "outer_year_risk_metrics.csv")
    assert set(outer_years["outer_year"]) == {2023, 2024, 2025, 2026}
    assert outer_years["selection_uses_outer_test"].eq(False).all()
    positions = pd.read_csv(
        recorder / "artifacts" / summary["experiment"] / "positions.csv",
        parse_dates=["signal_date", "trade_date", "exit_date"],
    )
    assert (positions["signal_date"] < positions["trade_date"]).all()
    assert (positions["trade_date"] < positions["exit_date"]).all()
    assert ((positions["soxl_weight"] > 0.0) & (positions["soxs_weight"] > 0.0)).sum() == 0
    assert positions["soxs_weight"].max() <= 0.10 + 1e-12
    assert (positions[["soxl_weight", "soxx_weight", "soxs_weight", "cash_weight"]].sum(axis=1) - 1.0).abs().max() < 1e-10
    for (_, variant, _), group in positions.groupby(["segment", "variant", "cost_bps"]):
        maximum = 0
        run = 0
        for active in group["soxs_weight"].gt(0.0):
            run = run + 1 if active else 0
            maximum = max(maximum, run)
        expected = 0 if variant == "cash_defense_no_soxs" else int(variant[-2])
        assert maximum <= expected


def test_notebook_is_executed_and_qlib_native() -> None:
    notebook = json.loads((CARD / "hierarchical_four_asset_policy_v1.ipynb").read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "import qlib" in source
    assert "qlib.init" in source
    assert "D.calendar" in source
    assert "DatasetH" in source
    assert "from qlib.workflow import R" in source
    assert "run_hierarchical_four_asset_policy_card" in source
    assert "中文结论" in source
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert all(cell["execution_count"] is not None for cell in code_cells)
