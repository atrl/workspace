from __future__ import annotations

import pandas as pd
import pytest

from src.soxl_quant.app.state_objective_budget_gap import (
    build_budget_usage_gap_diagnostic,
    find_flat_positive_runs,
)


def test_find_flat_positive_runs_detects_missed_reentry_window():
    daily = pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                ["2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"]
            ),
            "price": [100.0, 110.0, 132.0, 120.0, 144.0],
            "position": [0.0, 0.0, 0.0, 0.3, 0.0],
        }
    )

    runs = find_flat_positive_runs(daily, flat_position_threshold=0.05, min_run_return_pct=10.0)

    assert len(runs) == 1
    assert runs[0]["start"] == "2026-01-02"
    assert runs[0]["end"] == "2026-01-06"
    assert runs[0]["days"] == 3
    assert runs[0]["return_pct"] == pytest.approx(32.0)


def test_budget_usage_gap_diagnostic_blocks_ppo_when_ytd_gap_is_structural():
    ytd = pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                ["2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"]
            ),
            "strategy_nav": [100.0, 100.0, 104.0, 108.0, 110.0],
            "buyhold_nav": [100.0, 120.0, 150.0, 200.0, 240.0],
            "price": [10.0, 12.0, 15.0, 20.0, 24.0],
            "position": [0.0, 0.0, 0.0, 0.02, 0.0],
            "action": ["HOLD", "HOLD", "HOLD", "HOLD", "HOLD"],
        }
    )
    wf_2020 = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"]),
            "strategy_nav": [100.0, 78.5, 81.0],
            "buyhold_nav": [100.0, 120.0, 130.0],
            "price": [10.0, 12.0, 13.0],
            "position": [0.4, 0.4, 0.0],
            "action": ["HOLD", "HOLD", "HOLD"],
        }
    )
    metrics = pd.DataFrame(
        {
            "window": ["ytd_2026", "wf_2020"],
            "return_pct": [10.0, -19.0],
            "buyhold_return_pct": [140.0, 30.0],
            "max_dd_pct": [0.0, 21.5],
            "avg_position": [0.004, 0.266],
        }
    )

    diagnostic = build_budget_usage_gap_diagnostic(
        metrics=metrics,
        daily_by_window={"ytd_2026": ytd, "wf_2020": wf_2020},
        required_soxlbacktest_return_pct=132.0,
        required_best_return_pct=213.0,
        max_walkforward_drawdown_pct=20.0,
    )

    assert diagnostic.summary["decision"] == "revise_fold_local_reentry_participation_targets_before_ppo"
    assert diagnostic.summary["ppo_allowed_from_current_target"] is False
    assert diagnostic.summary["structural_ytd_participation_gap"] is True
    assert diagnostic.summary["walkforward_drawdown_gap"] is True
    assert diagnostic.summary["largest_ytd_flat_positive_run_return_pct"] == 140.0
    assert set(diagnostic.window_summary["window"]) == {"ytd_2026", "wf_2020"}
