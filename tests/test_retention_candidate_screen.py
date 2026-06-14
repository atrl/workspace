from __future__ import annotations

import pandas as pd


def test_overlay_metrics_use_prior_day_position_and_drawdown_caps():
    from src.soxl_quant.app.retention_candidate_screen import OverlaySpec, evaluate_position_path

    frame = pd.DataFrame(
        {
            "datetime": ["2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07"],
            "price": [100.0, 110.0, 88.0, 96.8],
            "position": [1.0, 1.0, 1.0, 1.0],
        }
    )

    metrics = evaluate_position_path(
        frame,
        OverlaySpec(
            name="cap_after_10pct_drawdown",
            base_scale=1.0,
            global_cap=1.0,
            drawdown_lookback=3,
            mild_drawdown=-0.10,
            mild_cap=0.25,
            severe_drawdown=-0.30,
            severe_cap=0.10,
        ),
    )

    assert metrics.return_pct == -9.8
    assert metrics.max_drawdown_pct == 20.0
    assert metrics.avg_position == 0.625
    assert metrics.last_position == 0.25


def test_strict_gate_requires_complete_wf_drawdown_and_ytd_return_alignment():
    from src.soxl_quant.app.retention_candidate_screen import CandidateMetrics, strict_gate_failures

    passing = CandidateMetrics(
        candidate="passing",
        ytd_return_pct=214.0,
        ytd_max_drawdown_pct=19.0,
        worst_wf_max_drawdown_pct=19.9,
        min_wf_return_pct=1.0,
        mean_wf_return_pct=3.0,
        wf_windows_present=("wf_2020", "wf_2022", "wf_2023_2024"),
        missing_wf_windows=(),
    )
    assert strict_gate_failures(
        passing,
        required_wf_windows=("wf_2020", "wf_2022", "wf_2023_2024"),
        max_walkforward_drawdown_pct=20.0,
        required_soxlbacktest_return_pct=132.077978,
        required_best_return_pct=213.143462,
    ) == []

    failing = CandidateMetrics(
        candidate="failing",
        ytd_return_pct=196.0,
        ytd_max_drawdown_pct=31.0,
        worst_wf_max_drawdown_pct=20.0,
        min_wf_return_pct=-23.0,
        mean_wf_return_pct=10.0,
        wf_windows_present=("wf_2020", "wf_2022"),
        missing_wf_windows=("wf_2023_2024",),
    )
    assert strict_gate_failures(
        failing,
        required_wf_windows=("wf_2020", "wf_2022", "wf_2023_2024"),
        max_walkforward_drawdown_pct=20.0,
        required_soxlbacktest_return_pct=132.077978,
        required_best_return_pct=213.143462,
    ) == [
        "missing_wf_window:wf_2023_2024",
        "worst_wf_max_drawdown_ge_20",
        "ytd_return_below_current_best_alignment",
    ]


def test_export_metrics_use_strategy_nav_when_available():
    from src.soxl_quant.app.retention_candidate_screen import evaluate_export_nav_path

    frame = pd.DataFrame(
        {
            "datetime": ["2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07"],
            "price": [100.0, 110.0, 88.0, 96.8],
            "position": [0.2, 0.2, 0.2, 0.2],
            "strategy_nav": [100.0, 120.0, 90.0, 135.0],
        }
    )

    metrics = evaluate_export_nav_path(frame)

    assert metrics.return_pct == 35.0
    assert metrics.max_drawdown_pct == 25.0
    assert metrics.avg_position == 0.2
    assert metrics.last_position == 0.2
