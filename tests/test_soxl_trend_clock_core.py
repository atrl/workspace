from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.backtest.strategy import DirectTargetWeightStrategy
from src.workflow.soxl_trend_clock_core import (
    PROFILES,
    TrendClockProfile,
    replay_trend_clock,
)


ROOT = Path(__file__).resolve().parents[1]
CARD = ROOT / "research" / "soxl_trend_clock_core_v1"


def test_trend_clock_uses_close_then_next_open_return_and_keeps_state():
    dates = pd.date_range("2020-01-01", periods=6, freq="B")
    frame = pd.DataFrame(
        {
            "underlying_close": [100.0, 120.0, 119.0, 90.0, 110.0, 109.0],
            "ma_2": [np.nan, 100.0, 100.0, 100.0, 100.0, 100.0],
            "trade_date": dates + pd.offsets.BDay(1),
            "exit_date": dates + pd.offsets.BDay(2),
            "leveraged_return": [0.01, 0.20, 0.03, -0.20, 0.10, 0.02],
            "underlying_return": [0.01, 0.05, 0.01, -0.05, 0.03, 0.01],
        },
        index=dates,
    )
    profile = TrendClockProfile("test", (2,), entry_buffer=0.08)
    replay = replay_trend_clock(frame, profile, cost_bps=10.0)

    assert replay["leveraged_weight"].tolist() == [0.0, 1.0, 1.0, 0.0, 1.0, 1.0]
    assert replay["action"].tolist() == [False, True, False, True, True, False]
    assert replay.loc[1, "gross_return"] == pytest.approx(0.20)
    assert replay.loc[3, "gross_return"] == pytest.approx(-0.05)


def test_direct_target_weight_strategy_clips_and_scales_targets():
    strategy = object.__new__(DirectTargetWeightStrategy)
    strategy.max_weight = 1.0
    strategy.risk_degree = 1.0
    score = pd.Series(
        [0.8, 0.8, -0.2],
        index=pd.Index(["SOXL", "SOXX", "CASH"], name="instrument"),
    )
    target = strategy.generate_target_weight_position(score, None, None, None)

    assert target["SOXL"] == pytest.approx(0.5)
    assert target["SOXX"] == pytest.approx(0.5)
    assert target["CASH"] == 0.0


def test_generated_card_is_materially_above_soxl_but_fail_closed():
    summary = json.loads((CARD / "summary.json").read_text(encoding="utf-8"))
    outer = summary["outer_soxl_comparison"]
    overlap = summary["stage1_overlap_comparison"]

    assert summary["decision"] == (
        "not_promoted_outer_contaminated_and_cross_asset_gate_failed"
    )
    assert summary["production_change"] == "blocked"
    assert summary["selection_uses_outer_test"] is True
    assert summary["cross_asset_outer_gate_pass"] is False
    assert summary["cross_asset_outer_pass_count"] == 0
    assert outer["return_delta_vs_leveraged_pct_points"] > 100.0
    assert outer["action_days"] <= 20
    assert outer["max_drawdown_pct"] > outer["leveraged_buy_hold_max_drawdown_pct"]
    assert overlap["trend_clock"]["return_pct"] > overlap["stage1"]["return_pct"]
    assert summary["artifact_policy"]["policy"] == "slim_card_recorder_first"
    native_outer = next(
        row
        for row in summary["qlib_native_reconciliation"]
        if row["segment"] == "outer_reused_2023_2026"
    )
    assert native_outer["qlib_native_excess_vs_benchmark_pct_points"] > 100.0
    assert native_outer["qlib_native_max_drawdown_pct"] > -75.0
    assert abs(native_outer["qlib_minus_manual_return_pct_points"]) < 100.0


def test_card_is_slim_and_recorder_has_native_outputs():
    forbidden = {
        "predictions.csv",
        "signals.csv",
        "portfolio_report.csv",
        "qlib_portfolio_report.csv",
        "qlib_positions.csv",
        "all_model_predictions.csv",
    }
    assert not forbidden.intersection(path.name for path in CARD.iterdir())
    summary = json.loads((CARD / "summary.json").read_text(encoding="utf-8"))
    recorder = Path(summary["recorder_local_dir"]) / "artifacts"
    for relative in (
        "pred.pkl",
        "label.pkl",
        "sig_analysis/ic.pkl",
        "portfolio_analysis/report_normal_1day.pkl",
        "portfolio_analysis/positions_normal_1day.pkl",
        "portfolio_analysis/port_analysis_1day.pkl",
        "soxl_trend_clock_core_v1/portfolio_report.csv",
        "soxl_trend_clock_core_v1/target_weights.csv",
    ):
        assert (recorder / relative).exists(), relative


def test_notebook_is_executable_qlib_first_with_chinese_conclusion():
    notebook = json.loads(
        (CARD / "soxl_trend_clock_core_v1.ipynb").read_text(encoding="utf-8")
    )
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    for token in (
        "qlib.init",
        "D.calendar",
        "D.features",
        "DatasetH",
        "R",
        "run_soxl_trend_clock_core_card",
        "中文结论",
    ):
        assert token in source
    assert tuple(PROFILES[3].periods) == (140, 150, 160)
