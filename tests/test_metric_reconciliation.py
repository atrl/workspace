from __future__ import annotations

import pandas as pd

from src.soxl_quant.core.metric_reconciliation import (
    audit_source_vs_official_metrics,
    build_warm_start_replay_segment,
    compute_relative_metric_variants,
    summarize_source_vs_official_audit,
    slice_warm_start_daily_path,
)


def test_relative_metric_variants_detect_warmup_gate_flip() -> None:
    dates = pd.bdate_range("2024-01-02", periods=5)
    prices = pd.Series([100.0, 200.0, 200.0, 100.0, 100.0], index=dates)

    metrics = compute_relative_metric_variants(
        strategy_return=-5.0,
        price_series=prices,
        requested_seg=("2024-01-02", "2024-01-08"),
        effective_seg=("2024-01-02", "2024-01-08"),
        lookback=2,
        target_relative_return=30.0,
    )

    assert metrics["calendar_buy_and_hold_return"] == 0.0
    assert metrics["tradable_buy_and_hold_return"] == -50.0
    assert metrics["calendar_relative_return"] == -5.0
    assert metrics["tradable_relative_return"] == 45.0
    assert metrics["first_tradable_date"] == "2024-01-04"
    assert metrics["warmup_days"] == 2
    assert metrics["gate_flip"] is True
    assert metrics["mismatch_reason"] == "warmup_benchmark_alignment_gate_flip"


def test_build_warm_start_replay_segment_uses_prior_trading_rows_and_clamps() -> None:
    dates = pd.bdate_range("2024-01-02", periods=10)

    assert build_warm_start_replay_segment(
        dates,
        eval_seg=("2024-01-08", "2024-01-12"),
        lookback=3,
    ) == ("2024-01-03", "2024-01-12")

    assert build_warm_start_replay_segment(
        dates,
        eval_seg=("2024-01-03", "2024-01-12"),
        lookback=60,
    ) == ("2024-01-02", "2024-01-12")


def test_slice_warm_start_daily_path_rebases_only_eval_window() -> None:
    dates = pd.bdate_range("2024-01-02", periods=5)
    portfolio = pd.DataFrame(
        {"account": [950_000.0, 900_000.0, 1_000_000.0, 1_100_000.0, 1_050_000.0]},
        index=dates,
    )
    prices = pd.Series([100.0, 120.0, 80.0, 100.0, 120.0], index=dates)

    daily = slice_warm_start_daily_path(
        portfolio_df=portfolio,
        price_series=prices,
        eval_seg=("2024-01-04", "2024-01-08"),
        position_history=[0.0, 0.1, 0.2, 0.3, 0.4],
        trade_signals=[
            (pd.Timestamp("2024-01-03"), 120.0, 0.1, "buy"),
            (pd.Timestamp("2024-01-05"), 100.0, 0.3, "sell"),
        ],
        account_base=1_000_000.0,
    )

    assert daily["datetime"].tolist() == ["2024-01-04", "2024-01-05", "2024-01-08"]
    assert daily["strategy_nav"].tolist() == [1_000_000.0, 1_100_000.0, 1_050_000.0]
    assert daily["buyhold_nav"].tolist() == [1_000_000.0, 1_250_000.0, 1_500_000.0]
    assert daily["position"].tolist() == [0.2, 0.3, 0.4]
    assert daily["action"].tolist() == ["HOLD", "SELL", "HOLD"]
    assert daily["excess"].iloc[-1] == -450_000.0
    assert daily["excess_pct"].iloc[-1] == -30.0


def test_source_vs_official_audit_classifies_gate_flip_failure_modes() -> None:
    pk = pd.DataFrame(
        [
            {
                "candidate": "bc_initialization",
                "fold": "wf_2022",
                "source_bc_relative_return": 66.72,
                "official_warm_relative_return": 67.62,
                "source_bc_max_dd": 14.99,
                "official_warm_max_dd": 22.26,
            },
            {
                "candidate": "bc_initialization",
                "fold": "wf_2023_2024",
                "source_bc_relative_return": 31.12,
                "official_warm_relative_return": 5.02,
                "source_bc_max_dd": 13.36,
                "official_warm_max_dd": 16.69,
            },
        ]
    )

    audit = audit_source_vs_official_metrics(pk)
    summary = summarize_source_vs_official_audit(audit)

    by_fold = audit.set_index("fold")
    assert bool(by_fold.loc["wf_2022", "source_optimistic_gate_flip"]) is True
    assert by_fold.loc["wf_2022", "primary_mismatch"] == "max_drawdown_understated_gate_flip"
    assert by_fold.loc["wf_2023_2024", "primary_mismatch"] == "relative_and_drawdown_gate_flip"
    assert by_fold.loc["wf_2023_2024", "relative_overstatement_pp"] == 26.10
    assert by_fold.loc["wf_2022", "maxdd_understatement_pp"] == 7.27
    assert summary["decision"] == "source_backtest_optimistic_gate_flip_no_training"
    assert summary["source_optimistic_gate_flip_count"] == 2
    assert summary["next_mechanism"] == "replace_training_selection_gate_with_official_replay_metrics"
