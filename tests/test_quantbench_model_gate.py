from __future__ import annotations

from pathlib import Path

import pandas as pd


def _write_daily(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_compute_backtest_metrics_compares_strategy_to_buyhold_and_drawdown():
    from src.soxl_quant.app.quantbench_gates import compute_backtest_metrics

    frame = pd.DataFrame(
        [
            {"datetime": "2022-01-03", "strategy_nav": 100.0, "buyhold_nav": 100.0},
            {"datetime": "2022-01-04", "strategy_nav": 90.0, "buyhold_nav": 80.0},
            {"datetime": "2022-01-05", "strategy_nav": 130.0, "buyhold_nav": 120.0},
        ]
    )

    metrics = compute_backtest_metrics(frame, window="test")

    assert metrics.window == "test"
    assert metrics.rows == 3
    assert metrics.strategy_return_pct == 30.0
    assert metrics.buyhold_return_pct == 20.0
    assert metrics.relative_return_pct == 10.0
    assert metrics.max_drawdown_pct == 10.0
    assert metrics.gate_pass is True


def test_compute_backtest_metrics_blocks_equal_buyhold_and_large_drawdown():
    from src.soxl_quant.app.quantbench_gates import compute_backtest_metrics

    frame = pd.DataFrame(
        [
            {"datetime": "2022-01-03", "strategy_nav": 100.0, "buyhold_nav": 100.0},
            {"datetime": "2022-01-04", "strategy_nav": 70.0, "buyhold_nav": 90.0},
            {"datetime": "2022-01-05", "strategy_nav": 110.0, "buyhold_nav": 120.0},
        ]
    )

    metrics = compute_backtest_metrics(frame, window="test")

    assert metrics.strategy_return_pct < metrics.buyhold_return_pct
    assert metrics.max_drawdown_pct == 30.0
    assert metrics.gate_pass is False
    assert "relative_return_not_positive" in metrics.failures
    assert "max_drawdown_ge_20" in metrics.failures


def test_scan_model_replays_uses_regime_windows_not_only_full_history(tmp_path: Path):
    from src.soxl_quant.app.quantbench_gates import scan_model_replays

    good_path = tmp_path / "runtime" / "model" / "card" / "good" / "good_soxl_daily.csv"
    hidden_bad_path = tmp_path / "runtime" / "model" / "card" / "hidden_bad" / "hidden_bad_soxl_daily.csv"
    _write_daily(
        good_path,
        [
            {"datetime": "2022-01-03", "strategy_nav": 100.0, "buyhold_nav": 100.0},
            {"datetime": "2022-06-01", "strategy_nav": 98.0, "buyhold_nav": 90.0},
            {"datetime": "2022-12-30", "strategy_nav": 130.0, "buyhold_nav": 110.0},
            {"datetime": "2023-01-03", "strategy_nav": 130.0, "buyhold_nav": 110.0},
            {"datetime": "2024-12-31", "strategy_nav": 170.0, "buyhold_nav": 140.0},
        ],
    )
    _write_daily(
        hidden_bad_path,
        [
            {"datetime": "2022-01-03", "strategy_nav": 100.0, "buyhold_nav": 100.0},
            {"datetime": "2022-06-01", "strategy_nav": 60.0, "buyhold_nav": 90.0},
            {"datetime": "2022-12-30", "strategy_nav": 180.0, "buyhold_nav": 110.0},
            {"datetime": "2023-01-03", "strategy_nav": 180.0, "buyhold_nav": 110.0},
            {"datetime": "2024-12-31", "strategy_nav": 220.0, "buyhold_nav": 140.0},
        ],
    )

    result = scan_model_replays(
        root=tmp_path,
        required_windows=("test_2022", "test_2023_2024"),
        max_drawdown_pct=20.0,
    )

    assert result["summary"]["profiles_scanned"] == 2
    assert result["summary"]["passing_profiles"] == 1
    assert result["passing_profiles"] == ["good"]
    hidden_2022 = [
        row for row in result["rows"] if row["profile"] == "hidden_bad" and row["window"] == "test_2022"
    ][0]
    assert hidden_2022["gate_pass"] is False
    assert "max_drawdown_ge_20" in hidden_2022["failures"]
