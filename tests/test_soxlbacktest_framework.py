from __future__ import annotations

import json

import pandas as pd

from soxl_backtest.framework import ExperimentSpec, QlibBacktestRunner, load_qlib_frame
from soxl_backtest.reference import replay_soxlbacktest_reference


def test_load_qlib_frame_uses_repo_qlib_calendar_and_canonical_panel():
    frame = load_qlib_frame(start="2026-01-01", end="2026-06-09")

    assert frame.attrs["data_source"] == "qlib_calendar_plus_canonical_panel"
    assert frame.attrs["qlib_calendar_path"].endswith("data/qlib/soxl/calendars/day.txt")
    assert frame["date"].min().date().isoformat() == "2026-01-02"
    assert frame["date"].max().date().isoformat() == "2026-06-09"
    assert {"date", "instrument", "price", "close", "vix"}.issubset(frame.columns)
    assert set(frame["instrument"]) == {"SOXL"}
    assert frame["vix"].isna().sum() == 0


def test_soxlbacktest_reference_matches_known_ytd_and_june_windows():
    ytd = replay_soxlbacktest_reference(start="2026-01-01", end="2026-06-09")
    june = replay_soxlbacktest_reference(start="2026-06-01", end="2026-06-09")

    assert abs(ytd.metrics["rulebase_return_pct"] - 132.07797795135497) <= 0.05
    assert abs(ytd.metrics["end_position_pct"] - 15.91) <= 0.05
    assert ytd.metrics["trade_count"] == 3
    assert ytd.trades["date"].tolist() == ["2026-03-30", "2026-05-27", "2026-06-05"]
    assert ytd.trades["type"].tolist() == ["buy", "take_profit", "buy"]

    assert abs(june.metrics["rulebase_return_pct"] - 3.1389599999999973) <= 0.05
    assert abs(june.metrics["end_position_pct"] - 29.94) <= 0.05
    assert june.metrics["trade_count"] == 1
    assert june.trades["date"].tolist() == ["2026-06-05"]


def test_qlib_backtest_runner_writes_card_local_artifacts(tmp_path):
    spec = ExperimentSpec(
        experiment="soxlbacktest_qlib_reference_v1",
        direction="model_selection",
        start="2026-06-01",
        end="2026-06-09",
        changed_layer="selection",
        role="replace",
        baseline="legacy_soxl_backtest",
        comparison_key="soxlbacktest_qlib_reference_v1:june_2026",
    )
    card_dir = tmp_path / spec.experiment

    result = QlibBacktestRunner(spec).run_reference(card_dir)

    assert result.metrics["trade_count"] == 1
    expected_files = {
        "daily.csv",
        "trades.csv",
        "metrics.json",
        "summary.json",
        "experiment_spec.json",
    }
    assert expected_files == {path.name for path in card_dir.iterdir()}

    summary = json.loads((card_dir / "summary.json").read_text())
    assert summary["experiment"] == spec.experiment
    assert summary["direction"] == "model_selection"
    assert summary["artifacts"]["daily"] == "daily.csv"
    assert summary["artifacts"]["trades"] == "trades.csv"

    daily = pd.read_csv(card_dir / "daily.csv")
    assert daily["date"].min() == "2026-06-01"
    assert daily["date"].max() == "2026-06-09"
