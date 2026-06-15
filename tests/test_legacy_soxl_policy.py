from __future__ import annotations

import pandas as pd

from soxl_backtest import SOXLBacktest
from src.soxl_quant.policy.legacy_soxl import build_legacy_soxl_feature_frame, replay_legacy_soxl_policy


def _legacy_backtest(start: str, end: str) -> SOXLBacktest:
    return SOXLBacktest(
        {
            "USE_LONGBRIDGE": False,
            "START_DATE": start,
            "END_DATE": end,
        }
    )


def test_legacy_feature_builder_matches_non_qlib_soxl_backtest():
    legacy = _legacy_backtest("2026-01-01", "2026-06-09")
    actual = build_legacy_soxl_feature_frame(start="2026-01-01", end="2026-06-09")

    expected = legacy.df.copy()
    actual = actual.copy()
    expected["date"] = pd.to_datetime(expected["date"])
    actual["date"] = pd.to_datetime(actual["date"])

    merged = actual.set_index("date").join(
        expected.set_index("date"),
        how="inner",
        lsuffix="_actual",
        rsuffix="_expected",
    )
    assert len(merged) == len(expected) == len(actual)
    assert merged.index.min().date().isoformat() == "2026-01-02"
    assert merged.index.max().date().isoformat() == "2026-06-09"

    for column in ["price", "vix", "RSI6", "MA5", "MA20", "HV_20", "HV_30", "VOL_COEF", "panic_score"]:
        pd.testing.assert_series_equal(
            merged[f"{column}_actual"],
            merged[f"{column}_expected"],
            check_names=False,
            check_dtype=False,
            rtol=1e-10,
            atol=1e-10,
        )


def test_legacy_policy_replay_matches_ytd_trade_timing_and_return():
    result = replay_legacy_soxl_policy(start="2026-01-01", end="2026-06-09")

    assert abs(result.metrics["rulebase_return_pct"] - 132.07797795135497) <= 0.05
    assert abs(result.metrics["end_position_pct"] - 15.91) <= 0.05
    assert result.metrics["trade_count"] == 3
    assert set(result.trades["date"].tolist()) == {"2026-03-30", "2026-05-27", "2026-06-05"}
    assert result.trades["type"].tolist() == ["buy", "take_profit", "buy"]


def test_legacy_policy_replay_matches_june_fresh_start():
    result = replay_legacy_soxl_policy(start="2026-06-01", end="2026-06-09")

    assert abs(result.metrics["rulebase_return_pct"] - 3.1389599999999973) <= 0.05
    assert abs(result.metrics["end_position_pct"] - 29.94) <= 0.05
    assert result.metrics["trade_count"] == 1
    assert result.trades["date"].tolist() == ["2026-06-05"]
    assert result.trades["type"].tolist() == ["buy"]


def test_legacy_policy_replay_accepts_risk_parameter_overrides():
    default = replay_legacy_soxl_policy(start="2020-01-01", end="2020-12-31")
    capped = replay_legacy_soxl_policy(
        start="2020-01-01",
        end="2020-12-31",
        config_overrides={
            "max_position": 0.35,
            "min_position": 0.0,
            "buy_step_initial": 0.15,
            "buy_step_normal": 0.05,
        },
    )

    assert capped.trades.iloc[0]["position"] <= 0.16
    assert capped.metrics["average_position_pct"] < default.metrics["average_position_pct"]
    assert capped.metrics["rulebase_max_drawdown_pct"] < default.metrics["rulebase_max_drawdown_pct"]


def test_export_legacy_soxl_policy_replay_writes_sidecars(tmp_path):
    from scripts.export_legacy_soxl_policy_replay import write_legacy_soxl_policy_replay_outputs

    summary = write_legacy_soxl_policy_replay_outputs(tmp_path, end_date="2026-06-09")

    expected_files = {
        "new_policy_daily.csv",
        "new_policy_trades.csv",
        "legacy_comparison_metrics.csv",
        "trade_timing_diagnostics.csv",
        "summary.json",
    }
    assert expected_files == {path.name for path in tmp_path.iterdir()}
    assert summary["windows"]["ytd_2026"]["new_policy"]["trade_count"] == 3
    assert summary["windows"]["june_2026"]["new_policy"]["trade_count"] == 1

    metrics = pd.read_csv(tmp_path / "legacy_comparison_metrics.csv")
    ytd = metrics.loc[metrics["window"] == "ytd_2026"].iloc[0]
    june = metrics.loc[metrics["window"] == "june_2026"].iloc[0]
    assert abs(float(ytd["return_delta_pct"])) <= 0.05
    assert abs(float(june["return_delta_pct"])) <= 0.05

    timing = pd.read_csv(tmp_path / "trade_timing_diagnostics.csv")
    assert timing["date_delta_days"].abs().max() == 0


def test_export_legacy_soxl_policy_replay_defaults_to_latest_local_soxl_date(tmp_path):
    from scripts.export_legacy_soxl_policy_replay import write_legacy_soxl_policy_replay_outputs

    summary = write_legacy_soxl_policy_replay_outputs(tmp_path)
    expected_end = str(pd.read_csv("data/raw/symbols/SOXL.csv")["date"].max())
    daily = pd.read_csv(tmp_path / "new_policy_daily.csv")

    assert daily.loc[daily["window"] == "ytd_2026", "date"].max() == expected_end
    assert daily.loc[daily["window"] == "june_2026", "date"].max() == expected_end
    assert summary["windows"]["ytd_2026"]["new_policy"]["end"] == expected_end
