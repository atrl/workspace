from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.workflow.leveraged_etf_positive_underlying_excess import (
    LeveragedPair,
    _path_efficiency,
    build_pair_observations,
)


ROOT = Path(__file__).resolve().parents[1]
CARD = ROOT / "research" / "leveraged_etf_positive_underlying_excess_v1"


def test_directional_path_efficiency_is_one_and_chop_is_lower():
    assert np.isclose(_path_efficiency(np.array([0.01, 0.02, 0.03])), 1.0)
    choppy = _path_efficiency(np.array([0.10, -0.08, 0.02]))
    assert 0.0 < choppy < 0.30


def test_observation_uses_close_to_close_and_ideal_daily_reset():
    dates = pd.bdate_range("2022-01-03", periods=4)
    underlying = [100.0, 101.0, 102.01, 103.0301]
    leveraged = [100.0, 103.0, 106.09, 109.2727]
    frame = pd.DataFrame(
        {
            "underlying_close": underlying,
            "leveraged_close": leveraged,
            "starting_underlying_return_20d": [0.01] * 4,
            "starting_underlying_vol_20d": [0.20] * 4,
            "signal_position": range(4),
        },
        index=dates,
    )
    pair = LeveragedPair("TEST", "LEV", "BASE", "unused")
    result = build_pair_observations(
        frame,
        pair,
        horizons=(2,),
        switch_cost_per_leg_bps=25.0,
    )
    first = result.iloc[0]
    assert first["signal_date"] == dates[0]
    assert first["exit_date"] == dates[2]
    assert np.isclose(first["underlying_return"], 0.0201)
    assert np.isclose(first["leveraged_return"], 0.0609)
    assert np.isclose(first["ideal_daily_reset_return"], 0.0609)
    assert np.isclose(first["tracking_gap_vs_ideal"], 0.0)
    assert np.isclose(
        first["switch_net_excess_25bps"],
        first["raw_excess_return"] - 0.005,
    )


def test_card_contract_and_thresholds():
    summary = json.loads((CARD / "summary.json").read_text(encoding="utf-8"))
    notebook_text = (
        CARD / "leveraged_etf_positive_underlying_excess_v1.ipynb"
    ).read_text(encoding="utf-8")

    assert summary["experiment"] == "leveraged_etf_positive_underlying_excess_v1"
    assert summary["promotion"] == "blocked_descriptive_not_predictive"
    assert summary["timing"]["comparison"] == "close_to_close"
    assert summary["timing"]["is_ex_ante_signal"] is False
    assert summary["overfit_audit"]["condition_uses_future_information"] is True
    assert summary["overfit_audit"]["outer_is_clean_holdout"] is False
    assert [row["horizon_days"] for row in summary["stable_thresholds"]] == [
        1,
        5,
        21,
        63,
    ]
    assert summary["dataset_audit"]["api"] == [
        "DatasetH",
        "DataHandlerLP",
        "StaticDataLoader",
    ]
    assert (
        summary["artifact_policy"]["policy"]
        == "slim_card_recorder_first"
    )
    for sidecar in summary["sidecars"].values():
        assert (CARD / sidecar).exists(), sidecar
    for forbidden in (
        "forward_windows.csv",
        "predictions.csv",
        "signals.csv",
        "portfolio_report.csv",
    ):
        assert not (CARD / forbidden).exists(), forbidden
    for required in (
        "qlib.init",
        "D.calendar",
        "D.features",
        "DatasetH",
        "R",
    ):
        assert required in notebook_text
