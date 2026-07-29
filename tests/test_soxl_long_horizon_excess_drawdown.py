from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.workflow.soxl_long_horizon_excess_drawdown import (
    build_horizon_windows,
    build_underwater_episodes,
)


ROOT = Path(__file__).resolve().parents[1]
CARD = ROOT / "research" / "soxl_long_horizon_excess_drawdown_v1"


def _synthetic_frame() -> pd.DataFrame:
    index = pd.bdate_range("2024-01-02", periods=6, name="date")
    frame = pd.DataFrame(
        {
            "soxx_close": [100.0, 101.0, 102.01, 101.0, 103.0, 104.0],
            "soxl_close": [100.0, 103.0, 106.09, 100.0, 106.0, 109.0],
            "vix": [15.0, 14.5, 14.0, 18.0, 16.0, 15.0],
        },
        index=index,
    )
    frame["soxl_return"] = frame["soxl_close"].pct_change()
    frame["soxx_return"] = frame["soxx_close"].pct_change()
    frame["ideal_3x_return"] = 3.0 * frame["soxx_return"]
    frame["soxl_wealth"] = frame["soxl_close"] / frame["soxl_close"].iloc[0]
    frame["soxx_wealth"] = frame["soxx_close"] / frame["soxx_close"].iloc[0]
    frame["ideal_3x_wealth"] = (
        1.0 + frame["ideal_3x_return"].fillna(0.0)
    ).cumprod()
    frame["relative_wealth_vs_soxx"] = (
        frame["soxl_wealth"] / frame["soxx_wealth"]
    )
    frame["relative_wealth_vs_ideal_3x"] = (
        frame["soxl_wealth"] / frame["ideal_3x_wealth"]
    )
    return frame


def test_horizon_windows_use_daily_reset_ideal_path():
    windows = build_horizon_windows(_synthetic_frame(), horizons=(2,))
    first = windows.iloc[0]
    assert np.isclose(first["soxx_return"], 0.0201)
    assert np.isclose(first["soxl_return"], 0.0609)
    assert np.isclose(first["ideal_daily_reset_3x_return"], 0.0609)
    assert np.isclose(first["tracking_gap_vs_ideal"], 0.0)


def test_underwater_episode_records_depth_and_recovery():
    frame = _synthetic_frame()
    episodes = build_underwater_episodes(frame)
    absolute = episodes.loc[
        episodes["basis"].eq("absolute_soxl")
        & episodes["peak_date"].eq("2024-01-04")
    ].iloc[0]
    assert absolute["trough_date"] == "2024-01-05"
    assert absolute["recovery_or_end_date"] == "2024-01-09"
    assert bool(absolute["recovered"]) is True
    assert np.isclose(absolute["depth"], 100.0 / 106.09 - 1.0)


def test_card_contract_is_slim_and_qlib_native():
    summary = json.loads((CARD / "summary.json").read_text(encoding="utf-8"))
    notebook = (
        CARD / "soxl_long_horizon_excess_drawdown_v1.ipynb"
    ).read_text(encoding="utf-8")
    assert summary["experiment"] == "soxl_long_horizon_excess_drawdown_v1"
    assert summary["promotion"] == "blocked_descriptive_not_predictive"
    assert summary["data_audit"]["source_api"] == [
        "qlib.init",
        "qlib.data.D.calendar",
        "qlib.data.D.instruments",
        "qlib.data.D.features",
    ]
    assert summary["artifact_policy"]["policy"] == "slim_card_recorder_first"
    assert summary["dataset_audit"]["api"] == [
        "DatasetH",
        "DataHandlerLP",
        "StaticDataLoader",
    ]
    for sidecar in summary["sidecars"].values():
        assert (CARD / sidecar).exists(), sidecar
    for forbidden in (
        "daily_horizon_windows.csv",
        "persistent_regimes_full.csv",
        "underwater_episodes_full.csv",
    ):
        assert not (CARD / forbidden).exists(), forbidden
    for required in (
        "qlib.init",
        "D.calendar",
        "D.features",
        "DatasetH",
        "R",
    ):
        assert required in notebook
