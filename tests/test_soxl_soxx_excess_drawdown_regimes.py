from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.workflow.soxl_soxx_excess_drawdown_regimes import (
    _drawdown_band,
    build_forward_observations,
)


ROOT = Path(__file__).resolve().parents[1]
CARD = ROOT / "research" / "soxl_soxx_excess_drawdown_regimes_v1"


def test_drawdown_bands_are_fixed_and_exhaustive():
    drawdown = pd.Series([-0.25, -0.20, -0.15, -0.10, -0.07, -0.05, -0.01, 0.0])
    assert _drawdown_band(drawdown).tolist() == [
        "dd_20plus",
        "dd_20plus",
        "dd_10_20",
        "dd_10_20",
        "dd_5_10",
        "dd_5_10",
        "dd_0_5",
        "dd_0_5",
    ]


def test_forward_observation_uses_next_open_and_beta_matching():
    dates = pd.bdate_range("2022-01-03", periods=5)
    market = pd.DataFrame(
        {
            "soxl_open": [100.0, 100.0, 103.0, 106.09, 109.2727],
            "soxl_close": [100.0] * 5,
            "soxx_open": [100.0, 100.0, 101.0, 102.01, 103.0301],
            "soxx_close": [100.0] * 5,
            "soxx_drawdown": [-0.21] * 5,
            "soxx_return_5d": [0.01] * 5,
            "soxx_return_20d": [0.02] * 5,
            "soxx_realized_vol_20d": [0.20] * 5,
            "drawdown_band": ["dd_20plus"] * 5,
            "phase": ["rebound5"] * 5,
            "regime_phase": ["dd_20plus__rebound5"] * 5,
            "signal_position": range(5),
        },
        index=dates,
    )
    observations = build_forward_observations(
        market, horizons=(2,), cost_bps=10.0, stress_cost_bps=25.0
    )
    first = observations.iloc[0]
    assert first["entry_date"] == dates[1]
    assert first["exit_date"] == dates[3]
    assert np.isclose(first["soxl_return"], 0.0609)
    assert np.isclose(first["soxx_return"], 0.0201)
    assert np.isclose(first["beta_matched_soxl_return"], 0.0201)
    assert np.isclose(first["beta_matched_excess_return"], 0.0)
    assert np.isclose(
        first["raw_excess_net_10bps"],
        first["raw_excess_return"] - 0.002,
    )


def test_card_contract_and_no_root_recorder_artifacts():
    summary = json.loads((CARD / "summary.json").read_text(encoding="utf-8"))
    notebook_text = (
        CARD / "soxl_soxx_excess_drawdown_regimes_v1.ipynb"
    ).read_text(encoding="utf-8")

    assert summary["experiment"] == "soxl_soxx_excess_drawdown_regimes_v1"
    assert summary["promotion"] == "blocked"
    assert summary["regime_design"]["selection_uses_outer_test"] is False
    assert summary["regime_design"]["primary_cell_count"] == 16
    assert summary["overfit_audit"]["outer_is_clean_holdout"] is False
    assert summary["risk_efficient_zone_count"] == 0
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
        "forward_observations.csv",
        "predictions.csv",
        "signals.csv",
        "portfolio_report.csv",
        "qlib_positions.csv",
    ):
        assert not (CARD / forbidden).exists(), forbidden
    for required in ("qlib.init", "D.calendar", "D.features", "DatasetH", "R"):
        assert required in notebook_text
