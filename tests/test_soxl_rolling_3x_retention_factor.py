from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.workflow.soxl_rolling_3x_retention_factor import (
    build_retention_factor_panel,
)


ROOT = Path(__file__).resolve().parents[1]
CARD = ROOT / "research" / "soxl_rolling_3x_retention_factor_v1"


def _frame() -> pd.DataFrame:
    index = pd.bdate_range("2023-01-02", periods=400, name="date")
    base_return = np.tile([0.01, -0.008, 0.006, -0.004], 100)
    soxx = 100.0 * np.cumprod(1.0 + base_return)
    ideal_return = 3.0 * pd.Series(base_return, index=index)
    ideal_wealth = (1.0 + ideal_return).cumprod()
    tracking_return = ideal_return - 0.0001
    soxl = 100.0 * (1.0 + tracking_return).cumprod()
    frame = pd.DataFrame(
        {
            "soxx_close": soxx,
            "soxl_close": soxl,
            "vix": np.linspace(15.0, 20.0, len(index)),
            "soxx_return": pd.Series(soxx, index=index).pct_change(),
            "ideal_3x_wealth": ideal_wealth,
        },
        index=index,
    )
    return frame


def test_retention_decomposition_and_endpoint_gap_identity():
    panel = build_retention_factor_panel(_frame())
    row = panel.dropna(subset=["actual_3x_residual_log_21"]).iloc[-1]
    assert np.isclose(
        row["actual_3x_residual_log_21"],
        row["ideal_path_drag_log_21"] + row["product_tracking_log_21"],
    )
    assert np.isclose(
        row["endpoint_3x_gap_21"],
        row["ideal_endpoint_path_gap_21"]
        + row["product_tracking_gap_21"],
    )
    assert np.isclose(
        row["continuous_3x_retention_ratio_21"],
        np.exp(row["actual_3x_residual_log_21"]),
    )


def test_raw_multiple_is_unstable_near_zero_underlying_return():
    panel = build_retention_factor_panel(_frame())
    near_zero = panel["soxx_return_21"].abs().nsmallest(1).index[0]
    underlying = panel.loc[near_zero, "soxx_return_21"]
    multiple = panel.loc[near_zero, "effective_multiple_21"]
    assert abs(underlying) < 0.02
    assert abs(multiple - 3.0) > 0.40
    assert np.isclose(
        multiple - 3.0,
        panel.loc[near_zero, "endpoint_3x_gap_21"] / underlying,
    )


def test_card_contract_is_qlib_native_and_slim():
    summary = json.loads((CARD / "summary.json").read_text(encoding="utf-8"))
    notebook = (
        CARD / "soxl_rolling_3x_retention_factor_v1.ipynb"
    ).read_text(encoding="utf-8")
    assert summary["experiment"] == "soxl_rolling_3x_retention_factor_v1"
    assert summary["promotion"] == "blocked_not_stable_incremental"
    assert summary["artifact_policy"]["policy"] == "slim_card_recorder_first"
    assert summary["dataset_audit"]["api"] == [
        "DatasetH",
        "DataHandlerLP",
        "StaticDataLoader",
    ]
    assert all(
        not row["incremental_alpha_gate_pass"]
        for row in summary["factor_gate"]["gate"]
    )
    for sidecar in summary["sidecars"].values():
        assert (CARD / sidecar).exists(), sidecar
    for forbidden in ("factor_panel.csv", "walk_forward_predictions.csv"):
        assert not (CARD / forbidden).exists(), forbidden
    for required in (
        "qlib.init",
        "D.features",
        "DatasetH",
        "R",
    ):
        assert required in notebook
