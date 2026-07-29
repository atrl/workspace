from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.workflow.soxl_3x_state_factor_lab import (
    TARGET_HORIZON,
    build_state_factor_panel,
)


ROOT = Path(__file__).resolve().parents[1]
CARD = ROOT / "research" / "soxl_3x_state_factor_lab_v1"


def _frame() -> pd.DataFrame:
    index = pd.bdate_range("2020-01-02", periods=500, name="date")
    underlying_return = np.tile(
        [0.01, 0.006, -0.002, 0.008, -0.001],
        100,
    )
    soxx = 100.0 * np.cumprod(1.0 + underlying_return)
    ideal_return = 3.0 * pd.Series(underlying_return, index=index)
    ideal_wealth = (1.0 + ideal_return).cumprod()
    soxl = 100.0 * ideal_wealth
    return pd.DataFrame(
        {
            "soxx_close": soxx,
            "soxl_close": soxl,
            "vix": np.linspace(15.0, 24.0, len(index)),
            "soxx_return": pd.Series(
                soxx,
                index=index,
            ).pct_change(),
            "ideal_3x_wealth": ideal_wealth,
        },
        index=index,
    )


def test_state_factors_are_trailing_and_forward_label_is_complete():
    panel = build_state_factor_panel(_frame())
    assert panel["signed_path_efficiency_21"].dropna().between(
        -1.0,
        1.0,
    ).all()
    assert (
        panel["future_soxl_worst_return_21"]
        .tail(TARGET_HORIZON)
        .isna()
        .all()
    )
    assert panel["future_relative_excess_21"].tail(
        TARGET_HORIZON
    ).isna().all()


def test_exact_daily_reset_path_has_zero_product_tracking_volatility():
    panel = build_state_factor_panel(_frame())
    tracking_volatility = panel["soxl_tracking_error_vol_21"].dropna()
    assert np.allclose(tracking_volatility, 0.0, atol=1e-12)


def test_card_contract_is_qlib_native_slim_and_outer_safe():
    summary = json.loads(
        (CARD / "summary.json").read_text(encoding="utf-8")
    )
    notebook = (
        CARD / "soxl_3x_state_factor_lab_v1.ipynb"
    ).read_text(encoding="utf-8")
    model_comparison = pd.read_csv(CARD / "model_comparison.csv")

    assert summary["experiment"] == "soxl_3x_state_factor_lab_v1"
    assert summary["selection_uses_outer"] is False
    assert summary["promotion"] == "blocked_pending_new_forward_paper"
    assert summary["artifact_policy"]["policy"] == (
        "slim_card_recorder_first"
    )
    assert summary["dataset_audit"]["api"] == [
        "DatasetH",
        "DataHandlerLP",
        "StaticDataLoader",
    ]
    assert model_comparison["purged_rows"].eq(TARGET_HORIZON).all()
    for sidecar in summary["sidecars"].values():
        assert (CARD / sidecar).exists(), sidecar
    for forbidden in ("factor_panel.csv", "walk_forward_predictions.csv"):
        assert not (CARD / forbidden).exists(), forbidden
    for required in ("qlib.init", "D.features", "DatasetH", "R"):
        assert required in notebook
