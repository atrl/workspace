from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.workflow.stage1_elastic_soxl_overflow import (
    OverflowCandidate,
    _apply_overlay,
)


ROOT = Path(__file__).resolve().parents[1]
CARD = ROOT / "research" / "stage1_elastic_soxl_overflow_v1"


def test_overflow_moves_only_soxx_into_soxl_and_preserves_cash():
    dates = pd.to_datetime(["2022-01-03", "2022-01-04"])
    frame = pd.DataFrame(
        {
            "signal_date": dates,
            "trade_date": dates + pd.offsets.BDay(1),
            "exit_date": dates + pd.offsets.BDay(2),
            "year": [2022, 2022],
            "segment": ["development_walk_forward"] * 2,
            "cap_bind": [True, False],
            "open_soxl_weight": [0.30, 0.25],
            "open_soxx_weight": [0.50, 0.55],
            "open_cash_weight": [0.20, 0.20],
            "incremental_spread": [0.02, -0.01],
            "strategy_return": [0.01, -0.01],
        }
    )
    result = _apply_overlay(
        frame,
        probability=pd.Series([0.9, 0.9], index=dates),
        candidate=OverflowCandidate(
            "logistic_065_010", "logistic", 0.10, 0.65
        ),
        cost_bps=10.0,
    )

    assert np.isclose(result.loc[0, "target_soxl_weight"], 0.40)
    assert np.isclose(result.loc[0, "target_soxx_weight"], 0.40)
    assert np.isclose(result.loc[0, "target_cash_weight"], 0.20)
    assert result.loc[1, "overflow_weight"] == 0.0
    assert result["target_soxl_weight"].le(0.40).all()
    assert np.allclose(
        result[
            [
                "target_soxl_weight",
                "target_soxx_weight",
                "target_cash_weight",
            ]
        ].sum(axis=1),
        1.0,
    )


def test_stage1_elastic_overflow_card_is_frozen_and_fail_closed():
    summary = json.loads(
        (CARD / "summary.json").read_text(encoding="utf-8")
    )
    selected = json.loads(
        (CARD / "selected_model.json").read_text(encoding="utf-8")
    )
    notebook_text = (
        CARD / "stage1_elastic_soxl_overflow_v1.ipynb"
    ).read_text(encoding="utf-8")

    assert summary["experiment"] == "stage1_elastic_soxl_overflow_v1"
    assert summary["source_stage1"]["frozen"] is True
    assert summary["source_stage1"]["training_end"] == "2022-12-30"
    assert summary["promotion"] == "blocked"
    assert summary["candidate_search"]["candidate_count"] == 8
    assert summary["dataset_audit"]["api"] == [
        "DatasetH",
        "DataHandlerLP",
        "StaticDataLoader",
    ]
    assert selected["selection_uses_outer_test"] is False
    assert (
        selected["outer_role"]
        == "reused_diagnostic_only_not_model_selection"
    )
    assert selected["overflow_weight"] <= 0.10
    assert summary["timing"]["cash_unchanged"] is True
    assert summary["overfit_audit"]["outer_is_selection_input"] is False
    assert summary["overfit_audit"]["clean_unseen_holdout_available"] is False
    assert summary["artifact_policy"]["policy"] == (
        "slim_card_recorder_first"
    )
    for forbidden in (
        "predictions.csv",
        "positions.csv",
        "portfolio_report.csv",
        "all_model_predictions.csv",
    ):
        assert not (CARD / forbidden).exists(), forbidden
    for sidecar in summary["sidecars"].values():
        assert (CARD / sidecar).exists(), sidecar
    assert "qlib.init" in notebook_text
    assert "D.features" in notebook_text
    assert "DatasetH" in notebook_text
    assert "R" in notebook_text
    assert "run_stage1_elastic_soxl_overflow_card" in notebook_text
