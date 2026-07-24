from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import torch

from src.workflow.stateful_direct_allocation import (
    EXPERIMENT,
    StatefulDirectAllocationNet,
    select_stateful_development_candidate,
)


ROOT = Path(__file__).resolve().parents[1]


def test_stateful_net_emits_one_logit_triplet_per_day():
    model = StatefulDirectAllocationNet(
        primary_feature_count=5,
        common_feature_count=3,
        source_tail_units=2,
        source_hidden_units=4,
        state_feature_count=6,
        recurrent_units=3,
    )
    logits, trade_logits, hidden = model.forward_sequence(
        torch.zeros(7, 5),
        torch.zeros(7, 3),
        torch.zeros(7, 6),
    )

    assert logits.shape == (7, 3)
    assert trade_logits.shape == (7,)
    assert hidden.shape == (1, 1, 3)


def test_stateful_selection_excludes_outer_and_falls_back_honestly():
    rows = [
        {
            "candidate": "dev_gate",
            "segment": "development_walk_forward",
            "selection_uses_outer_test": False,
            "development_gate_pass": True,
            "risk_window_pass_count": 7,
            "positive_relative_window_count": 5,
            "relative_ratio_p25": 1.01,
            "terminal_wealth": 2.0,
            "open_turnover": 20.0,
        },
        {
            "candidate": "outer_leak",
            "segment": "outer_diagnostic",
            "selection_uses_outer_test": True,
            "development_gate_pass": True,
            "risk_window_pass_count": 7,
            "positive_relative_window_count": 7,
            "relative_ratio_p25": 9.0,
            "terminal_wealth": 99.0,
            "open_turnover": 1.0,
        },
    ]
    selected = select_stateful_development_candidate(pd.DataFrame(rows))
    assert selected["candidate"] == "dev_gate"

    rows[0]["development_gate_pass"] = False
    fallback = select_stateful_development_candidate(pd.DataFrame(rows))
    assert fallback["candidate"] == "dev_gate"


def test_stateful_selection_requires_development_rows():
    with pytest.raises(ValueError):
        select_stateful_development_candidate(
            pd.DataFrame(
                [
                    {
                        "segment": "outer_diagnostic",
                        "selection_uses_outer_test": True,
                        "development_gate_pass": True,
                        "risk_window_pass_count": 7,
                        "positive_relative_window_count": 7,
                        "relative_ratio_p25": 2.0,
                        "terminal_wealth": 10.0,
                        "open_turnover": 1.0,
                    }
                ]
            )
        )


def test_stateful_card_is_slim_and_recorder_first():
    card = ROOT / "research" / EXPERIMENT
    if not (card / "summary.json").exists():
        pytest.skip("stateful direct research card has not been executed")
    summary = json.loads(
        (card / "summary.json").read_text(encoding="utf-8")
    )
    notebook = (card / f"{EXPERIMENT}.ipynb").read_text(
        encoding="utf-8"
    )

    assert summary["selection_uses_outer_test"] is False
    assert summary["method"]["external_cap_wait_ramp"] is False
    assert summary["method"]["base_plus_overlay"] is False
    assert (
        summary["artifact_policy"]["policy"]
        == "slim_card_recorder_first"
    )
    assert summary["recorder_id"] in notebook
    artifact_dir = (
        Path(summary["recorder_local_dir"]) / "artifacts" / EXPERIMENT
    )
    for name in (
        "positions.csv",
        "strategy_returns.csv",
        "predictions.csv",
        "teacher_training_frame.csv",
        "model_selection_full.csv",
        "dataset_audit.json",
    ):
        assert (artifact_dir / name).exists(), name
        assert not (card / name).exists(), name
