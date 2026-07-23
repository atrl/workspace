from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.workflow.direct_continuous_soxx_allocation import (
    DIRECT_FEATURES,
    PROFILES,
    _project_weights_tensor,
)


ROOT = Path(__file__).resolve().parents[1]
CARD = ROOT / "research" / "direct_continuous_soxx_allocation_v1"


def test_direct_projection_emits_continuous_simplex_weights() -> None:
    logits = torch.tensor(
        [[4.0, 0.0, -1.0], [-1.0, 2.0, 1.0]], dtype=torch.float32
    )
    weights = _project_weights_tensor(logits, maximum_soxl_weight=0.30)
    values = weights.detach().numpy()
    assert np.all(values >= 0.0)
    assert np.allclose(values.sum(axis=1), 1.0)
    assert np.all(values[:, 0] <= 0.30 + 1e-6)
    assert not np.allclose(values[0], values[1])


def test_direct_features_exclude_prior_and_rule_outputs() -> None:
    forbidden = {
        "base_constrained_ref",
        "prior_share",
        "prior_weight",
        "cash_rule",
    }
    assert forbidden.isdisjoint(DIRECT_FEATURES)
    assert len(PROFILES) >= 3


def test_direct_card_contract_after_execution() -> None:
    summary_path = CARD / "summary.json"
    if not summary_path.exists():
        return
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["architecture"]["base_position"] is None
    assert summary["architecture"]["explicit_prior"] is None
    assert summary["architecture"]["candidate_weight_grid"] is False
    assert summary["development"]["selection_uses_outer_test"] is False
    assert summary["artifact_policy"]["policy"] == "slim_card_recorder_first"
    root_names = {path.name for path in CARD.iterdir() if path.is_file()}
    assert "predictions.csv" not in root_names
    assert "positions.csv" not in root_names
    assert "strategy_returns.csv" not in root_names


def test_direct_objective_audit_is_honest() -> None:
    audit_path = CARD / "objective_gate_audit.csv"
    if not audit_path.exists():
        return
    audit = pd.read_csv(audit_path)
    required = audit.loc[audit["required"]]
    summary = json.loads((CARD / "summary.json").read_text(encoding="utf-8"))
    assert summary["objective_pass"] == bool(required["passed"].all())
    assert set(
        audit.loc[
            audit["gate"].isin(
                ["no_base_or_prior", "no_candidate_weight_grid"]
            ),
            "passed",
        ]
    ) == {True}
