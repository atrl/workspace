from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.workflow.multitask_classification_allocation import (
    BASE_FEATURES,
    CANDIDATES,
    LABEL_COLUMNS,
    _attach_multitask_targets,
    _factorized_weights,
)


ROOT = Path(__file__).resolve().parents[1]
CARD = ROOT / "research" / "multitask_classification_allocation_v1"


def test_factorized_weights_are_continuous_and_bounded() -> None:
    logits = torch.tensor(
        [[-3.0, -2.0], [0.0, 0.0], [3.0, 2.0]],
        dtype=torch.float32,
    )
    weights = _factorized_weights(logits, maximum_soxl_weight=0.30)
    values = weights.detach().numpy()
    assert np.all(values >= 0.0)
    assert np.allclose(values.sum(axis=1), 1.0)
    assert np.all(values[:, 0] <= 0.30)
    assert len(np.unique(values[:, 0])) == 3


def test_multitask_targets_have_causal_feature_inputs() -> None:
    index = pd.date_range("2020-01-01", periods=30, freq="B")
    frame = pd.DataFrame(index=index)
    for feature in BASE_FEATURES:
        frame[feature] = np.linspace(0.0, 1.0, len(frame))
    frame["soxl_forward_open_return"] = 0.01
    frame["soxx_forward_open_return"] = 0.004
    enriched, base, fast = _attach_multitask_targets(frame)
    assert set(LABEL_COLUMNS).issubset(enriched.columns)
    assert len(base) == len(BASE_FEATURES)
    assert len(fast) > len(base)
    forbidden = {
        "base_constrained_ref",
        "prior_share",
        "prior_weight",
        "cash_rule",
    }
    assert forbidden.isdisjoint(fast)


def test_candidate_ablation_covers_labels_features_and_sequence() -> None:
    names = {candidate.name for candidate in CANDIDATES}
    assert "tabular_base_allocation_only" in names
    assert "tabular_base_multitask" in names
    assert "tabular_fast_multitask" in names
    assert "gru_fast_multitask" in names


def test_multitask_card_contract_after_execution() -> None:
    summary_path = CARD / "summary.json"
    if not summary_path.exists():
        return
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    architecture = summary["architecture"]
    assert architecture["base_position"] is None
    assert architecture["explicit_prior"] is None
    assert architecture["candidate_weight_grid"] is False
    assert architecture["final_weights_are_direct_model_outputs"] is True
    assert summary["development"]["selection_uses_outer_test"] is False
    assert summary["artifact_policy"]["policy"] == (
        "slim_card_recorder_first"
    )
    root_names = {path.name for path in CARD.iterdir() if path.is_file()}
    assert "predictions.csv" not in root_names
    assert "positions.csv" not in root_names
    assert "strategy_returns.csv" not in root_names


def test_multitask_objective_audit_is_honest() -> None:
    audit_path = CARD / "objective_gate_audit.csv"
    if not audit_path.exists():
        return
    audit = pd.read_csv(audit_path)
    required = audit.loc[audit["required"]]
    summary = json.loads((CARD / "summary.json").read_text(encoding="utf-8"))
    assert summary["objective_pass"] == bool(required["passed"].all())
