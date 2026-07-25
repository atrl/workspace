from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.workflow.cross_pair_auxiliary_training import (
    CashConstrainedAllocationNet,
    CrossPairCandidate,
    CrossPairTrainingSpec,
    _candidate_asset_weights,
    _fit_candidate,
)


ROOT = Path(__file__).resolve().parents[1]
CARD = ROOT / "research" / "cross_pair_auxiliary_training_v1"


def _panel() -> pd.DataFrame:
    dates = pd.date_range("2014-01-02", periods=900, freq="B")
    rows: list[dict[str, object]] = []
    for instrument in ("SOXL", "TQQQ"):
        for offset, date in enumerate(dates):
            rows.append(
                {
                    "datetime": date,
                    "instrument": instrument,
                    "ret_5_z20": np.sin(offset / 20.0),
                    "ret_20_pctile60": (offset % 60) / 60.0,
                    "target_soxl_weight": 0.15,
                    "target_soxx_weight": 0.65,
                    "target_cash_weight": 0.20,
                }
            )
    return pd.DataFrame(rows)


def test_cash_constrained_net_outputs_legal_weights():
    net = CashConstrainedAllocationNet(feature_count=3, hidden_units=4)
    weights = net(
        torch.zeros((8, 3)),
        maximum_leveraged_weight=0.30,
        minimum_cash_weight=0.15,
    ).detach().numpy()

    assert np.allclose(weights.sum(axis=1), 1.0)
    assert np.all(weights >= 0.0)
    assert np.all(weights[:, 0] <= 0.30)
    assert np.all(weights[:, 2] >= 0.15)


def test_auxiliary_candidate_uses_only_declared_training_assets():
    candidate = CrossPairCandidate("aux", 0.25, 0.02)

    assert _candidate_asset_weights(candidate, "SOXL") == {
        "SOXL": 1.0,
        "TQQQ": 0.25,
    }


def test_fit_candidate_uses_auxiliary_train_but_primary_test_only():
    panel = _panel()
    spec = CrossPairTrainingSpec(root=Path("."), epochs=3)
    train = panel.loc[panel["datetime"].lt("2017-01-01")]
    test = panel.loc[
        panel["datetime"].between("2017-01-02", "2017-02-01")
    ]
    candidate = CrossPairCandidate("aux", 0.25, 0.02)

    prediction, state, diagnostics = _fit_candidate(
        train=train,
        test=test,
        features=["ret_5_z20", "ret_20_pctile60"],
        primary_symbol="SOXL",
        candidate=candidate,
        spec=spec,
        seed=17,
    )

    assert state["asset_weights"] == {"SOXL": 1.0, "TQQQ": 0.25}
    assert diagnostics["auxiliary_train_rows"] > 0
    assert diagnostics["primary_train_rows"] > 0
    assert np.allclose(prediction.sum(axis=1), 1.0)
    assert prediction["cash_weight"].ge(0.15).all()


def test_cross_pair_card_contract():
    required = {
        "cross_pair_auxiliary_training_v1.ipynb",
        "workflow.yaml",
        "summary.json",
        "conclusion.md",
        "selected_model.json",
        "model_selection.csv",
        "backtest_benchmark_comparison.csv",
        "backtest_walk_forward_metrics.csv",
        "risk_analysis.csv",
        "training_data_audit.csv",
        "objective_gate_audit.csv",
        "analysis_summary.json",
        "failure_modes.md",
    }
    assert required.issubset(
        {path.name for path in CARD.iterdir() if path.is_file()}
    )
    summary = json.loads(
        (CARD / "summary.json").read_text(encoding="utf-8")
    )
    notebook = json.loads(
        (CARD / "cross_pair_auxiliary_training_v1.ipynb").read_text(
            encoding="utf-8"
        )
    )
    notebook_text = json.dumps(notebook, ensure_ascii=False)

    assert summary["training_end"] == "2022-12-30"
    assert summary["selection_uses_outer_test"] is False
    assert summary["outer_role"] == "report_only"
    assert (
        summary["artifact_policy"]["policy"]
        == "slim_card_recorder_first"
    )
    assert "import qlib" in notebook_text
    assert "DatasetH" in notebook_text
    assert "from qlib.workflow import R" in notebook_text
    assert "中文结论" in notebook_text

    selection = pd.read_csv(CARD / "model_selection.csv")
    assert selection.groupby("pair")["selected"].sum().eq(1).all()
    assert not selection["selection_uses_outer_test"].astype(bool).any()
