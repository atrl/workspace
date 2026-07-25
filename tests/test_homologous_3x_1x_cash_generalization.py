from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.workflow.homologous_3x_1x_cash_generalization import (
    HomologousAllocationSpec,
    _fit_pair,
)


ROOT = Path(__file__).resolve().parents[1]
CARD = ROOT / "research" / "homologous_3x_1x_cash_generalization_v1"


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


def test_pair_fit_uses_only_requested_pair_and_outputs_legal_weights():
    panel = _panel()
    spec = HomologousAllocationSpec(root=Path("."), epochs=3)
    train = panel.loc[panel["datetime"].lt("2017-01-01")]
    test = panel.loc[
        panel["datetime"].between("2017-01-02", "2017-02-01")
    ]

    prediction, state, diagnostics = _fit_pair(
        train=train,
        test=test,
        features=["ret_5_z20", "ret_20_pctile60"],
        leveraged_symbol="TQQQ",
        spec=spec,
        seed=17,
    )

    assert state["leveraged_symbol"] == "TQQQ"
    assert diagnostics["cross_pair_train_rows"] == 0.0
    assert np.allclose(
        prediction[
            ["leveraged_weight", "underlying_weight", "cash_weight"]
        ].sum(axis=1),
        1.0,
    )
    assert prediction["leveraged_weight"].le(0.30).all()
    assert prediction.ge(0.0).all().all()


def test_homologous_card_contract():
    required = {
        "homologous_3x_1x_cash_generalization_v1.ipynb",
        "workflow.yaml",
        "summary.json",
        "conclusion.md",
        "selected_model.json",
        "backtest_benchmark_comparison.csv",
        "annual_metrics.csv",
        "risk_analysis.csv",
        "training_data_audit.csv",
        "objective_gate_audit.csv",
        "analysis_summary.json",
    }

    assert required.issubset(
        {path.name for path in CARD.iterdir() if path.is_file()}
    )
    summary = json.loads(
        (CARD / "summary.json").read_text(encoding="utf-8")
    )
    notebook = json.loads(
        (
            CARD
            / "homologous_3x_1x_cash_generalization_v1.ipynb"
        ).read_text(encoding="utf-8")
    )
    notebook_text = json.dumps(notebook, ensure_ascii=False)

    assert set(summary["pairs"]) == {
        "SOXL_SOXX_CASH",
        "TQQQ_QQQ_CASH",
    }
    assert summary["training_end"] == "2022-12-30"
    assert summary["selection_uses_outer_test"] is False
    assert summary["weight_transfer_between_pairs"] is False
    assert (
        summary["artifact_policy"]["policy"]
        == "slim_card_recorder_first"
    )
    assert "import qlib" in notebook_text
    assert "DatasetH" in notebook_text
    assert "from qlib.workflow import R" in notebook_text
    assert "中文结论" in notebook_text

    diagnostics = pd.read_csv(CARD / "training_data_audit.csv")
    expected_assets = {
        "SOXL_SOXX_CASH": "SOXL",
        "TQQQ_QQQ_CASH": "TQQQ",
    }
    for pair, asset in expected_assets.items():
        rows = diagnostics.loc[diagnostics["pair"].eq(pair)]
        assert set(rows["training_assets"]) == {asset}
        assert set(rows["test_assets"]) == {asset}
        assert rows["cross_pair_train_rows"].eq(0.0).all()
        assert not rows["selection_uses_outer_test"].astype(bool).any()
    outer = diagnostics.loc[
        diagnostics["segment"].eq("outer_diagnostic")
    ]
    assert set(outer["train_end"]) == {"2022-12-30"}
