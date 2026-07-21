from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.workflow import LeveragedEtfPaperReplaySpec, run_leveraged_etf_paper_replay_card
from src.workflow.leveraged_etf_paper_replay import (
    EXPERIMENT,
    _inverse_variance_weights,
    _risk_metrics,
    _simulate_george_trainor_cppi,
)


ROOT = Path(__file__).resolve().parents[1]


def _toy_frame(rows: int = 260) -> pd.DataFrame:
    dates = pd.bdate_range("2018-01-02", periods=rows + 2)
    return pd.DataFrame(
        {
            "signal_date": dates[:-2],
            "trade_date": dates[1:-1],
            "exit_date": dates[2:],
            "soxl_return": np.zeros(rows),
            "soxx_return": np.zeros(rows),
            "soxx_close_return": np.linspace(-0.01, 0.01, rows),
            "soxx_close": 100.0 * np.cumprod(1.0 + np.linspace(-0.001, 0.001, rows)),
        }
    )


def test_inverse_variance_normalizer_is_sealed_before_outer_years():
    frame = _toy_frame(520)
    spec = LeveragedEtfPaperReplaySpec(root=ROOT, inverse_variance_calibration_end="2018-12-31")
    _, audit_before = _inverse_variance_weights(frame, spec)
    changed = frame.copy()
    changed.loc[changed["signal_date"].dt.year.ge(2019), "soxx_close_return"] *= 50.0
    _, audit_after = _inverse_variance_weights(changed, spec)
    assert audit_before["normalizer"] == audit_after["normalizer"]
    assert audit_after["selection_uses_outer_test"] is False


def test_cppi_matches_paper_initial_weight_and_cap():
    frame = _toy_frame(20)
    result = _simulate_george_trainor_cppi(frame, cost_bps=0)
    assert result.iloc[0]["soxl_weight"] == pytest.approx(1.0 / 6.0)
    assert result["soxl_weight"].between(0.0, 1.0 / 3.0).all()


def test_risk_metrics_uses_true_compounded_rolling_five_day_return():
    returns = np.array([-0.02, -0.02, -0.02, -0.02, -0.02, 0.01])
    frame = pd.DataFrame(
        {
            "strategy_return": returns,
            "soxl_return": returns,
            "soxl_weight": np.ones(len(returns)),
            "soxx_weight": np.zeros(len(returns)),
            "turnover": np.zeros(len(returns)),
        }
    )
    metrics = _risk_metrics(frame)
    expected = ((1.0 - 0.02) ** 5 - 1.0) * 100.0
    assert metrics["worst_rolling_5d_pct"] == pytest.approx(expected)


def test_paper_replay_card_is_qlib_recorder_first_and_outer_sealed():
    card = ROOT / "research" / EXPERIMENT
    summary = json.loads((card / "summary.json").read_text(encoding="utf-8"))
    notebook = (card / f"{EXPERIMENT}.ipynb").read_text(encoding="utf-8")
    conclusion = (card / "conclusion.md").read_text(encoding="utf-8")

    assert summary["experiment"] == EXPERIMENT
    assert summary["selection_uses_outer_test"] is False
    assert summary["diagnostic_ranking_uses_outer_test"] is True
    assert "never_parameter_selection" in summary["outer_test_use"]
    assert summary["data_audit"]["as_of"] == "2026-07-20"
    assert summary["data_audit"]["latest_refresh_provider"] == "yahoo_api"
    assert summary["data_audit"]["futu_refresh_status"].startswith("needs_data")
    assert summary["data_audit"]["futu_yahoo_overlap_audit"]["SOXL"]["status"] == "ok"
    assert summary["required_cost_bps"] == 10
    assert summary["stress_cost_bps"] == 25
    assert summary["artifact_policy"]["policy"] == "slim_card_recorder_first"
    assert not (card / "positions.csv").exists()
    assert not (card / "strategy_returns.csv").exists()
    recorder = Path(summary["recorder_local_dir"]) / "artifacts" / EXPERIMENT
    assert (recorder / "positions.csv").exists()
    assert (recorder / "strategy_returns.csv").exists()
    assert "run_leveraged_etf_paper_replay_card" in notebook
    assert "qlib" in notebook
    assert "不构成买卖建议" in conclusion

    audit = {row["strategy"]: row for row in summary["paper_implementation_audit"]}
    assert audit["george_trainor_cppi_3x_2p5pct"]["status"] == "near_exact_mechanism_replay"
    assert audit["smart_leverage_neural_control"]["status"] == "not_replicated_needs_data_and_scope"


def test_workflow_lazy_exports_paper_replay():
    assert LeveragedEtfPaperReplaySpec.__name__ == "LeveragedEtfPaperReplaySpec"
    assert run_leveraged_etf_paper_replay_card.__name__ == "run_leveraged_etf_paper_replay_card"
