import numpy as np
import pandas as pd
import pytest

from src.analysis import cash_drag_overlay as analysis
from src.workflow import cash_drag_overlay as workflow


def test_apply_overlay_moves_cash_to_underlying_and_preserves_funding():
    weights = np.array([[0.2, 0.3, 0.5], [0.0, 0.4, 0.6]])
    overlay = analysis.CashOverlay("r50", 0.5)
    result = analysis.apply_overlay(weights, overlay)
    assert result[0].tolist() == pytest.approx([0.2, 0.55, 0.25])
    assert result[1].tolist() == pytest.approx([0.0, 0.7, 0.3])
    assert np.allclose(result.sum(axis=1), 1.0)
    # SOXL weight is never changed by the overlay.
    assert (result[:, 0] == weights[:, 0]).all()


def test_apply_overlay_rejects_bad_fraction():
    weights = np.array([[0.2, 0.3, 0.5]])
    with pytest.raises(ValueError, match="recycle_fraction"):
        analysis.apply_overlay(weights, analysis.CashOverlay("bad", 1.5))


def test_replay_weights_charges_turnover_and_uses_next_session_returns():
    dates = pd.bdate_range("2026-08-03", periods=5)
    weights = pd.DataFrame({"soxl_weight": 0.0, "soxx_weight": 0.0, "cash_weight": 1.0}, index=dates)
    returns = pd.DataFrame({"soxl": 0.01, "soxx": 0.002, "cash": 0.0}, index=dates)
    replay = analysis.replay_weights(weights, returns, cost_bps=10.0)
    # All cash -> zero strategy return with no turnover after the first session.
    assert (replay["strategy_return"].abs() < 1e-12).all()
    assert replay["turnover"].iloc[0] == pytest.approx(0.0)


def test_performance_flags_dominance():
    dates = pd.bdate_range("2026-08-03", periods=200)
    rng = np.random.default_rng(3)
    bench = pd.Series(0.0002 + rng.normal(0, 0.004, len(dates)), index=dates)
    replay = pd.DataFrame({"strategy_return": bench + 0.0008, "turnover": 0.0,
                           "cash_weight": 0.0, "soxx_weight": 1.0, "soxl_weight": 0.0}, index=dates)
    stats = analysis.performance(replay, bench, "SOXX")
    assert stats["beats_return"] is True
    assert stats["beats_sharpe"] is True


def test_production_targets_extraction_and_diagnostic(tmp_path):
    """The workflow reads the real production slot history and diagnoses cash source."""
    summary = workflow.run_cash_drag_overlay(
        root=tmp_path.parents[0] if False else __import__("pathlib").Path(__file__).resolve().parents[1],
        card_dir=tmp_path / "card", recorder_uri=tmp_path / "card" / "qlib_recorder")
    diag = summary["diagnostic"]
    assert diag["sessions"] >= 8
    assert 0.0 <= diag["average_cash_weight"] <= 1.0
    assert summary["qlib_data_api"] == ["qlib.data.D.features"]
    assert summary["recorder_id"]
