"""Focused contracts for the H20 replay validator + the real mock-solver replay entry."""
import numpy as np
import pandas as pd
import pytest

from src.backtest import generic_horizon_replay_contract as R
from src.backtest import generic_horizon_replay as RP
from src.data import generic_horizon as G


def _inputs(n=6):
    signals = pd.bdate_range("2020-01-01", periods=n)
    trades = signals + pd.Timedelta(days=1)
    returns = np.full((n, 2), 0.001)
    variances = np.stack([np.eye(2) * 0.01 for _ in range(n)])
    daily = np.stack([np.eye(2) * 0.001 for _ in range(n)])
    return signals, trades, returns, variances, daily


def test_valid_contract_units_and_clocks():
    signals, trades, returns, variances, daily = _inputs()
    audit = R.validate_h20_replay_contract(signal_dates=signals, trade_dates=trades, predicted_returns=returns,
                                           mean_variances=variances, daily_covariances=daily,
                                           unit_tags=G.unit_tags(20), horizon_sessions=20, rebalance_sessions=20)
    assert audit["kind"] == "validator_only" and audit["matched_holding"]
    assert audit["allocation_after_prediction"] and audit["unit_tags_validated"]
    assert audit["mean_unit"] == "twenty_session_simple_return"


def test_h5_unit_tags_fed_as_20_blocked():
    signals, trades, returns, variances, daily = _inputs()
    with pytest.raises(ValueError, match="unit tag"):
        R.validate_h20_replay_contract(signal_dates=signals, trade_dates=trades, predicted_returns=returns,
                                       mean_variances=variances, daily_covariances=daily,
                                       unit_tags=G.unit_tags(5), horizon_sessions=20, rebalance_sessions=20)


def test_rebalance_must_match_horizon():
    signals, trades, returns, variances, daily = _inputs()
    with pytest.raises(ValueError, match="rebalance_sessions == horizon_sessions"):
        R.validate_h20_replay_contract(signal_dates=signals, trade_dates=trades, predicted_returns=returns,
                                       mean_variances=variances, daily_covariances=daily,
                                       unit_tags=G.unit_tags(20), horizon_sessions=20, rebalance_sessions=5)


def test_covariance_structure_checks():
    signals, trades, returns, variances, daily = _inputs()
    rng = np.random.default_rng(0)
    a = rng.normal(size=(2, 2))
    negcorr = np.stack([a @ a.T + 1e-3 * np.eye(2) for _ in range(len(signals))])  # SPD with legal negative offdiag
    ok = R.validate_h20_replay_contract(signal_dates=signals, trade_dates=trades, predicted_returns=returns,
                                        mean_variances=negcorr, daily_covariances=daily, unit_tags=G.unit_tags(20))
    assert ok["unit_tags_validated"]

    bad_returns = returns.copy()
    bad_returns[0, 0] = -1.5
    with pytest.raises(ValueError, match="simple returns in"):
        R.validate_h20_replay_contract(signal_dates=signals, trade_dates=trades, predicted_returns=bad_returns,
                                       mean_variances=variances, daily_covariances=daily, unit_tags=G.unit_tags(20))

    asym = daily.copy()
    asym[0] = np.array([[0.001, 0.002], [0.0, 0.001]])
    with pytest.raises(ValueError, match="daily covariance must be symmetric"):
        R.validate_h20_replay_contract(signal_dates=signals, trade_dates=trades, predicted_returns=returns,
                                       mean_variances=variances, daily_covariances=asym, unit_tags=G.unit_tags(20))

    nonpsd = daily.copy()
    nonpsd[0] = np.array([[0.001, 0.002], [0.002, 0.001]])  # symmetric, negative eigenvalue
    with pytest.raises(ValueError, match="daily covariance must be positive semidefinite"):
        R.validate_h20_replay_contract(signal_dates=signals, trade_dates=trades, predicted_returns=returns,
                                       mean_variances=variances, daily_covariances=nonpsd, unit_tags=G.unit_tags(20))

    neg_diag = variances.copy()
    neg_diag[0, 0, 0] = -0.01  # negative diagonal is non-PSD (no variance floor imposed for mean error cov)
    with pytest.raises(ValueError, match="mean error covariance must be positive semidefinite"):
        R.validate_h20_replay_contract(signal_dates=signals, trade_dates=trades, predicted_returns=returns,
                                       mean_variances=neg_diag, daily_covariances=daily, unit_tags=G.unit_tags(20))


# ---- real replay entry (mock solver) ----
def _market_frame(n=60):
    sessions = pd.bdate_range("2020-01-01", periods=n + 2)
    signals, trades, exits = sessions[:-2], sessions[1:-1], sessions[2:]
    return pd.DataFrame({
        "signal_date": signals, "trade_date": trades, "exit_date": exits,
        "decision_at": G.allocation_clock(trades),
        "soxl_return": np.full(n, 0.002), "soxx_return": np.full(n, 0.001),
    })


def _arrays(n=60):
    return np.full((n, 2), 0.001), np.stack([np.eye(2) * 0.01 for _ in range(n)]), np.stack([np.eye(2) * 0.001 for _ in range(n)])


def _solver(held, mean, mean_var, daily_cov):
    from src.workflow.stage1_economic_action import fee_affordable_target
    net = np.array([0.3, 0.5, 0.2])
    execution = fee_affordable_target(net, pretrade_weights=np.asarray(held), cost_bps=10.0)
    return {"target_net_weights": net, "execution_target_weights": execution}


def test_replay_entry_hold_trade_fee_and_cashqty():
    frame = _market_frame(60)
    means, variances, daily = _arrays(60)
    calendar = pd.bdate_range("2020-01-01", periods=62)
    path, audit = RP.replay_matched_horizon(frame, means, variances, daily, solver=_solver, calendar=calendar,
                                            horizon_sessions=20, cost_bps=10.0, unit_tags=G.unit_tags(20))
    assert audit["kind"] == "replay_entry"
    assert audit["sessions"] == 60 and audit["replay_day_completeness"]
    assert audit["trades"] == 3 and audit["hold_sessions"] == 57          # rebalance at 0,20,40
    assert audit["hold_requires_zero_fee"]
    assert (path.loc[path["execute"], "turnover"] > 0).all()
    assert (path.loc[~path["execute"], "turnover"] == 0).all()
    # cash/qty settlement consistency: ending values sum to the session net equity
    assert audit["cash_plus_stocks_equals_equity"]
    assert np.allclose(path["ending_soxl_weight"] + path["ending_soxx_weight"] + path["ending_cash_weight"], 1.0)
    assert audit["total_fees"] > 0


def test_replay_entry_blocks_wrong_clock_and_wrong_horizon_units():
    frame = _market_frame(30)
    means, variances, daily = _arrays(30)
    bad = frame.copy()
    bad["decision_at"] = G.predictor_clock(bad["signal_date"])  # prior 16:15, not the quote clock
    calendar = pd.bdate_range("2020-01-01", periods=32)
    with pytest.raises(ValueError, match="trade-open quote clock"):
        RP.replay_matched_horizon(bad, means, variances, daily, solver=_solver, calendar=calendar,
                                  horizon_sessions=20, cost_bps=10.0, unit_tags=G.unit_tags(20))
    with pytest.raises(ValueError, match="unit tag"):
        RP.replay_matched_horizon(frame, means, variances, daily, solver=_solver, calendar=calendar,
                                  horizon_sessions=20, cost_bps=10.0, unit_tags=G.unit_tags(5))
    with pytest.raises(ValueError, match="unit tags are mandatory"):
        RP.replay_matched_horizon(frame, means, variances, daily, solver=_solver, calendar=calendar,
                                  horizon_sessions=20, cost_bps=10.0, unit_tags=None)


def test_mean_error_covariance_allows_zero_psd_but_blocks_asymmetry():
    signals, trades, returns, variances, daily = _inputs()
    zero = np.zeros((len(signals), 2, 2))  # legal zero PSD, no imposed variance floor
    ok = R.validate_h20_replay_contract(signal_dates=signals, trade_dates=trades, predicted_returns=returns,
                                        mean_variances=zero, daily_covariances=daily, unit_tags=G.unit_tags(20))
    assert ok["unit_tags_validated"]
    asym = variances.copy()
    asym[0] = np.array([[0.01, 0.02], [0.0, 0.01]])
    with pytest.raises(ValueError, match="mean error covariance must be symmetric"):
        R.validate_h20_replay_contract(signal_dates=signals, trade_dates=trades, predicted_returns=returns,
                                       mean_variances=asym, daily_covariances=daily, unit_tags=G.unit_tags(20))


def test_hold_budget_zero_uses_same_tolerance():
    frame = _market_frame(10)
    means, variances, daily = _arrays(10)
    daily = daily.copy()
    daily[1] = np.array([[0.01, 0.0], [0.0, 0.0]])  # SOXX budget 0 at a HOLD row, SOXL variance > 0
    calendar = pd.bdate_range("2020-01-01", periods=12)
    path, audit = RP.replay_matched_horizon(frame, means, variances, daily, solver=_solver, calendar=calendar,
                                            horizon_sessions=20, cost_bps=10.0, unit_tags=G.unit_tags(20))
    assert audit["mid_hold_risk_drift_exceedance_sessions"] >= 1


def test_replay_entry_reuses_covariance_symmetry_guard():
    frame = _market_frame(30)
    means, variances, daily = _arrays(30)
    asym = daily.copy()
    asym[0] = np.array([[0.001, 0.002], [0.0, 0.001]])
    calendar = pd.bdate_range("2020-01-01", periods=32)
    with pytest.raises(ValueError, match="daily covariance must be symmetric"):
        RP.replay_matched_horizon(frame, means, variances, asym, solver=_solver, calendar=calendar,
                                  horizon_sessions=20, cost_bps=10.0, unit_tags=G.unit_tags(20))
