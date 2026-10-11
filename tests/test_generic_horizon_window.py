"""Window integration: benchmark fees once and uncertified optimization fails closed."""
import numpy as np
import pandas as pd
import pytest
from src.backtest import generic_horizon_window as window
from src.data.generic_horizon import predictor_clock


def _market():
    calendar = pd.bdate_range('2020-01-02', periods=7)
    dates = calendar[:-2]
    frame = pd.DataFrame(dict(signal_date=dates, trade_date=calendar[1:-1], exit_date=calendar[2:],
        decision_at=predictor_clock(dates), feature_available_at=predictor_clock(dates),
        soxl_return=.02, soxx_return=.01, soxl_return_h20=np.nan, soxx_return_h20=np.nan,
        label_h20_available_at=pd.NaT))
    return frame, calendar, np.tile(np.eye(2)*.001,(len(frame),1,1))


@pytest.mark.parametrize('cost',[10.,25.])
def test_soxx_window_pays_only_initial_entry_fee_and_retains_terminal_position(cost):
    frame, calendar, sigma = _market()
    result = window.replay_horizon_window(frame, {}, sigma, calendar=calendar, cost_bps=cost, benchmark_only=True)
    path = result['path']
    assert path.execute_open_target.tolist() == [True,False,False,False,False]
    assert (path.fees.iloc[1:] == 0).all() and (path.turnover.iloc[1:] == 0).all()
    np.testing.assert_allclose(path.ending_soxx_weight, 1., rtol=0, atol=1e-12)
    assert result['replay_audit']['target_decision_calls'] == 1
    np.testing.assert_allclose(path.equity.iloc[-1], 1.01**5/(1+cost/10000),rtol=0,atol=1e-12)


def test_unverified_optimizer_certificate_cannot_become_a_path(monkeypatch):
    frame, calendar, sigma = _market()
    monkeypatch.setattr(window, 'predict_horizon', lambda model, part, **kwargs:
        dict(ridge=np.zeros((len(part),2)),prior=np.zeros((len(part),2)),Vmean=np.tile(np.eye(2)*.001,(len(part),1,1))))
    class Unverified:
        def __init__(self, **kwargs):
            pass
        def solve(self, *args, **kwargs):
            return {'optimality_verified':False}
    import src.model.certified_convex_allocation as certified
    monkeypatch.setattr(certified, 'CertifiedConvexAllocationSolver', Unverified)
    with pytest.raises(ValueError,match='verified optimality certificate'):
        window.replay_horizon_window(frame,{2020:{'selected_fit':{}}},sigma,calendar=calendar,cost_bps=25.)
