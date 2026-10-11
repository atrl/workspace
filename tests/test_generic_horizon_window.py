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


def test_source_fallback_preserves_cross_year_cadence_and_unknown_forecasts(monkeypatch):
    calendar=pd.bdate_range('2016-12-28',periods=14)
    dates=calendar[:-2]
    known=np.ones(len(dates),int);known[[5,7]]=0
    frame=pd.DataFrame(dict(signal_date=dates,trade_date=calendar[1:-1],exit_date=calendar[2:],
        decision_at=predictor_clock(dates),feature_available_at=predictor_clock(dates),
        soxl_return=.02,soxx_return=.01,soxl_return_h5=np.nan,soxx_return_h5=np.nan,
        h5_available_at=pd.NaT,source_known=known))
    queried=[]
    def predict(model,part,**kwargs):
        assert part.source_known.eq(1).all() and (part.index.year==2017).all()
        assert kwargs['feature_columns']==('extra_feature',)
        queried.extend(part.index)
        return dict(ridge=np.zeros((len(part),2)),prior=np.zeros((len(part),2)),Vmean=np.zeros((len(part),2,2)))
    monkeypatch.setattr(window,'predict_horizon',predict)
    class CertifiedHold:
        def __init__(self,**kwargs): pass
        def solve(self,held,*args,**kwargs):
            from src.workflow.stage1_economic_action import fee_affordable_target
            target=np.array([0.,1.,0.])
            return dict(target_net_weights=target,
                execution_target_weights=fee_affordable_target(target,pretrade_weights=held,cost_bps=10.),
                optimality_verified=True)
    import src.model.certified_convex_allocation as certified
    monkeypatch.setattr(certified,'CertifiedConvexAllocationSolver',CertifiedHold)
    out=window.replay_horizon_window(frame,{2017:{'selected_fit':{}}},np.tile(np.eye(2)*.001,(len(frame),1,1)),
        calendar=calendar,cost_bps=10.,horizon_sessions=5,feature_columns=('extra_feature',),
        availability_column='h5_available_at',eligibility_column='source_known',fallback_years=(2016,))
    path=out['path'];forecast=out['forecast']
    assert path.index[path.execute].tolist()==[0,5,10]
    assert path.index[path.fallback_rebalance].tolist()==[0,5]
    assert path.loc[7,'execute']==False and path.loc[7,'fees']==0 and path.loc[7,'turnover']==0
    assert path.loc[~path.execute,'fees'].eq(0).all()
    assert path.fees.iloc[1:].abs().max()<1e-18
    unknown=~forecast.model_used
    assert forecast.loc[unknown,['predicted_soxl','predicted_soxx']].isna().all().all()
    assert len(forecast)==len(frame) and len(queried)==int(forecast.model_used.sum())
