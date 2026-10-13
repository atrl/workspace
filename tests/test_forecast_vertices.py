import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest.forecast_vertices import ForecastVertices, feasible_vertices
from src.backtest.full_path_tiered import replay_full_path


def test_feasible_known_price_decision_and_true_hold_across_fill_move():
    root = Path(__file__).resolve().parents[1]
    plan = json.loads((root/'research/cushion_target_tiered_20261013/plan.json').read_text())
    plan.update(initial_cash=1000., decision_cost_multiplier=2.)
    for beta in [0., .2, .85, 1.2]:
        for weights in feasible_vertices(beta, .15, .3):
            assert weights.min() >= 0 and np.isclose(weights.sum(), 1)
            assert weights[0] <= .3+1e-10 and weights[2] >= .15-1e-10
            assert weights[:2] @ [3.,1.] <= beta+1e-10
    dates = pd.to_datetime(['2023-01-04','2023-01-05'])
    frame = pd.DataFrame({'signal_date':dates-pd.Timedelta(days=1),'trade_date':dates,'exit_date':dates+pd.Timedelta(days=1),
        'known_soxl':10.,'known_soxx':20.,'raw_known_soxl':10.,'raw_known_soxx':20.,
        'soxl_open':[10.,20.],'soxx_open':[20.,10.],'raw_soxl_open':[10.,20.],'raw_soxx_open':[20.,10.],
        'close_soxl':[10.,20.],'close_soxx':[20.,10.],'soxl_exit_open':20.,'soxx_exit_open':10.},index=dates)
    predictions = pd.DataFrame({'cash_relative_h5':-.01,'leverage_relative_h5':.03},index=dates)
    policy = ForecastVertices(predictions,plan=plan,slippage_bps=2)
    row=frame.iloc[0]; changed=row.copy();changed['soxl_open']=1000.;changed['raw_soxl_open']=1000.
    a=policy.target(row,np.zeros(2),1000.,np.array([10.,20.]),1000.)
    b=policy.target(changed,np.zeros(2),1000.,np.array([10.,20.]),1000.)
    np.testing.assert_array_equal(a,b)
    class BuyThenHold:
        def target(self,row,q,cash,known,peak):
            return np.array([.1,.4,.5]) if row.name==dates[0] else None
    result=replay_full_path(frame,models=[BuyThenHold()],plan=plan,slippage_bps=2)
    assert result['orders'].trade_date.eq(dates[0]).all()
    assert not result['replay'].execute_open_target.iloc[1]
    assert result['replay'].ending_soxl_quantity.iloc[0]==result['replay'].ending_soxl_quantity.iloc[1]
