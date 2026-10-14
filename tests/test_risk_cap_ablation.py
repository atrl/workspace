from types import SimpleNamespace
import numpy as np
import pandas as pd
from src.backtest.risk_cap_ablation import LegalCapOpenPolicy


def test_open_policy_ignores_future_values_and_keeps_hold_and_legal_repairs():
    class Skip:
        def predict_proba(self,frame):return np.array([[.99,.01]])
    spec=SimpleNamespace(execute_probability_threshold=.5,maximum_skippable_turnover=.05,mandatory_derisk_beta_delta=.1)
    def policy():return LegalCapOpenPolicy(open_model=Skip(),action_spec=spec,maximum_soxl_weight=.3,minimum_cash_weight=.15)
    row=pd.Series(dict(trade_date=pd.Timestamp('2024-01-02'),soxl_weight=.2,soxx_weight=.5,cash_weight=.3,tail_probability=.2,alpha_probability=.6,close_soxx=100,soxx_exit_open=100,early_ret_soxx=.01))
    assert policy().target(row,np.array([.2,.5]),.3,np.array([1.,1.]),1.) is None
    changed=row.copy();changed[['close_soxx','soxx_exit_open','early_ret_soxx']]=9999
    a=policy();b=policy();x=a.target(row,np.array([.31,.39]),.3,np.array([1.,1.]),1.);y=b.target(changed,np.array([.31,.39]),.3,np.array([1.,1.]),1.)
    np.testing.assert_array_equal(x,y);assert a.audit[-1]['mandatory_position_repair'];assert x[0]<=.3 and x[2]>=.15
