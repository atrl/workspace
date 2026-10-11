"""Actual quantities and cash must support fees, NAV and all three reported weights."""
import numpy as np
import pandas as pd
import pytest
from src.analysis.generic_native_reconciliation import audit_actual_positions


class Position:
    def get_stock_amount_dict(self):
        return {'SOXX': 10.0}
    def get_cash(self):
        return 0.0


def _case():
    days = pd.date_range('2020-07-01', periods=4)
    frame = pd.DataFrame({'signal_date':days[:2], 'trade_date':days[1:3], 'exit_date':days[2:]})
    path = frame.copy()
    path['execute_open_target'] = [True, False]
    path['ending_soxl_weight'], path['ending_soxx_weight'], path['ending_cash_weight'] = 0., 1., 0.
    # Cash 101 buys ten shares at 10 plus a one-dollar fee; shares remain owned.
    path['fees'], path['turnover'], path['equity'] = [1/101, 0], [100/101, 0], [110/101, 120/101]
    opening = pd.DataFrame({'SOXL':[20.,20.,20.], 'SOXX':[10.,11.,12.]}, index=days[1:])
    native = {'positions':{days[1]:Position(), days[2]:Position()}}
    return frame, path, native, opening


def test_cash_quantity_and_fee_parity_with_exact_hold():
    frame, path, native, opening = _case()
    result, audit = audit_actual_positions(frame,path,native,opening,initial_cash=101.)
    assert audit['actual_cash_quantity_checked'] and audit['all_three_ending_weights_checked']
    np.testing.assert_allclose(result.native_next_open_nav,[110.,120.])
    assert audit['maximum_fee_fraction_error'] < 1e-15


@pytest.mark.parametrize('column,value', [('ending_cash_weight',.1), ('fees',0.), ('turnover',0.), ('equity',2.)])
def test_false_native_parity_is_rejected(column, value):
    frame, path, native, opening = _case()
    path.loc[0,column] = value
    with pytest.raises(ValueError,match='cash/quantity reconciliation differs'):
        audit_actual_positions(frame,path,native,opening,initial_cash=101.)


def test_unused_hold_roundoff_adapter_does_not_change_research_or_executed_target(monkeypatch):
    import src.analysis.generic_native_reconciliation as module
    frame,path,_,_ = _case()
    path['open_soxl_weight'],path['open_soxx_weight'],path['open_cash_weight'] = 0.,1.,0.
    path.loc[1,'open_soxx_weight'],path.loc[1,'open_cash_weight'] = 1+2e-16,-2e-16
    before = path.copy(deep=True)
    captured = {}
    def capture(frame,schedule,**kwargs):
        captured['schedule'] = schedule
        raise RuntimeError('captured without replay')
    monkeypatch.setattr(module,'native_open_reconciliation',capture)
    with pytest.raises(RuntimeError,match='captured without replay'):
        module.reconcile_native_horizon(frame,path,provider_uri='unused',cost_bps=10.,account=101.)
    assert captured['schedule'].loc[1,'open_cash_weight'] == 0
    pd.testing.assert_series_equal(captured['schedule'].loc[0],path.loc[0])
    pd.testing.assert_frame_equal(path,before)
    path.loc[0,'open_cash_weight'] = -2e-16
    with pytest.raises(ValueError,match='executed negative'):
        module.reconcile_native_horizon(frame,path,provider_uri='unused',cost_bps=10.,account=101.)
    path.loc[0,'open_cash_weight'] = 0.
    path.loc[1,'open_cash_weight'] = -1e-6
    with pytest.raises(ValueError,match='materially negative'):
        module.reconcile_native_horizon(frame,path,provider_uri='unused',cost_bps=10.,account=101.)
