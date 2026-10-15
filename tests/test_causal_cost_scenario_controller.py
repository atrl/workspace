import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
from src.model.nominal_beta_composition import AuditedComposition


def test_single_scenario_equivalence_and_causal_order_invariance(monkeypatch):
    import src.backtest.full_path_tiered as original
    import src.backtest.causal_cost_scenario_controller as coupled
    settings=json.loads((Path(__file__).resolve().parents[1]/'research/causal_cost_scenario_controller_20261015/plan.json').read_text())
    dates=pd.date_range('2023-01-04',periods=3)
    frame=pd.DataFrame({'signal_date':dates-pd.Timedelta(days=1),'trade_date':dates,'exit_date':dates+pd.Timedelta(days=1),'soxl_weight':.2,'soxx_weight':.4,'cash_weight':.4})
    for field in ['known_soxl','known_soxx','soxl_open','soxx_open','soxl_exit_open','soxx_exit_open','close_soxl','close_soxx','raw_soxl_open','raw_soxx_open','soxl_recovery','soxx_recovery','raw_soxl_recovery','raw_soxx_recovery']:frame[field]=100.
    spec=SimpleNamespace(minimum_cash_weight=.15,maximum_soxl_weight=.3)
    def opened(row,**kwargs):
        drawdown=max(0,1-kwargs['equity']/kwargs['peak'])
        return dict(risk_state=0,risk_clock=0,desired_open_weights=np.array([.2,.4,.4]),execute_open_target=True,current_beta_cap=1.,drawdown=drawdown,raw_beta=1.,history=list(kwargs['history']))
    def recovered(row,**kwargs):
        yes=row.trade_date==dates[0]
        return dict(recovery_action=yes,recovery_probability=.9 if yes else np.nan,target_weights=np.array([.2,.4,.4]) if yes else None)
    for module in [original,coupled]:
        monkeypatch.setattr(module,'prepare_on_policy_open',opened);monkeypatch.setattr(module,'prepare_on_policy_recovery',recovered)
    controller={'frozen_open_action_value_model':None,'frozen_recovery_model':None}
    kwargs=dict(plan=settings,controller=controller,recovery_spec=spec,action_spec=None,selection=None)
    one,trace=coupled.replay_cost_scenarios(frame,**kwargs,scenarios=[2])
    baseline=original.replay_full_path(frame,models=None,slippage_bps=2,**kwargs,target_transform=AuditedComposition())
    for key in ['replay','orders','targets']:pd.testing.assert_frame_equal(one[2][key],baseline[key])
    group,trace=coupled.replay_cost_scenarios(frame,**kwargs)
    reversed_group,reversed_trace=coupled.replay_cost_scenarios(frame,**kwargs,scenarios=[5,2,0])
    for cost in [0,2,5]:
        for key in ['replay','orders','targets']:pd.testing.assert_frame_equal(group[cost][key],reversed_group[cost][key])
    changed=frame.copy();changed.loc[0,['soxl_open','soxx_open','soxl_recovery','soxx_recovery','close_soxl','close_soxx']]*=1.2
    future,other=coupled.replay_cost_scenarios(changed,**kwargs)
    for cost in [0,2,5]:pd.testing.assert_frame_equal(group[cost]['targets'].iloc[:1],future[cost]['targets'].iloc[:1])
    columns=['open_after_T_soxl','open_after_T_soxx','open_after_T_cash','recovery_consensus','recovery_after_T_soxl','recovery_after_T_soxx','recovery_after_T_cash']
    pd.testing.assert_frame_equal(trace.loc[trace.trade_date.eq(dates[0]),columns],other.loc[other.trade_date.eq(dates[0]),columns])
