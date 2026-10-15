import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.backtest.ibkr_tiered import TieredFees, settle_tiered_event
from src.model.full_path_tiered import FullPathTieredModel, PathPolicy, PRICE_FIELDS, roll_full_path, tiered_tensor_quote, tiered_tensor_settle
from src.workflow.full_path_tiered import full_path_dataset


def plan():
    return json.loads((Path(__file__).resolve().parents[1] / "research/full_path_tiered_allocation_20261013/plan.json").read_text())


def test_tensor_whole_share_fees_and_monthly_margins_equal_independent_cash_ledger():
    settings = plan()
    shares, raw, months = torch.tensor([200., 1000., 10., 0.], dtype=torch.float64), torch.tensor([50., 50., .2, 1.], dtype=torch.float64), torch.tensor([0., 299990., 1234., 0.], dtype=torch.float64)
    quoted, counted = tiered_tensor_quote(shares, raw, True, months, tariff=settings["tariff"], slippage_bps=2)
    for number in range(4):
        fees = TieredFees(settings["tariff"], slippage_bps=2)
        fees.monthly_shares["2023-01"] = months[number].item()
        expected = fees.quote(shares=int(shares[number]), raw_price=raw[number].item(), sell=True, stamp="2023-01-04")
        fees.commit(int(shares[number]), stamp="2023-01-04", cap_applied=expected["cap_applied"])
        assert np.isclose(quoted[number].item(), expected["total"], rtol=0, atol=1e-10)
        assert counted[number].item() == fees.monthly_shares["2023-01"]
    q, cash = np.array([62., 33.]), 25000.
    prices, raw_prices, desired = np.array([100., 200.]), np.array([317., 607.]), np.array([.2, .3, .5])
    fees = TieredFees(settings["tariff"], slippage_bps=2); fees.monthly_shares["2023-01"] = 299950
    expected = settle_tiered_event(q, cash, desired=desired, adjusted_prices=prices, raw_prices=raw_prices, stamp="2023-01-04", fees=fees)
    tq, tc, tm, charge = tiered_tensor_settle(torch.tensor(q[None,:]), torch.tensor([cash]), torch.tensor(desired[None,:]),
        torch.tensor(prices[None,:]), torch.tensor(raw_prices[None,:]), torch.tensor([299950.]), tariff=settings["tariff"], slippage_bps=2)
    np.testing.assert_allclose(tq[0], expected["quantities"], rtol=0, atol=1e-10)
    assert np.isclose(tc[0], expected["cash"], rtol=0, atol=1e-8)
    assert np.isclose(charge[0], sum(order["total"] for order in expected["orders"]), rtol=0, atol=1e-8)
    assert tm[0] == fees.monthly_shares["2023-01"]


def test_current_decision_excludes_future_fill_price_and_has_finite_path_gradient():
    settings = plan()
    torch.manual_seed(71)
    policy = PathPolicy(2,1,4,family="minute",minimum_cash=.15,maximum_soxl=.3).double()
    features = torch.tensor([[[.3,.2],[.2,.4]]],dtype=torch.float64)
    prices = torch.tensor([[[100.,200.,101.,201.,101.,201.,102.,204.],[102.,204.,103.,205.,103.,205.,104.,206.]]],dtype=torch.float64)
    months = torch.tensor([[202301,202301]])
    values=[]
    handle=policy.register_forward_hook(lambda model,args,output: values.append(output.detach().clone()))
    result=roll_full_path(policy,features,prices,months,plan=settings)
    first=values[0].clone(); values.clear()
    changed=prices.clone(); changed[:,0,2:6]*=1.2; changed[:,0,6:8]*=.8
    roll_full_path(policy,features,changed,months,plan=settings)
    torch.testing.assert_close(first,values[0],rtol=0,atol=0)
    handle.remove()
    torch.log(result["equity"][:,-1]).sum().backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in policy.parameters())


def test_qlib_full_path_maturity_is_strict_and_scaler_ignores_unmatured_rows():
    settings=plan();settings.update(epochs=1,block_length=2,warmup_sessions=1,hidden_units=4)
    dates=pd.date_range("2023-01-04",periods=7,freq="1D")
    frame=pd.DataFrame({"a":np.linspace(0,1,7),"m":np.linspace(1,0,7),"decision_at":dates+pd.Timedelta(hours=9,minutes=46),
        "feature_available_at":dates-pd.Timedelta(days=1)+pd.Timedelta(hours=16,minutes=1),"label_available_at":dates+pd.Timedelta(days=1)},index=dates)
    for column in PRICE_FIELDS:frame[column]=100.
    frame.loc[dates[-2:],"label_available_at"]=pd.to_datetime(["2024-01-01","2024-01-02"])
    altered=frame.copy();altered.loc[dates[-2:],["a","m"]]=1e8
    fitted=[FullPathTieredModel(columns=["a","m"],base_count=1,family="minute",seed=71,cutoff="2024-01-01",plan=settings)
        .fit(full_path_dataset(data,["a","m"],"2024-01-01")) for data in (frame,altered)]
    assert fitted[0].audit["training_rows"]==5
    np.testing.assert_array_equal(fitted[0].mean,fitted[1].mean)
    for actual,expected in zip(fitted[0].network.parameters(),fitted[1].network.parameters()):
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)


def test_optional_composition_hook_identity_hold_and_both_execution_slots(monkeypatch):
    import src.backtest.full_path_tiered as module
    from src.model.nominal_beta_composition import AuditedComposition, cancel_leverage_cash
    settings=plan()
    dates=pd.date_range('2023-01-04',periods=2)
    frame=pd.DataFrame({'signal_date':dates-pd.Timedelta(days=1),'trade_date':dates,'exit_date':dates+pd.Timedelta(days=1)})
    for field in ['known_soxl','known_soxx','soxl_open','soxx_open','soxl_exit_open','soxx_exit_open','close_soxl','close_soxx','raw_soxl_open','raw_soxx_open','soxl_recovery','soxx_recovery','raw_soxl_recovery','raw_soxx_recovery']:frame[field]=100.
    target=np.array([.2,.4,.4])
    def opened(row,**kwargs):return dict(risk_state=0,risk_clock=0,desired_open_weights=target,execute_open_target=row.trade_date==dates[0],current_beta_cap=1.)
    def recovered(row,**kwargs):return dict(recovery_action=row.trade_date==dates[0],target_weights=target)
    monkeypatch.setattr(module,'prepare_on_policy_open',opened);monkeypatch.setattr(module,'prepare_on_policy_recovery',recovered)
    kwargs=dict(models=None,plan=settings,slippage_bps=2,controller={'frozen_open_action_value_model':None,'frozen_recovery_model':None})
    baseline=module.replay_full_path(frame,**kwargs)
    identity=module.replay_full_path(frame,**kwargs,target_transform=lambda weights,**context:weights)
    for key in ['replay','orders','targets']:pd.testing.assert_frame_equal(baseline[key],identity[key])
    transform=AuditedComposition();changed=module.replay_full_path(frame,**kwargs,target_transform=transform)
    assert [r['slot'] for r in transform.rows]==['open','recovery']
    np.testing.assert_array_equal(target,[.2,.4,.4])
    np.testing.assert_allclose(changed['targets'].iloc[0][['soxl_weight','soxx_weight','cash_weight']].to_numpy(float),[0.,1.,0.])
    assert changed['replay'].ending_soxl_quantity.eq(0).all()
    for value in [[0.,1.,0.],[.3,.6,.1],[0.,0.,1.]]:
        result=cancel_leverage_cash(value);assert abs(3*result[0]+result[1]-(3*value[0]+value[1]))<1e-12
