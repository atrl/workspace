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
