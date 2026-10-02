"""Money identities and state/fee derivatives, independent of market outcomes."""
import numpy as np
import pytest
import torch

from src.model.post_fee_settlement import settle_torch, scalar_settle
from src.model.benchmark_anchor_utility import NumericalFitFailure
from src.model.integrated_allocation import _settle as legacy
from src.workflow.stage1_economic_action import fee_affordable_target
from src.backtest.position_accounting import settle_target_session


def tensor(value, **kw):
    return torch.tensor(value, dtype=torch.float64, **kw)


def independent(h, z, r, rate):
    executable = fee_affordable_target(np.asarray(z), pretrade_weights=np.asarray(h), cost_bps=rate*10000)
    return settle_target_session(h, open_target_weights=executable, asset_returns=r, cost_bps=rate*10000)


@pytest.mark.parametrize('rate', [0., .001, .0025])
def test_exact_accounting_against_bisection_and_independent_cash_ledger(rate):
    rng = np.random.default_rng(814)
    states = list(zip(rng.dirichlet([1, 1, 1], 31), rng.dirichlet([1, 1, 1], 31)))
    vertices = np.eye(3)
    states += [(h, z) for h in vertices for z in vertices]
    states += [(np.array([.2, .6, .2]), np.array([.2, .6, .2]))]
    for h, z in states:
        r = np.r_[rng.uniform(-.2, .2, 2), 0.]
        actual = settle_torch(tensor(h), tensor(z), tensor(r), rate)
        old = legacy(tensor(h), tensor(z), tensor(r), rate)
        ref = independent(h, z, r, rate)
        for key in ('net_equity', 'fees', 'turnover', 'ending_weights', 'open_net_equity', 'open_net_weights'):
            np.testing.assert_allclose(actual[key].numpy(), ref[key], atol=5e-15, rtol=0)
            torch.testing.assert_close(actual[key], old[key], atol=5e-15, rtol=0)
        out = scalar_settle(*h, *z, *r[:2], rate)
        for value, key in zip(out[:4], ('ending_weights', 'net_equity', 'fees', 'turnover')):
            np.testing.assert_allclose(value, actual[key].numpy(), atol=5e-15, rtol=0)
        np.testing.assert_allclose(actual['net_equity'], (1-actual['fees'])*np.dot(z, 1+r), atol=1e-15, rtol=0)


@pytest.mark.parametrize('rate', [.001, .0025])
@pytest.mark.parametrize('h,dz,dh', [
    ([0.,1.,0.], [0.,-1.,1.], [0.,0.,0.]),
    ([0.,1.,0.], [1.,-1.,0.], [0.,0.,0.]),
    ([.2,.6,.2], [.3,-.4,.1], [0.,0.,0.]),
    ([.2,.6,.2], [0.,0.,0.], [.3,-.4,.1]),
    ([.2,.6,.2], [.3,-.4,.1], [.3,-.4,.1]),
])
def test_unique_state_jvp_at_no_trade_is_target_only_and_cost_independent(rate, h, dz, dh):
    h, dz, dh = map(tensor, (h, dz, dh)); r = tensor([.03,.01,0.])
    fn = lambda z, held: settle_torch(held, z, r, rate)['ending_weights']
    _, actual = torch.autograd.functional.jvp(fn, (h, h), (dz, dh))
    a = 1+r; gross = (h*a).sum()
    expected = dz*a/gross - h*a*(dz*a).sum()/gross.square()
    torch.testing.assert_close(actual, expected, atol=4e-15, rtol=0)


@pytest.mark.parametrize('rate', [.001,.0025])
def test_hold_composition_has_zero_fee_gradient_and_correct_state_gradient(rate):
    h = tensor([.2,.6,.2], requires_grad=True); r = tensor([.03,.01,0.])
    result = settle_torch(h, h, r, rate)
    assert float(result['turnover']) == 0 and float(result['fees']) == 0
    assert float(result['open_net_equity']) == 1
    for key in ('fees', 'open_net_equity'):
        grad, = torch.autograd.grad(result[key], h, retain_graph=True)
        torch.testing.assert_close(grad, torch.zeros_like(h), atol=0, rtol=0)
    torch.testing.assert_close(result['ending_weights'], h*(1+r)/(h*(1+r)).sum(), atol=0, rtol=0)


@pytest.mark.parametrize('rate', [.001, .0025])
def test_away_from_kinks_scalar_and_state_derivatives_match_independent_ledger(rate):
    h = np.array([.2,.6,.2]); z = np.array([.6,.2,.2]); r = np.array([.03,-.01,0.])
    dz = np.array([.3,-.1,-.2]); dh = np.array([-.2,.3,-.1]); step = 1e-6
    fn = lambda zt, ht: torch.cat((settle_torch(ht,zt,tensor(r),rate)['net_equity'].reshape(1),
                                  settle_torch(ht,zt,tensor(r),rate)['ending_weights']))
    _, actual = torch.autograd.functional.jvp(fn,(tensor(z),tensor(h)),(tensor(dz),tensor(dh)))
    plus = independent(h+step*dh,z+step*dz,r,rate); minus = independent(h-step*dh,z-step*dz,r,rate)
    expected = (np.r_[plus['net_equity'],plus['ending_weights']]-np.r_[minus['net_equity'],minus['ending_weights']])/(2*step)
    np.testing.assert_allclose(actual,expected,atol=3e-10,rtol=0)


@pytest.mark.parametrize('rate', [.001, .0025])
def test_implicit_fee_derivative_obeys_funding_identity_including_partial_zero_trade(rate):
    for h,z in [(tensor([.2,.6,.2]),tensor([.2,.6,.2])),
                (tensor([0.,.6,.4]),tensor([0.,.2,.8])),
                (tensor([.2,.6,.2]),tensor([.7,.1,.2]))]:
        h=h.requires_grad_();z=z.requires_grad_()
        result=settle_torch(h,z,tensor([.03,.01,0.]),rate)
        residual=result['open_net_equity']+result['fees']-1
        assert abs(float(residual))<1e-15
        gradients=torch.autograd.grad(residual,(z,h))
        for grad in gradients:torch.testing.assert_close(grad,torch.zeros_like(grad),atol=1e-17,rtol=0)


def test_zero_trade_fee_gradient_is_declared_selection_not_all_one_sided_derivatives():
    h=tensor([0.,1.,0.]); z=h.clone().requires_grad_();r=tensor([.03,.01,0.]);rate=.0025
    out=settle_torch(h,z,r,rate); grad,=torch.autograd.grad(out['fees'],z)
    assert torch.count_nonzero(grad)==0
    dz=np.array([1.,-1.,0.]);eps=1e-7
    right=independent(h.numpy(),h.numpy()+eps*dz,r.numpy(),rate)['fees']/eps
    assert right>.004  # A buy/sell departure has positive directional cost.


def test_vectorized_kernel_preserves_own_holdings_and_never_invents_a_trade_band():
    h=tensor([[.2,.6,.2],[0.,1.,0.]])
    z=h.clone();z[0,0]+=1e-10;z[0,2]-=1e-10
    out=settle_torch(h,z,tensor([[.03,.01,0.],[.03,.01,0.]]),.0025)
    assert out['turnover'][0]>0 and out['fees'][0]>0 and out['turnover'][1]==0
    for i in range(2):
        one=settle_torch(h[i],z[i],tensor([.03,.01,0.]),.0025)
        for key in out:torch.testing.assert_close(out[key][i],one[key],atol=0,rtol=0)


@pytest.mark.parametrize('bad', ['nan','negative','unnormalized','return','cost'])
def test_invalid_money_inputs_fail_closed(bad):
    h=tensor([.2,.6,.2]);z=h.clone();r=tensor([.03,.01,0.]);rate=.0025
    if bad=='nan':z[0]=float('nan')
    if bad=='negative':z=tensor([-.1,.6,.5])
    if bad=='unnormalized':z*=.9
    if bad=='return':r[0]=-1.
    if bad=='cost':rate=1.
    with pytest.raises((ValueError,NumericalFitFailure)):settle_torch(h,z,r,rate)
