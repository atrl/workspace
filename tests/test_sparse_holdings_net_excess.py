"""Synthetic sparse action, exact-state and constrained-training contracts."""
from copy import deepcopy
import math
import pickle

import numpy as np
import pytest
import torch

import src.model.sparse_holdings_net_excess as m
from src.model.benchmark_anchor_utility import BenchmarkAnchorPolicy, numpy_rollout as parent_rollout
from src.model.post_fee_settlement import settle_torch


def fixture(n=64):
    rng=np.random.default_rng(601)
    x=rng.normal(size=(n,27));under=.0003+.009*np.sin(np.arange(n)*.73)
    return x,np.c_[2.9*under+rng.normal(0,.0003,n),under]


def spec(**kwargs):
    return m.ConstrainedNetExcessSpec(updates=kwargs.pop('updates',2),minimum_rows=63,**kwargs)


@pytest.mark.parametrize('scores,expected',[
    ([0.,0.,0.,0.],[.25,.25,.25,.25]),
    ([0.,1.,0.,0.],[0.,1.,0.,0.]),
    ([1.,1.,0.,0.],[.5,.5,0.,0.]),
    ([-10.,0.,-10.,2.],[0.,0.,0.,1.]),
])
def test_sparsemax_exact_vertices_and_simplex(scores,expected):
    tensor=torch.tensor(scores,dtype=torch.float64,requires_grad=True)
    q=m.sparsemax(tensor)
    np.testing.assert_array_equal(q.detach().numpy(),expected)
    np.testing.assert_array_equal(m.sparsemax_numpy(scores),expected)
    assert float(q.sum())==1 and bool((q>=0).all())
    jac=torch.autograd.functional.jacobian(m.sparsemax,tensor).numpy()
    support=np.asarray(expected)>0;wanted=np.diag(support.astype(float))-np.outer(support,support)/support.sum()
    np.testing.assert_allclose(jac,wanted,atol=0,rtol=0)


def test_sparsemax_dense_support_finite_difference_translation_and_invalid():
    score=np.array([.2,.3,.1,-.6]);t=torch.tensor(score,dtype=torch.float64,requires_grad=True)
    jac=torch.autograd.functional.jacobian(m.sparsemax,t).numpy();eps=1e-6
    finite=np.column_stack([(m.sparsemax_numpy(score+np.eye(4)[j]*eps)-m.sparsemax_numpy(score-np.eye(4)[j]*eps))/(2*eps) for j in range(4)])
    np.testing.assert_allclose(jac,finite,atol=1e-9,rtol=0)
    np.testing.assert_allclose(m.sparsemax_numpy(score),m.sparsemax_numpy(score+40),atol=1e-14,rtol=0)
    with pytest.raises(m.NumericalFitFailure):m.sparsemax_numpy([0.,np.nan,1.,2.])


@pytest.mark.parametrize('seed',[17,41,71])
def test_parent_class_embedding_initialization_and_full_cost_paths(seed):
    x,r=fixture(100);old=BenchmarkAnchorPolicy(seed=seed);new=m.SparseHoldingsPolicy(seed=seed)
    assert sum(p.numel() for p in new.parameters())==272
    assert torch.equal(old.hidden.weight,new.hidden.weight) and torch.equal(old.hidden.bias,new.hidden.bias)
    rng=np.random.default_rng(91);held=rng.dirichlet(np.ones(3),size=len(x))
    np.testing.assert_allclose(new(torch.from_numpy(x),torch.from_numpy(held)).detach().numpy(),old(torch.from_numpy(x),torch.from_numpy(held)).detach().numpy(),atol=1e-12,rtol=0)
    for cost in [10.,25.]:
        before=parent_rollout(old,x,r,cost_bps=cost);after=m.numpy_rollout(new,x,r,cost_bps=cost)
        for key in ['target_weights','pretrade_weights','ending_weights','fees','net_equity']:
            np.testing.assert_allclose(after[key],before[key],atol=1e-12,rtol=0)
        np.testing.assert_allclose(after['equity']/before['equity'],1.,atol=1e-10,rtol=0)
    # The *whole* legal parent class embeds, including a genuine HOLD vertex.
    for theta in [np.zeros((3,9)),rng.dirichlet(np.ones(28))[:27].reshape(3,9),np.pad([[1.]],((2,0),(0,8)))]:
        with torch.no_grad():old.theta.copy_(torch.from_numpy(theta))
        embedded=m.embed_parent_policy(old)
        np.testing.assert_allclose(embedded(torch.from_numpy(x),torch.from_numpy(held)).detach().numpy(),old(torch.from_numpy(x),torch.from_numpy(held)).detach().numpy(),atol=1e-12,rtol=0)


def test_finite_same_holdings_cash_hold_and_leverage_witness():
    model=m.SparseHoldingsPolicy(anchor=True)
    with torch.no_grad():
        model.hidden.weight.zero_();model.hidden.bias.zero_();model.hidden.weight[0,0]=1.
        model.A.zero_();model.a.zero_();model.K.zero_()
        model.A[0,0]=12.;model.A[2,0]=-12.;model.a[0]=-8.;model.a[2]=4.;model.a[3]=2.
    h=torch.tensor([0.,1.,0.],dtype=torch.float64)
    for value,wanted in [(-math.log(23),[0.,0.,1.]),(0.,[0.,1.,0.]),(math.log(23),[1.,0.,0.])]:
        x=torch.zeros(27,dtype=torch.float64);x[0]=value
        np.testing.assert_array_equal(model(x,h).detach().numpy(),wanted)
    x=np.zeros((5,27));r=np.tile([.03,-.01],(5,1))
    path=m.numpy_rollout(model,x,r,cost_bps=25,initial_weights=np.array([.2,.6,.2]))
    assert path['exact_hold'].all() and path['exact_zero_order'].all()
    np.testing.assert_array_equal(path['fees'],np.zeros(5))
    np.testing.assert_array_equal(path['target_weights'],path['pretrade_weights'])


def test_direct_holdings_skip_has_a_nonempty_exact_hold_region():
    model=m.SparseHoldingsPolicy(anchor=True)
    with torch.no_grad():
        model.A.zero_();model.a.zero_();model.a[3]=2.;model.K.zero_();model.K[0,0]=4.;model.K[2,2]=4.
    x=torch.zeros(27,dtype=torch.float64)
    for h in [[.1,.8,.1],[.25,.5,.25]]:
        held=torch.tensor(h,dtype=torch.float64,requires_grad=True)
        q=model.expert_weights(x,held);np.testing.assert_array_equal(q.detach().numpy(),[0,0,0,1])
        target=model(x,held);np.testing.assert_array_equal(target.detach().numpy(),h)
        jac=torch.autograd.functional.jacobian(lambda z:model(x,z),held)
        torch.testing.assert_close(jac,torch.eye(3,dtype=torch.float64),atol=0,rtol=0)
    for h in [[.4,.5,.1],[.1,.5,.4]]:
        q=model.expert_weights(x,torch.tensor(h,dtype=torch.float64))
        assert not bool(q[3]==1)


def test_exact_anchor_huge_finite_inputs_has_declared_zero_jacobian():
    model=m.zero_anchor();x=torch.full((27,),1e308,dtype=torch.float64,requires_grad=True)
    h=torch.tensor([.2,.6,.2],dtype=torch.float64,requires_grad=True)
    target=model(x,h);np.testing.assert_array_equal(target.detach().numpy(),[0,1,0])
    target.sum().backward()
    assert all(p.grad is not None and torch.count_nonzero(p.grad)==0 for p in model.parameters())
    assert torch.count_nonzero(x.grad)==0 and torch.count_nonzero(h.grad)==0
    np.testing.assert_array_equal(m.policy_target(model,np.full(27,1e308),h.detach().numpy()),[0,1,0])
    payload=m.policy_to_dict(model);restored=m.policy_from_dict(payload)
    assert m.is_exact_anchor(payload) and m.is_exact_anchor(restored)
    payload['state_dict']['a'][0]=1e-8
    with pytest.raises(ValueError,match='hash'):m.policy_from_dict(payload)


def test_causal_same_date_force_and_cost_independent_normalized_state():
    x,r=fixture(80);model=m.SparseHoldingsPolicy()
    with torch.no_grad():model.K[3,1]=.15
    a=m.numpy_rollout(model,x,r,cost_bps=10);b=m.numpy_rollout(model,x,r,cost_bps=25)
    for name in ['expert_weights','target_weights','pretrade_weights','ending_weights']:
        np.testing.assert_array_equal(a[name],b[name])
    mask=np.zeros(80,bool);mask[40]=True
    forced=m.numpy_rollout(model,x,r,cost_bps=10,force_soxx_mask=mask)
    np.testing.assert_array_equal(a['net_equity'][:40],forced['net_equity'][:40])
    np.testing.assert_array_equal(forced['target_weights'][40],[0,1,0])
    np.testing.assert_array_equal(forced['pretrade_weights'][41],forced['ending_weights'][40])
    changed=x.copy();changed[50:]*=-3;rr=r.copy();rr[50:]*=-1
    suffix=m.numpy_rollout(model,changed,rr,cost_bps=10)
    np.testing.assert_array_equal(a['net_equity'][:50],suffix['net_equity'][:50])


def test_shared_holdings_ensemble_targets_are_equal_mean():
    x,r=fixture(12);models=[m.SparseHoldingsPolicy(seed=s) for s in [17,41,71]]
    with torch.no_grad():models[0].K[0,0]=.5;models[1].K[3,2]=.3
    path=m.numpy_rollout(models,x,r,cost_bps=25)
    for i,h in enumerate(path['pretrade_weights']):
        target=np.mean([m.policy_target(model,x[i],h) for model in models],axis=0)
        np.testing.assert_allclose(path['target_weights'][i],target,atol=1e-15,rtol=0)


def test_initial_fit_audit_duals_and_feasible_incumbent_protocol():
    x,r=fixture(64);model,info=m.fit_policy(x,r,seed=17,spec=spec())
    assert info['failure'] is None and len(info['epochs'])==3
    assert info['training_feasibility_passed'] and info['F']>=0
    assert info['initial_path_parity']['worst_F_error']<=1e-12
    best=0.;epoch='anchor'
    for i,h in enumerate(info['epochs']):
        if h['feasible'] and h['F']>best:best=h['F'];epoch=i
        assert h['selected_epoch_so_far']==epoch
        if i:
            np.testing.assert_array_equal(h['lambda_used'],info['epochs'][i-1]['lambda_next'])
            np.testing.assert_allclose(h['lambda_next'],np.maximum(0,np.asarray(h['lambda_used'])+.003*np.asarray(h['postg'])),atol=0,rtol=0)
            assert h['gradient']['support_mismatch_count']==0
            assert max(h['gradient']['forward_parity_by_field'].values())<1e-10
    assert info['selected_epoch']==epoch and info['F']==best
    assert info['optimizer_diagnostics']['parameter_projection'] is False
    assert info['optimizer_state_parameter_epoch']==2


def test_no_legacy_veto_and_explicit_failed_anchor_accounting(monkeypatch):
    x=np.zeros((126,27));under=np.r_[np.full(63,.002),np.full(63,-.003)];r=np.c_[3*under,under]
    cash=[m.SparseHoldingsPolicy(seed=s,anchor=True) for s in [17,41,71]]
    for model in cash:
        with torch.no_grad():model.a.copy_(model.a.new_tensor([0.,0.,1.,0.]))
    selected,ensemble=m.evaluate_ensemble(cash,x,r,spec=spec())
    assert len(selected)==3 and ensemble['F']>0 and ensemble['legacy_J']<0 and not ensemble['anchor_selected']
    def fail(*args,**kwargs):raise m.NumericalFitFailure('synthetic failure')
    monkeypatch.setattr(m,'truncated_utility_gradient',fail)
    x,r=fixture(64);model,info=m.fit_policy(x,r,seed=17,spec=spec())
    assert info['failure']['epoch']==1 and info['qualification_eligible'] is False
    assert info['training_feasibility_passed'] and m.is_exact_anchor(model) and info['F']==0
    assert info['optimizer_state_parameter_epoch']==0


def test_own_state_gradient_route_and_compact_diagnostics():
    x,r=fixture(13)
    cfg=m.ConstrainedNetExcessSpec(updates=1,minimum_rows=4,risk_group_length=4,gradient_chunk_length=4,context_sessions=20)
    model=m.SparseHoldingsPolicy(seed=41)
    with torch.no_grad():model.K.copy_(torch.arange(12,dtype=torch.float64).reshape(4,3)*.001)
    current=m.evaluate_policy(model,x,r,spec=cfg);proxy,_=m.gradient_proxy(current,np.array([.01,.02,.03,.04]),spec=cfg)
    out=m.truncated_utility_gradient(model,x,r,proxy,spec=cfg)
    assert out['K_gradient_norm']>0 and out['head_gradient_norm']>0 and out['encoder_gradient_norm']>0
    assert out['scored_rows']==13 and out['support_mismatch_count']==0
    assert max(out['forward_parity_by_field'].values())<1e-12


def test_bad_inputs_and_schema_fail_closed():
    x,r=fixture();x[0,0]=np.nan
    with pytest.raises(ValueError):m.numpy_rollout(m.SparseHoldingsPolicy(),x,r,cost_bps=10)
    with pytest.raises(ValueError):m.policy_from_dict({'schema_version':1})
    with pytest.raises(ValueError):m.numpy_rollout([],np.zeros((2,27)),np.zeros((2,2)),cost_bps=10)
