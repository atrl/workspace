"""Synthetic constrained-training contracts; no market data or tuning."""
from copy import deepcopy
from dataclasses import replace
import pickle

import numpy as np
import pytest
import torch

import src.model.constrained_net_excess as m
from src.model.benchmark_anchor_utility import BenchmarkAnchorPolicy, policy_to_dict
from src.model.integrated_allocation import _settle


def fixture(n=126):
    rng=np.random.default_rng(203)
    x=rng.normal(size=(n,27))
    underlying=.0003+.008*np.sin(np.arange(n)*.73)
    returns=np.c_[2.9*underlying+rng.normal(0,.0004,n),underlying]
    return x,returns


def small_spec(**kwargs):
    return m.ConstrainedNetExcessSpec(updates=kwargs.pop('updates',2),minimum_rows=63,**kwargs)


def cash_models():
    models=[BenchmarkAnchorPolicy(seed=s,anchor=True) for s in [17,41,71]]
    for model in models:
        with torch.no_grad():model.theta[1,0]=1.
    return models


def test_spec_enforces_new_F_without_mutating_parent_objective():
    with pytest.raises(ValueError,match='zero RMS'):m.ConstrainedNetExcessSpec(downside_coefficient=1.)
    spec=small_spec()
    parent=m._parent_spec(spec,downside=1)
    assert parent.downside_coefficient==1 and spec.downside_coefficient==0
    assert type(parent).__name__=='AnchorUtilitySpec'


@pytest.mark.parametrize('seed',[17,41,71])
def test_initial_tensors_exact_parent_and_analytic_anchor(seed):
    x,r=fixture(64)
    _,info=m.fit_policy(x,r,seed=seed,spec=small_spec(updates=0))
    original=policy_to_dict(BenchmarkAnchorPolicy(seed=seed))
    assert info['initial_tensor_sha256']==original['tensor_sha256']
    assert info['epochs'][0]['checkpoint']['tensor_sha256']==original['tensor_sha256']
    assert info['epochs'][0]['lambda_used']==[0.]*4
    assert info['epochs'][0]['lambda_next']==[0.]*4
    assert not info['epochs'][0]['dual_update_applied']
    for cost,audit in info['anchor_audit'].items():
        assert audit['maximum_multiplier_error']<1e-12
        assert audit['maximum_later_fee']<1e-12
    anchor=m.evaluate_policy(BenchmarkAnchorPolicy(anchor=True),x,r,spec=small_spec())
    assert anchor['F']==0 and anchor['feasible']
    np.testing.assert_array_equal(anchor['normalized_g'],np.zeros(4))


def test_signed_post_primal_dual_update_decreases_and_has_no_cap():
    old=np.array([.01,.2,0.,1.]);g=np.array([-10.,.5,-.3,1000.])
    actual=m.update_multipliers(old,g,learning_rate=.003)
    np.testing.assert_allclose(actual,[0.,.2015,0.,4.],atol=1e-15,rtol=0)
    np.testing.assert_array_equal(old,[.01,.2,0.,1.])
    with pytest.raises(m.NumericalFitFailure):m.update_multipliers(old,[np.nan,0,0,0],learning_rate=.003)


def test_proxy_preserves_F_and_includes_inactive_cost_risk_gradients():
    x,r=fixture(64);spec=small_spec();model=BenchmarkAnchorPolicy(seed=17)
    current=m.evaluate_policy(model,x,r,spec=spec);before=pickle.dumps(current)
    inactive=[c for c,w in current['cost_weights'].items() if w==0]
    assert len(inactive)==1
    dual=np.array([.1,.2,.3,.4]);proxy,hashes=m.gradient_proxy(current,dual,spec=spec)
    assert pickle.dumps(current)==before
    assert proxy['cost_weights']=={10.:1.,25.:1.}
    for i,cost in enumerate(spec.cost_bps):
        expected=np.full(len(x),63/len(x))*current['cost_weights'][cost]-dual[2*i:2*i+2]@current['constraints_by_cost'][cost]['normalized_adjoints']
        np.testing.assert_allclose(proxy['components'][cost]['daily_weights'],expected,atol=1e-14,rtol=0)
    assert np.any(proxy['components'][inactive[0]]['daily_weights']!=0)
    assert len(hashes['normalized_risk_adjoints_sha256'])==64


def torch_lagrangian(model,x,r,evaluation,dual,spec):
    objective=[];constraints=[]
    for cost in spec.cost_bps:
        held=torch.tensor([0.,0.,1.],dtype=torch.float64);multipliers=[]
        for f,ret in zip(torch.from_numpy(x),torch.from_numpy(r)):
            result=_settle(held,model(f,held),torch.cat((ret,torch.zeros(1,dtype=torch.float64))),cost/10000)
            multipliers.append(result['net_equity']);held=result['ending_weights']
        net=torch.stack(multipliers);logs=torch.log(net)
        reference=1+r[:,1].copy();reference[0]/=1+cost/10000
        objective.append(spec.risk_group_length/len(x)*(logs-torch.log(torch.from_numpy(reference))).sum())
        nav=torch.cat((logs.new_zeros(1),logs.cumsum(0)))
        drawdown=-torch.expm1((nav-torch.cummax(nav,dim=0).values).min())
        sorted_returns=torch.sort(net-1).values;mass=len(x)*.05;whole=int(np.floor(mass))
        cvar=-(sorted_returns[:whole].sum()+(mass-whole)*sorted_returns[whole])/mass
        B=torch.from_numpy(evaluation['constraints_by_cost'][cost]['normalizers'])
        constraints.extend([(drawdown-B[0])/B[0],(cvar-B[1])/B[1]])
    return -torch.stack(objective).min()+(torch.stack(constraints)*torch.from_numpy(dual)).sum()


def test_fullcontext_combined_gradient_matches_whole_lagrangian_autograd():
    x,r=fixture(13)
    spec=m.ConstrainedNetExcessSpec(updates=1,minimum_rows=4,risk_group_length=4,gradient_chunk_length=4,context_sessions=20)
    model=BenchmarkAnchorPolicy(seed=41);current=m.evaluate_policy(model,x,r,spec=spec);dual=np.array([.01,.02,.03,.04])
    proxy,_=m.gradient_proxy(current,dual,spec=spec)
    diagnostic=m.truncated_utility_gradient(model,x,r,proxy,spec=m._parent_spec(spec,downside=0))
    actual=[p.grad.clone() for p in model.parameters()]
    assert diagnostic['forward_parity_max_error']<1e-12
    model.zero_grad(set_to_none=True);loss=torch_lagrangian(model,x,r,current,dual,spec);loss.backward()
    for left,p in zip(actual,model.parameters()):torch.testing.assert_close(left,p.grad,atol=2e-12,rtol=1e-9)
    assert model.hidden.weight.grad.norm()>0


def test_positive_F_feasible_negative_legacy_J_is_eligible_including_ensemble():
    x=np.zeros((126,27));under=np.r_[np.full(63,.002),np.full(63,-.003)];r=np.c_[3*under,under]
    models=cash_models();spec=small_spec()
    evaluation=m.evaluate_policy(models,x,r,spec=spec)
    assert evaluation['F']>0 and evaluation['legacy_J']<0 and evaluation['feasible']
    assert m.feasible_improvement(evaluation,0.)
    selected,audit=m.evaluate_ensemble(models,x,r,spec=spec)
    assert len(selected)==3 and not audit['anchor_selected']
    assert audit['training_feasibility_passed'] and audit['legacy_J']<0
    assert audit['centered_worst_utility']>0


def test_ensemble_tail_failure_vetoes_positive_mean_excess():
    x=np.zeros((126,27));under=.001+.009*np.sin(np.arange(126)*.72);r=np.c_[3*under,under]
    models=[BenchmarkAnchorPolicy(seed=s,anchor=True) for s in [17,41,71]]
    for model in models:
        with torch.no_grad():model.theta[0,0]=1.
    raw=m.evaluate_policy(models,x,r,spec=small_spec())
    assert raw['F']>0 and not raw['feasible']
    chosen,audit=m.evaluate_ensemble(models,x,r,spec=small_spec())
    assert len(chosen)==1 and audit['anchor_selected']
    assert audit['training_feasibility_passed'] and audit['centered_worst_utility']==0
    assert not audit['ensemble_feasible'] and audit['failure'] is None


def test_fit_records_exact_dual_order_fixed_objective_incumbent_and_compact_history():
    x,r=fixture(126);spec=small_spec(updates=3)
    model,info=m.fit_policy(x,r,seed=17,spec=spec)
    assert info['failure'] is None and len(info['epochs'])==4
    assert info['training_feasibility_passed'] and info['centered_worst_utility']>=0
    best=0.;epoch='anchor'
    for i,h in enumerate(info['epochs']):
        if h['feasible'] and h['F']>best:best=h['F'];epoch=i
        assert h['selected_epoch_so_far']==epoch
        if i:
            prev=info['epochs'][i-1]
            np.testing.assert_array_equal(h['lambda_used'],prev['lambda_next'])
            np.testing.assert_allclose(h['lambda_next'],np.maximum(0,np.array(h['lambda_used'])+.003*np.array(h['postg'])),atol=0,rtol=0)
            assert h['pre_primal_lagrangian']==pytest.approx(-prev['F']+np.dot(h['lambda_used'],prev['postg']))
            assert h['post_primal_lagrangian_at_lambda_used']==pytest.approx(-h['F']+np.dot(h['lambda_used'],h['postg']))
            assert 'chunk_boundaries' not in h['gradient'] and 'daily_weights' not in h['gradient']
        assert 'normalized_adjoints' not in h['risks_by_cost'][10.]
    assert info['selected_epoch']==epoch and info['centered_worst_utility']==best
    assert info['final_optimizer_state'] is not None and info['optimizer_state_parameter_epoch']==3
    assert info['optimizer_initial_settings']['weight_decay']==0
    assert info['gradient_chunk_boundaries']
    if epoch!='anchor':assert policy_to_dict(model)['tensor_sha256']==info['epochs'][epoch]['checkpoint']['tensor_sha256']


def test_declared_failure_is_feasible_anchor_accounting_but_disqualified(monkeypatch):
    x,r=fixture(64)
    def fail(*args,**kwargs):raise m.NumericalFitFailure('synthetic gradient failure')
    monkeypatch.setattr(m,'gradient_proxy',fail)
    model,info=m.fit_policy(x,r,seed=17,spec=small_spec())
    assert info['failure']['epoch']==1 and not info['qualification_eligible']
    assert info['selected_epoch']=='anchor' and info['training_feasibility_passed']
    assert info['F']==0 and torch.count_nonzero(model.theta)==0
    assert info['optimizer_state_parameter_epoch']==0
    assert info['optimizer_diagnostics']['updates_completed']==0
    def bug(*args,**kwargs):raise RuntimeError('programming error')
    monkeypatch.setattr(m,'gradient_proxy',bug)
    with pytest.raises(RuntimeError,match='programming error'):m.fit_policy(x,r,seed=17,spec=small_spec())


def test_invalid_reference_has_no_epsilon_rescue_or_fake_qualified_null():
    x=np.zeros((64,27));r=np.tile([.03,.01],(64,1))
    _,info=m.fit_policy(x,r,seed=17,spec=small_spec())
    assert info['failure']['epoch']=='reference'
    assert not info['qualification_eligible'] and info['null_selected']
    assert info['training_feasibility_passed']  # Same raw null/benchmark, despite unusable optimizer normalizers.
    assert info['selected_normalized_g'] is None
    assert not info['epochs'] and info['final_optimizer_state'] is None


def test_repeatability_and_no_state_warmstart():
    x,r=fixture(64);spec=small_spec(updates=1)
    first,a=m.fit_policy(x,r,seed=71,spec=spec);second,b=m.fit_policy(x,r,seed=71,spec=spec)
    assert policy_to_dict(first)['tensor_sha256']==policy_to_dict(second)['tensor_sha256']
    assert [h['F'] for h in a['epochs']]==[h['F'] for h in b['epochs']]
    assert a['epochs'][0]['lambda_used']==b['epochs'][0]['lambda_used']==[0.]*4


def test_registered_optimizer_recognizes_fixed_synthetic_known_information():
    # Declared once before execution: this is neither market tuning nor an
    # assertion that projected Adam reaches the known feasible class optimum.
    n=504;sign=np.where(np.arange(n)%2==0,1.,-1.)
    x=np.zeros((n,27));x[:,0]=sign
    underlying=.002+.008*sign;r=np.c_[3*underlying,underlying]
    spec=m.ConstrainedNetExcessSpec()
    witness=BenchmarkAnchorPolicy(seed=17,anchor=True)
    with torch.no_grad():
        witness.hidden.weight.zero_();witness.hidden.bias.zero_()
        witness.hidden.weight[:4,0]=2.;witness.hidden.weight[4:,0]=-2.
        witness.theta[0,1:5]=.125;witness.theta[1,5:9]=.125
    witness_eval=m.evaluate_policy(witness,x,r,spec=spec)
    assert witness_eval['F']>0 and witness_eval['feasible']
    models=[]
    for seed in [17,41,71]:
        model,info=m.fit_policy(x,r,seed=seed,spec=spec)
        assert len(info['epochs'])==121 and info['failure'] is None
        assert info['F']>0 and info['training_feasibility_passed'] and not info['null_selected']
        models.append(model)
    _,ensemble=m.evaluate_ensemble(models,x,r,spec=spec)
    assert ensemble['F']>0 and ensemble['training_feasibility_passed']
