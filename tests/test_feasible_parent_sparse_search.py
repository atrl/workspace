"""Synthetic algebra, physical incumbents and finite evaluation-budget checks.

No registered504-row parent/COBYLA campaign is executed by these tests.
"""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.model.benchmark_anchor_utility import BenchmarkAnchorPolicy, policy_to_dict as parent_payload
import src.model.feasible_parent_sparse_search as m
from src.model.sparse_holdings_net_excess import evaluate_policy as independent_evaluate


def fixture(n=64):
    rng = np.random.default_rng(921)
    x = rng.normal(size=(n,27))
    s = .0004 + .009*np.sin(np.arange(n)*.71)
    return x, np.c_[2.95*s,s]


def conditional_fixture(n=64):
    x=np.zeros((n,27)); x[:,0]=np.where(np.arange(n)%2==0,1.,-1.)
    s=.002+.008*x[:,0]
    return x,np.c_[3*s,s]


def spec(**kw):
    return m.SearchSpec(minimum_rows=63, max_evaluations=kw.pop('max_evaluations',3),
                        scipy_maxiter=kw.pop('scipy_maxiter',3), **kw)


def witness(seed=17, scale=1.):
    p=BenchmarkAnchorPolicy(seed=seed)
    with torch.no_grad():
        p.hidden.weight.zero_(); p.hidden.bias.zero_()
        p.hidden.weight[:4,0]=2.; p.hidden.weight[4:,0]=-2.
        p.theta.zero_(); p.theta[0,1:5]=.125*scale; p.theta[1,5:9]=.125*scale
    return p


def unchanged_solver(fun,x0,*,method,constraints,options):
    assert method=='COBYLA' and options['catol']==0
    constraints[0]['fun'](x0)
    f=fun(x0)
    return SimpleNamespace(success=False,status=3,message='fixed evaluation budget',nfev=1,
                           fun=f,maxcv=0.,x=x0.copy())


def test_registered_defaults_and_invalid_specs():
    s=m.SearchSpec()
    assert (s.max_evaluations,s.scipy_maxiter,s.rhobeg,s.tol,s.catol)==(2000,2000,1.,1e-6,0.)
    for kwargs in [{'max_evaluations':0},{'scipy_maxiter':0},{'minimum_rows':62},
                   {'catol':1e-5},{'rhobeg':float('nan')},{'cost_bps':(0.,25.)},
                   {'risk_tolerance_percentage_points':1e-8}]:
        with pytest.raises(ValueError):m.SearchSpec(**kwargs)


def test_two_gauges_preserve_arbitrary_holdings_targets_and_encoder():
    model=m.SparseHoldingsPolicy(seed=41)
    rng=np.random.default_rng(390)
    with torch.no_grad():
        model.A.copy_(torch.tensor(rng.normal(size=(4,8))))
        model.a.copy_(torch.tensor(rng.normal(size=4)))
        model.K.copy_(torch.tensor(rng.normal(size=(4,3))))
    coords=m.gauge_coordinates(model); rebuilt=m.model_from_coordinates(model,coords)
    assert coords.shape==(33,) and m.encoder_hash(rebuilt)==m.encoder_hash(model)
    assert torch.count_nonzero(rebuilt.A[1])==0 and rebuilt.a[1]==0
    assert torch.count_nonzero(rebuilt.K[1])==0 and torch.count_nonzero(rebuilt.K[:,1])==0
    x=rng.normal(size=(40,27));h=rng.dirichlet([1,1,1],size=40)
    with torch.no_grad():
        a=model(torch.tensor(x),torch.tensor(h)).numpy()
        b=rebuilt(torch.tensor(x),torch.tensor(h)).numpy()
    np.testing.assert_allclose(a,b,atol=2e-15,rtol=0)
    np.testing.assert_array_equal(m.gauge_coordinates(rebuilt),coords)
    assert not any(p.requires_grad for p in rebuilt.parameters())


def test_gauge_equivalent_null_and_canonical_return_fastpath():
    old=BenchmarkAnchorPolicy(seed=17,anchor=True)
    _, template=m.prepare_parent(parent_payload(old),seed=17)
    head=m.gauge_coordinates(template).reshape(3,11)
    np.testing.assert_array_equal(head[:,8],[-1.,-1.,-1.])
    x,h=np.ones(27),np.array([.2,.5,.3])
    np.testing.assert_array_equal(m.policy_target(template,x,h),[0,1,0])
    canonical=m._anchor_like(template)
    assert m.is_exact_anchor(canonical)
    np.testing.assert_array_equal(m.policy_target(canonical,np.full(27,1e308),h),[0,1,0])
    assert m.encoder_hash(canonical)==m.encoder_hash(template)


def test_finite_head_move_can_leave_singleton_without_gradient():
    _, template=m.prepare_parent(BenchmarkAnchorPolicy(seed=17,anchor=True))
    z=m.gauge_coordinates(template)
    initial=m.policy_target(template,np.zeros(27),np.array([0,1,0]))
    z[8]+=2.
    altered=m.model_from_coordinates(template,z)
    np.testing.assert_array_equal(initial,[0,1,0])
    np.testing.assert_array_equal(m.policy_target(altered,np.zeros(27),[0,1,0]),[1,0,0])


def test_full_parent_paths_and_serialization_roundtrip():
    x,r=conditional_fixture();p=witness()
    template,evaluation,audit=m.revalidate_parent(parent_payload(p),x,r,spec=spec(),seed=17)
    assert audit['passed'] and evaluation['feasible'] and evaluation['F']>0
    assert audit['maximum_A_F_error']<=1e-12
    roundtrip=m.policy_from_dict(m.policy_to_dict(template))
    for cost in [10,25]:
        a=m.numpy_rollout(template,x,r,cost_bps=cost)
        b=m.numpy_rollout(roundtrip,x,r,cost_bps=cost)
        for field in a:np.testing.assert_array_equal(a[field],b[field])


@pytest.mark.parametrize('count',[1,3])
def test_paired_path_full_objective_and_raw_risk_exact_reference_parity(count):
    x,r=fixture(137);models=[m.SparseHoldingsPolicy(seed=s) for s in [17,41,71]][:count]
    for i,model in enumerate(models):
        with torch.no_grad():model.K.copy_(torch.tensor(np.random.default_rng(10+i).normal(0,.2,(4,3))))
    paired=m.evaluate_policy(models,x,r,spec=spec())
    independent=independent_evaluate(models,x,r,spec=spec().evaluation_spec())
    for key in ['F','A_by_cost','legacy_J','legacy_centered_utility_by_cost','feasible']:
        assert paired[key]==independent[key]
    np.testing.assert_array_equal(paired['normalized_g'],independent['normalized_g'])
    for cost in [10.,25.]:
        for field in paired['paths'][cost]:
            np.testing.assert_array_equal(paired['paths'][cost][field],independent['paths'][cost][field])
        np.testing.assert_array_equal(paired['constraints_by_cost'][cost]['raw_percentage_point_gaps'],
                                      independent['constraints_by_cost'][cost]['raw_percentage_point_gaps'])


def test_source_seed_and_infeasible_parent_block_before_solver(monkeypatch):
    x,r=conditional_fixture();p=witness()
    with pytest.raises(m.ParentContractError,match='seed'):
        m.prepare_parent(parent_payload(p),seed=41)
    with torch.no_grad():p.theta.zero_();p.theta[0,0]=1.
    monkeypatch.setattr(m,'minimize',lambda *a,**kw:pytest.fail('must not start solver'))
    with pytest.raises(m.ParentContractError,match='risk'):
        m.fit_policy(x,r,parent_model=p,seed=17,spec=spec())


def test_shared_exact_byte_cache_and_hard_budget(monkeypatch):
    x,r=fixture();p=BenchmarkAnchorPolicy(seed=17,anchor=True)
    _,anchors,references,_=m._reference(x,r,spec(max_evaluations=2),17)
    template,e,a=m.revalidate_parent(p,x,r,spec=spec(),anchors=anchors,references=references)
    evaluator=m.SearchEvaluator(template,x,r,spec=spec(max_evaluations=2),anchors=anchors,references=references,parent_evaluation=e)
    head=m.gauge_coordinates(template)
    for _ in range(2): evaluator.objective(head);evaluator.constraints(head)
    assert len(evaluator.trials)==1 and evaluator.cache_hits==4
    changed=head.copy();changed[0]=np.nextafter(changed[0],1.)
    evaluator.constraints(changed);evaluator.objective(changed)
    assert len(evaluator.trials)==2 and evaluator.cache_hits==5
    changed2=head.copy();changed2[1]=.1
    with pytest.raises(m.EvaluationBudgetExceeded):evaluator.objective(changed2)
    assert len(evaluator.trials)==2
    assert all('paths' not in t and t['coordinates'].shape==(33,) for t in evaluator.trials)


def test_actual_small_cobyla_run_obeys_cap_and_retains_raw_feasible_policy():
    x,r=fixture();p=BenchmarkAnchorPolicy(seed=17,anchor=True)
    with pytest.warns(UserWarning,match='MAXFUN'):
        model,a=m.fit_policy(x,r,parent_model=p,seed=17,spec=spec(max_evaluations=2,scipy_maxiter=2))
    assert a['unique_evaluations']==2 and a['termination']=='unique_evaluation_budget'
    assert a['failure'] is None and a['training_feasible'] and a['F']>=0
    assert a['frozen_encoder_sha256_before']==a['frozen_encoder_sha256_after']
    assert 'epochs' not in a and 'final_optimizer_state' not in a


def test_null_ties_and_solver_non_success_are_not_numerical_failures(monkeypatch):
    monkeypatch.setattr(m,'minimize',unchanged_solver)
    x,r=fixture();p=BenchmarkAnchorPolicy(seed=41,anchor=True)
    model,a=m.fit_policy(x,r,parent_model=p,seed=41,spec=spec())
    assert a['selected_origin']=='null' and m.is_exact_anchor(model) and a['F']==0
    assert a['failure'] is None and a['qualification_eligible'] and a['cache_hits']==2
    np.testing.assert_array_equal(a['selected_normalized_g'],np.zeros(4))


def test_positive_parent_inheritance_is_not_new_optimizer_edge(monkeypatch):
    monkeypatch.setattr(m,'minimize',unchanged_solver)
    x,r=conditional_fixture();p=witness()
    model,a=m.fit_policy(x,r,parent_model=p,seed=17,spec=spec())
    assert a['selected_origin']=='parent' and a['F']>0 and a['F_delta']==0
    assert a['selected_trial']=='parent' and a['unique_evaluations']==1
    assert m.encoder_hash(model)==m.encoder_hash(p)


def test_positive_F_but_tail_infeasible_proposal_is_rejected(monkeypatch):
    def propose(fun,x0,*,method,constraints,options):
        z=np.zeros(33);z.reshape(3,11)[:,8]=[1.,-2.,-2.]
        f=fun(z);g=constraints[0]['fun'](z)
        assert f<0 and min(g)<0
        return SimpleNamespace(success=True,status=1,message='mock optimizer claims success',nfev=1,
                               fun=f,maxcv=float(max(-g)),x=z)
    monkeypatch.setattr(m,'minimize',propose)
    x,r=conditional_fixture();p=BenchmarkAnchorPolicy(seed=17,anchor=True)
    model,a=m.fit_policy(x,r,parent_model=p,seed=17,spec=spec())
    assert a['trials'][1]['F']>0 and not a['trials'][1]['feasible']
    assert a['selected_origin']=='null' and a['F']==0 and m.is_exact_anchor(model)


def test_nonfinite_trial_is_explicit_failure_and_keeps_valid_parent(monkeypatch):
    def fail(fun,x0,**kw):
        bad=x0.copy();bad[0]=np.nan;fun(bad)
    monkeypatch.setattr(m,'minimize',fail)
    x,r=conditional_fixture();p=witness()
    model,a=m.fit_policy(x,r,parent_model=p,seed=17,spec=spec())
    assert a['failure'] is not None and not a['qualification_eligible']
    assert a['failure']['kind']=='numerical_search_failure' and a['failure']['accounting_only']
    assert a['accounting_only']
    assert a['training_feasible'] and a['selected_origin']=='parent' and a['unique_evaluations']==1
    assert np.isnan(a['failure']['requested_coordinates'][0])


def test_ensemble_compares_whole_parent_candidate_and_null_books():
    x,r=conditional_fixture();parents=[witness(s,.5) for s in [17,41,71]]
    candidates=[m.prepare_parent(witness(s,1.))[1] for s in [17,41,71]]
    selected,a=m.evaluate_ensemble(candidates,x,r,parent_models=parents,spec=spec())
    assert a['selected_origin']=='search' and a['candidate_feasible']
    assert a['candidate_minus_parent_F']>1e-8 and len(selected)==3
    smaller=[m._anchor_like(c) for c in candidates]
    selected,b=m.evaluate_ensemble(smaller,x,r,parent_models=parents,spec=spec())
    assert b['selected_origin']=='parent' and b['candidate_F']==0 and b['candidate_minus_parent_F']<0
    assert b['F']==b['parent_ensemble_F']


def test_parent_ensemble_ties_win_and_encoder_binding_is_enforced():
    x,r=conditional_fixture();parents=[witness(s) for s in [17,41,71]]
    candidates=[m.prepare_parent(p)[1] for p in parents]
    _,a=m.evaluate_ensemble(candidates,x,r,parent_models=parents,spec=spec())
    assert a['selected_origin']=='parent' and a['candidate_minus_parent_F']==0
    bad=deepcopy(candidates)
    with torch.no_grad():bad[1].hidden.bias[0]+=1.
    with pytest.raises(m.ParentContractError,match='encoder'):
        m.evaluate_ensemble(bad,x,r,parent_models=parents,spec=spec())


def test_healthy_null_ensemble_has_measured_zero_normalized_g():
    x,r=fixture();parents=[BenchmarkAnchorPolicy(seed=s,anchor=True) for s in [17,41,71]]
    candidates=[m._anchor_like(m.prepare_parent(p)[1]) for p in parents]
    selected,a=m.evaluate_ensemble(candidates,x,r,parent_models=parents,spec=spec())
    assert a['selected_origin']=='null' and len(selected)==1 and a['failure'] is None
    np.testing.assert_array_equal(a['selected_normalized_g'],np.zeros(4))


@pytest.mark.parametrize('exception',[RuntimeError,ValueError,TypeError])
def test_unexpected_solver_exception_retains_evidence_for_campaign_abort(monkeypatch,exception):
    def broken(fun,x0,**kwargs):
        changed=x0.copy();changed[0]+=.01;fun(changed)
        raise exception('unexpected solver contract error')
    monkeypatch.setattr(m,'minimize',broken)
    x,r=conditional_fixture()
    model,a=m.fit_policy(x,r,parent_model=witness(),seed=17,spec=spec())
    assert a['failure']['kind']=='unexpected_solver_exception'
    assert a['failure']['campaign_abort_required'] and a['failure']['accounting_only']
    assert exception.__name__ in a['failure']['traceback']
    assert a['trial_count']==2 and a['attempted_full_path_evaluations']==2
    assert a['training_feasible'] and a['F']>=a['parent_F']
    assert not a['qualification_eligible'] and a['termination']=='unexpected_solver_exception'


@pytest.mark.parametrize('exception',[KeyboardInterrupt,SystemExit])
def test_process_interrupt_is_not_swallowed_by_solver_failure_boundary(monkeypatch,exception):
    def interrupt(*args,**kwargs):raise exception('requested stop')
    monkeypatch.setattr(m,'minimize',interrupt)
    x,r=fixture()
    with pytest.raises(exception):
        m.fit_policy(x,r,parent_model=BenchmarkAnchorPolicy(seed=17,anchor=True),seed=17,spec=spec())
