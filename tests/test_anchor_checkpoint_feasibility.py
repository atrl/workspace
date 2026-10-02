"""Synthetic saved-state diagnostics; no optimization or market evaluation."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import pickle

import numpy as np
import pandas as pd
import pytest
import torch

import src.analysis.anchor_checkpoint_feasibility as a
from src.model.benchmark_anchor_utility import (
    AnchorUtilitySpec, BenchmarkAnchorPolicy, evaluate_policy, policy_from_dict, policy_to_dict, risk_groups,
)


def fixture(n=64):
    rng = np.random.default_rng(1301)
    x = rng.normal(size=(n,27))
    under = .0008 + .008*np.sin(np.arange(n)*.71)
    returns = np.c_[2.9*under-.0001,under]
    calendar = pd.bdate_range("2015-01-05",periods=n+2)
    dates = pd.DataFrame({"signal_date":calendar[:-2],"trade_date":calendar[1:-1],"exit_date":calendar[2:]})
    spec = AnchorUtilitySpec(updates=2,minimum_rows=63)
    history = []
    for epoch,expert in enumerate([None,1,0]):
        model = BenchmarkAnchorPolicy(seed=17,anchor=True)
        if expert is not None:
            with torch.no_grad():model.theta[expert,0]=1.
        result = evaluate_policy(model,x,returns,spec=spec)
        history.append({"epoch":epoch,"checkpoint":policy_to_dict(model),"J":result["J"],"centered_utility":result["centered_utility"]})
    metadata = {"variant":"price","year":2016,"seed":17,"failure":None,"qualification_eligible":True,
        "training_rows":n,"training_feature_sha256":hashlib.sha256(x.tobytes()).hexdigest(),
        "training_return_sha256":hashlib.sha256(returns.tobytes()).hexdigest(),"spec":asdict(spec),
        "risk_groups":risk_groups(n),"epochs":history,"selected_epoch":"anchor"}
    diagnostic = a.CheckpointFeasibilitySpec(expected_epochs=(0,1,2),minimum_rows=63)
    return x,returns,dates,metadata,diagnostic


def run_fixture(parts):
    x,r,d,m,s=parts
    return a.evaluate_checkpoint_fit(x,r,d,m,variant="price",model_year=2016,seed=17,spec=s)


def metric(gap=0,excess=-.02):
    return {"max_drawdown_pct":gap,"benchmark_max_drawdown_pct":0.,
            "cvar5_pct":gap,"benchmark_cvar5_pct":0.,"log_excess_sum":excess}


def status(epoch,J,explicit=False):
    return {"epoch":epoch,"explicit_null":explicit,"J":J,"stored_J":J,"J_sign_disagreement":False,
            **a.classify_checkpoint(J,{10.:metric(),25.:metric()},spec=a.CheckpointFeasibilitySpec())}


def test_complete_saved_inventory_and_independent_fee_funded_null():
    parts=fixture();result=run_fixture(parts)
    metrics,states=result['checkpoint_metrics'],result['checkpoint_status']
    assert len(metrics)==8 and len(states)==4
    assert states.epoch.tolist()==['exact_SOXX_null',0,1,2]
    assert int(metrics.explicit_null.sum())==2 and int(states.explicit_null.sum())==1
    null=metrics[metrics.explicit_null]
    assert null.J.eq(0).all() and null.centered_utility.eq(0).all()
    assert null.sum_session_fee_rates.to_numpy()==pytest.approx(np.array([.001/1.001,.0025/1.0025]))
    assert result['fit_summary']['learner_checkpoint_count']==3
    assert result['fit_summary']['null_risk_feasible']
    assert result['audit']['maximum_independent_benchmark_multiplier_error']<1e-12
    assert result['audit']['global_mean_observations']==64
    assert set(result['benchmark_risk_reports'])=={10.,25.}
    assert all('max_drawdown' in report.index for report in result['benchmark_risk_reports'].values())
    assert states.loc[states.epoch.eq(0),'theta_is_zero'].item()  # A saved zero-theta state is still an inventoried learner.
    assert not states.loc[states.epoch.eq(0),'explicit_null'].item()
    assert 'model' not in result and 'selected_model' not in result


def test_risk_tolerance_is_in_percentage_points_and_has_no_J_rescue():
    spec=a.CheckpointFeasibilitySpec();tol=spec.risk_tolerance_percentage_points
    exactly={10.:metric(-tol),25.:metric(-tol)}
    flags=a.classify_checkpoint(-5e-11,exactly,spec=spec)
    assert flags['risk_feasible'] and not flags['risk_and_nonnegative_J']
    failed={10.:metric(np.nextafter(-tol,-np.inf)),25.:metric()}
    assert not a.classify_checkpoint(10.,failed,spec=spec)['risk_feasible']
    for J,positive,margin in [(0.,False,False),(1e-12,True,False),(1e-8,True,False),(1.01e-8,True,True)]:
        flags=a.classify_checkpoint(J,{10.:metric(),25.:metric()},spec=spec)
        assert flags['risk_and_strict_positive_J']==positive
        assert flags['risk_and_positive_J_margin']==margin
        assert not flags['both_costs_raw_positive_log_excess']


def test_raw_count_classes_null_exclusion_and_ties_are_diagnostic():
    rows=[status('exact_SOXX_null',0.,True),*[status(i,j) for i,j in enumerate([-1e-12,0.,1e-12,1e-8,1.01e-8])]]
    result=a.summarize_checkpoint_statuses(pd.DataFrame(rows))
    assert result['learner_checkpoint_count']==5
    assert [result[k+'_count'] for k in ['risk_feasible','risk_and_nonnegative_J','risk_and_strict_positive_J','risk_and_positive_J_margin']]==[5,4,3,1]
    assert result['first_risk_and_nonnegative_J_epoch']==1
    assert result['null_inclusive_best_risk_feasible_epoch']==4
    rows=[status('exact_SOXX_null',0.,True),status(0,-3.),status(1,-1.),status(2,-1.)]
    result=a.summarize_checkpoint_statuses(pd.DataFrame(rows))
    assert result['best_risk_feasible_epoch']==1 and result['best_risk_feasible_J']==-1.
    assert result['null_inclusive_best_risk_feasible_epoch']=='exact_SOXX_null'
    assert result['null_inclusive_best_risk_feasible_J']==0
    assert result['risk_and_nonnegative_J_count']==0
    rows.append(status(3,0.))
    assert a.summarize_checkpoint_statuses(pd.DataFrame(rows))['null_inclusive_best_risk_feasible_epoch']=='exact_SOXX_null'


def test_actual_cached_utility_is_reproduced_and_wrong_scalar_retains_paths():
    parts=fixture();parts[3]['epochs'][1]['J']+=.01
    parts[3]['epochs'][1]['centered_utility']={c:v+.01 for c,v in parts[3]['epochs'][1]['centered_utility'].items()}
    with pytest.raises(a.CheckpointIntegrityError,match='recomputed cached utility') as caught:run_fixture(parts)
    evidence=caught.value.evidence
    assert evidence['epoch']==1 and set(evidence['paths'])=={10.,25.}
    assert all(len(path['net_equity'])==64 for path in evidence['paths'].values())
    restored=pickle.loads(pickle.dumps(caught.value))
    assert restored.evidence['epoch']==1


def test_within_parity_tolerance_sign_difference_does_not_change_raw_gate():
    parts=fixture();h=parts[3]['epochs'][0]
    h['J']=5e-12;h['centered_utility']={10.:5e-12,25.:5e-12}
    result=run_fixture(parts);row=result['checkpoint_status'].loc[lambda x:x.epoch.eq(0)].iloc[0]
    assert row.J==0 and row.stored_J>0 and row.J_sign_disagreement
    assert row.risk_and_nonnegative_J and not row.risk_and_strict_positive_J
    assert row.recomputed_minus_stored_J==-5e-12


@pytest.mark.parametrize('mutation',['missing','duplicate','extra','reordered'])
def test_exact_epoch_inventory_fails_closed(mutation):
    parts=fixture();h=parts[3]['epochs']
    if mutation=='missing':h.pop(1)
    elif mutation=='duplicate':h[1]=deepcopy(h[0])
    elif mutation=='extra':h.append(deepcopy(h[-1]))
    else:h[0],h[1]=h[1],h[0]
    with pytest.raises(a.CheckpointIntegrityError,match='cached epochs'):run_fixture(parts)


@pytest.mark.parametrize('mutation',['features','returns','tensor','groups'])
def test_input_checkpoint_and_risk_group_identity_failures(mutation):
    parts=fixture()
    if mutation=='features':parts[0][1,1]+=1
    elif mutation=='returns':parts[1][1,1]+=.01
    elif mutation=='tensor':parts[3]['epochs'][0]['checkpoint']['state_dict']['theta'][0,0]=.2
    else:parts[3]['risk_groups']=[(0,63)]
    with pytest.raises(a.CheckpointIntegrityError):run_fixture(parts)


def test_chronology_scope_and_native_dates_are_not_silently_trimmed():
    parts=fixture();parts[2].loc[0,'exit_date']+=pd.Timedelta(days=1)
    with pytest.raises(ValueError,match='noncontiguous'):run_fixture(parts)
    parts=fixture();parts[2]['exit_date']=parts[2].exit_date+pd.Timedelta(days=365)
    with pytest.raises(ValueError):run_fixture(parts)
    x,r,d,m,s=fixture()
    with pytest.raises(a.CheckpointIntegrityError,match='development fit inventory'):
        a.evaluate_checkpoint_fit(x,r,d,m,variant='price',model_year=2023,seed=17,spec=s)


def test_no_fitting_or_optimizer_route_and_parent_inputs_are_immutable(monkeypatch):
    import src.model.benchmark_anchor_utility as model_module
    parts=fixture();before=pickle.dumps(parts)
    def forbidden(*args,**kwargs):raise AssertionError('fitting prohibited')
    monkeypatch.setattr(model_module,'fit_policy',forbidden)
    monkeypatch.setattr(torch.optim.Adam,'step',forbidden)
    result=run_fixture(parts)
    assert pickle.dumps(parts)==before
    assert not result['audit']['model_fitting_performed']
    assert not result['audit']['out_of_sample_rows_used']
    assert not result['audit']['deployable_model_exported']


def test_partial_final_group_and_exact_multiple_are_preserved():
    for n,groups in [(64,[(0,63),(1,64)]),(126,[(0,63),(63,126)])]:
        result=run_fixture(fixture(n))
        assert result['audit']['risk_groups']==groups
        assert result['audit']['global_mean_observations']==n


def test_default_full121_inventory_does_not_deduplicate_equal_states():
    x,r,d,metadata,_=fixture(252)
    examples=metadata['epochs']
    metadata['epochs']=[{**deepcopy(examples[i%3]),'epoch':i} for i in range(121)]
    metadata['spec']['updates']=120
    result=a.evaluate_checkpoint_fit(x,r,d,metadata,variant='price',model_year=2016,seed=17)
    assert len(result['checkpoint_status'])==122
    assert len(result['checkpoint_metrics'])==244
    assert result['fit_summary']['learner_checkpoint_count']==121
    assert result['checkpoint_status'].explicit_null.sum()==1


def test_independent_analytic_reference_matches_registered_benchmark_path():
    from src.workflow.stable_excess_integration import benchmark_path
    _,ret,dates,_,_=fixture()
    original=dates.assign(soxl_return=ret[:,0],soxx_return=ret[:,1])
    for cost in [10.,25.]:
        analytic=a._independent_benchmark(dates,ret,cost)
        registered=benchmark_path(original,cost_bps=cost)
        for field in ['strategy_return','fees','turnover']:
            np.testing.assert_allclose(analytic[field],registered[field],atol=1e-12,rtol=0)
        np.testing.assert_array_equal(analytic.exit_date,registered.exit_date)


def test_initial_capital_drawdown_and_fractional_tail_mass_are_not_lost():
    x,ret,dates,metadata,spec=fixture()
    ret[:]=0.;ret[0,0]=-.2;ret[1,0]=.25
    metadata['training_return_sha256']=hashlib.sha256(ret.tobytes()).hexdigest()
    for checkpoint in metadata['epochs']:
        evaluated=evaluate_policy(policy_from_dict(checkpoint['checkpoint']),x,ret,spec=AnchorUtilitySpec(**metadata['spec']))
        checkpoint['J']=evaluated['J'];checkpoint['centered_utility']=evaluated['centered_utility']
    result=a.evaluate_checkpoint_fit(x,ret,dates,metadata,variant='price',model_year=2016,seed=17,spec=spec)
    for cost in [10.,25.]:
        row=result['checkpoint_metrics'].loc[lambda df:df.epoch.eq(2)&df.cost_bps.eq(cost)].iloc[0]
        first_loss=.8/(1+cost/10000)-1
        assert row.max_drawdown_pct==pytest.approx(100*first_loss,abs=1e-10)
        assert abs(row.qlib_max_drawdown_without_initial_mark_pct)<1e-10
        assert row.cvar5_pct==pytest.approx(100*first_loss/(64*.05),abs=1e-10)
