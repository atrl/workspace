"""Synthetic workflow integrity tests; never execute registered market fits."""
import copy
from contextlib import contextmanager
import hashlib
import json
import pickle
from dataclasses import asdict
from pathlib import Path
import sys
import types

import numpy as np
import pandas as pd
import pytest

from test_benchmark_anchor_workflow import history, _InlinePool, _Recorder
from src.data.benchmark_anchor_utility import build_anchor_dataset
from src.model.benchmark_anchor_utility import _scalar_settle
from src.model.constrained_net_excess import ConstrainedNetExcessSpec
from src.workflow import sparse_holdings_net_excess as workflow


@pytest.fixture
def plan():
    p=json.loads((Path(__file__).parents[1]/'research/sparse_holdings_net_excess_20261003/selection_plan.json').read_text())
    p['training']['minimum_rows']=63;p['training']['updates']=0
    return p


@pytest.fixture
def prepared(history,plan):
    return workflow.prepare_folds(build_anchor_dataset(history,as_of='2017-01-10'),
        ['price_holdings','price_vol_holdings'],[2015,2016],plan)


class FakePolicy:
    def __init__(self,seed=17,anchor=False,mode=None):
        self.seed=seed;self.mode=mode or ('anchor' if anchor else 'buy_then_hold')


def _payload(model):return {'mode':model.mode,'seed':model.seed}
def _restore(value):return FakePolicy(seed=value['seed'],mode=value['mode'])


def _rollout(models,x,r,*,cost_bps,initial_weights=None,force_soxx_mask=None):
    held=np.array([0.,0.,1.] if initial_weights is None else initial_weights,dtype=float)
    force=np.zeros(len(r),bool) if force_soxx_mask is None else force_soxx_mask
    values={k:[] for k in ('pretrade_weights','ending_weights','target_weights','expert_weights','exact_hold','net_equity','fees','turnover')}
    for i,ret in enumerate(r):
        rows=[]
        for m in models:
            if m.mode=='anchor':q=[0.,1.,0.,0.]
            elif m.mode=='cash':q=[0.,0.,1.,0.]
            else:q=[0.,0.,0.,1.] if held[0]>.5 else [1.,0.,0.,0.]
            rows.append(q)
        q=np.mean(rows,axis=0)
        if force[i]:q=np.array([0.,1.,0.,0.])
        hold=q[3]==1 and np.count_nonzero(q)==1
        target=held.copy() if hold else np.array([q[0],q[1],q[2]])+q[3]*held
        ending,net,fee,turnover,_,_=_scalar_settle(*held,*target,*ret,float(cost_bps)/10000)
        for k,v in [('pretrade_weights',held.copy()),('ending_weights',ending),('target_weights',target),
                    ('expert_weights',q),('exact_hold',hold),('net_equity',net),('fees',fee),('turnover',turnover)]:values[k].append(v)
        held=np.array(ending)
    return {k:np.asarray(v,dtype=bool if k=='exact_hold' else float) for k,v in values.items()}


@pytest.fixture
def fake_model(monkeypatch):
    module=types.ModuleType('src.model.sparse_holdings_net_excess')
    module.SparseHoldingsPolicy=FakePolicy;module.policy_to_dict=_payload;module.policy_from_dict=_restore
    module.is_exact_anchor=lambda model:model.mode=='anchor'
    module.numpy_rollout=_rollout;module.ConstrainedNetExcessSpec=ConstrainedNetExcessSpec
    monkeypatch.setitem(sys.modules,module.__name__,module)
    return module


def _record(prepared,variant,year):
    fold=prepared[(variant,year)];p=_payload(FakePolicy())
    return {'variant':variant,'parent_market_variant':fold['parent_market_variant'],'model_year':year,
        'scaler':fold['scaler'],'scaler_sha256':fold['scaler']['scaler_sha256'],
        'seed_models':{s:copy.deepcopy(p) for s in workflow.SEEDS},'ensemble_models':[p]*3,
        'ensemble_audit':{'anchor_selected':False},'invalid_training':False,'numerical_failure':False}


def test_owned_state_survives_annual_boundary_and_exact_hold_has_no_orders(history,prepared,plan,fake_model):
    frame=history.loc['2015-12-29':'2016-01-07']
    annual={year:_record(prepared,'price_holdings',year) for year in (2015,2016)}
    paths={c:workflow.replay_models(frame,annual,variant='price_holdings',cost_bps=c,plan=plan) for c in (10,25)}
    for cost,path in paths.items():
        assert path.execute_open_trade.tolist()==[True]+[False]*(len(path)-1)
        assert path.exact_hold_support.tolist()==[False]+[True]*(len(path)-1)
        assert path.fees.iloc[1:].eq(0).all()
        np.testing.assert_array_equal(path[[f'pretrade_{a}_weight' for a in workflow.ASSETS]].iloc[1:],
                                      path[[f'ending_{a}_weight' for a in workflow.ASSETS]].iloc[:-1])
        assert workflow.independent_path_audit(frame,path,cost_bps=cost)['passed']
    assert paths[25].fees.iloc[0]>paths[10].fees.iloc[0]


def test_bad_vol_forces_only_current_date_and_resumes_on_resulting_book(history,prepared,plan,fake_model):
    frame=history.loc['2016-01-01':'2016-01-08'].copy()
    annual={2016:_record(prepared,'price_vol_holdings',2016)}
    good=workflow.replay_models(frame,annual,variant='price_vol_holdings',cost_bps=25,plan=plan)
    frame.loc[frame.index[3],'future_vol_prediction']=np.nan
    changed=workflow.replay_models(frame,annual,variant='price_vol_holdings',cost_bps=25,plan=plan)
    pd.testing.assert_frame_equal(good.iloc[:3],changed.iloc[:3])
    assert changed.observed_forecast_fallback.tolist()==[False,False,False,True,False,False]
    assert changed.soxx_weight.iloc[3]==1
    assert changed.pretrade_soxx_weight.iloc[4]==pytest.approx(1,abs=1e-14)
    assert changed.soxl_weight.iloc[4]==1 and not changed.exact_hold_support.iloc[4]
    assert workflow.independent_path_audit(frame,changed,cost_bps=25)['passed']
    price={2016:_record(prepared,'price_holdings',2016)}
    pd.testing.assert_frame_equal(
        workflow.replay_models(history.loc[frame.index],price,variant='price_holdings',cost_bps=25,plan=plan),
        workflow.replay_models(frame,price,variant='price_holdings',cost_bps=25,plan=plan))


@pytest.mark.parametrize('fault',['mapping','scaler','future','boolean','payload'])
def test_new_variant_and_old_scaler_binding_is_fail_closed(history,prepared,plan,fake_model,fault):
    annual={2015:_record(prepared,'price_holdings',2015)}
    if fault=='mapping':annual[2015]['parent_market_variant']='price_vol'
    if fault=='scaler':annual[2015]['scaler_sha256']='changed'
    if fault=='future':annual[2015]['model_year']=2017
    if fault=='boolean':annual[2015]['numerical_failure']='false'
    if fault=='payload':annual[2015]['scaler']=copy.deepcopy(annual[2015]['scaler']);annual[2015]['scaler']['price_mean'][0]+=1
    with pytest.raises(ValueError,match='binding|bundle|identity|boolean|scaler'):
        workflow.replay_models(history.loc['2016-01-01':'2016-01-08'],annual,variant='price_holdings',
                               cost_bps=10,plan=plan,frozen_year=2015)


def _seal(tmp_path):
    card=tmp_path/'card';card.mkdir()
    tests={}
    for name in ('test_sparse_holdings_net_excess_data.py','test_sparse_holdings_net_excess_workflow.py',
                 'test_sparse_holdings_net_excess.py','test_post_fee_settlement.py'):
        path=tmp_path/'tests'/name;path.parent.mkdir(exist_ok=True);path.write_text('synthetic test bytes')
        tests['tests/'+name]=workflow.sha(path)
    value={'plan_sha256':'plan','source_sha256':{'source':'digest'},'test_sha256':tests,
        'sealed_at_utc':'2026-10-03T00:00:00+00:00','market_model_fits_started':False,'focused_passed':21,
        'synthetic_known_information_all_seeds_and_ensemble_passed':True,'pre_fit_contracts_passed':True,
        'source_state_derivative_audit_passed':True,'objective_achieved':False}
    (card/'initial_seal.json').write_text(json.dumps(value));return card,value


@pytest.mark.parametrize('fault',['missing','source','test','flag','flag_string','boolean_count','missing_core_test'])
def test_initial_seal_requires_exact_sources_tests_and_true_prefit_evidence(tmp_path,fault):
    card,value=_seal(tmp_path)
    assert workflow._require_initial_seal(tmp_path,card,'plan',{'source':'digest'})['file_sha256']
    if fault=='missing':(card/'initial_seal.json').unlink()
    elif fault=='test':(tmp_path/'tests/test_sparse_holdings_net_excess_data.py').write_text('changed')
    else:
        if fault=='source':value['source_sha256']['source']='changed'
        if fault=='flag':value['pre_fit_contracts_passed']=False
        if fault=='flag_string':value['pre_fit_contracts_passed']='true'
        if fault=='boolean_count':value['focused_passed']=True
        if fault=='missing_core_test':del value['test_sha256']['tests/test_post_fee_settlement.py']
        (card/'initial_seal.json').write_text(json.dumps(value))
    with pytest.raises(ValueError,match='blocked'):
        workflow._require_initial_seal(tmp_path,card,'plan',{'source':'digest'})


def test_execution_diagnostics_are_an_identity_on_saved_targets(history,prepared,plan,fake_model):
    frame=history.loc['2016-01-01':'2016-01-08'];annual={2016:_record(prepared,'price_holdings',2016)}
    path=workflow.replay_models(frame,annual,variant='price_holdings',cost_bps=25,plan=plan)
    reference=workflow.benchmark_path(frame,cost_bps=25)
    rows=[path.assign(seed=str(seed),variant='price_holdings',segment='development') for seed in workflow.SEEDS]
    rows.extend([path.assign(seed='ensemble',variant='price_holdings',segment='development'),
                 reference.assign(seed='benchmark',variant='SOXX_buy_hold',segment='development')])
    support,fees,equality=workflow.execution_diagnostics(pd.concat(rows,ignore_index=True),{'development':frame})
    assert support.exact_hold_support_days.eq(len(frame)-1).all()
    assert equality.exact_path_equality.all()
    np.testing.assert_allclose(fees.gross_position_log_excess+fees.fee_log_excess,fees.net_log_excess,atol=1e-12,rtol=0)


def _metadata(failure=None):
    risk={'max_drawdown_pct':-10.,'cvar5_pct':-1.,'benchmark_max_drawdown_pct':-10.,'benchmark_cvar5_pct':-1.,
        'raw_percentage_point_gaps':[0.,0.],'normalizers':[.1,.01],'normalizers_valid_for_optimization':True}
    return {'training_feasibility_passed':True,'centered_worst_utility':0.,'centered_utility_by_cost':{10:0.,25:0.},
        'selected_risks_by_cost':{10:risk,25:copy.deepcopy(risk)},'qualification_eligible':failure is None,'failure':failure}


def test_retained_failure_requires_new_architecture_exact_anchor(plan,fake_model):
    bad={'accounting_anchor_only':True}
    valid=_metadata(bad)
    workflow._validate_retained([_payload(FakePolicy(anchor=True))],valid,plan)
    with pytest.raises(ValueError,match='explicit exact accounting null'):
        workflow._validate_retained([_payload(FakePolicy())],valid,plan)
    with pytest.raises(ValueError,match='true boolean'):
        workflow._validate_retained([_payload(FakePolicy(anchor=True))],{**valid,'training_feasibility_passed':'true'},plan)


def test_ensemble_only_failure_is_retained_in_new_replay_integrity_flag(prepared,plan,fake_model,monkeypatch):
    fold=prepared[('price_holdings',2015)]
    def worker(payload):
        name,year,seed,x,r,_=payload
        info=_metadata();info.update({'selected_epoch':'anchor','elapsed_seconds':0.,
            'training_feature_sha256':hashlib.sha256(x.astype(np.float64).tobytes()).hexdigest(),
            'training_return_sha256':hashlib.sha256(r.astype(np.float64).tobytes()).hexdigest()})
        return name,year,seed,_payload(FakePolicy(seed=seed,anchor=True)),info
    def ensemble(models,x,r,*,spec):
        info=_metadata({'accounting_anchor_only':True,'reason':'synthetic ensemble failure'})
        info['anchor_selected']=True
        return [FakePolicy(anchor=True)],info
    fake_model.evaluate_ensemble=ensemble
    monkeypatch.setattr(workflow,'ProcessPoolExecutor',_InlinePool);monkeypatch.setattr(workflow,'_fit_job',worker)
    fitted,diagnostics,audits=workflow.fit_folds({('price_holdings',2015):fold},plan,_Recorder())
    assert fitted['price_holdings'][2015]['numerical_failure'] is True
    assert all(d['failure'] is None for d in diagnostics) and audits[0]['failure'] is not None


def test_native_hold_callback_creates_no_order_and_retains_owned_shares(tmp_path,monkeypatch):
    import qlib
    from qlib.data import D
    from qlib.backtest.exchange import Exchange
    from src.data.qlib_native import build_qlib_store_from_frame
    from src.analysis.convex_qlib import native_open_reconciliation
    from src.backtest.position_accounting import settle_target_session
    from src.workflow.stage1_economic_action import fee_affordable_target
    dates=pd.bdate_range('2016-01-04',periods=5)
    rows=[]
    for symbol,scale in [('SOXL',20.),('SOXX',100.)]:
        for i,date in enumerate(dates):
            price=scale*(1+.01*i)
            rows.append({'instrument':symbol,'datetime':date,'open':price,'close':price,
                         'high':price*1.01,'low':price*.99,'volume':100000.,'factor':1.})
    provider=tmp_path/'provider'
    build_qlib_store_from_frame(pd.DataFrame(rows),provider_uri=provider,as_of=str(dates[-1].date()),
        field_map={name:name for name in ('open','close','high','low','volume','factor')})
    qlib.init(provider_uri=str(provider),region='us',kernels=1)
    opens=D.features(['SOXL','SOXX'],['$open'],start_time=dates[0],end_time=dates[-1])['$open'].unstack('instrument').astype(float)
    held=np.array([0.,0.,1.]);market=[];path=[]
    for i,desired in enumerate(([1.,0.,0.],None,[0.,1.,0.])):
        ret=np.r_[(opens.loc[dates[i+2]]/opens.loc[dates[i+1]]-1).to_numpy(),0.]
        target=held.copy() if desired is None else np.array(desired)
        execution=None if desired is None else fee_affordable_target(target,pretrade_weights=held,cost_bps=25)
        settled=settle_target_session(held,open_target_weights=execution,asset_returns=ret,cost_bps=25)
        row={'signal_date':dates[i],'trade_date':dates[i+1],'exit_date':dates[i+2],'soxl_return':ret[0],'soxx_return':ret[1]}
        market.append(row);path.append({**row,'strategy_return':settled['net_return'],'fees':settled['fees'],
            'turnover':settled['turnover'],'cost_bps':25,'execute_open_target':desired is not None,
            **{f'{a}_weight':target[j] for j,a in enumerate(workflow.ASSETS)}})
        # The new state contract uses the post-fee normalized target directly;
        # old cash subtraction can leave a -4e-17 boundary residual. This is
        # the exact algebraic identity, not clipping an invalid test target.
        held=target*(1+ret)/np.dot(target,1+ret)
        np.testing.assert_allclose(held,settled['ending_weights'],atol=1e-14,rtol=0)
    calls=[];original=Exchange.generate_order_for_target_amount_position
    def track(self,*args,**kwargs):
        calls.append(pd.Timestamp(args[2]));return original(self,*args,**kwargs)
    monkeypatch.setattr(Exchange,'generate_order_for_target_amount_position',track)
    result=workflow.native_allocation_reconciliation(pd.DataFrame(market),pd.DataFrame(path),
        provider_uri=provider,cost_bps=25,native_reconciler=native_open_reconciliation)
    assert result['audit']['status']=='passed'
    assert calls==[dates[1],dates[3]]
    assert result['positions'][dates[2]].get_stock_amount('SOXL')==result['positions'][dates[1]].get_stock_amount('SOXL')


def _actual_policy(mode='buy_then_hold',seed=17):
    """Analytic API fixtures; no optimizer or market outcomes choose weights."""
    import torch
    from src.model.sparse_holdings_net_excess import SparseHoldingsPolicy
    policy=SparseHoldingsPolicy(seed=seed,anchor=True)
    if mode=='anchor':return policy
    with torch.no_grad():
        policy.A.zero_();policy.a.zero_();policy.K.zero_()
        if mode=='cash':policy.a[2]=1.
        else:
            policy.a[0]=2.
            policy.K[3,0]=4.  # CASH-start -> SOXL; owned SOXL -> exact HOLD.
    return policy


def _actual_record(prepared,variant,year,models=None):
    from src.model.sparse_holdings_net_excess import policy_to_dict
    models=models or [_actual_policy(seed=seed) for seed in workflow.SEEDS]
    record=_record(prepared,variant,year)
    record['seed_models']={seed:policy_to_dict(policy) for seed,policy in zip(workflow.SEEDS,models)}
    record['ensemble_models']=[policy_to_dict(policy) for policy in models]
    return record


def test_actual_model_serialization_annual_state_and_exact_hold(history,prepared,plan):
    from src.model.sparse_holdings_net_excess import policy_from_dict,policy_to_dict
    annual={year:_actual_record(prepared,'price_holdings',year) for year in (2015,2016)}
    for record in annual.values():
        for payload in record['ensemble_models']:
            assert policy_to_dict(policy_from_dict(payload))['tensor_sha256']==payload['tensor_sha256']
    frame=history.loc['2015-12-29':'2016-01-07']
    paths={c:workflow.replay_models(frame,annual,variant='price_holdings',cost_bps=c,plan=plan) for c in (10,25)}
    for cost,path in paths.items():
        assert path.exact_hold_support.tolist()==[False]+[True]*(len(path)-1)
        assert path.execute_open_target.tolist()==[True]+[False]*(len(path)-1)
        assert path.turnover.iloc[1:].eq(0).all() and path.fees.iloc[1:].eq(0).all()
        assert path.loc[path.signal_date.dt.year.eq(2016),'pretrade_soxl_weight'].eq(1).all()
        assert workflow.independent_path_audit(frame,path,cost_bps=cost)['passed']
    np.testing.assert_array_equal(paths[10][['soxl_weight','soxx_weight','cash_weight']],
                                  paths[25][['soxl_weight','soxx_weight','cash_weight']])
    assert paths[25].fees.iloc[0]>paths[10].fees.iloc[0]


def test_actual_model_forecast_override_changes_only_observed_date_then_uses_new_holdings(history,prepared,plan):
    frame=history.loc['2016-01-01':'2016-01-08'].copy()
    annual={2016:_actual_record(prepared,'price_vol_holdings',2016)}
    normal=workflow.replay_models(frame,annual,variant='price_vol_holdings',cost_bps=25,plan=plan)
    frame.loc[frame.index[3],'future_vol_prediction']=np.nan
    bad=workflow.replay_models(frame,annual,variant='price_vol_holdings',cost_bps=25,plan=plan)
    pd.testing.assert_frame_equal(normal.iloc[:3],bad.iloc[:3])
    assert bad.observed_forecast_fallback.tolist()==[False,False,False,True,False,False]
    assert bad.expert_soxx_weight.iloc[3]==1 and bad.pretrade_soxx_weight.iloc[4]==1
    assert bad.expert_soxl_weight.iloc[4]==1 and bad.exact_hold_support.iloc[5]
    assert workflow.independent_path_audit(frame,bad,cost_bps=25)['passed']


def test_actual_three_model_ensemble_uses_shared_h_not_private_component_books(history,prepared,plan):
    from src.model.sparse_holdings_net_excess import numpy_rollout,policy_target
    from src.model.post_fee_settlement import scalar_settle
    models=[_actual_policy('buy_then_hold'),_actual_policy('cash',41),_actual_policy('anchor',71)]
    annual={2016:_actual_record(prepared,'price_holdings',2016,models)}
    frame=history.loc['2016-01-01':'2016-01-08']
    path=workflow.replay_models(frame,annual,variant='price_holdings',cost_bps=25,plan=plan)
    x=workflow.transform_for_variant(frame,annual[2016]['scaler'],variant='price_holdings',plan=plan)['features']
    returns=frame[['soxl_return','soxx_return']].to_numpy(float);held=np.array([0.,0.,1.])
    expected=[]
    for feature,ret in zip(x,returns):
        target=policy_target(models,feature,held);expected.append(target)
        held=np.asarray(scalar_settle(*held,*target,*ret,.0025)[0])
    np.testing.assert_allclose(path[['soxl_weight','soxx_weight','cash_weight']],expected,atol=1e-14,rtol=0)
    private_average=np.mean([numpy_rollout(model,x,returns,cost_bps=25)['target_weights'] for model in models],axis=0)
    assert np.max(abs(private_average-np.asarray(expected)))>1e-3
    assert workflow.independent_path_audit(frame,path,cost_bps=25)['passed']


@pytest.fixture
def failed_preflight_fixture(tmp_path,monkeypatch):
    """A fabricated unit fixture for archive plumbing, never research evidence."""
    import qlib
    import src.model.sparse_holdings_net_excess as model
    root=Path(__file__).parents[1]
    plan=json.loads((root/'research/sparse_holdings_net_excess_20261003/selection_plan.json').read_text())
    card=tmp_path/'card';card.mkdir();(card/'selection_plan.json').write_text(json.dumps(plan))
    source=tmp_path/'probe_input';source.mkdir()
    retention=(root/'src/model/sparse_holdings_net_excess.py').read_bytes()
    initial=retention.replace(b'reported_trade_count_gt_1e_minus12',b'reported_trade_count_gt_1e12')
    (source/'retention.py').write_bytes(retention);(source/'initial.py').write_bytes(initial)
    source_hash=hashlib.sha256(retention).hexdigest()
    x=np.zeros((504,27));x[:,0]=np.where(np.arange(504)%2==0,1.,-1.)
    soxx=.002+.008*x[:,0];returns=np.c_[3*soxx,soxx]
    spec=workflow.training_spec(plan);fits=[];old=[]
    for seed in workflow.SEEDS:
        anchor=model.zero_anchor(seed);payload=model.policy_to_dict(anchor)
        paths={c:model.numpy_rollout(anchor,x,returns,cost_bps=c) for c in (10,25)}
        epochs=[{'epoch':i,'F':0.,'feasible':True,'checkpoint':payload} for i in range(121)]
        metadata={'seed':seed,'spec':asdict(spec),'training_rows':504,
            'training_feature_sha256':hashlib.sha256(x.tobytes()).hexdigest(),
            'training_return_sha256':hashlib.sha256(returns.tobytes()).hexdigest(),
            'epochs':epochs,'history':epochs,'optimizer_diagnostics':{'updates_completed':120},
            'final_optimizer_state':{'unit_fixture':True},'training_paths':paths,
            'benchmark_net_equity':{c:p['net_equity'] for c,p in paths.items()},
            'centered_utility_by_cost':{10:0.,25:0.},'F':0.,'centered_worst_utility':0.,
            'selected_epoch':'anchor','null_selected':True,'failure':None}
        fits.append({'seed':seed,'selected_model':payload,'metadata':metadata,'latest_model':payload,'latest_evaluation':{}})
        old.append({'seed':seed,'F':0.,'selected_epoch':'anchor','epochs':121,'feasible_positive_F_states':0,'maximum_trajectory_F':0.})
    prior={'market_data_used':False,'fits':old,'witness':{'F':.2,'feasible':True}}
    (source/'initial_summary.json').write_text(json.dumps(prior))
    probe={'purpose':'unit archive fixture, no training performed','features':x,'returns':returns,'spec':spec,
           'source_sha256':source_hash,'fits':fits,'prior_probe':prior}
    probe_file=source/'probe.pkl';probe_file.write_bytes(pickle.dumps(probe))
    receipt={'initial_failed_probe_source_sha256':workflow.sha(source/'initial.py'),
        'retention_reproduction_source_sha256':source_hash,'initial_source_path':str(source/'initial.py'),
        'retention_source_path':str(source/'retention.py'),'initial_probe_summary_path':str(source/'initial_summary.json'),
        'initial_probe_summary_sha256':workflow.sha(source/'initial_summary.json'),
        'retention_full_payload_sha256':workflow.sha(probe_file)}
    receipt_path=source/'receipt.json';receipt_path.write_text(json.dumps(receipt))
    provider=tmp_path/'provider';provider.mkdir();state={'native_reads':0}
    def native_read(*args):
        state['native_reads']+=1
        qlib.init(provider_uri=str(provider),region='us',kernels=1)
        return {'source_audit':{'unit_fixture':True},'source_details':{'unit_fixture':True}}
    monkeypatch.setenv('MLFLOW_ALLOW_FILE_STORE','true');monkeypatch.setenv('MLFLOW_DISABLE_AGENT_HINT','1')
    monkeypatch.setattr(workflow,'load_sparse_inputs',native_read)
    monkeypatch.setattr(workflow,'_source_hashes',lambda *_:{'src/model/sparse_holdings_net_excess.py':source_hash})
    monkeypatch.setattr(workflow,'CANONICAL_ROOT',tmp_path)
    monkeypatch.setattr(workflow,'fit_folds',lambda *a,**k:pytest.fail('diagnostic started a market fitter'))
    monkeypatch.setattr(model,'fit_policy',lambda *a,**k:pytest.fail('diagnostic reran a toy fitter'))
    return tmp_path,card,probe_file,receipt_path,probe,state


def test_failed_preflight_archives_exact_ordinal_probe_and_never_opens_market_gate(failed_preflight_fixture):
    import shutil
    from qlib.workflow import R
    root,card,probe,receipt,_,state=failed_preflight_fixture;digest=workflow.sha(probe)
    original=probe.read_bytes()
    summary=workflow.record_preflight_diagnostic(root,card,probe_path=probe,expected_probe_sha256=digest,source_receipt_path=receipt)
    assert summary['market_model_fits_started'] is False and summary['market_model_fits']==summary['market_backtests']==0
    assert summary['record_type']=='failed_registered_synthetic_preflight_diagnostic'
    assert summary['preflight']['registered_positive_control_pass'] is False and summary['selected'] is None
    assert not (card/'initial_seal.json').exists()
    with R.uri_context(summary['recorder_uri']):
        rec=R.get_recorder(experiment_name=workflow.EXPERIMENT,recorder_id=summary['recorder_id'])
        assert rec.client.get_run(rec.id).info.status=='FINISHED'
        assert (Path(rec.get_local_dir())/'artifacts/preflight/original_payload.pkl').read_bytes()==original
        table=rec.load_object('analysis/synthetic_seed_summary.pkl')
        assert table.ordinal_rows.eq(504).all() and not table.registered_positive_control_pass.any()
        assert not any('date' in name for name in table.columns)
    shutil.rmtree(probe.parent)
    cached=workflow.record_preflight_diagnostic(root,card)
    assert cached['probe_sha256']==digest and state['native_reads']==2
    with pytest.raises(ValueError,match='initial source/test seal'):
        workflow.run_experiment(root,card)
    assert state['native_reads']==2  # Market gate fails before another source read.


def test_preflight_probe_sha_is_checked_before_deserialization(failed_preflight_fixture,monkeypatch):
    root,card,probe,receipt,_,_=failed_preflight_fixture
    monkeypatch.setattr(workflow.pickle,'loads',lambda *_:pytest.fail('corrupt diagnostic was deserialized'))
    with pytest.raises(ValueError,match='before deserialization'):
        workflow.record_preflight_diagnostic(root,card,probe_path=probe,expected_probe_sha256='wrong',source_receipt_path=receipt)


def test_failed_diagnostic_refuses_fixture_or_trace_membership_changes(failed_preflight_fixture):
    root,card,probe_path,receipt_path,probe,_=failed_preflight_fixture
    plan=json.loads((card/'selection_plan.json').read_text());receipt=json.loads(receipt_path.read_text())
    files={'initial_model.py':Path(receipt['initial_source_path']).read_bytes(),
        'retention_model.py':Path(receipt['retention_source_path']).read_bytes(),
        'initial_probe_summary.json':Path(receipt['initial_probe_summary_path']).read_bytes()}
    changed=copy.deepcopy(probe);changed['features'][0,0]=-1
    with pytest.raises(ValueError,match='ordinal fixture'):
        workflow._preflight_probe_audit(changed,plan,receipt,files)
    changed=copy.deepcopy(probe);changed['fits'][0]['metadata']['epochs'].pop()
    with pytest.raises(ValueError,match='traces are incomplete'):
        workflow._preflight_probe_audit(changed,plan,receipt,files)


@pytest.mark.parametrize("tamper_development", [False, True])
def test_parent_input_binding_seal_order_frozen_parameters_and_actual_aug3_trade_origin(plan, tmp_path, monkeypatch, tamper_development):
    """Exercise orchestration with sentinel models, never a financial replay."""
    import qlib.workflow as native_workflow
    from src.data.integrated_allocation import COMMON_PRIMARY_FEATURES
    from src.workflow import conditional_variance as prior_analysis

    card,seal = _seal(tmp_path)
    (card / "selection_plan.json").write_text(json.dumps(plan))
    seal["plan_sha256"]=workflow.sha(card/"selection_plan.json")
    seal["source_sha256"]={}
    (card/"initial_seal.json").write_text(json.dumps(seal))
    dates = pd.to_datetime(["2016-01-04", "2022-12-30", "2023-01-03", "2026-07-31", "2026-08-03", "2026-09-23"])
    frame = pd.DataFrame(0., index=dates, columns=list(COMMON_PRIMARY_FEATURES))
    frame["signal_date"] = dates;frame["trade_date"] = dates + pd.offsets.BDay(1);frame["exit_date"] = dates + pd.offsets.BDay(2)
    frame["soxl_return"] = .003;frame["soxx_return"] = .001
    frame["cash_advantage_prediction"] = 0.;frame["future_vol_prediction"] = .018;frame["forecast_available"] = 1.
    frame["forecast_model_available_at"] = pd.Timestamp("2016-01-01", tz="UTC")
    names = [v["name"] for v in plan["variants"]]
    mapping=workflow.variant_mapping(plan)
    dev_years = plan["training"]["development_signal_years"]
    outer_years = plan["training"]["diagnostic_signal_years"]
    parent_folds = {(mapping[name], year): {"scaler": {"fixture_year": year}, "features": np.zeros((1, 27)), "returns": np.zeros((1, 2))}
                    for name in names for year in dev_years}
    development_folds={(name,year):copy.deepcopy(parent_folds[(mapping[name],year)]) for name in names for year in dev_years}
    if tamper_development:development_folds[(names[0],dev_years[0])]["features"][0,0]=1.
    state = {"active": False, "prepared": [], "fit_calls": [], "outer_replays": [], "head_calls": []}

    class Recorder(_Recorder):
        id = "synthetic-phase8"
        def save_objects(self, *, artifact_path, **objects):
            for key, value in objects.items(): self.objects[f"{artifact_path}/{key}"] = value
        def load_object(self, name): return self.objects[name]
        def get_local_dir(self): return str(tmp_path / "synthetic_recorder")
    recorder = Recorder()

    class FakeWorkflow:
        @staticmethod
        def set_uri(uri): pass
        @staticmethod
        @contextmanager
        def start(**kwargs):
            state["active"] = True
            try: yield
            finally: state["active"] = False
        @staticmethod
        def get_recorder(): return recorder
        @staticmethod
        def log_params(**kwargs): pass
        @staticmethod
        def log_metrics(**kwargs): pass
        @staticmethod
        def save_objects(**kwargs): recorder.save_objects(**kwargs)

    class FrozenHead:
        def predict(self, features):
            assert list(features.columns) == list(COMMON_PRIMARY_FEATURES)
            state["head_calls"].append(features.index.tolist())
            return np.c_[np.zeros(len(features)), np.full(len(features), .123)]

    class ReachedNativeAfterClosedRecorder(Exception):
        pass

    outer_records = {year: {"scaler": {"year": year}, "seed_models": {seed: (year, seed) for seed in workflow.SEEDS},
                            "ensemble_models": [(year, seed) for seed in workflow.SEEDS],
                            "ensemble_audit": {"anchor_selected": year == 2023}}
                     for year in outer_years}

    def prepare(dataset, variants, years, current_plan):
        state["prepared"].append(tuple(years))
        assert years == outer_years and variants == ["price_vol_holdings"]
        seal = card / "selection_seal.json"
        assert seal.exists() and ("inputs", str(seal)) in recorder.artifacts
        assert json.loads(seal.read_text())["selection_uses_diagnostic_signal_years"] is False
        return {("price_vol_holdings", year): {} for year in outer_years}

    def fit(folds, current_plan, current_recorder, **kwargs):
        years = {year for _, year in folds}
        state["fit_calls"].append(years)
        return ({name: {} for name in names}, [], []) if years == set(dev_years) else ({"price_vol_holdings": outer_records}, [], [])

    def path(rows, *, cost_bps):
        return rows[["signal_date", "trade_date", "exit_date"]].reset_index(drop=True).assign(
            strategy_return=.001, fees=0., turnover=0., cost_bps=cost_bps)

    def replay(rows, annual, *, variant, cost_bps, seed=None, frozen_year=None,plan=None):
        if rows.index.year.min() >= 2023:
            assert (card / "selection_seal.json").exists()
            assert annual is outer_records
            if frozen_year is not None:
                assert frozen_year == 2023
                assert annual[2023]["ensemble_audit"]["anchor_selected"] is True
                assert rows.future_vol_prediction.eq(.123).all()
                assert pd.to_datetime(rows.forecast_model_available_at, utc=True).eq(pd.Timestamp("2023-01-01", tz="America/New_York").tz_convert("UTC")).all()
            else:
                assert rows.future_vol_prediction.eq(.018).all()
            state["outer_replays"].append({"dates": rows.index.tolist(), "frozen_year": frozen_year, "cost": cost_bps})
        else:
            assert rows.index.year.max() <= 2022
        return path(rows, cost_bps=cost_bps)

    def metrics(left, right):
        return {"log_excess_sum": 0., "return_pct": 0., "relative_wealth": 1., "max_drawdown_pct": 0.,
                "cvar5_pct": 0., "sessions": len(left), "first_trade": str(left.trade_date.iloc[0].date())}

    def qualification(*args, **kwargs):
        return {"metrics": pd.DataFrame({"cost_bps": [10., 25.], "return_pct": 0., "relative_wealth": 1., "max_drawdown_pct": 0., "cvar5_pct": 0.}),
                "annual": pd.DataFrame({"signal_year": [2016], "log_excess_sum": [0.]}),
                "gates": pd.DataFrame({"cost_bps": [10., 25.], "all_pass": False}), "bootstrap": pd.DataFrame({"lower_bound": [-1.]})}

    def native_boundary(*args):
        assert state["active"] is False
        raise ReachedNativeAfterClosedRecorder()

    monkeypatch.setattr(native_workflow, "R", FakeWorkflow)
    monkeypatch.setattr(workflow, "CANONICAL_ROOT", tmp_path)
    monkeypatch.setattr(workflow, "_source_hashes", lambda root: {})
    monkeypatch.setattr(workflow, "load_sparse_inputs", lambda *args: {"raw": frame.copy(), "prices": pd.DataFrame(), "forecasts": frame.copy(),
        "forecast_audit": pd.DataFrame(), "forecast_models": {2023: FrozenHead()}, "dataset": object(),
        "source_audit": {}, "source_details": {}, "parent_dev_folds": parent_folds,
        "variant_mapping":mapping,"development_folds":development_folds})
    monkeypatch.setattr(workflow, "prepare_folds", prepare)
    monkeypatch.setattr(workflow,"execution_diagnostics",lambda *args:(pd.DataFrame(),pd.DataFrame(),pd.DataFrame()))
    monkeypatch.setattr(workflow, "fit_folds", fit)
    monkeypatch.setattr(workflow, "benchmark_path", path)
    monkeypatch.setattr(workflow, "replay_models", replay)
    monkeypatch.setattr(workflow, "paired_metrics", metrics)
    monkeypatch.setattr(workflow, "annual_paired_metrics", lambda *args: pd.DataFrame({"signal_year": [2023], "log_excess_sum": [0.]}))
    monkeypatch.setattr(workflow, "independent_path_audit", lambda *args, **kwargs: {"passed": True})
    monkeypatch.setattr(workflow, "qualification", qualification)
    monkeypatch.setattr(workflow, "choose_family", lambda evidence: ({"variant": "price_vol_holdings", "historical_qualification_pass": False}, pd.DataFrame()))
    monkeypatch.setattr(prior_analysis, "hac_attribution", lambda *args: {"realized_beta": 1.})
    monkeypatch.setattr(workflow, "_finalize_native", native_boundary)
    if tamper_development:
        with pytest.raises(ValueError, match="changed prepared development inputs"):
            workflow.run_experiment(tmp_path, card, workers=1)
        assert state["fit_calls"] == []
        return
    with pytest.raises(ReachedNativeAfterClosedRecorder):
        workflow.run_experiment(tmp_path, card, workers=1)
    assert state["prepared"] == [tuple(outer_years)]
    assert len(state["head_calls"]) == 1
    assert len(state["outer_replays"]) == 8  # Full/fresh x two costs x two modes.
    fresh = [r for r in state["outer_replays"] if r["dates"][0].year == 2026]
    assert len(fresh) == 4 and all(r["dates"][0] == pd.Timestamp("2026-07-31") for r in fresh)
    windows = recorder.objects["analysis/window_metrics.pkl"]
    assert len(windows) == 8 and windows.first_trade.eq("2026-08-03").all()


@pytest.mark.parametrize("native_fails", [False, True])
def test_phase8_native_reports_resume_own_recorder_after_qlib_reinitialization(tmp_path, monkeypatch, native_fails):
    import qlib
    from qlib.workflow import R
    from qlib.workflow.recorder import Recorder
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    monkeypatch.setenv("MLFLOW_DISABLE_AGENT_HINT", "1")
    provider = tmp_path / "provider";provider.mkdir()
    uri = str(tmp_path / "phase8_recorder")
    qlib.init(provider_uri=str(provider), region="us", kernels=1)
    with R.start(experiment_name=workflow.EXPERIMENT, uri=uri):
        identifier = R.get_recorder().id
        R.save_objects(artifact_path="inputs", **{"marker.pkl": "retained phase8 evidence"})
    def build():
        assert R.exp_manager.active_experiment is None
        qlib.init(provider_uri=str(provider), region="us", kernels=1)
        R.set_uri(str(tmp_path / "different_default"))
        if native_fails: raise RuntimeError("synthetic native failure")
        return {"synthetic": True}
    if native_fails:
        with pytest.raises(RuntimeError, match="synthetic native failure"):
            workflow._finalize_native(identifier, uri, build)
    else:
        assert workflow._finalize_native(identifier, uri, build) == {"synthetic": True}
    with R.uri_context(uri):
        retained = R.get_recorder(experiment_name=workflow.EXPERIMENT, recorder_id=identifier)
        assert retained.status == (Recorder.STATUS_FA if native_fails else Recorder.STATUS_FI)
        assert retained.load_object("inputs/marker.pkl") == "retained phase8 evidence"
        if not native_fails:
            assert retained.load_object("portfolio_analysis/native_reconciliation.pkl") == {"synthetic": True}
