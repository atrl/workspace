"""Workflow gates/provenance only; no registered toy or market search runs."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from test_benchmark_anchor_workflow import _InlinePool,_Recorder
from src.model.benchmark_anchor_utility import BenchmarkAnchorPolicy,policy_to_dict as parent_payload
from src.model.feasible_parent_sparse_search import policy_to_dict,prepare_parent,zero_anchor
from src.workflow import feasible_parent_sparse_search as wf


@pytest.fixture
def plan():
    return json.loads((Path(__file__).parents[1]/'research/feasible_parent_sparse_search_20261003/selection_plan.json').read_text())


def _risk():
    return {'max_drawdown_pct':-10.,'cvar5_pct':-1.,'benchmark_max_drawdown_pct':-10.,'benchmark_cvar5_pct':-1.,
        'raw_percentage_point_gaps':[0.,0.],'normalizers':[.1,.01],'normalizers_valid_for_optimization':True}


def _audit(F=.1,origin='parent',failure=None):
    return {'F':F,'parent_F':.1,'F_delta_from_parent':F-.1,'selected_origin':origin,'selected_trial':None,
        'centered_worst_utility':F,'centered_utility_by_cost':{10:F,25:F},
        'training_feasibility_passed':True,'qualification_eligible':failure is None,'failure':failure,
        'selected_risks_by_cost':{10:_risk(),25:_risk()},'trials':[{'unique_id':0}],
        'frozen_encoder_sha256_before':'same','frozen_encoder_sha256_after':'same',
        'optimizer_diagnostics':{'unique_full_path_evaluations':1,'requested_callbacks':{'objective':1,'constraints':1},
            'cache_hits':2,'termination':'fixture','solver_result':{'nfev':1}}}


def test_parent_inheritance_cannot_pass_synthetic_increment_gate(plan):
    parents={s:{'healthy':True,'feasible':True,'F':.1} for s in wf.SEEDS}
    searches={s:_audit() for s in wf.SEEDS}
    ensemble={'failure':None,'parent_ensemble':{'F':.1,'feasible':True},
              'candidate_ensemble':{'F':.1,'feasible':True},'selected_origin':'parent'}
    table,gate=wf._incremental_gate(parents,searches,ensemble,plan)
    assert gate['parent_stage_passed'] and not gate['passed']
    assert not table.incremental_seed_gate_passed.any()
    for s in wf.SEEDS:searches[s]=_audit(.2,'search')
    ensemble['candidate_ensemble']={'F':.2,'feasible':False}
    ensemble['F']=999.  # A selected/fallback field must not replace actual candidate evidence.
    _,gate=wf._incremental_gate(parents,searches,ensemble,plan)
    assert gate['all_three_seed_increments_passed'] and not gate['actual_candidate_ensemble_increment_passed']
    ensemble['candidate_ensemble']['feasible']=True
    ensemble['selected_origin']='search'
    assert wf._incremental_gate(parents,searches,ensemble,plan)[1]['passed']
    ensemble['parent_ensemble']['F']=-.2;ensemble['candidate_ensemble']['F']=-.1;ensemble['selected_origin']='null'
    assert not wf._incremental_gate(parents,searches,ensemble,plan)[1]['passed']
    ensemble['candidate_ensemble']['F']=.1;ensemble['selected_origin']='search'
    assert not wf._incremental_gate(parents,searches,ensemble,plan)[1]['passed']
    parents[17]['healthy']='false'
    with pytest.raises(ValueError,match='boolean'):wf._incremental_gate(parents,searches,ensemble,plan)


def test_numerical_failure_can_keep_valid_parent_but_never_hide_failure(plan):
    parent=BenchmarkAnchorPolicy(seed=17);_,embedded=prepare_parent(parent)
    payload=policy_to_dict(embedded)
    failure={'kind':'numerical_search_failure','accounting_only':True}
    audit=_audit(failure=failure)
    wf._validate_retained([payload],audit,plan)
    wrong=deepcopy(audit);wrong['qualification_eligible']=True
    with pytest.raises(ValueError,match='hidden'):wf._validate_retained([payload],wrong,plan)
    wrong=deepcopy(audit);wrong['failure']['accounting_only']=False
    with pytest.raises(ValueError,match='accounting-only'):wf._validate_retained([payload],wrong,plan)
    for normalizers in ([0.,.01],[.1,np.nan],[.2,.01]):
        wrong=_audit();wrong['selected_risks_by_cost'][10]['normalizers']=normalizers
        with pytest.raises(ValueError,match='normalizers'):wf._validate_retained([payload],wrong,plan)
    wrong=_audit();wrong['centered_worst_utility']=True
    with pytest.raises(ValueError,match='two-cost F'):wf._validate_retained([payload],wrong,plan)
    wrong=_audit();wrong['optimizer_diagnostics']['unique_full_path_evaluations']=True
    with pytest.raises(ValueError,match='budget'):wf._validate_retained([payload],wrong,plan)


def _fold_and_parents():
    x=np.zeros((252,27));r=np.c_[np.tile([.03,-.018],126),np.tile([.01,-.006],126)]
    fold={'features':x,'returns':r,'rows':pd.DataFrame(index=range(252)),
        'scaler':{'variant':'price','scaler_sha256':'fixture'},'parent_market_variant':'price',
        'invalid_training':False,'invalid_training_dates':[],'audit':{'cutoff':'2016-01-01T00:00:00-05:00'}}
    parents={'parent_seed_models':{('price_holdings',2016,s):parent_payload(BenchmarkAnchorPolicy(seed=s)) for s in wf.SEEDS},
             'parent_seed_audits':{('price_holdings',2016,s):{'failure':None,'source':'fixture'} for s in wf.SEEDS}}
    return fold,parents


def test_unexpected_solver_trace_is_saved_before_campaign_abort(plan,monkeypatch):
    fold,parents=_fold_and_parents();rec=_Recorder()
    def worker(payload):
        name,year,seed,x,r,parent,identity,spec=payload
        _,embedded=prepare_parent(parent,seed=seed)
        audit=_audit(failure={'kind':'unexpected_solver_exception','accounting_only':True,'traceback':'synthetic stack'})
        audit.update({'training_feature_sha256':wf._hash_array(x),'training_return_sha256':wf._hash_array(r)})
        return name,year,seed,policy_to_dict(embedded),audit
    monkeypatch.setattr(wf,'ProcessPoolExecutor',_InlinePool);monkeypatch.setattr(wf,'_fit_job',worker)
    with pytest.raises(RuntimeError,match='trace archived'):
        wf.fit_folds({('price_holdings',2016):fold},parents,plan,rec)
    saved=[v['audit.pkl'] for v in rec.objects.values() if 'audit.pkl' in v]
    assert saved and all(a['failure']['kind']=='unexpected_solver_exception' for a in saved)


def test_invalid_used_training_forecast_never_starts_search_and_stays_disqualified(plan,monkeypatch):
    fold,parents=_fold_and_parents();fold['invalid_training']=True;fold['invalid_training_dates']=['synthetic_invalid_row']
    monkeypatch.setattr(wf,'ProcessPoolExecutor',_InlinePool)
    monkeypatch.setattr(wf,'_fit_job',lambda *_:pytest.fail('invalid source started search'))
    fitted,audits,ensemble=wf.fit_folds({('price_holdings',2016):fold},parents,plan,_Recorder())
    assert len(audits)==3 and all(a['selected_origin']=='null' and not a['qualification_eligible'] for a in audits)
    assert fitted['price_holdings'][2016]['invalid_training']
    assert ensemble[0]['selected_origin']=='null' and ensemble[0]['failure']['kind']=='invalid_available_training_volatility'


def test_parent_revalidation_failure_is_accounting_null_not_a_replacement_start(plan,monkeypatch):
    from src.model import feasible_parent_sparse_search as model
    fold,parents=_fold_and_parents()
    def reject(*a,**k):raise model.ParentContractError('synthetic parent parity failure')
    monkeypatch.setattr(model,'fit_policy',reject)
    payload=('price_holdings',2016,17,fold['features'],fold['returns'],parents['parent_seed_models'][('price_holdings',2016,17)],{},wf.search_spec(plan))
    *_,saved,audit=wf._fit_job(payload)
    wf._validate_retained([saved],audit,plan)
    assert audit['selected_origin']=='null' and not audit['qualification_eligible']
    assert audit['failure']['kind']=='parent_revalidation_failure'
    assert audit['optimizer_diagnostics']['unique_full_path_evaluations']==0


def test_implementation_seal_checks_static_source_data_gauge_and_tests(plan,tmp_path):
    card=tmp_path/'card';card.mkdir()
    original=Path(__file__).parents[1]/'research/feasible_parent_sparse_search_20261003'
    for name in ['selection_plan.json','registration_clarification.json']:(card/name).write_bytes((original/name).read_bytes())
    names=['test_feasible_parent_sparse_search.py','test_feasible_parent_sparse_search_data.py',
           'test_feasible_parent_sparse_search_workflow.py','test_post_fee_settlement.py','test_paired_sparse_paths.py']
    tests={}
    for name in names:
        p=tmp_path/'tests'/name;p.parent.mkdir(exist_ok=True);p.write_text('fixture');tests[str(p.relative_to(tmp_path))]=wf.sha(p)
    seal={'plan_sha256':wf.sha(card/'selection_plan.json'),'source_sha256':{'fixture':'sha'},'test_sha256':tests,
        'registration_clarification_sha256':wf.sha(card/'registration_clarification.json'),
        'focused_passed':5,'static_contracts_passed':True,'source_contracts_passed':True,
        'data_contracts_passed':True,'gauge_contracts_passed':True}
    p=card/'implementation_seal.json';p.write_text(json.dumps(seal))
    assert wf._implementation_seal(tmp_path,card,plan,{'fixture':'sha'})['file_sha256']
    for key,value in [('focused_passed',True),('gauge_contracts_passed',False),('data_contracts_passed','true')]:
        changed={**seal,key:value};p.write_text(json.dumps(changed))
        with pytest.raises(ValueError,match='seal'):wf._implementation_seal(tmp_path,card,plan,{'fixture':'sha'})


@pytest.mark.parametrize('fail',[False,True])
def test_closed_outer_or_native_stage_preserves_preflight_and_marks_exact_market_recorder(tmp_path,monkeypatch,fail):
    import qlib
    from qlib.workflow import R
    monkeypatch.setenv('MLFLOW_ALLOW_FILE_STORE','true');monkeypatch.setenv('MLFLOW_DISABLE_AGENT_HINT','1')
    provider=tmp_path/'provider';provider.mkdir();uri=str(tmp_path/'recorders')
    qlib.init(provider_uri=str(provider),region='us',kernels=1)
    with R.start(experiment_name=wf.EXPERIMENT,uri=uri):preflight=R.get_recorder().id
    with R.start(experiment_name=wf.EXPERIMENT,uri=uri):market=R.get_recorder().id
    def operation():
        assert R.exp_manager.active_experiment is None
        qlib.init(provider_uri=str(provider),region='us',kernels=1)
        R.set_uri(str(tmp_path/'other'))
        if fail:raise RuntimeError('synthetic outer/native read error')
        return 'read only'
    if fail:
        with pytest.raises(RuntimeError,match='outer/native'):wf._closed_stage(market,uri,operation)
    else:assert wf._closed_stage(market,uri,operation)=='read only'
    with R.uri_context(uri):
        assert R.get_recorder(experiment_name=wf.EXPERIMENT,recorder_id=preflight).client.get_run(preflight).info.status=='FINISHED'
        assert R.get_recorder(experiment_name=wf.EXPERIMENT,recorder_id=market).client.get_run(market).info.status==('FAILED' if fail else 'FINISHED')


def test_reference_checks_bytes_before_load_and_refuses_partial_restart(tmp_path,monkeypatch):
    import qlib
    from qlib.workflow import R
    monkeypatch.setenv('MLFLOW_ALLOW_FILE_STORE','true');monkeypatch.setenv('MLFLOW_DISABLE_AGENT_HINT','1')
    provider=tmp_path/'provider';provider.mkdir();qlib.init(provider_uri=str(provider),region='us',kernels=1)
    uri=str(tmp_path/'recorders')
    with R.start(experiment_name=wf.EXPERIMENT,uri=uri):
        r=R.get_recorder();r.save_objects(artifact_path='preflight',**{'gate.pkl':{'passed':False}})
        identity=wf._artifact_identity(r,'preflight/gate.pkl');directory=Path(r.get_local_dir())
    ref={'recorder_uri':uri,'recorder_id':r.id,'plan_sha256':'plan','source_sha256':{'x':'sha'},'complete':True,'artifacts':{'gate':identity}}
    assert wf._load_reference(ref,'plan',{'x':'sha'})['gate']=={'passed':False}
    with pytest.raises(ValueError,match='partial'):wf._load_reference({**ref,'complete':False},'plan',{'x':'sha'})
    (directory/'artifacts/preflight/gate.pkl').write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='before deserialization'):wf._load_reference(ref,'plan',{'x':'sha'})


@pytest.mark.parametrize('gate_pass,allow,starts',[(False,False,False),(False,True,False),(True,False,False),(True,True,True)])
def test_first_market_run_requires_both_verified_gate_and_explicit_flag(plan,tmp_path,monkeypatch,gate_pass,allow,starts):
    from src.data import feasible_parent_sparse_search as data
    card=tmp_path/'card';card.mkdir();(card/'selection_plan.json').write_text(json.dumps(plan))
    monkeypatch.setattr(wf,'_source_hashes',lambda *_:{})
    monkeypatch.setattr(wf,'_implementation_seal',lambda *a:{'static':True})
    monkeypatch.setattr(data,'load_feasible_parent_inputs',lambda *a:{'unit':True})
    monkeypatch.setattr(wf,'ensure_preflight',lambda *a:({},pd.DataFrame(),{'passed':gate_pass}))
    monkeypatch.setattr(wf,'_preflight_card',lambda *a:{'market':False})
    called=[]
    monkeypatch.setattr(wf,'_run_market',lambda *a,**k:called.append(True) or {'market':True})
    assert wf.run(tmp_path,card,allow_market_fits=allow)['market'] is starts
    assert bool(called) is starts


def test_reproductions_are_only_explicitly_missing_scheduled_outer_parents(plan,monkeypatch):
    parent={'parent_seed_models':{},'parent_seed_audits':{},'genuinely_missing':[],'folds':{}}
    monkeypatch.setattr(wf,'ProcessPoolExecutor',_InlinePool)
    monkeypatch.setattr(wf,'_parent_reproduction_job',lambda *_:pytest.fail('existing parent was refitted'))
    assert wf.reproduce_missing_outer(parent,plan,_Recorder())['parent_seed_models']=={}


@pytest.mark.parametrize('fault',['no_seal','cached','duplicate','wrong_year','wrong_certificate'])
def test_parent_reproduction_certificate_fails_before_any_old_fit(plan,tmp_path,monkeypatch,fault):
    card=tmp_path/'card';card.mkdir();(card/'selection_plan.json').write_text(json.dumps(plan))
    seal={'plan_sha256':wf.sha(card/'selection_plan.json'),'selection_uses_diagnostic_signal_years':False,
          'selected':{'variant':'price_holdings'}}
    path=card/'selection_seal.json';path.write_text(json.dumps(seal))
    identities=[('price_holdings',2023,s) for s in wf.SEEDS]
    data={'parent_seed_models':{},'parent_seed_audits':{},'genuinely_missing':identities.copy(),
        'folds':{('price_holdings',2023):{}},'selection_seal_sha256':wf.sha(path),
        'parent_source_audit':{'scheduled_variant':'price_holdings','scheduled_years':[2023]}}
    if fault=='no_seal':path.unlink()
    if fault=='cached':data['parent_seed_models'][identities[0]]={}
    if fault=='duplicate':data['genuinely_missing'].append(identities[0])
    if fault=='wrong_year':data['genuinely_missing'][0]=('price_holdings',2016,17)
    if fault=='wrong_certificate':data['selection_seal_sha256']='wrong'
    monkeypatch.setattr(wf,'ProcessPoolExecutor',lambda *a,**k:pytest.fail('bad certificate reached a worker'))
    with pytest.raises(ValueError,match='blocked'):
        wf.reproduce_missing_outer(data,plan,_Recorder(),selection_seal=path)


def test_preflight_is_archived_once_and_reused_without_another_parent_or_search(plan,tmp_path,monkeypatch):
    import qlib
    monkeypatch.setenv('MLFLOW_ALLOW_FILE_STORE','true');monkeypatch.setenv('MLFLOW_DISABLE_AGENT_HINT','1')
    provider=tmp_path/'provider';provider.mkdir();qlib.init(provider_uri=str(provider),region='us',kernels=1)
    card=tmp_path/'research'/wf.EXPERIMENT;card.mkdir(parents=True)
    original=Path(__file__).parents[1]/'research/feasible_parent_sparse_search_20261003'
    for name in ['selection_plan.json','registration_clarification.json']:(card/name).write_bytes((original/name).read_bytes())
    (card/'implementation_seal.json').write_text('{}')
    source=tmp_path/'src/model/fixture.py';source.parent.mkdir(parents=True);source.write_text('# unit fixture')
    hashes={'src/model/fixture.py':wf.sha(source)}
    monkeypatch.setattr(wf,'CANONICAL_ROOT',tmp_path)
    monkeypatch.setattr(wf,'_source_hashes',lambda *_:hashes)
    monkeypatch.setattr(wf,'_implementation_seal',lambda *a:{})
    calls=[]
    def fake_preflight(record,current_plan,current_hashes):
        calls.append(True)
        parents={s:{'healthy':True,'feasible':True,'F':.1} for s in wf.SEEDS}
        searches={s:_audit(.2,'search') for s in wf.SEEDS}
        ensemble={'failure':None,'parent_ensemble':{'F':.1,'feasible':True},
            'candidate_ensemble':{'F':.2,'feasible':True},'selected_origin':'search'}
        for seed in wf.SEEDS:
            record.save_objects(artifact_path=f'preflight/parents/{seed}',**{'revalidation.pkl':parents[seed]})
            record.save_objects(artifact_path=f'preflight/search/{seed}',**{'audit.pkl':searches[seed]})
        table,gate=wf._incremental_gate(parents,searches,ensemble,current_plan)
        gate.update({'source_sha256':current_hashes,'parent_spec':asdict(wf.parent_spec(current_plan)),
            'search_spec':asdict(wf.search_spec(current_plan))})
        record.save_objects(artifact_path='preflight',**{'ensemble_audit.pkl':ensemble,'gate.pkl':gate,'seed_summary.pkl':table})
        return table,gate
    monkeypatch.setattr(wf,'_run_preflight',fake_preflight)
    bundle={'source_audit':{},'source_details':{},'parent_source_audit':{}}
    first=wf.ensure_preflight(tmp_path,card,plan,bundle,hashes,{})
    second=wf.ensure_preflight(tmp_path,card,plan,bundle,hashes,{})
    assert len(calls)==1 and first[0]['recorder_id']==second[0]['recorder_id'] and second[2]['passed']
    assert 'preflight/source_bytes/src/model/fixture.py' in first[0]['artifacts']


def test_parent_full_evidence_is_saved_before_a_later_preflight_search_bug(plan,monkeypatch):
    from src.model import constrained_net_excess as old_model,feasible_parent_sparse_search as search
    class AppendRecorder(_Recorder):
        id='unit-preflight'
        def save_objects(self,*,artifact_path,**objects):self.objects.setdefault(artifact_path,{}).update(objects)
    record=AppendRecorder();calls=[]
    def parent_fit(x,r,*,seed,spec):
        parent=BenchmarkAnchorPolicy(seed=seed);p=parent_payload(parent)
        info={'epochs':[{'checkpoint':p}],'failure':None,'F':.1,'training_feasibility_passed':True}
        calls.append(seed);return parent,info
    monkeypatch.setattr(old_model,'fit_policy',parent_fit)
    monkeypatch.setattr(old_model,'evaluate_policy',lambda *a,**k:{'unit_fixture':True})
    monkeypatch.setattr(search,'revalidate_parent',lambda parent,*a,**k:(prepare_parent(parent)[1],{'F':.1,'feasible':True},{'unit_fixture':True}))
    def buggy(*args,**kwargs):
        assert calls==list(wf.SEEDS)
        for seed in wf.SEEDS:
            assert {'model.pkl','audit.pkl','latest_evaluation.pkl','revalidation.pkl'}.issubset(record.objects[f'preflight/parents/{seed}'])
        raise RuntimeError('unexpected unit search setup bug')
    monkeypatch.setattr(search,'fit_policy',buggy)
    with pytest.raises(RuntimeError,match='setup bug'):wf._run_preflight(record,plan,{})
    assert calls==list(wf.SEEDS)
