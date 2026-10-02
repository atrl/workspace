"""Registered feasible-parent search, guarded synthetic gate, and owned books.

Preflight and market evidence have separate Recorder identities. Notebook
reruns consume sealed evidence; they never silently start another campaign.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import time

import numpy as np
import pandas as pd

from src.analysis.integrated_allocation import paired_metrics, annual_paired_metrics, native_allocation_reconciliation
from src.workflow.stable_excess_integration import CANONICAL_ROOT, benchmark_path, choose_family, json_write, sha
from src.workflow.benchmark_anchor_utility import qualification
from src.workflow.sparse_holdings_net_excess import (
    replay_models, independent_path_audit, execution_diagnostics,
    _source_hashes as _inherited_source_hashes, _boolean,
)
from src.data.sparse_holdings_net_excess import prepare_folds, variant_mapping

EXPERIMENT='feasible_parent_sparse_search_20261003'
SEEDS=(17,41,71)
ASSETS=('soxl','soxx','cash')


def search_spec(plan):
    from src.model.feasible_parent_sparse_search import FeasibleParentSearchSpec
    p=plan['search'];options=p['options'];training=plan['training']
    if (tuple(training['seed_ids'])!=SEEDS or plan['architecture']['trainable_search_coordinates']!=33
            or plan['evaluation']['family_size_adjustment']!=26):
        raise ValueError('blocked: registered search architecture/family/seed contract differs')
    return FeasibleParentSearchSpec(max_evaluations=p['unique_whole_two_cost_path_evaluation_cap'],
        scipy_maxiter=options['maxiter'],rhobeg=options['rhobeg'],tol=options['tol'],catol=options['catol'],
        cost_bps=tuple(training['cost_bps']),minimum_rows=training['minimum_rows'],torch_threads=1,
        risk_tolerance_percentage_points=plan['constraints']['risk_tolerance_percentage_points'],
        risk_group_length=training['risk_group_length'],rms_epsilon=training['rms_epsilon'])


def parent_spec(plan):
    from src.workflow.constrained_net_excess import training_spec
    original=deepcopy(plan['parent_reproduction_protocol'])
    original['architecture']={'feature_count':27,'hidden_units':8}
    return training_spec(original)


def _source_hashes(root):
    names=set(_inherited_source_hashes(root)) | {
        'src/workflow/feasible_parent_sparse_search.py','src/data/feasible_parent_sparse_search.py',
        'src/model/feasible_parent_sparse_search.py','src/model/paired_sparse_paths.py',
    }
    return {name:sha(Path(root)/name) for name in sorted(names)}


def _implementation_seal(root,card,plan,hashes):
    import scipy
    path=Path(card)/'implementation_seal.json'
    if not path.is_file():raise ValueError('blocked: implementation seal required before registered toy or market search')
    seal=json.loads(path.read_text())
    clarification_path=Path(card)/'registration_clarification.json'
    if not clarification_path.is_file():raise ValueError('blocked: registered clarification receipt is missing')
    clarification=json.loads(clarification_path.read_text())
    if (seal.get('registration_clarification_sha256')!=sha(clarification_path)
            or clarification['current_plan_sha256']!=sha(Path(card)/'selection_plan.json')
            or hashlib.sha256(clarification['prior_plan_utf8'].encode()).hexdigest()!=clarification['prior_plan_sha256']
            or clarification['registered_experiment_fits_started'] is not False):
        raise ValueError('blocked: original/current registration clarification identity differs')
    count=seal.get('focused_passed')
    flags=('static_contracts_passed','source_contracts_passed','data_contracts_passed','gauge_contracts_passed')
    if (seal.get('plan_sha256')!=sha(Path(card)/'selection_plan.json') or seal.get('source_sha256')!=hashes
            or not isinstance(count,int) or isinstance(count,bool) or count<=0
            or any(seal.get(flag) is not True for flag in flags)):
        raise ValueError('blocked: implementation seal or static contracts differ')
    if scipy.__version__!=plan['search']['installed_scipy_at_registration']:
        raise ValueError('blocked: installed COBYLA implementation version differs')
    required={'tests/test_feasible_parent_sparse_search.py','tests/test_feasible_parent_sparse_search_data.py',
              'tests/test_feasible_parent_sparse_search_workflow.py','tests/test_post_fee_settlement.py',
              'tests/test_paired_sparse_paths.py'}
    tests=seal.get('test_sha256')
    if not isinstance(tests,dict) or not required.issubset(tests):raise ValueError('blocked: incomplete implementation test seal')
    for name,digest in tests.items():
        path=Path(name)
        if path.is_absolute() or '..' in path.parts or not path.parts or path.parts[0]!='tests' or sha(Path(root)/path)!=digest:
            raise ValueError('blocked: sealed test bytes differ')
    return {**seal,'file_sha256':sha(Path(card)/'implementation_seal.json')}


def _hash_array(value):
    return hashlib.sha256(np.asarray(value,dtype='<f8').tobytes()).hexdigest()


def _number(value):
    return isinstance(value,(int,float,np.integer,np.floating)) and not isinstance(value,(bool,np.bool_)) and bool(np.isfinite(value))


def _validate_retained(payloads,metadata,plan):
    """A numerical failure can retain a valid parent, but cannot qualify."""
    from src.model.feasible_parent_sparse_search import policy_from_dict,is_exact_anchor
    models=[policy_from_dict(payload) for payload in payloads]
    if not _boolean(metadata['training_feasibility_passed'],'retained feasibility'):
        raise ValueError('blocked: retained search policy is not training feasible')
    costs=set(plan['training']['cost_bps']);parts=metadata['centered_utility_by_cost'];F=metadata['centered_worst_utility']
    if (not isinstance(parts,dict) or set(parts)!=costs or not all(_number(v) for v in [F,*parts.values()])
            or F<0 or F!=min(parts.values())):
        raise ValueError('blocked: inconsistent retained two-cost F')
    if _boolean(metadata['qualification_eligible'],'qualification eligibility')!=(metadata['failure'] is None):
        raise ValueError('blocked: numerical failure was hidden in eligibility')
    if metadata['failure'] is not None:
        failure=metadata['failure']
        if not isinstance(failure,dict) or failure.get('accounting_only') is not True or not failure.get('kind'):
            raise ValueError('blocked: failed retained policy lacks explicit accounting-only failure kind')
    origin=metadata['selected_origin']
    if origin not in ('null','parent','search') or (origin=='null' and (F!=0 or not all(is_exact_anchor(m) for m in models))):
        raise ValueError('blocked: retained origin or canonical null differs')
    risks=metadata['selected_risks_by_cost']
    if set(risks)!=costs:raise ValueError('blocked: retained risk cost coverage differs')
    for risk in risks.values():
        values=[risk[k] for k in ('max_drawdown_pct','cvar5_pct','benchmark_max_drawdown_pct','benchmark_cvar5_pct')]
        gaps=np.asarray(risk['raw_percentage_point_gaps'],float)
        expected=np.array([values[2]-values[0],values[3]-values[1]])
        if (not all(_number(v) for v in values) or gaps.shape!=(2,)
                or not all(_number(v) for v in np.asarray(risk['raw_percentage_point_gaps'],dtype=object).ravel())
                or not np.allclose(gaps,expected,atol=1e-12,rtol=0)
                or (gaps>plan['constraints']['risk_tolerance_percentage_points']).any()):
            raise ValueError('blocked: retained raw four-tail gates fail')
        if metadata['failure'] is None:
            normalizers=np.asarray(risk['normalizers'],float)
            if (risk.get('normalizers_valid_for_optimization') is not True or normalizers.shape!=(2,)
                    or not all(_number(v) for v in np.asarray(risk['normalizers'],dtype=object).ravel()) or (normalizers<=0).any()
                    or not np.allclose(normalizers,-np.asarray(values[2:])/100,atol=1e-14,rtol=0)):
                raise ValueError('blocked: healthy retained reference risk normalizers differ')
    if 'trials' in metadata:
        info=metadata['optimizer_diagnostics'];trials=metadata['trials']
        count=info['unique_full_path_evaluations']
        if (not isinstance(count,(int,np.integer)) or isinstance(count,(bool,np.bool_)) or count!=len(trials)
                or not 1<=count<=plan['search']['unique_whole_two_cost_path_evaluation_cap']):
            raise ValueError('blocked: actual unique-evaluation budget differs')
        if metadata['frozen_encoder_sha256_before']!=metadata['frozen_encoder_sha256_after']:
            raise ValueError('blocked: parent encoder changed during search')
    return models


def _accounting_null(parent,seed,x,returns,spec,reason,kind):
    """Registered source/parent failure route, never a replacement search start."""
    from src.model.feasible_parent_sparse_search import zero_anchor,numpy_rollout,policy_to_dict,encoder_hash
    from src.model.benchmark_anchor_utility import policy_from_dict as parent_from_dict
    from src.model.constrained_net_excess import _fallback_evaluation,_risk_record
    model=zero_anchor(seed)
    if parent is not None:model.hidden.load_state_dict(parent_from_dict(parent).hidden.state_dict())
    paths={c:numpy_rollout(model,x,returns,cost_bps=c) for c in spec.cost_bps}
    evaluated=_fallback_evaluation(paths,spec.cost_bps)
    audit={'schema_version':'unstarted_parent_search_accounting_v1','seed':seed,'spec':asdict(spec),
        'training_rows':len(x),'selected_origin':'null','selected_trial':None,'F':0.,'parent_F':None,
        'F_delta_from_parent':None,'centered_worst_utility':0.,'centered_utility_by_cost':{c:0. for c in spec.cost_bps},
        'selected_risks_by_cost':{c:_risk_record(v) for c,v in evaluated['constraints_by_cost'].items()},
        'training_feasibility_passed':True,'training_feasible':True,'training_paths':paths,
        'failure':{'kind':kind,'reason':str(reason),'accounting_only':True,'qualification_eligible':False},
        'qualification_eligible':False,'null_selected':True,'legacy_J':0.,
        'training_feature_sha256':_hash_array(x),'training_return_sha256':_hash_array(returns),
        'frozen_encoder_sha256_before':encoder_hash(model),'frozen_encoder_sha256_after':encoder_hash(model),
        'optimizer_diagnostics':{'method':'not_started','unique_full_path_evaluations':0,
            'requested_callbacks':{'objective':0,'constraints':0},'cache_hits':0,'termination':kind,'solver_result':None}}
    return policy_to_dict(model),audit


def _fit_job(payload):
    from src.model.feasible_parent_sparse_search import fit_policy,policy_to_dict,ParentContractError
    from src.model.benchmark_anchor_utility import policy_from_dict as parent_from_dict
    from src.objective.benchmark_tail_constraints import TailConstraintError
    name,year,seed,x,returns,parent,identity,spec=payload
    started=time.monotonic()
    try:
        model,audit=fit_policy(x,returns,parent_model=parent_from_dict(parent),seed=seed,spec=spec,parent_identity=identity)
        model_payload=policy_to_dict(model)
    except (ParentContractError,TailConstraintError) as error:
        model_payload,audit=_accounting_null(parent,seed,x,returns,spec,error,'parent_revalidation_failure')
    audit['elapsed_seconds']=time.monotonic()-started
    return name,year,seed,model_payload,audit


def _fit_index(rows):
    keys=('variant','year','seed','selected_origin','selected_trial','parent_F','F_delta_from_parent',
          'F','failure','qualification_eligible','training_feasibility_passed','optimizer_diagnostics',
          'training_feature_sha256','training_return_sha256','frozen_encoder_sha256_before','frozen_encoder_sha256_after')
    return [{**{k:a.get(k) for k in keys},'recorder_artifact':f"search_trials/{a['variant']}/{a['year']}/{a['seed']}/audit.pkl"} for a in rows]


def _ensemble_index(rows):
    omit={'training_paths','candidate_training_paths','parent_training_paths'}
    return [{**{k:v for k,v in a.items() if k not in omit},
        'recorder_artifact':f"search_trials/{a['variant']}/{a['year']}/ensemble/audit.pkl"} for a in rows]


def fit_folds(folds,parents,plan,recorder,*,workers=3):
    from src.model.feasible_parent_sparse_search import evaluate_ensemble,policy_to_dict,policy_from_dict,prepare_parent
    spec=search_spec(plan);fitted={};audits=[];ensembles=[];jobs=[]
    for (name,year),fold in folds.items():
        fitted.setdefault(name,{})[year]={'variant':name,'parent_market_variant':fold['parent_market_variant'],
            'model_year':year,'scaler':fold['scaler'],'scaler_sha256':fold['scaler']['scaler_sha256'],
            'seed_models':{},'parent_seed_models':{},'invalid_training':bool(fold['invalid_training']),
            'invalid_training_dates':fold['invalid_training_dates'],
            'training_feature_sha256':_hash_array(fold['features']),'training_return_sha256':_hash_array(fold['returns'])}
        for seed in SEEDS:
            key=(name,year,seed)
            if key not in parents['parent_seed_models']:raise ValueError('blocked: missing authorized parent identity')
            payload=parents['parent_seed_models'][key];identity=parents['parent_seed_audits'][key]
            _,embedded=prepare_parent(payload,seed=seed)
            fitted[name][year]['parent_seed_models'][seed]=policy_to_dict(embedded)
            if fold['invalid_training'] or identity.get('failure') is not None:
                kind='invalid_available_training_volatility' if fold['invalid_training'] else 'parent_revalidation_failure'
                reason=fold['invalid_training_dates'] if fold['invalid_training'] else identity['failure']
                selected,audit=_accounting_null(payload,seed,fold['features'],fold['returns'],spec,
                    reason,kind)
                fitted[name][year]['seed_models'][seed]=selected
                complete={'variant':name,'year':year,'seed':seed,**audit};audits.append(complete)
                recorder.save_objects(artifact_path=f'search_trials/{name}/{year}/{seed}',**{'model.pkl':selected,'audit.pkl':complete})
            else:
                jobs.append((name,year,seed,fold['features'],fold['returns'],payload,identity,spec))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending=[pool.submit(_fit_job,job) for job in jobs]
        fatal=None
        for future in as_completed(pending):
            if future.cancelled():continue
            name,year,seed,payload,audit=future.result()
            record=fitted[name][year]
            for key in ('training_feature_sha256','training_return_sha256'):
                if audit[key]!=record[key]:raise ValueError('blocked: search worker consumed different mature rows')
            complete={'variant':name,'year':year,'seed':seed,**audit};audits.append(complete)
            recorder.save_objects(artifact_path=f'search_trials/{name}/{year}/{seed}',**{'model.pkl':payload,'audit.pkl':complete})
            _validate_retained([payload],audit,plan)
            record['seed_models'][seed]=payload
            print(f"Parent search {name}/{year}/{seed}:origin={audit['selected_origin']} F={audit['F']:.8g} unique={audit['optimizer_diagnostics']['unique_full_path_evaluations']}",flush=True)
            if audit['failure'] is not None and audit['failure'].get('kind')=='unexpected_solver_exception':
                fatal=RuntimeError('blocked: unexpected solver exception; retained trace archived before campaign abort')
                for remaining in pending:
                    if not remaining.done():remaining.cancel()
        if fatal is not None:raise fatal
    for (name,year),fold in folds.items():
        record=fitted[name][year]
        if set(record['seed_models'])!=set(SEEDS):raise ValueError('blocked: missing registered search seed')
        models=[policy_from_dict(record['seed_models'][seed]) for seed in SEEDS]
        originals=[policy_from_dict(record['parent_seed_models'][seed]) for seed in SEEDS]
        source_failures=[a for a in audits if a['variant']==name and a['year']==year and a['failure'] is not None
                         and a['failure'].get('kind') in ('invalid_available_training_volatility','parent_revalidation_failure')]
        if source_failures:
            # Parent/source failures mandate S accounting, even if the whole
            # parent ensemble would otherwise win the economic comparison.
            payload,fallback=_accounting_null(parents['parent_seed_models'][(name,year,SEEDS[0])],SEEDS[0],
                fold['features'],fold['returns'],spec,source_failures[0]['failure']['reason'],source_failures[0]['failure']['kind'])
            selected=[policy_from_dict(payload)]
            audit={**fallback,'anchor_selected':True,'all_three_seed_models_used':True,'component_count':3,
                'candidate_ensemble':None,'parent_ensemble':None,'candidate_ensemble_F':None,
                'parent_ensemble_F':None,'candidate_minus_parent_F':None,
                'candidate_training_paths':None,'parent_training_paths':None,
                'ensemble_comparison_skipped_reason':'source/parent revalidation failure; canonical S accounting only'}
        else:
            selected,audit=evaluate_ensemble(models,fold['features'],fold['returns'],parent_models=originals,spec=spec)
        selected_payloads=[policy_to_dict(model) for model in selected]
        _validate_retained(selected_payloads,audit,plan)
        audit['candidate_component_origins']=[next(a['selected_origin'] for a in audits if (a['variant'],a['year'],a['seed'])==(name,year,seed)) for seed in SEEDS]
        record['ensemble_models']=selected_payloads
        record['candidate_ensemble_models']=[record['seed_models'][seed] for seed in SEEDS]
        record['parent_ensemble_models']=[record['parent_seed_models'][seed] for seed in SEEDS]
        record['ensemble_audit']={k:v for k,v in audit.items() if k not in ('training_paths','candidate_training_paths','parent_training_paths')}
        record['numerical_failure']=bool(audit['failure'] is not None or any(a['failure'] is not None for a in audits if a['variant']==name and a['year']==year))
        ensembles.append({'variant':name,'year':year,**audit})
        recorder.save_objects(artifact_path=f'search_trials/{name}/{year}/ensemble',**{'bundle.pkl':record,'audit.pkl':audit})
    return fitted,audits,ensembles


def _parent_control(annual):
    result=deepcopy(annual)
    for record in result.values():
        record['seed_models']=record['parent_seed_models']
        record['ensemble_models']=record['parent_ensemble_models']
        record['ensemble_audit']={**record['ensemble_audit'],'anchor_selected':False}
        record['numerical_failure']=False
    return result


def _artifact_identity(recorder,artifact):
    path=Path(recorder.get_local_dir())/'artifacts'/artifact
    return {'artifact':artifact,'sha256':sha(path),'bytes':path.stat().st_size}


def _load_reference(reference,plan_hash,hashes,*,complete=True):
    """Verify immutable recorded evidence; partial campaigns are never restarted."""
    from qlib.workflow import R
    if (reference.get('plan_sha256')!=plan_hash or reference.get('source_sha256')!=hashes
            or (complete and reference.get('complete') is not True)):
        raise ValueError('blocked: campaign reference is partial or source/registration changed')
    with R.uri_context(reference['recorder_uri']):
        record=R.get_recorder(experiment_name=EXPERIMENT,recorder_id=reference['recorder_id'])
        if record.client.get_run(record.id).info.status!='FINISHED':
            raise ValueError('blocked: referenced campaign did not finish; automatic restart prohibited')
        base=(Path(record.get_local_dir())/'artifacts').resolve()
        objects={}
        for key,identity in reference['artifacts'].items():
            path=(base/identity['artifact']).resolve()
            if (not path.is_relative_to(base) or not path.is_file() or path.stat().st_size!=identity['bytes']
                    or sha(path)!=identity['sha256']):
                raise ValueError('blocked: referenced campaign artifact is missing or differs before deserialization')
            if identity.get('deserialize',True):objects[key]=record.load_object(identity['artifact'])
            if sha(path)!=identity['sha256']:raise ValueError('blocked: campaign artifact changed during read')
    return objects


def _registered_toy(plan):
    toy=plan['synthetic_known_information']
    if toy['rows']!=504 or toy['feature_count']!=27 or tuple(toy['seed_ids'])!=SEEDS:
        raise ValueError('blocked: registered ordinal positive-control fixture differs')
    x=np.zeros((504,27),dtype=np.float64);x[:,0]=np.where(np.arange(504)%2==0,1.,-1.)
    rS=.002+.008*x[:,0]
    return x,np.c_[3*rS,rS]


def _incremental_gate(parent_audits,search_audits,ensemble,plan):
    margin=plan['synthetic_known_information']['incremental_margin_F']
    if margin!=1e-8:raise ValueError('blocked: registered synthetic incremental margin differs')
    rows=[]
    for seed in SEEDS:
        parent=parent_audits[seed];search=search_audits.get(seed)
        if not _number(parent['F']) or (search is not None and not _number(search['F'])):
            raise ValueError('blocked: synthetic F must be a finite number, not a boolean')
        healthy=_boolean(parent['healthy'],'parent health');feasible=_boolean(parent['feasible'],'parent feasibility')
        parent_pass=healthy and feasible and parent['F']>0
        if search is not None:
            child_feasible=_boolean(search['training_feasibility_passed'],'search feasibility')
            eligible=_boolean(search['qualification_eligible'],'search eligibility')
            if eligible!=(search['failure'] is None):raise ValueError('blocked: inconsistent synthetic search failure/eligibility')
        child_pass=bool(search is not None and search['failure'] is None and child_feasible
            and search['F']>parent['F']+margin and search['selected_origin']=='search')
        rows.append({'seed':seed,'parent_F':parent['F'],'parent_stage_passed':parent_pass,
            'search_ran':search is not None,'retained_origin':search['selected_origin'] if search else None,
            'search_F':search['F'] if search else None,'F_delta':search['F']-parent['F'] if search else None,
            'incremental_seed_gate_passed':child_pass,
            'unique_evaluations':search['optimizer_diagnostics']['unique_full_path_evaluations'] if search else 0,
            'search_failure':search['failure'] if search else None})
    parent_ensemble=ensemble.get('parent_ensemble') if ensemble else None
    candidate=ensemble.get('candidate_ensemble') if ensemble else None
    if parent_ensemble is not None and candidate is not None:
        if not _number(parent_ensemble['F']) or not _number(candidate['F']):raise ValueError('blocked: ensemble F is not finite numeric evidence')
        parent_feasible=_boolean(parent_ensemble['feasible'],'parent ensemble feasibility')
        candidate_feasible=_boolean(candidate['feasible'],'candidate ensemble feasibility')
    ensemble_pass=bool(ensemble is not None and ensemble['failure'] is None and parent_ensemble is not None
        and candidate is not None and parent_feasible and candidate_feasible
        and parent_ensemble['F']>=0 and candidate['F']>=0 and candidate['F']>parent_ensemble['F']+margin
        and ensemble['selected_origin']=='search')
    return pd.DataFrame(rows),{'passed':all(r['parent_stage_passed'] and r['incremental_seed_gate_passed'] for r in rows) and ensemble_pass,
        'parent_stage_passed':all(r['parent_stage_passed'] for r in rows),
        'all_three_seed_increments_passed':all(r['incremental_seed_gate_passed'] for r in rows),
        'actual_candidate_ensemble_increment_passed':ensemble_pass,
        'parent_ensemble_F':parent_ensemble['F'] if parent_ensemble else None,
        'actual_candidate_ensemble_F':candidate['F'] if candidate else None,
        'retained_ensemble_origin':ensemble['selected_origin'] if ensemble else None,
        'margin_F':margin,'rows':504,'chronology':'synthetic ordinal0..503; not market calendar dates',
        'scope':'Exact registered parent reproduction plus one incremental constrained search per seed; not financial qualification'}


def _run_preflight(recorder,plan,hashes):
    """One parent generation and one registered search, with immediate retention."""
    from src.model import constrained_net_excess as old_fit
    from src.model.benchmark_anchor_utility import policy_to_dict as old_serialize,policy_from_dict as old_restore
    from src.model import feasible_parent_sparse_search as model
    from src.objective.benchmark_tail_constraints import TailConstraintError
    x,returns=_registered_toy(plan);old_spec=parent_spec(plan);new_spec=search_spec(plan)
    recorder.save_objects(artifact_path='preflight',**{'inputs.pkl':{'features':x,'returns':returns,
        'chronology':'ordinal','parent_spec':asdict(old_spec),'search_spec':asdict(new_spec),
        'source_sha256':hashes,'declared_fixture':plan['synthetic_known_information']}})
    parent_models={};parent_audits={};search_models={};search_audits={}
    for seed in SEEDS:
        parent,full=old_fit.fit_policy(x,returns,seed=seed,spec=old_spec)
        payload=old_serialize(parent)
        # Persist immediately. A later audit/search failure cannot erase these
        # original120-update states or trigger an unrecorded parent rerun.
        recorder.save_objects(artifact_path=f'preflight/parents/{seed}',**{'model.pkl':payload,'audit.pkl':full})
        latest=old_restore(full['epochs'][-1]['checkpoint']) if full['epochs'] else parent
        latest_eval=old_fit.evaluate_policy(latest,x,returns,spec=old_spec)
        recorder.save_objects(artifact_path=f'preflight/parents/{seed}',**{'latest_evaluation.pkl':latest_eval})
        parent_models[seed]=parent
        status={'healthy':full['failure'] is None,'F':full['F'],'feasible':full['training_feasibility_passed'],
            'original_source':'src/model/constrained_net_excess.py','original_spec':asdict(old_spec)}
        try:
            _,evaluated,parity=model.revalidate_parent(parent,x,returns,spec=new_spec,seed=seed)
            status.update({'F':evaluated['F'],'feasible':evaluated['feasible'],'current_kernel_parity':parity})
        except (model.ParentContractError,TailConstraintError) as error:
            status.update({'healthy':False,'feasible':False,'revalidation_error':str(error)})
        parent_audits[seed]=status
        recorder.save_objects(artifact_path=f'preflight/parents/{seed}',**{'revalidation.pkl':status})
    if all(a['healthy'] and a['feasible'] and a['F']>0 for a in parent_audits.values()):
        for seed in SEEDS:
            selected,full=model.fit_policy(x,returns,parent_model=parent_models[seed],seed=seed,spec=new_spec,
                parent_identity={'scope':'registered_synthetic_parent','seed':seed,'recorder_id':recorder.id})
            search_models[seed]=selected;search_audits[seed]=full
            recorder.save_objects(artifact_path=f'preflight/search/{seed}',
                **{'model.pkl':model.policy_to_dict(selected),'audit.pkl':full})
            _validate_retained([model.policy_to_dict(selected)],full,plan)
            if full['failure'] is not None and full['failure'].get('kind')=='unexpected_solver_exception':break
    ensemble=None
    if len(search_models)==3:
        _,ensemble=model.evaluate_ensemble([search_models[s] for s in SEEDS],x,returns,
            parent_models=[parent_models[s] for s in SEEDS],spec=new_spec)
        ensemble['candidate_component_origins']=[search_audits[s]['selected_origin'] for s in SEEDS]
        recorder.save_objects(artifact_path='preflight',**{'ensemble_audit.pkl':ensemble})
    table,gate=_incremental_gate(parent_audits,search_audits,ensemble,plan)
    gate.update({'source_sha256':hashes,'parent_spec':asdict(old_spec),'search_spec':asdict(new_spec),
        'features_sha256':_hash_array(x),'returns_sha256':_hash_array(returns)})
    recorder.save_objects(artifact_path='preflight',**{'gate.pkl':gate,'seed_summary.pkl':table})
    return table,gate


def ensure_preflight(root,card,plan,bundle,hashes,seal):
    from qlib.workflow import R
    path=card/'preflight_reference.json';plan_hash=sha(card/'selection_plan.json')
    if path.exists():
        reference=json.loads(path.read_text());stored=_load_reference(reference,plan_hash,hashes)
        gate=stored['gate']
        if (gate['source_sha256']!=hashes or gate['parent_spec']!=asdict(parent_spec(plan))
                or gate['search_spec']!=asdict(search_spec(plan))):
            raise ValueError('blocked: cached preflight options or source differ')
        parents={seed:stored[f'preflight/parents/{seed}/revalidation.pkl'] for seed in SEEDS}
        searches={seed:stored[f'preflight/search/{seed}/audit.pkl'] for seed in SEEDS
                  if f'preflight/search/{seed}/audit.pkl' in stored}
        _,checked=_incremental_gate(parents,searches,stored.get('preflight/ensemble_audit.pkl'),plan)
        if any(gate[key]!=value for key,value in checked.items()):
            raise ValueError('blocked: cached incremental gate differs from retained seed/whole-ensemble audits')
        return reference,stored['seed_summary'],gate
    uri=CANONICAL_ROOT/'research'/EXPERIMENT/'qlib_recorder';uri.mkdir(parents=True,exist_ok=True)
    with R.start(experiment_name=EXPERIMENT,uri=str(uri)):
        recorder=R.get_recorder()
        reference={'recorder_id':recorder.id,'recorder_uri':str(uri),'plan_sha256':plan_hash,
            'source_sha256':hashes,'complete':False,'scope':'registered_synthetic_incremental_preflight'}
        reference['registration_clarification_sha256']=sha(card/'registration_clarification.json')
        reference['recorder_local_dir']=str(recorder.get_local_dir())
        reference['recorder_relative_dir']=str(Path(recorder.get_local_dir()).relative_to(CANONICAL_ROOT/'research'/EXPERIMENT))
        json_write(path,reference)
        R.set_tags(role='registered_incremental_preflight',market_model_fits_started=False)
        recorder.log_artifact(str(card/'selection_plan.json'),artifact_path='inputs')
        recorder.log_artifact(str(card/'implementation_seal.json'),artifact_path='inputs')
        recorder.log_artifact(str(card/'registration_clarification.json'),artifact_path='inputs')
        for name,digest in hashes.items():
            source=Path(root)/name
            if sha(source)!=digest:raise ValueError('blocked: source changed before synthetic parent reproduction')
            recorder.log_artifact(str(source),artifact_path='preflight/source_bytes/'+str(Path(name).parent))
        recorder.save_objects(artifact_path='inputs',**{'native_source_audit.pkl':bundle['source_audit'],
            'native_source_details.pkl':bundle['source_details'],'parent_source_audit.pkl':bundle['parent_source_audit'],
            'observer_source_hashes.pkl':hashes})
        table,gate=_run_preflight(recorder,plan,hashes)
        if _source_hashes(root)!=hashes or sha(card/'selection_plan.json')!=plan_hash:
            raise ValueError('blocked: registered preflight implementation changed')
        _implementation_seal(root,card,plan,hashes)
        R.set_tags(preflight_passed=gate['passed'],market_model_fits_started=False)
        # Bind every retained parent/search byte, not only a small pass flag.
        artifacts={}
        base=Path(recorder.get_local_dir())/'artifacts'
        for file in sorted(base.rglob('*')):
            if not file.is_file():continue
            artifact=file.relative_to(base).as_posix()
            if artifact.startswith('code_'):continue
            key='gate' if artifact=='preflight/gate.pkl' else 'seed_summary' if artifact=='preflight/seed_summary.pkl' else artifact
            artifacts[key]={**_artifact_identity(recorder,artifact),'deserialize':file.suffix=='.pkl'}
    reference.update({'complete':True,'passed':gate['passed'],'artifacts':artifacts})
    json_write(path,reference)
    return reference,table,gate


def _preflight_card(card,plan,reference,table,gate,bundle,hashes):
    decision='preflight_passed_market_not_started' if gate['passed'] else 'aborted_before_market_search_failed_incremental_preflight'
    summary={'experiment':EXPERIMENT,'campaign_status':'active_research','objective_achieved':False,
        'record_type':'registered_incremental_synthetic_preflight','decision':decision,
        'recorder_id':reference['recorder_id'],'recorder_uri':reference['recorder_uri'],
        'recorder_local_dir':reference['recorder_local_dir'],'recorder_relative_dir':reference['recorder_relative_dir'],
        'preflight_reference':'preflight_reference.json','preflight':gate,
        'market_model_fits_started':False,'market_model_fits':0,'market_backtests':0,'selected':None,
        'historical_qualification_pass':False,'new_forward_qualified':False,
        'source':plan['source'],'source_audit':bundle['source_audit'],'parent_source_audit':bundle['parent_source_audit'],
        'source_sha256':hashes,'plan_sha256':sha(card/'selection_plan.json'),
        'provider_uri':plan['source']['provider_uri'],'region':'us','market':'all',
        'sidecars':['preflight_seed_summary.csv','preflight_reference.json'],
        'artifact_policy':plan['artifact_policy'],'real_broker_orders':False,'production_change':False,
        'qlib_docs_reference':plan['qlib_docs'],
        'recorder_completion_scope':'FINISHED means synthetic evidence recorded; market search needs separate explicit flag and a passing incremental gate.'}
    table.to_csv(card/'preflight_seed_summary.csv',index=False)
    json_write(card/'summary.json',summary);json_write(card/'selected_model.json',{'selected':None,'reason':decision})
    import yaml
    (card/'workflow.yaml').write_text(yaml.safe_dump({'experiment':EXPERIMENT,'provider_uri':plan['source']['provider_uri'],
        'region':'us','recorder_uri':reference['recorder_uri'],'mode':'registered_synthetic_preflight',
        'market_fits':0,'chronology':'ordinal','artifact_policy':plan['artifact_policy']},sort_keys=False))
    (card/'conclusion.md').write_text('# 可行父模型搜索预检结论\n\n'
        f"完成固定504行人工序列的父模型与增量搜索证据登记。预检通过={gate['passed']}；市场拟合与回测均为0。"
        '序列使用ordinal编号，不是交易日期；F是63倍平均净对数超额，不是投资收益百分比。\n\n'
        '任何子种子或实际三组件集合不能超越父基准时，保持市场门阻断。没有增加预算、换起点或重跑挑结果。'
        '通过合成预检也不等于历史或未来稳定超额。原SIM不变，目标尚未完成。\n')
    return summary


def _closed_stage(recorder_id,uri,operation):
    """Run native/provider work outside R; stamp the same campaign FAILED on error."""
    from qlib.workflow import R
    try:return operation()
    except Exception:
        with R.start(experiment_name=EXPERIMENT,recorder_id=recorder_id,uri=uri,resume=True):raise


def _parent_reproduction_job(payload):
    from src.model import constrained_net_excess as old_fit
    from src.model.benchmark_anchor_utility import policy_to_dict
    name,year,seed,features,returns,spec=payload
    started=time.monotonic();model,audit=old_fit.fit_policy(features,returns,seed=seed,spec=spec)
    audit={**audit,'reproduction_elapsed_seconds':time.monotonic()-started}
    return name,year,seed,policy_to_dict(model),audit


def reproduce_missing_outer(parents,plan,recorder,*,selection_seal=None,workers=3):
    """Only loader-certified never-materialized outer parents; original protocol."""
    from src.model import constrained_net_excess as old_fit
    from src.model.benchmark_anchor_utility import policy_from_dict
    result={**parents,'parent_seed_models':dict(parents['parent_seed_models']),
        'parent_seed_audits':dict(parents['parent_seed_audits'])}
    missing=list(parents['genuinely_missing'])
    if not missing:return result
    if selection_seal is None or not Path(selection_seal).is_file():
        raise ValueError('blocked: missing-parent reproduction requires the persisted development seal')
    seal_path=Path(selection_seal);seal=json.loads(seal_path.read_text());plan_path=seal_path.parent/'selection_plan.json'
    certificate=parents.get('parent_source_audit',{});variant=certificate.get('scheduled_variant');years=certificate.get('scheduled_years',[])
    if (not plan_path.is_file() or json.loads(plan_path.read_text())!=plan or seal['plan_sha256']!=sha(plan_path)
            or parents.get('selection_seal_sha256')!=sha(seal_path)
            or seal['selection_uses_diagnostic_signal_years'] is not False or seal['selected']['variant']!=variant
            or variant not in variant_mapping(plan) or not years or len(years)!=len(set(years))
            or any(not isinstance(y,int) or isinstance(y,bool) for y in years)
            or not set(years).issubset(plan['training']['diagnostic_signal_years'])):
        raise ValueError('blocked: missing-parent certificate is not bound to scheduled sealed outer evaluation')
    expected={(variant,year,seed) for year in years for seed in SEEDS}
    if (any(not isinstance(key,tuple) or len(key)!=3 or not isinstance(key[0],str)
            or not isinstance(key[1],int) or isinstance(key[1],bool)
            or not isinstance(key[2],int) or isinstance(key[2],bool) for key in missing)
            or len(missing)!=len(set(missing)) or not set(parents['parent_seed_models']).issubset(expected)
            or set(missing)!=(expected-set(parents['parent_seed_models']))
            or any((name,year) not in parents['folds'] for name,year,seed in missing)):
        raise ValueError('blocked: duplicate, cached or unscheduled parent reproduction identity')
    spec=parent_spec(plan)
    jobs=[(name,year,seed,parents['folds'][(name,year)]['features'],parents['folds'][(name,year)]['returns'],spec)
          for name,year,seed in missing]
    reproduction_summary=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(_parent_reproduction_job,job) for job in jobs]):
            name,year,seed,payload,audit=future.result();key=(name,year,seed);fold=parents['folds'][(name,year)]
            where=f'parent_reproductions/{name}/{year}/{seed}'
            provenance={'origin':'new_reproduction_of_never_materialized_outer_parent','variant':name,
                'parent_market_variant':fold['parent_market_variant'],'year':year,'seed':seed,
                'original_protocol':plan['parent_reproduction_protocol'],'training_cutoff':fold['audit']['cutoff'],
                'scaler_sha256':fold['scaler']['scaler_sha256'],'recorder_id':recorder.id,'artifact':where}
            recorder.save_objects(artifact_path=where,**{'model.pkl':payload,'audit.pkl':audit,'provenance.pkl':provenance})
            if (audit['spec']!=asdict(spec) or audit['training_rows']!=len(fold['rows'])
                    or audit['training_feature_sha256']!=_hash_array(fold['features'])
                    or audit['training_return_sha256']!=_hash_array(fold['returns'])):
                raise ValueError('blocked: reproduced parent consumed different mature source/protocol')
            latest=policy_from_dict(audit['epochs'][-1]['checkpoint']) if audit['epochs'] else policy_from_dict(payload)
            latest_eval=old_fit.evaluate_policy(latest,fold['features'],fold['returns'],spec=spec)
            recorder.save_objects(artifact_path=where,**{'latest_evaluation.pkl':latest_eval})
            if audit['failure'] is not None:
                provenance['original_parent_failure']=audit['failure']
            result['parent_seed_models'][key]=payload
            result['parent_seed_audits'][key]={**provenance,'spec':audit['spec'],'F':audit['F'],
                'failure':audit['failure'],'selected_epoch':audit['selected_epoch'],'null_selected':audit['null_selected']}
            reproduction_summary.append({'variant':name,'year':year,'seed':seed,'failure':audit['failure'],
                'original_protocol_updates_completed':audit['optimizer_diagnostics']['updates_completed'],
                'elapsed_seconds':audit.get('reproduction_elapsed_seconds'),'artifact':where})
    result['parent_reproduction_summary']=reproduction_summary
    return result


def run(root,card=None,*,allow_market_fits=False,workers=3):
    """First verify/reuse the registered toy; a separate root flag permits markets."""
    from src.data.feasible_parent_sparse_search import load_feasible_parent_inputs
    root=Path(root).resolve();card=Path(card) if card else root/'research'/EXPERIMENT
    plan=json.loads((card/'selection_plan.json').read_text());plan_hash=sha(card/'selection_plan.json')
    if plan['experiment']!=EXPERIMENT or plan['registered_before_market_model_fits_or_portfolio_results'] is not True:
        raise ValueError('blocked: registered Phase9 notebook plan required')
    hashes=_source_hashes(root);seal=_implementation_seal(root,card,plan,hashes)
    bundle=load_feasible_parent_inputs(root,plan)
    reference,table,gate=ensure_preflight(root,card,plan,bundle,hashes,seal)
    market_reference=card/'market_reference.json'
    if market_reference.exists():
        market_ref=json.loads(market_reference.read_text())
        if not gate['passed'] or market_ref.get('preflight_reference_sha256')!=sha(card/'preflight_reference.json'):
            raise ValueError('blocked: cached market evidence does not bind this passing preflight')
        stored=_load_reference(market_ref,plan_hash,hashes)
        return stored['summary']
    if not gate['passed'] or allow_market_fits is not True:
        return _preflight_card(card,plan,reference,table,gate,bundle,hashes)
    return _run_market(root,card,plan,bundle,hashes,seal,reference,workers=workers)


def _run_market(root,card,plan,bundle,hashes,implementation_seal,preflight_reference,*,workers=3):
    import qlib
    from qlib.workflow import R
    from src.workflow.conditional_variance import hac_attribution
    from src.data.integrated_allocation import COMMON_PRIMARY_FEATURES
    from src.analysis.convex_qlib import native_open_reconciliation as precise_native
    from src.workflow.trainable_regime_policy_nested_wf import _remove_qlib_code_snapshot_artifacts
    root=Path(root).resolve();card=Path(card)
    plan_hash=sha(card/'selection_plan.json');source=plan['source'];costs=plan['evaluation']['cost_bps']
    raw=bundle['raw'];prices=bundle['prices'];forecasts=bundle['forecasts']
    forecast_audit=bundle['forecast_audit'];forecast_models=bundle['forecast_models'];dataset=bundle['dataset']
    uri=CANONICAL_ROOT/'research'/EXPERIMENT/'qlib_recorder';uri.mkdir(parents=True,exist_ok=True)
    R.set_uri(str(uri))
    reference_path=card/'market_reference.json'
    with R.start(experiment_name=EXPERIMENT,uri=str(uri)):
        recorder=R.get_recorder()
        market_reference={'recorder_id':recorder.id,'recorder_uri':str(uri),'plan_sha256':plan_hash,
            'source_sha256':hashes,'complete':False,'stage':'development_search',
            'preflight_reference_sha256':sha(card/'preflight_reference.json')}
        json_write(reference_path,market_reference)
        R.set_tags(role='market_research_after_incremental_preflight',stage='development_search')
        R.log_params(plan_sha256=plan_hash,source_recorder_id=source['recorder_id'],
            parent_input_recorder_id=plan['parent_input_source']['recorder_id'],objective='mean_net_log_excess_with_benchmark_tail_constraints',campaign_status='active_research')
        recorder.log_artifact(str(card/'selection_plan.json'),artifact_path='inputs')
        recorder.log_artifact(str(card/'implementation_seal.json'),artifact_path='inputs')
        recorder.log_artifact(str(card/'registration_clarification.json'),artifact_path='inputs')
        recorder.log_artifact(str(card/'preflight_reference.json'),artifact_path='inputs')
        R.save_objects(artifact_path='inputs',**{'source_hashes.pkl':hashes,'source_audit.pkl':bundle['source_audit'],
            'primary_frame.pkl':raw,'execution_prices.pkl':prices})
        R.save_objects(artifact_path='inputs',**{'source_details.pkl':bundle['source_details']})
        R.save_objects(artifact_path='inputs',**{'development_parent_source_audit.pkl':bundle['parent_source_audit'],
            'development_parent_source_artifacts.pkl':bundle['parent_source_artifacts'],
            'development_parent_bindings.pkl':bundle['parent_seed_audits']})
        valid=raw[['soxl_return','soxx_return']].notna().all(axis=1)
        development=forecasts.loc[valid & raw.index.year.isin(plan['training']['development_signal_years'])]
        outer=forecasts.loc[valid & raw.index.year.isin(plan['training']['diagnostic_signal_years'])]
        variants=[v['name'] for v in plan['variants']]
        folds=bundle['development_folds']
        for identity,fold in folds.items():
            previous=bundle['parent_dev_folds'][(bundle['variant_mapping'][identity[0]],identity[1])]
            if (fold['scaler']!=previous['scaler'] or not np.array_equal(fold['features'],previous['features'])
                    or not np.array_equal(fold['returns'],previous['returns'])):
                raise ValueError('blocked: constrained study changed prepared development inputs')
        fitted,diagnostics,ensemble_audits=fit_folds(folds,bundle,plan,recorder,workers=workers)
        R.save_objects(artifact_path='inputs',**{'forecasts.pkl':forecasts,'forecast_fit_audit.pkl':forecast_audit,
            'development_fold_inputs.pkl':folds})
        R.save_objects(artifact_path='models',**{'forecast_models.pkl':forecast_models,'development.pkl':fitted})
        benchmarks={c:benchmark_path(development,cost_bps=c) for c in costs}
        evidence={};paths={};parent_paths={};replays=[];seed_rows=[];path_audits=[];parent_comparisons=[]
        for name in variants:
            candidate={}
            for cost in costs:
                path=replay_models(development,fitted[name],variant=name,cost_bps=cost,plan=plan)
                paths[(name,cost)]=path;candidate[cost]=path
                replays.append(path.assign(variant=name,seed='ensemble',segment='development'))
                path_audits.append({'variant':name,'seed':'ensemble',**independent_path_audit(development,path,cost_bps=cost)})
                parent_path=replay_models(development,_parent_control(fitted[name]),variant=name,cost_bps=cost,plan=plan)
                parent_paths[(name,cost)]=parent_path
                replays.append(parent_path.assign(variant=name+'_embedded_parent_control',seed='ensemble',segment='development'))
                parent_comparisons.append({'variant':name,'cost_bps':cost,'segment':'development','reference':'all_three_embedded_parent_seed_models',**paired_metrics(path,parent_path)})
                path_audits.append({'variant':name+'_embedded_parent_control','seed':'ensemble',**independent_path_audit(development,parent_path,cost_bps=cost)})
                for seed in SEEDS:
                    trial=replay_models(development,fitted[name],variant=name,cost_bps=cost,seed=seed,plan=plan)
                    seed_rows.append({'variant':name,'seed':seed,'cost_bps':cost,**paired_metrics(trial,benchmarks[cost])})
                    replays.append(trial.assign(variant=name,seed=str(seed),segment='development'))
                    path_audits.append({'variant':name,'seed':seed,**independent_path_audit(development,trial,cost_bps=cost)})
            evidence[name]=qualification(candidate,benchmarks,pd.DataFrame(seed_rows).query('variant==@name'),plan=plan)
        selected,ranking=choose_family(evidence)
        seal={'plan_sha256':plan_hash,'selected':selected,'selection_uses_diagnostic_signal_years':False,
            'development_last_exit':str(development.exit_date.max().date()),'campaign_status':'active_research'}
        json_write(card/'selection_seal.json',seal);recorder.log_artifact(str(card/'selection_seal.json'),artifact_path='inputs')
        print(f'Feasible parent search development sealed: {selected}',flush=True)
        chosen=selected['variant']
        R.set_tags(stage='development_sealed')
    # Parent loader may initialize/read Qlib globals. It must run with no active
    # writable child Recorder; failures are stamped on this exact market run.
    from src.data.feasible_parent_sparse_search import load_cached_outer_parents
    outer_parents=_closed_stage(recorder.id,str(uri),lambda:load_cached_outer_parents(root,plan,bundle,
        variant=chosen,years=plan['training']['diagnostic_signal_years'],selection_seal=card/'selection_seal.json'))
    market_reference['stage']='outer_parent_and_search';json_write(reference_path,market_reference)
    with R.start(experiment_name=EXPERIMENT,recorder_id=recorder.id,uri=str(uri),resume=True):
        if R.get_recorder().id!=recorder.id:raise ValueError('blocked: resumed another market Recorder')
        outer_parents=reproduce_missing_outer(outer_parents,plan,recorder,selection_seal=card/'selection_seal.json',workers=workers)
        outer_folds=outer_parents['folds']
        outer_fitted,outer_diagnostics,outer_ensemble=fit_folds(outer_folds,outer_parents,plan,recorder,workers=workers)
        R.save_objects(artifact_path='inputs',**{'outer_parent_source_audit.pkl':outer_parents['parent_source_audit'],
            'outer_parent_source_artifacts.pkl':outer_parents['parent_source_artifacts'],
            'outer_parent_bindings.pkl':outer_parents['parent_seed_audits'],
            'parent_reproduction_summary.pkl':outer_parents.get('parent_reproduction_summary',[])})
        diagnostics+=outer_diagnostics;ensemble_audits+=outer_ensemble
        comparisons=[r['metrics'].assign(variant=name,segment='development') for name,r in evidence.items()]
        annuals=[r['annual'].assign(variant=name,segment='development') for name,r in evidence.items()]
        controls=[];attribution=[];windows=[]
        for name in variants:
            for cost in costs:
                path=paths[(name,cost)]
                if name!='holdings_only': controls.append({'variant':name,'reference':'holdings_only','cost_bps':cost,
                    **paired_metrics(path,paths[('holdings_only',cost)])})
                attribution.append({'variant':name,'cost_bps':cost,'signal_year':'all',**hac_attribution(path,benchmarks[cost])})
                for year in plan['training']['development_signal_years']:
                    mask=path.signal_date.dt.year.eq(year)
                    attribution.append({'variant':name,'cost_bps':cost,'signal_year':str(year),
                        **hac_attribution(path.loc[mask].reset_index(drop=True),benchmarks[cost].loc[mask].reset_index(drop=True))})
        outer_frames={}
        for mode in ('annual_refit','frozen2022_parameters'):
            frame=outer.copy();frozen=None
            if mode=='frozen2022_parameters':
                frozen=2023
                if chosen=='price_vol_holdings':
                    prediction=forecast_models[2023].predict(frame.loc[:,list(COMMON_PRIMARY_FEATURES)])
                    frame.loc[:,['cash_advantage_prediction','future_vol_prediction']]=prediction
                    frame.loc[:,'forecast_available']=1.
                    frame['forecast_model_available_at']=pd.Timestamp('2023-01-01',tz='America/New_York').tz_convert('UTC')
            outer_frames[mode]=frame
            for cost in costs:
                reference=benchmark_path(frame,cost_bps=cost)
                path=replay_models(frame,outer_fitted[chosen],variant=chosen,cost_bps=cost,frozen_year=frozen,plan=plan)
                name=chosen+'_'+mode
                replays.append(path.assign(variant=name,seed='ensemble',segment='reused_diagnostic'))
                comparisons.append(pd.DataFrame([{'variant':name,'segment':'reused_diagnostic','cost_bps':cost,**paired_metrics(path,reference)}]))
                annuals.append(annual_paired_metrics(path,reference).assign(variant=name,segment='reused_diagnostic',cost_bps=cost))
                path_audits.append({'variant':name,'seed':'ensemble',**independent_path_audit(frame,path,cost_bps=cost)})
                mask=path.trade_date.ge('2026-08-03')
                if mask.any():
                    windows.append({'variant':name,'cost_bps':cost,'window':'trade_date>=2026-08-03; through2026-09-25open','start_position':'carried from diagnostic2023start',
                        **paired_metrics(path.loc[mask],reference.loc[mask])})
                    fresh=replay_models(frame.loc[mask.to_numpy()],outer_fitted[chosen],variant=chosen,cost_bps=cost,frozen_year=frozen,plan=plan)
                    fresh_ref=benchmark_path(frame.loc[mask.to_numpy()],cost_bps=cost)
                    windows.append({'variant':name,'cost_bps':cost,'window':'trade_date>=2026-08-03; through2026-09-25open','start_position':'CASH at first window trade',
                        **paired_metrics(fresh,fresh_ref)})
        for cost in costs:
            replays.append(benchmarks[cost].assign(variant='SOXX_buy_hold',seed='benchmark',segment='development'))
            replays.append(benchmark_path(outer,cost_bps=cost).assign(variant='SOXX_buy_hold',seed='benchmark',segment='reused_diagnostic'))
        comparison=pd.concat(comparisons,ignore_index=True);annual=pd.concat(annuals,ignore_index=True)
        gates=pd.concat([r['gates'].assign(variant=n) for n,r in evidence.items()],ignore_index=True)
        bootstrap=pd.concat([r['bootstrap'].assign(variant=n) for n,r in evidence.items()],ignore_index=True)
        all_replays=pd.concat(replays,ignore_index=True)
        sparsity,fee_decomposition,seed_equality=execution_diagnostics(all_replays,
            {'development':development,'reused_diagnostic':outer})
        if (sha(card/'selection_plan.json')!=plan_hash or _source_hashes(root)!=hashes
                or sha(card/'implementation_seal.json')!=implementation_seal['file_sha256']
                or _implementation_seal(root,card,plan,hashes)!=implementation_seal):
            raise ValueError('blocked: registered implementation changed during experiment')
        R.save_objects(artifact_path='models',**{'selected_annual_refits.pkl':outer_fitted})
        R.save_objects(artifact_path='inputs',**{'diagnostic_fold_inputs.pkl':outer_folds,'diagnostic_frames.pkl':outer_frames})
        R.save_objects(artifact_path='analysis',**{'portfolio_replays.pkl':all_replays,
            'comparison.pkl':comparison,'annual.pkl':annual,'gates.pkl':gates,'bootstrap.pkl':bootstrap,
            'seed_metrics.pkl':pd.DataFrame(seed_rows),'training_diagnostics.pkl':_fit_index(diagnostics),
            'ensemble_audits.pkl':_ensemble_index(ensemble_audits),'independent_path_audits.pkl':path_audits,
            'controls.pkl':pd.DataFrame(controls),'attribution.pkl':pd.DataFrame(attribution),'window_metrics.pkl':pd.DataFrame(windows),
            'execution_sparsity.pkl':sparsity,'saved_path_fee_decomposition.pkl':fee_decomposition,
            'seed_path_equality.pkl':seed_equality,'paired_parent_comparison.pkl':pd.DataFrame(parent_comparisons)})
        R.log_metrics(historical_qualification_pass=float(selected['historical_qualification_pass']),registered_variants=3,seeds_per_variant=3)
        recdir=Path(recorder.get_local_dir())
        pd.testing.assert_frame_equal(comparison,recorder.load_object('analysis/comparison.pkl'))
    native=_closed_stage(recorder.id,str(uri),lambda: {c:native_allocation_reconciliation(
        development.reset_index(drop=True),paths[(chosen,c)],provider_uri=root/source['provider_uri'],
        cost_bps=c,native_reconciler=precise_native) for c in costs})
    _remove_qlib_code_snapshot_artifacts(recdir)
    training_rows=[]
    for audit in diagnostics:
        stats=audit['optimizer_diagnostics'];solver=stats.get('solver_result') or {}
        training_rows.append({k:audit.get(k) for k in ('variant','year','seed','selected_origin','selected_trial','parent_F','F_delta_from_parent','F','failure','training_feasibility_passed')} | {
            'unique_evaluations':stats['unique_full_path_evaluations'],'cache_hits':stats['cache_hits'],
            'requested_objective_callbacks':stats['requested_callbacks'].get('objective',0),
            'requested_constraint_callbacks':stats['requested_callbacks'].get('constraints',0),
            'solver_nfev':solver.get('nfev'),'termination':stats['termination']})
    tables={'model_selection.csv':ranking,'backtest_benchmark_comparison.csv':comparison,'annual_metrics.csv':annual,
        'development_gates.csv':gates,'bootstrap_analysis.csv':bootstrap,'seed_metrics.csv':pd.DataFrame(seed_rows),
        'training_summary.csv':pd.DataFrame(training_rows),'forecast_training_audit.csv':forecast_audit,
        'control_comparison.csv':pd.DataFrame(controls),'beta_alpha_attribution.csv':pd.DataFrame(attribution),
        'window_metrics.csv':pd.DataFrame(windows),'execution_sparsity.csv':sparsity,
        'saved_path_fee_decomposition.csv':fee_decomposition,'seed_path_equality.csv':seed_equality,
        'paired_parent_comparison.csv':pd.DataFrame(parent_comparisons)}
    for filename,table in tables.items():table.to_csv(card/filename,index=False)
    summary={'experiment':EXPERIMENT,'campaign_status':'active_research','objective_achieved':False,
        'decision':'historical_candidate_requires_source_sensitivity_and_new_forward' if selected['historical_qualification_pass'] else 'continue_research_no_historical_qualification',
        'selected':selected,'historical_qualification_pass':bool(selected['historical_qualification_pass']),
        'new_forward_qualified':False,'source_sensitivity_completed':False,'source':source,'source_audit':bundle['source_audit'],
        'seed_scope':'Optimization-start stability only. holdings_only masks market inputs but retains owned state; deterministic initial symmetry can make its seeds identical. Seeds are never independent market observations.',
        'source_sha256':hashes,'plan_sha256':plan_hash,'implementation_seal_sha256':implementation_seal['file_sha256'],'provider_uri':source['provider_uri'],'region':'us','market':'all',
        'source_coverage':'Explicit project data/model/workflow/accounting/risk/native-strategy and evidence-helper footprint; not a full transitive or third-party dependency hash.',
        'training_objective':'F=min_c63mean_net_log_excess; legacy_J_report_only',
        'parent_work':{'development_cached_parents':len(bundle['parent_seed_models']),
            'scheduled_outer_cached_parents':outer_parents['parent_source_audit']['cached_parent_count'],
            'scheduled_outer_reproductions':len(outer_parents.get('parent_reproduction_summary',[])),
            'original_parent_updates_completed':sum(a['original_protocol_updates_completed'] for a in outer_parents.get('parent_reproduction_summary',[])),
            'new_COBYLA_fit_count':len(diagnostics),
            'new_unique_two_cost_evaluations':sum(a['optimizer_diagnostics']['unique_full_path_evaluations'] for a in diagnostics),
            'budget_comparability':plan['training']['budget_comparability']},
        'architecture':plan['architecture'],'search':plan['search'],'settlement':plan['settlement'],'preflight_reference':preflight_reference,'parent_variant_mapping':bundle['variant_mapping'],
        'training_constraints':'same_benchmark_anchored_MaxDD_and_exact_CVaR5_at_both_costs',
        'normalized_cost_state_scope':'Post-fee normalized targets are cost-blind; fee funding changes NAV/trade amounts, not necessarily normalized holdings.',
        'recorder_id':recorder.id,'recorder_uri':str(uri),'recorder_local_dir':str(recdir),
        'recorder_relative_dir':str(recdir.relative_to(CANONICAL_ROOT/'research'/EXPERIMENT)),
        'selection_uses_diagnostic_signal_years':False,'development_last_exit':str(development.exit_date.max().date()),
        'native_qlib_audit':{str(c):v['audit'] for c,v in native.items()},'independent_accounting_audit':path_audits,
        'production_change':False,'real_broker_orders':False,'market_model_fits_started':True,'market_search_fits':len(diagnostics),'sidecars':list(tables),
        'training_audit_storage':'Unique33-coordinate COBYLA trials and callback/cache counts in per-fit Recorder artifacts; original parent Adam states remain separately identified; no new Adam epochs are claimed.',
        'artifact_policy':plan['artifact_policy'],
        'qlib_docs_reference':plan['qlib_docs'],
        'limitations':['Historical years have been reviewed; no fresh forward evidence or guaranteed future excess.',
            'Sparse actions and path risks are nonsmooth; finite derivative-free search has no global optimality guarantee.',
            'New search uses finite COBYLA function evaluations, not gradients or TBPTT. The inherited parent was trained with its original approximate derivative/truncation conventions.',
            'Retained training models satisfy historical mean-excess and tail constraints; this does not guarantee future wealth/risk or optimizer convergence.',
            'Source basis retains published-versus-server adjustment differences and16secondary observations.',
            'Historical open prices and continuous adjusted shares are not evidence of actual spread, impact or partial fills.']}
    json_write(card/'summary.json',summary);json_write(card/'selected_model.json',{'selected':selected,'research_only':True,'new_forward_qualified':False})
    import yaml
    (card/'workflow.yaml').write_text(yaml.safe_dump({'experiment':EXPERIMENT,'provider_uri':source['provider_uri'],'region':'us',
        'recorder_uri':str(uri),'model_selection':'selection_plan.json','artifact_policy':summary['artifact_policy'],
        'segments':{k:plan['training'][k] for k in ('development_signal_years','diagnostic_signal_years')}},sort_keys=False))
    lines=['# 可行父模型上的稀疏搜索：训练结论','',f'完成3组信息对照、每组3个种子；固定原父模型编码器，从同身份可行父模型进行登记的33坐标COBYLA搜索。完整路径计入所有费用，优化F并保留四风险门可行状态；继承父模型不能视为新的搜索改进。历史资格通过={selected["historical_qualification_pass"]}；稳定净超额目标仍未完成。','',
        '开发期2016–2022，10bp：']
    for row in comparison[comparison.segment.eq('development') & comparison.cost_bps.eq(10)].itertuples():
        lines.append(f'- {row.variant}：收益{row.return_pct:.2f}%，相对SOXX终值{row.relative_wealth:.4f}，MaxDD{row.max_drawdown_pct:.2f}%，CVaR5{row.cvar5_pct:.2f}%。')
    lines+=['','各年差异见annual_metrics.csv，8月3日实际交易日起的持续持仓与独立现金起点分别见window_metrics.csv。三种子不是三条独立市场历史。',
        f'决定：{summary["decision"]}。2023以后为已查看的诊断期，不能作为新留出样本；原SIM未切换。',
        '风险：本轮是有限预算的COBYLA全路径函数评估，没有使用梯度、TBPTT或对偶更新，仍无全局收敛与未来风险保证。继承父模型保留其原训练导数和截断近似的局限；行情口径和真实成交差异仍存在。',
        '下一步：按失败门与训练诊断继续定位；若历史门通过，先完成来源敏感性，再冻结进入新的前向证据期。',f'Qlib Recorder：{recorder.id}。']
    (card/'conclusion.md').write_text('\n'.join(lines)+'\n')
    with R.start(experiment_name=EXPERIMENT,recorder_id=recorder.id,uri=str(uri),resume=True):
        R.set_tags(stage='complete',historical_qualification_pass=selected['historical_qualification_pass'])
        R.save_objects(artifact_path='portfolio_analysis',**{'native_reconciliation.pkl':native})
        R.save_objects(artifact_path='analysis',**{'final_summary.pkl':summary})
        bound={}
        base=Path(recorder.get_local_dir())/'artifacts'
        for file in sorted(base.rglob('*.pkl')):
            artifact=file.relative_to(base).as_posix();key='summary' if artifact=='analysis/final_summary.pkl' else artifact
            bound[key]=_artifact_identity(recorder,artifact)
    market_reference.update({'complete':True,'stage':'complete','artifacts':bound})
    json_write(reference_path,market_reference)
    return summary
