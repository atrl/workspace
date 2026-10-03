"""Nested sparse-expert research with immutable inputs and owned-state replay."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import hashlib
import json
import pickle
from dataclasses import asdict
from tempfile import TemporaryDirectory
import time
import numpy as np
import pandas as pd

from src.analysis.integrated_allocation import paired_metrics,annual_paired_metrics,native_allocation_reconciliation
from src.workflow.stable_excess_integration import CANONICAL_ROOT,benchmark_path,choose_family,json_write,sha
from src.workflow.benchmark_anchor_utility import qualification
from src.data.sparse_holdings_net_excess import load_sparse_inputs, prepare_folds, transform_for_variant, variant_mapping
from src.analysis.integrated_allocation import _dates, validate_replay
from src.workflow.constrained_net_excess import _source_hashes as _parent_source_hashes

EXPERIMENT='sparse_holdings_net_excess_20261003'
SEEDS=(17,41,71)
ASSETS=('soxl','soxx','cash')


def _preflight_probe_audit(probe, plan, receipt, source_files):
    """Check retained ordinal evidence only; never fit or regenerate targets."""
    from src.workflow.anchor_checkpoint_feasibility import _same_evidence
    from src.objective.benchmark_tail_constraints import evaluate_constraints,reference_risk
    x=np.asarray(probe['features']);returns=np.asarray(probe['returns'])
    toy=plan['synthetic_known_information'];n=int(toy['rows'])
    expected_x=np.zeros((n,27));expected_x[:,0]=np.where(np.arange(n)%2==0,1.,-1.)
    underlying=.002+.008*expected_x[:,0];expected_r=np.c_[3*underlying,underlying]
    if (not isinstance(probe['features'],np.ndarray) or not isinstance(probe['returns'],np.ndarray)
            or x.dtype!=np.float64 or returns.dtype!=np.float64 or not np.array_equal(x,expected_x)
            or not np.array_equal(returns,expected_r) or asdict(probe['spec'])!=asdict(training_spec(plan))):
        raise ValueError('blocked: retained synthetic arrays/spec differ from registered ordinal fixture')
    if probe['source_sha256']!=receipt['retention_reproduction_source_sha256']:
        raise ValueError('blocked: full traces must retain their reproduction source identity')
    for name,key in [('initial_model.py','initial_failed_probe_source_sha256'),
                     ('retention_model.py','retention_reproduction_source_sha256'),
                     ('initial_probe_summary.json','initial_probe_summary_sha256')]:
        if hashlib.sha256(source_files[name]).hexdigest()!=receipt[key]:
            raise ValueError('blocked: supplied synthetic source snapshot identity differs')
    if source_files['initial_model.py'].replace(b'reported_trade_count_gt_1e12',b'reported_trade_count_gt_1e_minus12')!=source_files['retention_model.py']:
        raise ValueError('blocked: probe source change is not the declared diagnostic-key-only change')
    prior=json.loads(source_files['initial_probe_summary.json'])
    if prior!=probe['prior_probe'] or prior.get('market_data_used') is not False:
        raise ValueError('blocked: initial probe summary differs or uses market data')
    fits=probe['fits']
    if len(fits)!=len(SEEDS) or {fit['seed'] for fit in fits}!=set(SEEDS):
        raise ValueError('blocked: retained synthetic seed inventory differs')
    feature_hash=hashlib.sha256(x.tobytes()).hexdigest();return_hash=hashlib.sha256(returns.tobytes()).hexdigest()
    rows=[]
    for fit in fits:
        info=fit['metadata'];epochs=info['epochs'];seed=fit['seed']
        if (info['seed']!=seed or info['spec']!=asdict(probe['spec']) or info['training_rows']!=n
                or info['training_feature_sha256']!=feature_hash or info['training_return_sha256']!=return_hash
                or [entry['epoch'] for entry in epochs]!=list(range(plan['training']['updates']+1))
                or any('checkpoint' not in entry or not np.isfinite(entry['F']) for entry in epochs)
                or info['optimizer_diagnostics']['updates_completed']!=plan['training']['updates']
                or info.get('final_optimizer_state') is None
                or not _same_evidence(epochs,info['history'])):
            raise ValueError('blocked: retained toy training metadata/traces are incomplete')
        parts=[];risk_pass=[]
        for cost in plan['training']['cost_bps']:
            path=info['training_paths'][cost];net=np.asarray(path['net_equity']);benchmark=np.asarray(info['benchmark_net_equity'][cost])
            expected_benchmark=1+returns[:,1];expected_benchmark=expected_benchmark.copy();expected_benchmark[0]/=1+cost/10000
            if (net.shape!=(n,) or benchmark.shape!=(n,) or not np.isfinite(net).all() or (net<=0).any()
                    or not np.allclose(benchmark,expected_benchmark,atol=1e-12,rtol=0)):
                raise ValueError('blocked: saved synthetic path or fee-funded benchmark differs')
            F=float(63*np.mean(np.log(net)-np.log(benchmark)))
            if not np.isfinite(info['centered_utility_by_cost'][cost]) or abs(F-info['centered_utility_by_cost'][cost])>1e-12:
                raise ValueError('blocked: saved synthetic utility does not reconstruct')
            parts.append(F)
            risk=evaluate_constraints(np.log(net),reference_risk(np.log(benchmark)),
                risk_tolerance_percentage_points=plan['constraints']['risk_tolerance_percentage_points'])
            risk_pass.append(risk['feasible'])
        F=min(parts)
        if (not np.isfinite([info['F'],info['centered_worst_utility']]).all()
                or abs(F-info['F'])>1e-12 or abs(F-info['centered_worst_utility'])>1e-12):
            raise ValueError('blocked: saved retained worst-cost F is inconsistent')
        positive=sum(bool(e['feasible']) and e['F']>0 for e in epochs)
        margin=sum(bool(e['feasible']) and e['F']>1e-8 for e in epochs)
        old=next(item for item in prior['fits'] if item['seed']==seed)
        if (old['F']!=info['F'] or old['selected_epoch']!=info['selected_epoch'] or old['epochs']!=len(epochs)
                or old['feasible_positive_F_states']!=positive
                or old['maximum_trajectory_F']!=max(e['F'] for e in epochs)):
            raise ValueError('blocked: retained reproduction financial summary differs from original probe')
        healthy=info['failure'] is None
        passed=healthy and F>1e-8 and all(risk_pass)
        rows.append({'seed':seed,'ordinal_rows':n,'saved_states':len(epochs),'healthy':healthy,
            'selected_epoch':info['selected_epoch'],'retained_F':F,'four_tail_constraints_pass':all(risk_pass),
            'null_selected':info['null_selected'],'feasible_positive_F_states':positive,
            'feasible_F_above_numerical_margin_states':margin,'maximum_trajectory_F':max(e['F'] for e in epochs),
            'registered_positive_control_pass':passed})
    if all(row['registered_positive_control_pass'] for row in rows):
        raise ValueError('blocked: failed-preflight diagnostic refuses a passing probe')
    witness=prior['witness']
    return pd.DataFrame(rows),{'fixture_rows':n,'chronology':'ordinal synthetic rows0..503; no market calendar dates',
        'features_sha256':feature_hash,'returns_sha256':return_hash,
        'registered_positive_control_pass':False,'market_data_used_for_toy':False,
        'retention_reproduction_source_sha256':probe['source_sha256'],
        'initial_failed_probe_source_sha256':receipt['initial_failed_probe_source_sha256'],
        'financial_summary_matches_initial':True,'full_traces_belong_to_retention_reproduction':True,
        'witness_from_original_summary':witness,'witness_newly_evaluated':False,
        'F_units':'63 times mean net log excess on synthetic ordinal steps; not observed financial performance'}


def record_preflight_diagnostic(root,card=None,*,probe_path=None,expected_probe_sha256=None,source_receipt_path=None):
    """Archive a FAILED synthetic gate after native source verification only.

    This entry never invokes fit_folds/run_experiment or creates a passing
    initial seal. A FINISHED Recorder means the diagnostic evidence was saved.
    The archived original bytes permit rerunning without temporary files.
    """
    from qlib.workflow import R
    root=Path(root).resolve();card=Path(card) if card else root/'research'/EXPERIMENT
    plan=json.loads((card/'selection_plan.json').read_text());plan_hash=sha(card/'selection_plan.json')
    hashes=_source_hashes(root)
    bundle=load_sparse_inputs(root,plan)  # Native readback before writable R.
    reference_path=card/'preflight_probe_reference.json';reference=None
    source_files={}
    if probe_path is None:
        reference=json.loads(reference_path.read_text())
        if reference['plan_sha256']!=plan_hash:
            raise ValueError('blocked: archived preflight probe belongs to another registration')
        with R.uri_context(reference['recorder_uri']):
            record=R.get_recorder(experiment_name=EXPERIMENT,recorder_id=reference['recorder_id'])
            if record.client.get_run(record.id).info.status!='FINISHED':
                raise ValueError('blocked: archived preflight diagnostic is incomplete')
            base=(Path(record.get_local_dir())/'artifacts').resolve()
            for name,identity in reference['artifacts'].items():
                path=(base/identity['artifact']).resolve()
                if not path.is_relative_to(base) or sha(path)!=identity['sha256']:
                    raise ValueError('blocked: archived preflight bytes differ')
                source_files[name]=path.read_bytes()
        expected_probe_sha256=reference['probe_sha256'] if expected_probe_sha256 is None else expected_probe_sha256
    else:
        if expected_probe_sha256 is None or source_receipt_path is None:
            raise ValueError('blocked: first preflight archive requires exact probe SHA and source receipt')
        source_files['original_payload.pkl']=Path(probe_path).read_bytes()
        source_files['identity_receipt.json']=Path(source_receipt_path).read_bytes()
        receipt=json.loads(source_files['identity_receipt.json'])
        source_files['initial_model.py']=Path(receipt['initial_source_path']).read_bytes()
        source_files['retention_model.py']=Path(receipt['retention_source_path']).read_bytes()
        source_files['initial_probe_summary.json']=Path(receipt['initial_probe_summary_path']).read_bytes()
    probe_bytes=source_files['original_payload.pkl'];probe_hash=hashlib.sha256(probe_bytes).hexdigest()
    if probe_hash!=expected_probe_sha256:
        raise ValueError('blocked: synthetic probe bytes differ before deserialization')
    receipt=json.loads(source_files['identity_receipt.json'])
    if receipt['retention_full_payload_sha256']!=probe_hash:
        raise ValueError('blocked: probe receipt points to different payload bytes')
    probe=pickle.loads(probe_bytes)
    table,audit=_preflight_probe_audit(probe,plan,receipt,source_files)
    if hashes['src/model/sparse_holdings_net_excess.py']!=audit['retention_reproduction_source_sha256']:
        raise ValueError('blocked: current sparse model differs from retained diagnostic source')
    seal_path=card/'initial_seal.json';seal_before=sha(seal_path) if seal_path.exists() else None
    uri=CANONICAL_ROOT/'research'/EXPERIMENT/'qlib_recorder';uri.mkdir(parents=True,exist_ok=True)
    identities={}
    with R.start(experiment_name=EXPERIMENT,uri=str(uri)):
        record=R.get_recorder()
        R.set_tags(role='failed_registered_synthetic_preflight',market_model_fits_started=False,
                   historical_qualification_pass=False,objective_achieved=False)
        R.log_params(plan_sha256=plan_hash,probe_sha256=probe_hash,chronology='synthetic_ordinal',
                     market_fits=0,market_backtests=0)
        record.log_artifact(str(card/'selection_plan.json'),artifact_path='inputs')
        with TemporaryDirectory(prefix='sparse-prefit-archive-') as temporary:
            for name,contents in source_files.items():
                path=Path(temporary)/name;path.write_bytes(contents)
                record.log_artifact(str(path),artifact_path='preflight')
                identities[name]={'artifact':'preflight/'+name,'sha256':hashlib.sha256(contents).hexdigest()}
        R.save_objects(artifact_path='inputs',**{'native_frozen_source_audit.pkl':bundle['source_audit'],
            'native_frozen_source_details.pkl':bundle['source_details'],'observer_source_hashes.pkl':hashes})
        R.save_objects(artifact_path='analysis',**{'synthetic_seed_summary.pkl':table,'preflight_audit.pkl':audit})
        if _source_hashes(root)!=hashes or sha(card/'selection_plan.json')!=plan_hash:
            raise ValueError('blocked: diagnostic observer source/registration changed')
        base=Path(record.get_local_dir())/'artifacts'
        if any(sha(base/identity['artifact'])!=identity['sha256'] for identity in identities.values()):
            raise ValueError('blocked: recorded original diagnostic bytes differ')
        recdir=Path(record.get_local_dir())
    if (sha(seal_path) if seal_path.exists() else None)!=seal_before:
        raise ValueError('blocked: preflight diagnostic changed the market-start seal')
    if reference is None:
        reference={'experiment':EXPERIMENT,'recorder_id':record.id,'recorder_uri':str(uri),
            'plan_sha256':plan_hash,'probe_sha256':probe_hash,'artifacts':identities}
        json_write(reference_path,reference)
    summary={'experiment':EXPERIMENT,'campaign_status':'active_research','objective_achieved':False,
        'record_type':'failed_registered_synthetic_preflight_diagnostic',
        'decision':'aborted_before_market_fitting_failed_registered_positive_control',
        'recorder_id':record.id,'recorder_uri':str(uri),'recorder_local_dir':str(recdir),
        'recorder_relative_dir':str(recdir.relative_to(CANONICAL_ROOT/'research'/EXPERIMENT)),
        'recorder_completion_scope':'FINISHED records successful evidence archival only; synthetic gate FAILED.',
        'market_model_fits_started':False,'market_model_fits':0,'market_backtests':0,
        'historical_qualification_pass':False,'new_forward_qualified':False,'selected':None,
        'source':plan['source'],'source_audit':bundle['source_audit'],'source_sha256':hashes,
        'source_coverage':'Current archival observer project footprint; retained model source version is separately archived in preflight evidence.',
        'plan_sha256':plan_hash,'probe_sha256':probe_hash,'preflight':audit,
        'provider_uri':plan['source']['provider_uri'],'region':'us','market':'all',
        'sidecars':['preflight_probe_reference.json','preflight_seed_summary.csv'],
        'production_change':False,'real_broker_orders':False,'qlib_docs_reference':plan['qlib_docs'],
        'artifact_policy':{'policy':'slim_card_recorder_first','workflow_config':'workflow.yaml',
            'recorder_only_artifacts':['preflight/original_payload.pkl','preflight/*.py','preflight/*.json',
                'inputs/*.pkl','analysis/*.pkl']},
        'limitations':['Synthetic ordinal rows are not market dates or realized investment evidence.',
            'Full retained traces belong to the reproduction source version; the initial probe source and summary remain separate.',
            'A failed positive control does not prove the policy class has no edge; no market models were fitted.']}
    table.to_csv(card/'preflight_seed_summary.csv',index=False)
    json_write(card/'summary.json',summary);json_write(card/'selected_model.json',{'selected':None,'reason':summary['decision']})
    import yaml
    (card/'workflow.yaml').write_text(yaml.safe_dump({'experiment':EXPERIMENT,'provider_uri':plan['source']['provider_uri'],
        'region':'us','market':'all','recorder_uri':str(uri),'workflow_mode':'synthetic_preflight_diagnostic_only',
        'market_fits':0,'market_backtests':0,'toy_chronology':'ordinal','artifact_policy':summary['artifact_policy']},sort_keys=False))
    return summary


def _require_initial_seal(root, card, plan_hash, hashes):
    """Fail before any writable Recorder when the pre-fit proof is incomplete."""
    path=Path(card)/'initial_seal.json'
    if not path.is_file():
        raise ValueError('blocked: initial source/test seal is required before market fitting')
    seal=json.loads(path.read_text())
    flags=('synthetic_known_information_all_seeds_and_ensemble_passed',
           'pre_fit_contracts_passed','source_state_derivative_audit_passed')
    count=seal.get('focused_passed')
    if (seal.get('plan_sha256')!=plan_hash or seal.get('source_sha256')!=hashes
            or not isinstance(count,int) or isinstance(count,bool) or count<=0
            or any(seal.get(flag) is not True for flag in flags)
            or seal.get('market_model_fits_started') is not False
            or seal.get('objective_achieved') is not False):
        raise ValueError('blocked: initial seal identity or pre-fit evidence differs')
    stamp=pd.Timestamp(seal.get('sealed_at_utc'))
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise ValueError('blocked: initial seal needs an explicit UTC timestamp')
    tests=seal.get('test_sha256')
    required={'tests/test_sparse_holdings_net_excess_data.py','tests/test_sparse_holdings_net_excess_workflow.py',
              'tests/test_sparse_holdings_net_excess.py','tests/test_post_fee_settlement.py'}
    if not isinstance(tests,dict) or not required.issubset(tests):
        raise ValueError('blocked: initial seal omits focused data/workflow tests')
    for name,digest in tests.items():
        relative=Path(name)
        if (relative.is_absolute() or '..' in relative.parts or relative.parts[0]!='tests'
                or sha(Path(root)/relative)!=digest):
            raise ValueError('blocked: sealed test bytes changed')
    return {**seal,'file_sha256':sha(path)}


def _boolean(value, name):
    if not isinstance(value,(bool,np.bool_)):
        raise ValueError(f'blocked: {name} must be an explicit boolean')
    return bool(value)


def replay_models(frame, annual, *, variant, cost_bps, plan, seed=None, frozen_year=None):
    """State-dependent gates see this book, including across annual refits."""
    from src.model.sparse_holdings_net_excess import policy_from_dict, numpy_rollout
    mapping=variant_mapping(plan)
    if variant not in mapping:
        raise ValueError('blocked: unknown sparse replay family')
    market=_dates(frame); dates=pd.DatetimeIndex(market.signal_date)
    current=np.array([0.,0.,1.]);parts=[]
    for year in sorted(dates.year.unique()):
        use_year=int(frozen_year or year)
        if use_year not in annual:
            raise ValueError('blocked: missing sparse annual policy')
        record=annual[use_year];positions=np.flatnonzero(dates.year==year);rows=frame.iloc[positions]
        cutoff=pd.Timestamp(f'{use_year}-01-01',tz='America/New_York')
        scaler=record['scaler']
        if (record.get('variant')!=variant or record.get('parent_market_variant')!=mapping[variant]
                or record.get('model_year')!=use_year or scaler.get('variant')!=mapping[variant]
                or scaler.get('model_year')!=use_year or record.get('scaler_sha256')!=scaler.get('scaler_sha256')
                or pd.Timestamp(scaler['cutoff'])!=cutoff
                or cutoff>pd.Timestamp(rows.signal_date.min()).tz_localize('America/New_York')):
            raise ValueError('blocked: sparse bundle/scaler identity or availability differs')
        transformed=transform_for_variant(rows,scaler,variant=variant,plan=plan)
        invalid=np.asarray(transformed['invalid_forecast_mask'])
        if invalid.dtype!=bool or invalid.shape!=(len(rows),):
            raise ValueError('blocked: forecast fallback mask is not a complete boolean vector')
        if variant!='price_vol_holdings' and invalid.any():
            raise ValueError('blocked: unused forecasts changed a sparse control')
        stored=record['ensemble_models'] if seed is None else [record['seed_models'][seed]]
        models=[policy_from_dict(payload) for payload in stored]
        result=numpy_rollout(models,transformed['features'],rows[['soxl_return','soxx_return']].to_numpy(float),
            cost_bps=cost_bps,initial_weights=current,force_soxx_mask=invalid)
        q=np.asarray(result['expert_weights']);hold=np.asarray(result['exact_hold'])
        if (q.shape!=(len(rows),4) or not np.isfinite(q).all() or (q<0).any()
                or not np.allclose(q.sum(axis=1),1,atol=1e-12,rtol=0)
                or hold.dtype!=bool or hold.shape!=(len(rows),)
                or not np.array_equal(hold,(q[:,3]==1)&(np.count_nonzero(q,axis=1)==1))):
            raise ValueError('blocked: sparse expert/HOLD evidence is inconsistent')
        if invalid.any() and not np.array_equal(q[invalid],np.tile([0.,1.,0.,0.],(int(invalid.sum()),1))):
            raise ValueError('blocked: current invalid forecast did not force exact SOXX')
        if (hold.any() and (np.any(result['fees'][hold]!=0) or np.any(result['turnover'][hold]!=0)
                or not np.array_equal(result['target_weights'][hold],result['pretrade_weights'][hold]))):
            raise ValueError('blocked: exact HOLD placed an order or changed its target')
        part=market.iloc[positions][['signal_date','trade_date','exit_date']].copy()
        part['strategy_return']=result['net_equity']-1
        part['fees']=result['fees'];part['turnover']=result['turnover'];part['cost_bps']=cost_bps
        part['execute_open_target']=~hold;part['execute_open_trade']=result['turnover']>1e-12
        part['exact_zero_order']=result['turnover']==0;part['exact_hold_support']=hold
        part['expert_support_size']=np.count_nonzero(q,axis=1)
        for number,expert in enumerate(('soxl','soxx','cash','hold')):part[f'expert_{expert}_weight']=q[:,number]
        part['observed_forecast_fallback']=invalid
        part['invalid_training_fallback']=_boolean(record['invalid_training'],'invalid training flag')
        part['numerical_failure']=_boolean(record['numerical_failure'],'numerical failure flag')
        part['model_year']=use_year;part['parent_market_variant']=mapping[variant]
        part['ensemble_anchor_selected']=_boolean(record['ensemble_audit']['anchor_selected'],'ensemble anchor flag') if seed is None else False
        for number,asset in enumerate(ASSETS):
            part[f'{asset}_weight']=result['target_weights'][:,number]
            part[f'pretrade_{asset}_weight']=result['pretrade_weights'][:,number]
            part[f'ending_{asset}_weight']=result['ending_weights'][:,number]
        current=np.asarray(result['ending_weights'][-1]).copy();parts.append(part)
    path=pd.concat(parts,ignore_index=True);path['equity']=np.cumprod(1+path.strategy_return.to_numpy())
    return validate_replay(path)


def independent_path_audit(frame,path,*,cost_bps):
    """Independent frozen NumPy settlement, explicitly exercising true HOLD."""
    from src.backtest.position_accounting import settle_target_session
    from src.workflow.stage1_economic_action import fee_affordable_target
    market=_dates(frame);path=validate_replay(path)
    if not market[['signal_date','trade_date','exit_date']].equals(path[['signal_date','trade_date','exit_date']]):
        raise ValueError('blocked: sparse independent audit dates differ')
    held=np.array([0.,0.,1.]);errors={key:0. for key in ('pretrade','ending','return','fee','turnover')}
    for i,row in path.iterrows():
        target=row[[f'{asset}_weight' for asset in ASSETS]].to_numpy(float)
        pre=row[[f'pretrade_{asset}_weight' for asset in ASSETS]].to_numpy(float)
        execute=_boolean(row.execute_open_target,'execution schedule')
        if not execute and not np.array_equal(target,pre):
            raise ValueError('blocked: HOLD schedule differs from owned pretrade holdings')
        execution=fee_affordable_target(target,pretrade_weights=held,cost_bps=cost_bps) if execute else None
        settled=settle_target_session(held,open_target_weights=execution,
            asset_returns=[market.soxl_return.iloc[i],market.soxx_return.iloc[i],0.],cost_bps=cost_bps)
        errors['pretrade']=max(errors['pretrade'],float(np.max(abs(pre-held))))
        errors['ending']=max(errors['ending'],float(np.max(abs(row[[f'ending_{asset}_weight' for asset in ASSETS]].to_numpy(float)-settled['ending_weights']))))
        for key,field in [('return','net_return'),('fee','fees'),('turnover','turnover')]:
            observed=row.strategy_return if key=='return' else row[field]
            errors[key]=max(errors[key],abs(float(observed)-settled[field]))
        held=settled['ending_weights']
    if max(errors.values())>1e-10:
        raise ValueError(f'blocked: sparse independent accounting differs {errors}')
    return {'sessions':len(path),'cost_bps':cost_bps,'maximum_absolute_errors':errors,'passed':True,
            'exact_hold_sessions':int((~path.execute_open_target).sum())}


def execution_diagnostics(replays, market_frames):
    """Describe saved books only; no alternate actions or zero-cost rollout."""
    support_rows=[];fee_rows=[];equality_rows=[]
    for (segment,variant,cost,seed),part in replays[replays.seed.ne('benchmark')].groupby(
            ['segment','variant','cost_bps','seed']):
        part=part.sort_values('signal_date').reset_index(drop=True)
        market=market_frames[segment].loc[pd.DatetimeIndex(part.signal_date)]
        benchmark=replays[replays.segment.eq(segment)&replays.cost_bps.eq(cost)&replays.seed.eq('benchmark')].sort_values('signal_date').reset_index(drop=True)
        if not part[['signal_date','trade_date','exit_date']].equals(benchmark[['signal_date','trade_date','exit_date']]):
            raise ValueError('blocked: diagnostic benchmark dates differ')
        q=part[[f'expert_{a}_weight' for a in ('soxl','soxx','cash','hold')]].to_numpy(float)
        size=np.count_nonzero(q,axis=1)
        support_rows.append({'segment':segment,'variant':variant,'cost_bps':cost,'seed':seed,
            'sessions':len(part),'exact_zero_orders':int(part.exact_zero_order.sum()),
            'reported_trade_days':int(part.execute_open_trade.sum()),'turnover':float(part.turnover.sum()),
            'exact_hold_support_days':int(part.exact_hold_support.sum()),
            'forecast_forced_soxx_days':int(part.observed_forecast_fallback.sum()),
            'support_changes':int(np.any((q[1:]>0)!=(q[:-1]>0),axis=1).sum()),
            **{f'support_{k}_fraction':float(np.mean(size==k)) for k in (1,2,3,4)},
            **{f'sole_{a}_fraction':float(np.mean((size==1)&(q[:,j]==1)))
               for j,a in enumerate(('soxl','soxx','cash','hold'))},
            **{f'average_{a}_target':float(part[f'{a}_weight'].mean()) for a in ASSETS}})
        gross=1+(part[['soxl_weight','soxx_weight']].to_numpy(float)*market[['soxl_return','soxx_return']].to_numpy(float)).sum(1)
        bgross=1+market.soxx_return.to_numpy(float)
        if not np.allclose(1+part.strategy_return,(1-part.fees)*gross,atol=1e-12,rtol=0):
            raise ValueError('blocked: saved sparse fees do not reconstruct net returns')
        if not np.allclose(1+benchmark.strategy_return,(1-benchmark.fees)*bgross,atol=1e-12,rtol=0):
            raise ValueError('blocked: saved SOXX fees do not reconstruct net returns')
        g=np.log(gross)-np.log(bgross)
        f=np.log1p(-part.fees.to_numpy())-np.log1p(-benchmark.fees.to_numpy())
        n=np.log1p(part.strategy_return.to_numpy())-np.log1p(benchmark.strategy_return.to_numpy())
        if np.max(abs(g+f-n))>1e-12:
            raise ValueError('blocked: saved-path gross/fee decomposition differs')
        for year in ['all',*sorted(part.signal_date.dt.year.unique())]:
            mask=np.ones(len(part),bool) if year=='all' else part.signal_date.dt.year.eq(year).to_numpy()
            fee_rows.append({'segment':segment,'variant':variant,'cost_bps':cost,'seed':seed,'signal_year':year,
                'sessions':int(mask.sum()),'gross_position_log_excess':float(g[mask].sum()),
                'fee_log_excess':float(f[mask].sum()),'net_log_excess':float(n[mask].sum()),
                'scope':'exact identity on saved holdings; not an alternative zero-cost strategy'})
    for (segment,variant,cost),part in replays[replays.seed.isin([str(v) for v in SEEDS])].groupby(
            ['segment','variant','cost_bps']):
        names=[f'{a}_weight' for a in ASSETS]+['strategy_return','fees','turnover']
        reference=part[part.seed.eq(str(SEEDS[0]))].sort_values('signal_date')
        for seed in SEEDS[1:]:
            other=part[part.seed.eq(str(seed))].sort_values('signal_date')
            if not np.array_equal(reference.signal_date.to_numpy(),other.signal_date.to_numpy()):
                raise ValueError('blocked: registered seed coverage differs')
            left,right=reference[names].to_numpy(float),other[names].to_numpy(float)
            equality_rows.append({'segment':segment,'variant':variant,'cost_bps':cost,'reference_seed':SEEDS[0],
                'seed':seed,'exact_path_equality':bool(np.array_equal(left,right)),
                'maximum_absolute_difference':float(np.max(abs(left-right))),
                'interpretation':'optimization-start symmetry/dispersion, not independent market samples'})
    return pd.DataFrame(support_rows),pd.DataFrame(fee_rows),pd.DataFrame(equality_rows)

def training_spec(plan):
    from src.model.sparse_holdings_net_excess import ConstrainedNetExcessSpec
    p = plan['training']
    if (tuple(p['seed_ids']) != SEEDS or plan['architecture']['feature_count'] != 27
            or plan['architecture']['hidden_units'] != 8 or plan['architecture']['parameter_count'] != 272
            or plan['evaluation']['family_size_adjustment'] != 23):
        raise ValueError('blocked: implementation architecture or seeds differ from plan')
    return ConstrainedNetExcessSpec(updates=p['updates'], learning_rate=p['learning_rate'],
        gradient_clip=p['gradient_clip'], risk_group_length=p['risk_group_length'],
        gradient_chunk_length=p['gradient_scored_chunk'], context_sessions=p['gradient_prior_context'],
        cost_bps=tuple(p['cost_bps']), minimum_rows=p['minimum_rows'], torch_threads=p['torch_threads'],
        downside_coefficient=p['downside_coefficient'], rms_epsilon=p['rms_epsilon'],
        dual_learning_rate=p['dual_learning_rate'],risk_tolerance_percentage_points=plan['constraints']['risk_tolerance_percentage_points'])


def _fit_job(payload):
    from src.model.sparse_holdings_net_excess import fit_policy
    from src.model.sparse_holdings_net_excess import policy_to_dict
    variant, year, seed, x, returns, spec = payload
    start = time.monotonic()
    model, metadata = fit_policy(x, returns, seed=seed, spec=spec)
    metadata['elapsed_seconds'] = time.monotonic() - start
    return variant, year, seed, policy_to_dict(model), metadata


def _validate_retained(payloads,metadata,plan):
    """Separate feasible economic incumbents from failed accounting-only nulls."""
    from src.model.sparse_holdings_net_excess import policy_from_dict, is_exact_anchor
    costs=tuple(plan['training']['cost_bps'])
    def scalar(value):
        return isinstance(value,(int,float,np.integer,np.floating)) and not isinstance(value,(bool,np.bool_)) and bool(np.isfinite(value))
    flag=metadata.get('training_feasibility_passed')
    if not isinstance(flag,(bool,np.bool_)) or not bool(flag):
        raise ValueError('blocked: retained training feasibility needs a true boolean')
    F=metadata.get('centered_worst_utility');parts=metadata.get('centered_utility_by_cost')
    if not scalar(F) or F<0 or not isinstance(parts,dict) or set(parts)!=set(costs) or any(not scalar(v) for v in parts.values()) or F!=min(parts.values()):
        raise ValueError('blocked: retained mean-excess evidence is nonfinite, negative or inconsistent')
    policies=[policy_from_dict(payload) for payload in payloads]
    failure=metadata.get('failure')
    if failure is None:
        if metadata.get('qualification_eligible') is not True:
            raise ValueError('blocked: healthy retained fit lacks qualification integrity')
    else:
        if (not isinstance(failure,dict) or failure.get('accounting_anchor_only') is not True
                or metadata.get('qualification_eligible') is not False or F!=0 or any(v!=0 for v in parts.values())
                or any(not is_exact_anchor(policy) for policy in policies)):
            raise ValueError('blocked: numerical failure did not preserve an explicit exact accounting null')
    risks=metadata.get('selected_risks_by_cost')
    if not isinstance(risks,dict) or set(risks)!=set(costs):raise ValueError('blocked: missing retained raw risk evidence')
    for risk in risks.values():
        keys=('max_drawdown_pct','cvar5_pct','benchmark_max_drawdown_pct','benchmark_cvar5_pct')
        if any(not scalar(risk.get(k)) for k in keys):raise ValueError('blocked: nonfinite retained risk values')
        gaps=np.asarray(risk.get('raw_percentage_point_gaps'),dtype=float)
        if gaps.shape!=(2,) or not np.isfinite(gaps).all():raise ValueError('blocked: invalid retained risk gaps')
        implied=np.array([risk['benchmark_max_drawdown_pct']-risk['max_drawdown_pct'],risk['benchmark_cvar5_pct']-risk['cvar5_pct']])
        if not np.allclose(gaps,implied,atol=1e-12,rtol=0) or (gaps>plan['constraints']['risk_tolerance_percentage_points']).any():
            raise ValueError('blocked: retained raw tail constraints fail')
        if failure is None:
            normalizers=np.asarray(risk.get('normalizers'),dtype=float)
            expected_reference=-np.array([risk['benchmark_max_drawdown_pct'],risk['benchmark_cvar5_pct']])/100
            if (risk.get('normalizers_valid_for_optimization') is not True or normalizers.shape!=(2,)
                    or not np.isfinite(normalizers).all() or (normalizers<=0).any()
                    or not np.allclose(normalizers,expected_reference,atol=1e-14,rtol=0)):
                raise ValueError('blocked: healthy retained fit has invalid risk normalizers')
    return policies


def fit_folds(folds, plan, recorder, *, workers=3):
    from src.model.sparse_holdings_net_excess import (SparseHoldingsPolicy, policy_to_dict,
        policy_from_dict)
    from src.model.sparse_holdings_net_excess import evaluate_ensemble
    spec = training_spec(plan)
    fitted = {}; diagnostics = []; ensemble_audits = []
    jobs = []
    for (name, year), fold in folds.items():
        fitted.setdefault(name, {})[year] = {'scaler':fold['scaler'], 'seed_models':{},
            'variant':name,'parent_market_variant':fold['parent_market_variant'],'model_year':year,'scaler_sha256':fold['scaler']['scaler_sha256'],
            'training_feature_sha256':hashlib.sha256(np.asarray(fold['features'],dtype=np.float64).tobytes()).hexdigest(),
            'training_return_sha256':hashlib.sha256(np.asarray(fold['returns'],dtype=np.float64).tobytes()).hexdigest(),
            'invalid_training':fold['invalid_training'], 'invalid_training_dates':fold['invalid_training_dates']}
        for seed in SEEDS:
            if fold['invalid_training']:
                null = SparseHoldingsPolicy(seed=seed,anchor=True)
                payload=policy_to_dict(null)
                fitted[name][year]['seed_models'][seed] = payload
                source_failure={'variant':name,'year':year,'seed':seed,'failure':'invalid_available_training_volatility',
                    'qualification_eligible':False,'accounting_anchor_only':True,'selected_epoch':'anchor',
                    'centered_worst_utility':0.,'centered_utility_by_cost':{c:0. for c in plan['training']['cost_bps']},
                    'null_selected':True,'epochs':[],'invalid_training_dates':fold['invalid_training_dates']}
                diagnostics.append(source_failure)
                recorder.save_objects(artifact_path=f'fit_checkpoints/{name}/{year}/{seed}',
                    **{'model.pkl':payload,'training_audit.pkl':source_failure})
            else:
                jobs.append((name,year,seed,fold['features'],fold['returns'],spec))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending = [pool.submit(_fit_job, job) for job in jobs]
        for future in as_completed(pending):
            name,year,seed,model,metadata = future.result()
            for field in ('training_feature_sha256','training_return_sha256'):
                if metadata[field] != fitted[name][year][field]:
                    raise ValueError('blocked: fitted worker consumed different mature inputs')
            _validate_retained([model],metadata,plan)
            fitted[name][year]['seed_models'][seed] = model
            diagnostic = {'variant':name,'year':year,'seed':seed,**metadata}
            diagnostics.append(diagnostic)
            recorder.save_objects(artifact_path=f'fit_checkpoints/{name}/{year}/{seed}',
                **{'model.pkl':model,'training_audit.pkl':diagnostic})
            print(f'Sparse holdings {name} year={year} seed={seed} epoch={metadata["selected_epoch"]} F={metadata["centered_worst_utility"]:.8g} seconds={metadata["elapsed_seconds"]:.1f}',flush=True)
    for (name,year),fold in folds.items():
        record = fitted[name][year]
        if set(record['seed_models']) != set(SEEDS):
            raise ValueError('blocked: missing registered seed fit')
        models = [policy_from_dict(record['seed_models'][seed]) for seed in SEEDS]
        accepted,audit = evaluate_ensemble(models,fold['features'],fold['returns'],spec=spec)
        _validate_retained([policy_to_dict(model) for model in accepted],audit,plan)
        record['ensemble_models'] = [policy_to_dict(model) for model in accepted]
        record['ensemble_audit'] = {k:v for k,v in audit.items() if k not in ('training_paths','rejected_ensemble_training_paths')}
        record['ensemble_audit']['recorder_artifact']=f'fit_checkpoints/{name}/{year}/ensemble/training_audit.pkl'
        record['numerical_failure'] = bool(audit.get('failure') is not None or any(
            d.get('failure') is not None and d.get('failure')!='invalid_available_training_volatility'
            for d in diagnostics if d['variant']==name and d['year']==year))
        ensemble_audits.append({'variant':name,'year':year,**audit})
        recorder.save_objects(artifact_path=f'fit_checkpoints/{name}/{year}/ensemble',
            **{'model_bundle.pkl':record,'training_audit.pkl':audit})
    return fitted,diagnostics,ensemble_audits


def _training_index(diagnostics):
    fields=('variant','year','seed','selected_epoch','failure','qualification_eligible',
        'training_feasibility_passed','centered_worst_utility','centered_utility_by_cost','legacy_J',
        'null_selected','training_feature_sha256','training_return_sha256','optimizer_diagnostics')
    return [{**{k:a.get(k) for k in fields},
             'recorder_artifact':f"fit_checkpoints/{a['variant']}/{a['year']}/{a['seed']}/training_audit.pkl"}
            for a in diagnostics]


def _ensemble_index(audits):
    omitted=('training_paths','rejected_ensemble_training_paths')
    return [{**{k:v for k,v in a.items() if k not in omitted},
             'recorder_artifact':f"fit_checkpoints/{a['variant']}/{a['year']}/ensemble/training_audit.pkl"}
            for a in audits]


def _source_hashes(root):
    """Explicit project dependency footprint, not a third-party closure hash."""
    sources=set(_parent_source_hashes(root)) | {
        'src/data/sparse_holdings_net_excess.py','src/workflow/sparse_holdings_net_excess.py',
        'src/model/sparse_holdings_net_excess.py','src/model/post_fee_settlement.py',
        'src/workflow/trainable_regime_policy_nested_wf.py',
        'src/backtest/strategy.py','src/analysis/stage1_regime_validation.py',
        'src/core/__init__.py','src/core/contracts.py',
    }
    return {path:sha(root/path) for path in sorted(sources)}


def _finalize_native(recorder_id,uri,build_reports):
    """Native backtest may reinitialize Qlib; run it outside an active Recorder.

    Explicitly resume the original identity/URI to persist the reports. A native
    failure is also propagated inside that resumed context, retaining FAILED.
    """
    from qlib.workflow import R
    try:
        reports=build_reports()
    except Exception:
        with R.start(experiment_name=EXPERIMENT,recorder_id=recorder_id,uri=uri,resume=True):
            raise
    with R.start(experiment_name=EXPERIMENT,recorder_id=recorder_id,uri=uri,resume=True):
        if R.get_recorder().id!=recorder_id:
            raise ValueError('blocked: native reports resumed a different Recorder')
        R.save_objects(artifact_path='portfolio_analysis',**{'native_reconciliation.pkl':reports})
    return reports


def run_experiment(root,card=None,*,workers=3):
    import qlib
    from qlib.workflow import R
    from src.workflow.conditional_variance import hac_attribution
    from src.data.integrated_allocation import COMMON_PRIMARY_FEATURES
    from src.analysis.convex_qlib import native_open_reconciliation as precise_native
    from src.workflow.trainable_regime_policy_nested_wf import _remove_qlib_code_snapshot_artifacts
    root=Path(root).resolve();card=Path(card) if card else root/'research'/EXPERIMENT
    plan=json.loads((card/'selection_plan.json').read_text());plan_hash=sha(card/'selection_plan.json')
    hashes=_source_hashes(root);source=plan['source'];costs=plan['evaluation']['cost_bps']
    initial_seal=_require_initial_seal(root,card,plan_hash,hashes)
    bundle=load_sparse_inputs(root,plan)
    raw=bundle['raw'];prices=bundle['prices'];forecasts=bundle['forecasts']
    forecast_audit=bundle['forecast_audit'];forecast_models=bundle['forecast_models'];dataset=bundle['dataset']
    uri=CANONICAL_ROOT/'research'/EXPERIMENT/'qlib_recorder';uri.mkdir(parents=True,exist_ok=True)
    R.set_uri(str(uri))
    with R.start(experiment_name=EXPERIMENT):
        recorder=R.get_recorder()
        R.log_params(plan_sha256=plan_hash,source_recorder_id=source['recorder_id'],
            parent_input_recorder_id=plan['parent_input_source']['recorder_id'],objective='mean_net_log_excess_with_benchmark_tail_constraints',campaign_status='active_research')
        recorder.log_artifact(str(card/'selection_plan.json'),artifact_path='inputs')
        recorder.log_artifact(str(card/'initial_seal.json'),artifact_path='inputs')
        R.save_objects(artifact_path='inputs',**{'source_hashes.pkl':hashes,'source_audit.pkl':bundle['source_audit'],
            'primary_frame.pkl':raw,'execution_prices.pkl':prices})
        R.save_objects(artifact_path='inputs',**{'source_details.pkl':bundle['source_details']})
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
        fitted,diagnostics,ensemble_audits=fit_folds(folds,plan,recorder,workers=workers)
        R.save_objects(artifact_path='inputs',**{'forecasts.pkl':forecasts,'forecast_fit_audit.pkl':forecast_audit,
            'development_fold_inputs.pkl':folds})
        R.save_objects(artifact_path='models',**{'forecast_models.pkl':forecast_models,'development.pkl':fitted})
        benchmarks={c:benchmark_path(development,cost_bps=c) for c in costs}
        evidence={};paths={};replays=[];seed_rows=[];path_audits=[]
        for name in variants:
            candidate={}
            for cost in costs:
                path=replay_models(development,fitted[name],variant=name,cost_bps=cost,plan=plan)
                paths[(name,cost)]=path;candidate[cost]=path
                replays.append(path.assign(variant=name,seed='ensemble',segment='development'))
                path_audits.append({'variant':name,'seed':'ensemble',**independent_path_audit(development,path,cost_bps=cost)})
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
        print(f'Sparse holdings development sealed: {selected}',flush=True)
        chosen=selected['variant']
        outer_folds=prepare_folds(dataset,[chosen],plan['training']['diagnostic_signal_years'],plan)
        outer_fitted,outer_diagnostics,outer_ensemble=fit_folds(outer_folds,plan,recorder,workers=workers)
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
                or sha(card/'initial_seal.json')!=initial_seal['file_sha256']
                or _require_initial_seal(root,card,plan_hash,hashes)!=initial_seal):
            raise ValueError('blocked: registered implementation changed during experiment')
        R.save_objects(artifact_path='models',**{'selected_annual_refits.pkl':outer_fitted})
        R.save_objects(artifact_path='inputs',**{'diagnostic_fold_inputs.pkl':outer_folds,'diagnostic_frames.pkl':outer_frames})
        R.save_objects(artifact_path='analysis',**{'portfolio_replays.pkl':all_replays,
            'comparison.pkl':comparison,'annual.pkl':annual,'gates.pkl':gates,'bootstrap.pkl':bootstrap,
            'seed_metrics.pkl':pd.DataFrame(seed_rows),'training_diagnostics.pkl':_training_index(diagnostics),
            'ensemble_audits.pkl':_ensemble_index(ensemble_audits),'independent_path_audits.pkl':path_audits,
            'controls.pkl':pd.DataFrame(controls),'attribution.pkl':pd.DataFrame(attribution),'window_metrics.pkl':pd.DataFrame(windows),
            'execution_sparsity.pkl':sparsity,'saved_path_fee_decomposition.pkl':fee_decomposition,
            'seed_path_equality.pkl':seed_equality})
        R.log_metrics(historical_qualification_pass=float(selected['historical_qualification_pass']),registered_variants=3,seeds_per_variant=3)
        recdir=Path(recorder.get_local_dir())
        pd.testing.assert_frame_equal(comparison,recorder.load_object('analysis/comparison.pkl'))
    native=_finalize_native(recorder.id,str(uri),lambda: {c:native_allocation_reconciliation(
        development.reset_index(drop=True),paths[(chosen,c)],provider_uri=root/source['provider_uri'],
        cost_bps=c,native_reconciler=precise_native) for c in costs})
    _remove_qlib_code_snapshot_artifacts(recdir)
    training_rows=[]
    for audit in diagnostics:
        item={k:audit.get(k) for k in ('variant','year','seed','failure','selected_epoch','centered_worst_utility','legacy_J','training_feasibility_passed','null_selected','elapsed_seconds')}
        epochs=audit.get('epochs',[])
        item['feasible_epochs']=sum(bool(e['feasible']) for e in epochs)
        item['positive_F_feasible_epochs']=sum(bool(e['feasible']) and e['F']>0 for e in epochs)
        item['F_above_numerical_margin_feasible_epochs']=sum(bool(e['feasible']) and e['F']>1e-8 for e in epochs)
        item['last_trajectory_feasible']=bool(epochs[-1]['feasible']) if epochs else None
        item['updates_completed']=audit.get('optimizer_diagnostics',{}).get('updates_completed')
        item['selected_maximum_risk_excess_percentage_points']=max([0.]+[float(gap) for risk in audit.get('selected_risks_by_cost',{}).values() for gap in risk['raw_percentage_point_gaps']]) if audit.get('selected_risks_by_cost') else None
        training_rows.append(item)
    tables={'model_selection.csv':ranking,'backtest_benchmark_comparison.csv':comparison,'annual_metrics.csv':annual,
        'development_gates.csv':gates,'bootstrap_analysis.csv':bootstrap,'seed_metrics.csv':pd.DataFrame(seed_rows),
        'training_summary.csv':pd.DataFrame(training_rows),'forecast_training_audit.csv':forecast_audit,
        'control_comparison.csv':pd.DataFrame(controls),'beta_alpha_attribution.csv':pd.DataFrame(attribution),
        'window_metrics.csv':pd.DataFrame(windows),'execution_sparsity.csv':sparsity,
        'saved_path_fee_decomposition.csv':fee_decomposition,'seed_path_equality.csv':seed_equality}
    for filename,table in tables.items():table.to_csv(card/filename,index=False)
    summary={'experiment':EXPERIMENT,'campaign_status':'active_research','objective_achieved':False,
        'decision':'historical_candidate_requires_source_sensitivity_and_new_forward' if selected['historical_qualification_pass'] else 'continue_research_no_historical_qualification',
        'selected':selected,'historical_qualification_pass':bool(selected['historical_qualification_pass']),
        'new_forward_qualified':False,'source_sensitivity_completed':False,'source':source,'source_audit':bundle['source_audit'],
        'seed_scope':'Optimization-start stability only. holdings_only masks market inputs but retains owned state; deterministic initial symmetry can make its seeds identical. Seeds are never independent market observations.',
        'source_sha256':hashes,'plan_sha256':plan_hash,'initial_seal_sha256':initial_seal['file_sha256'],'provider_uri':source['provider_uri'],'region':'us','market':'all',
        'source_coverage':'Explicit project data/model/workflow/accounting/risk/native-strategy and evidence-helper footprint; not a full transitive or third-party dependency hash.',
        'training_objective':'F=min_c63mean_net_log_excess; legacy_J_report_only',
        'architecture':plan['architecture'],'settlement':plan['settlement'],'parent_variant_mapping':bundle['variant_mapping'],
        'training_constraints':'same_benchmark_anchored_MaxDD_and_exact_CVaR5_at_both_costs',
        'normalized_cost_state_scope':'Post-fee normalized targets are cost-blind; fee funding changes NAV/trade amounts, not necessarily normalized holdings.',
        'recorder_id':recorder.id,'recorder_uri':str(uri),'recorder_local_dir':str(recdir),
        'recorder_relative_dir':str(recdir.relative_to(CANONICAL_ROOT/'research'/EXPERIMENT)),
        'selection_uses_diagnostic_signal_years':False,'development_last_exit':str(development.exit_date.max().date()),
        'native_qlib_audit':{str(c):v['audit'] for c,v in native.items()},'independent_accounting_audit':path_audits,
        'production_change':False,'real_broker_orders':False,'sidecars':list(tables),
        'training_audit_storage':'Complete epoch states/selected paths/optimizer state only in per-fit Recorder artifacts; aggregate analysis files are small indices with exact artifact pointers.',
        'artifact_policy':{'policy':'slim_card_recorder_first','workflow_config':'workflow.yaml',
            'recorder_only_artifacts':['inputs/*.pkl','models/*.pkl','fit_checkpoints/*','analysis/portfolio_replays.pkl','analysis/training_diagnostics.pkl','analysis/ensemble_audits.pkl','portfolio_analysis/*.pkl']},
        'qlib_docs_reference':plan['qlib_docs'],
        'limitations':['Historical years have been reviewed; no fresh forward evidence or guaranteed future excess.',
            'Sparse single-support actions have zero score Jacobian and may stall finite-step optimization.',
            '21-session state-gradient truncation and nonsmooth fee gradients are optimizer approximations; exact incumbent forward audits do not certify optimality.',
            'Retained training models satisfy historical mean-excess and tail constraints; this does not guarantee future wealth/risk or optimizer convergence.',
            'Source basis retains published-versus-server adjustment differences and16secondary observations.',
            'Historical open prices and continuous adjusted shares are not evidence of actual spread, impact or partial fills.']}
    json_write(card/'summary.json',summary);json_write(card/'selected_model.json',{'selected':selected,'research_only':True,'new_forward_qualified':False})
    import yaml
    (card/'workflow.yaml').write_text(yaml.safe_dump({'experiment':EXPERIMENT,'provider_uri':source['provider_uri'],'region':'us',
        'recorder_uri':str(uri),'model_selection':'selection_plan.json','artifact_policy':summary['artifact_policy'],
        'segments':{k:plan['training'][k] for k in ('development_signal_years','diagnostic_signal_years')}},sort_keys=False))
    lines=['# 稀疏专家与真实持仓输入：训练结论','',f'完成3组信息对照、每组3个种子；保留原市场编码器和初始路径，改用稀疏四专家及直接持仓分支，并采用登记的费用后状态图/隐式广义梯度。完整路径计入所有费用，优化F并保留四风险门可行状态；原季度效用仅作诊断。历史资格通过={selected["historical_qualification_pass"]}；稳定净超额目标仍未完成。','',
        '开发期2016–2022，10bp：']
    for row in comparison[comparison.segment.eq('development') & comparison.cost_bps.eq(10)].itertuples():
        lines.append(f'- {row.variant}：收益{row.return_pct:.2f}%，相对SOXX终值{row.relative_wealth:.4f}，MaxDD{row.max_drawdown_pct:.2f}%，CVaR5{row.cvar5_pct:.2f}%。')
    lines+=['','各年差异见annual_metrics.csv，8月3日实际交易日起的持续持仓与独立现金起点分别见window_metrics.csv。三种子不是三条独立市场历史。',
        f'决定：{summary["decision"]}。2023以后为已查看的诊断期，不能作为新留出样本；原SIM未切换。',
        '风险：有限模型容量、截断梯度、零换手边界导数、行情口径和真实成交差异。训练期净超额与尾部可行性不保证样本外超额或风险；对偶迭代没有全局收敛证明。',
        '下一步：按失败门与训练诊断继续定位；若历史门通过，先完成来源敏感性，再冻结进入新的前向证据期。',f'Qlib Recorder：{recorder.id}。']
    (card/'conclusion.md').write_text('\n'.join(lines)+'\n')
    return summary
