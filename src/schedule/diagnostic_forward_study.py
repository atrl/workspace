"""Finite diagnostic event runner and offline-only implementation receipt.

Live transports are not wired while role-specific input mapping is pending.
The event runner executes real state transitions, not an unconditional pilot stub.
"""
from copy import deepcopy
import hashlib,json,os,pickle
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from src.data.diagnostic_forward_inputs import PLAN_SHA,require_phase,network_denied,digest,canonical,utc,validate_daily_bundle,validate_bars,validate_live_open,validate_quote,validate_close_bars
from src.backtest.diagnostic_forward_state import ForwardPaperState,FAMILIES,COSTS
from src.data.recovery_first_hour import aggregate_first_hour,FIELDS
from src.core.recorder_sources import load_verified_recorder_objects
from src.data.minute_intake import atomic_json
NAME='diagnostic_forward_study_20261016'
MODULES=['src/data/diagnostic_forward_inputs.py','src/backtest/diagnostic_forward_state.py','src/schedule/diagnostic_forward_study.py']

def persist_checkpoint(directory,cursor,record,payload):
    """Commit marker is written last; a quote may consume only a committed seal."""
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    event=json.dumps(canonical(record),sort_keys=True,allow_nan=False).encode();state=pickle.dumps(payload,protocol=pickle.HIGHEST_PROTOCOL)
    names={f'event_{cursor:03d}.json':event,f'state_{cursor:03d}.pkl':state}
    hashes={name:hashlib.sha256(body).hexdigest() for name,body in names.items()}
    names[f'commit_{cursor:03d}.json']=json.dumps({'cursor':cursor,'files':hashes,'plan_sha256':PLAN_SHA},sort_keys=True).encode()
    for name,body in names.items():
        path=directory/name
        if path.exists():
            if path.read_bytes()!=body:raise ValueError('conflicting durable checkpoint')
            continue
        with path.open('xb') as stream:stream.write(body);stream.flush();os.fsync(stream.fileno())
    descriptor=os.open(directory,os.O_RDONLY)
    try:os.fsync(descriptor)
    finally:os.close(descriptor)
    return hashes

def load_checkpoint(directory):
    directory=Path(directory);commits=sorted(directory.glob('commit_*.json'))
    if not commits:return None
    meta=json.loads(commits[-1].read_text())
    if meta['plan_sha256']!=PLAN_SHA:raise ValueError('checkpoint plan changed')
    for name,expected in meta['files'].items():
        if hashlib.sha256((directory/name).read_bytes()).hexdigest()!=expected:raise ValueError('checkpoint byte mismatch')
    return pickle.loads((directory/f"state_{meta['cursor']:03d}.pkl").read_bytes())

def registered_slots(session):
    import exchange_calendars as xc
    day=pd.Timestamp(session).normalize().tz_localize(None)
    calendar=xc.get_calendar('XNYS',start=day-pd.Timedelta(days=10),end=day+pd.Timedelta(days=10))
    dates=pd.DatetimeIndex(calendar.sessions).tz_localize(None);position=dates.get_loc(day)
    previous=dates[position-1];terminal=dates[position+1]
    at=lambda date,h,m:date.tz_localize('America/New_York')+pd.Timedelta(hours=h,minutes=m)
    return {'session':str(day.date()),'previous_session':str(previous.date()),'terminal_session':str(terminal.date()),'known_cutoff':at(day,9,45),'open_fill':at(day,9,47),'recovery_cutoff':at(day,10,30),'recovery_fill':at(day,10,32),'close':calendar.session_close(day).tz_convert('America/New_York'),'terminal_fill':at(terminal,9,47)}

def assert_fixed_time(value,expected):
    if utc(value)!=utc(expected):raise ValueError('event differs from registered fixed slot')

class FiniteStudyRunner:
    SLOTS=('daily','known','open_intent','open_quote','recovery_data','recovery_intent','recovery_quote','close','terminal')
    def __init__(self,card,*,session,previous_session,expected_dates,tariff,recovery_spec,action_spec,controller,selection,role_approval,bar_mapping,clock_now):
        self.permit=require_phase(card,'pilot',at=clock_now)
        if self.permit.get('session')!=session:raise PermissionError('pilot session differs from permit')
        self.journal_dir=Path(card).resolve().parents[1]/'runtime/research'/NAME/self.permit['registration_sha256'][:16]
        if role_approval.get('status')!='APPROVED_ROLE_MAPPING':raise ValueError('source mapping remains pending')
        self.slots=registered_slots(session)
        if previous_session!=self.slots['previous_session']:raise ValueError('previous date is not prior XNYS session')
        self.session=session;self.previous_session=previous_session;self.expected_dates=expected_dates;self.role_approval=role_approval;self.bar_mapping=bar_mapping
        self.state=ForwardPaperState(tariff);self.spec=recovery_spec;self.action_spec=action_spec;self.controller=controller;self.selection=selection
        self.cursor=0;self.journal=[];self.rows=None;self.bundle_hash=None;self.pending_targets=None;self.recovery_calls=0;self.durable_intents=set()
        saved=load_checkpoint(self.journal_dir)
        if saved is not None:
            for key,value in saved.items():setattr(self,key,value)
    def dispatch(self,event):
        slot=event['slot']
        if self.cursor>=len(self.SLOTS) or slot!=self.SLOTS[self.cursor]:raise ValueError('finite slot order violation')
        state_before=deepcopy(self.state)
        record={'slot':slot,'input_sha256':digest(event),'state_before':self.state.state_digest()}
        try:
            if slot=='daily':
                self.bundle_hash=validate_daily_bundle(event['bundle'],seal=event['seal'],role_approval=self.role_approval,expected_dates=self.expected_dates,required_roles=self.role_approval['roles'],session=self.session)
                self.rows=deepcopy(event['model_rows'])
                if set(self.rows)!=set(FAMILIES[:-1]) or event['proposal_bundle_hash']!=self.bundle_hash:raise ValueError('proposals not bound to daily bundle')
            elif slot=='known':
                assert_fixed_time(event['cutoff'],self.slots['known_cutoff'])
                bars=validate_bars(event['bars'],event['receipt'],session=self.session,cutoff=event['cutoff'],seal=event['seal'],mapping_evidence=self.bar_mapping)
                first=bars.groupby('symbol').open.first().to_dict();returns=validate_live_open(event['open_receipt'],session=self.session,previous_session=self.previous_session,decision_seal=event['seal'],futu_opens=first)
                proposals={f:np.array([self.rows[f][c] for c in ['soxl_weight','soxx_weight','cash_weight']]) for f in FAMILIES[:-1]};smoothed=self.state.smooth(proposals,returns)
                for f in smoothed:
                    for column,value in zip(['soxl_weight','soxx_weight','cash_weight'],smoothed[f]):self.rows[f][column]=value
                self.state.apply_actions(self.session,event['schedule']);prices=bars.groupby('symbol').close.last().reindex(['SOXL','SOXX']).to_numpy()
                self.pending_targets=self.state.open_targets(self.rows,prices,recovery_spec=self.spec,action_spec=self.action_spec)
            elif slot in ['open_intent','recovery_intent']:
                assert_fixed_time(event['fill_slot_at'],self.slots['open_fill' if slot=='open_intent' else 'recovery_fill'])
                if not self.state.trading_disabled:self.state.seal(slot,self.pending_targets,event['seal'],slot_at=event['fill_slot_at'],bindings={'feature_bundle':self.bundle_hash,'plan':PLAN_SHA})
            elif slot=='recovery_data':
                assert_fixed_time(event['cutoff'],self.slots['recovery_cutoff'])
                if not self.state.trading_disabled:
                    used=validate_bars(event['bars'],event['receipt'],session=self.session,cutoff=event['cutoff'],seal=event['seal'],mapping_evidence=self.bar_mapping)
                    points=used.copy();points['timestamp']=points.bar_start+pd.Timedelta(minutes=1);points['date']=pd.Timestamp(self.session);points['price_valid']=True;points['is_final']=True
                    aggregated=aggregate_first_hour(points).loc[pd.Timestamp(self.session)]
                    if aggregated.first_hour_ready!=1:raise ValueError('incomplete recovery aggregation')
                    fields={name:float(aggregated[name]) for name in FIELDS}
                    rows={f:pd.Series({**dict(self.rows[f]),**fields}) for f in FAMILIES[:-1]}
                    record['recovery_fields_from_validated_bars']=fields
                    self.pending_targets,votes,calls=self.state.recovery_targets(rows,recovery_spec=self.spec,controller=self.controller,selection=self.selection);self.recovery_calls+=calls
                    if self.recovery_calls>9:raise ValueError('recovery call budget exceeded')
                    record['votes']=votes
            elif slot in ['open_quote','recovery_quote']:
                intent='open_intent' if slot=='open_quote' else 'recovery_intent'
                fixed=self.slots['open_fill' if slot=='open_quote' else 'recovery_fill']
                if 'slot_at' in event:assert_fixed_time(event['slot_at'],fixed)
                if not self.state.trading_disabled:
                    if intent not in self.durable_intents:raise ValueError('intent not durably committed before quote')
                    record['execution']=self.state.execute_slot(intent,event['receipt'])
                else:
                    prices=validate_quote(event['receipt'],seal=event['prior_seal'],slot_at=fixed);record['passive_nav']=self.state.mark(prices,update_peak=False)
            elif slot=='close':
                assert_fixed_time(event['session_close'],self.slots['close'])
                used=validate_close_bars(event['bars'],event['receipt'],session=self.session,session_close=event['session_close'],commit_clock=event['commit_clock'],mapping_evidence=self.bar_mapping)
                prices=used.groupby('symbol').close.last().reindex(['SOXL','SOXX']).to_numpy();record['nav']=self.state.mark(prices,update_peak=True)
            elif slot=='terminal':
                if event['session']!=self.slots['terminal_session']:raise ValueError('terminal session differs from registration')
                assert_fixed_time(event['slot_at'],self.slots['terminal_fill'])
                self.state.apply_actions(event['session'],event['schedule']);prices=validate_quote(event['receipt'],seal=event['prior_seal'],slot_at=event['slot_at']);record['nav']=self.state.mark(prices,update_peak=False)
            record['status']='committed'
        except Exception as error:
            self.state=state_before
            self.state.disable(type(error).__name__+': '+str(error));record.update(status='unqualified_slot',error=str(error))
        record['state_after']=self.state.state_digest();self.journal.append(record);self.cursor+=1
        if slot in ['open_intent','recovery_intent'] and record['status']=='committed':self.durable_intents.add(slot)
        payload={key:getattr(self,key) for key in ['state','cursor','journal','rows','bundle_hash','pending_targets','recovery_calls','durable_intents']}
        try:persist_checkpoint(self.journal_dir,self.cursor,{'event':event,'result':record},payload)
        except Exception:
            self.durable_intents.discard(slot);self.state.disable('durable commit failed');raise
        return deepcopy(record)
    def run(self,events):
        for event in events:self.dispatch(event)
        if self.cursor!=len(self.SLOTS):raise ValueError('incomplete finite pilot stream')
        return {'journal':self.journal,'state':self.state,'qualified_interval':not self.state.trading_disabled,'recovery_calls':self.recovery_calls}


def _local_source(root,ref,artifacts):
    card=root/ref['card'];summary=json.loads((card/'summary.json').read_text());local=Path(summary['recorder_local_dir'])
    missing=[str(local/'artifacts'/key) for key in artifacts if not (local/'artifacts'/key).is_file()]
    if missing:raise FileNotFoundError('local-only source missing; no NAS permitted: '+json.dumps(missing))
    binding=json.loads((card/'nas_binding.json').read_text())
    if binding['evidence']['sha256']!=ref['archive']:raise ValueError('archive identity changed')
    return load_verified_recorder_objects(card,expected_recorder_id=ref['recorder_id'],expected_manifest_sha256=ref['inventory_sha256'],artifacts=artifacts)

def _fingerprint(model):
    return hashlib.sha256(b''.join(k.encode()+v.detach().cpu().numpy().tobytes() for k,v in sorted(model.network.state_dict().items()))).hexdigest()

def _dataset(features,common):
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP
    frame=pd.concat({'feature':features,'common':common},axis=1);frame.index=pd.MultiIndex.from_arrays([frame.index,['SOXL_SOXX_CASH']*len(frame)],names=['datetime','instrument'])
    return DatasetH(handler=DataHandlerLP.from_df(frame),segments={'predict':(str(features.index.min().date()),str(features.index.max().date()))})

def run_offline(root):
    import qlib
    from qlib.data import D
    from qlib.workflow import R
    from src.model.minute_pit_allocator import _scaled
    from src.workflow.direct_continuous_soxx_allocation import _load_frame,DirectContinuousAllocationSpec
    from src.workflow.pit_constituent_auxiliary_training_allocation import _soxx_common_training_frame,COMMON_FEATURES
    root=Path(root).resolve();card=root/'research'/NAME;require_phase(card,'implementation');plan=json.loads((card/'plan.json').read_text())
    if (card/'offline_summary.json').exists():return json.loads((card/'offline_summary.json').read_text())
    with network_denied() as network_attempts:
        qlib.init(provider_uri=str(root/'data/qlib/soxl'),region='us',kernels=1)
        index_ref=plan['models']['index_candidate_and_control'];seeds=[71,172,273]
        keys=['inputs/plan']+[f'models/{a}_{s}' for a in ['candidate','control'] for s in seeds]+[f'predictions/{a}_{s}' for a in ['candidate','control'] for s in seeds]
        index,ia=_local_source(root,index_ref,keys)
        reference,ra=_local_source(root,plan['models']['reference27baa'],['inputs/features','inputs/feature_manifest']+[f'models/candidate_{s}' for s in seeds]+[f'analysis/predictions/candidate_{s}' for s in seeds])
        training,ta=_local_source(root,index['inputs/plan']['training_source'],['inputs/features','inputs/common'])
        sources=json.loads((card/'frozen_controller_sources.json').read_text())
        controller,ca=_local_source(root,sources['controller'],list(sources['controller']['artifacts']));action,aa=_local_source(root,sources['action_and_recovery_spec'],list(sources['action_and_recovery_spec']['artifacts']));raw,rwa=_local_source(root,sources['raw_inheritance'],list(sources['raw_inheritance']['artifacts']))
        # Dates fixed by saved-index identity only, never selected from errors/outcomes.
        common_dates=pd.DatetimeIndex(index['predictions/candidate_71'].index).rename('datetime')
        for a in ['candidate','control']:
            for seed in seeds:pd.testing.assert_index_equal(pd.DatetimeIndex(index[f'predictions/{a}_{seed}'].index).rename('datetime'),common_dates)
        for seed in seeds:
            if not common_dates.isin(reference[f'analysis/predictions/candidate_{seed}'].index).all():raise ValueError('reference missing common dates')
        dates=common_dates[-5:];assert len(dates)==5
        primary_columns=index['models/candidate_71'].base_features
        base=training['inputs/features'].reindex(dates)[primary_columns];common=training['inputs/common'].reindex(dates)[list(COMMON_FEATURES)];extra=reference['inputs/features'].reindex(dates)
        if not np.isfinite(base.to_numpy()).all() or not np.isfinite(common.to_numpy()).all():raise ValueError('missing canonical fixture inputs')
        native_checks=[]
        for label,provider,columns,expected in [('primary',root/'data/qlib/minute_information_pit_allocator_20261012/allocation_features',primary_columns,base),('reference_extra',Path(reference['inputs/feature_manifest']['provider_uri']),list(extra.columns),extra)]:
            if not provider.is_dir():raise FileNotFoundError(str(provider))
            qlib.init(provider_uri=str(provider),region='us',kernels=1)
            read=D.features(['SOXL_SOXX_CASH'],['$'+c for c in columns],start_time=dates[0],end_time=dates[-1],freq='day').xs('SOXL_SOXX_CASH',level='instrument').rename(columns=lambda x:x.removeprefix('$')).reindex(dates)
            np.testing.assert_allclose(read.to_numpy(float),expected.to_numpy(float),rtol=0,atol=0,equal_nan=True)
            native_checks.append({'surface':label,'dates':len(dates),'fields':len(columns),'exact_D_features_match':True})
        # Genuine original-formula reconstruction is a separate check, not DatasetH integrity.
        reconstruction=[]
        try:
            rebuilt,_,_= _load_frame(DirectContinuousAllocationSpec(root=root),include_incomplete_latest=True)
            rebuilt_common=_soxx_common_training_frame(index['models/candidate_71'].spec,end=str(dates[-1].date()))
            for label,observed,expected in [('primary39',rebuilt.reindex(dates)[primary_columns],base),('common16',rebuilt_common.reindex(dates)[list(COMMON_FEATURES)],common)]:
                a,b=observed.to_numpy(float),expected.to_numpy(float);passed=bool(np.isfinite(a).all() and np.allclose(a,b,rtol=1e-6,atol=1e-8,equal_nan=False))
                reconstruction.append({'surface':label,'passed':passed,'maximum_absolute_error':float(np.nanmax(np.abs(a-b))),'method':'original frozen formula on existing local providers; not source selection/fetch'})
        except (ValueError,FileNotFoundError,KeyError) as error:reconstruction.append({'surface':'formula_rebuild','passed':False,'error':str(error)})
        models={};datasets={};expected_predictions={};initial_hashes={};model_preflight=[]
        for family in ['reference27baa','index_candidate','index_control']:
            features=base.join(extra) if family=='reference27baa' else base
            dataset=_dataset(features,common);prepared=dataset.prepare('predict',col_set=['feature','common'])
            np.testing.assert_array_equal(prepared['common'].to_numpy(),common.to_numpy())
            for seed in seeds:
                key=f'{family}_{seed}';model=reference[f'models/candidate_{seed}'] if family=='reference27baa' else index[f'models/{family.removeprefix("index_")}_{seed}']
                px=torch.cat([_scaled(prepared['feature'],model.base_features,model.primary_scaler),model._extra_tensor(prepared['feature'])],dim=1);cx=_scaled(prepared['common'],list(COMMON_FEATURES),model.common_scaler)
                if not torch.isfinite(px).all() or not torch.isfinite(cx).all() or cx.shape!=(5,16):raise ValueError('invalid transformed fixture')
                if model.network.allocation_head.in_features!=px.shape[1]+8 or model.network.training:raise ValueError('model shape/mode mismatch')
                models[key]=model;datasets[key]=dataset;initial_hashes[key]=_fingerprint(model)
                expected_predictions[key]=(reference[f'analysis/predictions/candidate_{seed}'] if family=='reference27baa' else index[f'predictions/{family.removeprefix("index_")}_{seed}']).reindex(dates)
                if not np.isfinite(expected_predictions[key].to_numpy()).all():raise ValueError('saved prediction coverage invalid')
                model_preflight.append({'key':key,'rows':5,'primary_tensor_columns':px.shape[1],'common_columns':16,'state_sha256':initial_hashes[key]})
        progress_path=card/'offline_attempts.json';code_hashes={f:hashlib.sha256((root/f).read_bytes()).hexdigest() for f in MODULES}
        progress=json.loads(progress_path.read_text()) if progress_path.exists() else {'plan_sha256':PLAN_SHA,'code_sha256':code_hashes,'calls_attempted':[],'calls_saved':[]}
        if progress['code_sha256']!=code_hashes:raise ValueError('code changed after offline registration; requires explicit repair record')
        kw={'experiment_name':NAME+'_implementation','uri':str(card/'qlib_recorder')}
        if progress.get('recorder_id'):kw.update(recorder_id=progress['recorder_id'],resume=True)
        with R.start(**kw):
            rec=R.get_recorder();progress.update(recorder_id=rec.id,experiment_id=str(rec.experiment_id));atomic_json(progress,progress_path)
            if not progress.get('sources_saved'):
                rec.save_objects(artifact_path='sources',**{f.replace('/','__'):(root/f).read_text() for f in MODULES});rec.save_objects(artifact_path='inputs',plan=plan,approval=json.loads((card/'implementation_approval.json').read_text()),fixture_dates=dates,primary=base,common=common,reference_extra=extra,expected_predictions=expected_predictions,model_preflight=model_preflight,source_audits={'index':ia,'reference':ra,'training':ta,'controller':ca,'action':aa,'raw':rwa},native_readback=native_checks,formula_reconstruction=reconstruction)
                progress['sources_saved']=True;atomic_json(progress,progress_path)
            checks=[]
            for key,model in models.items():
                if key in progress['calls_saved']:prediction=rec.load_object('offline_predictions/'+key)
                else:
                    if key in progress['calls_attempted'] or len(progress['calls_attempted'])>=9:raise ValueError('nine-call budget; no silent model retry')
                    progress['calls_attempted'].append(key);atomic_json(progress,progress_path)
                    prediction=model.predict(datasets[key]);rec.save_objects(artifact_path='offline_predictions',**{key:prediction});progress['calls_saved'].append(key);atomic_json(progress,progress_path)
                flat=prediction.copy();flat.index=flat.index.get_level_values('datetime') if isinstance(flat.index,pd.MultiIndex) else flat.index
                if len(flat)!=5 or not flat.index.equals(dates):raise ValueError('prediction dates changed')
                np.testing.assert_allclose(flat.to_numpy(),expected_predictions[key].to_numpy(),rtol=1e-6,atol=1e-8)
                if _fingerprint(model)!=initial_hashes[key]:raise ValueError('inference mutated frozen model')
                checks.append({'model':key,'batch_rows':5,'maximum_prediction_error':float(np.max(np.abs(flat.to_numpy()-expected_predictions[key].to_numpy()))),'model_unchanged':True});rec.log_metrics(saved_allocator_calls=len(progress['calls_saved']));print('saved one batch',key,flush=True)
            result={'experiment':NAME,'phase':'implementation_only','decision':'offline_contract_only_source_mapping_and_live_activation_pending','plan_sha256':PLAN_SHA,'allocator_calls':len(progress['calls_attempted']),'model_row_outputs':45,'model_parity_passed':True,'source_dataset_readback_passed':True,'formula_reconstruction_passed':all(x['passed'] for x in reconstruction),'formula_reconstruction':reconstruction,'source_role_mapping_passed':False,'network_calls':len(network_attempts),'probe_calls':0,'pilot_launches':0,'new_fits':0,'new_forward_days':0,'old_days_credited':0,'real_orders':0,'pilot_ready':False,'native_policy_backtests':0,'live_recovery_fill_qualified':False,'implemented':['strict phase/receipt/bar/quote/volume validators','12-book clone/commit state and raw settlement','finite receipted-event runner','local-only nine-model batch/readback notebook'],'not_implemented_or_unqualified':['approved live role-to-provider mapping','live collector transport wiring','observed bar mapping proof','probe/pilot phase permissions','live recovery order evidence'],'recorder_id':rec.id,'experiment_id':str(rec.experiment_id),'recorder_uri':str(card/'qlib_recorder'),'recorder_local_dir':str(card/'qlib_recorder'/str(rec.experiment_id)/rec.id),'artifact_policy':{'policy':'slim_card_recorder_first','workflow_config':'workflow.yaml','recorder_only_artifacts':['inputs/*','sources/*','offline_predictions/*','analysis/*']}}
            rec.save_objects(artifact_path='analysis',summary=result,model_checks=pd.DataFrame(checks));rec.log_metrics(registered_artifact_count=sum(x.is_file() for x in (Path(result['recorder_local_dir'])/'artifacts').rglob('*')))
        pd.DataFrame(checks).to_csv(card/'offline_model_checks.csv',index=False);pd.DataFrame(reconstruction).to_csv(card/'offline_formula_checks.csv',index=False)
        artifact=Path(result['recorder_local_dir'])/'artifacts';atomic_json({'recorder_id':rec.id,'entries':[{'path':str(x.relative_to(card)),'sha256':hashlib.sha256(x.read_bytes()).hexdigest(),'bytes':x.stat().st_size} for x in sorted(artifact.rglob('*')) if x.is_file()]},card/'verification_manifest.json');atomic_json(result,card/'offline_summary.json');return result

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path.cwd());args=parser.parse_args();print(json.dumps(run_offline(args.root),ensure_ascii=False,indent=2))
