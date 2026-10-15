"""Bounded existing-evidence assessment. No scheduler, quote, fit or replay calls."""
import hashlib,json
from datetime import datetime,timezone
from pathlib import Path
from urllib.parse import urlparse,unquote
import pandas as pd
from src.workflow.minute_action_value import _source
from src.data.minute_intake import atomic_json
from src.data.nas_market_data import archive_nas_recorder


def run(*,root,card,plan):
    import qlib,yaml
    from qlib.data import D
    from qlib.workflow import R
    if (card/'summary.json').exists():return json.loads((card/'summary.json').read_text())
    digest=hashlib.sha256((card/'plan.json').read_bytes()).hexdigest();assert digest=='02583876910a238a0d42119fe1ae5b0cb41e7d86427dc8c9a41914e5b2da2aed'
    started=datetime.now(timezone.utc).isoformat();snapshots={};identities=[];parsed={}
    def capture(relative,expected=None,optional=False):
        path=root/relative
        if not path.exists():
            if not optional:raise ValueError('missing frozen input '+relative)
            identities.append({'path':relative,'status':'not_present','observed_at_utc':datetime.now(timezone.utc).isoformat()});return None
        raw=path.read_bytes();sha=hashlib.sha256(raw).hexdigest()
        if expected and sha!=expected:raise ValueError('frozen source changed '+relative)
        identities.append({'path':relative,'status':'read_only_snapshot','sha256':sha,'bytes':len(raw),'observed_at_utc':datetime.now(timezone.utc).isoformat()})
        if relative.endswith('/config.json'):
            config=json.loads(raw);snapshots[relative]=json.dumps({k:config[k] for k in ['release_manifest','expected_manifest_sha256','enabled'] if k in config}).encode()
        else:snapshots[relative]=raw
        if relative.endswith('.json'):parsed[relative]=json.loads(raw)
        return raw
    for section in ['static_code_sha256','static_release_sha256']:
        for name,sha in plan['bounded_sources'][section].items():capture(name,sha)
    for name in plan['bounded_sources']['mutable_runtime_files']+plan['bounded_sources']['provider_manifests']:capture(name,optional=True)
    for source in [plan['reference'],plan['fundamental_source']]:
        local=json.loads((root/source['card']/'summary.json').read_text())['recorder_local_dir']
        if not Path(local).is_dir():raise ValueError('pinned local source absent; network restore not authorized')
    reference,ra=_source(root,plan['reference'],plan['reference_artifacts'])
    fundamental,fa=_source(root,plan['fundamental_source'],plan['fundamental_artifacts'])
    model_identities=[]
    for seed in [71,172,273]:
        model=reference[f'models/candidate_{seed}'];sha=hashlib.sha256()
        for k,v in sorted(model.network.state_dict().items()):sha.update(k.encode());sha.update(v.detach().cpu().numpy().tobytes())
        scalers={k:getattr(model,k) for k in ['primary_scaler','common_scaler','extra_scaler']}
        model_identities.append({'seed':seed,'trained_tensor_sha256':sha.hexdigest(),'scalers_sha256':hashlib.sha256(json.dumps(scalers,sort_keys=True).encode()).hexdigest(),'base_feature_count':len(model.base_features),'numeric_extra_features':model.extra_features,'training_rows':model.audit['training_rows'],'cutoff':str(model.cutoff),'new_predict_or_fit_calls':0})
    calendars=[]
    for provider,freq in [(plan['provider_uri'],'1min'),('data/qlib/published_futu_affine_20260925_v1','day')]:
        try:
            qlib.init(provider_uri=str(root/provider),region='us',kernels=1)
            dates=pd.DatetimeIndex(D.calendar(freq=freq));calendars.append({'provider':provider,'freq':freq,'rows':len(dates),'first':str(dates.min()),'last':str(dates.max()),'role':'existingcalendarcoverage_not_receiptclock'})
        except Exception as error:calendars.append({'provider':provider,'freq':freq,'status':'needs_data','error':str(error)})
    candidates=[];book_summaries=[]
    for name,obj in parsed.items():
        if '/books/' not in name:continue
        book_summaries.append({'path':name,'candidate_identity':obj.get('candidate_identity'),'cost_bps':obj.get('cost_bps'),'last_event_at':obj.get('last_event_at'),'stored_event_count':len(obj.get('events',{})),'credited_new_reference_days':0,'liveness_inferred':False})
        for event_id,receipt in obj.get('events',{}).items():
            link=receipt.get('qlib_evidence')
            if link and receipt.get('observed_at'):candidates.append((pd.Timestamp(receipt['observed_at']),name,event_id,link))
    old={};old_audit={'status':'not_observed_in_bounded_readback'}
    if candidates:
        _,book,event_id,link=sorted(candidates,key=lambda x:(x[0],x[1],x[2]))[-1]
        uri=Path(link['recorder_uri']).resolve();allowed=(root/'runtime/soxl/stage1_regime_paper/stage1_regime_paper_20261001/qlib_recorder').resolve()
        if uri!=allowed:raise ValueError('linked old Recorder outside bounded runtime')
        with R.uri_context(str(uri)):
            exp=R.get_exp(experiment_name=link['experiment_name'],create=False)
            rec=exp.get_recorder(recorder_id=link['recorder_id'],create=False)
            artifact=Path(unquote(urlparse(rec.artifact_uri).path)).resolve()
            if not artifact.is_relative_to(uri):raise ValueError('old artifacts outside local runtime')
            hashes={}
            for key in ['event.pkl','receipt.pkl','state.pkl']:
                path=artifact/key;before=path.read_bytes();hashes[key]=hashlib.sha256(before).hexdigest();old[key]=rec.load_object(key)
                if path.read_bytes()!=before:raise ValueError('old record changed during readback')
        old_audit={'status':'one_linked_existing_triplet_read','book':book,'event_id':event_id,'recorder_id':rec.id,'artifact_sha256':hashes,'observed_at':old['event.pkl'].get('observed_at'),'quote_as_of':old['event.pkl'].get('quote_as_of'),'feature_available_at':old['event.pkl'].get('feature_available_at'),'event_fields':list(old['event.pkl']),'credited_new_reference_days':0}
    old_release=parsed['production/stage1_regime_paper_20261001/release_manifest.json'];production=parsed['production/stage1_futu_sim_v1/runtime_release.json']
    pair=fundamental['analysis/pairs'];latest=pair.loc[pair.actual_report_available_at.notna()].sort_values('actual_report_available_at').iloc[-1]
    trace=reference['inputs/feature_trace'];frozen_frame=reference['inputs/frame']
    basis=parsed.get('data/qlib/full_path_tiered_allocation_20261013/qlib_store_manifest.json',{})
    def loc(path,needle):
        lines=snapshots[path].decode().splitlines()
        matches=[str(i+1) for i,line in enumerate(lines) if needle in line]
        return path+':'+(','.join(matches[:3]) if matches else 'not_found')
    findings=[
      {'requirement':'immutable_model_identity','verdict':'incompatible_old_identity_or_clock','evidence':f"27baa three models read/hashed; productionallocation={production['allocation']['recorder_id']}; oldpaperallocation={old_release['allocation_source']['recorder_id']}",'locator':'reference Recorder models/candidate_71,172,273; production release manifests','missing':'No isolated27baa diagnostic release/own-ledger state identity registered; oldcounts remain0.'},
      {'requirement':'Qlib_record_before_state_commit','verdict':'reusable_component_only','evidence':old_audit['status']+'; oldwriter logs event/receipt/state before localcommit; hashing/idempotency pattern reusable','locator':loc('src/schedule/stage1_regime_paper.py','recorder.save_objects')+'; '+loc('src/workflow/stage1_regime_paper.py','evidence_writer('),'missing':'Existing adapter is old10/25bpsfractionalstrategy, not27baa coupled0/2/5 Tiered policy.'},
      {'requirement':'actual_source_receipt_clock','verdict':'needs_data_proof','evidence':'Historicalfeaturetrace fields='+','.join(trace.columns)+'; oldtriplet timestamps='+json.dumps({k:old_audit.get(k) for k in ['observed_at','quote_as_of','feature_available_at']}),'locator':loc('src/schedule/us_open_quotes.py','update_time')+'; reference inputs/feature_trace','missing':'Bar/quote eventtime and assumedavailability do not establish immutable payload source_received_at/used_rows before every09:29/09:46/10:31cutoff.'},
      {'requirement':'decision_before_independent_fill','verdict':'incompatible_old_identity_or_clock','evidence':'Oldopen09:31–09:36; samecapturedprices enterdecision andpapertrade. Referenceknown09:45→decision09:46→fill09:47, recoverydecision10:31→fill10:32.','locator':loc('src/schedule/stage1_regime_paper.py','571 <=')+'; '+loc('src/schedule/stage1_regime_paper.py','"prices": group["prices"]'),'missing':'No27baa sealed decisionreceipt before separatelycapturedfuturefill; no liveproof ofall3prefillcaps/votes barrier.'},
      {'requirement':'fundamental_refresh_and_unknown_reset','verdict':'needs_implementation','evidence':f"Latestfrozenrealization={latest.target_quarter}, delayedavailable={latest.actual_report_available_at}; sourcefirst_public_timestamp_verified=False; aligner handlesunknown/futureguide rules.",'locator':loc('src/data/nvda_own_guidance_realization.py','actual_report_available_at')+'; fundamental Recorder analysis/pairs','missing':'Append-onlyneworiginalreport/scopequalification/payloadreceipt/version adapter not established. Latestknown maypersist, but newlyreceivedblockedreport mustreset; delayedreceipt cannotbackdateearlierdecisions.'},
      {'requirement':'affine_vs_raw_share_cash_CA_basis','verdict':'needs_data_proof','evidence':'Frozenprovider live_receipt_fill_dividend_parity_qualified='+str(basis.get('live_receipt_fill_dividend_parity_qualified'))+'; affine A/B priceoperator PIT_or_total_return_certified=False.','locator':loc('src/data/minute_price_basis.py','PIT_or_total_return_certified')+'; '+loc('src/data/full_path_prices.py','live_receipt_fill_dividend_parity_qualified'),'missing':'Frozenpastpriceplane/rawshareunitmapping plus split/dividendcash effective/receivedevents andNAV reconciliation notqualified. No newrehabpastrebasing or doubledividendcredit. No exactbrokerparity.'},
      {'requirement':'fresh_observer_state_and_pairing','verdict':'needs_implementation','evidence':'Existingbooks are separateoldcandidate/baseline10/25 identity; coupledresearchpolicy owns0/2/5ledgers but noauthorizednew observer registered.','locator':loc('src/backtest/causal_cost_scenario_controller.py','ledgers=')+'; old release initial_cash/cost_bps','missing':'Freeze ownnew100000cash/start,positions/monthlyfees/EMAinitialstate andpairedSOXX/reference clocks, missedslots/corrections beforelaunch. No oldSIMholdings/dayreuse.'},
      {'requirement':'prospective_evaluation_contract','verdict':'needs_implementation','evidence':'Historical27baa rolling70 andbootstrapremainfalse; campaignminimum60paired/15perstate/3transitions isminimumonly; oldplanpendingotheridentity.','locator':'docs/three_asset_research_contract.md; reference analysis/summary; old forward_plan.json','missing':'Separate diagnosticnotpromoted registration withnot_before,exactstates/fixedevaluationtimes/newdaysonly;126day/annualneedactualnewcoverage. No optionaldailytestuntilpass.'}
    ]
    table=pd.DataFrame(findings)
    changed=[]
    for item in identities:
        path=root/item['path']
        if item.get('sha256') and (not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']):changed.append(item['path'])
    next_action='Freeze ONE isolated decision-receipt adapter contract for diagnostic research-price-proxy observation: immutable27baa inputs/state/0-2-5 ledger schema, received_at/used-row hashes, seal decision before separately observed laterfill, failmissing slots; offline existing/synthetic clock-contract qualification first. No schedulerlaunch orquote/datafetch untilseparatereview. Rawshare/dividend economicqualification remainsseparateharddependency.'
    tracking=card/'qlib_recorder'
    with R.start(experiment_name=plan['experiment'],recorder_name='bounded_read_only_assessment',uri=str(tracking)):
        rec=R.get_recorder();rec.log_params(plan_sha256=digest,new_fits=0,new_replays=0,new_launches=0)
        rec.save_objects(artifact_path='inputs',plan=plan,file_snapshots=snapshots,file_identities=identities,source_audits={'reference':ra,'fundamental':fa},old_linked_triplet=old,old_linked_audit=old_audit)
        rec.save_objects(artifact_path='sources',assessment_workflow=Path(__file__).read_text())
        policy={'policy':'slim_card_recorder_first','workflow_config':'workflow.yaml','recorder_only_artifacts':['inputs/*','sources/*']}
        summary={'experiment':plan['experiment'],'plan_sha256':digest,'role':plan['role'],'decision':'not_ready_for_qualified_forward_separate_diagnostic_adapter_contract_needed','objective_achieved':False,'historical_rolling_gate':False,'historical_bootstrap_gate':False,'new_qualified_forward_days':0,'old_SIM_days_credited':0,'new_fits':0,'new_replays':0,'new_native_accounts':0,'new_data_fetches':0,'launches':0,'orders':0,'production_scheduler_changes':0,'readback_started_at_utc':started,'source_changed_during_readback':changed,'existing_calendar_coverage':calendars,'model_identities':model_identities,'old_runtime_book_summaries':book_summaries,'old_linked_receipt':old_audit,'frozen_frame_first_trade':str(frozen_frame.trade_date.min()),'frozen_frame_last_exit':str(frozen_frame.exit_date.max()),'reference_feature_rows':len(reference['inputs/features']),'latest_frozen_realization_quarter':latest.target_quarter,'economic_forward_qualified':False,'diagnostic_proxy_observer_currently_launched':False,'smallest_next_action':next_action,'recorder_id':rec.id,'experiment_id':str(rec.experiment_id),'recorder_uri':str(tracking),'recorder_local_dir':str(tracking/str(rec.experiment_id)/rec.id),'provider_uri':str(root/plan['provider_uri']),'artifact_policy':policy,'qlib_docs':plan['qlib_docs']}
        rec.save_objects(artifact_path='analysis',requirements=table,summary=summary)
        artifacts=Path(summary['recorder_local_dir'])/'artifacts';rec.log_metrics(registered_artifact_count=sum(p.is_file() for p in artifacts.rglob('*')),requirements_assessed=len(table),new_forward_days=0)
    table.to_csv(card/'requirement_evidence.csv',index=False);atomic_json(identities,card/'readback_identities.json')
    atomic_json({'recorder_id':rec.id,'entries':[{'path':str(p.relative_to(card)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in sorted(artifacts.rglob('*')) if p.is_file()]},card/'verification_manifest.json')
    binding=json.loads((root/plan['reference']['card']/'nas_binding.json').read_text());binding.pop('evidence');atomic_json(binding,card/'nas_binding.json')
    (card/'workflow.yaml').write_text(yaml.safe_dump({'experiment':plan['experiment'],'provider_uri':summary['provider_uri'],'recorder_uri':str(tracking),'artifact_policy':policy},sort_keys=False))
    summary['nas_evidence']=archive_nas_recorder(card/'nas_binding.json',workspace_root=root,card=card,recorder_id=rec.id,experiment_id=str(rec.experiment_id));atomic_json(summary,card/'summary.json');return summary
