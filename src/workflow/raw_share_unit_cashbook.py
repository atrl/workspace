"""Three frozen unit books, native Qlib Recorder evidence, no strategy replay."""
import hashlib,json
from pathlib import Path
import pandas as pd
from src.workflow.minute_action_value import _source
from src.backtest.raw_share_cashbook import new_unit_book,apply_unit_event,dec
from src.data.minute_intake import atomic_json
from src.data.nas_market_data import archive_nas_recorder


def run(*,root,card,plan):
    from qlib.workflow import R
    import yaml
    if (card/'summary.json').exists():return json.loads((card/'summary.json').read_text())
    digest=hashlib.sha256((card/'plan.json').read_bytes()).hexdigest();assert digest=='470922f2e293fe7ce6f0844f274bf7e57ee3f49fdd6d8aba82929e1ea99e127a'
    bridge,ba=_source(root,plan['sources']['bridge'],plan['source_artifacts']['bridge'])
    cash,ca=_source(root,plan['sources']['cash_event'],plan['source_artifacts']['cash_event'])
    event=cash['analysis/event_qualification'];convention=cash['analysis/paper_convention']
    assert event['currency'] is None and event['gross_cash_per_original_share_decimal']=='0.02501'
    assert str(pd.Timestamp(convention['conditional_paper_available_cash_at']))=='2023-03-29 09:30:00-04:00'
    marks=bridge['analysis/marks'];trace=[];checks=[];final={};receipts={};duplicates=[];events_by_case={}
    for case in plan['cases']:
        local=marks.loc[marks['case'].eq(case['id'])].sort_values('mark_at')
        assert local.mark_at.astype(str).tolist()==case['marks']
        first=pd.Timestamp(case['marks'][0]).tz_localize('America/New_York')
        state=new_unit_book(case['symbol'],shares=case['initial_raw_shares'],cash=case['initial_spendable_cash_decimal'],initial_at=first)
        queue=[]
        for row in local.itertuples():
            stamp=pd.Timestamp(row.mark_at).tz_localize('America/New_York')
            queue.append({'event_id':case['id']+':mark:'+stamp.isoformat(),'kind':'mark','symbol':case['symbol'],'effective_at':stamp.isoformat(),'raw_price':str(row.qlib_raw_price),'source':'pinnedbridge/analysis/marks'})
        for action in case['actions']:
            stamp=pd.Timestamp(action['effective_at'].replace(' America/New_York','')).tz_localize('America/New_York')
            payload={'event_id':action['source_event_id']+':'+action['kind'],'kind':action['kind'],'symbol':case['symbol'],'source_event_id':action['source_event_id'],'effective_at':stamp.isoformat()}
            if action['kind']=='split':
                split=bridge['analysis/cash_fields'].loc[bridge['analysis/cash_fields']['case'].eq(case['id'])].iloc[0]
                ratio=dec(split.split_ert)/dec(split.split_base);assert ratio==3
                payload['new_shares_per_old_share']=str(ratio)
            elif action['kind']=='dividend_entitlement':payload.update(eligible_raw_shares='100',gross_rate_per_original_share=event['gross_cash_per_original_share_decimal'],declared_currency_assumption='USD',declared_available_at=pd.Timestamp(convention['conditional_paper_available_cash_at']).isoformat())
            else:payload['declared_available_at']=pd.Timestamp(convention['conditional_paper_available_cash_at']).isoformat()
            queue.append(payload)
        queue.sort(key=lambda x:(pd.Timestamp(x['effective_at']),x['event_id']))
        case_receipts=[]
        for item in queue:
            state,receipt=apply_unit_event(state,item);case_receipts.append(receipt);trace.append({'case':case['id'],**receipt})
        duplicated,receipt=apply_unit_event(state,queue[0]);assert duplicated==state and receipt['status']=='already_recorded'
        duplicates.append({'case':case['id'],'old_first_event_duplicate_is_noop':True})
        expected_shares=300 if case['id']=='split_declared_no_cash' else 100
        expected_cash=dec('2.50100') if case['id']=='cash_dividend_multiplicative' else dec(0)
        assert dec(state['raw_shares'])==expected_shares and dec(state['spendable_cash'])==expected_cash and not state['receivables']
        if case['id']=='cash_dividend_multiplicative':
            for receipt in case_receipts:
                if receipt['kind']=='dividend_entitlement' or (receipt['kind']=='mark' and pd.Timestamp(receipt['effective_at'])>pd.Timestamp('2023-03-21',tz='UTC')):
                    assert dec(receipt['spendable_cash'])==0 and dec(receipt['nonspendable_receivable'])==dec('2.50100')
            assert case_receipts[-1]['NAV_if_same_time_raw_mark'] is None
        checks.append({'case':case['id'],'unique_events':len(queue),'final_raw_shares':state['raw_shares'],'final_spendable_cash':state['spendable_cash'],'final_receivable':'0','invariants_passed':True,'currency_evidence':'explicit_USD_paper_assumption','source_currency_certified':False,'same_time_mark_NAV_count':sum(x['NAV_if_same_time_raw_mark'] is not None for x in case_receipts),'duplicate_noop_passed':True})
        final[case['id']]=state;receipts[case['id']]=case_receipts;events_by_case[case['id']]=queue
    assert len(trace)==11 and sum(r['kind']=='mark' for r in trace)==8
    tracking=card/'qlib_recorder'
    with R.start(experiment_name=plan['experiment'],recorder_name='fixed_raw_unit_books',uri=str(tracking)):
        rec=R.get_recorder();rec.log_params(plan_sha256=digest,unit_books=3,strategy_replays=0)
        rec.save_objects(artifact_path='inputs',plan=plan,source_audits={'bridge':ba,'cash':ca},marks=marks,cash_event=event,paper_convention=convention,events=events_by_case)
        rec.save_objects(artifact_path='sources',execution_workflow=Path(__file__).read_text(),accounting=(root/'src/backtest/raw_share_cashbook.py').read_text())
        for name in final:rec.save_objects(artifact_path='unit_books/'+name,events=events_by_case[name],receipts=receipts[name],final_state=final[name])
        policy={'policy':'slim_card_recorder_first','workflow_config':'workflow.yaml','recorder_only_artifacts':['inputs/*','sources/*','unit_books/*']}
        summary={'experiment':plan['experiment'],'plan_sha256':digest,'decision':'fixed_raw_unit_cashbook_contract_passed_not_strategy_or_forward_qualification','objective_achieved':False,'unit_books':3,'unique_events':11,'saved_raw_marks_reused':8,'duplicate_noop_checks':3,'all_unit_invariants_passed':True,'currency_contract':plan['new_currency_contract'],'source_currency_certified':False,'historical_PIT_qualified':False,'new_price_reads':0,'new_calendar_reads':0,'new_fetches':0,'new_model_fits':0,'new_strategy_replays':0,'new_native_backtest_accounts':0,'orders':0,'observer_launches':0,'production_changes':0,'case_checks':checks,'next_action':plan['next_action_if_passed'],'recorder_id':rec.id,'experiment_id':str(rec.experiment_id),'recorder_uri':str(tracking),'recorder_local_dir':str(tracking/str(rec.experiment_id)/rec.id),'provider_uri':str(root/plan['provider_uri']),'artifact_policy':policy}
        rec.save_objects(artifact_path='analysis',summary=summary,trace=pd.DataFrame(trace),checks=pd.DataFrame(checks),duplicate_checks=duplicates)
        artifacts=Path(summary['recorder_local_dir'])/'artifacts';rec.log_metrics(registered_artifact_count=sum(p.is_file() for p in artifacts.rglob('*')),unit_books=3,unit_events=11)
    pd.DataFrame(trace).to_csv(card/'unit_event_ledger.csv',index=False);pd.DataFrame(checks).to_csv(card/'unit_checks.csv',index=False)
    atomic_json({'recorder_id':rec.id,'entries':[{'path':str(p.relative_to(card)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in sorted(artifacts.rglob('*')) if p.is_file()]},card/'verification_manifest.json')
    binding=json.loads((root/plan['sources']['bridge']['card']/'nas_binding.json').read_text());binding.pop('evidence');atomic_json(binding,card/'nas_binding.json')
    (card/'workflow.yaml').write_text(yaml.safe_dump({'experiment':plan['experiment'],'provider_uri':summary['provider_uri'],'recorder_uri':str(tracking),'artifact_policy':policy},sort_keys=False))
    summary['nas_evidence']=archive_nas_recorder(card/'nas_binding.json',workspace_root=root,card=card,recorder_id=rec.id,experiment_id=str(rec.experiment_id));atomic_json(summary,card/'summary.json');return summary
