"""One existing corporate-action event and its conditional paper convention."""
from decimal import Decimal
import hashlib,io,json
from pathlib import Path
import pandas as pd
from src.core.recorder_sources import load_verified_recorder_objects
from src.workflow.minute_action_value import _source
from src.data.minute_intake import atomic_json
from src.data.nas_market_data import archive_nas_recorder


def run(*,root,card,plan):
    from qlib.workflow import R
    import yaml
    if (card/'summary.json').exists():return json.loads((card/'summary.json').read_text())
    digest=hashlib.sha256((card/'plan.json').read_bytes()).hexdigest();assert digest=='82a3085dee6023be3a3035fac51e47855bbd35c5c6f56ff363ee41cee0e5cafa'
    source=plan['corporate_source'];parent=root/source['card']
    objects,audit=load_verified_recorder_objects(parent,expected_recorder_id=source['recorder_id'],expected_manifest_sha256=source['inventory_sha256'],artifacts=source['native_artifacts'])
    metadata=json.loads((parent/'summary.json').read_text());artifact_root=Path(metadata['recorder_local_dir'])/'artifacts'
    raw={}
    for name,sha in source['raw_artifact_sha256'].items():
        path=(artifact_root/name).resolve();assert path.is_relative_to(artifact_root.resolve())
        raw[name]=path.read_bytes();assert hashlib.sha256(raw[name]).hexdigest()==sha
    bridge,ba=_source(root,plan['bridge_source'],plan['bridge_source']['artifacts'])
    cash=objects['data/cash_events.pkl'];found=cash.loc[cash.event_id.eq(plan['event_id'])]
    assert len(found)==1
    event=found.iloc[0].to_dict();assert event['instrument']=='SOXL'
    table=pd.read_csv(io.BytesIO(raw['sources/soxl/issuer_soxl_table_observed.csv']),dtype=str,keep_default_na=False)
    issuer=table.loc[pd.to_datetime(table['Ex Date']).dt.strftime('%Y-%m-%d').eq('2023-03-21')];assert len(issuer)==1;row=issuer.iloc[0].to_dict()
    assert row=={'Record Date':'03/22/2023','Ex Date':'03/21/2023','Pay Date':'03/28/2023','Income Dividend':'0.02501','Short-Term Capital Gain':'—','Long-Term Capital Gain':'—'}
    components=[Decimal(0) if row[k]=='—' else Decimal(row[k]) for k in ['Income Dividend','Short-Term Capital Gain','Long-Term Capital Gain']]
    total=sum(components);assert total==Decimal(str(event['original_total_decimal']))==Decimal('0.02501')
    assert Decimal(str(event['original_share_multiplier_decimal']))==1
    for key,column in [('ex_date','Ex Date'),('record_date','Record Date'),('pay_date','Pay Date')]:assert str(pd.Timestamp(event[key]).date())==str(pd.Timestamp(row[column]).date())
    vendor=bridge['analysis/cash_fields'];vendor=vendor.loc[vendor['case'].eq('cash_dividend_multiplicative')&vendor.symbol.eq('SOXL')];assert len(vendor)==1
    assert Decimal(str(vendor.iloc[0].per_cash_div_original))==total
    inventory=json.loads(raw['sources/soxl/issuer_soxl_inventory.json'])
    failed_receipts={k:json.loads(v) for k,v in raw.items() if k.endswith('.receipt.json')}
    assert all(x['status_code']==403 for x in failed_receipts.values())
    contract=objects['inputs/source_contract.pkl'];time_contract=contract['time_contract']
    parent_convention=time_contract.get('pay_date_intraday_availability','')
    inherited='nextnativeopenafterpaydateclose' in parent_convention.replace(' ','').lower()
    calendar=pd.DatetimeIndex(objects['inputs/native_calendar.pkl']);ex=pd.Timestamp(event['ex_date']);pay=pd.Timestamp(event['pay_date'])
    assert ex in calendar and pay in calendar
    previous=calendar[calendar<ex][-1];next_open=calendar[calendar>pay][0].tz_localize('America/New_York')+pd.Timedelta(hours=9,minutes=30)
    assert str(next_open.date())=='2023-03-29'
    source_currency=event.get('currency') or row.get('Currency')
    # The bounded issuer row/native event have no currency field. USD may not be
    # imported from another event, a US ticker or a fee-validation convention.
    assert source_currency is None
    qualified={'event_id':plan['event_id'],'instrument':'SOXL','ex_date':'2023-03-21','record_date':'2023-03-22','pay_date':'2023-03-28','income_per_original_share_decimal':str(components[0]),'short_term_original_token':row['Short-Term Capital Gain'],'long_term_original_token':row['Long-Term Capital Gain'],'gross_cash_per_original_share_decimal':str(total),'original_share_multiplier_decimal':str(event['original_share_multiplier_decimal']),'original_share_basis_evidence_grade':event['basis_evidence_grade'],'source_cash_basis':event['source_cash_basis'],'currency':None,'currency_status':'needs_data_no_explicit_currency_in_bounded_sources','vendor_per_cash_div_decimal':str(Decimal(str(vendor.iloc[0].per_cash_div_original))),'vendor_amount_correspondence':'exact_decimal_corroboration_not_currency_or_date_receipt_proof','source_capture_kind':event['source_capture_kind'],'source_hash':event['source_hash'],'source_version':event['source_version'],'DOM_observation_completed_no_later_than_utc':inventory['browser_observation_completed_before_utc'],'transcription_written_at_utc':inventory['written_at_utc'],'exact_DOM_request_started_at':None,'exact_DOM_received_at':None,'failed_HTTP_receipts':[{'url':v['requested_url'],'status':v['status_code'],'request_started_at_utc':v['request_started_at_utc'],'response_received_at_utc':v['response_received_at_utc'],'role':'failed_HTTP_only_not_DOM_content_or_receipt_authority'} for v in failed_receipts.values()],'announcement_at':event.get('announcement_at'),'source_evidence_known_by_utc':event.get('source_evidence_known_by_utc'),'archive_timestamp_utc':event.get('archive_timestamp_utc'),'historical_2023_received_before_decision_qualified':False,'actual_broker_cash_receipt_verified':False,'amount_and_economic_dates_qualified_under_parent_contract':True}
    convention={'event_id':plan['event_id'],'role':'hypothetical_conditional_gross_paper_convention_not_account_run','convention_origin':'existing_parent_contract' if inherited else 'newly_declared_conservative_paper_assumption','parent_exact_paydate_convention':parent_convention,'eligible_unit_lot_original_shares':100,'eligible_previous_native_session':str(previous.date()),'conditional_entitlement_at':str(ex.tz_localize('America/New_York')+pd.Timedelta(hours=9,minutes=30)),'conditional_gross_receivable_source_units_decimal':str(100*total),'currency':None,'receivable_in_NAV':'only_if_currency_and_entitlement_qualified; no_USD_NAV_calculation_performed','spendable_before_declared_payment_transfer':False,'conditional_paper_available_cash_at':str(next_open),'source_pay_date_intraday_credit_time':None,'cash_transfer_semantics':'Move same receivable to cash once; scheduled date convention, not actual broker receipt','actual_credit_claimed':False,'DRIP':False,'withholding':'not modeled; gross pretax convention only','double_credit_prohibited':'No affine-return plus additional same dividend cash; future cash book must be raw-share plane','historical_decision_time_evidence_created':False,'conditional_convention_defined':True,'unconditional_USD_book_eligible':False,'account_ledger_run':False}
    tracking=card/'qlib_recorder'
    with R.start(experiment_name=plan['experiment'],recorder_name='fixed_existing_cash_event',uri=str(tracking)):
        rec=R.get_recorder();rec.log_params(plan_sha256=digest,fixed_events=1,new_fetches=0,new_accounts=0)
        rec.save_objects(artifact_path='inputs',plan=plan,source_audits={'corporate':audit,'bridge':ba},raw_source_artifacts=raw,selected_native_event=event,selected_issuer_row=row,parent_time_contract=time_contract,vendor_fixed_event=vendor,calendar=calendar)
        rec.save_objects(artifact_path='sources',execution_workflow=Path(__file__).read_text())
        rec.save_objects(artifact_path='analysis',event_qualification=qualified,paper_convention=convention)
        policy={'policy':'slim_card_recorder_first','workflow_config':'workflow.yaml','recorder_only_artifacts':['inputs/*','sources/*']}
        summary={'experiment':plan['experiment'],'plan_sha256':digest,'decision':'fixed_cash_amount_dates_qualified_conditional_paper_convention_currency_PIT_unqualified','objective_achieved':False,'fixed_events':1,'native_parent_objects':3,'raw_parent_artifacts':4,'bridge_objects':1,'new_market_reads':0,'new_source_fetches':0,'new_fits':0,'new_replays':0,'new_accounts':0,'observer_launches':0,'production_changes':0,'amount_original_share_qualified_under_parent_contract':True,'economic_ex_record_pay_dates_qualified':True,'currency_qualified':False,'paper_convention_defined_conditionally':True,'paper_convention_inherited_from_parent':inherited,'paper_available_cash_date_conditional':'2023-03-29','historical_PIT_qualified':False,'actual_broker_credit_verified':False,'unconditional_USD_cash_NAV_qualified':False,'next_action':'Separatelyfreeze minimal primaryissuer currencyproof lookup/intake for THIS event or explicitresearchcurrencyassumption; no newfetch inthisroute. Historicalfirstreceipt remainsunqualified and cannotbe backfilled. No observer/account launch.', 'recorder_id':rec.id,'experiment_id':str(rec.experiment_id),'recorder_uri':str(tracking),'recorder_local_dir':str(tracking/str(rec.experiment_id)/rec.id),'provider_uri':str(root/plan['provider_uri']),'artifact_policy':policy,'qlib_docs':plan['qlib_docs']}
        rec.save_objects(artifact_path='analysis',summary=summary);artifacts=Path(summary['recorder_local_dir'])/'artifacts';rec.log_metrics(registered_artifact_count=sum(p.is_file() for p in artifacts.rglob('*')),qualified_fixed_cash_events=1)
    atomic_json(qualified,card/'event_qualification.json');atomic_json(convention,card/'paper_convention.json')
    atomic_json({'recorder_id':rec.id,'entries':[{'path':str(p.relative_to(card)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in sorted(artifacts.rglob('*')) if p.is_file()]},card/'verification_manifest.json')
    binding=json.loads((root/plan['bridge_source']['card']/'nas_binding.json').read_text());binding.pop('evidence');atomic_json(binding,card/'nas_binding.json')
    (card/'workflow.yaml').write_text(yaml.safe_dump({'experiment':plan['experiment'],'provider_uri':summary['provider_uri'],'recorder_uri':str(tracking),'artifact_policy':policy},sort_keys=False))
    summary['nas_evidence']=archive_nas_recorder(card/'nas_binding.json',workspace_root=root,card=card,recorder_id=rec.id,experiment_id=str(rec.experiment_id));atomic_json(summary,card/'summary.json');return summary
