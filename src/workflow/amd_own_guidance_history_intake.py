"""Qualify the bounded saved AMD source intake; no automatic further downloads."""
import hashlib,json,re
from pathlib import Path
import pandas as pd
from bs4 import BeautifulSoup
from src.data.amd_guidance_history_intake import report_facts
from src.workflow.minute_action_value import _source
from src.data.minute_intake import atomic_json
from src.data.nas_market_data import archive_nas_recorder


def run(*,root,card,plan):
    from qlib.data import D
    from qlib.workflow import R
    import yaml
    if (card/'summary.json').exists():return json.loads((card/'summary.json').read_text())
    digest=hashlib.sha256((card/'plan.json').read_bytes()).hexdigest();assert digest=='5e419a90afbb7daa21e372f1f5ffd4b31cb2e97e78bc2456fda31781c7a45f07'
    parent,audit=_source(root,plan['parent_source'],plan['parent_source']['artifacts'])
    directory=root/'data/raw/external'/plan['experiment'];ledger=json.loads((directory/'operation_ledger.json').read_text());docs=json.loads((directory/'document_intake.json').read_text())
    assert ledger['plan_sha256']==digest
    ops=ledger['operations'];urls=[o['url'] for o in ops if o.get('url')];counts=pd.Series([o['category'] for o in ops]).value_counts().to_dict()
    assert len(ops)<=140 and len(set(urls))<=85 and counts.get('discovery_query',0)<=28
    assert max(pd.Series(urls).value_counts())<=2
    limits={'financial_index':6,'all_index':7,'sec_metadata':4,'sec_index':28,'regular':28,'update':8,'bridge':4}
    for k,limit in limits.items():assert counts.get(k,0)<=limit
    payloads={p.name:p.read_bytes() for p in directory.iterdir() if p.is_file() and p.suffix in ['.bin','.html','.json'] and p.name!='extraction_preview.json'}
    for op in ops:
        if op.get('payload'):assert hashlib.sha256(payloads[op['payload']]).hexdigest()==op['sha256']
    assert hashlib.sha256(payloads['existing_2011_index.html']).hexdigest()==plan['existing_2011_probe']['sha256']
    calendar=pd.DatetimeIndex(D.calendar(start_time='2011-01-01',end_time=plan['universe']['as_of'],freq='day'))
    def available(day):
        after=calendar[calendar>pd.Timestamp(day).normalize()]
        return after[0].tz_localize('America/New_York')+pd.Timedelta(hours=9,minutes=31) if len(after) else pd.NaT
    facts=[];timeline=[];scope=[];status=[]
    for e in docs:
        receipt=e['receipt'];raw=payloads.get(receipt.get('payload'));day=pd.Timestamp(e['report_date']);ready=available(day)
        timeline.append({'reported_quarter':e['reported_quarter'],'report_date':e['report_date'],'role':e['role'],'published_clock_raw':e.get('published_clock_raw'),'available_at':ready,'source_url':e['url'],'source_sha256':receipt.get('sha256'),'retrieval_status':receipt['status'],'all_updates_coverage_verified':False})
        if e['role']!='regular':continue
        row={'reported_quarter':e['reported_quarter'],'source_url':e['url'],'status':'needs_data','error':None}
        try:
            if receipt['status']!='captured':raise ValueError('original report retrieval incomplete')
            f=report_facts(raw,e);f['available_at']=ready;f['date_bound']=day;facts.append(f);row['status']='original_regular_report_extracted'
            text=f['full_text']
            for term in ['SeaMicro','discontinued operations','deconsolidation','joint venture','excludes','excluding']:
                parts=[text[max(0,m.start()-130):m.end()+360] for m in list(re.finditer(re.escape(term),text,re.I))[:4]]
                if parts:scope.append({'reported_quarter':f['reported_quarter'],'source_url':f['source_url'],'source_sha256':f['source_sha256'],'term':term,'excerpt':'\n'.join(parts)})
        except ValueError as error:row['error']=str(error)
        status.append(row)
    requested={f'FY{p.year}Q{p.quarter}' for p in pd.period_range('2010Q4','2017Q3',freq='Q')}
    for q in requested-{x['reported_quarter'] for x in status}:status.append({'reported_quarter':q,'status':'needs_data_original_document','error':'not acquired within frozen budget'})
    assert len(status)==28
    facts=pd.DataFrame(facts).sort_values('report_date');timeline=pd.DataFrame(timeline);scope=pd.DataFrame(scope);status=pd.DataFrame(status).sort_values('reported_quarter')
    current=parent['analysis/original_report_facts'].set_index('reported_quarter');old=facts.set_index('reported_quarter');guides=facts.set_index('target_quarter')
    rows=[]
    for p in pd.period_range('2011Q1','2017Q4',freq='Q'):
        target=f'FY{p.year}Q{p.quarter}';row={'target_quarter':target,'qualification':'needs_data_original_guide','guide_available_at':pd.NaT,'actual_report_available_at':pd.NaT,'pair_available_at':pd.NaT,'measurement_known':False,'freshness':'unknown_update_coverage','all_updates_coverage_verified':False,'scope_reason':'original guide not obtained'}
        actual=old.loc[target] if target in old.index else current.loc[target] if target in current.index else None
        if actual is not None:row.update(actual_report_date=actual.report_date,actual_report_available_at=actual.available_at,actual_revenue_million=actual.actual_revenue_million,actual_rounding_half_million=actual.actual_rounding_half_million,actual_source_url=actual.source_url,actual_source_sha256=actual.source_sha256)
        if target in guides.index:
            g=guides.loc[target];row.update(guide_reported_quarter=g.reported_quarter,guide_report_date=g.report_date,guide_available_at=g.available_at,guide_mid=g.guide_mid,guide_lower=g.guide_lower,guide_upper=g.guide_upper,guide_unit=g.guide_unit,guide_status=g.guide_status,guide_base_actual_million=g.guide_base_actual_million,guide_base_rounding_half_million=g.guide_base_rounding_half_million,guide_source_url=g.source_url,guide_source_sha256=g.source_sha256,guide_excerpt=g.guide_excerpt)
            if actual is None:row.update(qualification='needs_data_role_clock_unit_or_pair',scope_reason='target original regular actual absent')
            else:
                assert pd.Timestamp(actual.report_date)>pd.Timestamp(g.report_date)
                row['pair_available_at']=max(actual.available_at,g.available_at)
                if g.guide_status=='original_numeric_sequential_percent':row.update(qualification='comparable_original_regular_report_pair_under_delayed_assumption',measurement_known=True,scope_reason='OriginalcompanyGAAPtotalquarterlyactual and explicit signedsequentialguide; originalsame-report printedbase/precision retained, no restatedcomparative.')
                else:row.update(qualification='needs_data_role_clock_unit_or_pair',scope_reason='Original guide qualitative flat-to-slightly-down; no numeric midpoint assigned.')
                if target=='FY2012Q1':row.update(qualification='blocked_scope_mismatch_or_missing_bridge',measurement_known=False,scope_reason='SeaMicro acquisition completed in targetquarter; priororiginalguide does not establish acquisitionperimeter and no originalsame-scopeGAAPbridge retained. No subtraction.')
                if target=='FY2015Q1':row['scope_reason']='Originalreport saysSeaMicroexit effectiveimmediately atreport; currentquarteractual and originalpriorguide companytotal retained. Impairment charge isexpense, notrevenueaddback.'
                if target=='FY2016Q2':row['scope_reason']='ATMPJV closing anticipated in originalguide report; targetreport separatelyattributes$150m disposal gain to netincome, notcurrentquarterrevenue. No gain added/subtracted fromrevenue.'
                updates=timeline.loc[timeline.role.eq('update')&timeline.reported_quarter.eq(target)]
                updates=updates.loc[updates.report_date.ge(g.report_date)&updates.report_date.le(actual.report_date)]
                row['known_intervening_dates']=';'.join(updates.report_date)
                if len(updates):row['freshness']='known_preannounced_or_updated'
        row['realization_event_known']=pd.notna(row['actual_report_available_at']);row['percentage_conversion_supported']=row['measurement_known']
        rows.append(row)
    new=pd.DataFrame(rows)
    combined=pd.concat([new,parent['analysis/pairs'].loc[~parent['analysis/pairs'].target_quarter.eq('FY2017Q4')]],ignore_index=True)
    assert len(combined)==63 and combined.target_quarter.nunique()==63
    original_blocks=parent['analysis/pairs'].loc[parent['analysis/pairs'].qualification.eq('blocked_scope_mismatch_or_missing_bridge'),'target_quarter']
    assert combined.set_index('target_quarter').loc[original_blocks,'measurement_known'].eq(False).all()
    tracking=card/'qlib_recorder'
    with R.start(experiment_name=plan['experiment'],recorder_name='bounded_AMD_history_only',uri=str(tracking)):
        rec=R.get_recorder();rec.log_params(plan_sha256=digest,new_fits=0,new_accounts=0)
        rec.save_objects(artifact_path='inputs',plan=plan,parent_audit=audit,capture_payloads=payloads,operation_ledger=ledger,documents=docs,calendar=calendar)
        rec.save_objects(artifact_path='sources',execution_workflow=Path(__file__).read_text(),collector_parser=(root/'src/data/amd_guidance_history_intake.py').read_text())
        rec.save_objects(artifact_path='analysis',pairs=combined,new_pairs=new,original_report_facts=facts,report_status=status,timeline=timeline,scope_evidence=scope)
        policy={'policy':'slim_card_recorder_first','workflow_config':'workflow.yaml','recorder_only_artifacts':['inputs/*','sources/*','analysis/original_report_facts','analysis/scope_evidence']}
        summary={'experiment':plan['experiment'],'plan_sha256':digest,'decision':'bounded_AMD_history_qualified_with_explicit_gaps','objective_achieved':False,'new_fits':0,'new_accounts':0,'new_error_features':0,'market_outcome_reads':0,'network_operations':len(ops),'new_URLs':len(set(urls)),'operation_category_counts':counts,'report_status_counts':status.status.value_counts().to_dict(),'guide_status_counts':facts.guide_status.value_counts().to_dict(),'combined_target_rows':len(combined),'combined_status_counts':combined.qualification.value_counts().to_dict(),'pre2023_comparable_events':int(((combined.actual_report_available_at<pd.Timestamp('2023-01-01',tz='America/New_York'))&combined.measurement_known).sum()),'known_preannounced_combined_events':int(combined.freshness.eq('known_preannounced_or_updated').sum()),'all_updates_coverage_verified':False,'first_public_timestamp_verified':False,'historical_original_bytes_certified':False,'next_action':'Independent source review then separate frozen AMDincrement design; sourcecompletion doesnotauthorizefits. No more retrievaltofill qualitative/scopemissing.', 'provider_uri':str(root/plan['provider_uri']),'region':'us','as_of':plan['universe']['as_of'],'recorder_id':rec.id,'experiment_id':str(rec.experiment_id),'recorder_uri':str(tracking),'recorder_local_dir':str(tracking/str(rec.experiment_id)/rec.id),'artifact_policy':policy}
        rec.save_objects(artifact_path='analysis',summary=summary);artifacts=Path(summary['recorder_local_dir'])/'artifacts';rec.log_metrics(registered_artifact_count=sum(p.is_file() for p in artifacts.rglob('*')),network_operations=len(ops))
    for name,data in [('pair_qualification',combined),('report_status',status),('disclosure_timeline',timeline),('scope_evidence',scope)]:data.to_csv(card/(name+'.csv'),index=False)
    atomic_json(ledger,card/'operation_ledger.json')
    atomic_json({'recorder_id':rec.id,'entries':[{'path':str(p.relative_to(card)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in sorted(artifacts.rglob('*')) if p.is_file()]},card/'verification_manifest.json')
    binding=json.loads((root/plan['parent_source']['card']/'nas_binding.json').read_text());binding.pop('evidence');atomic_json(binding,card/'nas_binding.json')
    (card/'workflow.yaml').write_text(yaml.safe_dump({'experiment':plan['experiment'],'provider_uri':summary['provider_uri'],'recorder_uri':str(tracking),'artifact_policy':policy},sort_keys=False))
    summary['nas_evidence']=archive_nas_recorder(card/'nas_binding.json',workspace_root=root,card=card,recorder_id=rec.id,experiment_id=str(rec.experiment_id));atomic_json(summary,card/'summary.json');return summary
