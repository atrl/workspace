"""Bounded existing AMD issuer corpus qualification, without forecast-error features."""
import hashlib,json,re
from pathlib import Path
import pandas as pd
from bs4 import BeautifulSoup
from src.workflow.minute_action_value import _source
from src.data.minute_intake import atomic_json
from src.data.nas_market_data import archive_nas_recorder


def extract_guide(text,reported):
    match=re.search(r'Current Outlook',text,re.I)
    if not match:return {'guide_status':'needs_data_outlook_section','excerpt':''}
    tail=text[match.start():]
    ending=re.search(r'AMD Teleconference|Cautionary Statement',tail[20:],re.I)
    # Cautionary Statement is mentioned inside the introductory outlook sentence;
    # only the actual Teleconference section is a reliable ending here.
    ending=re.search(r'AMD Teleconference',tail,re.I)
    section=tail[:ending.start()] if ending else tail[:2500]
    pattern=r'For\s+(?:the\s+(first|second|third|fourth)\s+quarter\s+(?:of\s+)?|Q([1-4])\s+)(20\d{2}),?\s+AMD expects revenue to be approximately\s*\$([\d.]+)\s*(billion|million),?\s+plus or minus\s*\$([\d.]+)\s*(billion|million)'
    m=re.search(pattern,section,re.I)
    if not m:return {'guide_status':'needs_data_numeric_quarter_guide','excerpt':section}
    q=int(m[2]) if m[2] else ['first','second','third','fourth'].index(m[1].lower())+1
    target=f'FY{m[3]}Q{q}';year,rq=int(reported[2:6]),int(reported[-1]);expected=f'FY{year+(rq==4)}Q{rq%4+1}'
    if target!=expected:raise ValueError('next-quarter identity differs')
    mid=float(m[4])*(1000 if m[5].lower()=='billion' else 1);width=float(m[6])*(1000 if m[7].lower()=='billion' else 1)
    return {'guide_status':'original_next_quarter_amount','target_quarter':target,'guide_mid_million':mid,'guide_lower_million':mid-width,'guide_upper_million':mid+width,'guide_unit':'USD_million','original_midpoint_text':m[4]+' '+m[5],'original_tolerance_text':m[6]+' '+m[7],'bounds_derivation':'published midpoint +/- published dollar tolerance; no error calculation','excerpt':section}


def qualify(objects,plan,calendar):
    manifest=objects['sources/manifest'];html=objects['sources/html']
    entries=[e for e in manifest['entries'] if e['symbol']=='AMD'];regular=[e for e in entries if e['event_role']=='quarterly'];prelim=[e for e in entries if e['event_role']=='preliminary']
    assert len(regular)==35 and len(prelim)<=1
    expected={(e['report_date'],e['expected_year'],e['expected_quarter']) for e in plan['universe']['regular_reports']}
    assert {(e['report_date'],e['expected_year'],e['expected_quarter']) for e in regular}==expected
    def available(day):
        dates=calendar[calendar>pd.Timestamp(day).normalize()]
        return dates[0].tz_localize('America/New_York')+pd.Timedelta(hours=9,minutes=31) if len(dates) else pd.NaT
    facts=[];timeline=[];evidence=[];selected_html={}
    for e in entries:
        raw=html[e['path']];assert hashlib.sha256(raw).hexdigest()==e['sha256'];selected_html[e['path']]=raw
        text=BeautifulSoup(raw,'html.parser').get_text(' ',strip=True)
        dates=[pd.Timestamp(e['report_date'])]
        if e.get('published_clock_raw'):dates.append(pd.Timestamp(e['published_clock_raw']).tz_localize(None).normalize())
        bound=max(dates);ready=available(bound)
        timeline.append({'report_date':e['report_date'],'role':e['event_role'],'source_url':e['url'],'source_sha256':e['sha256'],'published_clock_raw':e.get('published_clock_raw'),'published_clock_display':e.get('published_clock_display'),'date_bound':bound,'available_at':ready,'full_update_coverage':False})
        if e['event_role']!='quarterly':
            evidence.append({'target_quarter':'FY2022Q3','kind':'preliminary_do_not_replace_regular','report_date':e['report_date'],'source_url':e['url'],'source_sha256':e['sha256'],'excerpt':text[:5500]});continue
        quarter=f"FY{e['expected_year']}Q{e['expected_quarter']}";guide=extract_guide(text,quarter)
        actual=e['parsed']['basis']['gaap'];f={'reported_quarter':quarter,'report_date':e['report_date'],'available_at':ready,'date_bound':bound,'published_clock_raw':e.get('published_clock_raw'),'source_url':e['url'],'source_sha256':e['sha256'],'actual_details':json.dumps(actual,sort_keys=True),'actual_revenue_million':actual['revenue_million_usd'],'actual_rounding_half_million':actual['revenue_rounding_half_million'],**guide}
        facts.append(f)
        if quarter in ['FY2017Q4','FY2018Q1','FY2021Q4','FY2022Q1','FY2022Q2','FY2025Q1','FY2025Q2','FY2025Q3','FY2025Q4','FY2026Q1','FY2026Q2']:
            for term in ['ASC 606','Xilinx','Pensando','ZT Systems','discontinued operations','MI308']:
                snippets=[text[max(0,m.start()-140):m.end()+380] for m in list(re.finditer(re.escape(term),text,re.I))[:4]]
                if snippets:evidence.append({'target_quarter':quarter,'kind':'scope_observation_not_automatic_bridge','report_date':e['report_date'],'source_url':e['url'],'source_sha256':e['sha256'],'excerpt':'\n'.join(snippets),'term':term})
    facts=pd.DataFrame(facts).sort_values('report_date');actuals=facts.set_index('reported_quarter');guides=facts.loc[facts.guide_status.eq('original_next_quarter_amount')].set_index('target_quarter')
    assert not guides.index.duplicated().any()
    blocks={'FY2022Q1':'Original guide does not explicitly establish Xilinx inclusion/exclusion; targetactual includespartialXilinx. OriginalexXilinx5.3bn is separatelylabellednonGAAP; retained evidence only, no inferredsameGAAPbridge.',
        'FY2022Q2':'Original guide namesXilinx but Pensando pendingacquisition scope unspecified; targetreport statesPensandocompleted/included. No explicit comparabletotalbridge retained.',
        'FY2025Q1':'ZT acquisition announcedcompleted by originalregularreport but exactcompletion-versusquarterend/perimeter not established in boundedcapturedreport; no inferreddatefromcurrentlink.',
        'FY2025Q2':'Guide issuedbefore newdiscontinuedoperations classification; targetreport classifiesZTmanufacturingheldforsale/discontinued. Originalguide perimeter/explicitmatchedrevenuebridge not established.'}
    rows=[]
    for period in pd.period_range('2017Q4','2026Q3',freq='Q'):
        target=f'FY{period.year}Q{period.quarter}';row={'target_quarter':target,'qualification':'needs_data_original_guide','guide_available_at':pd.NaT,'actual_report_available_at':pd.NaT,'pair_available_at':pd.NaT,'measurement_known':False,'freshness':'unknown_update_coverage','all_updates_coverage_verified':False,'scope_reason':'original precedingquarter guide absent fromexistingcorpus'}
        if target in actuals.index:
            a=actuals.loc[target];row.update(actual_report_date=a.report_date,actual_report_available_at=a.available_at,actual_revenue_million=a.actual_revenue_million,actual_rounding_half_million=a.actual_rounding_half_million,actual_source_url=a.source_url,actual_source_sha256=a.source_sha256,actual_details=a.actual_details)
        if target in guides.index:
            g=guides.loc[target];row.update(guide_reported_quarter=g.reported_quarter,guide_report_date=g.report_date,guide_available_at=g.available_at,guide_mid_million=g.guide_mid_million,guide_lower_million=g.guide_lower_million,guide_upper_million=g.guide_upper_million,guide_unit=g.guide_unit,guide_source_url=g.source_url,guide_source_sha256=g.source_sha256,guide_excerpt=g.excerpt)
            if target not in actuals.index:row.update(qualification='not_yet_reported_target',scope_reason='guide known but targetregularreport absent asof')
            else:
                assert pd.Timestamp(a.report_date)>pd.Timestamp(g.report_date)
                row.update(pair_available_at=max(a.available_at,g.available_at),qualification='comparable_original_regular_report_pair_under_delayed_assumption',measurement_known=True,scope_reason='originalcompanytotalquarterlyguide vsoriginalsamequarterregularGAAPtotal; no latercomparative')
                if target in blocks:row.update(qualification='blocked_scope_mismatch_or_missing_bridge',measurement_known=False,scope_reason=blocks[target])
                if target=='FY2018Q1':row['scope_reason']='Originalguide explicitlyASC606; originaltargetreportcurrentquarter followsASC606. No restated2017comparative used.'
                if target in ['FY2025Q3','FY2025Q4','FY2026Q1','FY2026Q2']:row['scope_reason']='Priororiginalreport/currenttarget bothafterZTdiscontinuedclassification; usecurrentquartertotalnotdiscontinuednetincome. MI308anticipatedsales/no-sales isforecastassumption notmeasurementaddback.'
                if target=='FY2022Q3':row['freshness']='known_preannounced_or_updated';row['known_preannouncement_date']='2022-10-06'
        row['realization_event_known']=pd.notna(row['actual_report_available_at'])
        rows.append(row)
    pairs=pd.DataFrame(rows);assert len(pairs)==36 and pairs.target_quarter.nunique()==36
    assert pairs.loc[pairs.target_quarter.eq('FY2026Q3'),'pair_available_at'].isna().all()
    indexes=[e for e in manifest['indexes'] if 'ir.amd.com/' in e['url']];assert len(indexes)<=10
    for e in indexes:
        raw=html[e['path']];assert hashlib.sha256(raw).hexdigest()==e['sha256'];selected_html[e['path']]=raw
    return pairs,facts,pd.DataFrame(timeline),pd.DataFrame(evidence),selected_html,{'entries':entries,'indexes':indexes}


def run(*,root,card,plan):
    from qlib.data import D
    from qlib.workflow import R
    import yaml
    if (card/'summary.json').exists():return json.loads((card/'summary.json').read_text())
    digest=hashlib.sha256((card/'plan.json').read_bytes()).hexdigest();assert digest=='60401a091aa770c8459f663c66dc4f57996c8362a839b136063843e0742e9044'
    objects,audit=_source(root,plan['source'],plan['source']['artifacts']);calendar=pd.DatetimeIndex(D.calendar(start_time='2018-01-01',end_time=plan['universe']['as_of'],freq='day'))
    pairs,facts,timeline,evidence,raw,manifest=qualify(objects,plan,calendar)
    tracking=card/'qlib_recorder'
    with R.start(experiment_name=plan['experiment'],recorder_name='existing_AMD_only',uri=str(tracking)):
        rec=R.get_recorder();rec.log_params(plan_sha256=digest,new_fetches=0,new_fits=0,new_accounts=0)
        rec.save_objects(artifact_path='inputs',plan=plan,source_audit=audit,amd_html=raw,amd_manifest=manifest,calendar=calendar)
        rec.save_objects(artifact_path='sources',execution_workflow=Path(__file__).read_text())
        rec.save_objects(artifact_path='analysis',pairs=pairs,original_report_facts=facts,timeline=timeline,scope_evidence=evidence)
        policy={'policy':'slim_card_recorder_first','workflow_config':'workflow.yaml','recorder_only_artifacts':['inputs/*','sources/*','analysis/original_report_facts','analysis/scope_evidence']}
        summary={'experiment':plan['experiment'],'plan_sha256':digest,'decision':'existing_AMD_subset_qualified_with_explicit_scope_gaps','objective_achieved':False,'new_fetches':0,'new_fits':0,'new_accounts':0,'new_error_features':0,'market_outcome_reads':0,'target_rows':len(pairs),'regular_reports':len(facts),'status_counts':pairs.qualification.value_counts().to_dict(),'guide_status_counts':facts.guide_status.value_counts().to_dict(),'pre2023_comparable_events':int(((pairs.actual_report_available_at<pd.Timestamp('2023-01-01',tz='America/New_York'))&pairs.measurement_known).sum()),'overall_intended_coverage':plan['overall_intended_coverage'],'all_updates_coverage_verified':False,'first_public_timestamp_verified':False,'historical_original_bytes_certified':False,'next_action':'Independentreview; separatelydecide bounded2011-2017historyrepair, notautomaticmodel fitting. No scopegap fetch permitted inthisrun.','provider_uri':str(root/plan['provider_uri']),'region':'us','as_of':plan['universe']['as_of'],'recorder_id':rec.id,'experiment_id':str(rec.experiment_id),'recorder_uri':str(tracking),'recorder_local_dir':str(tracking/str(rec.experiment_id)/rec.id),'artifact_policy':policy,'qlib_docs':plan['qlib_docs']}
        rec.save_objects(artifact_path='analysis',summary=summary);artifacts=Path(summary['recorder_local_dir'])/'artifacts';rec.log_metrics(registered_artifact_count=sum(p.is_file() for p in artifacts.rglob('*')),qualified_targets=int(pairs.measurement_known.sum()))
    pairs.to_csv(card/'pair_qualification.csv',index=False);timeline.to_csv(card/'disclosure_timeline.csv',index=False);evidence.to_csv(card/'scope_evidence.csv',index=False)
    atomic_json({'recorder_id':rec.id,'entries':[{'path':str(p.relative_to(card)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in sorted(artifacts.rglob('*')) if p.is_file()]},card/'verification_manifest.json')
    binding=json.loads((root/plan['source']['card']/'nas_binding.json').read_text());binding.pop('evidence');atomic_json(binding,card/'nas_binding.json')
    (card/'workflow.yaml').write_text(yaml.safe_dump({'experiment':plan['experiment'],'provider_uri':summary['provider_uri'],'recorder_uri':str(tracking),'artifact_policy':policy},sort_keys=False))
    summary['nas_evidence']=archive_nas_recorder(card/'nas_binding.json',workspace_root=root,card=card,recorder_id=rec.id,experiment_id=str(rec.experiment_id));atomic_json(summary,card/'summary.json');return summary
