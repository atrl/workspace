"""Existing-corpus qualification of delayed own-guidance realization; no error features."""
import hashlib
import json
from pathlib import Path
import pandas as pd
from src.core.recorder_sources import load_verified_recorder_objects
from src.data.minute_intake import atomic_json
from src.data.nas_market_data import archive_nas_recorder
from src.workflow.minute_action_value import _source


def qualify(events, guides, facts, texts, metadata, calendar):
    """Keep guide scope separate from the actual document's *next* guide scope."""
    raw = {r['accession']: r for r in facts}
    assert len(raw) == len(events) == 66 and len(guides) == 63
    events = events.drop(columns=['guidance_growth'], errors='ignore').copy()
    guides = guides.drop(columns=['guidance_growth'], errors='ignore').copy()
    def available(date):
        after = calendar[calendar > pd.Timestamp(date).normalize()]
        return after[0].tz_localize('America/New_York') + pd.Timedelta(hours=9, minutes=31) if len(after) else pd.NaT
    def actual_detail(r):
        return r.get('actual_revenue', {'value':r.get('actual_revenue_million'), 'unit':'USD_million',
            'original_value':r.get('actual_revenue_original_value'), 'original_unit':r.get('actual_revenue_original_unit'),
            'precision_million':r.get('actual_revenue_precision_million'), 'basis':r.get('actual_revenue_basis'),
            'status':r.get('actual_revenue_status')})
    evidence=[]
    for quarter, patterns in {
        'FY2020Q4':['pending acquisition of Mellanox','Jan. 26, 2020'],
        'FY2021Q1':['April 26, 2020','April 27, 2020','first day of the second quarter'],
        'FY2021Q2':['contributed 14 percent of revenue','July 26, 2020'],
        'FY2026Q1':['H20','Revenue is expected']}.items():
        payload=texts[quarter+'.txt']; text=payload.decode() if isinstance(payload,bytes) else payload
        for pattern in patterns:
            lines=[s for s in text.splitlines() if pattern.lower() in s.lower()]
            assert lines, (quarter,pattern)
            evidence.append({'quarter':quarter,'accession':metadata[quarter]['accession'],
                'source_url':metadata[quarter].get('source_url') or metadata[quarter]['sources'][0]['url'],'capture_sha256':hashlib.sha256(text.encode()).hexdigest(),
                'pattern':pattern,'verbatim_lines':'\n'.join(lines)})
    timeline=[]
    for _, e in events.iterrows():
        r=raw[e.accession]
        role='original_regular_report' if pd.notna(e.actual_revenue_million) else 'preliminary_or_guidance_update'
        timeline.append({**e.to_dict(),'observation_role':role,'event_subtype':r.get('event_subtype'),
            'preliminary_revenue_million':r.get('preliminary_revenue_million'),
            'source_actual_details':json.dumps(actual_detail(r),sort_keys=True),
            'available_at':available(e.date_bound),'all_updates_coverage_verified':False})
    timeline=pd.DataFrame(timeline)
    pairs=[]
    regular=events.loc[events.actual_revenue_million.notna()]
    for _, g in guides.iterrows():
        gr=raw[g.accession]; target=g.target_quarter
        candidates=regular.loc[regular.reported_quarter.eq(target)]
        assert len(candidates)<=1, target
        row={'target_quarter':target,'guide_accession':g.accession,'guide_reported_quarter':g.reported_quarter,
            'guide_release_date':g.release_date,'guide_date_bound':g.date_bound,'guide_available_at':available(g.date_bound),
            'guide_api_accepted_raw':g.api_accepted_raw,'guide_index_accepted_raw':g.index_accepted_raw,
            'guide_lower':g.guidance_lower,'guide_mid':g.guidance_mid,'guide_upper':g.guidance_upper,
            'guide_unit':g.guidance_unit,'guide_scope':g.guidance_scope,
            'guide_direct_total_revenue_comparison_qualified':gr.get('direct_total_revenue_comparison_qualified'),
            'guide_original_details':json.dumps(gr.get('guidance', {k:v for k,v in gr.items() if k.startswith('guidance_') or k=='bounds_derivation'}),sort_keys=True),
            'guide_base_actual_million':g.actual_revenue_million,'guide_base_actual_details':json.dumps(actual_detail(gr),sort_keys=True),
            'guide_source_urls':g.source_urls,'original_html_bytes_verified':False,'first_public_timestamp_verified':False,
            'all_updates_coverage_verified':False,'actual_accession':None,'actual_report_available_at':pd.NaT,
            'pair_available_at':pd.NaT,'actual_revenue_million':None,
            'qualification':'not_yet_reported_target','freshness':'unknown_update_coverage',
            'scope_reason':'target has no regular report in frozen as-of corpus','known_intervening_accessions':''}
        if len(candidates):
            a=candidates.iloc[0]; ar=raw[a.accession]
            assert a.release_date>g.release_date and a.reported_quarter==target
            detail=actual_detail(ar)
            assert detail['value']==a.actual_revenue_million and detail['unit'] in ('million','USD_million')
            if target in metadata:
                assert metadata[target]['accession']==a.accession
                assert abs(actual_detail(metadata[target])['value']-a.actual_revenue_million)<1e-8
            intervening=events.loc[(events.release_date>=g.release_date)&(events.release_date<=a.release_date)&
                (events.target_quarter.eq(target)|events.reported_quarter.eq(target))&~events.accession.isin([g.accession,a.accession])]
            row.update(actual_accession=a.accession,actual_report_release_date=a.release_date,
                actual_report_date_bound=a.date_bound,actual_report_available_at=available(a.date_bound),
                pair_available_at=available(max(g.date_bound,a.date_bound)),actual_revenue_million=a.actual_revenue_million,
                actual_original_details=json.dumps(detail,sort_keys=True),actual_source_urls=a.source_urls,
                actual_api_accepted_raw=a.api_accepted_raw,actual_index_accepted_raw=a.index_accepted_raw,
                actual_document_next_guide_comparison_flag=ar.get('direct_total_revenue_comparison_qualified'),
                qualification='comparable_original_regular_report_pair_under_delayed_assumption',
                scope_reason='original company-total guide and original regular-report consolidated current-quarter revenue',
                known_intervening_accessions=';'.join(intervening.accession),
                freshness='known_preannounced_or_updated' if len(intervening) else 'unknown_update_coverage')
            if 'Icera' in str(g.guidance_scope):
                assert gr.get('direct_total_revenue_comparison_qualified') is False
                row.update(qualification='blocked_scope_mismatch_or_missing_bridge',scope_reason='Guide excludes Icera; original consolidated actual has no retained same-scope bridge. No subtraction/estimate.')
            elif 'Mellanox' in str(g.guidance_scope):
                if target=='FY2020Q4':
                    row['scope_reason']='Original FY2020Q4 report says Mellanox acquisition still pending; actual quarter ended Jan26,2020. Contemporaneous pending statement supports excluded-target match.'
                elif target=='FY2021Q1':
                    row['scope_reason']='Original regular report quarter ended Apr26,2020; acquisition closed Apr27,2020 first day Q2. Original Q1 guide excluded pending Mellanox; Q1 actual precedes acquisition.'
                elif target=='FY2021Q2':
                    row['scope_reason']='Original guide includes Mellanox; original Q2 report says first quarter included, contributing14% revenue. No acquisition subtraction.'
                else:raise ValueError('Unreviewed Mellanox scope')
            if target=='FY2026Q2':
                row['scope_reason']='Original outlook H20 anticipated lost-sales assumption retained; both guide and actual company-total revenue, no addback of $8bn.'
            if pd.isna(row['pair_available_at']):row['qualification']='needs_data_role_clock_unit_or_pair'
        row['percentage_conversion_supported']=bool(g.guidance_unit=='pct_sequential_revenue_growth' and pd.notna(g.actual_revenue_million) and row['qualification'].startswith('comparable'))
        row['realization_event_known']=pd.notna(row['actual_report_available_at'])
        row['measurement_known']=row['qualification'].startswith('comparable')
        pairs.append(row)
    pairs=pd.DataFrame(pairs)
    assert pairs.target_quarter.nunique()==63
    assert pairs.loc[pairs.target_quarter.eq('FY2027Q3'),'pair_available_at'].isna().all()
    assert pairs.loc[pairs.target_quarter.eq('FY2012Q2'),'qualification'].eq('blocked_scope_mismatch_or_missing_bridge').all()
    assert pairs.loc[pairs.target_quarter.eq('FY2012Q1'),'measurement_known'].all()
    return pairs,timeline,pd.DataFrame(evidence)


def run(*,root,card,plan):
    from qlib.data import D
    from qlib.workflow import R
    import yaml
    if (card/'summary.json').exists():return json.loads((card/'summary.json').read_text())
    digest=hashlib.sha256((card/'plan.json').read_bytes()).hexdigest()
    assert digest=='130fbe46bb32dbd7475bce70fe89d7c741316d47b35c0951a48a8f7d75ee7165'
    s=plan['sources']['guidance_parent']
    primary,audit=load_verified_recorder_objects(root/s['card'],expected_recorder_id=s['recorder_id'],expected_manifest_sha256=s['inventory_sha256'],artifacts=s['artifacts'])
    s=plan['sources']['secondary_existing_sec_text'];secondary,secondary_audit=_source(root,s,s['artifacts'])
    manifest_bytes=(root/plan['sources']['fact_manifest']['path']).read_bytes()
    assert hashlib.sha256(manifest_bytes).hexdigest()==plan['sources']['fact_manifest']['sha256']
    factbytes={}
    for e in json.loads(manifest_bytes)['entries']:
        b=(root/e['path']).read_bytes();assert hashlib.sha256(b).hexdigest()==e['sha256'];factbytes[e['accession']]=b
    facts=primary['source/raw_facts.pkl'];assert {r['accession']:r for r in facts}=={k:json.loads(v) for k,v in factbytes.items()}
    texts=secondary['inputs/web_text_bytes'];metadata=json.loads(secondary['inputs/metadata_bytes']);assert len(texts)==len(metadata)==48
    calendar=pd.DatetimeIndex(D.calendar(start_time='2011-01-01',end_time=plan['universe']['as_of'],freq='day'))
    pairs,timeline,evidence=qualify(primary['source/normalized_events.pkl'],primary['source/original_guidance_events.pkl'],facts,texts,metadata,calendar)
    tracking=card/'qlib_recorder'
    with R.start(experiment_name=plan['experiment'],recorder_name='existing_corpus_only',uri=str(tracking)):
        rec=R.get_recorder();rec.log_params(plan_sha256=digest,new_fits=0,new_fetches=0,new_accounts=0)
        rec.save_objects(artifact_path='inputs',plan=plan,primary_facts=facts,fact_bytes=factbytes,manifest_bytes=manifest_bytes,secondary_texts=texts,secondary_metadata=metadata,calendar=calendar,source_audits={'primary':audit,'secondary':secondary_audit})
        rec.save_objects(artifact_path='sources',execution_workflow=Path(__file__).read_text())
        rec.save_objects(artifact_path='analysis',pairs=pairs,timeline=timeline,scope_evidence=evidence)
        policy={'policy':'slim_card_recorder_first','workflow_config':'workflow.yaml','recorder_only_artifacts':['inputs/*','sources/*']}
        summary={'experiment':plan['experiment'],'plan_sha256':digest,'decision':'source_qualified_subset_under_delayed_regular_report_assumption','objective_achieved':False,'new_fits':0,'new_fetches':0,'new_accounts':0,'market_outcome_reads':0,'error_features_computed':0,
            'targets':len(pairs),'captured_filing_events':len(timeline),'status_counts':pairs.qualification.value_counts().to_dict(),'freshness_counts':pairs.freshness.value_counts().to_dict(),
            'known_realization_events_pre2023':int((pairs.actual_report_available_at<pd.Timestamp('2023-01-01',tz='America/New_York')).sum()),
            'comparable_realization_events_pre2023':int(((pairs.actual_report_available_at<pd.Timestamp('2023-01-01',tz='America/New_York'))&pairs.measurement_known).sum()),
            'first_pair_available_at':str(pairs.pair_available_at.min()),'last_pair_available_at':str(pairs.pair_available_at.max()),
            'all_updates_coverage_verified':False,'first_public_timestamp_verified':False,'original_html_bytes_verified':False,
            'next_action':'Separate independently reviewed information/model design; source qualification does not authorize fits.',
            'provider_uri':str(root/plan['provider_uri']),'recorder_id':rec.id,'experiment_id':str(rec.experiment_id),'recorder_uri':str(tracking),'recorder_local_dir':str(tracking/str(rec.experiment_id)/rec.id),'artifact_policy':policy}
        rec.save_objects(artifact_path='analysis',summary=summary)
        artifacts=Path(summary['recorder_local_dir'])/'artifacts';rec.log_metrics(registered_artifact_count=sum(p.is_file() for p in artifacts.rglob('*')),source_target_count=len(pairs))
    pairs.to_csv(card/'pair_qualification.csv',index=False);timeline.to_csv(card/'disclosure_timeline.csv',index=False);evidence.to_csv(card/'scope_evidence.csv',index=False)
    atomic_json({'recorder_id':rec.id,'entries':[{'path':str(p.relative_to(card)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in sorted(artifacts.rglob('*')) if p.is_file()]},card/'verification_manifest.json')
    binding=json.loads((root/plan['sources']['secondary_existing_sec_text']['card']/'nas_binding.json').read_text());binding.pop('evidence');atomic_json(binding,card/'nas_binding.json')
    (card/'workflow.yaml').write_text(yaml.safe_dump({'experiment':plan['experiment'],'provider_uri':summary['provider_uri'],'recorder_uri':str(tracking),'artifact_policy':policy},sort_keys=False))
    summary['nas_evidence']=archive_nas_recorder(card/'nas_binding.json',workspace_root=root,card=card,recorder_id=rec.id,experiment_id=str(rec.experiment_id));atomic_json(summary,card/'summary.json');return summary
