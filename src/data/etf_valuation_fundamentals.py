"""Historical SOXX issuer fact intake, with period/vintage identity retained."""
import hashlib
import json
import os
from pathlib import Path
import pandas as pd

FIELDS = {'NetIncomeLoss','ProfitLoss','EarningsPerShareDiluted','DilutedEarningsLossPerShare',
          'CommonStockSharesOutstanding','EntityCommonStockSharesOutstanding',
          'StockholdersEquity','Equity','NumberOfSharesOutstanding'}


def accession_facts(payload):
    doc=json.loads(payload)
    rows=[]
    for taxonomy,tags in doc.get('facts',{}).items():
        for tag,definition in tags.items():
            if tag not in FIELDS: continue
            for unit,items in definition.get('units',{}).items():
                for fact in items:
                    rows.append({'cik':doc['cik'],'entity_name':doc['entityName'],
                                 'taxonomy':taxonomy,'tag':tag,'unit':unit,**fact})
    data=pd.DataFrame(rows)
    if data.empty:return data,data.copy()
    if 'start' not in data:data['start']=None
    data['duration_days']=(pd.to_datetime(data.end)-pd.to_datetime(data.start)).dt.days
    # Forms and tags stay distinct; quarterly, YTD and annual periods never merge.
    keys=['cik','taxonomy','tag','unit','start','end','form']
    first=data.groupby(keys,dropna=False).filed.transform('min')
    candidates=data.loc[data.filed.eq(first)].drop_duplicates(keys+['accn','val']).copy()
    candidates['first_filing_conflict']=candidates.groupby(keys,dropna=False).val.transform('nunique').gt(1)
    return data,candidates


def run(root,staging='/tmp/etf_pit_facts',card_name='etf_pit_fundamentals_intake_20261016'):
    import qlib
    from qlib.data import D
    from qlib.workflow import R
    from src.data.nas_market_data import archive_nas_recorder
    os.environ.setdefault('MLFLOW_ALLOW_FILE_STORE','true')
    root=Path(root).resolve();staging=Path(staging)
    card=root/'research'/card_name
    provider=root/'data/qlib/soxx_pit_leveraged_training_v1'
    qlib.init(provider_uri=str(provider),region='us',kernels=1)
    calendar=D.calendar(start_time='2019-01-01',end_time='2026-10-16')
    if not len(calendar):raise ValueError('missing Qlib calendar')
    if (card/'summary.json').exists():
        summary=json.loads((card/'summary.json').read_text())
        R.set_uri(summary['recorder_uri'])
        rec=R.get_recorder(recorder_id=summary['recorder_id'],experiment_id=summary['experiment_id'])
        payloads=rec.load_object('inputs/raw_payloads')
        for e in rec.load_object('inputs/acquisition_manifest'):
            assert hashlib.sha256(payloads[e['name']]).hexdigest()==e['sha256']
        return summary
    plan=json.loads((card/'plan.json').read_text())
    review=json.loads((card/'design_review.json').read_text())
    plan_hash=hashlib.sha256((card/'plan.json').read_bytes()).hexdigest()
    if review.get('plan_sha256') != plan_hash or review.get('decision') != 'approved_raw_source_intake_only':raise ValueError('independent review does not bind plan')
    entries=json.loads((staging/'entries.json').read_text())
    requests=sum(not x.get('reused_parent',False) for x in entries)+plan.get('mapping_requests',1)
    if requests>plan['max_new_http_requests']:raise ValueError('request budget exceeded')
    all_entries=list(entries)
    fallback=[]
    if (staging/'compressed_entries.json').exists():
        fallback=json.loads((staging/'compressed_entries.json').read_text())
        fallback_plan=json.loads((card/'compressed_transport_plan.json').read_text())
        fallback_review=json.loads((card/'design_review_compressed_transport.json').read_text())
        if fallback_review.get('plan_sha256')!=hashlib.sha256((card/'compressed_transport_plan.json').read_bytes()).hexdigest() or fallback_review.get('decision')!='approved_compressed_transport_fallback_only':raise ValueError('compressed transport not approved')
        if len(fallback)>fallback_plan['max_new_http_requests'] or len({x['cik'] for x in fallback})!=len(fallback):raise ValueError('compressed budget exceeded')
        by_cik={x['cik']:x for x in entries}
        for e in fallback:
            previous=by_cik[e['cik']]
            if e['cik'] not in fallback_plan['eligible_universe'] or previous.get('parse_valid') is not False:raise ValueError('compressed attempt not eligible')
            if e.get('parse_valid') is True:by_cik[e['cik']]=e
        all_entries+=fallback
        entries=list(by_cik.values())
    mapping=pd.read_csv(staging/'issuer_mapping.csv')
    holdings_path=root/'data/qlib/soxx_pit_leveraged_training_v1/sec_nport_holdings_history.csv'
    if hashlib.sha256(holdings_path.read_bytes()).hexdigest()!=plan['holdings_sha256']:raise ValueError('holdings changed')
    holdings=pd.read_csv(holdings_path)
    payloads={e['name']:(staging/e['name']).read_bytes() for e in all_entries if (staging/e['name']).exists()}
    for e in all_entries:
        if hashlib.sha256(payloads.get(e['name'],b'')).hexdigest()!=e['sha256']:raise ValueError('transport history payload changed')
    facts={};candidates={};rows=[]
    for e in entries:
        b=payloads.get(e['name'],b'')
        if hashlib.sha256(b).hexdigest()!=e['sha256']:raise ValueError('payload hash changed')
        row={**e,'fact_rows':0,'first_candidates':0,'status_label':'http_or_schema_missing'}
        if e['status']==200:
            try:
                doc=json.loads(b)
                if int(doc['cik'])!=e['cik']:raise ValueError('CIK mismatch')
                data,first=accession_facts(b)
                facts[e['cik']]=data;candidates[e['cik']]=first
                row.update(entity_name=doc['entityName'],fact_rows=len(data),first_candidates=len(first),
                           status_label='raw_facts_only_not_EP_qualified')
                if len(data):
                    row.update(first_filed=data.filed.min(),last_filed=data.filed.max(),
                               tags=';'.join(sorted(set(data.taxonomy+':'+data.tag))),
                               first_filing_conflicts=int(first.first_filing_conflict.sum()),
                               negative_income_candidates=int(((first.tag.isin(['NetIncomeLoss','ProfitLoss'])) & first.val.lt(0)).sum()))
            except (ValueError,KeyError,TypeError) as ex:row['error']=str(ex)
        rows.append(row)
    inventory=pd.DataFrame(rows)
    mapping=mapping.merge(inventory[['cik','entity_name','fact_rows','first_candidates','status_label']],on='cik',how='left')
    mapping['valuation_qualified']=False
    # Aggregate availability is inventory coverage, never a valid PE or zero-loss estimate.
    normal=mapping.loc[mapping.ticker.ne('MRVL_PRE_20210420')].set_index('ticker')
    holdings['facts_downloaded']=holdings.ticker.map(normal.fact_rows.gt(0)).fillna(False)
    coverage=holdings.groupby(['accession','report_date','filing_date']).apply(lambda g:pd.Series({'equity_weight_pct':g.weight_pct.sum(),'raw_facts_mapped_weight_pct':g.loc[g.facts_downloaded,'weight_pct'].sum(),'valuation_qualified_weight_pct':0}),include_groups=False).reset_index()
    income_plan=json.loads((card/'income_inventory_plan.json').read_text())
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest()!=income_plan['module_sha256']:raise ValueError('income inventory code changed since freeze')
    income_review=json.loads((card/'design_review_income_inventory.json').read_text())
    if income_review.get('plan_sha256') != hashlib.sha256((card/'income_inventory_plan.json').read_bytes()).hexdigest() or income_review.get('decision') != 'approved_candidate_income_coverage_only':raise ValueError('income subplan not approved')
    income_coverage=income_weight_inventory(holdings,mapping,candidates)
    policy={'policy':'slim_card_recorder_first','workflow_config':'workflow.yaml','recorder_only_artifacts':['inputs/*','analysis/facts','analysis/first_candidates','sources/*']}
    tracking=card/'qlib_recorder'
    with R.start(experiment_name=plan['experiment'],uri=str(tracking)):
        rec=R.get_recorder()
        rec.save_objects(artifact_path='inputs',plan=plan,income_inventory_plan=income_plan,acquisition_manifest=all_entries,raw_payloads=payloads,holdings=holdings,company_tickers=(staging/'company_tickers.json').read_bytes())
        rec.save_objects(artifact_path='analysis',facts=facts,first_candidates=candidates,issuer_mapping=mapping,coverage=coverage,inventory=inventory,income_coverage=income_coverage)
        rec.save_objects(artifact_path='sources',module=Path(__file__).read_text(),acquisition=(staging/'acquisition_source.py').read_text())
        summary={'experiment':plan['experiment'],'decision':'raw_issuer_facts_acquired_needs_original_TTM_share_price_identity_validation','objective_achieved':False,'new_fits':0,'new_backtests':0,'new_requests':requests+len(fallback),'base_repair_requests':requests,'compressed_fallback_requests':len(fallback),'reused_files':sum(x.get('reused_parent',False) for x in entries),'requested_ciks':len(entries),'http200_files':int(inventory.status.eq(200).sum()),'ciks_with_facts':len(facts),'fact_rows':int(inventory.fact_rows.sum()),'candidate_income_weight_min_pct':float(income_coverage.candidate_income_weight_pct.min()),'candidate_income_weight_max_pct':float(income_coverage.candidate_income_weight_pct.max()),'candidate_disclosures_reaching95pct':int(income_coverage.coverage_95pct_candidate_only.sum()),'first_candidates':int(inventory.first_candidates.sum()),'qualified_monthly_rows':0,'qualified_input_path':None,'provider_uri':str(provider),'region':'us','market':'historical SOXX constituent union','as_of':'2026-10-16','calendar_last':str(calendar[-1]),'recorder_id':rec.id,'experiment_id':str(rec.experiment_id),'recorder_uri':str(tracking),'recorder_local_dir':str(tracking/str(rec.experiment_id)/rec.id),'plan_sha256':plan_hash,'artifact_policy':policy,'limitations':['Current SEC mapping candidates need historic issuer/CUSIP audit; MRVL predecessor separate','Companyfacts is tagged accession data, not blanket original-document verification','Annual/quarter/YTD and currency/share units remain distinct; no TTM fabricated','Fiscal revisions, ADR share ratios, nominal price/splits and historical missing prices must be qualified before EP','Current provider ends2026-07-02; later filing signals cannot use nonexistent exchange calendar','Raw mapped weight coverage is not qualified valuation coverage; unknown losses remain missing'],'next_action':'Validate original filing periods and TTM/denominator/corporate actions per historical issuer; preserve missing historic prices and resolve index versions; only then build canonical monthly snapshots and Qlib Dataset.'}
        rec.save_objects(artifact_path='analysis',summary=summary)
        artifacts=Path(summary['recorder_local_dir'])/'artifacts';rec.log_metrics(registered_artifact_count=sum(p.is_file() for p in artifacts.rglob('*')),fact_rows=summary['fact_rows'],qualified_monthly_rows=0)
    for key,frame in {'issuer_mapping':mapping,'coverage':coverage,'inventory':inventory,'income_coverage':income_coverage}.items():frame.to_csv(card/(key+'.csv'),index=False)
    binding=json.loads((root/'research/etf_valuation_source_20261016/nas_binding.json').read_text());binding.pop('evidence',None);(card/'nas_binding.json').write_text(json.dumps(binding,indent=2)+'\n')
    (card/'verification_manifest.json').write_text(json.dumps({'recorder_id':rec.id,'entries':[{'path':str(p.relative_to(card)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in sorted(artifacts.rglob('*')) if p.is_file()]},indent=2)+'\n')
    summary['nas_evidence']=archive_nas_recorder(card/'nas_binding.json',workspace_root=root,card=card,recorder_id=rec.id,experiment_id=str(rec.experiment_id))
    (card/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


def candidate_income_at(first, cutoff):
    """Source-only TTM coverage diagnostic; not original-document certification."""
    if first is None or first.empty:return None
    cutoff=pd.Timestamp(cutoff)
    data=first.loc[first.tag.isin(['NetIncomeLoss','ProfitLoss']) & first.unit.isin(['USD','TWD','EUR']) & ~first.first_filing_conflict].copy()
    data=data.loc[pd.to_datetime(data.filed).le(cutoff) & pd.to_datetime(data.end).le(cutoff)]
    possibilities=[]
    for (taxonomy,tag,unit),g in data.groupby(['taxonomy','tag','unit']):
        # Multiple forms may report the same period; do not double-count it.
        g=g.sort_values(['filed','accn']).drop_duplicates(['start','end'],keep='first')
        annual=g.loc[g.duration_days.between(330,380)]
        for _,x in annual.iterrows():
            if 0 <= (cutoff-pd.Timestamp(x.end)).days <= 180:
                possibilities.append({'method':'annual_report','income':float(x.val),'period_end':x.end,'taxonomy':taxonomy,'tag':tag,'unit':unit})
        quarters=g.loc[g.duration_days.between(60,110)].sort_values('end')
        for i in range(3,len(quarters)):
            block=quarters.iloc[i-3:i+1]
            starts=pd.to_datetime(block.start).to_list();ends=pd.to_datetime(block.end).to_list()
            if all((starts[j]-ends[j-1]).days==1 for j in range(1,4)) and 0<=(cutoff-ends[-1]).days<=180:
                possibilities.append({'method':'four_contiguous_quarters','income':float(block.val.sum()),'period_end':str(ends[-1].date()),'taxonomy':taxonomy,'tag':tag,'unit':unit})
    if not possibilities:return None
    return sorted(possibilities,key=lambda x:(x['period_end'],x['method']))[-1]


def income_weight_inventory(holdings,mapping,candidates):
    rows=[];by=mapping.set_index('ticker').cik.to_dict()
    for (accn,report,filed),g in holdings.groupby(['accession','report_date','filing_date']):
        known=negative=annual_weight=quarter_weight=0.;missing=[];ages=[]
        for _,position in g.iterrows():
            ticker=position.ticker;cik=by.get(ticker)
            if ticker=='MRVL' and pd.Timestamp(report)<pd.Timestamp('2021-04-20'):cik=1058057
            result=candidate_income_at(candidates.get(int(cik)) if pd.notna(cik) else None,filed)
            if result is None:missing.append(ticker)
            else:
                known+=float(position.weight_pct)
                if result['income']<0:negative+=float(position.weight_pct)
                if result['method']=='annual_report':annual_weight+=float(position.weight_pct)
                else:quarter_weight+=float(position.weight_pct)
                ages.append((pd.Timestamp(filed)-pd.Timestamp(result['period_end'])).days)
        rows.append({'accession':accn,'report_date':report,'filing_date':filed,
                     'candidate_income_weight_pct':known,'known_negative_income_weight_pct':negative,'annual_fallback_weight_pct':annual_weight,'four_quarter_weight_pct':quarter_weight,'max_candidate_income_age_days':max(ages) if ages else None,
                     'missing_income_symbols':';'.join(sorted(missing)),'coverage_95pct_candidate_only':known>=95,
                     'qualified_valuation_weight_pct':0,'availability_basis':'filing-date diagnostic only; next-session signal requires calendar'})
    return pd.DataFrame(rows)
