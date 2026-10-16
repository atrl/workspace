"""Candidate monetary TTM source construction; no price or market outcome use."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd


def prepare_income(facts, calendar, allowed_tags=None):
    if facts.empty:return facts.copy()
    allowed_tags=allowed_tags or ['us-gaap:NetIncomeLoss','ifrs-full:ProfitLoss']
    data=facts.loc[(facts.taxonomy+':'+facts.tag).isin(allowed_tags)].copy()
    data=data.loc[data.start.notna() & np.isfinite(data.val)].copy()
    for c in ['start','end','filed']:data[c]=pd.to_datetime(data[c])
    positions=calendar.searchsorted(data.filed,side='right')
    data['available_session']=[calendar[i] if i<len(calendar) else pd.NaT for i in positions]
    data['duration_days']=(data.end-data.start).dt.days
    return data


def monetary_ttm_at(prepared, cutoff):
    """Latest-known candidate annual or annual+YTD-priorYTD, period-aligned.

    Source accounting revisions are preserved and may make component bases
    incompatible; the result is deliberately candidate TTM, not qualified EP.
    """
    if prepared.empty:return None
    cutoff=pd.Timestamp(cutoff)
    data=prepared.loc[prepared.available_session.le(cutoff) & prepared.end.le(cutoff) & prepared.end.ge(cutoff-pd.Timedelta(days=730))].copy()
    if data.empty:return None
    keys=['taxonomy','tag','unit','start','end']
    last=data.groupby(keys).filed.transform('max')
    data=data.loc[data.filed.eq(last)]
    conflict=data.groupby(keys).val.transform('nunique').gt(1)
    data=data.loc[~conflict].sort_values(['filed','accn']).drop_duplicates(keys,keep='last')
    options=[]
    def add(parts,method,value):
        end=parts[0]['end']
        if not 0<=(cutoff-end).days<=180:return
        source=[{'accession':x['accn'],'start':str(x['start'].date()),'end':str(x['end'].date()),
                 'filed':str(x['filed'].date()),'available_session':str(x['available_session'].date()),
                 'form':x['form'],'value':float(x['val'])} for x in parts]
        options.append({'income':float(value),'period_end':str(end.date()),'method':method,
                        'method_id':parts[0]['taxonomy']+':'+parts[0]['tag']+':'+method,
                        'taxonomy':parts[0]['taxonomy'],'tag':parts[0]['tag'],'unit':parts[0]['unit'],
                        'available_session':max(x['available_session'] for x in parts),
                        'income_age_days':int((cutoff-end).days),
                        'max_component_vintage_age_days':max(int((cutoff-x['filed']).days) for x in parts),
                        'component_accessions':json.dumps(source,sort_keys=True),
                        'accounting_scope_compatible_certified':False})
    for _,group in data.groupby(['taxonomy','tag','unit']):
        annual=group.loc[group.duration_days.between(330,380)]
        ytd=group.loc[group.duration_days.between(45,320)]
        for _,a in annual.iterrows():add([a],'annual_direct',a.val)
        for _,current in ytd.iterrows():
            priors=annual.loc[annual.end.eq(current.start-pd.Timedelta(days=1))]
            for _,a in priors.iterrows():
                matching=ytd.loc[ytd.start.eq(a.start) & (ytd.duration_days-current.duration_days).abs().le(7) & (current.end-ytd.end).dt.days.between(350,380)]
                if len(matching):
                    previous=matching.assign(distance=(matching.duration_days-current.duration_days).abs()).sort_values(['distance','end']).iloc[0]
                    add([current,a,previous],'annual_plus_ytd_minus_prior_ytd',a.val+current.val-previous.val)
    if not options:return None
    return sorted(options,key=lambda x:(x['period_end'],x['available_session'],x['method']=='annual_direct',x['taxonomy']=='us-gaap',x['tag']=='NetIncomeLoss'))[-1]


def build_coverage(facts,holdings,mapping,calendar,allowed_tags=None):
    prepared={int(cik):prepare_income(frame,calendar,allowed_tags) for cik,frame in facts.items()}
    holdings=holdings.copy()
    for c in ['report_date','filing_date']:holdings[c]=pd.to_datetime(holdings[c])
    filing_positions=calendar.searchsorted(holdings.filing_date,side='right')
    holdings['available_session']=[calendar[i] if i<len(calendar) else pd.NaT for i in filing_positions]
    idmap=mapping.set_index('ticker').cik.to_dict()
    cache={};details=[];coverage=[]
    def one(g,cutoff,kind,identity):
        known=loss=annual=ytd=0.;missing=[]
        for _,position in g.iterrows():
            cik=idmap.get(position.ticker)
            if position.ticker=='MRVL' and position.report_date<pd.Timestamp('2021-04-20'):cik=1058057
            key=(int(cik),pd.Timestamp(cutoff)) if pd.notna(cik) else None
            if key not in cache:cache[key]=monetary_ttm_at(prepared.get(int(cik),pd.DataFrame()),cutoff) if key else None
            value=cache[key]
            item={'kind':kind,'cutoff':str(pd.Timestamp(cutoff).date()),'basket_id':identity,
                  'ticker':position.ticker,'cik':cik,'weight_pct':float(position.weight_pct),
                  'holding_report_date':str(position.report_date.date()),'holding_filed':str(position.filing_date.date()),
                  'candidate_income_known':value is not None}
            if value:
                item.update(value);known+=float(position.weight_pct)
                if value['income']<0:loss+=float(position.weight_pct)
                if value['method']=='annual_direct':annual+=float(position.weight_pct)
                else:ytd+=float(position.weight_pct)
            else:missing.append(position.ticker)
            details.append(item)
        coverage.append({'kind':kind,'cutoff':str(pd.Timestamp(cutoff).date()),'basket_id':identity,
                         'holdings_rows':len(g),'equity_weight_pct':float(g.weight_pct.sum()),
                         'candidate_ttm_weight_pct':known,'known_negative_candidate_weight_pct':loss,
                         'annual_direct_weight_pct':annual,'derived_ttm_weight_pct':ytd,
                         'missing_income_symbols':';'.join(sorted(missing)),
                         'candidate95pct':known>=95,'qualified_ep_weight_pct':0})
    for accn,g in holdings.groupby('accession'):
        eligible=g.available_session.dropna()
        if len(eligible):one(g,eligible.iloc[0],'disclosure',accn)
        else:coverage.append({'kind':'disclosure','cutoff':None,'basket_id':accn,'holdings_rows':len(g),'candidate95pct':False,'missing_income_symbols':'calendar_missing','qualified_ep_weight_pct':0})
    months=pd.Series(calendar,index=calendar).groupby(calendar.to_period('M')).max()
    # Exclude terminal partial month; next-session execution must be observable.
    months=months.loc[months.index<calendar[-1].to_period('M')]
    for cutoff in months:
        known=holdings.loc[holdings.available_session.le(cutoff)]
        if known.empty:
            coverage.append({'kind':'monthly','cutoff':str(cutoff.date()),'basket_id':None,'holdings_rows':0,'candidate95pct':False,'missing_income_symbols':'holdings_not_yet_public','qualified_ep_weight_pct':0});continue
        report=known.report_date.max();latest=known.loc[known.report_date.eq(report)].filing_date.max()
        basket=known.loc[known.report_date.eq(report) & known.filing_date.eq(latest)]
        if basket.accession.nunique()!=1:raise ValueError('ambiguous same-date basket revisions')
        one(basket,cutoff,'monthly',basket.accession.iloc[0])
    return pd.DataFrame(details),pd.DataFrame(coverage)


def run(root,card_name='etf_pit_ttm_coverage_20261016'):
    import qlib
    from qlib.data import D
    from qlib.workflow import R
    from src.data.nas_market_data import archive_nas_recorder
    os.environ.setdefault('MLFLOW_ALLOW_FILE_STORE','true')
    root=Path(root).resolve();card=root/'research'/card_name
    plan=json.loads((card/'plan.json').read_text());review=json.loads((card/'design_review.json').read_text())
    plan_hash=hashlib.sha256((card/'plan.json').read_bytes()).hexdigest()
    if review.get('plan_sha256')!=plan_hash or review.get('decision')!='approved_candidate_TTM_source_coverage_only':raise ValueError('unapproved TTMplan')
    qlib.init(provider_uri=str(root/plan['provider_uri']),region='us',kernels=1)
    calendar=pd.DatetimeIndex(D.calendar(start_time=plan['start'],end_time=plan['end']))
    if not len(calendar):raise ValueError('Qlib calendar missing')
    if (card/'summary.json').exists():
        summary=json.loads((card/'summary.json').read_text());R.set_uri(summary['recorder_uri'])
        rec=R.get_recorder(recorder_id=summary['recorder_id'],experiment_id=summary['experiment_id'])
        assert len(rec.load_object('analysis/coverage'))==summary['coverage_rows']
        return summary
    binding=json.loads((card/'source_binding.json').read_text());source=root/binding['source_card']
    for filename,key in [('summary.json','source_summary_sha256'),('verification_manifest.json','verification_manifest_sha256')]:
        if hashlib.sha256((source/filename).read_bytes()).hexdigest()!=binding[key]:raise ValueError('pinned source changed')
    source_summary=json.loads((source/'summary.json').read_text())
    if source_summary['recorder_id']!=binding['recorder_id'] or source_summary['nas_evidence']['sha256']!=binding['nas_archive_sha256']:raise ValueError('source identity changed')
    manifest=json.loads((source/'verification_manifest.json').read_text())
    for entry in manifest['entries']:
        if hashlib.sha256((source/entry['path']).read_bytes()).hexdigest()!=entry['sha256']:raise ValueError('source artifact hash changed')
    R.set_uri(binding['recorder_uri']);source_rec=R.get_recorder(recorder_id=binding['recorder_id'],experiment_id=binding['experiment_id'])
    facts=source_rec.load_object('analysis/facts');holdings=source_rec.load_object('inputs/holdings');mapping=source_rec.load_object('analysis/issuer_mapping')
    detail,coverage=build_coverage(facts,holdings,mapping,calendar,plan['tags'])
    monthly=coverage.loc[coverage.kind.eq('monthly')];disclosure=coverage.loc[coverage.kind.eq('disclosure')]
    policy={'policy':'slim_card_recorder_first','workflow_config':'workflow.yaml','recorder_only_artifacts':['inputs/*','analysis/issuer_monthly_ttm','sources/*']}
    tracking=card/'qlib_recorder'
    with R.start(experiment_name=plan['experiment'],uri=str(tracking)):
        rec=R.get_recorder();rec.save_objects(artifact_path='inputs',plan=plan,source_binding=binding,calendar=calendar)
        rec.save_objects(artifact_path='analysis',issuer_monthly_ttm=detail,coverage=coverage)
        rec.save_objects(artifact_path='sources',module=Path(__file__).read_text())
        summary={'experiment':plan['experiment'],'decision':'candidate_monetary_TTM_constructed_not_EP_qualified','objective_achieved':False,'new_http_requests':0,'new_fits':0,'new_backtests':0,'coverage_rows':len(coverage),'issuer_detail_rows':len(detail),'monthly_dates':len(monthly),'months_with_public_holdings':int(monthly.holdings_rows.gt(0).sum()),'monthly_candidate95pct':int(monthly.candidate95pct.sum()),'disclosures_candidate95pct':int(disclosure.candidate95pct.sum()),'monthly_min_candidate_weight_pct':float(monthly.candidate_ttm_weight_pct.min()),'monthly_max_candidate_weight_pct':float(monthly.candidate_ttm_weight_pct.max()),'monthly_first':monthly.cutoff.min(),'monthly_last':monthly.cutoff.max(),'qualified_ep_monthly_rows':0,'qualified_input_path':None,'numerator_scope_compatible_certified':False,'provider_uri':str(root/plan['provider_uri']),'region':'us','market':'SOXX historical basket','as_of':'2026-07-02','calendar_last':str(calendar[-1]),'source_binding':binding,'plan_sha256':plan_hash,'recorder_id':rec.id,'experiment_id':str(rec.experiment_id),'recorder_uri':str(tracking),'recorder_local_dir':str(tracking/str(rec.experiment_id)/rec.id),'artifact_policy':policy,'limitations':['Candidate accounting components may differ in restatement basis; all component accessions retained','NI and IFRSprofit scope separate, not automatically attributable common income','No denominator, currency conversion, ADR ratio, stock split or price qualification; noEP computed','Reported holdings remain lagged; currentticker mapping alone cannot certify all historical securities','No return labels or model fitting; sourcecoverage is not alpha evidence']}
        rec.save_objects(artifact_path='analysis',summary=summary);artifacts=Path(summary['recorder_local_dir'])/'artifacts';rec.log_metrics(registered_artifact_count=sum(p.is_file() for p in artifacts.rglob('*')),monthly_dates=len(monthly),monthly_candidate95pct=summary['monthly_candidate95pct'])
    coverage.to_csv(card/'coverage.csv',index=False)
    nas=json.loads((source/'nas_binding.json').read_text());nas.pop('evidence',None);(card/'nas_binding.json').write_text(json.dumps(nas,indent=2)+'\n')
    (card/'verification_manifest.json').write_text(json.dumps({'recorder_id':rec.id,'entries':[{'path':str(p.relative_to(card)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in sorted(artifacts.rglob('*')) if p.is_file()]},indent=2)+'\n')
    summary['nas_evidence']=archive_nas_recorder(card/'nas_binding.json',workspace_root=root,card=card,recorder_id=rec.id,experiment_id=str(rec.experiment_id))
    (card/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');return summary
