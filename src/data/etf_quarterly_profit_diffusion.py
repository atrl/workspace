"""PIT reported-quarter profit diffusion; no prices, FX, targets or model fits."""
from pathlib import Path
import hashlib
import json
import os
import numpy as np
import pandas as pd

NAME='etf_quarterly_profit_diffusion_inputs_20261016'
TAGS={'us-gaap:NetIncomeLoss':0,'us-gaap:ProfitLoss':1,'ifrs-full:ProfitLoss':2}
CURRENCIES={'ASX':'TWD','ASML':'EUR','STM':'USD','ARM':'USD','NVMI':'USD','NXPI':'USD'}
FEATURES=['profit_improve_weight','profit_deteriorate_weight','profit_bounded_growth','quarterly_loss_weight']


def bounded_change(current,prior):
    if not np.isfinite(current) or not np.isfinite(prior):return np.nan
    denominator=abs(current)+abs(prior)
    return 0. if denominator==0 else (current-prior)/denominator


def prepare_facts(frame,calendar):
    if frame.empty:return frame.copy()
    d=frame.loc[(frame.taxonomy+':'+frame.tag).isin(TAGS)].copy()
    d=d.loc[d.start.notna() & np.isfinite(d.val)]
    for c in ['start','end','filed']:d[c]=pd.to_datetime(d[c])
    positions=calendar.searchsorted(d.filed,side='right')
    d['eligible_session']=[calendar[i] if i<len(calendar) else pd.NaT for i in positions]
    d['duration_days']=(d.end-d.start).dt.days
    return d


def _component(row):
    return {'accession':row.accn,'start':str(row.start.date()),'end':str(row.end.date()),'filed':str(row.filed.date()),'eligible_session':str(row.eligible_session.date()),'value':float(row.val),'taxonomy':row.taxonomy,'tag':row.tag,'currency':row.unit}


def domestic_quarters(frame,cutoff,currency):
    """Construct compatible quarters only from facts known at this cutoff."""
    if frame.empty:return []
    day=pd.Timestamp(cutoff).tz_localize(None).normalize()
    d=frame.loc[frame.unit.eq(currency)&frame.eligible_session.le(day)&frame.end.le(day)&frame.end.ge(day-pd.Timedelta(days=1000))].copy()
    if d.empty:return []
    keys=['taxonomy','tag','unit','start','end']
    d=d.loc[d.filed.eq(d.groupby(keys).filed.transform('max'))]
    conflicts=d.groupby(keys).val.transform('nunique').gt(1)
    conflicting=d.loc[conflicts].copy()
    blocked=set(tuple(x) for x in d.loc[conflicts,keys].itertuples(index=False,name=None))
    d=d.loc[~conflicts].sort_values(['filed','accn']).drop_duplicates(keys,keep='last')
    out=[]
    for (taxonomy,tag,unit,start,end),g in conflicting.groupby(keys):
        if 60<=(end-start).days<=110:
            out.append({'period_start':start,'period_end':end,'duration_days':(end-start).days,'profit':np.nan,'channel':taxonomy+':'+tag+'|consolidated_reported_profit|'+unit,'tag_priority':TAGS[taxonomy+':'+tag],'scope':'consolidated_reported_profit','currency':unit,'basis':taxonomy,'origin':'conflicting_direct_quarter','available_at':g.eligible_session.max(),'lineage':json.dumps([_component(r) for _,r in g.iterrows()],sort_keys=True),'unknown_reason':'conflicting_latest_quarter_vintage'})
    for (taxonomy,tag,unit,start,end),g in conflicting.groupby(keys):
        if 120<=(end-start).days<=380:
            direct=d.loc[d.taxonomy.eq(taxonomy)&d.tag.eq(tag)&d.unit.eq(unit)&d.end.eq(end)&d.duration_days.between(60,110)]
            if direct.empty:
                out.append({'period_start':pd.NaT,'period_end':end,'duration_days':np.nan,'profit':np.nan,'channel':taxonomy+':'+tag+'|consolidated_reported_profit|'+unit,'tag_priority':TAGS[taxonomy+':'+tag],'scope':'consolidated_reported_profit','currency':unit,'basis':taxonomy,'origin':'conflicting_YTD_quarter_unresolved','available_at':g.eligible_session.max(),'lineage':json.dumps([_component(r) for _,r in g.iterrows()],sort_keys=True),'unknown_reason':'latest_quarter_from_conflicting_YTD'})
    for (taxonomy,tag,unit),g in d.groupby(['taxonomy','tag','unit']):
        channel=taxonomy+':'+tag+'|consolidated_reported_profit|'+unit
        def add(start,end,value,parts,origin):
            if (taxonomy,tag,unit,start,end) in blocked:return
            out.append({'period_start':start,'period_end':end,'duration_days':(end-start).days,'profit':float(value),'channel':channel,'tag_priority':TAGS[taxonomy+':'+tag],'scope':'consolidated_reported_profit','currency':unit,'basis':taxonomy,'origin':origin,'available_at':max(x.eligible_session for x in parts),'lineage':json.dumps([_component(x) for x in parts],sort_keys=True)})
        for _,row in g.loc[g.duration_days.between(60,110)].iterrows():
            add(row.start,row.end,row.val,[row],'direct_quarter')
        for _,long in g.loc[g.duration_days.between(120,380)].iterrows():
            earlier=g.loc[g.start.eq(long.start)&g.end.lt(long.end)&(long.end-g.end).dt.days.between(61,111)]
            if len(earlier):
                prior=earlier.sort_values('end').iloc[-1]
                add(prior.end+pd.Timedelta(days=1),long.end,long.val-prior.val,[long,prior],'compatible_YTD_difference')
            elif g.loc[g.end.eq(long.end)&g.duration_days.between(60,110)].empty:
                out.append({'period_start':pd.NaT,'period_end':long.end,'duration_days':np.nan,'profit':np.nan,'channel':channel,'tag_priority':TAGS[taxonomy+':'+tag],'scope':'consolidated_reported_profit','currency':unit,'basis':taxonomy,'origin':'reported_YTD_quarter_underivable','available_at':long.eligible_session,'lineage':json.dumps([_component(long)],sort_keys=True),'unknown_reason':'latest_reported_period_quarter_underivable'})
    if not out:return []
    # Direct reported quarter wins over a derived value for the same period.
    ordered=sorted(out,key=lambda x:(x['period_end'],x['channel'],bool(x.get('unknown_reason')),x['origin']=='direct_quarter',x['available_at']),reverse=True)
    unique={}
    for x in ordered:unique.setdefault((x['period_start'],x['period_end'],x['channel']),x)
    return list(unique.values())


def foreign_quarters(frame,ticker,cutoff):
    day=pd.Timestamp(cutoff).tz_localize(None).normalize()
    d=frame.loc[frame.ticker.eq(ticker)&frame.accounting_basis.eq('Taiwan_IFRS')&frame.currency.eq('TWD')&frame.profit_concept.eq('attributable_to_parent')&frame.eligible_session.le(day)&frame.period_end.le(day)&frame.period_end.ge(day-pd.Timedelta(days=1000))].copy()
    reported=d.copy();d=d.loc[d.period_kind.eq('quarter')]
    out=[]
    for end,g in reported.loc[~reported.period_kind.eq('quarter')].groupby('period_end'):
        if d.loc[d.period_end.eq(end)].empty:
            out.append({'period_start':pd.NaT,'period_end':end,'duration_days':np.nan,'profit':np.nan,'channel':'Taiwan_IFRS|attributable_to_parent|TWD','tag_priority':0,'scope':'attributable_to_parent','currency':'TWD','basis':'Taiwan_IFRS','origin':'reported_foreign_period_quarter_underivable','available_at':g.eligible_session.max(),'lineage':json.dumps(g.component_lineage.tolist()),'unknown_reason':'latest_reported_period_quarter_underivable'})
    for (start,end),g in d.groupby(['period_start','period_end']):
        g=g.loc[g.available_at.eq(g.available_at.max())]
        if g.value.nunique()!=1:
            out.append({'period_start':start,'period_end':end,'duration_days':(end-start).days,'profit':np.nan,'channel':'Taiwan_IFRS|attributable_to_parent|TWD','tag_priority':0,'scope':'attributable_to_parent','currency':'TWD','basis':'Taiwan_IFRS','origin':'conflicting_foreign_quarter','available_at':g.eligible_session.max(),'lineage':json.dumps(g.component_lineage.tolist()),'unknown_reason':'conflicting_latest_quarter_vintage'})
            continue
        row=g.iloc[0]
        out.append({'period_start':start,'period_end':end,'duration_days':(end-start).days,'profit':float(row.value),'channel':'Taiwan_IFRS|attributable_to_parent|TWD','tag_priority':0,'scope':'attributable_to_parent','currency':'TWD','basis':'Taiwan_IFRS','origin':row.extraction,'available_at':row.eligible_session,'lineage':row.component_lineage})
    return out



def asml_quarters(frame,cutoff):
    day=pd.Timestamp(cutoff).tz_localize(None).normalize()
    d=frame.loc[frame.cik.eq(937966)&frame.period_kind.eq('quarter')&frame.accounting_basis.eq('US_GAAP')&frame.currency.eq('EUR')&frame.profit_concept.eq('consolidated_reported_net_income')&frame.eligible_session.le(day)&frame.period_end.le(day)&frame.period_end.ge(day-pd.Timedelta(days=1000))]
    out=[]
    for (start,end),g in d.groupby(['period_start','period_end']):
        g=g.loc[g.available_at.eq(g.available_at.max())];row=g.iloc[0]
        conflict=g.value.nunique()!=1
        out.append({'period_start':start,'period_end':end,'duration_days':(end-start).days,'profit':np.nan if conflict else float(row.value),'channel':'CIK937966|US_GAAP|consolidated_reported_net_income|EUR|direct_reported_quarter','tag_priority':0,'scope':'consolidated_reported_net_income','currency':'EUR','basis':'US_GAAP','origin':'direct_ASML_USGAAP_quarter_statement','available_at':row.eligible_session,'lineage':json.dumps(g.component_lineage.tolist()) if conflict else row.component_lineage,**({'unknown_reason':'conflicting_latest_quarter_vintage'} if conflict else {})})
    return out

def latest_quarter_pair(quarters,cutoff):
    if not quarters:return None,None,'no_reported_quarter'
    # Channel selection uses economic date and fixed tag priority, never growth.
    ordered=sorted(quarters,key=lambda x:(x['period_end'],-x['tag_priority'],bool(x.get('unknown_reason')),x['origin']=='direct_quarter',x['available_at']),reverse=True)
    current=ordered[0];day=pd.Timestamp(cutoff).tz_localize(None).normalize();age=(day-current['period_end']).days
    if not np.isfinite(current['profit']):return current,None,current.get('unknown_reason','latest_quarter_unknown')
    if age>180:return current,None,'current_quarter_stale'
    priors=[q for q in quarters if q['channel']==current['channel'] and 350<=(current['period_end']-q['period_end']).days<=380 and abs(current['duration_days']-q['duration_days'])<=7]
    if not priors:return current,None,'same_channel_previous_year_quarter_missing'
    rank=lambda q:(abs((current['period_end']-q['period_end']).days-365),abs(current['duration_days']-q['duration_days']))
    best=min(rank(q) for q in priors);priors=[q for q in priors if rank(q)==best]
    if any(not np.isfinite(q['profit']) for q in priors):return current,None,'previous_year_quarter_conflicting'
    if len({q['profit'] for q in priors})!=1:return current,None,'ambiguous_previous_year_quarter'
    return current,sorted(priors,key=lambda q:q['available_at'])[-1],'paired'


def aggregate_issuer_rows(details):
    d=pd.DataFrame(details)
    if d.empty:
        return {**{f:np.nan for f in FEATURES},'pair_coverage_weight':0.,'current_profit_coverage_weight':0.,'stale_profit_weight':0.,'known_profit_age_mean_days':np.nan,'max_profit_age_days':np.nan,'eligible':False,'raw_known_improve_weight':0.,'raw_known_deteriorate_weight':0.,'raw_known_bounded_growth':0.,'raw_known_loss_weight':0.}
    pair=d.pair_valid;current=d.current_valid;paired_weight=float(d.loc[pair,'weight'].sum());current_weight=float(d.loc[current,'weight'].sum());eligible=paired_weight>=.95
    improve=float(d.loc[pair & d.profit_change.gt(0),'weight'].sum());deteriorate=float(d.loc[pair & d.profit_change.lt(0),'weight'].sum());growth=float((d.loc[pair,'weight']*d.loc[pair,'bounded_growth']).sum());loss=float(d.loc[current & d.current_profit.lt(0),'weight'].sum())
    return {'profit_improve_weight':improve if eligible else np.nan,'profit_deteriorate_weight':deteriorate if eligible else np.nan,'profit_bounded_growth':growth if eligible else np.nan,'quarterly_loss_weight':loss if eligible else np.nan,'pair_coverage_weight':paired_weight,'current_profit_coverage_weight':current_weight,'stale_profit_weight':float(d.loc[d.reason.eq('current_quarter_stale'),'weight'].sum()),'known_profit_age_mean_days':float((d.loc[current,'weight']*d.loc[current,'quarter_age_days']).sum()/current_weight) if current_weight else np.nan,'max_profit_age_days':float(d.loc[current,'quarter_age_days'].max()) if current_weight else np.nan,'eligible':eligible,'raw_known_improve_weight':improve,'raw_known_deteriorate_weight':deteriorate,'raw_known_bounded_growth':growth,'raw_known_loss_weight':loss}


def build_monthly(facts,foreign,holdings,mapping,calendar,asml=None):
    prepared={int(cik):prepare_facts(frame,calendar) for cik,frame in facts.items()}
    h=holdings.copy()
    for col in ['report_date','filing_date']:h[col]=pd.to_datetime(h[col])
    pos=calendar.searchsorted(h.filing_date,side='right');h['eligible_session']=[calendar[i] if i<len(calendar) else pd.NaT for i in pos]
    if (h.weight_pct<0).any() or (h.weight_pct>100).any():raise ValueError('invalid disclosed long-only weights')
    idmap=mapping.set_index('ticker').cik.to_dict()
    foreign=foreign.copy()
    for col in ['period_start','period_end','eligible_session']:foreign[col]=pd.to_datetime(foreign[col])
    foreign['available_at']=pd.to_datetime(foreign.available_at,utc=True)
    if asml is not None:
        asml=asml.copy()
        for col in ['period_start','period_end','eligible_session']:asml[col]=pd.to_datetime(asml[col])
        asml['available_at']=pd.to_datetime(asml.available_at,utc=True)
    dates=pd.Series(calendar,index=calendar).groupby(calendar.to_period('M')).max()
    dates=dates.loc[(dates.index>=pd.Period('2019-01','M'))&(dates.index<calendar[-1].to_period('M'))]
    details=[];monthly=[]
    for day in dates:
        cutoff=pd.Timestamp(day).tz_localize('America/New_York')+pd.Timedelta(hours=16,minutes=15)
        known=h.loc[h.eligible_session.le(day)];issuer_rows=[]
        if known.empty:
            monthly.append({'signal_at':cutoff,'datetime':day,'basket_accession':None,'report_date':pd.NaT,'filing_date':pd.NaT,'holdings_age_days':np.nan,'reported_weight_sum':np.nan,'basket_status':'not_yet_disclosed',**aggregate_issuer_rows([])});continue
        report=known.report_date.max();latest=known.loc[known.report_date.eq(report)].filing_date.max();basket=known.loc[known.report_date.eq(report)&known.filing_date.eq(latest)]
        if basket.accession.nunique()!=1 or basket.ticker.duplicated().any():raise ValueError('ambiguous disclosed basket')
        for _,position in basket.iterrows():
            ticker=position.ticker;cik=idmap.get(ticker)
            transition=ticker=='MRVL' and report<pd.Timestamp('2021-04-20') and day>=pd.Timestamp('2021-04-20')
            if ticker=='MRVL' and report<pd.Timestamp('2021-04-20'):cik=1058057
            if pd.isna(cik):quarters=[]
            elif ticker=='ASML' and asml is not None:quarters=asml_quarters(asml,cutoff)
            elif ticker in ['TSM','UMC']:quarters=foreign_quarters(foreign,ticker,cutoff)
            else:quarters=domestic_quarters(prepared.get(int(cik),pd.DataFrame()),cutoff,CURRENCIES.get(ticker,'USD'))
            q,prior,reason=latest_quarter_pair(quarters,cutoff)
            row={'signal_at':cutoff,'datetime':day,'ticker':ticker,'cik':cik,'weight':float(position.weight_pct)/100.,'report_date':report,'filing_date':latest,'basket_accession':position.accession,'mapping_policy':'lagged_report_identity_not_current_legal_portfolio','identity_transition_flag':transition,'current_valid':q is not None and np.isfinite(q['profit']) and reason!='current_quarter_stale','pair_valid':reason=='paired','reason':reason,'current_profit':np.nan,'prior_profit':np.nan,'profit_change':np.nan,'bounded_growth':np.nan,'quarter_age_days':np.nan}
            if q is not None:
                row.update(current_profit=q['profit'],quarter_age_days=(day-q['period_end']).days,current_period_start=q['period_start'],current_period_end=q['period_end'],channel=q['channel'],profit_scope=q['scope'],currency=q['currency'],accounting_basis=q['basis'],current_origin=q['origin'],current_lineage=q['lineage'],current_eligible_session=q['available_at'],current_available_at=pd.Timestamp(q['available_at']).tz_localize('America/New_York')+pd.Timedelta(hours=9,minutes=30))
            if prior is not None:
                row.update(prior_profit=prior['profit'],profit_change=q['profit']-prior['profit'],bounded_growth=bounded_change(q['profit'],prior['profit']),prior_period_start=prior['period_start'],prior_period_end=prior['period_end'],prior_origin=prior['origin'],prior_lineage=prior['lineage'],prior_eligible_session=prior['available_at'],prior_available_at=pd.Timestamp(prior['available_at']).tz_localize('America/New_York')+pd.Timedelta(hours=9,minutes=30))
            issuer_rows.append(row)
        aggregate=aggregate_issuer_rows(issuer_rows)
        if not (-1-1e-10<=aggregate['raw_known_bounded_growth']<=1+1e-10):raise ValueError('unrenormalized diffusion out of bounds')
        ages=[r['quarter_age_days'] for r in issuer_rows if np.isfinite(r['quarter_age_days'])]
        monthly.append({'signal_at':cutoff,'datetime':day,'basket_accession':basket.accession.iloc[0],'report_date':report,'filing_date':latest,'holdings_age_days':(day-report).days,'reported_weight_sum':float(basket.weight_pct.sum())/100.,'basket_status':'lagged_disclosed_proxy','max_reported_quarter_age_days':max(ages) if ages else np.nan,**aggregate})
        details.extend(issuer_rows)
    return pd.DataFrame(monthly),pd.DataFrame(details)


def _load_source(root,binding):
    from qlib.workflow import R
    from src.data.nas_market_data import materialize_nas_evidence
    card=root/binding['card']
    if hashlib.sha256((card/'summary.json').read_bytes()).hexdigest()!=binding['summary_sha256']:raise ValueError('source summary changed')
    if hashlib.sha256((card/'verification_manifest.json').read_bytes()).hexdigest()!=binding['verification_manifest_sha256']:raise ValueError('source manifest changed')
    summary=json.loads((card/'summary.json').read_text())
    if summary['recorder_id']!=binding['recorder_id'] or summary['nas_evidence']['sha256']!=binding['nas_archive_sha256']:raise ValueError('source Recorder changed')
    if not Path(summary['recorder_local_dir']).exists():materialize_nas_evidence(card/'nas_binding.json',card=card)
    for e in json.loads((card/'verification_manifest.json').read_text())['entries']:
        if hashlib.sha256((card/e['path']).read_bytes()).hexdigest()!=e['sha256']:raise ValueError('source artifact hash mismatch')
    R.set_uri(summary['recorder_uri']);return R.get_recorder(recorder_id=summary['recorder_id'],experiment_id=summary['experiment_id'])


def run(root,card_name=NAME):
    import qlib
    from qlib.data import D
    from qlib.workflow import R
    from src.data.qlib_native import build_qlib_store_from_frame
    from src.data.nas_market_data import archive_nas_recorder,materialize_nas_evidence
    os.environ.setdefault('MLFLOW_ALLOW_FILE_STORE','true')
    root=Path(root).resolve();card=root/'research'/card_name
    plan=json.loads((card/'plan.json').read_text());review=json.loads((card/'design_review.json').read_text());ph=hashlib.sha256((card/'plan.json').read_bytes()).hexdigest()
    if review['plan_sha256']!=ph or review['decision']!='approved_for_execution':raise ValueError('unapproved source construction')
    qlib.init(provider_uri=str(root/plan['source_calendar']),region='us',kernels=1)
    calendar=pd.DatetimeIndex(D.calendar(end_time=plan['as_of']))
    previous=json.loads((card/'summary.json').read_text()) if (card/'summary.json').exists() else {}
    if previous.get('construction_version')==4 and previous.get('plan_sha256')==ph:
        s=previous
        if not Path(s['recorder_local_dir']).exists():materialize_nas_evidence(card/'nas_binding.json',card=card)
        R.set_uri(s['recorder_uri']);rec=R.get_recorder(recorder_id=s['recorder_id'],experiment_id=s['experiment_id']);assert len(rec.load_object('analysis/monthly_features'))==s['monthly_rows'];return s
    source=_load_source(root,plan['sources']['facts']);facts=source.load_object('analysis/facts');holdings=source.load_object('inputs/holdings');mapping=source.load_object('analysis/issuer_mapping')
    source=_load_source(root,plan['sources']['foreign']);foreign=source.load_object('analysis/events')
    asml=None
    if 'asml' in plan['sources']:
        source=_load_source(root,plan['sources']['asml']);asml=source.load_object('analysis/events')
    monthly,details=build_monthly(facts,foreign,holdings,mapping,calendar,asml=asml)
    diagnostics=['pair_coverage_weight','current_profit_coverage_weight','stale_profit_weight','known_profit_age_mean_days','max_profit_age_days','max_reported_quarter_age_days','holdings_age_days','reported_weight_sum','eligible','raw_known_improve_weight','raw_known_deteriorate_weight','raw_known_bounded_growth','raw_known_loss_weight']
    intended=FEATURES+diagnostics;writable=[f for f in intended if monthly[f].notna().any()];all_null=[f for f in intended if f not in writable]
    state=monthly[['datetime']+writable].copy();state.insert(0,'instrument','SOXX_QUARTER_PROFIT_DIFFUSION')
    for field in writable:state[field]=state[field].astype(float)
    provider=root/'data/qlib'/card_name
    manifest=build_qlib_store_from_frame(state,provider_uri=provider,region='us',freq='day',as_of=plan['as_of'],source=card_name,field_map={f:f for f in writable})
    manifest.update(monthly_calendar_only=True,source_calendar=str(root/plan['source_calendar']),fields_not_written_because_all_null=all_null,full_intended_fields=intended,monthly_rows_preserved=len(monthly),coverage_gate=.95,unrenormalized_fund_weights=True,no_returns_or_prices=True)
    (provider/'qlib_store_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    qlib.init(provider_uri=str(provider),region='us',kernels=1)
    back=D.features(['SOXX_QUARTER_PROFIT_DIFFUSION'],['$'+f for f in writable],start_time=monthly.datetime.min(),end_time=monthly.datetime.max()).reset_index()
    compare=state.merge(back,on=['instrument','datetime'],validate='one_to_one')
    for field in writable:np.testing.assert_allclose(compare[field],compare['$'+field],rtol=1e-6,atol=1e-7,equal_nan=True)
    snapshot=root/'data/snapshots'/card_name;snapshot.mkdir(parents=True,exist_ok=True);monthly.to_parquet(snapshot/'monthly_features.parquet',index=False)
    cache_files={}
    cache_entries=[]
    if 'asml' in plan['sources']:
        cache_names=[Path(plan['sources']['foreign']['card']).name,Path(plan['sources']['asml']['card']).name,Path(plan['parent_source_card']).name,card_name]
        for cache_name in cache_names:
            for cache_kind in ['qlib','snapshots']:
                cache_root=root/'data'/cache_kind/cache_name
                for path in sorted(cache_root.rglob('*')):
                    if path.is_symlink():raise ValueError('cachebundle symlink refused')
                    if not path.is_file():continue
                    relative=str(path.relative_to(root));body=path.read_bytes();cache_files[relative]=body
                    cache_entries.append({'path':relative,'sha256':hashlib.sha256(body).hexdigest(),'bytes':len(body)})
    policy={'policy':'slim_card_recorder_first','workflow_config':'workflow.yaml','recorder_only_artifacts':['inputs/*','analysis/monthly_features','analysis/issuer_quarter_pairs','analysis/qlib_readback','sources/*']};tracking=card/'qlib_recorder'
    with R.start(experiment_name=card_name,uri=str(tracking)):
        rec=R.get_recorder();rec.save_objects(artifact_path='inputs',plan=plan,source_bindings=plan['sources'],provider_manifest=manifest,source_calendar=calendar,data_cache_files=cache_files)
        rec.save_objects(artifact_path='analysis',monthly_features=monthly,issuer_quarter_pairs=details,qlib_readback=back)
        rec.save_objects(artifact_path='sources',module=Path(__file__).read_text())
        summary={'experiment':card_name,'explicit_additional_channels':['ASML_USGAAP_EUR'] if asml is not None else [],'data_cache_bundle_artifact':'inputs/data_cache_files' if cache_files else None,'data_cache_file_count':len(cache_files),'construction_version':4,'supersedes_recorder_id':previous.get('recorder_id'),'correction':'latest_quarter_conflict_and_underivable_YTD_unknown_barriers_and_explicit_0930NY_clock','decision':'quarterly_profit_diffusion_constructed_source_qualification_reported' if monthly.eligible.any() else 'needs_quarterly_profit_source_coverage_only','objective_achieved':False,'new_http_requests':0,'new_fits':0,'new_accounts':0,'new_target_series':0,'qlib_docs':['https://qlib.readthedocs.io/en/latest/component/data.html','https://qlib.readthedocs.io/en/latest/component/recorder.html'],'monthly_rows':len(monthly),'eligible_months':int(monthly.eligible.sum()),'first_month':str(monthly.datetime.min().date()),'last_month':str(monthly.datetime.max().date()),'first_eligible_month':str(monthly.loc[monthly.eligible,'datetime'].min().date()) if monthly.eligible.any() else None,'last_eligible_month':str(monthly.loc[monthly.eligible,'datetime'].max().date()) if monthly.eligible.any() else None,'issuer_month_rows':len(details),'valid_issuer_pairs':int(details.pair_valid.sum()),'pair_coverage_min':float(monthly.pair_coverage_weight.min()),'pair_coverage_max':float(monthly.pair_coverage_weight.max()),'fields_not_written_because_all_null':all_null,'provider_uri':str(provider),'canonical_monthly_path':str(snapshot/'monthly_features.parquet'),'region':'us','market':'SOXX_QUARTER_PROFIT_DIFFUSION information instrument','as_of':plan['as_of'],'plan_sha256':ph,'recorder_id':rec.id,'experiment_id':str(rec.experiment_id),'recorder_uri':str(tracking),'recorder_local_dir':str(tracking/str(rec.experiment_id)/rec.id),'artifact_policy':policy,'boundaries':['Reported lagged basket, not current legal portfolio; MRVL transition flagged','Consolidated profit channels not relabeled parent NI; Taiwan parent channels separate','No currency conversion or net monetary aggregation; same-channel dimensionless YoY only','180day freshness and95pct paircoverage gates unchanged; missingmonths retained','Known current-profit coverage differs paired coverage; loss uses freshcurrentknown profits','Historical source vintages are development inputs, not new independent market evaluation']}
        rec.save_objects(artifact_path='analysis',summary=summary);artifacts=Path(summary['recorder_local_dir'])/'artifacts';rec.log_metrics(registered_artifact_count=sum(p.is_file() for p in artifacts.rglob('*')),eligible_months=summary['eligible_months'],monthly_rows=len(monthly))
    if cache_entries:
        (card/'data_cache_manifest.json').write_text(json.dumps({'recorder_id':rec.id,'artifact':'inputs/data_cache_files','entries':cache_entries},indent=2)+'\n')
        (card/'provider_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    monthly[['datetime','basket_status','pair_coverage_weight','current_profit_coverage_weight','stale_profit_weight','holdings_age_days','eligible']].to_csv(card/'source_coverage.csv',index=False)
    gaps=details.loc[~details.pair_valid].groupby(['ticker','reason']).agg(months=('datetime','size'),mean_unqualified_weight=('weight','mean')).reset_index();gaps.to_csv(card/'source_gaps.csv',index=False)
    yearly=monthly.assign(year=monthly.datetime.dt.year).groupby('year').agg(months=('datetime','size'),eligible_months=('eligible','sum'),minimum_pair_coverage=('pair_coverage_weight','min'),maximum_pair_coverage=('pair_coverage_weight','max'));yearly.to_csv(card/'year_coverage.csv')
    (snapshot/'manifest.json').write_text(json.dumps({'source_card':str(card.relative_to(root)),'recorder_id':rec.id,'monthly_sha256':hashlib.sha256((snapshot/'monthly_features.parquet').read_bytes()).hexdigest(),'provider_uri':str(provider.relative_to(root)),'coverage_gate':.95,'no_targets':True},indent=2)+'\n')
    binding=json.loads((root/plan['sources']['facts']['card']/'nas_binding.json').read_text());binding.pop('evidence',None);(card/'nas_binding.json').write_text(json.dumps(binding,indent=2)+'\n')
    (card/'verification_manifest.json').write_text(json.dumps({'recorder_id':rec.id,'entries':[{'path':str(p.relative_to(card)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in sorted(artifacts.rglob('*')) if p.is_file()]},indent=2)+'\n')
    summary['nas_evidence']=archive_nas_recorder(card/'nas_binding.json',workspace_root=root,card=card,recorder_id=rec.id,experiment_id=str(rec.experiment_id));(card/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');return summary
