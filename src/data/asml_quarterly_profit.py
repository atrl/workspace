"""ASML quarterly reported profit source, separately labelled US GAAP EUR."""
from pathlib import Path
import hashlib,json,threading
from src.data.etf_foreign_quarterly_intake import Fetcher,submissions,fulltext_search

NAME='asml_quarterly_profit_source_20261016'
class ASMLFetcher(Fetcher):
    def __init__(self,root,staging='/tmp/asml_quarterly_profit_source'):
        self.root=Path(root).resolve();self.card=self.root/'research'/NAME
        self.plan=json.loads((self.card/'plan.json').read_text());review=json.loads((self.card/'design_review.json').read_text())
        if review['plan_sha256']!=hashlib.sha256((self.card/'plan.json').read_bytes()).hexdigest() or review['decision']!='approved_for_execution':raise ValueError('unapproved ASMLsource')
        self.staging=Path(staging);self.staging.mkdir(parents=True,exist_ok=True);self.path=self.staging/'receipts.json';self.lock=threading.Lock();self.receipts=json.loads(self.path.read_text()) if self.path.exists() else []


def _clean(soup):
    return ' '.join(soup.get_text(' ',strip=True).split())


def parse_statement(payload,target,receipt):
    """Read the first US-GAAP statement of operations, not ratios or EPS."""
    import re,calendar
    import pandas as pd
    from bs4 import BeautifulSoup
    text=_clean(BeautifulSoup(payload,'lxml'))
    if not re.search(r'U\.?S\.? GAAP',text[:1800],re.I):raise ValueError('USGAAP identity not explicit')
    match=re.search(r'\bNet income\s+(.{0,240}?)\s+(?:Basic net income|Diluted net income|Weighted average)',text,re.I)
    if not match:raise ValueError('net-income row unavailable')
    values=[];precisions=[]
    for token in re.findall(r'\(?\s*-?\d[\d,]*\.\d+\s*\)?',match[1]):
        neg='(' in token or '-' in token;value=float(token.replace(',','').replace('(','').replace(')','').replace('-','').strip());values.append(-value if neg else value);precisions.append(float(10**(6-len(token.strip().replace(')','').split('.')[-1].strip()))))
    start=text.lower().rfind('three months ended',0,match.start())
    first_sales=text.lower().find('net system sales',start)
    if start<0 or first_sales<0:raise ValueError('quarter header unavailable')
    header=text[start:first_sales]
    if not ('million' in header.lower() and ('EUR' in header or '€' in header or 'euro' in header.lower())):raise ValueError('EUR millions not explicit')
    dates=re.findall(r'\b(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?\s+(\d{1,2}),?',header,re.I)
    years=re.findall(r'\b20\d{2}\b',header)
    periods=re.findall(r'(Three|Six|Nine|Twelve) months ended',header,re.I)
    duration={'three':3,'six':6,'nine':9,'twelve':12}
    if len(dates)!=len(values) or len(years)!=len(values) or not periods or len(values)%len(periods):raise ValueError('header/NI columns do not align')
    repeats=len(values)//len(periods);events=[]
    for i,(date,year,value) in enumerate(zip(dates,years,values)):
        months=duration[periods[i//repeats].lower()]
        if months!=3:continue
        month=list(calendar.month_abbr).index(date[0][:3].capitalize());end=pd.Timestamp(int(year),month,int(date[1]));near=[(abs((end-q.end_time.normalize()).days),q) for q in pd.period_range(end-pd.Timedelta(days=100),end+pd.Timedelta(days=100),freq='Q')];distance,quarter=min(near,key=lambda x:x[0])
        if distance>7:raise ValueError('fiscal quarter cannot be labelled within7days')
        events.append({'ticker':'ASML','cik':937966,'period_end':end,'nominal_quarter':str(quarter),'period_months':3,'value':value*1e6,'currency':'EUR','profit_concept':'consolidated_reported_net_income','accounting_basis':'US_GAAP','source_unit':'million_EUR','source_precision_eur':precisions[i],'filed':target['filed'],'accepted_at':target['accepted_at'],'accession':target['accession'],'source_url':target['url'],'source_sha256':receipt['sha256'],'source_row':match[0],'source_header':header,'column_position':i})
    if not events:raise ValueError('no direct three-month NI columns')
    maxend=max(x['period_end'] for x in events)
    for x in events:x['comparative']=x['period_end']<maxend
    return events


def canonical_events(raw,calendar):
    import pandas as pd
    raw=raw.copy();raw['available_at']=pd.to_datetime(raw.accepted_at,utc=True)
    raw['nominal_period']=pd.PeriodIndex(raw.nominal_quarter,freq='Q')
    rows=[];gaps=[]
    for _,row in raw.iterrows():
        if row.nominal_period<pd.Period('2018Q1','Q') or row.nominal_period>pd.Period('2026Q2','Q'):continue
        prior=raw.loc[raw.nominal_period.eq(row.nominal_period-1)&raw.available_at.le(row.available_at)]
        if prior.empty:
            gaps.append({'quarter':row.nominal_quarter,'accession':row.accession,'reason':'preceding_fiscal_end_not_yet_documented'});continue
        prior=prior.loc[prior.available_at.eq(prior.available_at.max())]
        if prior.period_end.nunique()!=1:
            gaps.append({'quarter':row.nominal_quarter,'accession':row.accession,'reason':'ambiguous_preceding_fiscal_end'});continue
        anchor=prior.iloc[0];start=anchor.period_end+pd.Timedelta(days=1)
        if not 60<=(row.period_end-start).days<=110:raise ValueError('invalid true fiscal quarter duration')
        out=row.drop(labels=['nominal_period']).to_dict();out['period_start']=start;out['period_kind']='quarter';out['period_start_basis']='previous_disclosed_fiscal_quarter_end_plus1day';out['period_start_source_accession']=anchor.accession;out['period_start_source_available_at']=anchor.available_at
        day=row.available_at.tz_convert('America/New_York').tz_localize(None).normalize();idx=calendar.searchsorted(day,side='right');out['eligible_session']=calendar[idx] if idx<len(calendar) else pd.NaT;out['eligible_at']=pd.Timestamp(calendar[idx]).tz_localize('America/New_York')+pd.Timedelta(hours=9,minutes=30) if idx<len(calendar) else pd.NaT
        out['method_id']='CIK937966|US_GAAP|consolidated_reported_net_income|EUR|direct_reported_quarter'
        out['component_lineage']=json.dumps([{'accession':row.accession,'source_sha256':row.source_sha256,'value':row.value,'available_at':row.available_at.isoformat()},{'period_start_anchor_accession':anchor.accession,'period_start_anchor_end':str(anchor.period_end.date()),'anchor_available_at':anchor.available_at.isoformat()}],sort_keys=True)
        out['event_id']=hashlib.sha256((out['component_lineage']+str(row.period_end)).encode()).hexdigest();rows.append(out)
    events=pd.DataFrame(rows).drop_duplicates('event_id').sort_values(['period_end','available_at'])
    coverage=[]
    for quarter in pd.period_range('2018Q1','2026Q2',freq='Q'):
        g=events.loc[events.nominal_quarter.eq(str(quarter))];coverage.append({'nominal_quarter':str(quarter),'event_vintages':len(g),'first_available_at':str(g.available_at.min()) if len(g) else None,'status':'parsed_direct_USGAAP_EUR_quarter' if len(g) else 'missing_qualified_direct_quarter'})
    return events,pd.DataFrame(coverage),pd.DataFrame(gaps)


def run(root,staging='/tmp/asml_quarterly_profit_source'):
    import os
    import numpy as np
    import pandas as pd
    import qlib
    from qlib.data import D
    from qlib.workflow import R
    from src.data.qlib_native import build_qlib_store_from_frame
    from src.data.nas_market_data import archive_nas_recorder,materialize_nas_evidence
    os.environ.setdefault('MLFLOW_ALLOW_FILE_STORE','true')
    root=Path(root).resolve();staging=Path(staging);card=root/'research'/NAME
    qlib.init(provider_uri=str(root/'data/qlib/soxl'),region='us',kernels=1);calendar=pd.DatetimeIndex(D.calendar(start_time='2018-01-01',end_time='2026-09-30'))
    if (card/'summary.json').exists():
        s=json.loads((card/'summary.json').read_text())
        if not Path(s['recorder_local_dir']).exists():materialize_nas_evidence(card/'nas_binding.json',card=card)
        R.set_uri(s['recorder_uri']);rec=R.get_recorder(recorder_id=s['recorder_id'],experiment_id=s['experiment_id']);assert len(rec.load_object('analysis/events'))==s['event_vintages'];return s
    plan=json.loads((card/'plan.json').read_text());review=json.loads((card/'design_review.json').read_text());ph=hashlib.sha256((card/'plan.json').read_bytes()).hexdigest()
    if review['plan_sha256']!=ph or review['decision']!='approved_for_execution':raise ValueError('unapproved ASML source')
    receipts=json.loads((staging/'receipts.json').read_text())
    if len(receipts)>plan['max_http_requests'] or any(e['state']!='complete' for e in receipts):raise ValueError('source request budget/inflight state invalid')
    payloads={e['name']:(staging/e['name']).read_bytes() if (staging/e['name']).exists() else b'' for e in receipts}
    for e in receipts:
        if hashlib.sha256(payloads[e['name']]).hexdigest()!=e['sha256']:raise ValueError('ASML source bytes changed')
    targets=pd.read_csv(staging/'targets.csv').set_index('url',drop=False);rows=[];errors=[]
    for e in receipts:
        if e.get('valid') and e['url'] in targets.index:
            try:rows+=parse_statement(payloads[e['name']],targets.loc[e['url']].to_dict(),e)
            except ValueError as ex:errors.append({'source_url':e['url'],'error':str(ex)})
    raw=pd.DataFrame(rows);events,coverage,gaps=canonical_events(raw,calendar)
    state=[]
    for day in calendar:
        known=events.loc[events.eligible_session.le(day)]
        row={'instrument':'ASML_USGAAP_QUARTER','datetime':day,'reported_quarter_net_income_eur':np.nan,'quarter_end_day':np.nan,'quarter_start_day':np.nan,'quarter_age_days':np.nan,'known':0.}
        if len(known):
            known=known.loc[known.period_end.eq(known.period_end.max())];known=known.loc[known.available_at.eq(known.available_at.max())]
            if known.value.nunique()!=1:raise ValueError('conflicting ASML current-quarter source')
            event=known.iloc[0];row.update(reported_quarter_net_income_eur=event.value,quarter_end_day=(event.period_end-pd.Timestamp('1970-01-01')).days,quarter_start_day=(event.period_start-pd.Timestamp('1970-01-01')).days,quarter_age_days=(day-event.period_end).days,known=1.)
        state.append(row)
    state=pd.DataFrame(state);fields=[x for x in state if x not in ['instrument','datetime']];provider=root/'data/qlib'/NAME
    manifest=build_qlib_store_from_frame(state,provider_uri=provider,region='us',freq='day',as_of=plan['as_of'],source=NAME,field_map={f:f for f in fields})
    manifest.update(information_only=True,no_prices_or_targets=True,accounting_basis='US_GAAP',profit_concept='consolidated_reported_net_income',currency='EUR',direct_quarter_units='million_EUR_reported; canonicalEUR',nominal_quarter_only_for_coverage=True,true_fiscal_end_and_prior_known_end_anchors_retained=True,store_end='2026-09-30',target_quarter_end='2026Q2')
    (provider/'qlib_store_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');qlib.init(provider_uri=str(provider),region='us',kernels=1)
    back=D.features(['ASML_USGAAP_QUARTER'],['$'+x for x in fields],start_time=calendar[0],end_time=calendar[-1]).reset_index();merged=state.merge(back,on=['instrument','datetime'],validate='one_to_one')
    for field in fields:np.testing.assert_allclose(merged[field],merged['$'+field],rtol=1e-6,atol=1e-6,equal_nan=True)
    snapshot=root/'data/snapshots'/NAME;snapshot.mkdir(parents=True,exist_ok=True);events.to_parquet(snapshot/'events.parquet',index=False)
    policy={'policy':'slim_card_recorder_first','workflow_config':'workflow.yaml','recorder_only_artifacts':['inputs/*','analysis/events','analysis/raw_events','analysis/information_state','analysis/qlib_readback','sources/*']};tracking=card/'qlib_recorder'
    with R.start(experiment_name=NAME,uri=str(tracking)):
        rec=R.get_recorder();rec.save_objects(artifact_path='inputs',plan=plan,receipts=receipts,raw_payloads=payloads,document_targets=targets.reset_index(drop=True),source_calendar=calendar,provider_manifest=manifest)
        rec.save_objects(artifact_path='analysis',events=events,raw_events=raw,coverage=coverage,anchor_gaps=gaps,parser_errors=errors,information_state=state,qlib_readback=back)
        rec.save_objects(artifact_path='sources',module=Path(__file__).read_text())
        summary={'experiment':NAME,'decision':'ASML_direct_quarter_USGAAP_EUR_research_source_constructed','new_http_requests':len(receipts),'valid_responses':sum(e['valid'] for e in receipts),'source_documents':len(targets),'raw_three_month_vintages':len(raw),'event_vintages':len(events),'target_quarters':len(coverage),'parsed_target_quarters':int(coverage.status.eq('parsed_direct_USGAAP_EUR_quarter').sum()),'first_quarter':coverage.nominal_quarter.min(),'last_quarter':coverage.nominal_quarter.max(),'parser_errors':len(errors),'period_start_anchor_gaps':len(gaps),'source_precision_eur_values':sorted(events.source_precision_eur.unique().tolist()),'new_fits':0,'new_accounts':0,'new_target_series':0,'objective_achieved':False,'provider_uri':str(provider),'canonical_events_path':str(snapshot/'events.parquet'),'qlib_readback_rows':len(back),'region':'us','market':'ASML_USGAAP_QUARTER information instrument','as_of':plan['as_of'],'calendar_last':str(calendar[-1]),'plan_sha256':ph,'recorder_id':rec.id,'experiment_id':str(rec.experiment_id),'recorder_uri':str(tracking),'recorder_local_dir':str(tracking/str(rec.experiment_id)/rec.id),'artifact_policy':policy,'boundaries':['USGAAPconsolidatedreportednetincome not relabeledIFRS/parent','Actualfiscalendretained; priorgenuinefiscalendavailablebyrelease anchorsstart','Comparativequartersretainlaterpublication; no backwardrevision','EURmillion amountsscaled once; no EPS subtraction or currencyconversion','Through2026Q2 only; informationstore ends2026Sep beforelaterquarterrelease','Sourceonly; no financialinformation orportfoliogate asserted']}
        rec.save_objects(artifact_path='analysis',summary=summary);artifacts=Path(summary['recorder_local_dir'])/'artifacts';rec.log_metrics(registered_artifact_count=sum(p.is_file() for p in artifacts.rglob('*')),parsed_quarters=summary['parsed_target_quarters'],event_vintages=len(events))
    coverage.to_csv(card/'quarter_coverage.csv',index=False)
    (snapshot/'manifest.json').write_text(json.dumps({'source_card':str(card.relative_to(root)),'source_recorder_id':rec.id,'events_sha256':hashlib.sha256((snapshot/'events.parquet').read_bytes()).hexdigest(),'provider_uri':str(provider.relative_to(root)),'method_id':'CIK937966|US_GAAP|consolidated_reported_net_income|EUR|direct_reported_quarter'},indent=2)+'\n')
    binding=json.loads((root/'research/etf_foreign_quarterly_intake_20261016/nas_binding.json').read_text());binding.pop('evidence',None);(card/'nas_binding.json').write_text(json.dumps(binding,indent=2)+'\n');(card/'verification_manifest.json').write_text(json.dumps({'recorder_id':rec.id,'entries':[{'path':str(p.relative_to(card)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in sorted(artifacts.rglob('*')) if p.is_file()]},indent=2)+'\n')
    summary['nas_evidence']=archive_nas_recorder(card/'nas_binding.json',workspace_root=root,card=card,recorder_id=rec.id,experiment_id=str(rec.experiment_id));(card/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');return summary
