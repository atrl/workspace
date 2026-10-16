"""Source-backed foreign quarterly income intake, with every request receipted."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import subprocess
import threading
from urllib.parse import urlencode
import pandas as pd

NAME='etf_foreign_quarterly_intake_20261016'

class Fetcher:
    def __init__(self, root, staging='/tmp/etf_foreign_quarterly_intake'):
        self.root=Path(root).resolve();self.card=self.root/'research'/NAME
        self.plan=json.loads((self.card/'plan.json').read_text())
        review=json.loads((self.card/'design_review.json').read_text())
        if review['plan_sha256']!=hashlib.sha256((self.card/'plan.json').read_bytes()).hexdigest() or review['decision']!='approved_for_execution':raise ValueError('unapproved intake')
        self.staging=Path(staging);self.staging.mkdir(parents=True,exist_ok=True)
        self.path=self.staging/'receipts.json';self.lock=threading.Lock()
        self.receipts=json.loads(self.path.read_text()) if self.path.exists() else []
    def get(self,url,kind='html'):
        with self.lock:
            prior=[e for e in self.receipts if e['url']==url]
            for e in reversed(prior):
                if e.get('valid'):
                    b=(self.staging/e['name']).read_bytes()
                    if hashlib.sha256(b).hexdigest()!=e['sha256']:raise ValueError('cached source changed')
                    return b,e
        for attempt in range(len(prior)+1,self.plan['transport']['max_attempts_per_url']+1):
            with self.lock:
                if len(self.receipts)>=self.plan['max_http_requests']:raise ValueError('reviewed request budget exhausted')
                key=hashlib.sha256(url.encode()).hexdigest()[:20];name=f'{key}_{attempt}.{kind}'
                e={'url':url,'name':name,'kind':kind,'attempt':attempt,'started_at':dt.datetime.now(dt.timezone.utc).isoformat(),'state':'running'}
                self.receipts.append(e);self.path.write_text(json.dumps(self.receipts,indent=2))
            p=self.staging/name
            r=subprocess.run(['curl','--compressed','-L','--connect-timeout','10','--max-time','180','--max-filesize','32000000','-sS','-H','User-Agent: SOXL research codex@openai.com','-w','%{http_code}',url,'-o',str(p)],capture_output=True,text=True)
            b=p.read_bytes() if p.exists() else b''
            valid=r.returncode==0 and r.stdout=='200'
            reason=''
            if valid and kind=='json':
                try:json.loads(b)
                except (ValueError,UnicodeDecodeError) as ex:valid=False;reason=str(ex)
            if valid and kind=='html':
                valid=b'<html' in b[:3000].lower() or b'<!doctype' in b[:3000].lower()
                if b'Your Request Originates from an Undeclared Automated Tool' in b:valid=False
            with self.lock:
                e.update(state='complete',status=int(r.stdout) if r.stdout.isdigit() else 0,curl_returncode=r.returncode,bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),retrieved_at=dt.datetime.now(dt.timezone.utc).isoformat(),valid=valid,error=reason or r.stderr[-300:]);self.path.write_text(json.dumps(self.receipts,indent=2))
            print(url,e['status'],len(b),valid,flush=True)
            if valid:return b,e
        return None,prior[-1] if prior else e


def submissions(fetcher,ticker,cik):
    payload,e=fetcher.get(f'https://data.sec.gov/submissions/CIK{cik:010d}.json','json')
    if payload is None:raise ValueError('submissions unavailable')
    data=json.loads(payload);frames=[pd.DataFrame(data['filings']['recent'])]
    for f in data['filings'].get('files',[]):
        if f['filingTo']>='2017-01-01' and f['filingFrom']<=fetcher.plan['as_of']:
            b,_=fetcher.get('https://data.sec.gov/submissions/'+f['name'],'json')
            if b is not None:frames.append(pd.DataFrame(json.loads(b)))
    result=pd.concat(frames,ignore_index=True).drop_duplicates('accessionNumber')
    result=result.loc[result.filingDate.between('2017-01-01',fetcher.plan['as_of'])].copy()
    result['ticker']=ticker;result['cik']=cik
    result.to_pickle(fetcher.staging/(ticker+'_submissions.pkl'))
    return result


def fulltext_search(fetcher,cik,query='"net income"',offset=0):
    params={'dateRange':'custom','startdt':'2018-01-01','enddt':fetcher.plan['as_of'],'forms':'6-K','ciks':f'{cik:010d}','q':query,'from':offset,'size':100}
    b,e=fetcher.get('https://efts.sec.gov/LATEST/search-index?'+urlencode(params),'json')
    return json.loads(b) if b else {}


def _text(node):
    return ' '.join(node.get_text(' ',strip=True).split())


def _amounts(text):
    import re
    found=[]
    for match in re.finditer(r'\(?\s*-?\d{1,3}(?:,\d{3})+(?:\.\d+)?\s*\)?',text):
        token=match.group().strip();negative=token.startswith('(') or token.startswith('-')
        value=float(token.replace(',','').replace('(','').replace(')','').replace('-','').strip())
        found.append(-value if negative else value)
    return found


def _basis(text):
    import re
    intro=text[:13000]
    for match in re.finditer(r'in accordance with ([^.]{0,650})',intro,re.I):
        clause=match.group(1)
        if re.search(r'endorsed|Republic of China|Regulations Governing',clause,re.I):return 'Taiwan_IFRS'
        if re.search(r'IASB|International Accounting Standards Board',clause,re.I):return 'IASB_IFRS'
    if 'TIFRS' in intro or 'Taiwan-IFRS' in intro:return 'Taiwan_IFRS'
    return 'unresolved_accounting_basis'


def _periods(header,amount_count):
    import re,calendar
    header=re.split(r'Except for Earnings per Share\)',header,flags=re.I)[-1]
    header=re.split(r'Operating revenues?|NET REVENUE|OTHER COMPREHENSIVE',header,maxsplit=1,flags=re.I)[0]
    durations={'three':3,'six':6,'nine':9,'twelve':12}
    groups=list(re.finditer(r'(Three|Six|Nine|Twelve)[ -]Months?(?: Periods?)? Ended ([A-Za-z]+)\s+(\d{1,2})',header,re.I))
    years=[int(y) for y in re.findall(r'\b20\d{2}\b',header)]
    if len(years)>amount_count:years=years[-amount_count:]
    if not groups:
        if len(years)==amount_count:
            return [(pd.Timestamp(y,1,1),pd.Timestamp(y,12,31),'annual') for y in years]
        return []
    if len(years)!=amount_count or len(years)%len(groups):return []
    per=len(years)//len(groups);result=[]
    for k,group in enumerate(groups):
        months=durations[group[1].lower()]
        try:month=list(calendar.month_name).index(group[2].capitalize())
        except ValueError:return []
        for year in years[k*per:(k+1)*per]:
            end=pd.Timestamp(year,month,int(group[3]));start=pd.Timestamp(year,month,1)-pd.DateOffset(months=months-1)
            result.append((start,end,'quarter' if months==3 else 'annual' if months==12 else 'ytd'))
    return result


def parse_document(payload,target,receipt):
    """Extract explicit parent-profit statement rows and dated board summaries."""
    from bs4 import BeautifulSoup
    import re
    soup=BeautifulSoup(payload,'lxml');text=_text(soup);basis=_basis(text)
    common={k:target[k] for k in ['ticker','cik','accession','filed','url']}
    common.update(source_url=common.pop('url'),accepted_at=target.get('accepted_at'),source_sha256=receipt['sha256'],currency='TWD',profit_concept='attributable_to_parent',accounting_basis=basis)
    events=[];diagnostics=[]
    for table_number,table in enumerate(soup.find_all('table')):
        table_text=_text(table)
        if not re.search(r'NET INCOME(?: \(LOSS\))? ATTRIBUTABLE TO:',table_text,re.I):continue
        # Main statements declare currency in the nearby document headings.
        if not re.search(r'(Thousands of New Taiwan Dollars|thousands of New Taiwan dollars)',text,re.I):continue
        trs=table.find_all('tr')
        for i,tr in enumerate(trs[:-1]):
            label=_text(tr)
            if not re.fullmatch(r'NET INCOME(?: \(LOSS\))? ATTRIBUTABLE TO:',label,re.I):continue
            owner=_text(trs[i+1])
            if not re.match(r'(?:Shareholders|Owners|Stockholders) of the parent',owner,re.I):continue
            values=_amounts(owner)
            header=table_text.split('Amount')[0][:1500]
            periods=_periods(header,len(values))
            if len(periods)!=len(values) or not periods:
                diagnostics.append({'table':table_number,'reason':'period_header_unresolved','header':header[:400],'owner_row':owner[:350]});continue
            maxend=max(x[1] for x in periods)
            for (start,end,kind),value in zip(periods,values):
                events.append({**common,'period_start':str(start.date()),'period_end':str(end.date()),'period_kind':kind,'value':value*1000,'source_unit':'thousand_TWD','extraction':'explicit_statement_parent_row','table_number':table_number,'comparative':end.year<maxend.year,'source_row':owner[:550],'eps':None,'shares':None})
    # Taiwan exchange board announcements publish unambiguous current YTD fields.
    period=re.search(r'(?:reporting period|period covered)[^:]{0,120}:\s*(20\d{2})[/-](\d{1,2})[/-](\d{1,2})\s*[~～–-]\s*(20\d{2})[/-](\d{1,2})[/-](\d{1,2})',text,re.I)
    ni=re.search(r'Profit \(loss\) during the period attributable to owners of parent accumulated from 1/1 to end of the period \(thousand NTD\)\s*:\s*(\(?-?[\d,]+\)?)',text,re.I)
    if period and ni:
        start=pd.Timestamp(int(period[1]),int(period[2]),int(period[3]));end=pd.Timestamp(int(period[4]),int(period[5]),int(period[6]));token=ni[1];value=float(token.replace(',','').replace('(','').replace(')',''))*(-1 if token.startswith('(') else 1)
        eps=re.search(r'Basic earnings \(loss\) per share accumulated from 1/1 to end of the period \(NTD\)\s*:\s*(\(?-?[\d.]+\)?)',text,re.I)
        events.append({**common,'accounting_basis':'Taiwan_IFRS','basis_evidence':'Taiwan_board_reporting_framework_research_assumption','period_start':str(start.date()),'period_end':str(end.date()),'period_kind':'annual' if (end-start).days>=330 else 'quarter' if (end-start).days<=100 else 'ytd','value':value*1000,'source_unit':'thousand_TWD','extraction':'explicit_board_parent_YTD','table_number':None,'comparative':False,'source_row':ni[0],'eps':float(eps[1].replace('(','-').replace(')','')) if eps else None,'shares':None})
    # Optional, explicitly labelled EPS/share facts from the same statement.
    auxiliary={}
    for table in soup.find_all('table'):
        table_text=_text(table)
        for tr in table.find_all('tr'):
            row=_text(tr);field=None;values=[];factor=1.
            if re.match(r'(Basic earnings per share|Earnings per share-basic)',row,re.I):
                field='eps';values=[float(x.replace('(','-').replace(')','').replace(' ','')) for x in re.findall(r'\(?-?\d+\.\d+\)?',row)]
            elif re.match(r'Weighted[- ]average number of (?:ordinary|common) shares for basic earnings per share',row,re.I) and re.search('thousand',row,re.I):
                field='shares';values=_amounts(row);factor=1000.
            if field and values:
                contexts=_periods(table_text.split('Amount')[0][:1500],len(values))
                if len(contexts)==len(values):
                    for (start,end,kind),value in zip(contexts,values):auxiliary.setdefault((str(start.date()),str(end.date())),{})[field]=value*factor
    for event in events:
        for field,value in auxiliary.get((event['period_start'],event['period_end']),{}).items():
            event[field]=value
            if field=='shares':event['shares_definition']='weighted_average_basic_ordinary_shares_not_point_in_time_outstanding'
    return events,diagnostics


def canonicalize(raw,calendar):
    """Retain disclosure vintages and derive only same-basis monetary Q4."""
    import numpy as np
    raw=raw.copy()
    raw['available_at']=pd.to_datetime(raw.accepted_at,utc=True,errors='coerce')
    raw['availability_basis']='SEC_acceptance_timestamp'
    for idx,row in raw.loc[raw.available_at.isna()].iterrows():
        pos=calendar.searchsorted(pd.Timestamp(row.filed),side='right')
        if pos<len(calendar):
            raw.loc[idx,'available_at']=pd.Timestamp(calendar[pos]).tz_localize('America/New_York').tz_convert('UTC')
            raw.loc[idx,'availability_basis']='conservative_next_exchange_session_after_filing_date'
    raw['period_end']=pd.to_datetime(raw.period_end);raw['period_start']=pd.to_datetime(raw.period_start)
    limit=pd.Timestamp('2026-10-16 23:59:59',tz='America/New_York').tz_convert('UTC')
    if raw.available_at.dropna().gt(limit).any():raise ValueError('source publication beyond frozen asof')
    if raw.period_end.gt(pd.Timestamp('2026-09-30')).any():raise ValueError('period beyond frozen scope')
    raw=raw.loc[raw.period_start.ge('2018-01-01') | (raw.period_kind.eq('annual') & raw.period_start.eq('2017-01-01'))].copy()
    raw['numerator_method_id']=raw.accounting_basis+'|'+raw.profit_concept+'|'+raw.currency
    raw['component_lineage']=raw.apply(lambda x:json.dumps([{'accession':x.accession,'source_url':x.source_url,'source_sha256':x.source_sha256,'period_start':str(x.period_start.date()),'period_end':str(x.period_end.date()),'value':x.value,'available_at':x.available_at.isoformat() if pd.notna(x.available_at) else None}],sort_keys=True),axis=1)
    raw=raw.drop_duplicates(['ticker','accession','period_start','period_end','numerator_method_id','value','extraction'])
    derived=[]
    for _,annual in raw.loc[raw.period_kind.eq('annual') & raw.period_start.ge('2018-01-01')].iterrows():
        nine=raw.loc[raw.ticker.eq(annual.ticker) & raw.numerator_method_id.eq(annual.numerator_method_id) & raw.period_start.eq(annual.period_start) & raw.period_end.eq(pd.Timestamp(annual.period_end.year,9,30)) & raw.available_at.le(annual.available_at)]
        if nine.empty:continue
        latest=nine.available_at.max();nine=nine.loc[nine.available_at.eq(latest)]
        if nine.value.nunique()!=1:continue
        nine=nine.iloc[0];row=annual.to_dict();row.update(period_start=pd.Timestamp(annual.period_end.year,10,1),period_kind='quarter',value=float(annual.value-nine.value),extraction='annual_minus_nine_month_same_basis',eps=np.nan,shares=np.nan,source_unit='TWD_derived',component_lineage=json.dumps(json.loads(annual.component_lineage)+json.loads(nine.component_lineage),sort_keys=True),basis_evidence='inherits_annual_and_nine_month_basis_evidence',comparative=bool(annual.comparative))
        derived.append(row)
    events=pd.concat([raw,pd.DataFrame(derived)],ignore_index=True)
    # Never assign a session beyond the actually loaded Qlib calendar.
    dates=events.available_at.dt.tz_convert('America/New_York').dt.tz_localize(None).dt.normalize()
    pos=calendar.searchsorted(dates,side='right')
    events['eligible_session']=[calendar[i] if i<len(calendar) else pd.NaT for i in pos]
    events['eligibility_convention']='next_exchange_session_after_SEC_publication_local_date'
    events['quarter']=events.period_end.dt.to_period('Q').astype(str)
    events['event_id']=events.apply(lambda x:hashlib.sha256((x.ticker+str(x.period_start)+str(x.period_end)+x.numerator_method_id+x.component_lineage).encode()).hexdigest(),axis=1)
    events=events.drop_duplicates('event_id').sort_values(['ticker','period_end','available_at','event_id'])
    rows=[]
    for ticker in ['TSM','UMC']:
        for q in pd.period_range('2018Q1','2026Q3',freq='Q'):
            found=events.loc[events.ticker.eq(ticker)&events.period_kind.eq('quarter')&events.quarter.eq(str(q))]
            rows.append({'ticker':ticker,'quarter':str(q),'event_vintages':len(found),'first_available_at':str(found.available_at.min()) if len(found) else None,'status':'parsed_parent_income' if len(found) else 'no_qualified_parent_statement_in_asof_filings','quarterly_value_first':float(found.sort_values('available_at').iloc[0].value) if len(found) else None})
    return events,pd.DataFrame(rows)


def information_state(events,calendar):
    """Information-only daily state: newer filings never rewind economic periods."""
    import numpy as np
    frames=[]
    for ticker in ['TSM','UMC']:
        entity=events.loc[events.ticker.eq(ticker)&events.accounting_basis.eq('Taiwan_IFRS')]
        for day in calendar:
            row={'instrument':ticker+'_TAIWAN_IFRS','datetime':day}
            known=entity.loc[entity.eligible_session.le(day)]
            for kind in ['annual','quarter']:
                subset=known.loc[known.period_kind.eq(kind)]
                if len(subset):
                    end=subset.period_end.max();subset=subset.loc[subset.period_end.eq(end)];available=subset.available_at.max();subset=subset.loc[subset.available_at.eq(available)]
                    if subset.value.nunique()!=1:raise ValueError('conflicting latest event values')
                    event=subset.iloc[0]
                    row[kind+'_parent_income_twd']=float(event.value)
                    row[kind+'_period_day']=float((event.period_end-pd.Timestamp('1970-01-01')).days)
                    row[kind+'_age_days']=float((day-event.period_end).days)
                    row[kind+'_known']=1.
                else:
                    row[kind+'_parent_income_twd']=np.nan;row[kind+'_period_day']=np.nan;row[kind+'_age_days']=np.nan;row[kind+'_known']=0.
            frames.append(row)
    return pd.DataFrame(frames)


def parser_identity():
    import inspect
    return hashlib.sha256('\n'.join(inspect.getsource(f) for f in [_text,_amounts,_basis,_periods,parse_document]).encode()).hexdigest()


def run(root,staging='/tmp/etf_foreign_quarterly_intake'):
    import os
    import qlib
    import numpy as np
    from qlib.data import D
    from qlib.workflow import R
    from src.data.qlib_native import build_qlib_store_from_frame
    from src.data.nas_market_data import archive_nas_recorder,materialize_nas_evidence
    os.environ.setdefault('MLFLOW_ALLOW_FILE_STORE','true')
    root=Path(root).resolve();staging=Path(staging);card=root/'research'/NAME
    qlib.init(provider_uri=str(root/'data/qlib/soxl'),region='us',kernels=1)
    calendar=pd.DatetimeIndex(D.calendar(start_time='2018-01-01',end_time='2026-10-16'))
    if (card/'summary.json').exists():
        summary=json.loads((card/'summary.json').read_text())
        if not Path(summary['recorder_local_dir']).exists():materialize_nas_evidence(card/'nas_binding.json',card=card)
        R.set_uri(summary['recorder_uri']);rec=R.get_recorder(recorder_id=summary['recorder_id'],experiment_id=summary['experiment_id'])
        events=rec.load_object('analysis/events');assert len(events)==summary['event_vintages'];return summary
    plan=json.loads((card/'plan.json').read_text());review=json.loads((card/'design_review.json').read_text());ph=hashlib.sha256((card/'plan.json').read_bytes()).hexdigest()
    if review['plan_sha256']!=ph or review['decision']!='approved_for_execution':raise ValueError('unapproved source scope')
    receipts=json.loads((staging/'receipts.json').read_text())
    if len(receipts)>plan['max_http_requests'] or any(e['state']!='complete' for e in receipts):raise ValueError('request budget/inflight state invalid')
    payloads={e['name']:(staging/e['name']).read_bytes() if (staging/e['name']).exists() else b'' for e in receipts}
    for e in receipts:
        if hashlib.sha256(payloads[e['name']]).hexdigest()!=e['sha256']:raise ValueError('raw response changed')
    targets=pd.read_csv(staging/'document_targets.csv').set_index('url',drop=False)
    cache_manifest=staging/'parse_cache_manifest.json'
    cache=json.loads(cache_manifest.read_text()) if cache_manifest.exists() else {}
    if cache.get('parser_identity')==parser_identity() and cache.get('receipts_sha256')==hashlib.sha256((staging/'receipts.json').read_bytes()).hexdigest() and cache.get('frame_sha256')==hashlib.sha256((staging/'parsed_final.pkl').read_bytes()).hexdigest():
        raw=pd.read_pickle(staging/'parsed_final.pkl');diagnostics=json.loads((staging/'parse_diagnostics.json').read_text())
    else:
        rows=[];diagnostics=[]
        for e in receipts:
            if e.get('valid') and e['url'] in targets.index:
                parsed,issues=parse_document(payloads[e['name']],targets.loc[e['url']].to_dict(),e);rows+=parsed;diagnostics+=[{'source_url':e['url'],**x} for x in issues]
        raw=pd.DataFrame(rows)
    events,coverage=canonicalize(raw,calendar)
    state=information_state(events,calendar)
    fields=[x for x in state if x not in ['instrument','datetime']]
    provider=root/'data/qlib'/NAME
    manifest=build_qlib_store_from_frame(state,provider_uri=provider,region='us',freq='day',as_of=plan['as_of'],source=NAME,field_map={f:f for f in fields})
    manifest.update(information_only=True,not_trading_prices=True,numerator='Taiwan_IFRS_parent_attributable_income_TWD',source_calendar_provider='data/qlib/soxl',calendar_last=str(calendar[-1]),availability='next_Qlib_session_after_publication_local_date; canonical_actual_SEC_timestamp_retained',quantization='Qlib float32; canonical Recorder events float64',source_plan_sha256=ph)
    (provider/'qlib_store_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    qlib.init(provider_uri=str(provider),region='us',kernels=1)
    back=D.features(['TSM_TAIWAN_IFRS','UMC_TAIWAN_IFRS'],['$'+f for f in fields],start_time=calendar[0],end_time=calendar[-1]).reset_index()
    compare=state.merge(back,on=['instrument','datetime'],validate='one_to_one')
    for field in fields:np.testing.assert_allclose(compare[field],compare['$'+field],rtol=1e-6,atol=1e-4,equal_nan=True)
    snapshot=root/'data/snapshots'/NAME;snapshot.mkdir(parents=True,exist_ok=True);events.to_parquet(snapshot/'events.parquet',index=False)
    policy={'policy':'slim_card_recorder_first','workflow_config':'workflow.yaml','recorder_only_artifacts':['inputs/*','analysis/events','analysis/raw_extracted_events','analysis/information_state','analysis/qlib_readback','sources/*']}
    tracking=card/'qlib_recorder'
    with R.start(experiment_name=NAME,uri=str(tracking)):
        rec=R.get_recorder();rec.save_objects(artifact_path='inputs',plan=plan,receipts=receipts,raw_payloads=payloads,document_targets=targets.reset_index(drop=True),source_calendar=calendar,provider_manifest=manifest)
        rec.save_objects(artifact_path='analysis',events=events,raw_extracted_events=raw,coverage=coverage,information_state=state,qlib_readback=back,parse_diagnostics=diagnostics)
        rec.save_objects(artifact_path='sources',module=Path(__file__).read_text())
        summary={'experiment':NAME,'decision':'continuous_parent_income_source_reconstructed_research_proxy','objective_achieved':False,'new_fits':0,'new_accounts':0,'new_http_attempts':len(receipts),'valid_responses':sum(e['valid'] for e in receipts),'primary_html_documents':len(targets),'raw_extracted_events':len(raw),'event_vintages':len(events),'quarter_coverage':coverage.groupby(['ticker','status']).size().unstack(fill_value=0).to_dict(orient='index'),'annual_periods':events.loc[events.period_kind.eq('annual')].groupby('ticker').period_end.nunique().to_dict(),'eps_nonmissing':int(events.eps.notna().sum()),'weighted_average_shares_nonmissing':int(events.shares.notna().sum()),'negative_parent_income_events':int(events.value.lt(0).sum()),'unresolved_period_headers':len(diagnostics),'provider_uri':str(provider),'canonical_events_path':str(snapshot/'events.parquet'),'region':'us','market':'TSM_TAIWAN_IFRS/UMC_TAIWAN_IFRS information instruments','as_of':plan['as_of'],'calendar_first':str(calendar[0]),'calendar_last':str(calendar[-1]),'qlib_readback_rows':len(back),'events_pending_calendar':int(events.eligible_session.isna().sum()),'recorder_id':rec.id,'experiment_id':str(rec.experiment_id),'recorder_uri':str(tracking),'recorder_local_dir':str(tracking/str(rec.experiment_id)/rec.id),'plan_sha256':ph,'parser_identity':parser_identity(),'artifact_policy':policy,'boundaries':['Taiwan_IFRS parentNI distinct from IASB20F/companyfacts; no crossbasis TTM','Weighted-average EPS shares are not date-specific marketcap shares','NI in nativeTWD; no convenienceUSD or ADR conversion synthesized','Exact SEC acceptance or explicit delayed eligibility; no backdating comparative periods','Source evidence only; no ETFvaluation/model/alpha promotion certified']}
        rec.save_objects(artifact_path='analysis',summary=summary);artifacts=Path(summary['recorder_local_dir'])/'artifacts';rec.log_metrics(registered_artifact_count=sum(p.is_file() for p in artifacts.rglob('*')),event_vintages=len(events),quarter_periods_parsed=int(coverage.status.eq('parsed_parent_income').sum()))
    coverage.to_csv(card/'quarter_coverage.csv',index=False)
    annual=events.loc[events.period_kind.eq('annual')&~events.comparative].sort_values('available_at').drop_duplicates(['ticker','period_end','accounting_basis']);annual[['ticker','period_end','value','currency','accounting_basis','available_at','accession','eps','shares']].to_csv(card/'annual_summary.csv',index=False)
    (snapshot/'manifest.json').write_text(json.dumps({'source_card':str(card.relative_to(root)),'source_recorder_id':rec.id,'events_sha256':hashlib.sha256((snapshot/'events.parquet').read_bytes()).hexdigest(),'provider_uri':str(provider.relative_to(root)),'source_plan_sha256':ph,'canonical_not_model_input_without_qualification':True},indent=2)+'\n')
    binding=json.loads((root/'research/etf_pit_fundamentals_repair_20261016/nas_binding.json').read_text());binding.pop('evidence',None);(card/'nas_binding.json').write_text(json.dumps(binding,indent=2)+'\n')
    (card/'verification_manifest.json').write_text(json.dumps({'recorder_id':rec.id,'entries':[{'path':str(p.relative_to(card)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in sorted(artifacts.rglob('*')) if p.is_file()]},indent=2)+'\n')
    summary['nas_evidence']=archive_nas_recorder(card/'nas_binding.json',workspace_root=root,card=card,recorder_id=rec.id,experiment_id=str(rec.experiment_id));(card/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');return summary
