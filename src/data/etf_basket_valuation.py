"""Signed, lagged disclosed-basket annual earnings yield; source-only research proxy."""
from pathlib import Path
import hashlib, io, json, os, re
import numpy as np
import pandas as pd

NAME='etf_basket_valuation_inputs_20261016'
PLAN_SHA='246f1cc8be72a2f2b7980e1d94a339c471d5cb66ff3a4cf6a7e835df57447ebf'
PARENT_TAGS={'us-gaap:NetIncomeLoss','ifrs-full:ProfitLossAttributableToOwnersOfParent'}
SHARE_TAGS={'dei:EntityCommonStockSharesOutstanding','us-gaap:CommonStockSharesOutstanding','ifrs-full:NumberOfSharesOutstanding'}


def scope_repair(card):
    path=card/'scope_repair_plan.json'
    if not path.exists():return None
    repair=json.loads(path.read_text());review=json.loads((card/'design_review_scope_repair.json').read_text())
    if review.get('plan_sha256')!=hashlib.sha256(path.read_bytes()).hexdigest() or not str(review.get('decision','')).startswith('approved'):raise ValueError('scope proof repair unapproved')
    return repair


def issuer_completion(card):
    path=card/'issuer_completion_plan.json'
    if not path.exists():return None
    value=json.loads(path.read_text());review=json.loads((card/'design_review_issuer_completion.json').read_text())
    if review.get('plan_sha256')!=hashlib.sha256(path.read_bytes()).hexdigest() or not str(review.get('decision','')).startswith('approved'):raise ValueError('issuer source completion unapproved')
    return value


def document_targets(plan,card):
    repair=scope_repair(card);completion=issuer_completion(card)
    return plan['source_acquisition']['documents']+(repair['documents'] if repair else [])+(completion['documents'] if completion else [])


def acquire(root,staging='/tmp/etf_basket_valuation'):
    import requests,signal
    from lxml import html
    from urllib.parse import urljoin,urlparse,parse_qs
    root=Path(root);card=root/'research'/NAME;staging=Path(staging);staging.mkdir(exist_ok=True)
    plan=json.loads((card/'plan.json').read_text());review=json.loads((card/'design_review.json').read_text())
    if hashlib.sha256((card/'plan.json').read_bytes()).hexdigest()!=PLAN_SHA or review.get('plan_sha256')!=PLAN_SHA or review.get('decision')!='approved_for_execution':raise ValueError('frozen source plan not approved')
    ledger=staging/'acquisition.json';records=json.loads(ledger.read_text()) if ledger.exists() else []
    def get(url,name):
        previous=next((x for x in records if x['name']==name),None)
        if previous:return (staging/name).read_bytes() if (staging/name).exists() else b'',previous
        repair=scope_repair(card)
        completion=issuer_completion(card)
        limit=plan['source_acquisition']['max_http_requests']+(repair['max_new_http_requests'] if repair else 0)+(completion['max_new_http_requests'] if completion else 0)
        if len(records)>=limit:raise ValueError('source request budget')
        record={'name':name,'url':url,'attempted':True};records.append(record);ledger.write_text(json.dumps(records,indent=2)+'\n')
        def expired(*args):raise TimeoutError('source60second wallclock deadline')
        old=signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,60)
        payload=b''
        try:
            response=requests.get(url,timeout=30,headers={'User-Agent':'SOXL research codex@openai.com','Accept-Encoding':'gzip, deflate'});payload=response.content
            record.update(status=response.status_code,bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest());(staging/name).write_bytes(payload)
        except (requests.RequestException,TimeoutError) as error:record.update(status=0,error=str(error))
        finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old)
        ledger.write_text(json.dumps(records,indent=2)+'\n');print('source',name,record.get('status'),flush=True);return payload,record
    for item in document_targets(plan,card):
        index,record=get(item['index_url'],item['accn']+'.index.html')
        if record.get('status')!=200:continue
        tree=html.fromstring(index);links=[]
        for row in tree.xpath('//tr'):
            cells=row.xpath('./td')
            if len(cells)>=4 and cells[3].text_content().strip()==item['form']:
                links.extend(x for x in cells[2].xpath('.//a/@href') if re.search(r'\.html?(?:$|[?#])',x,re.I))
        if len(links)!=1:continue
        url=urljoin('https://www.sec.gov',links[0])
        if '/ix?doc=' in url or '/ixviewer/' in url:url=urljoin('https://www.sec.gov',parse_qs(urlparse(url).query)['doc'][0])
        expected=f'/Archives/edgar/data/{item["cik"]}/{item["accn"].replace("-","")}/'
        if urlparse(url).netloc!='www.sec.gov' or not urlparse(url).path.startswith(expected):raise ValueError('document leaves frozen accession')
        get(url,item['accn']+'.html')
    for fx in plan['source_acquisition']['fx']:get(fx['download_url'],fx['name']+'.csv')
    completion=issuer_completion(card)
    if completion:
        event=completion['corporate_event_source'];get(event['url'],event['name'])
    return records


def parse_original(payload,item):
    """Extract original dimension-free income/share tags and explicit unit measures."""
    from lxml import etree,html
    tree=etree.fromstring(payload,parser=etree.XMLParser(recover=True,resolve_entities=False,huge_tree=True))
    if tree is None:return pd.DataFrame(columns=['cik','taxonomy','tag','unit','start','end','val','filed','accn','form']),''
    def local(tag):return tag.split('}')[-1].split(':')[-1].lower() if isinstance(tag,str) else ''
    def qname(value,node):
        prefix,name=value.split(':',1) if ':' in value else (None,value)
        return '{'+str(node.nsmap.get(prefix,''))+'}'+name
    contexts={};units={};rows=[]
    for e in tree.iter():
        if local(e.tag)=='context':
            d={};bad=False
            for c in e.iter():
                tag=local(c.tag)
                if tag in ['startdate','enddate','instant']:d[tag]=c.text
                if tag=='identifier':d['cik']=c.text;d['scheme']=c.get('scheme')
                if tag in ['segment','scenario','typedmember','explicitmember']:bad=True
            if not bad:contexts[e.get('id')]=d
        elif local(e.tag)=='unit':
            top=[];bottom=[]
            for c in e.iter():
                if local(c.tag)=='measure':
                    (bottom if 'unitdenominator' in {local(a.tag) for a in c.iterancestors()} else top).append(qname(c.text.strip(),c))
            units[e.get('id')]=(top,bottom)
    allowed={x.split(':')[1] for x in PARENT_TAGS|SHARE_TAGS}|{'ProfitLoss','NetIncomeLossAttributableToNoncontrollingInterest','NetIncomeLossAvailableToCommonStockholdersBasic'}
    for e in tree.iter():
        if not isinstance(e.tag,str):continue
        attrs={k.lower():v for k,v in e.attrib.items()};full=qname(attrs['name'],e) if 'name'in attrs else e.tag
        if '}' not in full:continue
        ns,tag=full[1:].split('}',1)
        if tag not in allowed:continue
        taxonomy='us-gaap' if '/us-gaap/'in ns else ('ifrs-full' if 'ifrs' in ns else ('dei' if '/dei/' in ns else None))
        if not taxonomy:continue
        ctx=contexts.get(attrs.get('contextref'),{});unit=units.get(attrs.get('unitref'),([],[]))
        if str(ctx.get('cik','')).lstrip('0')!=str(item['cik']) or str(ctx.get('scheme','')).lower().rstrip('/')not in ['http://www.sec.gov/cik','https://www.sec.gov/cik']:continue
        if unit[1] or len(unit[0])!=1:continue
        measure=unit[0][0];currency=measure.split('}')[-1]
        if currency not in ['USD','TWD','EUR','shares']:continue
        text=''.join(e.itertext()).strip().replace(',','').replace(' ','').replace('\u00a0','')
        try:
            value=float(text)*10**int(attrs.get('scale','0'))
            if attrs.get('sign')=='-':value=-abs(value)
        except (ValueError,TypeError):continue
        rows.append({'cik':item['cik'],'taxonomy':taxonomy,'tag':tag,'unit':currency,'start':ctx.get('startdate'),'end':ctx.get('enddate',ctx.get('instant')),'val':value,'filed':item['filed'],'accn':item['accn'],'form':item['form'],'original_document_sha256':hashlib.sha256(payload).hexdigest(),'unit_measure':measure})
    normalized=' '.join(html.fromstring(payload).text_content().split())
    return pd.DataFrame(rows).drop_duplicates() if rows else pd.DataFrame(columns=['cik','taxonomy','tag','unit','start','end','val','filed','accn','form']),normalized


def raw_fact_frame(payloads):
    rows=[]
    wanted={x.split(':')[1] for x in PARENT_TAGS|SHARE_TAGS}|{'ProfitLoss','NetIncomeLossAttributableToNoncontrollingInterest','NetIncomeLossAvailableToCommonStockholdersBasic'}
    for payload in payloads.values():
        try:doc=json.loads(payload)
        except (ValueError,TypeError):continue
        if 'cik'not in doc:continue
        for taxonomy,tags in doc.get('facts',{}).items():
            for tag,item in tags.items():
                if tag not in wanted:continue
                for unit,facts in item.get('units',{}).items():
                    for fact in facts:rows.append({'cik':int(doc['cik']),'taxonomy':taxonomy,'tag':tag,'unit':unit,'source_description':item.get('description'),**fact})
    return pd.DataFrame(rows)


def prepare_facts(frame,calendar):
    frame=frame.copy()
    for col in ['start','end','filed']:frame[col]=pd.to_datetime(frame[col],errors='coerce')
    frame['concept']=frame.taxonomy+':'+frame.tag
    frame['duration']=(frame.end-frame.start).dt.days
    positions=calendar.searchsorted(frame.filed,side='right')
    frame['available_session']=[calendar[i] if i<len(calendar) else pd.NaT for i in positions]
    frame['vintage_concept']=frame.concept.where(~frame.concept.isin(PARENT_TAGS|{'source-proof:ParentAttributableNetIncome'}),'canonical_parent_attributable_income')
    keys=['cik','vintage_concept','unit','start','end']
    first=frame.groupby(keys,dropna=False).filed.transform('min')
    original=frame.loc[frame.filed.eq(first)].copy()
    original['conflict']=original.groupby(keys,dropna=False).val.transform('nunique').gt(1)
    original=original.sort_values(['filed','accn']).drop_duplicates(keys)
    return original


def annual_at(table,cik,cutoff,currency):
    known=table.loc[table.cik.eq(cik)&table.concept.isin(PARENT_TAGS|{'source-proof:ParentAttributableNetIncome'})&table.unit.eq(currency)&table.duration.between(330,380)&table.available_session.le(cutoff)&table.end.le(cutoff)]
    if known.empty:return None,None,'annual_income_missing'
    latest=known.sort_values(['end','filed']).iloc[-1]
    if latest.conflict or not np.isfinite(latest.val):return None,None,'annual_income_conflict'
    if (cutoff-latest.end).days>550:return None,None,'annual_income_stale'
    prior=known.loc[(latest.end-known.end).dt.days.between(350,380)&~known.conflict]
    previous=prior.sort_values(['end','filed']).iloc[-1] if len(prior) else None
    return latest,previous,None


def shares_at(table,cik,cutoff):
    known=table.loc[table.cik.eq(cik)&table.concept.isin(SHARE_TAGS)&table.unit.eq('shares')&table.available_session.le(cutoff)&table.end.le(cutoff)]
    if known.empty:return None,'shares_missing'
    known=known.loc[known.end.eq(known.end.max())].copy()
    known['priority']=known.concept.eq('dei:EntityCommonStockSharesOutstanding').astype(int)
    latest=known.sort_values(['priority','filed']).iloc[-1]
    if latest.conflict or not np.isfinite(latest.val) or latest.val<=0:return None,'shares_conflict_or_invalid'
    if (cutoff-latest.end).days>550:return None,'shares_stale'
    return latest,None


def fx_at(fx,currency,cutoff):
    if currency=='USD':return 1.,0,None
    name={'TWD':'DEXTAUS','EUR':'DEXUSEU'}.get(currency)
    if name not in fx:return None,None,'fx_missing'
    known=fx[name].loc[:cutoff-pd.Timedelta(days=7)].dropna()
    if known.empty:return None,None,'fx_missing'
    date=known.index[-1];value=float(known.iloc[-1]);age=(cutoff-date).days
    if age>14 or not np.isfinite(value) or value<=0:return None,age,'fx_stale_or_invalid'
    return (1/value if currency=='TWD' else value),age,None


def build_monthly(*,table,holdings,mapping,calendar,prices,rehab,old_wolf,fx,unit_review,plan):
    from src.data.earnings_price_basis import split_factor
    h=holdings.copy()
    for x in ['report_date','filing_date']:h[x]=pd.to_datetime(h[x])
    ix=calendar.searchsorted(h.filing_date,side='right');h['available_session']=[calendar[i] if i<len(calendar) else pd.NaT for i in ix]
    monthly=pd.Series(calendar,index=calendar).groupby(calendar.to_period('M')).max()
    dates=pd.DatetimeIndex(monthly[(monthly>=pd.Timestamp(plan['start']))&(monthly<=pd.Timestamp(plan['end']))])
    currency_map={'TSM':'TWD','UMC':'TWD','ASX':'TWD','ASML':'EUR'}
    foreign={'TSM','UMC','ASX','ASML','STM','ARM'}
    rows=[];details=[]
    for cutoff in dates:
        known=h.loc[h.available_session.le(cutoff)]
        common={'signal_at':(cutoff+pd.Timedelta(hours=16,minutes=15)).tz_localize('America/New_York'),'datetime':cutoff,'method_id':plan['method_id'],'source_qualified':False,'growth_source_qualified':False,'signed_annual_ep':np.nan,'annual_earnings_change':np.nan,'loss_weight':np.nan,'coverage_weight':0.,'growth_coverage_weight':0.,'unknown_weight':1.}
        if known.empty:rows.append({**common,'status':'holdings_not_yet_public'});continue
        basket=known.loc[known.report_date.eq(known.report_date.max())]
        basket=basket.loc[basket.filing_date.eq(basket.filing_date.max())]
        if basket.accession.nunique()!=1:raise ValueError('ambiguous latest disclosed basket')
        if not 0<basket.weight_pct.sum()<=101:raise ValueError('invalid disclosed weights')
        total=0.;coverage=0.;loss=0.;growth_cov=0.;growth_delta=0.;growth_base=0.;income_ages=[];share_ages=[];fx_ages=[];missing=[]
        for position in basket.itertuples():
            ticker=position.ticker;cik=int(mapping[ticker]);weight=float(position.weight_pct)/100
            if ticker=='MRVL' and position.report_date<pd.Timestamp('2021-04-20'):cik=1058057
            currency=currency_map.get(ticker,'USD')
            item={'signal_at':common['signal_at'],'ticker':ticker,'cik':cik,'weight':weight,'basket_id':position.accession,'currency':currency,'qualified':False}
            income,previous,error=annual_at(table,cik,cutoff,currency)
            shares,share_error=shares_at(table,cik,cutoff)
            error=error or share_error
            ratio=1.
            if ticker in foreign:
                proof=unit_review['listing_units'].get(ticker,{})
                if not proof.get('verified'):error=error or 'listing_unit_unverified'
                ratio=float(proof.get('ordinary_per_listed_unit',np.nan))
            rate,fx_age,fx_error=fx_at(fx,currency,cutoff);error=error or fx_error
            if ticker in ['CREE','WOLF']:
                quote=old_wolf.get(cutoff,np.nan);price_factor=1.;events=rehab.get('WOLF')
            else:
                quote=prices[ticker].get(cutoff,np.nan) if ticker in prices else np.nan;events=rehab.get(ticker)
                price_factor=split_factor(events,cutoff,plan['quote_snapshot_asof']) if events is not None else np.nan
            if not np.isfinite(quote) or quote<=0:error=error or 'qualified_quote_missing'
            if events is None:error=error or 'corporate_unit_history_missing'
            if shares is not None and events is not None:
                if split_factor(events,shares.end,shares.filed)!=1.:error=error or 'original_share_basis_ambiguous_intervening_split'
                share_factor=split_factor(events,shares.end,cutoff)
            else:share_factor=np.nan
            if error:
                item['missing_reason']=error;details.append(item);missing.append(ticker+':'+error);continue
            nominal=float(quote)/price_factor;decision_shares=float(shares.val)/share_factor
            cap=nominal*decision_shares/ratio
            if not np.isfinite(cap) or cap<=0:raise ValueError('invalid nominal marketcap estimate')
            ni=float(income.val)*rate;value=ni/cap
            # Same split-base change in price and sharecount leaves marketcap unchanged.
            np.testing.assert_allclose(cap,(nominal*price_factor)*(decision_shares/price_factor)/ratio,rtol=1e-12)
            total+=weight*value;coverage+=weight
            if ni<0:loss+=weight
            paired=previous is not None and np.isfinite(previous.val)
            if paired:
                previous_ni=float(previous.val)*rate;growth_cov+=weight;growth_delta+=weight*(ni-previous_ni)/cap;growth_base+=weight*abs(previous_ni)/cap
            ia=(cutoff-income.end).days;sa=(cutoff-shares.end).days;income_ages.append(ia);share_ages.append(sa);fx_ages.append(fx_age)
            item.update(qualified=True,parent_annual_income=float(income.val),income_usd=ni,income_accn=income.accn,income_end=str(income.end.date()),income_filed=str(income.filed.date()),income_concept=income.concept,income_age_days=ia,
                        shares_instant=str(shares.end.date()),shares_original_filed=str(shares.filed.date()),shares_accn=shares.accn,shares_concept=shares.concept,shares_reported=float(shares.val),shares_age_days=sa,shares_split_factor=share_factor,estimated_decision_ordinary_shares=decision_shares,
                        nominal_price=nominal,source_split_adjusted_price=float(quote),price_split_factor=price_factor,ordinary_per_listed_unit=ratio,fx_usd_per_reporting_currency=rate,fx_age_days=fx_age,estimated_marketcap_usd=cap,issuer_annual_ep=value,weighted_ep=weight*value,
                        prior_income=float(previous.val) if paired else np.nan,prior_income_end=str(previous.end.date()) if paired else None,prior_income_accn=previous.accn if paired else None,growth_pair_available=paired)
            details.append(item)
        ep_ok=coverage>=.95 and np.isfinite(total);growth_ok=growth_cov>=.95 and growth_base>0
        rows.append({**common,'basket_id':basket.accession.iloc[0],'holding_report_date':str(basket.report_date.iloc[0].date()),'holding_filed':str(basket.filing_date.iloc[0].date()),'holding_available_at':str(basket.available_session.iloc[0].date()),'holding_age_days':(cutoff-basket.report_date.iloc[0]).days,'disclosed_equity_weight':float(basket.weight_pct.sum()/100),
                     'signed_annual_ep':total if coverage else np.nan,'annual_earnings_change':growth_delta/growth_base if growth_ok else np.nan,'loss_weight':loss if coverage else np.nan,'coverage_weight':coverage,'growth_coverage_weight':growth_cov,'unknown_weight':max(0.,1-coverage),
                     'source_qualified':bool(ep_ok),'growth_source_qualified':bool(growth_ok),'max_annual_age_days':max(income_ages,default=np.nan),'max_share_age_days':max(share_ages,default=np.nan),'max_fx_age_days':max(fx_ages,default=np.nan),'status':'source_qualified_research_proxy' if ep_ok else 'insufficient_covered_weight','missing_reasons':';'.join(missing)})
    return pd.DataFrame(rows),pd.DataFrame(details)


def run(root,staging='/tmp/etf_basket_valuation'):
    import qlib,yaml
    from qlib.data import D
    from qlib.workflow import R
    from src.workflow.minute_action_value import _source
    from src.data.qlib_native import build_qlib_store_from_frame
    from src.data.minute_intake import atomic_json
    from src.data.nas_market_data import archive_nas_recorder
    os.environ.setdefault('MLFLOW_ALLOW_FILE_STORE','true')
    root=Path(root).resolve();staging=Path(staging);card=root/'research'/NAME
    plan=json.loads((card/'plan.json').read_text());review=json.loads((card/'design_review.json').read_text())
    if hashlib.sha256((card/'plan.json').read_bytes()).hexdigest()!=PLAN_SHA or review.get('plan_sha256')!=PLAN_SHA or review.get('decision')!='approved_for_execution':raise ValueError('frozen plan not approved')
    repair=scope_repair(card);completion=issuer_completion(card)
    completion_sha=hashlib.sha256((card/'issuer_completion_plan.json').read_bytes()).hexdigest() if completion else None
    repair_sha=hashlib.sha256((card/'scope_repair_plan.json').read_bytes()).hexdigest() if repair else None
    if (card/'summary.json').exists():
        existing=json.loads((card/'summary.json').read_text())
        if existing.get('scope_repair_plan_sha256')==repair_sha and existing.get('issuer_completion_plan_sha256')==completion_sha:return existing
    for path,digest in plan['source_files_sha256'].items():
        if hashlib.sha256((root/path).read_bytes()).hexdigest()!=digest:raise ValueError('source changed '+path)
    acquisition=acquire(root,staging)
    unit_review=json.loads((card/'basis_review.json').read_text())
    for item in unit_review['listing_units'].values():
        for proof in item['evidence']:
            if hashlib.sha256((staging/(proof['accn']+'.html')).read_bytes()).hexdigest()!=proof['source_sha256']:raise ValueError('listing proof changed')
    qlib.init(provider_uri=str(root/plan['provider']),region='us',kernels=1)
    sources,audit=_source(root,plan['source'],['inputs/raw_payloads','inputs/holdings','analysis/issuer_mapping'])
    full=raw_fact_frame(sources['inputs/raw_payloads']);mapping=sources['analysis/issuer_mapping'].set_index('ticker').cik.to_dict();holdings=sources['inputs/holdings']
    calendar=pd.DatetimeIndex(D.calendar(start_time='2010-01-01',end_time=plan['quote_snapshot_asof']))
    instruments=D.list_instruments(D.instruments('all'),as_list=True);wanted=[x for x in holdings.ticker.unique() if x in instruments]
    prices=D.features(wanted,['$close'],start_time=plan['start'],end_time=plan['quote_snapshot_asof'])['$close'].unstack('instrument')
    rehab={}
    for path in plan['source_files_sha256']:
        if path.endswith('_rehab.parquet'):
            symbol=Path(path).stem.replace('_rehab','');rehab[symbol]=pd.read_parquet(root/path)
            rehab[symbol]['split_ratio']=pd.to_numeric(rehab[symbol]['split_ratio'],errors='raise').astype(float)
    corporate_raw={}
    if completion:
        event=completion['corporate_event_source'];payload=(staging/event['name']).read_bytes()
        proof=unit_review.get('corporate_unit_sources',{}).get(event['symbol'],{})
        if not proof.get('verified') or hashlib.sha256(payload).hexdigest()!=proof.get('source_sha256'):raise ValueError('corporate unit source proof missing')
        response=json.loads(payload)['chart']['result'][0]
        if response['meta']['symbol']!=event['symbol']:raise ValueError('corporate source identity differs')
        records=[]
        for item in response.get('events',{}).get('splits',{}).values():
            day=pd.Timestamp(item['date'],unit='s',tz='UTC').tz_convert('America/New_York').strftime('%Y-%m-%d')
            if event['start']<=day<=event['end']:
                records.append({'ex_div_date':day,'split_ratio':float(item['denominator'])/float(item['numerator']),'cash_dividend':np.nan})
        for item in response.get('events',{}).get('dividends',{}).values():
            day=pd.Timestamp(item['date'],unit='s',tz='UTC').tz_convert('America/New_York').strftime('%Y-%m-%d')
            if event['start']<=day<=event['end']:
                records.append({'ex_div_date':day,'split_ratio':np.nan,'cash_dividend':float(item['amount'])})
        rehab[event['symbol']]=pd.DataFrame(records,columns=['ex_div_date','split_ratio','cash_dividend'])
        rehab[event['symbol']]['split_ratio']=pd.to_numeric(rehab[event['symbol']]['split_ratio'],errors='raise').astype(float)
        corporate_raw[event['symbol']]=payload
    qlib.init(provider_uri=str(root/'data/qlib/semis_wolf_recovery_20241126_v1'),region='us',kernels=1)
    old_wolf=D.features(['WOLF'],['$close'],start_time=plan['start'],end_time=plan['end'])['$close'].droplevel('instrument')
    originals={};original_tables={};original_text={};supplements=[]
    for item in document_targets(plan,card):
        path=staging/(item['accn']+'.html')
        if not path.exists():continue
        payload=path.read_bytes();originals[item['accn']]=payload;parsed,text=parse_original(payload,item);original_tables[item['accn']]=parsed;original_text[item['accn']]=text
        if len(parsed):supplements.append(parsed)
    if supplements:full=pd.concat([full,*supplements],ignore_index=True)
    for bridge in unit_review['income_bridges']:
        payload=originals[bridge['accn']]
        if hashlib.sha256(payload).hexdigest()!=bridge['source_sha256']:raise ValueError('income bridge proof changed')
        parsed=original_tables[bridge['accn']]
        selected=parsed.loc[parsed.start.eq(bridge['start'])&parsed.end.eq(bridge['end'])&parsed.unit.eq(bridge['unit'])&parsed.tag.eq(bridge['source_tag'].split(':')[-1])]
        if selected.empty and bridge.get('source_value_validation')=='source-bound_companyfacts_plus_visible_original_statement':
            selected=full.loc[full.cik.eq(bridge['cik'])&full.accn.eq(bridge['accn'])&full.start.eq(bridge['start'])&full.end.eq(bridge['end'])&full.unit.eq(bridge['unit'])&full.tag.eq(bridge['source_tag'].split(':')[-1])]
            displayed=bridge['statement_displayed_value'];multiplier=bridge['statement_unit_multiplier']
            if displayed*multiplier!=bridge['parent_income_value'] or f'{round(abs(displayed)):,}' not in bridge['evidence'] or not re.search('in thousands',original_text[bridge['accn']],re.I):raise ValueError('visible original amount/unit proof differs')
        if len(selected.val.unique())!=1 or selected.val.iloc[0]!=bridge['parent_income_value']:raise ValueError('income bridge value not original source')
        row=selected.iloc[0].to_dict();row.update(taxonomy='source-proof',tag='ParentAttributableNetIncome',source_description=bridge['rationale']);full=pd.concat([full,pd.DataFrame([row])],ignore_index=True)
    # Keep native-reporting currency and 20-F/IASB annual sources for Taiwan issuers.
    taiwan={1046179,1033767,1122411}
    full=full.loc[~full.cik.isin(taiwan)|full.form.isin(['20-F','20-F/A'])].copy()
    table=prepare_facts(full,calendar)
    fx={};fx_raw={}
    for item in plan['source_acquisition']['fx']:
        path=staging/(item['name']+'.csv')
        if not path.exists():continue
        fx_raw[item['name']]=path.read_bytes()
        data=pd.read_csv(io.BytesIO(fx_raw[item['name']]))
        if item['name']not in data:continue
        fx[item['name']]=pd.Series(pd.to_numeric(data[item['name']],errors='coerce').to_numpy(),index=pd.to_datetime(data.iloc[:,0])).sort_index()
    kwargs=dict(holdings=holdings,mapping=mapping,calendar=calendar,prices=prices,rehab=rehab,old_wolf=old_wolf,fx=fx,unit_review=unit_review,plan=plan)
    monthly,detail=build_monthly(table=table,**kwargs)
    # Append gross future revisions to the prepared first-vintage facts; no cutoff can use them.
    future=table.copy();future['available_session']=pd.Timestamp('2099-01-02');future['filed']=pd.Timestamp('2099-01-01');future['val']=1e30
    repeated,_=build_monthly(table=pd.concat([table,future],ignore_index=True),**kwargs)
    pd.testing.assert_frame_equal(monthly,repeated)
    if not ((detail.loc[detail.qualified,'income_filed'].map(pd.Timestamp).to_numpy()<pd.DatetimeIndex(detail.loc[detail.qualified,'signal_at']).tz_localize(None).normalize().to_numpy()).all()):raise ValueError('future income leakage')
    if not detail.loc[detail.qualified,'shares_reported'].gt(0).all():raise ValueError('invalid reported shares')
    fields=['signed_annual_ep','annual_earnings_change','loss_weight','coverage_weight','growth_coverage_weight','unknown_weight','max_annual_age_days','max_share_age_days','max_fx_age_days','source_qualified','growth_source_qualified']
    # A wholly missing optional growth field remains in lineage but cannot be an all-null Qlib feature.
    store_fields=[f for f in fields if monthly[f].notna().any()]
    ingestion=monthly[['datetime',*store_fields]].copy();ingestion['instrument']='SOXX_DISCLOSED_BASKET'
    for f in store_fields:ingestion[f]=pd.to_numeric(ingestion[f],errors='coerce').astype(float)
    provider=root/plan['output_provider'];manifest=build_qlib_store_from_frame(ingestion,provider_uri=provider,as_of=plan['end'],field_map={x:x for x in store_fields},source=NAME+':signed original annual parent income and dated share estimate')
    qlib.init(provider_uri=str(provider),region='us',kernels=1)
    features=D.features(['SOXX_DISCLOSED_BASKET'],['$'+x for x in store_fields],start_time=plan['start'],end_time=plan['end'])
    expected=ingestion.set_index(['instrument','datetime'])[store_fields].sort_index();actual=features.rename(columns=lambda x:x.lstrip('$')).sort_index()
    np.testing.assert_allclose(actual,expected,rtol=1e-6,atol=1e-8,equal_nan=True)
    # Qualification booleans retain full-precision decisions; values for models come from this Qlib readback.
    lineage=monthly.copy();lineage['source_qualified']=monthly.source_qualified.astype(bool);lineage['growth_source_qualified']=monthly.growth_source_qualified.astype(bool)
    stats=[]
    for year,g in lineage.groupby(lineage.datetime.dt.year):stats.append({'year':int(year),'months':len(g),'ep_qualified_months':int(g.source_qualified.sum()),'growth_qualified_months':int(g.growth_source_qualified.sum()),'min_coverage':float(g.coverage_weight.min()),'max_coverage':float(g.coverage_weight.max())})
    failure=detail.loc[~detail.qualified].groupby(['ticker','missing_reason']).agg(months=('weight','size'),mean_missing_weight=('weight','mean'),max_missing_weight=('weight','max')).reset_index()
    tracking=card/'qlib_recorder';policy={'policy':'slim_card_recorder_first','workflow_config':'workflow.yaml','recorder_only_artifacts':['inputs/*','sources/*','analysis/monthly_lineage','analysis/issuer_lineage','analysis/qlib_features','analysis/original_tables']}
    with R.start(experiment_name=NAME,uri=str(tracking)):
        rec=R.get_recorder();rec.save_objects(artifact_path='inputs',plan=plan,scope_repair=repair,issuer_completion=completion,source_audit=audit,basis_review=unit_review,acquisition=acquisition,original_documents=originals,corporate_event_payloads=corporate_raw,fx_raw=fx_raw,prepared_facts=table,holdings=holdings,prices=prices,old_wolf=old_wolf,rehab=rehab,provider_manifest=manifest)
        rec.save_objects(artifact_path='sources',module=Path(__file__).read_text())
        rec.save_objects(artifact_path='analysis',monthly_lineage=lineage,issuer_lineage=detail,qlib_features=features,original_tables=original_tables,year_coverage=pd.DataFrame(stats),missing_sources=failure)
        summary={'experiment':NAME,'decision':'signed_disclosed_basket_annual_EP_research_proxy_constructed','method_id':plan['method_id'],'objective_achieved':False,'new_fits':0,'new_accounts':0,'return_labels_read':False,'source_http_requests':len(acquisition),'monthly_rows':len(lineage),'ep_qualified_months':int(lineage.source_qualified.sum()),'growth_qualified_months':int(lineage.growth_source_qualified.sum()),'first_ep_qualified':str(lineage.loc[lineage.source_qualified,'datetime'].min().date()) if lineage.source_qualified.any() else None,'last_month':plan['end'],'issuer_rows':len(detail),'qualified_issuer_rows':int(detail.qualified.sum()),'negative_qualified_issuer_rows':int(detail.loc[detail.qualified,'parent_annual_income'].lt(0).sum()),'future_mutation_months_unchanged':len(lineage),'provider_uri':str(provider),'instrument':'SOXX_DISCLOSED_BASKET','field_map':{x:'$'+x for x in store_fields},'qualification_authority':'analysis/monthly_lineage.source_qualified and growth_source_qualified exact booleans','region':'us','as_of':plan['end'],'plan_sha256':PLAN_SHA,'scope_repair_plan_sha256':repair_sha,'issuer_completion_plan_sha256':completion_sha,'prior_issuer_completion_recorder_id':completion['prior_recorder_id'] if completion else None,'prior_source_recorder_id':repair['prior_recorder_id'] if repair else None,'recorder_id':rec.id,'experiment_id':str(rec.experiment_id),'recorder_uri':str(tracking),'recorder_local_dir':str(tracking/str(rec.experiment_id)/rec.id),'artifact_policy':policy,'limitations':['Research proxy not issuerPE, TTM, productionPIT or independent alpha evidence','Latest disclosed holdings weights and reported shares are stale estimates; no current ownership reconstruction','Parent profit can precede preferred dividends; common-share claim is not asserted','Intervening stable ADS listing ratios are research assumption backed by endpoint original filings and saved event histories','FX uses current downloaded vintage and7daylag, not archived originalpublication certification','Missingweights notrenormalized and unknownlossweight notzero; optionalgrowth/rank cannot erase primaryEP rows','All historical source observations have been used for research qualification, not independent evaluation']}
        rec.save_objects(artifact_path='analysis',summary=summary);artifacts=Path(summary['recorder_local_dir'])/'artifacts';rec.log_metrics(registered_artifact_count=sum(p.is_file() for p in artifacts.rglob('*')),ep_qualified_months=summary['ep_qualified_months'])
    pd.DataFrame(stats).to_csv(card/'year_coverage.csv',index=False);failure.to_csv(card/'missing_sources.csv',index=False)
    # Compact pre-label input table is permitted sidecar; detailed issuer histories remain Recorder-only.
    lineage.drop(columns=['missing_reasons'],errors='ignore').to_csv(card/'monthly_feature_summary.csv',index=False)
    atomic_json(manifest,card/'provider_manifest.json')
    atomic_json({'recorder_id':rec.id,'entries':[{'path':str(p.relative_to(card)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in sorted(artifacts.rglob('*')) if p.is_file()]},card/'verification_manifest.json')
    binding=json.loads((root/plan['source']['card']/'nas_binding.json').read_text());binding.pop('evidence',None);atomic_json(binding,card/'nas_binding.json')
    (card/'workflow.yaml').write_text(yaml.safe_dump({'experiment':NAME,'module':'src.data.etf_basket_valuation','plan':'plan.json','provider_uri':str(provider),'instrument':'SOXX_DISCLOSED_BASKET','artifact_policy':policy},sort_keys=False))
    summary['nas_evidence']=archive_nas_recorder(card/'nas_binding.json',workspace_root=root,card=card,recorder_id=rec.id,experiment_id=str(rec.experiment_id));atomic_json(summary,card/'summary.json');return summary
