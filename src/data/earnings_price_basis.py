"""Source-only delayed annual E/P on explicit original share and price bases."""
from pathlib import Path
import hashlib
import io
import json
import re
import os
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd

NAME = 'etf_earnings_price_basis_20261016'
PLAN_SHA = 'ace4aade262d40ce9e8568fec75887e886dd3756b4128b6586521befc18a64f7'
EPS = 'EarningsPerShareDiluted'
SHARES = 'WeightedAverageNumberOfDilutedSharesOutstanding'


def original_annual_facts(doc):
    facts = pd.DataFrame(doc['facts']['us-gaap'][EPS]['units']['USD/shares'])
    facts['duration_days'] = (pd.to_datetime(facts.end)-pd.to_datetime(facts.start)).dt.days
    facts = facts.loc[facts.form.eq('10-K') & facts.duration_days.between(330, 380)].copy()
    first = facts.groupby(['start', 'end']).filed.transform('min')
    original = facts[facts.filed.eq(first)].copy()
    if original.groupby(['start','end']).val.nunique().gt(1).any():
        raise ValueError('ambiguous first annual EPS')
    original = original.sort_values(['end','filed','accn']).drop_duplicates(['start','end'])
    return facts, original


def split_factor(rehab, after, through):
    dates = pd.to_datetime(rehab.ex_div_date)
    values = rehab.loc[(dates > pd.Timestamp(after)) & (dates <= pd.Timestamp(through)), 'split_ratio'].dropna()
    if not np.isfinite(values).all() or (values <= 0).any(): raise ValueError('invalid split factors')
    return float(values.prod())


def inline_facts(payload):
    """Resolve original XBRL namespace, unit measures and context entity identity."""
    from lxml import etree
    text = payload.decode('utf-8', errors='replace')
    documents = re.findall(r'<DOCUMENT>(.*?)</DOCUMENT>', text, flags=re.S|re.I) or [text]
    rows = []
    def local(tag): return tag.split('}')[-1].split(':')[-1].lower() if isinstance(tag,str) else ''
    def qname(value, element):
        prefix, name = value.split(':',1) if ':' in value else (None,value)
        return '{'+str(element.nsmap.get(prefix,''))+'}'+name
    for document in documents:
        match = re.search(r'<TYPE>\s*([^\r\n<]+)', document, re.I)
        if match and match.group(1).strip() not in ['10-K','EX-101.INS','EX-101.INS.']:continue
        body = re.search(r'<TEXT>(.*?)(?:</TEXT>|$)', document, flags=re.S|re.I)
        body = body.group(1) if body else document
        tree = etree.fromstring(body.strip().encode(), parser=etree.XMLParser(recover=True, huge_tree=True, resolve_entities=False))
        if tree is None:continue
        contexts, units = {}, {}
        for element in tree.iter():
            tag=local(element.tag)
            if tag=='unit':
                numerator,denominator=[],[]
                for measure in element.iter():
                    if local(measure.tag)!='measure':continue
                    resolved=qname(measure.text.strip(),measure)
                    ancestors={local(a.tag) for a in measure.iterancestors()}
                    (denominator if 'unitdenominator' in ancestors else numerator).append(resolved)
                units[element.get('id')]={'numerator':tuple(numerator),'denominator':tuple(denominator)}
            elif tag=='context':
                context={};dimension=False
                for child in element.iter():
                    tag=local(child.tag)
                    if tag in ['startdate','enddate']:context[tag]=child.text
                    if tag=='identifier':context.update(entity_identifier=child.text,entity_scheme=child.get('scheme'))
                    if tag in ['explicitmember','typedmember','segment','scenario']:dimension=True
                if not dimension:contexts[element.get('id')]=context
        for element in tree.iter():
            if not isinstance(element.tag,str):continue
            attrs={k.lower():v for k,v in element.attrib.items()}
            qualified=qname(attrs['name'],element) if 'name' in attrs else element.tag
            if not qualified.startswith(('{http://fasb.org/us-gaap/','{https://fasb.org/us-gaap/')):continue
            name=qualified.split('}')[-1]
            if name not in [EPS,SHARES,'NetIncomeLoss']:continue
            context=contexts.get(attrs.get('contextref'),{})
            if not {'startdate','enddate','entity_identifier','entity_scheme'}<=set(context):continue
            unit=units.get(attrs.get('unitref'),{})
            value=''.join(element.itertext()).strip().replace(',','').replace(' ','').replace('\u00a0','')
            if value.startswith('(') and value.endswith(')'):value='-'+value[1:-1]
            try:
                value=float(value)*(10**int(attrs.get('scale','0')))
                if attrs.get('sign')=='-':value=-abs(value)
            except (ValueError,TypeError):continue
            filename=re.search(r'<FILENAME>\s*([^\r\n<]+)',document,re.I)
            rows.append({'document_filename':filename.group(1).strip() if filename else 'standalone','document_type':match.group(1).strip() if match else 'standalone','document_sha256':hashlib.sha256(body.encode()).hexdigest(),'tag':name,'namespace':qualified.split('}')[0][1:],'start':context['startdate'],'end':context['enddate'],'value':value,'unit_ref':attrs.get('unitref'),'unit_numerator':unit.get('numerator',()),'unit_denominator':unit.get('denominator',()),'entity_identifier':context['entity_identifier'],'entity_scheme':context['entity_scheme'],'original_document_fact':True})
    columns=['tag','start','end','value','unit_numerator','unit_denominator','entity_identifier','entity_scheme']
    return pd.DataFrame(rows).drop_duplicates() if rows else pd.DataFrame(columns=columns)


def original_basis_excerpts(payload, accession):
    """Retain exact normalized source text and character offsets for human review."""
    from lxml import html
    text=payload.decode('utf-8',errors='replace');documents=re.findall(r'<DOCUMENT>(.*?)</DOCUMENT>',text,re.S|re.I)
    rows=[]
    for document in documents:
        if not re.search(r'<TYPE>\s*10-K\s',document):continue
        body=re.search(r'<TEXT>(.*?)(?:</TEXT>|$)',document,re.S|re.I)
        if not body:continue
        normalized=' '.join(html.fromstring(body.group(1)).text_content().split())
        for match in re.finditer(r'weighted.average|stock split|retroactively adjusted|retrospectively adjusted',normalized,re.I):
            start=max(0,match.start()-160);end=min(len(normalized),match.end()+400)
            filename=re.search(r'<FILENAME>\s*([^\r\n<]+)',document,re.I)
            rows.append({'accession':accession,'document':'original10-K','document_filename':filename.group(1).strip() if filename else 'standalone','normalized_document_sha256':hashlib.sha256(normalized.encode()).hexdigest(),'normalization':'HTML text_content whitespace collapsed','start_offset':start,'end_offset':end,'excerpt':normalized[start:end]})
    return rows


def fetch_originals(plan, staging, repair, card):
    """Reuse completed submissions; retrieve only reviewed accession-bound primary HTML."""
    import requests
    from lxml import html
    from urllib.parse import urljoin, urlparse
    ledger=staging/'acquisition.json';records=json.loads(ledger.read_text())
    repairs=staging/'repair_acquisition.json'
    added=json.loads(repairs.read_text()) if repairs.exists() else []
    headers={'User-Agent':'SOXL research codex@openai.com','Accept-Encoding':'gzip, deflate'}
    def bounded_get(url):
        import signal
        def expired(signum, frame): raise TimeoutError('SEC request exceeded60second wallclock')
        previous=signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,60)
        try:return requests.get(url,headers=headers,timeout=30)
        finally:
            signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous)
    for item in repair['remaining_documents']:
        existing=next((x for x in added if x['accn']==item['accn']),None)
        resume_index=bool(existing and existing.get('index_status')==200 and not existing.get('primary_url') and existing.get('error')=='ambiguous original10K primarydocument')
        if existing and not resume_index:continue
        if not resume_index and 2*(len(added)+1)>repair['max_new_requests']:raise ValueError('repair request budget')
        record=existing if resume_index else {**item,'attempted':True}
        if not resume_index:added.append(record)
        else:record['index_selection_repair']='Use primary HTML, exclude explicitly labelled courtesy PDF; no index refetch.'
        repairs.write_text(json.dumps(added,indent=2)+'\n')
        try:
            if resume_index:index=(staging/(item['accn']+'.index.html')).read_bytes()
            else:
                response=bounded_get(item['index_url'])
                index=response.content;(staging/(item['accn']+'.index.html')).write_bytes(index)
                record['index_status']=response.status_code;record['index_sha256']=hashlib.sha256(index).hexdigest();record['index_bytes']=len(index)
            if record['index_status']!=200:raise ValueError('original filing index unavailable')
            tree=html.fromstring(index);links=[]
            for row in tree.xpath('//tr'):
                cells=row.xpath('./td')
                if len(cells)>=4 and cells[3].text_content().strip()=='10-K':links.extend(cells[2].xpath('.//a/@href'))
            links=[x for x in links if re.search(r'\.html?(?:$|[?#])',x,re.I)]
            if len(links)!=1:raise ValueError('ambiguous original10K primarydocument')
            url=urljoin('https://www.sec.gov',links[0])
            # SEC indexes may use an inline-viewer wrapper; resolve only its explicit doc path.
            if '/ixviewer/' in url or '/ix?doc=' in url:
                from urllib.parse import parse_qs
                url=urljoin('https://www.sec.gov',parse_qs(urlparse(url).query)['doc'][0])
            expected=f'/Archives/edgar/data/{int(item["cik"])}/{item["accn"].replace("-","")}/'
            if urlparse(url).netloc!='www.sec.gov' or not urlparse(url).path.startswith(expected):raise ValueError('primarydocument leaves frozenaccession')
            record['primary_url']=url;record['primary_filename']=urlparse(url).path.rsplit('/',1)[-1]
            repairs.write_text(json.dumps(added,indent=2)+'\n')
            response=bounded_get(url);payload=response.content
            record.update(status=response.status_code,bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest())
            (staging/(item['accn']+'.txt')).write_bytes(payload)
        except (requests.RequestException,ValueError,KeyError,TimeoutError) as error:record.update(status=0,error=str(error))
        repairs.write_text(json.dumps(added,indent=2)+'\n');print('primary annual source',item['accn'],record.get('status'),record.get('bytes'),flush=True)
    combined=records+added
    raw={r['accn']:(staging/(r['accn']+'.txt')).read_bytes() for r in combined if r.get('status')==200}
    indexes={r['accn']:(staging/(r['accn']+'.index.html')).read_bytes() for r in added if (staging/(r['accn']+'.index.html')).exists()}
    return combined,raw,indexes


def parse_envelope(payload, record):
    """Parser-only envelope; preserved HTTP source bytes remain unmodified."""
    if b'<DOCUMENT>' in payload[:10000]:return payload
    filename=record.get('primary_filename','unavailable.htm')
    return ('<DOCUMENT>\n<TYPE>10-K\n<FILENAME>'+filename+'\n<TEXT>\n').encode()+payload+b'\n</TEXT>\n</DOCUMENT>'


def run(root, staging='/tmp/etf_earnings_basis_originals'):
    import qlib
    import yaml
    from qlib.data import D
    from qlib.workflow import R
    from src.workflow.minute_action_value import _source
    from src.data.qlib_native import build_qlib_store_from_frame
    from src.data.nas_market_data import archive_nas_recorder
    from src.data.minute_intake import atomic_json
    os.environ.setdefault('MLFLOW_ALLOW_FILE_STORE','true')
    root=Path(root).resolve();card=root/'research'/NAME
    if hashlib.sha256((card/'plan.json').read_bytes()).hexdigest()!=PLAN_SHA:raise ValueError('plan changed')
    plan=json.loads((card/'plan.json').read_text());review=json.loads((card/'design_review.json').read_text())
    if review.get('plan_sha256')!=PLAN_SHA or review.get('decision')!='approved_for_execution':raise ValueError('design not approved')
    if (card/'summary.json').exists():return json.loads((card/'summary.json').read_text())
    for name,digest in plan['source_files_sha256'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:raise ValueError('source changed '+name)
    qlib.init(provider_uri=str(root/'data/qlib/soxx_pit_leveraged_training_v1'),region='us',kernels=1)
    objects,source_audit=_source(root,plan['facts_source'],['inputs/raw_payloads'])
    raw={s:objects['inputs/raw_payloads'][name] for s,name in plan['raw_payload_names'].items()}
    docs={s:json.loads(b) for s,b in raw.items()}
    prices=D.features(plan['symbols'],['$close'],start_time=plan['signal_start'],end_time=plan['price_snapshot_asof'])['$close'].unstack('instrument')
    calendar=pd.DatetimeIndex(D.calendar(start_time=plan['signal_start'],end_time=plan['price_snapshot_asof']))
    months=pd.Series(calendar,index=calendar).groupby(calendar.to_period('M')).max()
    decisions=pd.DatetimeIndex(months[(months>=pd.Timestamp(plan['signal_start']))&(months<=pd.Timestamp(plan['signal_end']))])
    rehab={s:pd.read_parquet(root/p) for s,p in {'NVDA':'data/raw/external/semis_minute_none_20261012_v1/NVDA_rehab.parquet','INTC':'data/raw/external/constituent_intraday_history_20261014/b02/INTC_rehab.parquet'}.items()}
    repair=json.loads((card/'acquisition_repair_plan.json').read_text())
    repair_review=json.loads((card/'design_review_acquisition_repair.json').read_text())
    if repair_review.get('plan_sha256')!=hashlib.sha256((card/'acquisition_repair_plan.json').read_bytes()).hexdigest() or repair_review.get('decision')!='approved_exact_filing_primary_document_transport_repair':raise ValueError('acquisition repair not approved')
    parser_review=json.loads((card/'design_review_parser_repair.json').read_text())
    if parser_review.get('plan_sha256')!=hashlib.sha256((card/'parser_repair_plan.json').read_bytes()).hexdigest() or parser_review.get('decision')!='approved_parser_repair_design':raise ValueError('parser repair not approved')
    acquisition,original_bytes,index_bytes=fetch_originals(plan,Path(staging),repair,card)
    basis_review={x['accn']:x for x in json.loads((card/'original_basis_review.json').read_text())['entries']}
    all_excerpts=[]
    verified=[];original_parsed={};annual_tables={};all_facts={}
    for symbol in plan['symbols']:
        all_facts[symbol],annual_tables[symbol]=original_annual_facts(docs[symbol])
    for item in plan['original_documents']:
        rec=[x for x in acquisition if x['accn']==item['accn']][-1]
        envelope=parse_envelope(original_bytes.get(item['accn'],b''),rec)
        parsed=inline_facts(envelope) if rec.get('status')==200 else inline_facts(b'')
        excerpts=original_basis_excerpts(envelope,item['accn']) if rec.get('status')==200 else []
        all_excerpts.extend(excerpts)
        original_parsed[item['accn']]=parsed
        values=parsed.loc[parsed.start.eq(item['start'])&parsed.end.eq(item['end'])]
        original=annual_tables[item['symbol']].loc[lambda x:x.accn.eq(item['accn']) & x.start.eq(item['start']) & x.end.eq(item['end'])].iloc[0]
        values=values.loc[pd.to_numeric(values.entity_identifier,errors='coerce').eq(int(item['cik'])) & values.entity_scheme.str.lower().str.rstrip('/').isin(['http://www.sec.gov/cik','https://www.sec.gov/cik'])]
        dollar=('{http://www.xbrl.org/2003/iso4217}USD',);share=('{http://www.xbrl.org/2003/instance}shares',)
        eps=values.loc[values.tag.eq(EPS)&values.unit_numerator.map(lambda x:x==dollar)&values.unit_denominator.map(lambda x:x==share),'value'].unique()
        shares=values.loc[values.tag.eq(SHARES)&values.unit_numerator.map(lambda x:x==share)&values.unit_denominator.map(lambda x:x==()),'value'].unique()
        facts_shares=docs[item['symbol']]['facts']['us-gaap'][SHARES]['units']['shares']
        original_shares={x['val'] for x in facts_shares if x.get('accn')==item['accn'] and x.get('start')==item['start'] and x.get('end')==item['end']}
        no_interim_split=split_factor(rehab[item['symbol']],item['end'],item['filed'])==1.
        manual=basis_review.get(item['accn'],{})
        split_in_year=split_factor(rehab[item['symbol']],item['start'],item['filed'])!=1.
        basis_valid=bool(manual.get('reviewed_original_annual_basis') and manual.get('source_sha256')==rec.get('sha256') and excerpts and (not split_in_year or manual.get('split_policy_reviewed')))
        valid=basis_valid and len(eps)==1 and len(shares)==1 and len(original_shares)==1 and np.isclose(eps[0],original.val,rtol=0,atol=1e-10) and np.isclose(shares[0],next(iter(original_shares)),rtol=0,atol=1.) and no_interim_split
        verified.append({**item,'http_status':rec.get('status'),'original_eps_count':len(eps),'original_shares_count':len(shares),'original_eps':float(eps[0]) if len(eps)==1 else None,'original_diluted_shares':float(shares[0]) if len(shares)==1 else None,'companyfacts_eps':float(original.val),'document_values_match':bool(valid),'original_units_and_CIK_checked':True,'manual_basis_verified':basis_valid,'effective_split_in_fiscal_period':split_in_year,'no_split_between_year_end_and_filing':no_interim_split,'qualification':'original_annual_document_and_share_units_matched' if valid else 'needs_original_document_or_share_basis'})
    verification=pd.DataFrame(verified).set_index('accn',drop=False)
    rows=[];revision_rows=[];checks=[]
    for symbol in plan['symbols']:
        originals=annual_tables[symbol]
        for date in decisions:
            eligible=originals.loc[pd.to_datetime(originals.filed).lt(date)&pd.to_datetime(originals.end).lt(date)]
            if eligible.empty:raise ValueError('missing annual original')
            chosen=eligible.sort_values('end').iloc[-1]
            quote=float(prices.loc[date,symbol]);price_factor=split_factor(rehab[symbol],date,plan['price_snapshot_asof'])
            nominal=quote/price_factor
            eps_factor=split_factor(rehab[symbol],chosen.filed,date)
            eps=float(chosen.val)*eps_factor
            if not np.isfinite([quote,nominal,eps]).all() or nominal<=0:raise ValueError('invalid valuation input')
            ep=eps/nominal
            invariant_error=abs(ep-(eps*price_factor)/quote)
            if invariant_error>1e-12:raise ValueError('split invariant broken')
            known_doc=json.loads(raw[symbol]);u=known_doc['facts']['us-gaap'][EPS]['units']['USD/shares'];known_doc['facts']['us-gaap'][EPS]['units']['USD/shares']=[x for x in u if pd.Timestamp(x['filed'])<date]
            _,truncated=original_annual_facts(known_doc);past=truncated.loc[pd.to_datetime(truncated.end).lt(date)].sort_values('end').iloc[-1]
            if (past.accn,past.val)!=(chosen.accn,chosen.val):raise ValueError('future filing altered vintage')
            known_doc['facts']['us-gaap'][EPS]['units']['USD/shares'].append({**chosen.to_dict(),'filed':'2099-01-01','accn':'synthetic_future_mutation','val':999999.})
            _,mutated=original_annual_facts(known_doc);again=mutated.loc[pd.to_datetime(mutated.filed).lt(date)&pd.to_datetime(mutated.end).lt(date)].sort_values('end').iloc[-1]
            if (again.accn,again.val)!=(chosen.accn,chosen.val):raise ValueError('future mutation changed source')
            proof=verification.loc[chosen.accn]
            rows.append({'instrument':symbol,'datetime':date,'source_accn':chosen.accn,'fiscal_start':chosen.start,'fiscal_end':chosen.end,'original_filed':chosen.filed,'source_eps':float(chosen.val),'price_split_factor':price_factor,'eps_split_factor':eps_factor,'split_adjusted_quote':quote,'nominal_price':nominal,'decision_share_eps':eps,'delayed_annual_ep':ep,'loss_flag':int(eps<0),'annual_age_days':(date-pd.Timestamp(chosen.end)).days,'qualified_source_row':int(proof.document_values_match),'invariance_error':invariant_error})
            checks.append({'instrument':symbol,'datetime':date,'future_truncation_unchanged':True,'future_mutation_unchanged':True,'same_day_filing_excluded':pd.Timestamp(chosen.filed)<date})
        for _,initial in originals.loc[originals.accn.isin(verification.index)].iterrows():
            later=all_facts[symbol].loc[lambda x:x.start.eq(initial.start)&x.end.eq(initial.end)&x.filed.gt(initial.filed)&x.val.ne(initial.val)]
            for _,v in later.iterrows():revision_rows.append({'instrument':symbol,'start':initial.start,'end':initial.end,'original_accn':initial.accn,'original_filed':initial.filed,'original_eps':initial.val,'later_accn':v.accn,'later_filed':v.filed,'later_eps':v.val,'used_to_overwrite_original':False})
    frame=pd.DataFrame(rows);checks=pd.DataFrame(checks);revisions=pd.DataFrame(revision_rows)
    xml=(root/'data/snapshots/semis_wolf_recovery_20241126_v1/000175272419177706.xml').read_bytes();tree=ET.fromstring(xml);anchors=[]
    for element in tree.iter():
        if element.tag.split('}')[-1]!='invstOrSec':continue
        values={e.tag.split('}')[-1]:e.text for e in element.iter() if e.text and e.text.strip()}
        symbol={'67066G104':'NVDA','458140100':'INTC'}.get(values.get('cusip'))
        if not symbol:continue
        implied=float(values['valUSD'])/float(values['balance']);row=frame.loc[frame.instrument.eq(symbol)&frame.datetime.eq(pd.Timestamp('2019-09-30'))].iloc[0]
        np.testing.assert_allclose(row.nominal_price,implied,rtol=1e-6)
        anchors.append({'instrument':symbol,'date':'2019-09-30','cusip':values['cusip'],'sec_value':float(values['valUSD']),'sec_shares':float(values['balance']),'sec_nominal_price':implied,'restored_nominal_price':row.nominal_price,'relative_error':row.nominal_price/implied-1})
    if len(anchors)!=2:raise ValueError('both nominal anchors required')
    fields=['source_eps','price_split_factor','eps_split_factor','split_adjusted_quote','nominal_price','decision_share_eps','delayed_annual_ep','loss_flag','annual_age_days','qualified_source_row']
    provider=root/plan['output_provider'];manifest=build_qlib_store_from_frame(frame[['instrument','datetime',*fields]],provider_uri=provider,as_of=plan['signal_end'],field_map={x:x for x in fields},source=f'{NAME}:frozen original accession and split sources')
    qlib.init(provider_uri=str(provider),region='us',kernels=1)
    final=D.features(plan['symbols'],['$'+x for x in fields],start_time=plan['signal_start'],end_time=plan['signal_end'])
    expected=frame.set_index(['instrument','datetime'])[fields].sort_index();actual=final.rename(columns=lambda x:x.lstrip('$')).sort_index()
    np.testing.assert_allclose(actual,expected,rtol=1e-6,atol=1e-7)
    # Final reported E/P and signs are read from Qlib, not the raw ingestion frame.
    stats=actual.groupby(level='instrument').agg(months=('delayed_annual_ep','count'),minimum_ep=('delayed_annual_ep','min'),maximum_ep=('delayed_annual_ep','max'),loss_months=('loss_flag','sum'),qualified_months=('qualified_source_row','sum'))
    stats=stats.reset_index();tracking=card/'qlib_recorder';policy={'policy':'slim_card_recorder_first','workflow_config':'workflow.yaml','recorder_only_artifacts':['inputs/*','analysis/monthly_lineage','analysis/qlib_features','analysis/original_parsed','analysis/vintage_checks','sources/*']}
    with R.start(experiment_name=NAME,uri=str(tracking)):
        rec=R.get_recorder();rec.save_objects(artifact_path='inputs',plan=plan,source_audit=source_audit,companyfacts_payloads=raw,original_documents=original_bytes,original_indexes=index_bytes,acquisition=acquisition,acquisition_repair=repair,parser_repair=json.loads((card/'parser_repair_plan.json').read_text()),manual_basis_review=basis_review,rehab=rehab,source_prices=prices,anchor_xml=xml,provider_manifest=manifest)
        rec.save_objects(artifact_path='sources',module=Path(__file__).read_text())
        rec.save_objects(artifact_path='analysis',monthly_lineage=frame,qlib_features=final,original_parsed=original_parsed,original_basis_excerpts=all_excerpts,vintage_checks=checks,annual_verification=verification,revision_differences=revisions,anchors=pd.DataFrame(anchors),issuer_summary=stats)
        summary={'experiment':NAME,'decision':'two_issuer_delayed_annual_EP_source_pilot_only','objective_achieved':False,'new_fits':0,'new_accounts':0,'new_market_fetches':0,'original_SEC_requests':len(acquisition)+len(index_bytes),'cancelled_complete_submission_requests':sum(bool(x.get('cancelled')) for x in acquisition),'original_annual_documents':len(verification),'matched_original_annual_documents':int(verification.document_values_match.sum()),'monthly_rows':len(final),'qualified_source_rows':int(actual.qualified_source_row.sum()),'loss_rows':int(actual.loss_flag.sum()),'max_split_invariance_error':float(frame.invariance_error.max()),'future_vintage_checks':len(checks),'provider_uri':str(provider),'market':'NVDA_INTC_only','region':'us','as_of':plan['signal_end'],'plan_sha256':PLAN_SHA,'qlib_docs':['https://qlib.readthedocs.io/en/latest/component/data.html','https://qlib.readthedocs.io/en/latest/component/recorder.html'],'recorder_id':rec.id,'experiment_id':str(rec.experiment_id),'recorder_uri':str(tracking),'recorder_local_dir':str(tracking/str(rec.experiment_id)/rec.id),'artifact_policy':policy,'limitations':['Delayed annual EPS, not TTM, not ETF aggregate, no model evidence','Only original document matched annual vintages qualify; companyfacts-only rows diagnostic','SEC filing day excluded conservatively; no first public earnings-release timestamp claim','EPS diluted weighted-average denominator is original annual reporting convention, not point-in-time shares outstanding','Historical price split correction is mechanical unit restoration, not future tradable information','2019 NPORT anchors verify price scale, not every daily market print','Full47basket, ADR/FX/acquired issuer qualification and independent forward remain unresolved']}
        rec.save_objects(artifact_path='analysis',summary=summary);artifacts=Path(summary['recorder_local_dir'])/'artifacts';rec.log_metrics(registered_artifact_count=sum(p.is_file() for p in artifacts.rglob('*')),qualified_source_rows=summary['qualified_source_rows'])
    atomic_json(all_excerpts,card/'original_basis_excerpts.json')
    for name,table in {'annual_verification':verification,'revision_differences':revisions,'nominal_anchors':pd.DataFrame(anchors),'issuer_summary':stats}.items():table.to_csv(card/(name+'.csv'),index=False)
    atomic_json({'recorder_id':rec.id,'entries':[{'path':str(p.relative_to(card)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in sorted(artifacts.rglob('*')) if p.is_file()]},card/'verification_manifest.json')
    binding=json.loads((root/plan['facts_source']['card']/'nas_binding.json').read_text());binding.pop('evidence',None);atomic_json(binding,card/'nas_binding.json')
    (card/'workflow.yaml').write_text(yaml.safe_dump({'experiment':NAME,'plan':'plan.json','module':'src.data.earnings_price_basis','provider_uri':str(provider),'artifact_policy':policy},sort_keys=False))
    summary['nas_evidence']=archive_nas_recorder(card/'nas_binding.json',workspace_root=root,card=card,recorder_id=rec.id,experiment_id=str(rec.experiment_id));atomic_json(summary,card/'summary.json');return summary
