"""Budgeted primary AMD history intake. No market data, model or outcome access."""
import hashlib,json,re
from datetime import datetime,timezone
from pathlib import Path
from urllib.parse import urlparse,urljoin
import requests
from bs4 import BeautifulSoup
import pandas as pd
from src.data.minute_intake import atomic_json


class Intake:
    def __init__(self,root,card,plan):
        self.root,self.card,self.plan=root,card,plan
        self.directory=root/'data/raw/external'/plan['experiment'];self.directory.mkdir(parents=True,exist_ok=True)
        self.ledger_path=self.directory/'operation_ledger.json'
        self.ledger=json.loads(self.ledger_path.read_text()) if self.ledger_path.exists() else {'plan_sha256':hashlib.sha256((card/'plan.json').read_bytes()).hexdigest(),'operations':[]}
        assert self.ledger['plan_sha256']==hashlib.sha256((card/'plan.json').read_bytes()).hexdigest()
    def fetch(self,url,category):
        old=[x for x in self.ledger['operations'] if x['url']==url]
        if old:
            last=old[-1]
            if last.get('status')=='captured':
                raw=(self.directory/last['payload']).read_bytes();assert hashlib.sha256(raw).hexdigest()==last['sha256'];return raw,last
            return None,last  # No implicit retries after a failed/partial operation.
        caps={'financial_index':6,'all_index':7,'sec_metadata':4,'sec_index':28,'regular':28,'update':8,'bridge':4}
        if category not in caps:raise ValueError('unknown category')
        operations=self.ledger['operations']
        if sum(x['category']==category for x in operations)>=caps[category] or len(operations)>=140 or len({x['url'] for x in operations})>=85:raise ValueError('frozen retrieval budget exhausted')
        if urlparse(url).hostname not in self.plan['allowed_hosts']:raise ValueError('nonofficial host')
        item={'number':len(operations)+1,'url':url,'category':category,'operation':'HTTP_GET','observed_at_utc':datetime.now(timezone.utc).isoformat(),'status':'attempted','redirects':[]}
        operations.append(item);atomic_json(self.ledger,self.ledger_path)
        try:
            target=url
            for _ in range(6):
                if urlparse(target).hostname not in self.plan['allowed_hosts']:raise ValueError('redirect outside official hosts')
                response=requests.get(target,timeout=25,allow_redirects=False)
                if response.is_redirect:
                    item['redirects'].append({'url':target,'status':response.status_code,'location':response.headers.get('Location')});target=urljoin(target,response.headers['Location']);continue
                break
            else:raise ValueError('redirect limit')
            item.update(http_status=response.status_code,resolved_url=response.url,headers=dict(response.headers))
            raw=response.content;name=hashlib.sha256(url.encode()).hexdigest()[:24]+'.bin';(self.directory/name).write_bytes(raw)
            item.update(payload=name,sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),format='HTTP_response_bytes_not_historical_identity',status='captured' if response.status_code==200 else 'failed_HTTP')
        except Exception as error:item.update(status='failed_transport',error=str(error))
        atomic_json(self.ledger,self.ledger_path)
        return (raw if item['status']=='captured' else None),item
    def existing_index(self):
        spec=self.plan['existing_2011_probe'];raw=Path(spec['path']).read_bytes()
        assert hashlib.sha256(raw).hexdigest()==spec['sha256']
        path=self.directory/'existing_2011_index.html';path.write_bytes(raw)
        return raw,{'url':spec['url'],'sha256':spec['sha256'],'payload':path.name,'category':'existing_index','operation':'reuse','status':'captured'}


def catalog(raw,url):
    soup=BeautifulSoup(raw,'html.parser');rows=[]
    for a in soup.select('a[href*="/press-releases/detail/"]'):
        title=a.get_text(' ',strip=True)
        if not re.search(r'AMD.*(?:Reports.*Quarter|Preliminary|Revises.*(?:Revenue|Outlook)|Updates.*(?:Revenue|Outlook))',title,re.I):continue
        node=a
        time=None
        for _ in range(7):
            time=node.find('time',attrs={'datetime':True})
            if time is not None:break
            node=node.parent
            if node is None:break
        if time is None:continue
        stamp=pd.Timestamp(time['datetime']);day=str(stamp.date())
        q=re.search(r'(First|Second|Third|Fourth)\s+Quarter',title,re.I)
        year=re.search(r'20\d{2}',title)
        quarter=['first','second','third','fourth'].index(q[1].lower())+1 if q else None
        fiscalyear=int(year[0]) if year else stamp.year-(quarter==4 and stamp.month<=3)
        preliminary=bool(re.search(r'Preliminary|Revises|Updates',title,re.I))
        rows.append({'title':title,'url':urljoin(url,a['href']),'published_clock_raw':time['datetime'],'published_clock_display':time.get_text(' ',strip=True),'report_date':day,'reported_quarter':f'FY{fiscalyear}Q{quarter}' if quarter else None,'role':'update' if preliminary else 'regular','index_url':url})
    return rows


def collect_indexes(root,card,plan):
    intake=Intake(root,card,plan);raw,receipt=intake.existing_index();rows=catalog(raw,receipt['url']);receipts=[receipt]
    for url in plan['index_sources']['financial'][1:]:
        raw,receipt=intake.fetch(url,'financial_index');receipts.append(receipt)
        if raw:rows.extend(catalog(raw,url))
    expected={f'FY{p.year}Q{p.quarter}' for p in pd.period_range('2010Q4','2017Q3',freq='Q')}
    have={r['reported_quarter'] for r in rows if r['role']=='regular'}
    missing=expected-have
    years={int(q[2:6])+(q.endswith('Q4')) for q in missing}
    for url in plan['index_sources']['optional_all_category_missing_quarters_only']:
        if int(url.split('year=')[1]) not in years:continue
        raw,receipt=intake.fetch(url,'all_index');receipts.append(receipt)
        if raw:rows.extend(catalog(raw,url))
    rows=list({r['url']:r for r in rows}.values())
    atomic_json({'rows':rows,'index_receipts':receipts,'expected_quarters':sorted(expected),'missing_quarters':sorted(expected-{r['reported_quarter'] for r in rows if r['role']=='regular'})},intake.directory/'catalog.json')
    return intake,json.loads((intake.directory/'catalog.json').read_text())


def report_facts(raw,entry):
    """Original printed GAAP summary, including legacy preformatted tables."""
    text=re.sub(r'\s+',' ',BeautifulSoup(raw,'html.parser').get_text(' ',strip=True))
    quarter=entry['reported_quarter'];year=int(quarter[2:6]);q=int(quarter[-1])
    section=re.search(r'(?<!Non-)GAAP\s+Financial Results.*?(?=Non-GAAP\s+Financial Results|Quarterly Summary|$)',text,re.I)
    if not section:raise ValueError('originalGAAPquarterlysummary absent')
    section=section[0]
    header=re.search(r'Q([1-4])\s*[-– ]\s*(\d{2,4})',section)
    if not header or (int(header[1]),int(header[2])%100)!=(q,year%100):raise ValueError('firstGAAPsummaryperiod mismatch')
    amount=re.search(r'Revenue\s+\$?\s*([\d,.]+)\s*([BM])\b',section,re.I)
    if not amount:raise ValueError('printedcurrentquarterrevenueunit absent')
    number=amount[1].replace(',','');factor=1000 if amount[2].upper()=='B' else 1
    quantum=10**(-len(number.split('.')[1]) if '.' in number else 0)
    actual=float(number)*factor;half=quantum*factor/2
    start=re.search(r'Current Outlook',text,re.I)
    if not start:raise ValueError('originaloutlook absent')
    out=text[start.start():];out=out[:out.find('AMD Teleconference')] if 'AMD Teleconference' in out else out[:2200]
    target_match=re.search(r'(first|second|third|fourth) quarter (?:of )?(20\d{2})|Q([1-4])\s+(20\d{2})',out,re.I)
    if not target_match:raise ValueError('explicitguidancetarget absent')
    tq=int(target_match[3]) if target_match[3] else ['first','second','third','fourth'].index(target_match[1].lower())+1
    ty=int(target_match[4] or target_match[2]);target=f'FY{ty}Q{tq}'
    if (ty,tq)!=(year+(q==4),q%4+1):raise ValueError('guide notnextquarter')
    words={'zero':0,'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9,'ten':10}
    def numeric(s):return words[s.lower()] if s.lower() in words else float(s)
    pct=re.search(r'AMD expects(?: [^.]{0,45})? revenue to (increase|decrease)(?: approximately)? ([\d.]+|\w+) percent(?: sequentially)?,? plus or minus ([\d.]+|\w+) percent',out,re.I)
    base={'reported_quarter':quarter,'target_quarter':target,'report_date':entry['report_date'],'published_clock_raw':entry.get('published_clock_raw'),'source_url':entry['receipt']['resolved_url'],'source_sha256':entry['receipt']['sha256'],
        'actual_revenue_million':actual,'actual_rounding_half_million':half,'actual_original_cell':amount[0],'actual_summary_excerpt':section,'guide_excerpt':out,'guide_base_actual_million':actual,'guide_base_rounding_half_million':half,'full_text':text}
    if pct:
        mid=numeric(pct[2])*(1 if pct[1].lower()=='increase' else -1);width=numeric(pct[3]);base.update(guide_status='original_numeric_sequential_percent',guide_mid=mid,guide_lower=mid-width,guide_upper=mid+width,guide_unit='pct_sequential_revenue_growth',bounds_derivation='signedpublishedgrowthmidpoint +/- publisheddispersion percentagepoints; no revenueconversion/error here')
    elif re.search(r'flat to slightly down',out,re.I):base.update(guide_status='qualitative_original_no_numeric_midpoint',guide_mid=None,guide_lower=None,guide_upper=None,guide_unit='qualitative')
    else:base.update(guide_status='needs_data_numeric_guidance_parse',guide_mid=None,guide_lower=None,guide_upper=None,guide_unit=None)
    day=pd.Timestamp(entry['report_date']);variants=[day.strftime('%B %d, %Y'),day.strftime('%b. %d, %Y'),day.strftime('%m/%d/%y'),day.strftime('%B %-d, %Y'),day.strftime('%b. %-d, %Y')]
    if not any(v.lower() in text.lower() for v in variants):raise ValueError('fullsource release date not matched')
    return base
