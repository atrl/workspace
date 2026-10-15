"""Bounded official CPI release-header corpus; no CPI values or consensus."""
import hashlib,re
from datetime import datetime,timezone
from urllib.parse import urljoin,urlparse
from urllib.request import Request,urlopen
from urllib.error import HTTPError
import pandas as pd
from bs4 import BeautifulSoup


def selected_urls(raw,index_url):
    urls={}
    for anchor in BeautifulSoup(raw,'html.parser').find_all('a',href=True):
        url=urljoin(index_url,anchor['href']);parsed=urlparse(url)
        match=re.fullmatch(r'/news.release/archives/cpi_(\d{8})\.htm',parsed.path)
        if parsed.scheme!='https' or parsed.netloc!='www.bls.gov' or not match:continue
        date=pd.to_datetime(match[1],format='%m%d%Y')
        if pd.Timestamp('2019-01-01')<=date<=pd.Timestamp('2022-12-31'):urls[url]=date
    if len(urls)!=48 or any(sum(d.year==year for d in urls.values())!=12 for year in range(2019,2023)):
        raise ValueError('official index does not identify exactly48 actual-year releases')
    return sorted(urls,key=urls.get)


def parse_header(raw,url):
    text=BeautifulSoup(raw,'html.parser').get_text(' ',strip=True);text=re.sub(r'\s+',' ',text)
    marker=re.search(r'Transmission of material in this release is embargoed until',text,re.I)
    if not marker:raise ValueError('embargo header missing')
    header=text[marker.start():marker.start()+400]
    pattern=r'8:30\s*a\.?m\.?\s*\((EDT|EST|ET)\)\s*(?:(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),?\s+)?([A-Za-z]+\s+\d{1,2},\s*\d{4})'
    match=re.search(pattern,header,re.I)
    if not match:raise ValueError('08:30 Eastern header clock not parsed')
    day=pd.Timestamp(match[2]);stamp=(day+pd.Timedelta(hours=8,minutes=30)).tz_localize('America/New_York')
    explicit=match[1].upper()
    if explicit in ['EDT','EST'] and stamp.tzname()!=explicit:raise ValueError('header offset conflicts with Eastern calendar')
    filename=re.search(r'cpi_(\d{8})\.htm$',url)
    if not filename or pd.to_datetime(filename[1],format='%m%d%Y')!=day:raise ValueError('header date differs from filename')
    month=re.search(r'CONSUMER PRICE INDEX\s*[-–—�]\s*([A-Za-z]+)\s+(\d{4})',text,re.I)
    if not month:raise ValueError('observation month heading absent')
    observed=pd.Timestamp(f'{month[1]} 1, {month[2]}').to_period('M')
    return {'url':url,'release_date':str(day.date()),'release_at':stamp.isoformat(),'observation_month':str(observed),'header':header[:match.end()],'source_sha256':hashlib.sha256(raw).hexdigest()}


def intake(plan,calendar,*,record_attempt):
    documents={};receipts=[];events=[]
    def fetch(url,name):
        record_attempt(url)
        retrieved=datetime.now(timezone.utc).isoformat()
        try:
            with urlopen(Request(url,headers={'User-Agent':'SOXL-research-source-audit/1.0'}),timeout=30) as response:
                raw=response.read();status=response.status
        except HTTPError as error:
            raw=error.read();status=error.code
        documents[name]=raw
        receipts.append({'url':url,'name':name,'status':status,'retrieved_at_utc':retrieved,'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)})
        if status!=200:raise ValueError(f'official source HTTP{status}: {url}')
        return raw
    failure=None
    try:
        index=fetch(plan['intake']['index_url'],'index.html');urls=selected_urls(index,plan['intake']['index_url'])
        for url in urls:
            raw=fetch(url,url.rsplit('/',1)[-1]);event=parse_header(raw,url)
            release=pd.Timestamp(event['release_at']);day=release.tz_localize(None).normalize()
            if day not in calendar or not release<=day.tz_localize('America/New_York')+pd.Timedelta(hours=9,minutes=46):raise ValueError('release outside required native decision calendar')
            events.append(event)
        periods={x['observation_month'] for x in events}
        if periods!=set(str(x) for x in pd.period_range('2018-12','2022-11',freq='M')):raise ValueError('observation-month coverage/boundaries differ')
        if len({x['release_date'] for x in events})!=48:raise ValueError('duplicate actual release date')
    except Exception as error:
        failure={'type':type(error).__name__,'reason':str(error)}
    return documents,receipts,pd.DataFrame(events),{'passed':failure is None and len(events)==48,'parsed_events':len(events),'attempted_requests':len(receipts),'failure':failure}
