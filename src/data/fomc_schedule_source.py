"""Dated next-meeting announcements from official minutes, not hindsight dates."""
from datetime import datetime,timezone
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urljoin
import re,json,hashlib
from pathlib import Path
import requests
import pandas as pd
from bs4 import BeautifulSoup
from src.data.minute_intake import atomic_json


def release_links(raw,base_url):
    soup=BeautifulSoup(raw,'html.parser');output=[]
    for a in soup.find_all('a',href=True):
        m=re.search(r'fomcminutes(\d{8})\.htm$',a['href'])
        if not m:continue
        parent=a.parent;text=parent.get_text(' ',strip=True)
        date=re.search(r'Released\s+([A-Za-z]+\s+\d{1,2},\s+\d{4})',text)
        if date is None:raise ValueError('minutes link lacks release date: '+a['href'])
        output.append({'minutes_meeting_end':str(pd.Timestamp(m.group(1)).date()),'release_date':str(pd.Timestamp(date.group(1)).date()),'url':urljoin(base_url,a['href'])})
    return output


def next_meeting(raw):
    text=' '.join(BeautifulSoup(raw,'html.parser').get_text(' ',strip=True).replace('\u200d','').split())
    sentence=re.search(r'It was agreed that the next meeting of the Committee would be held[^.]*\.',text)
    if not sentence:return {'parse_status':'no_next_meeting_sentence','next_meeting_start':None,'next_meeting_end':None}
    found=re.search(r'([A-Z][a-z]+)\s+(\d{1,2})\s*[-–—]\s*(?:([A-Z][a-z]+)\s+)?(\d{1,2}),?\s+(\d{4})',sentence.group())
    if not found:return {'parse_status':'unparsed_date','next_meeting_start':None,'next_meeting_end':None,'sentence':sentence.group()}
    month,first,second_month,last,year=found.groups()
    begin=pd.Timestamp(f'{month} {first}, {year}');end=pd.Timestamp(f'{second_month or month} {last}, {year}')
    if begin>end:raise ValueError('meeting dates reversed')
    return {'parse_status':'parsed','next_meeting_start':str(begin.date()),'next_meeting_end':str(end.date()),'sentence':sentence.group()}


def collect_sources(snapshot,plan):
    snapshot=Path(snapshot);snapshot.mkdir(parents=True,exist_ok=True);manifest_path=snapshot/'manifest.json'
    manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    def get(url):
        name=hashlib.sha256(url.encode()).hexdigest()+('.pdf' if url.endswith('.pdf') else '.html');path=snapshot/name
        if url in manifest:
            if hashlib.sha256(path.read_bytes()).hexdigest()!=manifest[url]['sha256']:raise ValueError('immutable document hash differs')
            return url,path.read_bytes(),manifest[url]
        response=requests.get(url,timeout=45);response.raise_for_status();raw=response.content
        if url.endswith('.pdf') and not raw.startswith(b'%PDF-'):raise ValueError('not a PDF')
        if path.exists() and path.read_bytes()!=raw:raise ValueError('source bytes changed during incomplete intake')
        path.write_bytes(raw)
        meta={'file':name,'url':url,'final_url':response.url,'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'retrieved_at_utc':datetime.now(timezone.utc).isoformat()}
        return url,raw,meta
    blobs={};links={}
    for url in plan['calendar_urls']:
        url,raw,meta=get(url);manifest[url]=meta;blobs[url]=raw;atomic_json(manifest,manifest_path)
        for row in release_links(raw,url):
            if pd.Timestamp(plan['earliest_minutes_meeting'])<=pd.Timestamp(row['minutes_meeting_end']) and pd.Timestamp(row['release_date'])<=pd.Timestamp(plan['as_of']):
                if row['url'] in links and links[row['url']]!=row:raise ValueError('calendar release metadata conflicts')
                links[row['url']]=row
    urls=[*links,plan['cancellation_url']]
    with ThreadPoolExecutor(max_workers=3) as pool:
        for url,raw,meta in pool.map(get,urls):
            manifest[url]=meta;blobs[url]=raw;atomic_json(manifest,manifest_path)
    rows=[{**row,**next_meeting(blobs[url])} for url,row in links.items()]
    return rows,manifest,blobs
