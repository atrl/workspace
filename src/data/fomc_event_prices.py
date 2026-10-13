"""Official release clocks and additional observed pre-announcement minute marks."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
from pathlib import Path
import hashlib,json,re
import requests
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from src.data.minute_intake import atomic_json
from src.data.minute_price_basis import normalize_none_history,published_intraday_basis
from src.data.qlib_native import build_qlib_store_from_frame,QlibProviderConfig,load_qlib_feature_frame


def release_clock(raw,date):
    text=' '.join(BeautifulSoup(raw,'html.parser').get_text(' ',strip=True).split());day=pd.Timestamp(date)
    if 'Federal Reserve issues FOMC statement' not in text or not re.search(day.strftime('%B')+rf'\s+0?{day.day},\s+{day.year}',text):raise ValueError('not expected dated FOMC statement')
    m=re.search(r'For release at (\d{1,2}):(\d{2})\s+([ap])\.m\.\s+(EDT|EST)',text)
    if not m:raise ValueError('official statement clock unavailable')
    hour,minute,meridiem,zone=m.groups();hour=int(hour)%12+(12 if meridiem=='p' else 0)
    stamp=(day+pd.Timedelta(hours=hour,minutes=int(minute))).tz_localize('America/New_York')
    if stamp.tzname()!=zone:raise ValueError('statement timezone mismatch')
    return stamp


def collect_clocks(snapshot,dates):
    snapshot=Path(snapshot);snapshot.mkdir(parents=True,exist_ok=True);path=snapshot/'manifest.json';meta=json.loads(path.read_text()) if path.exists() else {}
    def get(date):
        key=str(pd.Timestamp(date).date());url='https://www.federalreserve.gov/newsevents/pressreleases/monetary'+pd.Timestamp(date).strftime('%Y%m%d')+'a.htm';file=snapshot/(key+'.html')
        if key in meta:
            raw=file.read_bytes()
            if hashlib.sha256(raw).hexdigest()!=meta[key]['sha256']:raise ValueError('clock source hash differs')
            return key,raw,meta[key]
        response=requests.get(url,timeout=45);response.raise_for_status();raw=response.content;stamp=release_clock(raw,key)
        file.write_bytes(raw);return key,raw,{'url':url,'final_url':response.url,'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'retrieved_at_utc':datetime.now(timezone.utc).isoformat(),'release_at':stamp.isoformat()}
    docs={}
    with ThreadPoolExecutor(max_workers=3) as pool:
        for key,raw,item in pool.map(get,dates):meta[key]=item;docs[key]=raw;atomic_json(meta,path)
    return meta,docs


def event_price_store(*,root,plan,exit_dates,provider):
    import qlib
    base=root/'data/qlib/full_path_tiered_allocation_20261013'
    base_manifest=json.loads((base/'qlib_store_manifest.json').read_text())
    if base_manifest['source_manifest_sha256']!=plan['source_id']:raise ValueError('base mark source differs')
    fields=['open','high','low','close','volume','execution_price','raw_execution_price']
    original=load_qlib_feature_frame(QlibProviderConfig(base,freq='1min'),instruments=['SOXL','SOXX'],fields=['$'+f for f in fields],start='2018-10-12',end=plan['as_of']+' 16:00:00').rename(columns=lambda x:x.lstrip('$')).reset_index()
    manifest_path=root/plan['source_manifest']
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest()!=plan['source_id']:raise ValueError('raw snapshot identity differs')
    receipt=json.loads(manifest_path.read_text());pieces=[];events={};desired=pd.DatetimeIndex(exit_dates).normalize()+pd.Timedelta(hours=14)
    for symbol in ['SOXL','SOXX']:
        entry=next(x for x in receipt['entries'] if x['symbol']==symbol);rehab=next(x for x in receipt['rehab'] if x['symbol']==symbol)
        for item in [entry,rehab]:
            if hashlib.sha256((manifest_path.parent/item['path']).read_bytes()).hexdigest()!=item['sha256']:raise ValueError('raw source object changed')
        raw=normalize_none_history(pd.read_parquet(manifest_path.parent/entry['path']),symbol=symbol,interval='1m',fetched_at=entry['fetched_at'])
        selected=raw.loc[raw.timestamp.dt.tz_localize(None).isin(desired)].copy()
        if len(selected)!=len(desired) or not selected.price_valid.all():raise ValueError('missing or invalid exit mark; no dropped events')
        pieces.append(selected);events[symbol]=pd.read_parquet(manifest_path.parent/rehab['path'])
    points,basis=published_intraday_basis(pd.concat(pieces,ignore_index=True),events,as_of=plan['as_of'])
    points['instrument']=points.symbol;points['datetime']=points.bar_start_at.dt.tz_localize(None);points['execution_price']=points.open;points['raw_execution_price']=points.raw_open
    for f in ['open','high','low','close']:points[f]=points.execution_price
    combined=pd.concat([original[['instrument','datetime',*fields]],points[['instrument','datetime',*fields]]],ignore_index=True).sort_values(['instrument','datetime'])
    if combined.duplicated(['instrument','datetime']).any():raise ValueError('exit mark duplicates existing slot')
    store=build_qlib_store_from_frame(combined,provider_uri=provider,freq='1min',as_of=plan['as_of'],field_map={f:f for f in fields},source='verified base native marks plus observed13:59 opening from14:00 right-edge bar')
    store.update(source_manifest_sha256=plan['source_id'],additional_mark_clock='13:59 open; input bar right edge14:00; no close from announcement bar',price_basis=basis,calendar_role='sparse observed marks, not complete-minute risk monitoring')
    atomic_json(store,provider/'qlib_store_manifest.json')
    native=load_qlib_feature_frame(QlibProviderConfig(provider,freq='1min'),instruments=['SOXL','SOXX'],fields=['$execution_price','$raw_execution_price'],start='2018-10-12',end=plan['as_of']+' 16:00:00')
    return native,store
