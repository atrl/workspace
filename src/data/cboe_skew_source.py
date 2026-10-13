"""Immutable public Cboe SKEW intake and explicit missing-session store."""
from datetime import datetime, timezone
import hashlib,io,json
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from src.data.qlib_native import build_qlib_store_from_frame
from src.data.minute_intake import atomic_json


def read_skew(raw):
    frame=pd.read_csv(io.BytesIO(raw))
    if list(frame.columns)!=['DATE','SKEW']:raise ValueError('unexpected SKEW schema')
    dates=pd.to_datetime(frame.DATE,format='%m/%d/%Y',errors='raise')
    value=pd.to_numeric(frame.SKEW,errors='raise')
    if dates.duplicated().any() or not dates.is_monotonic_increasing or not np.isfinite(value).all() or not value.gt(0).all():raise ValueError('invalid source dates or values')
    return pd.Series(value.to_numpy(float),index=pd.DatetimeIndex(dates),name='skew')


def collect(snapshot,*,urls):
    snapshot=Path(snapshot);snapshot.mkdir(parents=True,exist_ok=True);manifest=snapshot/'manifest.json'
    if manifest.exists():
        meta=json.loads(manifest.read_text())
        for name,item in meta.items():
            if item['url']!=urls[name] or hashlib.sha256((snapshot/item['file']).read_bytes()).hexdigest()!=item['sha256']:raise ValueError('immutable source identity differs')
        return meta
    meta={}
    for name,url in urls.items():
        response=requests.get(url,timeout=45);response.raise_for_status();raw=response.content
        if name=='history':read_skew(raw)
        elif not raw.startswith(b'%PDF-'):raise ValueError('methodology response is not PDF')
        filename=name+('.csv' if name=='history' else '.pdf');destination=snapshot/filename
        if destination.exists() and destination.read_bytes()!=raw:raise ValueError('refusing changed snapshot bytes')
        destination.write_bytes(raw)
        meta[name]={'url':url,'final_url':response.url,'retrieved_at_utc':datetime.now(timezone.utc).isoformat(),'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'file':filename}
    atomic_json(meta,manifest);return meta


def build_store(values,*,reference_calendar,provider,as_of,source_sha256):
    # Union adds missing native sessions with NaN, never a copied observation.
    calendar=values.index.union(reference_calendar[reference_calendar<=values.index.max()]).sort_values()
    aligned=values.reindex(calendar)
    frame=pd.DataFrame({'instrument':'SKEW','datetime':calendar,'skew':aligned.to_numpy(),'observed':aligned.notna().astype(float).to_numpy()})
    manifest=build_qlib_store_from_frame(frame,provider_uri=Path(provider),field_map={'skew':'skew','observed':'observed'},as_of=as_of,source='official Cboe complete download '+source_sha256)
    import qlib
    from qlib.data import D
    qlib.init(provider_uri=str(provider),region='us',kernels=1)
    native=D.features(['SKEW'],['$skew','$observed'],start_time=calendar.min(),end_time=calendar.max(),freq='day').xs('SKEW',level='instrument')
    np.testing.assert_allclose(native['$skew'],aligned.to_numpy(),rtol=1e-6,atol=1e-5,equal_nan=True)
    np.testing.assert_array_equal(native['$observed'],aligned.notna().astype(float))
    if not pd.DatetimeIndex(D.calendar(freq='day')).equals(calendar):raise ValueError('native calendar differs')
    return native,manifest
