"""Dated original quarterly guidance facts; revisions remain separate audit events.

This source measures the original management-defined next-quarter outlook, not
the latest consensus/current forecast. Publisher/filing date bounds are conservative;
API/index timestamps are retained without inventing a timezone correction.
"""
from __future__ import annotations
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import urlparse
import numpy as np
import pandas as pd


def _require(ok, message):
    if not ok:
        raise ValueError('blocked: '+message)


def period_ordinal(value):
    match=re.fullmatch(r'FY(\d{4})Q([1-4])',str(value))
    _require(match is not None,'explicit fiscal quarter required')
    return int(match[1])*4+int(match[2])-1


def normalize_record(record, *, as_of):
    """Explicit adapters for the two captured fact schemas, with no value filling."""
    _require(not record.get('consensus_used',False) and not record.get('revised_xbrl_used',False),
             'consensus or revised actuals cannot replace original guidance facts')
    nested='guidance' in record
    guide=record.get('guidance',{}) if nested else record
    if nested:
        _require(record.get('actual_revenue',{}).get('currency')=='USD' and guide.get('currency')=='USD',
                 'explicit USD financial units required')
        _require(record.get('actual_revenue',{}).get('unit') in {'million','USD_million'},
                 'reported actual must be explicitly in USD million')
    actual=record.get('actual_revenue',{}).get('value') if nested else record.get('actual_revenue_million')
    reported=record.get('reported_fiscal_period') if nested else record.get('reported_fiscal_quarter')
    target=guide.get('target_fiscal_period') if nested else record.get('target_fiscal_quarter')
    unit=guide.get('unit') if nested else record.get('guidance_unit')
    lo,mid,hi=[guide.get(k) for k in (('lower','midpoint','upper') if nested else ('guidance_lower','guidance_mid','guidance_upper'))]
    release=pd.Timestamp(record['release_date']);filing=pd.Timestamp(record['filing_date'])
    _require(release.tzinfo is None and filing.tzinfo is None and not pd.isna(release) and not pd.isna(filing),'explicit publication dates required')
    _require(max(release,filing)<=pd.Timestamp(as_of),'after-cutoff financial disclosure')
    api=record.get('api_accepted_at_raw') if nested else record.get('api_accepted_original')
    stamp=pd.Timestamp(api)
    _require(stamp.tzinfo is not None,'API raw clock must remain timezone explicit, even though semantics are unverified')
    api_day=stamp.tz_convert('America/New_York').tz_localize(None).normalize()
    index_raw=record.get('index_accepted_at_raw') if nested else record.get('index_accepted_original')
    index_day=pd.Timestamp(index_raw).normalize() if index_raw else release
    # Dates only: preserve the original two strings; this is not an inferred instant.
    anchor=max(release,filing,api_day,index_day)
    _require(anchor<=pd.Timestamp(as_of),'after-cutoff conservative disclosure bound')
    sources=record.get('sources') or [{'url':record.get('source_url'),'source_kind':record.get('source_type')}]
    for source in sources:
        url=urlparse(str(source.get('url')))
        _require(url.scheme=='https' and url.hostname in {'www.sec.gov','nvidianews.nvidia.com','investor.nvidia.com','www.nvidia.com'},'unqualified primary source URL')
    is_next=actual is not None and reported is not None and target is not None and mid is not None
    if is_next:
        is_next=period_ordinal(target)==period_ordinal(reported)+1
    growth=np.nan
    if is_next:
        _require(np.isfinite(actual) and actual>0,'invalid reported quarterly revenue')
        _require(all(v is not None and np.isfinite(v) for v in (lo,mid,hi)) and lo<=mid<=hi,'invalid original guidance bounds')
        if unit in {'million','USD_million'}:
            _require(lo>0,'nonpositive revenue guidance')
            growth=float(mid/actual-1)
        elif unit=='pct_sequential_revenue_growth':
            _require(lo>-100,'invalid sequential percent guidance')
            growth=float(mid/100)
        else:
            raise ValueError('blocked: unsupported explicit guidance unit '+str(unit))
    scope=guide.get('scope','') if nested else record.get('guidance_scope','')
    return dict(accession=record['accession'],release_date=release,filing_date=filing,date_bound=anchor,
        api_accepted_raw=api,index_accepted_raw=index_raw,reported_quarter=reported,target_quarter=target,
        actual_revenue_million=actual,guidance_lower=lo,guidance_mid=mid,guidance_upper=hi,guidance_unit=unit,
        guidance_growth=growth,next_quarter_numeric=bool(is_next),guidance_scope=scope,
        event_role=record.get('event_role'),source_urls=[s['url'] for s in sources],
        original_html_bytes_verified=False,first_public_timestamp_verified=False)


def read_guidance_facts(root, manifest, *, as_of):
    rows=[];records=[]
    for entry in manifest['entries']:
        path=Path(root)/entry['path'];payload=path.read_bytes()
        _require(hashlib.sha256(payload).hexdigest()==entry['sha256'],'captured financial fact bytes changed')
        record=json.loads(payload)
        _require(record['accession']==entry['accession'],'fact accession differs from manifest')
        _require(record['cik']==manifest['cik'],'fact issuer differs from manifest')
        records.append(record);rows.append(normalize_record(record,as_of=as_of))
    frame=pd.DataFrame(rows).sort_values(['date_bound','accession']).reset_index(drop=True)
    _require(frame.accession.is_unique,'duplicate captured filing')
    return records,frame


def original_guidance_events(frame, *, expected_target_quarters):
    events=frame.loc[frame.next_quarter_numeric].copy()
    # Earliest fully disclosed guide for each fiscal target; revisions never overwrite it.
    events=events.drop_duplicates('target_quarter',keep='first').copy()
    expected=list(expected_target_quarters)
    _require(list(events.target_quarter)==expected,'incomplete or unordered original-quarter guidance inventory')
    _require(np.diff([period_ordinal(x) for x in expected]).tolist()==[1]*(len(expected)-1),'target quarters must be contiguous')
    return events.reset_index(drop=True)


def guidance_information_state(events, calendar):
    calendar=pd.DatetimeIndex(calendar)
    _require(not calendar.hasnans and calendar.is_unique and calendar.is_monotonic_increasing,'complete ordered native calendar required')
    events=events.copy()
    pos=calendar.searchsorted(pd.DatetimeIndex(events.date_bound),side='right')
    _require((pos<len(calendar)).all(),'source events lack a later native availability session')
    events['availability_calendar_index']=pos
    events['available_at']=(calendar[pos].tz_localize('America/New_York')+pd.Timedelta(hours=9,minutes=31)).tz_convert('UTC')
    _require(events.available_at.is_monotonic_increasing,'original guide availability is unordered')
    rows=[]
    for i,day in enumerate(calendar):
        visible=events.loc[events.availability_calendar_index<=i]
        if visible.empty:
            rows.append(dict(datetime=day,known=0.,guidance_qoq=np.nan,event_index=np.nan,availability_calendar_index=np.nan))
        else:
            j=visible.index[-1];event=events.loc[j]
            rows.append(dict(datetime=day,known=1.,guidance_qoq=float(event.guidance_growth),
                event_index=float(j),availability_calendar_index=float(event.availability_calendar_index)))
    return pd.DataFrame(rows),events


def build_guidance_store(state, *, provider_uri, as_of, instrument, source):
    from src.data.qlib_native import build_qlib_store_from_frame
    import qlib
    from qlib.data import D
    fields=['known','guidance_qoq','event_index','availability_calendar_index']
    frame=state[['datetime',*fields]].copy();frame['instrument']=instrument
    manifest=build_qlib_store_from_frame(frame,provider_uri=Path(provider_uri),as_of=as_of,
        source=source,field_map={k:k for k in fields},region='us')
    manifest.update(information_instrument=True,not_trading_prices=True,original_quarterly_guidance_only=True,
        latest_current_guidance_claimed=False,intraday_first_public_timestamp_verified=False,
        original_html_byte_identity_verified=False,
        availability_contract='09:31NY first actual native session strictly after max of source/filing/API-parsed NY/index dates; no instant correction')
    (Path(provider_uri)/'qlib_store_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    qlib.init(provider_uri=str(Path(provider_uri).resolve()),region='us',kernels=1)
    _require(pd.DatetimeIndex(D.calendar(freq='day')).equals(pd.DatetimeIndex(state.datetime)),'information calendar changed')
    readback=D.features([instrument],['$'+k for k in fields],freq='day')
    for k in fields:
        _require(np.allclose(readback['$'+k],state[k],equal_nan=True,rtol=0,atol=1e-7),'information readback differs: '+k)
    return manifest,readback
