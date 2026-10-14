"""Date-level archived rates aligned after publication-day end, never effective-date backfill."""
import hashlib
from decimal import Decimal
import io
import json
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd



def reported_rate_spread_bp(rate_percent, reserve_percent):
    """Preserve ties in published decimal quotes, including fractional basis points."""
    if not np.isfinite([rate_percent, reserve_percent]).all():
        return np.nan
    return float((Decimal(str(rate_percent))-Decimal(str(reserve_percent)))*Decimal(100))


def read_archived_rates(directory):
    directory=Path(directory).resolve();manifest=json.loads((directory/'manifest.json').read_text());series=manifest['series_id'];parts=[]
    for entry in manifest['entries']:
        path=(directory/entry['path']).resolve()
        if not path.is_relative_to(directory) or hashlib.sha256(path.read_bytes()).hexdigest()!=entry['sha256']:
            raise ValueError('rate archive path/hash differs')
        archive=zipfile.ZipFile(io.BytesIO(path.read_bytes()));readme=archive.read('README.txt').decode()
        if 'Series ID: '+series not in readme or 'Output Format: Observations, Initial Release Only' not in readme:
            raise ValueError('rate archive format differs')
        frame=pd.read_csv(io.BytesIO(archive.read(entry['csv_name'])))
        if list(frame)!=['period_start_date',series,'realtime_start_date']:raise ValueError('rate columns differ')
        frame=frame.rename(columns={'period_start_date':'observation_date',series:'value','realtime_start_date':'release_date'})
        for name in ['observation_date','release_date']:frame[name]=pd.to_datetime(frame[name])
        frame['value']=pd.to_numeric(frame.value,errors='coerce');parts.append(frame)
    result=pd.concat(parts,ignore_index=True).sort_values('observation_date')
    if result.observation_date.duplicated().any():raise ValueError('duplicate rate observations')
    if result[['observation_date','release_date']].isna().any().any() or result.observation_date.gt(manifest['as_of']).any() or result.release_date.gt(manifest['as_of']).any():raise ValueError('unknown or future archive dates')
    return result,manifest


def funding_release_events(sofr,ioer,iorb,*,official_observation_dates,policy_events):
    official=pd.DatetimeIndex(official_observation_dates)
    valid=sofr.loc[sofr.value.notna()]
    if not pd.DatetimeIndex(valid.observation_date).isin(official).all():raise ValueError('initial SOFR has nonofficial observation dates')
    first=valid.observation_date.min()
    expected=official[(official>=first)&(official<=valid.observation_date.max())]
    if not expected.isin(pd.DatetimeIndex(valid.observation_date)).all():raise ValueError('internal initial SOFR coverage gap')
    if not valid.release_date.gt(valid.observation_date).all():raise ValueError('SOFR archive date not after underlying transactions')
    policies={'IOER':ioer.set_index('observation_date'),'IORB':iorb.set_index('observation_date')};events=[]
    overrides={pd.Timestamp(e['observation_date']):e for e in policy_events}
    for row in valid.itertuples(index=False):
        regime='IORB' if row.observation_date>=pd.Timestamp('2021-07-29') else 'IOER';series=policies[regime]
        p=series.loc[row.observation_date] if row.observation_date in series.index else None
        value=float(p.value) if p is not None else np.nan;release=p.release_date if p is not None else pd.NaT;original=value;source='ALFRED initial'
        if row.observation_date in overrides:
            update=overrides[row.observation_date]
            if update['series']!=regime or pd.Timestamp(update['public_date'])>row.observation_date:raise ValueError('invalid dated policy correction')
            value=float(update['value']);release=pd.Timestamp(update['public_date']);source='dated official implementation note'
        known=np.isfinite(value)
        day=max(d for d in [row.observation_date,row.release_date,release] if pd.notna(d))
        available=day.tz_localize('America/New_York')+pd.Timedelta(hours=23,minutes=59,seconds=59)
        events.append({'observation_date':row.observation_date,'sofr_archive_date':row.release_date,'policy_public_date':release,'available_at':available,'sofr_percent':float(row.value),'reserve_rate_percent':value,'original_policy_initial_value':original,'policy_source':source,'reserve_rate_is_iorb':float(regime=='IORB'),'known':float(known),'spread_bp':reported_rate_spread_bp(row.value,value) if known else np.nan})
    return pd.DataFrame(events).sort_values('observation_date')


def align_funding_information(events,calendar):
    calendar=pd.DatetimeIndex(calendar)
    if not calendar.is_unique or not calendar.is_monotonic_increasing:raise ValueError('ordered native calendar required')
    rows=[]
    for day in calendar:
        decision=day.tz_localize('America/New_York')+pd.Timedelta(hours=9,minutes=46)
        eligible=events.loc[events.available_at<decision]
        row={'datetime':day,'decision_at':decision,'known':0.,'spread_bp':np.nan,'sofr_percent':np.nan,'reserve_rate_percent':np.nan,'reserve_rate_is_iorb':np.nan,'observation_age_days':np.nan,'observation_date':pd.NaT,'available_at':pd.NaT}
        if not eligible.empty:
            event=eligible.iloc[-1];row.update({k:event[k] for k in ['known','spread_bp','sofr_percent','reserve_rate_percent','reserve_rate_is_iorb','observation_date','available_at']});row['observation_age_days']=(day-event.observation_date).days
        rows.append(row)
    return pd.DataFrame(rows)
