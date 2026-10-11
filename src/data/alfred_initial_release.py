"""Official initial-release archives and conservative calendar-bound information states."""
from __future__ import annotations
from pathlib import Path
import hashlib,io,json,zipfile
import numpy as np
import pandas as pd


def read_initial_release_archive(directory, *, as_of):
    directory = Path(directory)
    manifest = json.loads((directory/'manifest.json').read_text())
    series = manifest['series_id']
    if manifest['output_format'] != 'Observations, Initial Release Only' or manifest['as_of'] != as_of:
        raise ValueError('blocked: first-release format or cutoff differs')
    frames = []
    for entry in manifest['entries']:
        path = directory.parent/entry['path']
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != entry['sha256']:
            raise ValueError('blocked: raw initial-release archive changed')
        archive = zipfile.ZipFile(io.BytesIO(data))
        readme = archive.read('README.txt').decode()
        if 'Output Format: Observations, Initial Release Only' not in readme or ('Series ID: '+series) not in readme:
            raise ValueError('blocked: official archive identity/format differs')
        frame = pd.read_csv(io.BytesIO(archive.read(entry['csv_name'])))
        if list(frame.columns) != ['period_start_date',series,'realtime_start_date']:
            raise ValueError('blocked: unexpected initial-release schema')
        frame = frame.rename(columns={'period_start_date':'observation_date',series:'value','realtime_start_date':'release_date'})
        frame['batch_sha256'] = entry['sha256']
        frames.append(frame)
    result = pd.concat(frames,ignore_index=True).sort_values('observation_date').reset_index(drop=True)
    for name in ['observation_date','release_date']:
        result[name] = pd.to_datetime(result[name],errors='raise')
        if result[name].isna().any() or (result[name]>pd.Timestamp(as_of)).any():
            raise ValueError('blocked: unknown or after-cutoff raw date')
    if result.observation_date.duplicated().any():
        raise ValueError('blocked: duplicate initial-release observation')
    known = result.value.notna()
    if not np.isfinite(result.loc[known,'value'].to_numpy(float)).all() or (result.loc[known,'value']<=0).any():
        raise ValueError('blocked: volatility index must be positive or explicitly unknown')
    result['series_id'] = series
    return result


def qualify_paired_information(vxn,vix,calendar):
    """Preserve raw anomalies; quarantine premature/unknown/non-session observations.

    Pair the same observation day. Availability is after BOTH publication dates
    and the observation day, at09:31NY of the next actual native session. Invalid
    latest published observations invalidate the feature; no older-value fill.
    """
    calendar = pd.DatetimeIndex(calendar)
    if calendar.hasnans or calendar.has_duplicates or not calendar.is_monotonic_increasing:
        raise ValueError('blocked: unique complete native calendar required')
    first,last = calendar[0],calendar[-1]
    pair = vxn[['observation_date','value','release_date']].merge(
        vix[['observation_date','value','release_date']],on='observation_date',how='outer',suffixes=('_vxn','_vix'),validate='one_to_one').sort_values('observation_date')
    pair['on_native_session'] = pair.observation_date.isin(calendar)
    pair['premature_release'] = (pair.release_date_vxn < pair.observation_date)|(pair.release_date_vix < pair.observation_date)
    pair['both_dates_known'] = pair.release_date_vxn.notna() & pair.release_date_vix.notna()
    pair['both_values_known'] = pair.value_vxn.notna() & pair.value_vix.notna()
    anchor = pd.concat([pair.observation_date,pair.release_date_vxn,pair.release_date_vix],axis=1).max(axis=1)
    next_position = calendar.searchsorted(pd.DatetimeIndex(anchor),side='right')
    valid_next = next_position < len(calendar)
    available = pd.Series(pd.NaT,index=pair.index,dtype='datetime64[ns, UTC]')
    dates = calendar[next_position[valid_next]]
    clocks = (dates.tz_localize('America/New_York')+pd.Timedelta(hours=9,minutes=31)).tz_convert('UTC')
    available.loc[pair.index[valid_next]] = clocks
    available.loc[~pair.both_dates_known] = pd.NaT
    pair['available_at'] = available
    pair['qualified'] = pair.on_native_session & pair.both_dates_known & pair.both_values_known & ~pair.premature_release
    pair['log_vxn_vix'] = np.where(pair.qualified,np.log(pair.value_vxn/pair.value_vix),np.nan)
    rows = []
    # Non-session observations stay in raw evidence but never become synthetic closes.
    events = pair.loc[pair.on_native_session & pair.available_at.notna()].copy()
    for number,day in enumerate(calendar):
        decision = (day.tz_localize('America/New_York')+pd.Timedelta(hours=16,minutes=15)).tz_convert('UTC')
        eligible = events.loc[events.available_at<=decision]
        row = dict(datetime=day,known=0.,log_vxn_vix=np.nan,observation_calendar_index=np.nan,
                   availability_calendar_index=np.nan,vxn=np.nan,vix=np.nan,decision_at=decision,
                   observation_date=pd.NaT,available_at=pd.NaT,reason='no_qualified_source_yet')
        if not eligible.empty:
            event = eligible.sort_values('observation_date').iloc[-1]
            obs_index = int(calendar.get_indexer([event.observation_date])[0])
            availability_date = event.available_at.tz_convert('America/New_York').tz_localize(None).normalize()
            row.update(observation_calendar_index=obs_index,
                availability_calendar_index=int(calendar.get_indexer([availability_date])[0]),
                observation_date=event.observation_date,available_at=event.available_at,
                reason='eligible_initial_release' if event.qualified else 'latest_observation_quarantined_or_unknown')
            if event.qualified:
                row.update(known=1.,log_vxn_vix=float(event.log_vxn_vix),vxn=float(event.value_vxn),vix=float(event.value_vix))
        rows.append(row)
    state = pd.DataFrame(rows)
    audit = dict(raw_vxn_rows=len(vxn),raw_vix_rows=len(vix),calendar_rows=len(calendar),
        known_signal_rows=int(state.known.sum()),unknown_signal_rows=int((state.known==0).sum()),
        premature_market_observations=int((pair.premature_release & pair.on_native_session).sum()),
        non_session_raw_observations=int((~pair.on_native_session).sum()),
        unknown_latest_observations=int((state.reason=='latest_observation_quarantined_or_unknown').sum()),
        date_level_initial_release_verified=True,intraday_publication_timestamp_verified=False,
        availability_contract='next native session09:31NY after max(observation_date,both release dates); premature rows quarantined',
        same_observation_date_pairing=True,unknown_feature_not_zero_filled=True,
        price_evaluation_dates_dropped=0,source_range=[str(first.date()),str(last.date())])
    return state,pair,audit


def build_information_store(state, *, provider_uri, as_of, source):
    from src.data.qlib_native import build_qlib_store_from_frame
    fields = ['known','log_vxn_vix','vxn','vix','observation_calendar_index','availability_calendar_index']
    frame = state[['datetime',*fields]].copy()
    frame['instrument'] = 'TECH_IV_REL'
    manifest = build_qlib_store_from_frame(frame,provider_uri=Path(provider_uri),as_of=as_of,
        source=source,field_map={name:name for name in fields},region='us')
    manifest.update(information_instrument=True,not_trading_prices=True,initial_release_values_only=True,
        date_level_vintage_verified=True,intraday_publication_timestamp_verified=False,
        unknown_feature_not_zero_filled=True,availability_contract='next actual session09:31NY after both release dates; premature observations quarantined')
    (Path(provider_uri)/'qlib_store_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    import qlib
    from qlib.data import D
    qlib.init(provider_uri=str(Path(provider_uri).resolve()),region='us',kernels=1)
    cal = pd.DatetimeIndex(D.calendar(freq='day'))
    if not cal.equals(pd.DatetimeIndex(state.datetime)):
        raise ValueError('blocked: information store changed the native calendar')
    readback = D.features(['TECH_IV_REL'],['$'+name for name in fields],freq='day')
    for name in fields:
        if not np.allclose(readback['$'+name],state[name],equal_nan=True,rtol=0,atol=1e-5):
            raise ValueError('blocked: information store readback differs: '+name)
    return manifest,readback
