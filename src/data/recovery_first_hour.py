"""Complete first-hour recovery fields from the frozen canonical minute source."""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from src.data.full_path_prices import _source_frames
from src.data.minute_price_basis import published_intraday_basis
from src.data.qlib_native import build_qlib_store_from_frame,load_qlib_feature_frame,QlibProviderConfig

FIELDS=[f'early_{field}_{asset}' for field in ['ret','range','from_low','volume'] for asset in ['soxl','soxx']]


def aggregate_first_hour(points):
    rows=[]
    for day,group in points.groupby('date'):
        row={'datetime':pd.Timestamp(day),'first_hour_ready':0.};row.update({f:np.nan for f in FIELDS});values={}
        expected=pd.date_range(pd.Timestamp(day).tz_localize('America/New_York')+pd.Timedelta(hours=9,minutes=31),periods=60,freq='min')
        for symbol in ['SOXL','SOXX']:
            block=group[group.symbol.eq(symbol)];block=block[block.timestamp.isin(expected)].sort_values('timestamp')
            if not pd.DatetimeIndex(block.timestamp).equals(expected) or not block.price_valid.all() or not block.is_final.all():break
            if not np.isfinite(block[['open','high','low','close','volume']]).all().all():break
            if (block[['open','high','low','close']]<=0).any().any() or block.volume.lt(0).any():break
            asset=symbol.lower();opening=block.open.iloc[0];closing=block.close.iloc[-1];high=block.high.max();low=block.low.min()
            values.update({f'early_ret_{asset}':closing/opening-1,f'early_range_{asset}':low/high-1,f'early_from_low_{asset}':closing/low-1,f'early_volume_{asset}':block.volume.sum()})
        if len(values)==len(FIELDS):row.update(values);row['first_hour_ready']=1.
        rows.append(row)
    return pd.DataFrame(rows).set_index('datetime').sort_index()


def build_recovery_first_hour(*,source_manifest,source_sha256,provider_uri,as_of,workspace_root):
    path=Path(source_manifest)
    if hashlib.sha256(path.read_bytes()).hexdigest()!=source_sha256:raise ValueError('canonical minute source changed')
    receipt=json.loads(path.read_text());frames,events=_source_frames(path,receipt,as_of=as_of,workspace_root=workspace_root);parts=[]
    for raw in frames.values():
        if raw.date.gt(as_of).any():raise ValueError('source exceeds as_of')
        clock=raw.timestamp.dt.tz_convert('America/New_York').dt.hour*60+raw.timestamp.dt.tz_convert('America/New_York').dt.minute
        parts.append(raw[clock.between(571,630)].copy())
    points,basis=published_intraday_basis(pd.concat(parts,ignore_index=True),events,as_of=as_of);features=aggregate_first_hour(points)
    # Preserve every source day even if neither asset supplied a first-hour row.
    days=pd.DatetimeIndex(sorted(set().union(*(set(f.date) for f in frames.values()))));features=features.reindex(days);features['first_hour_ready']=features.first_hour_ready.fillna(0.)
    data=features.rename_axis('datetime').reset_index();data['instrument']='RECOVERY_PAIR';fields=FIELDS+['first_hour_ready'];manifest=build_qlib_store_from_frame(data,provider_uri=provider_uri,as_of=as_of,field_map={f:f for f in fields},source='Canonical first60 minute bars, published affine OHLC and split-adjusted share volume')
    manifest.update(source_manifest_sha256=source_sha256,price_basis=basis,feature_information='09:31..10:30 right-edge bars available10:31; historical latency assumption, not live receipt certification')
    (Path(provider_uri)/'qlib_store_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    q=load_qlib_feature_frame(QlibProviderConfig(Path(provider_uri)),instruments=['RECOVERY_PAIR'],fields=['$'+f for f in fields],start=str(days.min().date()),end=as_of).xs('RECOVERY_PAIR',level='instrument').rename(columns=lambda x:x.removeprefix('$'))
    q['recovery_feature_available_at']=q.index.tz_localize('America/New_York')+pd.Timedelta(hours=10,minutes=31)
    return q,manifest
