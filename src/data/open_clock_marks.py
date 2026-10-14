"""Qlib daily audit view of same-source regular open and execution marks."""
import hashlib
import json
from pathlib import Path
import pandas as pd
from src.data.full_path_prices import _source_frames
from src.data.minute_price_basis import published_intraday_basis
from src.data.qlib_native import build_qlib_store_from_frame,load_qlib_feature_frame,QlibProviderConfig


def build_open_clock_marks(*,source_manifest,source_sha256,provider_uri,as_of,workspace_root):
    source=Path(source_manifest)
    if hashlib.sha256(source.read_bytes()).hexdigest()!=source_sha256:
        raise ValueError('source manifest changed')
    receipt=json.loads(source.read_text())
    frames,events=_source_frames(source,receipt,as_of=as_of,workspace_root=workspace_root)
    parts=[]
    for symbol,raw in frames.items():
        if raw.date.gt(as_of).any():raise ValueError('source exceeds as_of')
        clock=raw.bar_start_at.dt.tz_convert('America/New_York').dt.strftime('%H:%M')
        selected=raw.loc[clock.isin(['09:30','09:47'])].copy()
        if selected.duplicated(['date','bar_start_at']).any() or not selected.groupby('date').size().reindex(sorted(raw.date.unique()),fill_value=0).eq(2).all():
            raise ValueError('missing or duplicate opening mark')
        if not selected.is_final.all():raise ValueError('unfinished opening mark')
        selected['slot']=clock.loc[selected.index].map({'09:30':'open930','09:47':'open947'})
        parts.append(selected)
    adjusted,basis=published_intraday_basis(pd.concat(parts,ignore_index=True),events,as_of=as_of)
    rows=[]
    for symbol,g in adjusted.groupby('symbol'):
        p=g.pivot(index='date',columns='slot',values='open')
        raw=g.pivot(index='date',columns='slot',values='raw_open')
        p['raw930']=raw.open930;p['raw947']=raw.open947
        valid=g.pivot(index='date',columns='slot',values='price_valid')
        p['valid930']=valid.open930.astype(float);p['valid947']=valid.open947.astype(float)
        p.loc[~valid.open930,['open930','raw930']]=float('nan')
        p.loc[~valid.open947,['open947','raw947']]=float('nan')
        first=g[g.slot.eq('open930')].set_index('date')
        p['affine_a']=first.affine_A;p['affine_b']=first.affine_B;p['split_factor']=first.split_price_factor
        p['instrument']=symbol;p['datetime']=pd.to_datetime(p.index);rows.append(p.reset_index(drop=True))
    fields=['open930','open947','raw930','raw947','affine_a','affine_b','split_factor','valid930','valid947']
    manifest=build_qlib_store_from_frame(pd.concat(rows,ignore_index=True),provider_uri=provider_uri,as_of=as_of,field_map={f:f for f in fields},source='same frozen NONE minute source with published affine adjustment')
    manifest.update(source_manifest_sha256=source_sha256,price_basis=basis,role='diagnostic_only_full_day_marks_not_decision_features')
    (Path(provider_uri)/'qlib_store_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    result=load_qlib_feature_frame(QlibProviderConfig(Path(provider_uri)),instruments=['SOXL','SOXX'],fields=['$'+f for f in fields],start=receipt['start'],end=as_of)
    return result,manifest
