"""One explicitly available information field on otherwise identical native DatasetH arms."""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from src.data.convex_allocation import FEATURE_COLUMNS,LABEL_COLUMNS,METADATA_COLUMNS
from src.data.generic_horizon import attach_horizon_scope
from src.data.integrated_allocation import build_native_dataset
from src.data.provider_content_hash import provider_content_hash


def load_information_state(*,provider_uri,content_sha256,instrument,field,signal_dates):
    import qlib
    from qlib.data import D
    provider_uri=Path(provider_uri)
    if provider_content_hash(provider_uri)[0]!=content_sha256:
        raise ValueError('blocked: information provider identity differs')
    qlib.init(provider_uri=str(provider_uri.resolve()),region='us',kernels=1)
    calendar=pd.DatetimeIndex(D.calendar(freq='day'))
    raw=D.features([instrument],['$known',field,'$availability_calendar_index'],freq='day')
    if set(raw.index.get_level_values('instrument'))!={instrument}:
        raise ValueError('blocked: information instrument differs')
    raw=raw.droplevel('instrument')
    if not raw.index.equals(calendar) or not calendar.equals(pd.DatetimeIndex(signal_dates)):
        raise ValueError('blocked: full information and price signal calendars must match')
    if not raw['$known'].isin([0,1]).all():
        raise ValueError('blocked: known must be explicit0/1')
    known=raw['$known'].to_numpy(bool)
    positions=raw.loc[known,'$availability_calendar_index'].to_numpy(float)
    if not (np.isfinite(positions).all() and np.array_equal(positions,positions.astype(int))
            and (positions>=0).all() and (positions<len(calendar)).all()):
        raise ValueError('blocked: invalid information calendar indices')
    available=pd.Series(pd.NaT,index=calendar,dtype='datetime64[ns, UTC]')
    available.loc[known]=(calendar[positions.astype(int)].tz_localize('America/New_York')+pd.Timedelta(hours=9,minutes=31)).tz_convert('UTC')
    if not np.isfinite(raw.loc[known,field]).all() or not raw.loc[~known,field].isna().all():
        raise ValueError('blocked: known/unknown information values differ')
    if provider_content_hash(provider_uri)[0]!=content_sha256:
        raise ValueError('blocked: information provider changed during readback')
    return pd.DataFrame(dict(source_known=known.astype(int),information_value=raw[field],source_available_at=available),index=calendar)


def bind_matched_information_datasets(bundle,*,as_of):
    """Also restores native caches from frozen Recorder frames without a source refetch."""
    result=dict(bundle);frame=result['frame']
    for arm,features in [('control',FEATURE_COLUMNS),('candidate',(*FEATURE_COLUMNS,'information_value'))]:
        metadata=(*METADATA_COLUMNS,'source_known','source_available_at','base_feature_available_at')
        dataset=build_native_dataset(frame,feature_columns=features,label_columns=LABEL_COLUMNS,metadata_columns=metadata,as_of=as_of)
        attach_horizon_scope(dataset,5);dataset.calendar=bundle['calendar']
        prepared=dataset.prepare('all',col_set=['feature','label','metadata'])
        if arm+'_prepared' in bundle:
            pd.testing.assert_frame_equal(prepared,bundle[arm+'_prepared'],check_exact=True)
        if not pd.DatetimeIndex(prepared.index.get_level_values('datetime')).equals(frame.index):
            raise ValueError('blocked: DatasetH removed or reordered information dates')
        result[arm]=dataset;result[arm+'_prepared']=prepared;result[arm+'_features']=tuple(features)
    pd.testing.assert_frame_equal(result['control_prepared']['metadata'],result['candidate_prepared']['metadata'],check_exact=True)
    pd.testing.assert_frame_equal(result['control_prepared']['label'],result['candidate_prepared']['label'],check_exact=True)
    return result


def build_matched_information_datasets(price_bundle,information_state,*,as_of):
    frame=price_bundle['frame'].copy()
    if not information_state.index.equals(frame.index):
        raise ValueError('blocked: information must retain the complete signal axis')
    known=information_state.source_known
    if not known.isin([0,1]).all():
        raise ValueError('blocked: source known must be explicit0/1')
    mask=known.astype(bool)
    from src.model.generic_horizon_estimator import _aware
    price_at=_aware(frame.feature_available_at,'price feature clock')
    source_at=_aware(information_state.source_available_at,'information clock')
    if price_at.loc[mask].isna().any() or source_at.loc[mask].isna().any():
        raise ValueError('blocked: known information needs both explicit clocks')
    combined=pd.Series(pd.NaT,index=frame.index,dtype='datetime64[ns, UTC]')
    combined.loc[mask]=pd.concat([price_at,source_at],axis=1).max(axis=1).loc[mask]
    if (combined.loc[mask]>pd.to_datetime(frame.loc[mask,'decision_at'],utc=True)).any():
        raise ValueError('blocked: information unavailable at prediction')
    frame['base_feature_available_at']=frame.feature_available_at
    for name in ('source_known','information_value','source_available_at'):
        frame[name]=information_state[name]
    frame['feature_available_at']=combined
    if not np.isfinite(frame.loc[mask,'information_value']).all() or not frame.loc[~mask,'information_value'].isna().all():
        raise ValueError('blocked: information was missing or unknown was filled')
    result=bind_matched_information_datasets({**price_bundle,'frame':frame},as_of=as_of)
    result.update(rows=len(frame),known_rows=int(mask.sum()),unknown_rows=int((~mask).sum()),fit_performed=False,
        eligible_dates_sha256=hashlib.sha256('\n'.join(map(str,frame.index[mask])).encode()).hexdigest())
    return result
