"""Exchange-local observations mapped to an explicitly later US decision clock."""
import pandas as pd
import numpy as np
from pathlib import Path
from src.data.qlib_native import build_qlib_store_from_frame,QlibProviderConfig,load_qlib_feature_frame


def align_asian_closes(raw,us_dates,*,markets,availability_hour):
    if raw.index.has_duplicates or not raw.index.is_monotonic_increasing:raise ValueError('invalid source local dates')
    dates=pd.DatetimeIndex(us_dates);parts=[];audit=[]
    for market,zone in markets.items():
        observed=raw[market].dropna()
        if not np.isfinite(observed).all() or not observed.gt(0).all():raise ValueError('invalid Asian close')
        own=pd.DataFrame({'return_value':observed/observed.shift(1)-1,'gap_days':observed.index.to_series().diff().dt.days})
        values=own.reindex(dates)
        available=(dates.tz_localize(zone)+pd.Timedelta(hours=availability_hour)).tz_convert('America/New_York')
        decision=(dates+pd.Timedelta(hours=9,minutes=46)).tz_localize('America/New_York')
        known=values.return_value.notna()
        if not (available[known]<decision[known]).all():raise ValueError('Asian daily observation not available before US decision')
        parts.append(pd.DataFrame({'instrument':market.upper(),'datetime':dates,'return_value':values.return_value.to_numpy(),'known':known.astype(float).to_numpy(),'gap_days':values.gap_days.to_numpy()}))
        audit.append(pd.DataFrame({'market':market,'trade_date':dates,'assumed_available_at':available.astype(str),'decision_at':decision.astype(str),'known':known.to_numpy()}))
    return pd.concat(parts,ignore_index=True),pd.concat(audit,ignore_index=True)


def build_asia_features(raw,us_dates,*,plan,provider):
    points,alignment=align_asian_closes(raw,us_dates,markets=plan['markets'],availability_hour=plan['availability_local_hour'])
    manifest=build_qlib_store_from_frame(points,provider_uri=Path(provider),as_of=plan['as_of'],field_map={k:k for k in ['return_value','known','gap_days']},source='frozen Asian exchange-local dates;18:00local assumed availability; missing stays unknown')
    native=load_qlib_feature_frame(QlibProviderConfig(Path(provider)),instruments=[x.upper() for x in plan['markets']],fields=['$return_value','$known','$gap_days'],start=str(min(us_dates).date()),end=plan['as_of'])
    features=pd.DataFrame(index=pd.DatetimeIndex(us_dates))
    for market in plan['markets']:
        data=native.xs(market.upper(),level='instrument').reindex(features.index)
        expected=points.loc[points.instrument.eq(market.upper())].set_index('datetime').reindex(features.index)
        for column in ['return_value','known','gap_days']:
            np.testing.assert_allclose(data['$'+column],expected[column],rtol=1e-6,atol=1e-7,equal_nan=True)
            features[market+'_'+column]=data['$'+column].to_numpy()
    return features,alignment,manifest
