"""Conservative report-time state for the fixed own-original forecast realization."""
import numpy as np
import pandas as pd
FEATURE='nvda_own_original_revenue_realization'
FLAGS=['nvda_realization_report_seen','nvda_realization_comparable','nvda_realization_known_preannounced']


def align_realizations(pairs, decision_at):
    events=pairs.loc[pairs.actual_report_available_at.notna()].copy()
    events['available_utc']=pd.to_datetime(events.actual_report_available_at,utc=True)
    if events.available_utc.duplicated().any():raise ValueError('ambiguous simultaneous realized reports')
    events=events.sort_values('available_utc')
    events[FEATURE]=np.nan
    known=events.measurement_known.astype(bool)
    amounts=events.guide_unit.isin(['million','USD_million'])
    percent=events.guide_unit.eq('pct_sequential_revenue_growth') & events.percentage_conversion_supported.astype(bool)
    if not (amounts|percent).loc[known].all():raise ValueError('unknown qualified guidance unit')
    midpoint=events.guide_mid.astype(float).where(amounts,events.guide_base_actual_million*(1+events.guide_mid/100))
    if not midpoint.loc[known].gt(0).all():raise ValueError('nonpositive original guide midpoint')
    events.loc[known,FEATURE]=events.loc[known,'actual_revenue_million']/midpoint.loc[known]-1
    events[FLAGS[0]]=1.;events[FLAGS[1]]=known.astype(float)
    events[FLAGS[2]]=events.freshness.eq('known_preannounced_or_updated').astype(float)
    decisions=pd.to_datetime(decision_at,utc=True)
    output=pd.DataFrame(index=decision_at.index,columns=[*FLAGS,FEATURE],dtype=float)
    output[FLAGS]=0.
    trace=pd.DataFrame(index=decision_at.index,columns=['target_quarter','source_available_at','decision_at'])
    good=decisions.notna()
    left=pd.DataFrame({'decision_at':decisions.loc[good], 'row':np.flatnonzero(good)}).sort_values('decision_at')
    joined=pd.merge_asof(left,events[['available_utc','target_quarter',*FLAGS,FEATURE]],left_on='decision_at',right_on='available_utc',direction='backward',allow_exact_matches=True)
    for r in joined.itertuples():
        idx=output.index[r.row]
        for flag in FLAGS:output.loc[idx,flag]=getattr(r,flag) if pd.notna(getattr(r,flag)) else 0.
        output.loc[idx,FEATURE]=getattr(r,FEATURE)
        trace.loc[idx]=[r.target_quarter,r.available_utc,r.decision_at]
        if pd.notna(r.available_utc) and r.available_utc>r.decision_at:raise ValueError('source timing leakage')
    if output.loc[output[FLAGS[1]].eq(0),FEATURE].notna().any():raise ValueError('blocked/unseen realization carried forward')
    return output,trace,events
