"""Fixed AMD source normalization using the existing delayed event-state contract."""
import pandas as pd
from src.data.nvda_own_guidance_realization import align_realizations as _align,FEATURE as NVDA_FEATURE,FLAGS as NVDA_FLAGS
FEATURE='amd_own_original_revenue_realization'
FLAGS=['amd_realization_report_seen','amd_realization_comparable','amd_realization_known_preannounced']


def align_realizations(pairs,decision_at):
    data=pairs.copy()
    amount=data.guide_unit.eq('USD_million')
    data.loc[amount,'guide_mid']=data.loc[amount,'guide_mid_million']
    data['percentage_conversion_supported']=data.percentage_conversion_supported.fillna(False).astype(bool)
    known=data.measurement_known.astype(bool)
    pair=pd.to_datetime(data.pair_available_at,utc=True)
    actual=pd.to_datetime(data.actual_report_available_at,utc=True)
    if not (pair.loc[known].notna()&pair.loc[known].le(actual.loc[known])).all():raise ValueError('guide/pair not known at actual event')
    feature,trace,events=_align(data,decision_at)
    rename={NVDA_FEATURE:FEATURE,**dict(zip(NVDA_FLAGS,FLAGS))}
    return feature.rename(columns=rename),trace,events.rename(columns=rename)
