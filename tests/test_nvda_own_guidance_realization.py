import numpy as np
import pandas as pd
from src.data.nvda_own_guidance_realization import align_realizations,FEATURE,FLAGS


def test_timestamp_blocked_reset_future_guide_and_original_pct_base():
    pairs=pd.DataFrame({'target_quarter':['FY2012Q1','FY2012Q2','FY2012Q3'],
        'actual_report_available_at':['2020-01-02 09:31-05:00','2020-01-06 09:31-05:00',None],
        'measurement_known':[True,False,False],'guide_unit':['pct_sequential_revenue_growth']*3,
        'percentage_conversion_supported':[True,False,False],'guide_mid':[10.,5.,20.],
        'guide_base_actual_million':[100.,999.,300.],'actual_revenue_million':[121.,400.,np.nan],
        'freshness':['known_preannounced_or_updated','unknown_update_coverage','unknown_update_coverage']})
    decisions=pd.Series(pd.to_datetime(['2020-01-02 09:29-05:00','2020-01-03 09:29-05:00','2020-01-06 09:29-05:00','2020-01-07 09:29-05:00'],utc=True))
    x,trace,events=align_realizations(pairs,decisions)
    assert x[FLAGS[1]].tolist()==[0,1,1,0]
    assert np.isnan(x[FEATURE].iloc[0]) and np.isnan(x[FEATURE].iloc[-1])
    np.testing.assert_allclose(x[FEATURE].iloc[1:3],[.1,.1])
    assert len(events)==2 and trace.target_quarter.iloc[-1]=='FY2012Q2'
    assert x[FLAGS[2]].tolist()==[0,1,1,0]
